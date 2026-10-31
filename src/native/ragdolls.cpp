#include "ragdolls.h"

#include "snapshot.h"

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace pj {

namespace {

constexpr double kTwoPi = 6.283185307179586;

JPH::Vec3 vec3(const double *v) { return JPH::Vec3(float(v[0]), float(v[1]), float(v[2])); }

JPH::Quat quat(const double *v) {
    return JPH::Quat(float(v[0]), float(v[1]), float(v[2]), float(v[3])).Normalized();
}

JPH::Quat slerp(JPH::QuatArg a, JPH::QuatArg b, float s) {
    return a.SLERP(a.Dot(b) < 0.0f ? -b : b, s).Normalized();
}

JPH::SwingTwistConstraint *swing_twist(JPH::Ragdoll &ragdoll, int bone) {
    const int index = ragdoll.GetRagdollSettings()->GetConstraintIndexForBodyIndex(bone);
    if (index < 0)
        return nullptr;
    return static_cast<JPH::SwingTwistConstraint *>(ragdoll.GetConstraint(index));
}

}  // namespace

RagdollSystem::~RagdollSystem() {
    for (auto &doll : ragdolls_)
        doll->ragdoll->RemoveFromPhysicsSystem(false);
}

Ragdoll *RagdollSystem::find(RagdollId id) {
    auto at = std::lower_bound(ragdolls_.begin(), ragdolls_.end(), id,
                               [](const std::unique_ptr<Ragdoll> &d, RagdollId v) { return d->desc.id < v; });
    return at != ragdolls_.end() && (*at)->desc.id == id ? at->get() : nullptr;
}

bool RagdollSystem::owns_ragdoll(RagdollId id) const {
    for (const auto &doll : ragdolls_)
        if (doll->desc.id == id)
            return true;
    return false;
}

void RagdollSystem::bodies_of(RagdollId id, std::vector<JPH::BodyID> &bodies, std::vector<Uid> &uids) const {
    for (const auto &doll : ragdolls_)
        if (doll->desc.id == id) {
            const JPH::Array<JPH::BodyID> &ids = doll->ragdoll->GetBodyIDs();
            bodies.assign(ids.begin(), ids.end());
            uids = doll->desc.uids;
        }
}

bool RagdollSystem::owns(Uid uid) const { return bones_.count(uid) != 0; }

bool RagdollSystem::driving() const {
    for (const auto &doll : ragdolls_)
        if (doll->mode != RagdollMode::Limp || doll->strength > 0.0f)
            return true;
    return false;
}

std::vector<RagdollId> RagdollSystem::ids() const {
    std::vector<RagdollId> out;
    for (const auto &doll : ragdolls_)
        out.push_back(doll->desc.id);
    return out;
}

Ragdoll *RagdollSystem::add(const RagdollDesc &desc, JPH::ObjectLayer layer, const JPH::GroupFilter *filter,
                            std::vector<std::pair<int, int>> &pairs, double time) {
    const size_t n = desc.bone_count();
    const double *p = desc.params.data();
    auto doll = std::make_unique<Ragdoll>();
    doll->desc = desc;
    doll->position = JPH::RVec3(p[kRagX], p[kRagY], p[kRagZ]);
    doll->rotation = quat(p + kRagQX);
    doll->parent.resize(n);
    doll->rotations.resize(n);
    doll->translations.resize(n);
    for (size_t i = 0; i < n; ++i) {
        doll->parent[i] = int(desc.bone(i, kBoneParent));
        doll->rotations[i] = quat(&desc.bones[i * kBoneParamCount + kBoneRX]);
        doll->translations[i] = vec3(&desc.bones[i * kBoneParamCount + kBoneTX]);
    }

    // The rest pose in the world, and in the root's frame for the overlap test.
    std::vector<JPH::RVec3> position;
    std::vector<JPH::Quat> rotation;
    target(*doll, time, position, rotation);
    JPH::Ref<JPH::RagdollSettings> settings = new JPH::RagdollSettings;
    settings->mSkeleton = new JPH::Skeleton;
    settings->mParts.resize(n);
    const auto flags = uint32_t(p[kRagFlags]);
    for (size_t i = 0; i < n; ++i) {
        settings->mSkeleton->AddJoint(std::to_string(i), doll->parent[i]);
        JPH::RagdollSettings::Part &part = settings->mParts[i];
        part.SetShape(desc.shapes[i]);
        part.mPosition = position[i];
        part.mRotation = rotation[i];
        part.mMotionType = JPH::EMotionType::Dynamic;
        part.mAllowDynamicOrKinematic = true;
        part.mObjectLayer = layer;
        part.mFriction = float(p[kRagFriction]);
        part.mRestitution = float(p[kRagRestitution]);
        part.mLinearDamping = float(p[kRagLinearDamping]);
        part.mAngularDamping = float(p[kRagAngularDamping]);
        part.mGravityFactor = float(p[kRagGravityFactor]);
        part.mAllowSleeping = (flags & flag::kNoSleep) == 0;
        part.mMotionQuality = (flags & flag::kContinuous) ? JPH::EMotionQuality::LinearCast
                                                          : JPH::EMotionQuality::Discrete;
        part.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        part.mMassPropertiesOverride.mMass = float(desc.bone(i, kBoneMass));
        if (doll->parent[i] < 0)
            continue;
        JPH::Ref<JPH::SwingTwistConstraintSettings> joint = new JPH::SwingTwistConstraintSettings;
        joint->mSpace = JPH::EConstraintSpace::WorldSpace;
        joint->mPosition1 = joint->mPosition2 = position[i];
        const JPH::Vec3 twist = (rotation[i] * vec3(&desc.bones[i * kBoneParamCount + kBoneTwistX])).Normalized();
        JPH::Vec3 plane = rotation[i] * vec3(&desc.bones[i * kBoneParamCount + kBonePlaneX]);
        plane = (plane - twist * twist.Dot(plane)).NormalizedOr(twist.GetNormalizedPerpendicular());
        joint->mTwistAxis1 = joint->mTwistAxis2 = twist;
        joint->mPlaneAxis1 = joint->mPlaneAxis2 = plane;
        joint->mTwistMinAngle = float(desc.bone(i, kBoneTwistMin));
        joint->mTwistMaxAngle = float(desc.bone(i, kBoneTwistMax));
        joint->mNormalHalfConeAngle = float(desc.bone(i, kBoneNormalCone));
        joint->mPlaneHalfConeAngle = float(desc.bone(i, kBonePlaneCone));
        joint->mMaxFrictionTorque = float(desc.bone(i, kBoneFriction));
        part.mToParent = joint;
    }
    settings->CalculateBodyIndexToConstraintIndex();
    settings->CalculateConstraintIndexToBodyIdxPair();

    // The pairs kept apart: parent and child, and bones overlapping at rest;
    // every pair without self-collision.
    pairs.clear();
    if (p[kRagSelfCollision] != 0.0) {
        std::vector<JPH::Mat44> joints(n);
        for (size_t i = 0; i < n; ++i)
            joints[i] = JPH::Mat44::sRotationTranslation(rotation[i], JPH::Vec3(position[i] - position[0]));
        settings->DisableParentChildCollisions(joints.data(), 0.0f);
        const auto *table =
            static_cast<const JPH::GroupFilterTable *>(settings->mParts[0].mCollisionGroup.GetGroupFilter());
        for (size_t a = 0; a < n; ++a)
            for (size_t b = a + 1; b < n; ++b)
                if (!table->IsCollisionEnabled(JPH::CollisionGroup::SubGroupID(a), JPH::CollisionGroup::SubGroupID(b)))
                    pairs.emplace_back(int(a), int(b));
    } else {
        for (size_t a = 0; a < n; ++a)
            for (size_t b = a + 1; b < n; ++b)
                pairs.emplace_back(int(a), int(b));
    }
    for (size_t i = 0; i < n; ++i)
        settings->mParts[i].mCollisionGroup = JPH::CollisionGroup(filter, desc.uids[i], 0);

    doll->pairs = pairs;
    JPH::Ragdoll *made = settings->CreateRagdoll(0, 0, &system_);
    if (made == nullptr)
        return nullptr;
    doll->ragdoll = made;
    {
        const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
        for (size_t i = 0; i < n; ++i) {
            JPH::Body *body = locks.TryGetBody(made->GetBodyID(int(i)));
            body->SetUserData(user_data(desc.uids[i], flags, uint16_t(p[kRagMaterial])));
            body->GetCollisionGroup().SetGroupID(desc.uids[i]);
        }
    }
    made->AddToPhysicsSystem(JPH::EActivation::Activate, false);
    Ragdoll *out = doll.get();
    enter(*out, static_cast<RagdollMode>(uint8_t(p[kRagMode])), 0.0, time);
    auto at = std::lower_bound(ragdolls_.begin(), ragdolls_.end(), desc.id,
                               [](const std::unique_ptr<Ragdoll> &d, RagdollId v) { return d->desc.id < v; });
    ragdolls_.insert(at, std::move(doll));
    bones_.insert(desc.uids.begin(), desc.uids.end());
    return out;
}

bool RagdollSystem::remove(RagdollId id, std::vector<JPH::BodyID> &bodies, std::vector<Uid> &uids,
                           std::vector<std::pair<int, int>> &pairs) {
    auto at = std::lower_bound(ragdolls_.begin(), ragdolls_.end(), id,
                               [](const std::unique_ptr<Ragdoll> &d, RagdollId v) { return d->desc.id < v; });
    if (at == ragdolls_.end() || (*at)->desc.id != id)
        return false;
    Ragdoll &doll = **at;
    const JPH::Array<JPH::BodyID> &ids = doll.ragdoll->GetBodyIDs();
    bodies.assign(ids.begin(), ids.end());
    uids = doll.desc.uids;
    pairs = doll.pairs;
    for (Uid uid : doll.desc.uids)
        bones_.erase(uid);
    doll.ragdoll->RemoveFromPhysicsSystem(false);
    ragdolls_.erase(at);  // the Ragdoll destroys its bodies
    return true;
}

void RagdollSystem::enter(Ragdoll &doll, RagdollMode mode, double blend, double time) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const RagdollMode before = doll.mode;
    const JPH::Array<JPH::BodyID> &ids = doll.ragdoll->GetBodyIDs();
    doll.blend_time = 0.0;
    if (mode == RagdollMode::Kinematic) {
        if (blend > 0.0) {
            const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
            doll.from_position.resize(ids.size());
            doll.from_rotation.resize(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                const JPH::Body *body = locks.TryGetBody(ids[i]);
                doll.from_position[i] = body->GetPosition();
                doll.from_rotation[i] = body->GetRotation();
            }
            doll.blend_start = time;
            doll.blend_time = blend;
        }
        for (const JPH::BodyID &id : ids)
            bodies.SetMotionType(id, JPH::EMotionType::Kinematic, JPH::EActivation::Activate);
        doll.strength = doll.strength_target = 0.0f;
        doll.strength_rate = 0.0f;
    } else {
        if (before == RagdollMode::Kinematic) {
            for (const JPH::BodyID &id : ids)
                bodies.SetMotionType(id, JPH::EMotionType::Dynamic, JPH::EActivation::Activate);
            doll.strength = 1.0f;
        }
        doll.strength_target = mode == RagdollMode::Powered ? 1.0f : 0.0f;
        doll.strength_rate = blend > 0.0 ? float(1.0 / blend) : 0.0f;
        if (doll.strength_rate == 0.0f)
            doll.strength = doll.strength_target;
        bodies.ActivateBodies(ids.data(), int(ids.size()));
    }
    doll.mode = mode;
    doll.motors_dirty = true;
}

bool RagdollSystem::set_mode(RagdollId id, RagdollMode mode, double blend, double time) {
    Ragdoll *doll = find(id);
    if (doll == nullptr)
        return false;
    enter(*doll, mode, blend, time);
    return true;
}

bool RagdollSystem::set_pose(const SetRagdollPose &pose) {
    Ragdoll *doll = find(pose.id);
    if (doll == nullptr)
        return false;
    const size_t n = doll->desc.bone_count();
    if ((!pose.rotations.empty() && pose.rotations.size() != 4 * n) ||
        (!pose.translations.empty() && pose.translations.size() != 3 * n))
        return false;
    if (!pose.rotations.empty())
        for (size_t i = 0; i < n; ++i)
            doll->rotations[i] = quat(&pose.rotations[4 * i]);
    if (!pose.translations.empty())
        for (size_t i = 0; i < n; ++i)
            doll->translations[i] = vec3(&pose.translations[3 * i]);
    if (!std::isnan(pose.position[0])) {
        doll->position = JPH::RVec3(pose.position[0], pose.position[1], pose.position[2]);
        doll->rotation = quat(pose.rotation);
    }
    return true;
}

bool RagdollSystem::set_clip(const SetRagdollClip &clip, double time) {
    Ragdoll *doll = find(clip.id);
    if (doll == nullptr)
        return false;
    const size_t n = doll->desc.bone_count();
    const size_t keys = clip.times.size();
    if (clip.rotations.size() != 4 * n * keys || (!clip.translations.empty() && clip.translations.size() != 3 * n * keys))
        return false;
    doll->clip_times = clip.times;
    doll->clip_rotations = clip.rotations;
    doll->clip_translations = clip.translations;
    doll->clip_loop = clip.loop;
    doll->clip_start = time;
    return true;
}

bool RagdollSystem::set_strength(RagdollId id, const std::vector<double> &strength) {
    Ragdoll *doll = find(id);
    if (doll == nullptr || strength.size() != doll->desc.bone_count())
        return false;
    for (size_t i = 0; i < strength.size(); ++i)
        doll->desc.bones[i * kBoneParamCount + kBoneStrength] = strength[i];
    doll->motors_dirty = true;
    return true;
}

void RagdollSystem::target(Ragdoll &doll, double time, std::vector<JPH::RVec3> &position,
                           std::vector<JPH::Quat> &rotation, std::vector<JPH::Quat> *local) const {
    const size_t n = doll.parent.size();
    position.resize(n);
    rotation.resize(n);
    if (local)
        local->resize(n);
    const size_t keys = doll.clip_times.size();
    size_t k = 0;
    float s = 0.0f;
    if (keys) {
        double t = time - doll.clip_start;
        const double last = doll.clip_times.back();
        if (doll.clip_loop && last > 0.0)
            t = std::fmod(std::max(t, 0.0), last);
        t = std::clamp(t, 0.0, last);
        k = size_t(std::upper_bound(doll.clip_times.begin(), doll.clip_times.end(), t) - doll.clip_times.begin());
        k = k == 0 ? 0 : k - 1;
        if (k + 1 < keys)
            s = float((t - doll.clip_times[k]) / (doll.clip_times[k + 1] - doll.clip_times[k]));
        else
            s = 0.0f;
    }
    const size_t k1 = std::min(k + 1, keys ? keys - 1 : 0);
    for (size_t i = 0; i < n; ++i) {
        JPH::Quat r = doll.rotations[i];
        JPH::Vec3 t = doll.translations[i];
        if (keys) {
            r = slerp(quat(&doll.clip_rotations[(k * n + i) * 4]), quat(&doll.clip_rotations[(k1 * n + i) * 4]), s);
            if (!doll.clip_translations.empty()) {
                const JPH::Vec3 a = vec3(&doll.clip_translations[(k * n + i) * 3]);
                const JPH::Vec3 b = vec3(&doll.clip_translations[(k1 * n + i) * 3]);
                t = a + (b - a) * s;
            }
        }
        if (local)
            (*local)[i] = r;
        const int parent = doll.parent[i];
        const JPH::RVec3 base = parent < 0 ? doll.position : position[size_t(parent)];
        const JPH::Quat frame = parent < 0 ? doll.rotation : rotation[size_t(parent)];
        position[i] = base + JPH::RVec3(frame * t);
        rotation[i] = (frame * r).Normalized();
    }
}

void RagdollSystem::apply_motors(Ragdoll &doll) {
    const size_t n = doll.parent.size();
    const double *p = doll.desc.params.data();
    for (size_t i = 0; i < n; ++i) {
        JPH::SwingTwistConstraint *joint = swing_twist(*doll.ragdoll, int(i));
        if (joint == nullptr)
            continue;
        const float torque = float(doll.desc.bone(i, kBoneStrength)) * doll.strength;
        for (JPH::MotorSettings *motor : {&joint->GetSwingMotorSettings(), &joint->GetTwistMotorSettings()}) {
            motor->mSpringSettings.mFrequency = float(p[kRagMotorFrequency]);
            motor->mSpringSettings.mDamping = float(p[kRagMotorDamping]);
            motor->SetTorqueLimit(torque);
        }
        const JPH::EMotorState state = torque > 0.0f ? JPH::EMotorState::Position : JPH::EMotorState::Off;
        joint->SetSwingMotorState(state);
        joint->SetTwistMotorState(state);
    }
    doll.motors_dirty = false;
}

void RagdollSystem::before_step(double time, double dt) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const double end = time + dt;
    for (auto &owned : ragdolls_) {
        Ragdoll &doll = *owned;
        const JPH::Array<JPH::BodyID> &ids = doll.ragdoll->GetBodyIDs();
        if (doll.strength != doll.strength_target) {
            const float step = doll.strength_rate * float(dt);
            doll.strength = doll.strength < doll.strength_target
                                ? std::min(doll.strength_target, doll.strength + step)
                                : std::max(doll.strength_target, doll.strength - step);
            doll.motors_dirty = true;
        }
        if (doll.mode == RagdollMode::Kinematic) {
            target(doll, end, scratch_position_, scratch_rotation_);
            if (doll.blend_time > 0.0) {
                const double s = (end - doll.blend_start) / doll.blend_time;
                if (s >= 1.0) {
                    doll.blend_time = 0.0;
                } else {
                    for (size_t i = 0; i < ids.size(); ++i) {
                        const JPH::RVec3 a = doll.from_position[i];
                        scratch_position_[i] = a + (scratch_position_[i] - a) * s;
                        scratch_rotation_[i] = slerp(doll.from_rotation[i], scratch_rotation_[i], float(s));
                    }
                }
            }
            for (size_t i = 0; i < ids.size(); ++i)
                bodies.MoveKinematic(ids[i], scratch_position_[i], scratch_rotation_[i], float(dt));
            continue;
        }
        if (doll.motors_dirty)
            apply_motors(doll);
        if (doll.strength <= 0.0f)
            continue;
        target(doll, end, scratch_position_, scratch_rotation_, &scratch_local_);
        for (size_t i = 0; i < ids.size(); ++i)
            if (JPH::SwingTwistConstraint *joint = swing_twist(*doll.ragdoll, int(i)))
                joint->SetTargetOrientationBS(scratch_local_[i]);
        bool asleep = false;
        for (const JPH::BodyID &id : ids)
            asleep = asleep || !bodies.IsActive(id);
        if (asleep)
            bodies.ActivateBodies(ids.data(), int(ids.size()));
        const double *p = doll.desc.params.data();
        const float force_limit = float(p[kRagRootForce]) * doll.strength;
        const float torque_limit = float(p[kRagRootTorque]) * doll.strength;
        if (force_limit <= 0.0f && torque_limit <= 0.0f)
            continue;
        // The root's spring carries the whole ragdoll: its stiffness and the
        // gravity it holds are scaled by the ragdoll's mass.
        float mass = 0.0f;
        for (size_t i = 0; i < ids.size(); ++i)
            mass += float(doll.desc.bone(i, kBoneMass));
        const float omega = float(kTwoPi * p[kRagRootFrequency]);
        const float zeta = float(p[kRagRootDamping]);
        const JPH::Body *root = locks.TryGetBody(ids[0]);
        const JPH::Vec3 error(scratch_position_[0] - root->GetPosition());
        const JPH::Vec3 gravity = system_.GetGravity() * float(p[kRagGravityFactor]);
        JPH::Vec3 force = (error * (omega * omega) - root->GetLinearVelocity() * (2.0f * zeta * omega) - gravity) * mass;
        if (force.Length() > force_limit)
            force = force.NormalizedOr(JPH::Vec3::sZero()) * force_limit;
        JPH::Quat turn = (scratch_rotation_[0] * root->GetRotation().Conjugated()).Normalized();
        if (turn.GetW() < 0.0f)
            turn = -turn;
        JPH::Vec3 axis;
        float angle;
        turn.GetAxisAngle(axis, angle);
        const JPH::Vec3 alpha = axis * (angle * omega * omega) - root->GetAngularVelocity() * (2.0f * zeta * omega);
        const JPH::Mat44 inertia = root->GetInverseInertia().Inversed3x3();
        const float scale = mass * root->GetMotionProperties()->GetInverseMass();
        JPH::Vec3 torque = inertia.Multiply3x3(alpha) * scale;
        if (torque.Length() > torque_limit)
            torque = torque.NormalizedOr(JPH::Vec3::sZero()) * torque_limit;
        if (force_limit > 0.0f)
            bodies.AddForce(ids[0], force);
        if (torque_limit > 0.0f)
            bodies.AddTorque(ids[0], torque);
    }
}

void RagdollSystem::write(SnapshotBuffer &out) const {
    size_t bones = 0;
    for (const auto &doll : ragdolls_)
        bones += doll->parent.size();
    out.resize_ragdolls(uint32_t(ragdolls_.size()), uint32_t(bones));
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    uint32_t row = 0;
    uint32_t bone = 0;
    for (const auto &owned : ragdolls_) {
        const Ragdoll &doll = *owned;
        out.rag_id[row] = doll.desc.id;
        out.rag_mode[row] = uint8_t(doll.mode);
        out.rag_strength[row] = doll.strength;
        out.rag_self_collision[row] = doll.desc.params[kRagSelfCollision] != 0.0 ? 1 : 0;
        out.rag_first_bone[row] = bone;
        out.rag_bones[row] = uint32_t(doll.parent.size());
        for (size_t i = 0; i < doll.parent.size(); ++i, ++bone) {
            out.bone_uid[bone] = doll.desc.uids[i];
            const JPH::Body *body = locks.TryGetBody(doll.ragdoll->GetBodyID(int(i)));
            if (body == nullptr)
                continue;
            const JPH::RVec3 position = body->GetPosition();
            const JPH::Quat rotation = body->GetRotation();
            for (int c = 0; c < 3; ++c)
                out.bone_position[size_t(bone) * 3 + c] = position[c];
            out.bone_orientation[size_t(bone) * 4 + 0] = rotation.GetX();
            out.bone_orientation[size_t(bone) * 4 + 1] = rotation.GetY();
            out.bone_orientation[size_t(bone) * 4 + 2] = rotation.GetZ();
            out.bone_orientation[size_t(bone) * 4 + 3] = rotation.GetW();
        }
        ++row;
    }
}

}  // namespace pj
