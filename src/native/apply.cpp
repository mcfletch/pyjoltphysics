// Applying command batches, on the world thread with the world lock held
// exclusively. A command naming a uid that is not (or no longer) a body is
// skipped; one the body cannot take is skipped and counted as rejected.
#include "world.h"

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>

namespace pj {

namespace {

JPH::RVec3 rvec3(const std::vector<double> &values, size_t i) {
    return JPH::RVec3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

JPH::Vec3 vec3(const std::vector<double> &values, size_t i) {
    return JPH::Vec3(float(values[3 * i]), float(values[3 * i + 1]), float(values[3 * i + 2]));
}

JPH::Quat quat(const std::vector<double> &values, size_t i) {
    return JPH::Quat(float(values[4 * i]), float(values[4 * i + 1]), float(values[4 * i + 2]),
                     float(values[4 * i + 3]))
        .Normalized();
}

}  // namespace

const WorldCore::Row *WorldCore::find(Uid uid) const {
    if (uid >= row_of_uid_.size() || row_of_uid_[uid] == kNoRow)
        return nullptr;
    return &rows_[row_of_uid_[uid]];
}

void WorldCore::bind_row(Uid uid, const JPH::BodyID &id) {
    const auto row = uint32_t(rows_.size());
    rows_.push_back({uid, id});
    if (row_of_uid_.size() <= uid)
        row_of_uid_.resize(size_t(uid) + 1024, kNoRow);
    row_of_uid_[uid] = row;
    row_of_index_[id.GetIndex()] = row;
    publisher_.mark(row);
}

void WorldCore::unbind_row(Uid uid) {
    const uint32_t row = row_of_uid_[uid];
    row_of_index_[rows_[row].id.GetIndex()] = kNoRow;
    row_of_uid_[uid] = kNoRow;
    const auto last = uint32_t(rows_.size() - 1);
    if (row != last) {
        rows_[row] = rows_[last];
        row_of_uid_[rows_[row].uid] = row;
        row_of_index_[rows_[row].id.GetIndex()] = row;
        publisher_.mark(row);
    }
    rows_.pop_back();
}

void WorldCore::apply(Batch &batch, uint64_t index) {
    std::visit([&](auto &command) { apply(command, index); }, batch.command);
}

void WorldCore::apply(AddBodies &add, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const size_t count = add.uids.size();
    std::vector<JPH::BodyID> made;
    made.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto motion = static_cast<Motion>(add.motion[i]);
        const uint32_t flags = add.flags[i];
        const ShapeParams params{add.shape_params[4 * i], add.shape_params[4 * i + 1],
                                 add.shape_params[4 * i + 2], add.shape_params[4 * i + 3]};
        JPH::RefConst<JPH::Shape> shape =
            shapes_.get(static_cast<ShapeKind>(add.shape_kind[i]), params);
        const bool moving = motion != Motion::Static;
        JPH::BodyCreationSettings settings(shape, rvec3(add.position, i), quat(add.orientation, i),
                                           to_jolt(motion),
                                           layers_.layer(add.group[i], add.mask[i], moving));
        settings.mUserData = user_data(add.uids[i], flags);
        settings.mIsSensor = (flags & flag::kSensor) != 0;
        settings.mAllowSleeping = (flags & flag::kNoSleep) == 0;
        settings.mMotionQuality = (flags & flag::kContinuous) ? JPH::EMotionQuality::LinearCast
                                                              : JPH::EMotionQuality::Discrete;
        settings.mAllowDynamicOrKinematic = moving || (flags & flag::kMovable) != 0;
        settings.mFriction = add.friction[i];
        settings.mRestitution = add.restitution[i];
        settings.mLinearDamping = add.linear_damping[i];
        settings.mAngularDamping = add.angular_damping[i];
        settings.mGravityFactor = add.gravity_factor[i];
        settings.mLinearVelocity = vec3(add.linear_velocity, i);
        settings.mAngularVelocity = vec3(add.angular_velocity, i);
        if (settings.mAllowDynamicOrKinematic) {
            const double mass =
                add.mass[i] > 0.0 ? add.mass[i] : add.density[i] * double(shape->GetVolume());
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = float(mass);
        }
        JPH::Body *body = bodies.CreateBody(settings);
        if (body == nullptr) {
            ++rejected_;
            continue;
        }
        made.push_back(body->GetID());
        bind_row(add.uids[i], body->GetID());
    }
    if (made.empty())
        return;
    collector_.quiet = true;
    JPH::BodyInterface::AddState state = bodies.AddBodiesPrepare(made.data(), int(made.size()));
    bodies.AddBodiesFinalize(made.data(), int(made.size()), state, JPH::EActivation::Activate);
    collector_.quiet = false;
}

void WorldCore::apply(RemoveBodies &remove, uint64_t index) {
    std::vector<JPH::BodyID> ids;
    std::vector<Uid> uids;
    for (Uid uid : remove.uids) {
        const Row *row = find(uid);
        if (row == nullptr)
            continue;
        ids.push_back(row->id);
        uids.push_back(uid);
        unbind_row(uid);
    }
    if (ids.empty())
        return;
    pairs_.remove_bodies(ids, uint32_t(index + 1), tick_events_);
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    collector_.quiet = true;
    bodies.RemoveBodies(ids.data(), int(ids.size()));
    collector_.quiet = false;
    bodies.DestroyBodies(ids.data(), int(ids.size()));
    reserved_bodies_ -= int64_t(ids.size());
}

void WorldCore::apply(SetTransforms &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        const bool fixed = bodies.GetMotionType(row->id) == JPH::EMotionType::Static;
        bodies.SetPositionAndRotation(row->id, rvec3(set.position, i), quat(set.orientation, i),
                                      fixed ? JPH::EActivation::DontActivate
                                            : JPH::EActivation::Activate);
        publisher_.mark(row_of_uid_[set.uids[i]]);
    }
}

void WorldCore::apply(SetVelocities &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        if (bodies.GetMotionType(row->id) == JPH::EMotionType::Static) {
            ++rejected_;
            continue;
        }
        if (!set.linear.empty())
            bodies.SetLinearVelocity(row->id, vec3(set.linear, i));
        if (!set.angular.empty())
            bodies.SetAngularVelocity(row->id, vec3(set.angular, i));
        bodies.ActivateBody(row->id);
        publisher_.mark(row_of_uid_[set.uids[i]]);
    }
}

void WorldCore::apply(AddPushes &push, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < push.uids.size(); ++i) {
        const Row *row = find(push.uids[i]);
        if (row == nullptr)
            continue;
        if (bodies.GetMotionType(row->id) != JPH::EMotionType::Dynamic) {
            ++rejected_;
            continue;
        }
        const JPH::Vec3 value = vec3(push.values, i);
        switch (push.kind) {
        case Push::Force:
            bodies.AddForce(row->id, value);
            break;
        case Push::Impulse:
            bodies.AddImpulse(row->id, value);
            break;
        case Push::Torque:
            bodies.AddTorque(row->id, value);
            break;
        }
    }
}

void WorldCore::apply(SetGravity &set, uint64_t) {
    system_.SetGravity(
        JPH::Vec3(float(set.gravity[0]), float(set.gravity[1]), float(set.gravity[2])));
    // Sleeping bodies do not feel a new gravity until something wakes them.
    std::vector<JPH::BodyID> ids;
    ids.reserve(rows_.size());
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const Row &row : rows_)
        if (bodies.GetMotionType(row.id) == JPH::EMotionType::Dynamic)
            ids.push_back(row.id);
    if (!ids.empty())
        bodies.ActivateBodies(ids.data(), int(ids.size()));
}

void WorldCore::apply(Activate &activate, uint64_t) {
    std::vector<JPH::BodyID> ids;
    for (Uid uid : activate.uids)
        if (const Row *row = find(uid))
            ids.push_back(row->id);
    if (!ids.empty())
        system_.GetBodyInterfaceNoLock().ActivateBodies(ids.data(), int(ids.size()));
}

void WorldCore::apply(SetMotionTypes &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        const auto motion = static_cast<Motion>(set.motion[i]);
        const JPH::Body *body = locks.TryGetBody(row->id);
        if (body == nullptr || (motion != Motion::Static && !body->CanBeKinematicOrDynamic())) {
            ++rejected_;
            continue;
        }
        const bool moving = motion != Motion::Static;
        bodies.SetObjectLayer(row->id, layers_.with_motion(body->GetObjectLayer(), moving));
        bodies.SetMotionType(row->id, to_jolt(motion),
                             moving ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
        publisher_.mark(row_of_uid_[set.uids[i]]);
    }
}

void WorldCore::apply(SetMaterials &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        bodies.SetFriction(row->id, set.friction[i]);
        bodies.SetRestitution(row->id, set.restitution[i]);
    }
}

}  // namespace pj
