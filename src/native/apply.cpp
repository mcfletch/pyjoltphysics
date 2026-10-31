// Applying command batches, on the world thread with the world lock held
// exclusively. A command naming a uid that is not (or no longer) a body is
// skipped; one the body cannot take is skipped and counted as rejected.
#include "world.h"

#include <cmath>
#include <cstring>

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/StateRecorderImpl.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>

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
    std::vector<JPH::BodyID> sensors;
    size_t statics = 0;
    // Consecutive rows usually share their collision triple.
    uint32_t last_group = 0, last_mask = 0;
    uint16_t last_filter = 0;
    bool last_moving = false, last_sensor = false, have_layer = false;
    JPH::ObjectLayer layer = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto motion = static_cast<Motion>(add.motion[i]);
        const uint32_t flags = add.flags[i];
        const JPH::Shape *shape = add.shapes[i].GetPtr();
        const bool moving = motion != Motion::Static;
        const bool sensor = (flags & flag::kSensor) != 0;
        const uint16_t filter = add.filter[i];
        if (!have_layer || add.group[i] != last_group || add.mask[i] != last_mask ||
            moving != last_moving || sensor != last_sensor || filter != last_filter) {
            layer = layers_.layer(add.group[i], add.mask[i], moving, sensor ? LayerRole::Sensor : LayerRole::Body,
                                  filter);
            last_group = add.group[i];
            last_mask = add.mask[i];
            last_filter = filter;
            last_moving = moving;
            last_sensor = sensor;
            have_layer = true;
        }
        JPH::BodyCreationSettings settings(shape, rvec3(add.position, i), quat(add.orientation, i),
                                           to_jolt(motion), layer);
        settings.mUserData = user_data(add.uids[i], flags, add.material[i]);
        settings.mIsSensor = (flags & flag::kSensor) != 0;
        // A kinematic sensor meets static and kinematic bodies too.
        settings.mCollideKinematicVsNonDynamic = settings.mIsSensor;
        settings.mAllowSleeping = (flags & flag::kNoSleep) == 0;
        settings.mMotionQuality = (flags & flag::kContinuous) ? JPH::EMotionQuality::LinearCast
                                                              : JPH::EMotionQuality::Discrete;
        settings.mEnhancedInternalEdgeRemoval =
            (flags & flag::kSmoothEdges) != 0 ||
            (!moving && (flags & flag::kNoSmoothEdges) == 0 && has_internal_edges(*shape));
        settings.mAllowDynamicOrKinematic = moving || (flags & flag::kMovable) != 0;
        settings.mFriction = add.friction[i];
        settings.mRestitution = add.restitution[i];
        settings.mLinearDamping = add.linear_damping[i];
        settings.mAngularDamping = add.angular_damping[i];
        settings.mGravityFactor = add.gravity_factor[i];
        settings.mLinearVelocity = vec3(add.linear_velocity, i);
        settings.mAngularVelocity = vec3(add.angular_velocity, i);
        if (settings.mAllowDynamicOrKinematic) {
            if (can_be_dynamic(*shape)) {
                const double mass =
                    add.mass[i] > 0.0 ? add.mass[i] : add.density[i] * double(shape->GetVolume());
                settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
                settings.mMassPropertiesOverride.mMass = float(mass);
            } else {
                // Only kinematic: Jolt has no mass for a mesh or height field
                // and a kinematic body's mass meets no solver, so a unit box
                // of the given mass (1 kg by default) stands in.
                const float mass = add.mass[i] > 0.0 ? float(add.mass[i]) : 1.0f;
                settings.mOverrideMassProperties =
                    JPH::EOverrideMassProperties::MassAndInertiaProvided;
                settings.mMassPropertiesOverride.SetMassAndInertiaOfSolidBox(JPH::Vec3::sOne(),
                                                                             mass);
            }
        }
        JPH::Body *body = bodies.CreateBody(settings);
        if (body == nullptr) {
            ++rejected_;
            continue;
        }
        made.push_back(body->GetID());
        bind_row(add.uids[i], body->GetID());
        if (add.drag[i] > 0.0f)
            fields_.drag[body->GetID().GetIndex()] = add.drag[i];
        if (settings.mIsSensor)
            sensors.push_back(body->GetID());
        if (!moving)
            ++statics;
    }
    if (made.empty())
        return;
    collector_.quiet = true;
    JPH::BodyInterface::AddState state = bodies.AddBodiesPrepare(made.data(), int(made.size()));
    bodies.AddBodiesFinalize(made.data(), int(made.size()), state, JPH::EActivation::Activate);
    collector_.quiet = false;
    applied_bodies_ += made.size();
    statics_since_optimize_ += statics;
    for (const JPH::BodyID &id : sensors)
        wake_under_sensor(id);
}

// Jolt tests pairs only where a body is awake, so a sensor added or moved over
// sleeping bodies would not see them. Waking the moving bodies whose bounds
// meet the sensor's lets the next step report them; those at rest fall asleep
// again and are held as inside.
void WorldCore::wake_under_sensor(const JPH::BodyID &sensor) {
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const JPH::Body *body = locks.TryGetBody(sensor);
    if (body == nullptr)
        return;
    JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> hits;
    system_.GetBroadPhaseQuery().CollideAABox(body->GetWorldSpaceBounds(), hits,
                                              JPH::SpecifiedBroadPhaseLayerFilter(broadphase::kMoving));
    std::vector<JPH::BodyID> asleep;
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const JPH::BodyID &id : hits.mHits)
        if (id != sensor && !bodies.IsActive(id))
            asleep.push_back(id);
    if (!asleep.empty())
        bodies.ActivateBodies(asleep.data(), int(asleep.size()));
}

void WorldCore::apply(RemoveBodies &remove, uint64_t index) {
    std::vector<JPH::BodyID> ids;
    std::vector<Uid> uids;
    for (Uid uid : remove.uids) {
        const Row *row = find(uid);
        if (row == nullptr)
            continue;
        if (!ragdolls_->empty() && ragdolls_->owns(uid)) {
            // A bone goes with its ragdoll, by remove_ragdolls.
            ++rejected_;
            continue;
        }
        ids.push_back(row->id);
        uids.push_back(uid);
        unbind_row(uid);
    }
    if (ids.empty())
        return;
    const size_t soft_before = soft_->size();
    soft_->remove(uids);
    applied_soft_ -= soft_before - soft_->size();
    const size_t vehicles_before = vehicles_->size();
    vehicles_->remove_chassis(uids);
    applied_vehicles_ -= vehicles_before - vehicles_->size();
    remove_joints_of(uids, index);
    for (size_t i = 0; i < ids.size(); ++i) {
        fields_.drag.erase(ids[i].GetIndex());
        tracks_.erase(uids[i]);
    }
    pairs_.remove_bodies(ids, uint32_t(index + 1), tick_events_);
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    collector_.quiet = true;
    bodies.RemoveBodies(ids.data(), int(ids.size()));
    collector_.quiet = false;
    bodies.DestroyBodies(ids.data(), int(ids.size()));
    reserved_bodies_ -= int64_t(ids.size());
    applied_bodies_ -= ids.size();
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
        if (flags_of(bodies.GetUserData(row->id)) & flag::kSensor)
            wake_under_sensor(row->id);
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
        const bool at_point = !push.points.empty();
        switch (push.kind) {
        case Push::Force:
            if (at_point)
                bodies.AddForce(row->id, value, rvec3(push.points, i));
            else
                bodies.AddForce(row->id, value);
            break;
        case Push::Impulse:
            if (at_point)
                bodies.AddImpulse(row->id, value, rvec3(push.points, i));
            else
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

void WorldCore::apply(SwapBodies &swap, uint64_t index) {
    apply(swap.remove, index);
    apply(swap.add, index);
}

void WorldCore::apply(AddCharacter &add, uint64_t) {
    const JPH::ObjectLayer layer = layers_.layer(add.group, add.mask, true, LayerRole::Proxy);
    collector_.quiet = true;
    characters_->add(add, layer, *temp_);
    collector_.quiet = false;
    ++applied_characters_;
}

void WorldCore::remove_proxies(std::vector<JPH::BodyID> ids, uint64_t index) {
    if (ids.empty())
        return;
    pairs_.remove_bodies(ids, uint32_t(index + 1), tick_events_);
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    collector_.quiet = true;
    bodies.RemoveBodies(ids.data(), int(ids.size()));
    collector_.quiet = false;
    bodies.DestroyBodies(ids.data(), int(ids.size()));
}

void WorldCore::apply(RemoveCharacters &remove, uint64_t index) {
    const size_t before = characters_->size();
    remove_proxies(characters_->remove(remove.uids), index);
    const size_t removed = before - characters_->size();
    reserved_bodies_ -= int64_t(removed);
    applied_characters_ -= removed;
}

void WorldCore::apply(CharacterOps &ops, uint64_t index) {
    for (const CharacterOp &op : ops.ops)
        if (!characters_->apply(op, uint32_t(index + 1), *temp_))
            ++rejected_;
}

void WorldCore::apply(SetCharacterIgnore &set, uint64_t) { characters_->set_ignore(set.id, std::move(set.ignore)); }

void WorldCore::apply(SetBodyMaterials &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        const uint64_t user = bodies.GetUserData(row->id);
        bodies.SetUserData(row->id, user_data(uid_of(user), flags_of(user), set.material[i]));
    }
}

void WorldCore::apply(SetBodyFilters &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        bodies.SetObjectLayer(row->id, layers_.with_filter(bodies.GetObjectLayer(row->id), set.filter[i]));
        if (bodies.GetMotionType(row->id) != JPH::EMotionType::Static)
            bodies.ActivateBody(row->id);
    }
}

void WorldCore::apply(SetMaterialTable &set, uint64_t) {
    materials_.rows = std::move(set.rows);
    materials_.pairs = std::move(set.pairs);
    materials_.static_speed = set.static_speed;
}

// Every moving body is woken, so pairs the new table parts or joins are met
// at the next step.
void WorldCore::wake_dynamic() {
    std::vector<JPH::BodyID> ids;
    ids.reserve(rows_.size());
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const Row &row : rows_)
        if (bodies.GetMotionType(row.id) == JPH::EMotionType::Dynamic)
            ids.push_back(row.id);
    if (!ids.empty())
        bodies.ActivateBodies(ids.data(), int(ids.size()));
}

void WorldCore::apply(SetFilterTable &set, uint64_t) {
    layers_.set_table(set.count, std::move(set.table));
    wake_dynamic();
}

void WorldCore::apply(SetGravityZones &set, uint64_t) {
    std::stable_sort(set.zones.begin(), set.zones.end(),
                     [](const Zone &a, const Zone &b) { return a.priority < b.priority; });
    fields_.zones = std::move(set.zones);
    characters_->set_zones(&fields_.zones);
    wake_dynamic();
}

void WorldCore::apply(SetDrag &set, uint64_t) {
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        if (set.drag[i] > 0.0f)
            fields_.drag[row->id.GetIndex()] = set.drag[i];
        else
            fields_.drag.erase(row->id.GetIndex());
    }
}

void WorldCore::apply(SetTracks &set, uint64_t index) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (Uid uid : set.uids) {
        const Row *row = find(uid);
        if (row == nullptr)
            continue;
        if (set.track.times.empty()) {
            tracks_.erase(uid);
            continue;
        }
        if (bodies.GetMotionType(row->id) != JPH::EMotionType::Kinematic) {
            ++rejected_;
            continue;
        }
        tracks_[uid] = Tracked{set.track, double(index) * dt_};
    }
}

void WorldCore::apply(MoveKinematic &move, uint64_t) { driven_.push_back(std::move(move)); }

void WorldCore::move_kinematic_now(MoveKinematic &&move) { driven_.push_back(std::move(move)); }

// Tracks and Body.drive's targets move their kinematic bodies to where they
// are at the end of this tick.
void WorldCore::move_kinematics(uint64_t index) {
    if (tracks_.empty() && driven_.empty())
        return;
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const double end = double(index + 1) * dt_;
    const auto step = float(dt_);
    for (const auto &[uid, tracked] : tracks_) {
        const Row *row = find(uid);
        if (row == nullptr)
            continue;
        JPH::RVec3 position;
        JPH::Quat rotation;
        track_pose(tracked.track, end - tracked.start, position, rotation);
        bodies.MoveKinematic(row->id, position, rotation, step);
        publisher_.mark(row_of_uid_[uid]);
    }
    for (const MoveKinematic &move : driven_) {
        for (size_t i = 0; i < move.uids.size(); ++i) {
            const Row *row = find(move.uids[i]);
            if (row == nullptr || bodies.GetMotionType(row->id) != JPH::EMotionType::Kinematic) {
                ++rejected_;
                continue;
            }
            bodies.MoveKinematic(row->id, rvec3(move.positions, i), quat(move.rotations, i), step);
            publisher_.mark(row_of_uid_[move.uids[i]]);
        }
    }
    driven_.clear();
}

void WorldCore::apply(SetMotion &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (size_t i = 0; i < set.uids.size(); ++i) {
        const Row *row = find(set.uids[i]);
        if (row == nullptr)
            continue;
        JPH::Body *body = locks.TryGetBody(row->id);
        if (body == nullptr || body->IsStatic()) {
            ++rejected_;
            continue;
        }
        JPH::MotionProperties *motion = body->GetMotionProperties();
        if (!std::isnan(set.linear_damping[i]))
            motion->SetLinearDamping(set.linear_damping[i]);
        if (!std::isnan(set.angular_damping[i]))
            motion->SetAngularDamping(set.angular_damping[i]);
        if (!std::isnan(set.gravity_factor[i]))
            motion->SetGravityFactor(set.gravity_factor[i]);
        if (!std::isnan(set.mass[i]) && body->IsDynamic()) {
            // The shape keeps its distribution of mass: the inertia scales with the mass.
            const float old_inverse = motion->GetInverseMass();
            const float new_inverse = 1.0f / set.mass[i];
            const float scale = old_inverse > 0.0f ? new_inverse / old_inverse : 1.0f;
            motion->SetInverseMass(new_inverse);
            motion->SetInverseInertia(motion->GetInverseInertiaDiagonal() * scale, motion->GetInertiaRotation());
        }
        if (!std::isnan(set.inertia[9 * i]) && body->IsDynamic()) {
            JPH::MassProperties given;
            given.mMass = 1.0f / motion->GetInverseMass();
            given.mInertia = JPH::Mat44::sIdentity();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    given.mInertia(r, c) = set.inertia[9 * i + 3 * r + c];
            JPH::Mat44 rotation;
            JPH::Vec3 diagonal;
            if (given.DecomposePrincipalMomentsOfInertia(rotation, diagonal) && diagonal.ReduceMin() > 0.0f)
                motion->SetInverseInertia(diagonal.Reciprocal(), rotation.GetQuaternion());
            else
                ++rejected_;
        }
        if (body->IsDynamic())
            bodies.ActivateBody(row->id);
    }
}

void WorldCore::apply(SetFlags &set, uint64_t) {
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (Uid uid : set.uids) {
        const Row *row = find(uid);
        if (row == nullptr)
            continue;
        JPH::Body *body = locks.TryGetBody(row->id);
        if (body == nullptr)
            continue;
        const uint64_t user = body->GetUserData();
        const uint32_t flags = set.on ? (flags_of(user) | set.bits) : (flags_of(user) & ~set.bits);
        body->SetUserData(user_data(uid_of(user), flags, material_bits_of(user)));
        if ((set.bits & flag::kNoSleep) != 0 && !body->IsStatic()) {
            body->SetAllowSleeping(!set.on);
            if (set.on)
                bodies.ActivateBody(row->id);
        }
    }
}

void WorldCore::apply(RestoreState &restore, uint64_t) {
    // The state begins with the uids of the bodies it holds, in row order; a
    // world holding other bodies cannot take it.
    const std::string &data = *restore.data;
    uint64_t count = 0;
    if (data.size() < sizeof(count)) {
        ++rejected_;
        return;
    }
    std::memcpy(&count, data.data(), sizeof(count));
    const size_t header = sizeof(count) + count * sizeof(Uid);
    if (data.size() < header || count != rows_.size()) {
        ++rejected_;
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        Uid uid = 0;
        std::memcpy(&uid, data.data() + sizeof(count) + i * sizeof(Uid), sizeof(uid));
        if (uid != rows_[i].uid) {
            ++rejected_;
            return;
        }
    }
    JPH::StateRecorderImpl recorder;
    recorder.WriteBytes(data.data() + header, data.size() - header);
    recorder.Rewind();
    if (!system_.RestoreState(recorder)) {
        ++rejected_;
        return;
    }
    for (uint32_t row = 0; row < rows_.size(); ++row)
        publisher_.mark(row);
}

std::string WorldCore::save_state() {
    check();
    std::unique_lock<WorldLock> exclusive(world_lock_);
    JPH::StateRecorderImpl recorder;
    system_.SaveState(recorder);
    const uint64_t count = rows_.size();
    std::string out(sizeof(count) + count * sizeof(Uid), '\0');
    std::memcpy(out.data(), &count, sizeof(count));
    for (size_t i = 0; i < count; ++i)
        std::memcpy(out.data() + sizeof(count) + i * sizeof(Uid), &rows_[i].uid, sizeof(Uid));
    return out + recorder.GetData();
}

// -- vehicles ----------------------------------------------------------------------

void WorldCore::apply(AddVehicle &add, uint64_t) {
    const Row *row = find(add.desc.chassis);
    if (row == nullptr || !vehicles_->add(add.desc, row->id)) {
        ++rejected_;
        return;
    }
    ++applied_vehicles_;
}

void WorldCore::apply(RemoveVehicles &remove, uint64_t) {
    const size_t before = vehicles_->size();
    vehicles_->remove(remove.ids);
    applied_vehicles_ -= before - vehicles_->size();
}

void WorldCore::apply(VehicleInputs &inputs, uint64_t) {
    for (size_t i = 0; i < inputs.ids.size(); ++i)
        if (!vehicles_->input(inputs.ids[i], &inputs.values[4 * i]))
            ++rejected_;
}

void WorldCore::apply(SetVehicleSurfaces &set, uint64_t) {
    for (size_t i = 0; i < set.ids.size(); ++i)
        if (!vehicles_->surface(set.ids[i], set.wheel[i], set.grip[i], set.rolling[i]))
            ++rejected_;
}

void WorldCore::apply(SetSurfaceTable &set, uint64_t) {
    vehicles_->set_table(std::move(set.grip), std::move(set.rolling));
}



// -- ragdolls ----------------------------------------------------------------------

void WorldCore::apply(AddRagdoll &add, uint64_t) {
    const RagdollDesc &desc = add.desc;
    const double *p = desc.params.data();
    const JPH::ObjectLayer layer = layers_.layer(uint32_t(p[kRagGroup]), uint32_t(p[kRagMask]), true,
                                                 LayerRole::Body, uint16_t(p[kRagFilter]));
    if (joint_filter_ == nullptr)
        joint_filter_ = new JointPairFilter;
    std::vector<std::pair<int, int>> pairs;
    Ragdoll *doll = ragdolls_->add(desc, layer, joint_filter_, pairs, time_);
    if (doll == nullptr) {
        ++rejected_;
        reserved_bodies_ -= int64_t(desc.uids.size());
        return;
    }
    for (const auto &[a, b] : pairs)
        joint_filter_->add(desc.uids[size_t(a)], desc.uids[size_t(b)]);
    for (size_t i = 0; i < desc.uids.size(); ++i)
        bind_row(desc.uids[i], doll->ragdoll->GetBodyID(int(i)));
    applied_bodies_ += desc.uids.size();
    ++applied_ragdolls_;
}

void WorldCore::remove_ragdoll(RagdollId id, uint64_t index) {
    std::vector<JPH::BodyID> ids;
    std::vector<Uid> uids;
    std::vector<std::pair<int, int>> pairs;
    // The pairs end before the bodies go, so their END records name live rows.
    if (!ragdolls_->owns_ragdoll(id))
        return;
    ragdolls_->bodies_of(id, ids, uids);
    remove_joints_of(uids, index);
    pairs_.remove_bodies(ids, uint32_t(index + 1), tick_events_);
    for (Uid uid : uids)
        unbind_row(uid);
    collector_.quiet = true;
    ragdolls_->remove(id, ids, uids, pairs);
    collector_.quiet = false;
    if (joint_filter_ != nullptr)
        for (const auto &[a, b] : pairs)
            joint_filter_->remove(uids[size_t(a)], uids[size_t(b)]);
    reserved_bodies_ -= int64_t(uids.size());
    applied_bodies_ -= uids.size();
    --applied_ragdolls_;
}

void WorldCore::apply(RemoveRagdolls &remove, uint64_t index) {
    for (RagdollId id : remove.ids)
        remove_ragdoll(id, index);
}

void WorldCore::apply(SetRagdollModes &set, uint64_t) {
    for (RagdollId id : set.ids)
        if (!ragdolls_->set_mode(id, static_cast<RagdollMode>(set.mode), set.blend, time_))
            ++rejected_;
}

void WorldCore::apply(SetRagdollPose &set, uint64_t) {
    if (!ragdolls_->set_pose(set))
        ++rejected_;
}

void WorldCore::apply(SetRagdollClip &set, uint64_t) {
    if (!ragdolls_->set_clip(set, time_))
        ++rejected_;
}

void WorldCore::apply(SetRagdollStrength &set, uint64_t) {
    if (!ragdolls_->set_strength(set.id, set.strength))
        ++rejected_;
}

// -- soft bodies -------------------------------------------------------------------

void WorldCore::apply(AddSoftBody &add, uint64_t index) {
    const SoftBodyDesc &desc = add.desc;
    const double *p = desc.params.data();
    JPH::SoftBodyCreationSettings settings;
    settings.mSettings = desc.shared;
    settings.mPosition = JPH::RVec3(p[kSoftX], p[kSoftY], p[kSoftZ]);
    settings.mRotation = JPH::Quat(float(p[kSoftQX]), float(p[kSoftQY]), float(p[kSoftQZ]), float(p[kSoftQW])).Normalized();
    const auto flags = uint32_t(p[kSoftFlags]);
    settings.mUserData = user_data(desc.uid, flags, uint16_t(p[kSoftMaterial]));
    settings.mObjectLayer = layers_.layer(uint32_t(p[kSoftGroup]), uint32_t(p[kSoftMask]), true, LayerRole::Body,
                                          uint16_t(p[kSoftFilter]));
    settings.mNumIterations = uint32_t(p[kSoftIterations]);
    settings.mLinearDamping = float(p[kSoftLinearDamping]);
    settings.mMaxLinearVelocity = float(p[kSoftMaxVelocity]);
    settings.mRestitution = float(p[kSoftRestitution]);
    settings.mFriction = float(p[kSoftFriction]);
    settings.mPressure = float(p[kSoftPressure]);
    settings.mGravityFactor = float(p[kSoftGravityFactor]);
    settings.mVertexRadius = float(p[kSoftVertexRadius]);
    settings.mAllowSleeping = p[kSoftAllowSleep] != 0.0 && (flags & flag::kNoSleep) == 0;
    settings.mFacesDoubleSided = p[kSoftDoubleSided] != 0.0;
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    JPH::Body *body = bodies.CreateSoftBody(settings);
    if (body == nullptr) {
        ++rejected_;
        reserved_bodies_ -= 1;
        return;
    }
    bind_row(desc.uid, body->GetID());
    bodies.AddBody(body->GetID(), JPH::EActivation::Activate);
    soft_->add(desc, body->GetID(), index);
    applied_bodies_ += 1;
    ++applied_soft_;
}

void WorldCore::apply(SetSoftPins &set, uint64_t) {
    if (!soft_->set_pins(set.uid, std::move(set.pins)))
        ++rejected_;
}

void WorldCore::apply(SetWind &set, uint64_t) {
    for (size_t i = 0; i < set.uids.size(); ++i)
        if (!soft_->set_wind(set.uids[i], JPH::Vec3(float(set.force[3 * i]), float(set.force[3 * i + 1]),
                                                    float(set.force[3 * i + 2]))))
            ++rejected_;
}

}  // namespace pj
