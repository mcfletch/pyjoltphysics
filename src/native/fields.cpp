#include "fields.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <algorithm>
#include <cmath>

namespace pj {

namespace {

bool inside(const Zone &zone, JPH::RVec3Arg point) {
    const JPH::RVec3 origin(zone.position[0], zone.position[1], zone.position[2]);
    const JPH::Quat turn = JPH::Quat(float(zone.rotation[0]), float(zone.rotation[1]), float(zone.rotation[2]),
                                     float(zone.rotation[3]))
                               .Normalized();
    const JPH::Vec3 local = turn.Conjugated() * JPH::Vec3(point - origin);
    switch (static_cast<ZoneShape>(zone.shape)) {
    case ZoneShape::Box:
        return std::abs(local.GetX()) <= zone.half[0] && std::abs(local.GetY()) <= zone.half[1] &&
               std::abs(local.GetZ()) <= zone.half[2];
    case ZoneShape::Sphere:
        return double(local.LengthSq()) <= zone.half[0] * zone.half[0];
    case ZoneShape::Shape: {
        if (zone.region == nullptr)
            return false;
        JPH::AnyHitCollisionCollector<JPH::CollidePointCollector> hit;
        zone.region->CollidePoint(local - zone.region->GetCenterOfMass(), JPH::SubShapeIDCreator(), hit);
        return hit.HadHit();
    }
    }
    return false;
}

JPH::Vec3 field_at(const Zone &zone, JPH::RVec3Arg point) {
    JPH::Vec3 direction;
    if (static_cast<ZoneField>(zone.field) == ZoneField::Point)
        direction = JPH::Vec3(JPH::RVec3(zone.center[0], zone.center[1], zone.center[2]) - point);
    else
        direction = JPH::Vec3(float(zone.direction[0]), float(zone.direction[1]), float(zone.direction[2]));
    const float length = direction.Length();
    return length > 1e-12f ? direction * float(zone.gravity / length) : JPH::Vec3::sZero();
}

}  // namespace

JPH::Vec3 zone_gravity(const std::vector<Zone> &zones, JPH::RVec3Arg point, JPH::Vec3Arg base) {
    JPH::Vec3 g = base;
    for (const Zone &zone : zones) {
        if (!inside(zone, point))
            continue;
        if (zone.replace)
            g = field_at(zone, point);
        else if (zone.stop)
            g = JPH::Vec3::sZero();
        else
            g += field_at(zone, point);
    }
    return g;
}

void track_pose(const Track &track, double time, JPH::RVec3 &position, JPH::Quat &rotation) {
    const size_t k = track.times.size();
    auto pose = [&](size_t i) {
        return std::make_pair(
            JPH::RVec3(track.positions[3 * i], track.positions[3 * i + 1], track.positions[3 * i + 2]),
            JPH::Quat(float(track.rotations[4 * i]), float(track.rotations[4 * i + 1]),
                      float(track.rotations[4 * i + 2]), float(track.rotations[4 * i + 3]))
                .Normalized());
    };
    const double first = track.times.front();
    const double last = track.times.back();
    if (track.loop && last > first)
        time = first + std::fmod(std::fmod(time - first, last - first) + (last - first), last - first);
    if (k == 1 || time <= first) {
        std::tie(position, rotation) = pose(0);
        return;
    }
    if (time >= last) {
        std::tie(position, rotation) = pose(k - 1);
        return;
    }
    const size_t hi = size_t(std::upper_bound(track.times.begin(), track.times.end(), time) - track.times.begin());
    const size_t lo = hi - 1;
    const double span = track.times[hi] - track.times[lo];
    const double t = span > 0.0 ? (time - track.times[lo]) / span : 1.0;
    const auto [p0, q0] = pose(lo);
    const auto [p1, q1] = pose(hi);
    position = p0 + (p1 - p0) * t;
    rotation = q0.SLERP(q1, float(t)).Normalized();
}

void FieldListener::OnStep(const JPH::PhysicsStepListenerContext &context) {
    JPH::PhysicsSystem &system = *context.mPhysicsSystem;
    const JPH::BodyLockInterfaceNoLock &locks = system.GetBodyLockInterfaceNoLock();
    if (!zones.empty() || !drag.empty()) {
        const JPH::Vec3 base = system.GetGravity();
        const JPH::BodyID *active = system.GetActiveBodiesUnsafe(JPH::EBodyType::RigidBody);
        const JPH::uint32 count = system.GetNumActiveBodies(JPH::EBodyType::RigidBody);
        for (JPH::uint32 i = 0; i < count; ++i) {
            JPH::Body *body = locks.TryGetBody(active[i]);
            if (body == nullptr || !body->IsDynamic())
                continue;
            JPH::MotionProperties *motion = body->GetMotionProperties();
            const float inverse = motion->GetInverseMass();
            if (inverse <= 0.0f)
                continue;
            const float mass = 1.0f / inverse;
            if (!zones.empty()) {
                const JPH::Vec3 g = zone_gravity(zones, body->GetCenterOfMassPosition(), base);
                if (g != base)
                    body->AddForce((g - base) * (mass * motion->GetGravityFactor()));
            }
            if (!drag.empty()) {
                auto found = drag.find(active[i].GetIndex());
                if (found != drag.end()) {
                    const JPH::Vec3 v = body->GetLinearVelocity();
                    body->AddForce(v * (-found->second * mass * v.Length()));
                }
            }
        }
    }
    for (const Torque &t : torques) {
        JPH::Body *b = t.b.IsInvalid() ? nullptr : locks.TryGetBody(t.b);
        JPH::Body *a = t.a.IsInvalid() ? nullptr : locks.TryGetBody(t.a);
        const JPH::Vec3 world = a != nullptr ? a->GetRotation() * t.local : t.local;
        if (b != nullptr && b->IsDynamic() && b->IsActive())
            b->AddTorque(world);
        if (a != nullptr && a->IsDynamic() && a->IsActive())
            a->AddTorque(-world);
    }
}

}  // namespace pj
