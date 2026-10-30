#include "queries.h"

#include "world.h"

#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace pj {

void RayResults::resize(size_t n) {
    hit.assign(n, 0);
    uid.assign(n, kNoBody);
    fraction.assign(n, 1.0f);
    point.assign(n * 3, std::numeric_limits<double>::quiet_NaN());
    normal.assign(n * 3, 0.0f);
    sub_shape.assign(n, 0xffffffffu);
}

bool QueryLayerFilter::ShouldCollide(JPH::ObjectLayer layer) const { return layers_.meets(layer, group_, mask_); }

bool QueryBodyFilter::ShouldCollideLocked(const JPH::Body &body) const {
    const uint64_t user = body.GetUserData();
    if (flags_of(user) & flag::kCharacter) {
        if (!filter_.characters)
            return false;
    } else if (body.IsSensor() && !filter_.triggers) {
        return false;
    }
    const auto &ex = filter_.exclude;
    return ex.empty() || !std::binary_search(ex.begin(), ex.end(), uid_of(user));
}

namespace {

using Clock = std::chrono::steady_clock;

const JPH::BroadPhaseLayerFilter kEveryTree;

void raise_max(std::atomic<uint64_t> &target, uint64_t value) {
    uint64_t seen = target.load();
    while (value > seen && !target.compare_exchange_weak(seen, value)) {
    }
}

}  // namespace

WorldCore::QueryLock::QueryLock(WorldCore &world) : lock_(world.world_lock_, std::defer_lock) {
    world.check();
    if (world.on_world_thread())
        return;
    const auto start = Clock::now();
    lock_.lock();
    const auto waited = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    ++world.queries_;
    world.query_wait_ns_ += waited;
    raise_max(world.query_wait_max_ns_, waited);
}

void WorldCore::raycast(const double *origins, const double *directions, size_t count, double max_distance,
                        const QueryFilter &filter, RayResults &out) {
    out.resize(count);
    QueryLock lock(*this);
    const JPH::NarrowPhaseQuery &query = system_.GetNarrowPhaseQueryNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const QueryLayerFilter layers(layers_, filter.group, filter.mask);
    const QueryBodyFilter bodies(filter);
    JPH::RayCastSettings settings;
    settings.mTreatConvexAsSolid = true;
    for (size_t i = 0; i < count; ++i) {
        const double *o = origins + 3 * i;
        const double *d = directions + 3 * i;
        const double length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (!(length > 0.0) || !std::isfinite(length))
            continue;
        const double scale = max_distance / length;
        const JPH::RRayCast ray(JPH::RVec3(o[0], o[1], o[2]),
                                JPH::Vec3(float(d[0] * scale), float(d[1] * scale), float(d[2] * scale)));
        JPH::ClosestHitCollisionCollector<JPH::CastRayCollector> hit;
        query.CastRay(ray, settings, hit, kEveryTree, layers, bodies);
        if (!hit.HadHit())
            continue;
        const JPH::Body *body = locks.TryGetBody(hit.mHit.mBodyID);
        if (body == nullptr)
            continue;
        const JPH::RVec3 point = ray.GetPointOnRay(hit.mHit.mFraction);
        const JPH::Vec3 normal = body->GetWorldSpaceSurfaceNormal(hit.mHit.mSubShapeID2, point);
        out.hit[i] = 1;
        out.uid[i] = uid_of(body->GetUserData());
        out.fraction[i] = hit.mHit.mFraction;
        out.sub_shape[i] = hit.mHit.mSubShapeID2.GetValue();
        for (int k = 0; k < 3; ++k) {
            out.point[3 * i + size_t(k)] = point[k];
            out.normal[3 * i + size_t(k)] = normal[k];
        }
    }
}

std::vector<ShapeHit> WorldCore::cast_shape(const JPH::Shape &shape, const double start[3], const double rotation[4],
                                            const double end[3], bool all_hits, const QueryFilter &filter) {
    const JPH::RVec3 from(start[0], start[1], start[2]);
    const JPH::Quat q = JPH::Quat(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
                            .Normalized();
    const JPH::Vec3 travel(float(end[0] - start[0]), float(end[1] - start[1]), float(end[2] - start[2]));
    const JPH::RShapeCast cast =
        JPH::RShapeCast::sFromWorldTransform(&shape, JPH::Vec3::sOne(), JPH::RMat44::sRotationTranslation(q, from), travel);
    JPH::ShapeCastSettings settings;
    settings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
    settings.mReturnDeepestPoint = true;
    std::vector<JPH::ShapeCastResult> found;
    {
        QueryLock lock(*this);
        const JPH::NarrowPhaseQuery &query = system_.GetNarrowPhaseQueryNoLock();
        const QueryLayerFilter layers(layers_, filter.group, filter.mask);
        const QueryBodyFilter bodies(filter);
        if (all_hits) {
            JPH::AllHitCollisionCollector<JPH::CastShapeCollector> hits;
            query.CastShape(cast, settings, from, hits, kEveryTree, layers, bodies);
            hits.Sort();
            found.assign(hits.mHits.begin(), hits.mHits.end());
        } else {
            JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> hit;
            query.CastShape(cast, settings, from, hit, kEveryTree, layers, bodies);
            if (hit.HadHit())
                found.push_back(hit.mHit);
        }
        std::vector<ShapeHit> out;
        const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
        for (const JPH::ShapeCastResult &r : found) {
            const JPH::Body *body = locks.TryGetBody(r.mBodyID2);
            if (body == nullptr)
                continue;
            const Uid uid = uid_of(body->GetUserData());
            // The first hit of each body, in order along the cast.
            if (std::any_of(out.begin(), out.end(), [uid](const ShapeHit &h) { return h.uid == uid; }))
                continue;
            ShapeHit h{};
            h.uid = uid;
            h.sub_shape = r.mSubShapeID2.GetValue();
            h.fraction = r.mFraction;
            h.depth = r.mPenetrationDepth;
            const JPH::RVec3 point = from + r.mContactPointOn2;
            const JPH::Vec3 normal = -r.mPenetrationAxis.NormalizedOr(JPH::Vec3::sZero());
            for (int k = 0; k < 3; ++k) {
                h.point[k] = point[k];
                h.normal[k] = normal[k];
            }
            out.push_back(h);
        }
        return out;
    }
}

std::vector<Uid> WorldCore::overlap(const JPH::Shape &shape, const double position[3], const double rotation[4],
                                    const QueryFilter &filter) {
    const JPH::RVec3 at(position[0], position[1], position[2]);
    const JPH::Quat q = JPH::Quat(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
                            .Normalized();
    JPH::CollideShapeSettings settings;
    settings.mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> hits;
    std::vector<Uid> out;
    QueryLock lock(*this);
    const QueryLayerFilter layers(layers_, filter.group, filter.mask);
    const QueryBodyFilter bodies(filter);
    // CollideShape places the shape by its centre of mass.
    const JPH::RMat44 transform = JPH::RMat44::sRotationTranslation(q, at).PreTranslated(shape.GetCenterOfMass());
    system_.GetNarrowPhaseQueryNoLock().CollideShape(&shape, JPH::Vec3::sOne(), transform, settings, at, hits,
                                                     kEveryTree, layers, bodies);
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (const JPH::CollideShapeResult &r : hits.mHits)
        if (const JPH::Body *body = locks.TryGetBody(r.mBodyID2))
            out.push_back(uid_of(body->GetUserData()));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace pj
