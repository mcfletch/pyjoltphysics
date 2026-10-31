#include "seams.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/InternalEdgeRemovingCollector.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>

#include <algorithm>
#include <cfloat>
#include <utility>
#include <vector>

namespace pj {

namespace {

// The static bodies whose bounds meet `box`.
class StaticBodies final : public JPH::CollideShapeBodyCollector {
public:
    void AddHit(const JPH::BodyID &body) override { found.push_back(body); }
    std::vector<JPH::BodyID> found;
};

using Face = JPH::CollideShapeResult::Face;

// The unit normal of `face` as its winding gives it; zero for a face of fewer
// than three points or no area.
JPH::Vec3 face_normal(const Face &face) {
    if (face.size() < 3)
        return JPH::Vec3::sZero();
    return (face[1] - face[0]).Cross(face[2] - face[0]).NormalizedOr(JPH::Vec3::sZero());
}

// Marks each hit with the body it was made against, which the collision
// itself leaves unset, so the hits of several bodies can share a collector,
// and keeps every hit's face.
class Marking final : public JPH::CollideShapeCollector {
public:
    explicit Marking(JPH::CollideShapeCollector &next) : next_(next) {}
    void AddHit(const JPH::CollideShapeResult &result) override {
        JPH::CollideShapeResult marked = result;
        marked.mBodyID2 = body;
        faces.push_back({body, result.mShape2Face});
        next_.AddHit(marked);
    }
    JPH::BodyID body;
    std::vector<std::pair<JPH::BodyID, Face>> faces;

private:
    JPH::CollideShapeCollector &next_;
};

// Passes on the hits made against one body. A hit on an edge that the edge
// removal kept, where the edge is shared with a face of another body lying in
// the same plane, is turned to that plane's normal: the two faces are one
// surface, and the edge's own normal would push the moving body off it.
class Keeping final : public JPH::CollideShapeCollector {
public:
    Keeping(JPH::CollideShapeCollector &next, JPH::BodyID body, const Marking &marking, float tolerance_sq)
        : next_(next), body_(body), marking_(marking), tolerance_sq_(tolerance_sq) {}
    void AddHit(const JPH::CollideShapeResult &result) override {
        if (result.mBodyID2 != body_)
            return;
        JPH::CollideShapeResult kept = result;
        flatten(kept);
        next_.AddHit(kept);
        UpdateEarlyOutFraction(next_.GetEarlyOutFraction());
    }

private:
    bool holds(const Face &face, JPH::Vec3Arg point) const {
        for (const JPH::Vec3 &v : face)
            if (v.IsClose(point, tolerance_sq_))
                return true;
        return false;
    }

    void flatten(JPH::CollideShapeResult &hit) const {
        const JPH::Vec3 normal = face_normal(hit.mShape2Face);
        const JPH::Vec3 axis = hit.mPenetrationAxis.NormalizedOr(JPH::Vec3::sZero());
        if (normal.IsNearZero() || axis.IsNearZero() || normal.Dot(-axis) > 0.999848f)  // cos(1 degree)
            return;
        // The edge of the hit's face nearest the contact.
        const Face &face = hit.mShape2Face;
        const JPH::uint count = JPH::uint(face.size());
        float best = FLT_MAX;
        JPH::uint first = 0;
        for (JPH::uint i = 0; i < count; ++i) {
            const JPH::Vec3 a = face[i], b = face[(i + 1) % count];
            const JPH::Vec3 along = b - a;
            const float length_sq = along.LengthSq();
            const float t = length_sq > 0.0f ? std::clamp((hit.mContactPointOn2 - a).Dot(along) / length_sq, 0.0f, 1.0f)
                                             : 0.0f;
            const float distance_sq = (a + t * along - hit.mContactPointOn2).LengthSq();
            if (distance_sq < best) {
                best = distance_sq;
                first = i;
            }
        }
        const JPH::Vec3 a = face[first], b = face[(first + 1) % count];
        for (const auto &[body, other] : marking_.faces) {
            if (body == body_)
                continue;
            const JPH::Vec3 beside = face_normal(other);
            if (beside.Dot(normal) > 0.999848f && holds(other, a) && holds(other, b)) {
                const float length = hit.mPenetrationAxis.Length();
                hit.mPenetrationAxis = -normal * length;
                hit.mPenetrationDepth = (hit.mContactPointOn2 - hit.mContactPointOn1).Dot(normal);
                return;
            }
        }
    }

    JPH::CollideShapeCollector &next_;
    JPH::BodyID body_;
    const Marking &marking_;
    float tolerance_sq_;
};

}  // namespace

void collide_across_seams(const JPH::PhysicsSystem &system, const JPH::Body &moving, const JPH::Body &other,
                          JPH::Mat44Arg moving_transform, JPH::Mat44Arg other_transform,
                          JPH::CollideShapeSettings &settings, JPH::CollideShapeCollector &collector,
                          const JPH::ShapeFilter &filter) {
    if (!moving.GetEnhancedInternalEdgeRemoval() || !other.IsStatic() || moving.IsSensor() || other.IsSensor()) {
        JPH::PhysicsSystem::sDefaultSimCollideBodyVsBody(moving, other, moving_transform, other_transform, settings,
                                                         collector, filter);
        return;
    }
    // The transforms are relative to the moving body's centre of mass.
    const JPH::RVec3 offset = moving.GetCenterOfMassPosition() - JPH::RVec3(moving_transform.GetTranslation());
    JPH::AABox reach = moving.GetWorldSpaceBounds();
    reach.ExpandBy(JPH::Vec3::sReplicate(settings.mMaxSeparationDistance));
    StaticBodies beside;
    system.GetBroadPhaseQuery().CollideAABox(reach, beside, JPH::SpecifiedBroadPhaseLayerFilter(broadphase::kStatic),
                                             system.GetDefaultLayerFilter(moving.GetObjectLayer()));
    // Sorted, so the collector meets the bodies in the same order on every run.
    std::sort(beside.found.begin(), beside.found.end());

    // Marking feeds the edge removal, which feeds Keeping; Keeping reads the
    // faces Marking kept, which are complete by the time the removal's flush
    // passes on the edge hits.
    struct Chain {
        explicit Chain(JPH::CollideShapeCollector &out, JPH::BodyID body, float tolerance_sq)
            : keeping(out, body, marking, tolerance_sq), removing(keeping, tolerance_sq), marking(removing) {}
        Keeping keeping;
        JPH::InternalEdgeRemovingCollector removing;
        Marking marking;
    };
    Chain chain(collector, other.GetID(), settings.mInternalEdgeRemovalVertexToleranceSq);
    Marking &marking = chain.marking;
    JPH::InternalEdgeRemovingCollector &removing = chain.removing;
    settings.mActiveEdgeMode = JPH::EActiveEdgeMode::CollideWithAll;
    const JPH::BodyLockInterfaceNoLock &locks = system.GetBodyLockInterfaceNoLock();
    const JPH::ShapeFilter everything;
    const JPH::SubShapeIDCreator part1, part2;

    marking.body = other.GetID();
    JPH::CollisionDispatch::sCollideShapeVsShape(moving.GetShape(), other.GetShape(), JPH::Vec3::sOne(),
                                                 JPH::Vec3::sOne(), moving_transform, other_transform, part1, part2,
                                                 settings, marking, filter);
    for (const JPH::BodyID &id : beside.found) {
        if (id == other.GetID())
            continue;
        const JPH::Body *body = locks.TryGetBody(id);
        if (body == nullptr || !body->IsStatic() || body->IsSensor() ||
            !moving.GetCollisionGroup().CanCollide(body->GetCollisionGroup()))
            continue;
        marking.body = id;
        const JPH::Mat44 transform = body->GetCenterOfMassTransform().PostTranslated(-offset).ToMat44();
        JPH::CollisionDispatch::sCollideShapeVsShape(moving.GetShape(), body->GetShape(), JPH::Vec3::sOne(),
                                                     JPH::Vec3::sOne(), moving_transform, transform, part1, part2,
                                                     settings, marking, everything);
    }
    removing.Flush();
}

}  // namespace pj
