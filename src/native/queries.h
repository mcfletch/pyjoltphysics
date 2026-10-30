// Queries: rays, shape casts and overlaps on the caller's thread.
//
// A query reads the world as the last completed tick left it. It takes the
// world lock's shared side, so it waits while the world thread holds the lock
// exclusively (applying batches, moving characters, stepping) and runs beside
// everything else the tick does. On the world thread itself, inside a
// callback, the tick already holds the shared side and a query takes nothing.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <vector>

namespace pj {

class CollisionLayers;

struct QueryFilter {
    uint32_t group = 0xffffffffu;
    uint32_t mask = 0xffffffffu;
    std::vector<Uid> exclude;  // ascending
    bool triggers = false;     // sensor bodies are met
    bool characters = true;    // character proxies are met
};

struct RayResults {
    std::vector<uint8_t> hit;
    std::vector<Uid> uid;
    std::vector<float> fraction;    // of max_distance; 1 for a miss
    std::vector<double> point;      // (n, 3); NaN for a miss
    std::vector<float> normal;      // (n, 3); 0 for a miss
    std::vector<uint32_t> sub_shape;
    void resize(size_t n);
};

struct ShapeHit {
    Uid uid;
    uint32_t sub_shape;
    float fraction;    // of the way from the start to the end
    float depth;       // penetration at the hit, metres; positive where the start overlaps
    double point[3];   // on the body hit
    float normal[3];   // from the body hit towards the cast shape
};

// The object layer side of a filter: a body's group and mask against the query's.
class QueryLayerFilter final : public JPH::ObjectLayerFilter {
public:
    QueryLayerFilter(const CollisionLayers &layers, uint32_t group, uint32_t mask)
        : layers_(layers), group_(group), mask_(mask) {}
    bool ShouldCollide(JPH::ObjectLayer layer) const override;

private:
    const CollisionLayers &layers_;
    uint32_t group_, mask_;
};

// The body side: excluded uids, sensors and character proxies.
class QueryBodyFilter final : public JPH::BodyFilter {
public:
    explicit QueryBodyFilter(const QueryFilter &filter) : filter_(filter) {}
    bool ShouldCollideLocked(const JPH::Body &body) const override;

private:
    const QueryFilter &filter_;
};

}  // namespace pj
