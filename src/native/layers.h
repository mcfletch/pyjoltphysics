// Collision layers: which bodies may touch.
//
// Each body carries a collision group and a mask, both 32-bit. Two bodies
// collide when each one's group shares a bit with the other's mask. Jolt
// names a body's layer by a 16-bit ObjectLayer, so every distinct
// (group, mask, moving) triple a world uses gets one, assigned on first use.
// Static bodies sit in one broad-phase tree and moving bodies in another, and
// two static bodies never collide.
#pragma once

#include "common.h"

#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <map>
#include <tuple>
#include <vector>

namespace pj {

namespace broadphase {
constexpr JPH::BroadPhaseLayer kStatic(0);
constexpr JPH::BroadPhaseLayer kMoving(1);
constexpr JPH::uint kCount = 2;
}  // namespace broadphase

// The table is written only by the world thread while it holds the world lock
// exclusively (applying commands); Jolt's job threads read it during Update.
class CollisionLayers final : public JPH::BroadPhaseLayerInterface,
                              public JPH::ObjectVsBroadPhaseLayerFilter,
                              public JPH::ObjectLayerPairFilter {
public:
    // The layer for this triple, assigned on first use. Throws when every
    // 16-bit layer is in use.
    JPH::ObjectLayer layer(uint32_t group, uint32_t mask, bool moving);
    // The same group and mask with the other motion class, for set_motion_type.
    JPH::ObjectLayer with_motion(JPH::ObjectLayer current, bool moving);
    size_t size() const { return entries_.size(); }

    JPH::uint GetNumBroadPhaseLayers() const override { return broadphase::kCount; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return entries_[layer].moving ? broadphase::kMoving : broadphase::kStatic;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        return layer == broadphase::kMoving ? "moving" : "static";
    }
#endif
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer tree) const override {
        return entries_[layer].moving || tree == broadphase::kMoving;
    }
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        const Entry &x = entries_[a];
        const Entry &y = entries_[b];
        return (x.moving || y.moving) && (x.group & y.mask) != 0 && (y.group & x.mask) != 0;
    }

private:
    struct Entry {
        uint32_t group;
        uint32_t mask;
        bool moving;
    };
    std::vector<Entry> entries_;
    std::map<std::tuple<uint32_t, uint32_t, bool>, JPH::ObjectLayer> index_;
};

}  // namespace pj
