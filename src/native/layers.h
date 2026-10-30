// Collision layers: which bodies may touch.
//
// Each body carries a collision group and a mask, both 32-bit, and a filter
// index. Two bodies collide when each one's group shares a bit with the
// other's mask and the world's (F, F) filter table lets their two filters
// meet; a filter index outside the table meets every filter. Jolt names a
// body's layer by a 16-bit ObjectLayer, so every distinct
// (group, mask, filter, moving, role) a world uses gets one, assigned on
// first use.
// Static bodies sit in one broad-phase tree and moving bodies in another, and
// two static bodies never collide. A character's proxy meets sensors only.
#pragma once

#include "common.h"

#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <map>
#include <tuple>
#include <vector>

namespace pj {

enum class LayerRole : uint8_t { Body = 0, Sensor = 1, Proxy = 2 };

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
    JPH::ObjectLayer layer(uint32_t group, uint32_t mask, bool moving, LayerRole role = LayerRole::Body,
                           uint16_t filter = 0);
    // The same layer with the other motion class, for set_motion_type.
    JPH::ObjectLayer with_motion(JPH::ObjectLayer current, bool moving);
    // The same layer with another filter index.
    JPH::ObjectLayer with_filter(JPH::ObjectLayer current, uint16_t filter);
    // The (F, F) collides-with table, row major; an empty table lets every filter meet.
    void set_table(uint32_t count, std::vector<uint8_t> table) {
        filter_count_ = count;
        table_ = std::move(table);
    }
    bool filters_meet(uint16_t a, uint16_t b) const {
        return a >= filter_count_ || b >= filter_count_ || table_[size_t(a) * filter_count_ + b] != 0;
    }
    size_t size() const { return entries_.size(); }
    // Whether a query or a character of this group and mask meets bodies of `layer`.
    bool meets(JPH::ObjectLayer layer, uint32_t group, uint32_t mask) const {
        const Entry &e = entries_[layer];
        return (e.group & mask) != 0 && (group & e.mask) != 0;
    }

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
        if ((x.role == LayerRole::Proxy || y.role == LayerRole::Proxy) &&
            !(x.role == LayerRole::Sensor || y.role == LayerRole::Sensor))
            return false;
        return (x.moving || y.moving) && (x.group & y.mask) != 0 && (y.group & x.mask) != 0 &&
               filters_meet(x.filter, y.filter);
    }

private:
    struct Entry {
        uint32_t group;
        uint32_t mask;
        bool moving;
        LayerRole role;
        uint16_t filter;
    };
    std::vector<Entry> entries_;
    std::map<std::tuple<uint32_t, uint32_t, bool, LayerRole, uint16_t>, JPH::ObjectLayer> index_;
    uint32_t filter_count_ = 0;
    std::vector<uint8_t> table_;
};

}  // namespace pj
