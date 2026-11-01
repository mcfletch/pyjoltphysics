// Collision layers: which bodies may touch.
//
// Each body carries a collision group and a mask, both 32-bit, and a filter
// index. Two bodies collide when each one's group shares a bit with the
// other's mask and the world's (F, F) filter table lets their two filters
// meet; a filter index outside the table meets every filter. Jolt names a
// body's layer by a 16-bit ObjectLayer, so every distinct
// (group, mask, filter, moving, role) a world uses gets one, assigned on
// first use and counted by its users: a body, a character's proxy and a
// vehicle's wheel caster each hold one use. A layer whose last user goes is
// free for the next new key, so a world holds at most 65535 distinct keys at
// once, however many it uses over its life.
// Static bodies sit in one broad-phase tree and moving bodies in another, and
// two static bodies never collide. A character's proxy meets sensors only.
#pragma once

#include "common.h"

#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace pj {

enum class LayerRole : uint8_t { Body = 0, Sensor = 1, Proxy = 2 };

namespace broadphase {
constexpr JPH::BroadPhaseLayer kStatic(0);
constexpr JPH::BroadPhaseLayer kMoving(1);
constexpr JPH::uint kCount = 2;
}  // namespace broadphase

struct LayerKey {
    uint32_t group = 0;
    uint32_t mask = 0;
    bool moving = false;
    LayerRole role = LayerRole::Body;
    uint16_t filter = 0;
    auto tie() const { return std::make_tuple(group, mask, moving, role, filter); }
    bool operator<(const LayerKey &other) const { return tie() < other.tie(); }
};

// The table is written only by the world thread while it holds the world lock
// exclusively (applying commands); Jolt's job threads read it during Update.
// The key index is also read by submitting threads, under its own mutex.
class CollisionLayers final : public JPH::BroadPhaseLayerInterface,
                              public JPH::ObjectVsBroadPhaseLayerFilter,
                              public JPH::ObjectLayerPairFilter {
public:
    // Layers 0 to 65534; Jolt reserves 65535 as its invalid layer.
    static constexpr size_t kMaxLayers = JPH::cObjectLayerInvalid;
    static constexpr JPH::ObjectLayer kNoLayer = JPH::cObjectLayerInvalid;

    // The layer for this key with `users` more uses, assigned on first use;
    // kNoLayer when every layer is held by another key.
    JPH::ObjectLayer acquire(const LayerKey &key, uint32_t users = 1);
    JPH::ObjectLayer acquire(uint32_t group, uint32_t mask, bool moving, LayerRole role = LayerRole::Body,
                             uint16_t filter = 0, uint32_t users = 1) {
        return acquire(LayerKey{group, mask, moving, role, filter}, users);
    }
    // One more use of a layer in use.
    void retain(JPH::ObjectLayer layer);
    // `users` uses fewer; the last frees the layer for another key.
    void release(JPH::ObjectLayer layer, uint32_t users = 1);
    // The layer with the other motion class, for set_motion_type, with one use; kNoLayer when none is free.
    JPH::ObjectLayer with_motion(JPH::ObjectLayer current, bool moving);
    // The layer with another filter index, with one use; kNoLayer when none is free.
    JPH::ObjectLayer with_filter(JPH::ObjectLayer current, uint16_t filter);
    // Throws std::length_error (ValueError in Python) when the keys not in
    // use now would take the world past kMaxLayers. Any thread; a batch
    // queued ahead of this one can still take the last layers first.
    void check_room(const std::vector<LayerKey> &keys) const;
    // Layers held by at least one user.
    size_t in_use() const {
        std::lock_guard<std::mutex> guard(lock_);
        return index_.size();
    }
    // The (F, F) collides-with table, row major; an empty table lets every filter meet.
    void set_table(uint32_t count, std::vector<uint8_t> table) {
        filter_count_ = count;
        table_ = std::move(table);
    }
    bool filters_meet(uint16_t a, uint16_t b) const {
        return a >= filter_count_ || b >= filter_count_ || table_[size_t(a) * filter_count_ + b] != 0;
    }
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
        uint32_t users;
    };
    std::vector<Entry> entries_;
    std::vector<JPH::ObjectLayer> free_;  // released layers, reused before the table grows
    mutable std::mutex lock_;             // index_, for check_room on submitting threads
    std::map<LayerKey, JPH::ObjectLayer> index_;  // the keys in use
    uint32_t filter_count_ = 0;
    std::vector<uint8_t> table_;
};

}  // namespace pj
