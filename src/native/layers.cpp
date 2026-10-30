#include "layers.h"

#include <limits>

namespace pj {

JPH::ObjectLayer CollisionLayers::layer(uint32_t group, uint32_t mask, bool moving, LayerRole role) {
    const auto key = std::make_tuple(group, mask, moving, role);
    auto found = index_.find(key);
    if (found != index_.end())
        return found->second;
    if (entries_.size() >= std::numeric_limits<JPH::ObjectLayer>::max())
        throw std::length_error("the world uses every collision layer Jolt can name");
    const auto made = JPH::ObjectLayer(entries_.size());
    entries_.push_back({group, mask, moving, role});
    index_.emplace(key, made);
    return made;
}

JPH::ObjectLayer CollisionLayers::with_motion(JPH::ObjectLayer current, bool moving) {
    const Entry entry = entries_[current];
    return layer(entry.group, entry.mask, moving, entry.role);
}

}  // namespace pj
