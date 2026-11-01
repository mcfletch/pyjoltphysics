#include "layers.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace pj {

JPH::ObjectLayer CollisionLayers::acquire(const LayerKey &key, uint32_t users) {
    std::lock_guard<std::mutex> guard(lock_);
    auto found = index_.find(key);
    if (found != index_.end()) {
        entries_[found->second].users += users;
        return found->second;
    }
    JPH::ObjectLayer made;
    const Entry entry{key.group, key.mask, key.moving, key.role, key.filter, users};
    if (!free_.empty()) {
        made = free_.back();
        free_.pop_back();
        entries_[made] = entry;
    } else if (entries_.size() < kMaxLayers) {
        made = JPH::ObjectLayer(entries_.size());
        entries_.push_back(entry);
    } else {
        return kNoLayer;
    }
    index_.emplace(key, made);
    return made;
}

void CollisionLayers::retain(JPH::ObjectLayer layer) {
    std::lock_guard<std::mutex> guard(lock_);
    ++entries_[layer].users;
}

void CollisionLayers::release(JPH::ObjectLayer layer, uint32_t users) {
    std::lock_guard<std::mutex> guard(lock_);
    Entry &entry = entries_[layer];
    if (entry.users == 0)
        return;
    entry.users -= std::min(users, entry.users);
    if (entry.users != 0)
        return;
    index_.erase(LayerKey{entry.group, entry.mask, entry.moving, entry.role, entry.filter});
    free_.push_back(layer);
}

JPH::ObjectLayer CollisionLayers::with_motion(JPH::ObjectLayer current, bool moving) {
    const Entry &entry = entries_[current];
    return acquire(LayerKey{entry.group, entry.mask, moving, entry.role, entry.filter});
}

JPH::ObjectLayer CollisionLayers::with_filter(JPH::ObjectLayer current, uint16_t filter) {
    const Entry &entry = entries_[current];
    return acquire(LayerKey{entry.group, entry.mask, entry.moving, entry.role, filter});
}

void CollisionLayers::check_room(const std::vector<LayerKey> &keys) const {
    std::set<LayerKey> fresh;
    std::lock_guard<std::mutex> guard(lock_);
    for (const LayerKey &key : keys)
        if (index_.count(key) == 0)
            fresh.insert(key);
    if (index_.size() + fresh.size() > kMaxLayers)
        throw std::length_error("the batch needs " + std::to_string(fresh.size()) +
                                " collision layers the world does not have: a world holds at most " +
                                std::to_string(kMaxLayers) +
                                " distinct (group, mask, filter, motion, role) at once, and " +
                                std::to_string(index_.size()) + " are in use");
}

}  // namespace pj
