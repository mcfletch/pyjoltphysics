#include "events.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/EstimateCollisionResponse.h>

#include <algorithm>
#include <tuple>
#include <unordered_set>

namespace pj {

namespace {

void store(float out[3], JPH::Vec3Arg v) {
    out[0] = v.GetX();
    out[1] = v.GetY();
    out[2] = v.GetZ();
}

}  // namespace

// -- Collector: Jolt job threads ------------------------------------------------

void Collector::record(const JPH::Body &body1, const JPH::Body &body2,
                       const JPH::ContactManifold &manifold, const JPH::ContactSettings &settings,
                       RawContact::Type type) {
    RawContact raw{};
    raw.key = pair_key(body1.GetID(), body2.GetID());
    raw.sub = (uint64_t(manifold.mSubShapeID1.GetValue()) << 32) | manifold.mSubShapeID2.GetValue();
    raw.type = type;
    raw.sensors = uint8_t((body1.IsSensor() ? 1 : 0) | (body2.IsSensor() ? 2 : 0));
    raw.report = ((flags_of(body1.GetUserData()) | flags_of(body2.GetUserData())) & flag::kReport) != 0;
    raw.uid1 = uid_of(body1.GetUserData());
    raw.uid2 = uid_of(body2.GetUserData());
    raw.material1 = material_of(body1, manifold.mSubShapeID1);
    raw.material2 = material_of(body2, manifold.mSubShapeID2);

    const JPH::uint count = manifold.mRelativeContactPointsOn1.size();
    JPH::Vec3 sum = JPH::Vec3::sZero();
    for (JPH::uint i = 0; i < count; ++i)
        sum += manifold.mRelativeContactPointsOn1[i];
    const JPH::RVec3 point =
        manifold.mBaseOffset + (count ? sum / float(count) : JPH::Vec3::sZero());
    raw.point[0] = point.GetX();
    raw.point[1] = point.GetY();
    raw.point[2] = point.GetZ();
    store(raw.normal, manifold.mWorldSpaceNormal);
    const JPH::Vec3 closing = body1.GetPointVelocity(point) - body2.GetPointVelocity(point);
    raw.approach = closing.Dot(manifold.mWorldSpaceNormal);
    raw.depth = manifold.mPenetrationDepth;

    if (type == RawContact::Added && !raw.sensors && (body1.IsDynamic() || body2.IsDynamic())) {
        JPH::CollisionEstimationResult result;
        JPH::EstimateCollisionResponse(body1, body2, manifold, result, settings.mCombinedFriction,
                                       settings.mCombinedRestitution,
                                       min_velocity_for_restitution);
        float impulse = 0.0f;
        for (float part : result.mContactImpulse)
            impulse += part;
        raw.impulse = impulse;
    }
    std::lock_guard<std::mutex> guard(lock_);
    contacts_.push_back(raw);
}

namespace {

// Friction and restitution from the two sides' materials, where either has one.
void apply_materials(const MaterialTable *table, const JPH::Body &body1, const JPH::Body &body2,
                     const JPH::ContactManifold &manifold, JPH::ContactSettings &settings) {
    if (table == nullptr)
        return;
    const uint16_t m1 = material_of(body1, manifold.mSubShapeID1);
    const uint16_t m2 = material_of(body2, manifold.mSubShapeID2);
    if (m1 == 0 && m2 == 0)
        return;
    const JPH::uint count = manifold.mRelativeContactPointsOn1.size();
    JPH::Vec3 sum = JPH::Vec3::sZero();
    for (JPH::uint i = 0; i < count; ++i)
        sum += manifold.mRelativeContactPointsOn1[i];
    const JPH::RVec3 point = manifold.mBaseOffset + (count ? sum / float(count) : JPH::Vec3::sZero());
    const JPH::Vec3 relative = body1.GetPointVelocity(point) - body2.GetPointVelocity(point);
    const JPH::Vec3 n = manifold.mWorldSpaceNormal;
    const float slip = (relative - n * relative.Dot(n)).Length();
    table->combine(body1, m1, body2, m2, slip, settings);
}

}  // namespace

void Collector::OnContactAdded(const JPH::Body &body1, const JPH::Body &body2,
                               const JPH::ContactManifold &manifold,
                               JPH::ContactSettings &settings) {
    apply_materials(materials, body1, body2, manifold, settings);
    record(body1, body2, manifold, settings, RawContact::Added);
}

JPH::ValidateResult Collector::OnContactValidate(const JPH::Body &body1, const JPH::Body &body2, JPH::RVec3Arg base,
                                                 const JPH::CollideShapeResult &result) {
    const uint32_t first = flags_of(body1.GetUserData()), second = flags_of(body2.GetUserData());
    const uint32_t marked = filter_either.load(std::memory_order_relaxed) ? (first | second) : (first & second);
    if (filter != nullptr && !body1.IsSensor() && !body2.IsSensor() && (marked & flag::kFilterContacts) &&
        !filter->solve_pair(body1, body2, base, result))
        return JPH::ValidateResult::RejectAllContactsForThisBodyPair;
    return JPH::ValidateResult::AcceptAllContactsForThisBodyPair;
}

void Collector::OnContactPersisted(const JPH::Body &body1, const JPH::Body &body2,
                                   const JPH::ContactManifold &manifold,
                                   JPH::ContactSettings &settings) {
    apply_materials(materials, body1, body2, manifold, settings);
    // Every resting contact persists every step; only reporting bodies pay.
    if (((flags_of(body1.GetUserData()) | flags_of(body2.GetUserData())) & flag::kReport) == 0)
        return;
    record(body1, body2, manifold, settings, RawContact::Persisted);
}

void Collector::OnContactRemoved(const JPH::SubShapeIDPair &pair) {
    RawContact raw{};
    raw.key = pair_key(pair.GetBody1ID(), pair.GetBody2ID());
    raw.sub = (uint64_t(pair.GetSubShapeID1().GetValue()) << 32) | pair.GetSubShapeID2().GetValue();
    raw.type = RawContact::Removed;
    std::lock_guard<std::mutex> guard(lock_);
    contacts_.push_back(raw);
}

void Collector::OnBodyActivated(const JPH::BodyID &id, JPH::uint64 user) {
    if (quiet.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> guard(lock_);
    activations_.push_back({id, uid_of(user), true});
}

void Collector::OnBodyDeactivated(const JPH::BodyID &id, JPH::uint64 user) {
    if (quiet.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> guard(lock_);
    activations_.push_back({id, uid_of(user), false});
}

void Collector::take(std::vector<RawContact> &contacts, std::vector<RawActivation> &activations) {
    std::lock_guard<std::mutex> guard(lock_);
    contacts.swap(contacts_);
    activations.swap(activations_);
    contacts_.clear();
    activations_.clear();
}

// -- PairTracker: the world thread ----------------------------------------------

PairTracker::Pair PairTracker::start(const RawContact &raw) {
    Pair pair;
    pair.trigger = raw.sensors != 0;
    // A trigger record names the sensor first; a contact names the lower uid first.
    if (pair.trigger)
        pair.flip = (raw.sensors & 1) == 0;
    else
        pair.flip = raw.uid2 < raw.uid1;
    pair.a = pair.flip ? raw.uid2 : raw.uid1;
    pair.b = pair.flip ? raw.uid1 : raw.uid2;
    return pair;
}

Event PairTracker::make(const Pair &pair, EventPhase phase, uint32_t tick, const RawContact *raw) {
    Event event{};
    event.kind = uint8_t(pair.trigger ? EventKind::Trigger : EventKind::Contact);
    event.phase = uint8_t(phase);
    event.tick = tick;
    event.a = pair.a;
    event.b = pair.b;
    if (raw) {
        const float sign = pair.flip ? -1.0f : 1.0f;
        for (int i = 0; i < 3; ++i) {
            event.point[i] = raw->point[i];
            event.normal[i] = sign * raw->normal[i];
        }
        event.approach = raw->approach;
        event.impulse = raw->impulse;
        event.depth = raw->depth;
        event.material_a = pair.flip ? raw->material2 : raw->material1;
        event.material_b = pair.flip ? raw->material1 : raw->material2;
    }
    return event;
}

void PairTracker::fold(uint32_t tick, std::vector<RawContact> &contacts,
                       std::vector<RawActivation> &activations, const JPH::BodyInterface &bodies,
                       std::vector<Event> &out) {
    // Additions before removals within a body pair, so a contact moving from
    // one sub-shape to another does not end and begin the pair.
    std::sort(contacts.begin(), contacts.end(), [](const RawContact &x, const RawContact &y) {
        return std::tie(x.key, x.type, x.sub) < std::tie(y.key, y.type, y.sub);
    });
    for (const RawContact &raw : contacts) {
        switch (raw.type) {
        case RawContact::Added: {
            auto [found, made] = pairs_.try_emplace(raw.key, start(raw));
            Pair &pair = found->second;
            if (pair.held) {
                pair.held = false;
                pair.woke_tick = kNever;
            } else if (pair.contacts == 0) {
                out.push_back(make(pair, EventPhase::Begin, tick, &raw));
            }
            (void)made;
            ++pair.contacts;
            break;
        }
        case RawContact::Persisted: {
            auto found = pairs_.find(raw.key);
            if (found != pairs_.end() && found->second.persisted_tick != tick) {
                found->second.persisted_tick = tick;
                out.push_back(make(found->second, EventPhase::Persist, tick, &raw));
            }
            break;
        }
        case RawContact::Removed: {
            auto found = pairs_.find(raw.key);
            if (found == pairs_.end() || found->second.contacts == 0)
                break;
            Pair &pair = found->second;
            if (--pair.contacts > 0)
                break;
            const JPH::BodyID id1(uint32_t(raw.key >> 32));
            const JPH::BodyID id2(uint32_t(raw.key & 0xffffffffu));
            if (!bodies.IsActive(id1) && !bodies.IsActive(id2)) {
                pair.held = true;
            } else {
                out.push_back(make(pair, EventPhase::End, tick, nullptr));
                pairs_.erase(found);
            }
            break;
        }
        }
    }

    std::sort(activations.begin(), activations.end(),
              [](const RawActivation &x, const RawActivation &y) {
                  return std::tie(x.uid, x.active) < std::tie(y.uid, y.active);
              });
    bool woke = false;
    for (const RawActivation &change : activations) {
        Event event{};
        event.kind = uint8_t(EventKind::State);
        event.phase = uint8_t(change.active ? EventPhase::Wake : EventPhase::Sleep);
        event.tick = tick;
        event.a = change.uid;
        out.push_back(event);
        woke = woke || change.active;
    }
    end_parted_held(tick, woke, bodies, out);
}

// A held pair whose body is awake again and whose contacts have not come back
// within one more step has parted.
void PairTracker::end_parted_held(uint32_t tick, bool woke, const JPH::BodyInterface &bodies,
                                  std::vector<Event> &out) {
    if (woke) {
        for (auto &[key, pair] : pairs_) {
            if (!pair.held || pair.woke_tick != kNever)
                continue;
            const JPH::BodyID id1(uint32_t(key >> 32));
            const JPH::BodyID id2(uint32_t(key & 0xffffffffu));
            if (bodies.IsActive(id1) || bodies.IsActive(id2)) {
                pair.woke_tick = tick;
                waking_.push_back(key);
            }
        }
        std::sort(waking_.begin(), waking_.end());
        waking_.erase(std::unique(waking_.begin(), waking_.end()), waking_.end());
    }
    if (waking_.empty())
        return;
    std::vector<uint64_t> still;
    for (uint64_t key : waking_) {
        auto found = pairs_.find(key);
        if (found == pairs_.end() || !found->second.held)
            continue;
        Pair &pair = found->second;
        if (pair.woke_tick == tick) {
            still.push_back(key);
            continue;
        }
        const JPH::BodyID id1(uint32_t(key >> 32));
        const JPH::BodyID id2(uint32_t(key & 0xffffffffu));
        if (bodies.IsActive(id1) || bodies.IsActive(id2)) {
            out.push_back(make(pair, EventPhase::End, tick, nullptr));
            pairs_.erase(found);
        } else {
            pair.woke_tick = kNever;  // asleep again, still held
        }
    }
    waking_.swap(still);
}

void PairTracker::remove_bodies(const std::vector<JPH::BodyID> &ids, uint32_t tick,
                                std::vector<Event> &out) {
    if (ids.empty() || pairs_.empty())
        return;
    std::unordered_set<uint32_t> gone;
    for (const JPH::BodyID &id : ids)
        gone.insert(id.GetIndexAndSequenceNumber());
    std::vector<uint64_t> ended;
    for (const auto &[key, pair] : pairs_)
        if (gone.count(uint32_t(key >> 32)) || gone.count(uint32_t(key & 0xffffffffu)))
            ended.push_back(key);
    std::sort(ended.begin(), ended.end());
    for (uint64_t key : ended) {
        Event event = make(pairs_[key], EventPhase::End, tick, nullptr);
        event.reason = uint16_t(EndReason::Removed);
        out.push_back(event);
        pairs_.erase(key);
    }
}

// -- EventRing: any thread ------------------------------------------------------

size_t EventRing::drop(EventPhase first, EventPhase last, size_t wanted) {
    size_t dropped = 0;
    auto keep = std::remove_if(events_.begin(), events_.end(), [&](const Event &event) {
        if (dropped < wanted && event.phase >= uint8_t(first) && event.phase <= uint8_t(last)) {
            ++dropped;
            return true;
        }
        return false;
    });
    events_.erase(keep, events_.end());
    return dropped;
}

void EventRing::append(const std::vector<Event> &incoming) {
    if (incoming.empty())
        return;
    std::lock_guard<std::mutex> guard(lock_);
    events_.insert(events_.end(), incoming.begin(), incoming.end());
    if (events_.size() <= capacity_)
        return;
    size_t over = events_.size() - capacity_;
    size_t dropped = drop(EventPhase::Persist, EventPhase::Persist, over);
    if (dropped < over)
        dropped += drop(EventPhase::Sleep, EventPhase::Wake, over - dropped);
    dropped_ += dropped;
}

std::vector<Event> EventRing::drain() {
    std::vector<Event> out;
    std::lock_guard<std::mutex> guard(lock_);
    out.swap(events_);
    return out;
}

size_t EventRing::size() {
    std::lock_guard<std::mutex> guard(lock_);
    return events_.size();
}

}  // namespace pj
