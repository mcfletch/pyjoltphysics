// Events: contacts, trigger overlaps, sleep and wake, as fixed 64-byte records.
//
// Jolt's listeners run on its job threads during Update and report each pair
// of sub-shapes. They append raw reports under a mutex; after Update the world
// thread sorts them, so a run's events come out in the same order whatever the
// thread count, and folds them into one record per body pair:
//
//   begin    the pair's first touching sub-shape pair appears
//   persist  once a tick while the pair touches, only where either body has
//            the report flag
//   end      the pair's last touching sub-shape pair goes, or a body of the
//            pair is removed (reason 1)
//
// Jolt drops a sleeping body's contacts and adds them again when it wakes.
// A pair whose contacts go while none of its bodies is awake is held as
// touching: there is no end for it, and no begin when the contacts come back
// on waking. Should the pair have parted once awake, its end follows in the
// tick the contacts fail to come back.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace JPH {
class BodyInterface;
}

namespace pj {

enum class EventKind : uint8_t { Contact = 0, Trigger = 1, State = 2 };
enum class EventPhase : uint8_t { Begin = 0, Persist = 1, End = 2, Sleep = 3, Wake = 4 };
enum class EndReason : uint16_t { None = 0, Removed = 1 };

// The record Python reads as pyjoltphysics.EVENT_DTYPE. For a contact a is the
// lower uid and normal points from a to b; for a trigger a is the sensor; for
// sleep and wake a is the body and b is 0.
struct Event {
    uint8_t kind;      // EventKind
    uint8_t phase;     // EventPhase
    uint16_t reason;   // EndReason, for end records
    uint32_t tick;     // the tick the event happened in (the snapshot's tick)
    Uid a;
    Uid b;
    double point[3];   // world position of the contact, metres
    float normal[3];   // unit vector from a to b
    float approach;    // closing speed along the normal at the contact, m/s
    // Estimated normal impulse of the collision, N s, contact begin only.
    // Jolt calls its listeners before the solver runs, so no record carries
    // the impulse the solver applied.
    float impulse;
    uint32_t reserved;  // zero
};
static_assert(sizeof(Event) == 64, "the event record is 64 bytes");

// What one listener call reports, before aggregation.
struct RawContact {
    enum Type : uint8_t { Added = 0, Persisted = 1, Removed = 2 };
    uint64_t key;  // the body pair, from both bodies' index and sequence numbers
    uint64_t sub;  // the sub-shape pair, which orders reports for one body pair
    Type type;
    uint8_t sensors;  // bit 0: body 1 is a sensor, bit 1: body 2
    bool report;
    Uid uid1, uid2;  // in Jolt's body order (body 1 has the lower BodyID)
    double point[3];
    float normal[3];  // from body 1 to body 2
    float approach;
    float impulse;
};

struct RawActivation {
    JPH::BodyID id;
    Uid uid;
    bool active;
};

// Jolt's listener interfaces, collecting raw reports from any thread.
class Collector final : public JPH::ContactListener, public JPH::BodyActivationListener {
public:
    void OnContactAdded(const JPH::Body &body1, const JPH::Body &body2,
                        const JPH::ContactManifold &manifold,
                        JPH::ContactSettings &settings) override;
    void OnContactPersisted(const JPH::Body &body1, const JPH::Body &body2,
                            const JPH::ContactManifold &manifold,
                            JPH::ContactSettings &settings) override;
    void OnContactRemoved(const JPH::SubShapeIDPair &pair) override;
    void OnBodyActivated(const JPH::BodyID &id, JPH::uint64 user) override;
    void OnBodyDeactivated(const JPH::BodyID &id, JPH::uint64 user) override;

    // Activations while the world thread adds or removes bodies are not
    // sleep or wake events.
    std::atomic<bool> quiet{false};
    float min_velocity_for_restitution = 1.0f;

    void take(std::vector<RawContact> &contacts, std::vector<RawActivation> &activations);

private:
    void record(const JPH::Body &body1, const JPH::Body &body2,
                const JPH::ContactManifold &manifold, const JPH::ContactSettings &settings,
                RawContact::Type type);
    std::mutex lock_;
    std::vector<RawContact> contacts_;
    std::vector<RawActivation> activations_;
};

inline uint64_t pair_key(const JPH::BodyID &a, const JPH::BodyID &b) {
    return (uint64_t(a.GetIndexAndSequenceNumber()) << 32) | b.GetIndexAndSequenceNumber();
}

// Body-pair state on the world thread: turns raw reports into records.
class PairTracker {
public:
    // Appends this tick's records to `out`, sorting the raw reports first.
    void fold(uint32_t tick, std::vector<RawContact> &contacts,
              std::vector<RawActivation> &activations, const JPH::BodyInterface &bodies,
              std::vector<Event> &out);
    // Ends every pair a removed body was in, as the removal is applied.
    void remove_bodies(const std::vector<JPH::BodyID> &ids, uint32_t tick, std::vector<Event> &out);
    size_t pairs() const { return pairs_.size(); }

private:
    static constexpr uint32_t kNever = 0xffffffffu;
    struct Pair {
        Uid a = kNoBody, b = kNoBody;  // as the records name them
        bool flip = false;             // a is Jolt's body 2
        bool trigger = false;
        uint32_t contacts = 0;         // touching sub-shape pairs
        bool held = false;             // contacts dropped while asleep; still touching
        uint32_t woke_tick = kNever;   // a body of a held pair woke at this tick
        uint32_t persisted_tick = kNever;
    };
    static Pair start(const RawContact &raw);
    static Event make(const Pair &pair, EventPhase phase, uint32_t tick, const RawContact *raw);
    void end_parted_held(uint32_t tick, bool woke, const JPH::BodyInterface &bodies,
                         std::vector<Event> &out);
    std::unordered_map<uint64_t, Pair> pairs_;
    std::vector<uint64_t> waking_;  // held pairs with a woken body, waiting for contacts
};

// Events waiting for Python. Holds `capacity` records; past that, persist
// records go first, oldest first, then sleep and wake records. Begin and end
// records are never dropped, so a ring that holds only those grows past its
// capacity until Python drains it.
class EventRing {
public:
    explicit EventRing(size_t capacity) : capacity_(capacity) {}
    void append(const std::vector<Event> &events);
    std::vector<Event> drain();
    uint64_t dropped() const { return dropped_.load(); }
    size_t size();

private:
    size_t drop(EventPhase first, EventPhase last, size_t wanted);
    size_t capacity_;
    std::mutex lock_;
    std::vector<Event> events_;
    std::atomic<uint64_t> dropped_{0};
};

}  // namespace pj
