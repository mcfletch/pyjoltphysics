// Characters: omi_physics' controller state machine over Jolt's CharacterVirtual.
//
// A character is a capsule moved by collision queries rather than by the
// solver. The world thread moves every character once a tick, under the
// exclusive world lock and before PhysicsSystem::Update, with the inputs the
// tick's batches applied. Characters whose reach shares no dynamic body and no
// other character are independent: each such island is moved by one job, so
// the result is the same whatever the number of job threads.
//
// The vertical motion is the controller's own: a constant gravity integrated
// exactly (v dt - g dt^2 / 2 a tick), handed to CharacterVirtual as the
// velocity that covers that distance in the tick. CharacterVirtual resolves
// the move against the world, walks stairs and sticks to the floor.
#pragma once

#include "common.h"
#include "events.h"

#include <Jolt/Core/JobSystem.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace JPH {
class PhysicsSystem;
}

namespace pj {

class CollisionLayers;
struct SnapshotBuffer;

// Codes shared with pyjoltphysics.CharacterState: the first that holds, in this order.
enum class CharacterState : uint8_t {
    Standing = 0,
    Crouching = 1,
    Sliding = 2,
    Airborne = 3,
    Down = 4,
    Swimming = 5,
    Flying = 6,
};

enum class MoveMode : uint8_t { Walk = 0, Run = 1, Sprint = 2, Crouch = 3 };

// One character input; v holds the values each code names.
enum class CharacterOpCode : uint8_t {
    Move = 0,         // v: direction x, y, z, speed (NaN: the mode's); code: MoveMode
    Jump = 1,
    Slide = 2,        // v: duration, rise; code: steer
    ExtendSlide = 3,  // v: duration
    EndSlide = 4,
    KnockDown = 5,
    GetUp = 6,        // v: start speed (0: none)
    Knockback = 7,    // v: impulse x, y, z (N s)
    Impulse = 8,      // v: velocity x, y, z (m/s), replacing the motion
    Crouch = 9,       // code: 1 crouch, 0 stand
    Fly = 10,         // code: 1 fly, 0 walk
    FlyMove = 11,     // v: direction x, y, z
    Swim = 12,        // code: 1 swim, 0 leave the water; v: buoyancy
    Teleport = 13,    // v: centre x, y, z, search (m below for a floor)
    Gravity = 14,     // v: m/s2 downward
    PassGroup = 15,   // v: group (0: none)
    Count = 16,
};

struct CharacterOp {
    Uid id;
    uint8_t op;
    uint8_t code;
    double v[4];
};

// omi_physics' CharacterCapabilities, and what Jolt adds.
struct CharacterCaps {
    float walk_speed = 3.0f, run_speed = 6.0f, sprint_speed = 9.0f, crouch_speed = 1.5f;
    bool can_sprint = true, can_crouch = true, can_jump = true, can_fly = true;
    float fly_speed = 8.0f, swim_speed = 2.0f, swim_drag = 4.0f;
    float jump_height = 1.2f;
    bool jump_follows_slope = false;
    float step_height = 0.35f;
    float max_slope = 50.0f;  // degrees
    float eye_height = 1.6f, stand_height = 1.8f, crouch_height = 1.0f, radius = 0.3f;
    float air_control = 0.4f;
    float push_friction = 6.0f;      // 1/s, bleeds carried speed on the ground
    float terminal_velocity = 55.0f; // m/s; 0 for none
    bool suppress_over_void = false;
    float coyote_time = 0.12f, jump_buffer = 0.12f;
    float mass = 80.0f;
    float push_mass = 200.0f;        // kg, the heaviest dynamic body walked into and moved
    float push_speed = std::numeric_limits<float>::quiet_NaN();  // NaN: walk_speed
    float push_force = 1000.0f;      // N, the most a character pushes with
    float acceleration = std::numeric_limits<float>::infinity();  // m/s2 toward the input speed
    bool stick_to_floor = true;
    bool smooth_edges = true;        // Jolt's enhanced internal edge removal
};

struct AddCharacter {
    Uid uid = kNoBody;  // assigned when the batch is submitted
    double position[3] = {0, 0, 0};  // capsule centre
    CharacterCaps caps;
    double gravity = 9.81;
    uint32_t group = 1, mask = 0xffffffffu, pass_group = 0;
    bool proxy = true;
    std::vector<Uid> ignore;  // ascending
};

struct RemoveCharacters {
    std::vector<Uid> uids;
};

struct CharacterOps {
    std::vector<CharacterOp> ops;
};

struct SetCharacterIgnore {
    Uid id;
    std::vector<Uid> ignore;  // ascending
};

struct Character;

class CharacterSystem {
public:
    CharacterSystem(JPH::PhysicsSystem &system, const CollisionLayers &layers, int threads);
    ~CharacterSystem();
    CharacterSystem(const CharacterSystem &) = delete;
    CharacterSystem &operator=(const CharacterSystem &) = delete;

    // On the world thread with the world lock held exclusively.
    void add(const AddCharacter &add, JPH::ObjectLayer proxy_layer, JPH::TempAllocator &temp);
    // Returns the proxy bodies of the characters removed, for the world to remove.
    std::vector<JPH::BodyID> remove(const std::vector<Uid> &uids);
    // False for an op naming no character.
    bool apply(const CharacterOp &op, uint32_t tick, JPH::TempAllocator &temp);
    // Moves the events the ops made into `events`, in id order.
    void flush_events(std::vector<Event> &events);
    void set_ignore(Uid id, std::vector<Uid> ignore);
    // Moves every character by dt; `tick` is the tick the move belongs to.
    void step(double dt, uint32_t tick, JPH::JobSystem &jobs, std::vector<Event> &events);

    // On the world thread with at least the shared side of the world lock.
    void write(SnapshotBuffer &buffer) const;
    size_t size() const { return characters_.size(); }
    bool empty() const { return characters_.empty(); }
    // Every proxy body, with the characters forgotten; for the world's destructor.
    std::vector<JPH::BodyID> remove_all();

private:
    Character *find(Uid id);
    const Character *find(Uid id) const;
    void islands(double dt);

    JPH::PhysicsSystem &system_;
    const CollisionLayers &layers_;
    std::vector<std::unique_ptr<Character>> characters_;  // ascending id
    std::vector<std::unique_ptr<JPH::TempAllocator>> temps_;  // one a job
    std::vector<uint32_t> island_of_;
    std::vector<std::vector<uint32_t>> islands_;
    struct Listener;
    std::unique_ptr<Listener> listener_;
};

}  // namespace pj
