// Command batches: changes to the world, submitted from any thread and
// applied by the world thread at the start of a tick, before the step.
//
// Every batch holds its values in structure-of-arrays form, one entry per
// body named by uid. Vectors of 3 or 4 components are flattened row by row.
#pragma once

#include "characters.h"
#include "common.h"
#include "fields.h"
#include "joints.h"
#include "materials.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <deque>
#include <mutex>
#include <variant>
#include <memory>
#include <string>
#include <vector>

namespace pj {

struct AddBodies {
    std::vector<Uid> uids;
    std::vector<double> position;      // (n, 3) metres
    std::vector<double> orientation;   // (n, 4) unit quaternions x, y, z, w
    // Resolved when the batch is submitted: the library's shape, decorated
    // with the body's scale, offset and centre of mass. Rows of one shape
    // share one reference.
    std::vector<JPH::RefConst<JPH::Shape>> shapes;
    std::vector<uint8_t> motion;       // Motion
    std::vector<double> mass;          // kg; 0 takes the mass from density
    std::vector<double> density;       // kg/m3, where mass is 0
    std::vector<float> friction;
    std::vector<float> restitution;
    std::vector<float> linear_damping;   // 1/s
    std::vector<float> angular_damping;  // 1/s
    std::vector<float> gravity_factor;
    std::vector<double> linear_velocity;   // (n, 3) m/s
    std::vector<double> angular_velocity;  // (n, 3) rad/s
    std::vector<uint32_t> group;
    std::vector<uint32_t> mask;
    std::vector<uint32_t> flags;  // flag::*
    std::vector<uint16_t> material;  // material index; 0 is the body's own friction and restitution
    std::vector<uint16_t> filter;    // collision filter index
    std::vector<float> drag;         // quadratic drag, 1/m
};

struct RemoveBodies {
    std::vector<Uid> uids;
};

struct SetTransforms {
    std::vector<Uid> uids;
    std::vector<double> position;     // (n, 3)
    std::vector<double> orientation;  // (n, 4)
};

// Either array may be empty, which leaves that velocity as it is.
struct SetVelocities {
    std::vector<Uid> uids;
    std::vector<double> linear;   // (n, 3) m/s
    std::vector<double> angular;  // (n, 3) rad/s
};

// A force or torque acts over the one step that follows; an impulse changes
// the velocity at once.
enum class Push : uint8_t { Force, Impulse, Torque };

struct AddPushes {
    Push kind;
    std::vector<Uid> uids;
    std::vector<double> values;  // (n, 3) N, N s or N m
};

struct SetGravity {
    double gravity[3];  // m/s2
};

struct Activate {
    std::vector<Uid> uids;
};

struct SetMotionTypes {
    std::vector<Uid> uids;
    std::vector<uint8_t> motion;
};

struct SetMaterials {
    std::vector<Uid> uids;
    std::vector<float> friction;
    std::vector<float> restitution;
};

struct SetBodyMaterials {
    std::vector<Uid> uids;
    std::vector<uint16_t> material;
};

struct SetBodyFilters {
    std::vector<Uid> uids;
    std::vector<uint16_t> filter;
};

// Replaces the world's material table.
struct SetMaterialTable {
    std::vector<MaterialRow> rows;
    std::vector<float> pairs;
    float static_speed;
};

// Replaces the world's (F, F) collides-with table, row major.
struct SetFilterTable {
    uint32_t count;
    std::vector<uint8_t> table;
};

// Removes bodies and adds others in one tick, as ground streamed around a
// focus swaps tiles.
struct SwapBodies {
    RemoveBodies remove;
    AddBodies add;
};

// Motion properties of moving bodies; a NaN keeps a value.
struct SetMotion {
    std::vector<Uid> uids;
    std::vector<float> mass;             // kg; scales the inertia with it
    std::vector<float> linear_damping;   // 1/s
    std::vector<float> angular_damping;  // 1/s
    std::vector<float> gravity_factor;
};

// Sets or clears flag bits a body may change after it is added (report, no sleep).
struct SetFlags {
    std::vector<Uid> uids;
    uint32_t bits = 0;
    bool on = false;
};

// A state from save_state, read back into the bodies and constraints.
struct RestoreState {
    std::shared_ptr<const std::string> data;
};

using Command = std::variant<AddBodies, RemoveBodies, SetTransforms, SetVelocities, AddPushes,
                             SetGravity, Activate, SetMotionTypes, SetMaterials, SwapBodies,
                             AddCharacter, RemoveCharacters, CharacterOps, SetCharacterIgnore, AddJoints,
                             RemoveJoints, SetJointMotors, SetJointsEnabled, SetBodyMaterials, SetBodyFilters,
                             SetMaterialTable, SetFilterTable, SetGravityZones, SetDrag, SetTracks, MoveKinematic,
                             SetMotion, SetFlags, RestoreState>;

// The uids a batch takes when it is submitted: one a body it adds, one a character.
uint32_t uids_wanted(const Command &command);

struct Batch {
    uint64_t seq = 0;
    uint64_t at_tick = 0;  // applied at the first tick whose index is at least this
    Command command;
};

// Batches in submission order. The lock is held only to append or to take
// the due batches; sequence numbers and uids are assigned under it, so a
// replay that submits the same batches in the same order gets the same uids.
class CommandQueue {
public:
    // Appends the batch and returns its sequence number. For a batch that adds
    // bodies or a character, `uid_count` consecutive uids are assigned first,
    // written into the batch, and the first is stored in `first_uid`. A batch
    // that adds joints takes consecutive joint ids, the first stored in
    // `first_joint`.
    uint64_t push(Batch &&batch, uint32_t uid_count = 0, Uid *first_uid = nullptr, JointId *first_joint = nullptr);
    // Moves every batch due at `tick` into `out`, in sequence order.
    void take_due(uint64_t tick, std::vector<Batch> &out);
    bool empty() const;
    uint64_t last_seq() const;

private:
    mutable std::mutex lock_;
    std::deque<Batch> queue_;
    uint64_t seq_ = 0;
    Uid next_uid_ = 1;
    JointId next_joint_ = 1;
};

}  // namespace pj
