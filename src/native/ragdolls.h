// Ragdolls: a skeleton of dynamic bodies joined by Jolt swing-twist
// constraints, built with Jolt's Ragdoll.
//
// Each bone is a body whose frame is its joint's frame: the body stands where
// the joint stands, turned as the joint is turned, and its shape is placed in
// that frame. The snapshot's bone poses are therefore joint poses, the
// matrices a renderer skins with. A bone's uid is assigned when its batch is
// submitted, as a body's is, and the bone is a body row of every snapshot.
//
// A ragdoll is in one of three modes:
//   kinematic  every bone is a kinematic body moved each tick to the pose
//              target (MoveKinematic), so it reaches the pose at the tick's end
//   powered    dynamic bodies; each joint's swing and twist motors turn the
//              bone towards its target rotation relative to its parent, with
//              a torque limited per bone, and the root bone is pulled towards
//              its target pose by a damped spring of limited force and torque
//   limp       dynamic bodies, motors off; joint friction only
// The pose target is a held pose (local rotations and translations per
// joint, in the parent joint's frame) or a sampled clip evaluated natively at
// the end of each tick, placed in the world by the ragdoll's placement.
//
// A mode change may blend. Into powered or limp the motors' strength moves
// linearly to 1 or 0 over the blend; kinematic to either keeps the bones'
// velocities. Into kinematic the targets move from the bones' current poses
// to the pose target over the blend.
//
// Bones of one ragdoll do not collide with their parent or with bones whose
// shapes overlap in the rest pose (with self-collision off, with none of
// them); the world's groups, masks and filter table decide the rest.
#pragma once

#include "common.h"
#include "shapes.h"

#include <Jolt/Physics/Ragdoll/Ragdoll.h>

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace JPH {
class PhysicsSystem;
}

namespace pj {

struct SnapshotBuffer;

using RagdollId = uint32_t;

// Codes shared with pyjoltphysics.RagdollMode.
enum class RagdollMode : uint8_t { Kinematic = 0, Powered = 1, Limp = 2 };

// The per-bone settings, in the order pyjoltphysics.ragdolls sends them.
enum BoneParam : uint32_t {
    kBoneParent,                        // -1 for the root; parents before children
    kBoneTX, kBoneTY, kBoneTZ,          // rest translation in the parent joint's frame
    kBoneRX, kBoneRY, kBoneRZ, kBoneRW, // rest rotation in the parent joint's frame
    kBoneOX, kBoneOY, kBoneOZ,          // shape offset in the joint's frame
    kBoneQX, kBoneQY, kBoneQZ, kBoneQW, // shape rotation in the joint's frame
    kBoneMass,                          // kg
    kBoneTwistX, kBoneTwistY, kBoneTwistZ,  // twist axis, joint frame
    kBonePlaneX, kBonePlaneY, kBonePlaneZ,  // plane axis, joint frame, square to the twist axis
    kBoneTwistMin, kBoneTwistMax,       // rad
    kBoneNormalCone, kBonePlaneCone,    // half angles, rad
    kBoneStrength,                      // N m: the motors' torque limit when powered
    kBoneFriction,                      // N m: joint friction
    kBoneParamCount,
};

// The ragdoll's scalar settings, in the order pyjoltphysics.ragdolls sends them.
enum RagdollParam : uint32_t {
    kRagX, kRagY, kRagZ, kRagQX, kRagQY, kRagQZ, kRagQW,  // placement
    kRagMode,
    kRagSelfCollision,          // 1: non-neighbouring bones collide
    kRagMotorFrequency,         // Hz of the joint motors' springs
    kRagMotorDamping,           // their damping ratio
    kRagRootFrequency,          // Hz of the root's spring when powered
    kRagRootDamping,
    kRagRootForce,              // N; 0: the root is not driven
    kRagRootTorque,             // N m
    kRagFriction, kRagRestitution, kRagLinearDamping, kRagAngularDamping, kRagGravityFactor,
    kRagGroup, kRagMask, kRagFilter, kRagMaterial, kRagFlags,
    kRagParamCount,
};

struct RagdollDesc {
    RagdollId id = 0;            // assigned when the batch is submitted
    std::vector<Uid> uids;       // one a bone, assigned when the batch is submitted
    std::vector<double> params;  // kRagParamCount
    std::vector<double> bones;   // (B, kBoneParamCount)
    std::vector<JPH::RefConst<JPH::Shape>> shapes;  // one a bone, decorated
    size_t bone_count() const { return bones.size() / kBoneParamCount; }
    double bone(size_t i, BoneParam p) const { return bones[i * kBoneParamCount + p]; }
};

struct AddRagdoll {
    RagdollDesc desc;
};

struct RemoveRagdolls {
    std::vector<RagdollId> ids;
};

struct SetRagdollModes {
    std::vector<RagdollId> ids;
    uint8_t mode = 0;
    double blend = 0.0;  // seconds
};

// A held pose target. Empty rotations keep the current ones; empty
// translations keep the current ones; NaN placement keeps the placement.
struct SetRagdollPose {
    RagdollId id = 0;
    std::vector<double> rotations;     // (B, 4)
    std::vector<double> translations;  // (B, 3)
    double position[3];
    double rotation[4];
};

// A clip sampled at `times` (ascending from 0, seconds after the tick that
// applies the batch); empty times clear it, and the held pose serves again.
struct SetRagdollClip {
    RagdollId id = 0;
    std::vector<double> times;         // (K,)
    std::vector<double> rotations;     // (K, B, 4)
    std::vector<double> translations;  // (K, B, 3), or empty for the held translations
    bool loop = false;
};

struct SetRagdollStrength {
    RagdollId id = 0;
    std::vector<double> strength;  // (B,) N m
};

struct Ragdoll {
    RagdollDesc desc;
    JPH::Ref<JPH::Ragdoll> ragdoll;
    std::vector<int> parent;
    RagdollMode mode = RagdollMode::Limp;
    float strength = 0.0f;       // the motors' share of their torque, 0 to 1
    float strength_target = 0.0f;
    float strength_rate = 0.0f;  // per second; 0 for at once
    // The pose target: local rotations and translations, and the placement.
    std::vector<JPH::Quat> rotations;
    std::vector<JPH::Vec3> translations;
    JPH::RVec3 position;
    JPH::Quat rotation;
    // The clip, if any.
    std::vector<double> clip_times, clip_rotations, clip_translations;
    bool clip_loop = false;
    double clip_start = 0.0;
    // A blend into kinematic: the bones' world poses when it began.
    std::vector<JPH::RVec3> from_position;
    std::vector<JPH::Quat> from_rotation;
    double blend_start = 0.0, blend_time = 0.0;
    bool motors_dirty = true;
    // Bone pairs kept apart by the group filter, as indices.
    std::vector<std::pair<int, int>> pairs;
};

class RagdollSystem {
public:
    explicit RagdollSystem(JPH::PhysicsSystem &system) : system_(system) {}
    ~RagdollSystem();
    RagdollSystem(const RagdollSystem &) = delete;
    RagdollSystem &operator=(const RagdollSystem &) = delete;

    // On the world thread with the world lock held exclusively. The bodies
    // are created and added with their uids in their user data; `layer` is
    // their object layer and `filter` the group filter that keeps the pairs
    // `pairs` reports apart. Null where Jolt has no room for the bodies.
    Ragdoll *add(const RagdollDesc &desc, JPH::ObjectLayer layer, const JPH::GroupFilter *filter,
                 std::vector<std::pair<int, int>> &pairs, double time);
    // Removes the ragdoll's constraints and bodies; returns false for an unknown id.
    bool remove(RagdollId id, std::vector<JPH::BodyID> &bodies, std::vector<Uid> &uids,
                std::vector<std::pair<int, int>> &pairs);
    bool set_mode(RagdollId id, RagdollMode mode, double blend, double time);
    bool set_pose(const SetRagdollPose &pose);
    bool set_clip(const SetRagdollClip &clip, double time);
    bool set_strength(RagdollId id, const std::vector<double> &strength);
    // Drives every ragdoll towards its pose target at `time + dt`.
    void before_step(double time, double dt);
    void write(SnapshotBuffer &buffer) const;
    bool owns_ragdoll(RagdollId id) const;
    // The ragdoll's bodies and their uids, bone by bone.
    void bodies_of(RagdollId id, std::vector<JPH::BodyID> &bodies, std::vector<Uid> &uids) const;
    // Whether a bone of some ragdoll is this uid.
    bool owns(Uid uid) const;
    // Whether any ragdoll is driven, which keeps a lockstep world stepping.
    bool driving() const;
    std::vector<RagdollId> ids() const;
    bool empty() const { return ragdolls_.empty(); }
    size_t size() const { return ragdolls_.size(); }

private:
    Ragdoll *find(RagdollId id);
    // The pose target's world joint transforms at `time`.
    void target(Ragdoll &doll, double time, std::vector<JPH::RVec3> &position, std::vector<JPH::Quat> &rotation,
                std::vector<JPH::Quat> *local = nullptr) const;
    void enter(Ragdoll &doll, RagdollMode mode, double blend, double time);
    void apply_motors(Ragdoll &doll);

    JPH::PhysicsSystem &system_;
    std::vector<std::unique_ptr<Ragdoll>> ragdolls_;  // ascending id
    std::unordered_set<Uid> bones_;                   // every ragdoll's bone uids
    std::vector<JPH::RVec3> scratch_position_;
    std::vector<JPH::Quat> scratch_rotation_, scratch_local_;
};

}  // namespace pj
