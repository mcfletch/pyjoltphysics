// Joints: Jolt constraints between two bodies, or a body and the world.
//
// A joint is described in world coordinates as the bodies stand when the
// batch adding it is applied, and each body carries its anchor and axes from
// then on. Joint ids are assigned when the batch is submitted, from 1, and
// never reused. Every joint kind but MOTOR is one Jolt constraint; MOTOR (X3D's
// MotorJoint) is torques about up to three axes, applied every step.
//
// After each step the world reads each joint's total lambda (the impulse its
// constraint applied, N s or N m s) and divides it by the step to give the
// force and torque the joint carried. A joint whose force or torque passes its
// break limit is taken out of the world and reported once.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/GroupFilter.h>
#include <Jolt/Physics/Constraints/Constraint.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace JPH {
class PhysicsSystem;
class Body;
}  // namespace JPH

namespace pj {

using JointId = uint32_t;

// Codes shared with pyjoltphysics.JointKind.
enum class JointKind : uint8_t { Fixed = 0, Ball = 1, Hinge = 2, Slider = 3, Distance = 4, SixDOF = 5, Motor = 6 };
// Codes shared with pyjoltphysics.MotorMode: Jolt's EMotorState.
enum class MotorMode : uint8_t { Off = 0, Velocity = 1, Position = 2 };
// Codes shared with pyjoltphysics.JointState.
enum class JointState : uint8_t { Active = 0, Disabled = 1, Broken = 2 };

constexpr double kInf = std::numeric_limits<double>::infinity();

// One joint as a batch carries it. Axis i of the limit, motor and spring
// arrays is the hinge's or slider's one axis at 0; for SIXDOF the six axes
// translation x, y, z then rotation x, y, z of the joint's frame (x along
// `axis`, y along `axis2` made square to it, z their cross product).
struct JointDesc {
    JointId id = 0;  // assigned when the batch is submitted
    uint8_t kind = uint8_t(JointKind::Fixed);
    Uid a = kNoBody;  // the body `b` is held to; 0 for the world
    Uid b = kNoBody;
    double anchor[3] = {0, 0, 0};     // the pivot, or the distance's end on b
    double to_anchor[3] = {0, 0, 0};  // the distance's end on a
    double axis[3] = {1, 0, 0};
    double axis2[3] = {0, 1, 0};
    double axis3[3] = {0, 0, 1};      // MOTOR's third axis
    double lower[6] = {-kInf, -kInf, -kInf, -kInf, -kInf, -kInf};
    double upper[6] = {kInf, kInf, kInf, kInf, kInf, kInf};
    float spring[2] = {0, 0};         // limit spring: frequency Hz (0: rigid), damping ratio
    float friction = 0;               // N or N m holding the free axes
    uint8_t motor_mode[6] = {0, 0, 0, 0, 0, 0};
    double motor_target[6] = {0, 0, 0, 0, 0, 0};  // m/s, rad/s, m, rad; MOTOR: N m about each axis
    double motor_max[6] = {kInf, kInf, kInf, kInf, kInf, kInf};  // N or N m
    float motor_spring[2] = {2, 1};   // position motor: frequency Hz, damping ratio
    float break_force = 0, break_torque = 0;  // N and N m; 0 never breaks
    bool collide = false;             // the two bodies collide with each other
};

struct AddJoints {
    std::vector<JointDesc> joints;
};

struct RemoveJoints {
    std::vector<JointId> ids;
};

// One motor axis of each joint; NaN leaves a value as it is.
struct SetJointMotors {
    std::vector<JointId> ids;
    std::vector<uint8_t> axis;
    std::vector<uint8_t> mode;
    std::vector<double> target;
    std::vector<double> max;
};

struct SetJointsEnabled {
    std::vector<JointId> ids;
    std::vector<uint8_t> enabled;
};

// Pairs of bodies held by a joint that does not let them collide. A body in
// any such pair carries this filter in its collision group, so the broad
// phase asks it about that body's pairs only.
class JointPairFilter final : public JPH::GroupFilter {
public:
    bool CanCollide(const JPH::CollisionGroup &a, const JPH::CollisionGroup &b) const override;
    // On the world thread with the world lock held exclusively.
    void add(Uid a, Uid b);
    void remove(Uid a, Uid b);

private:
    static uint64_t key(Uid a, Uid b) {
        return a < b ? (uint64_t(a) << 32) | b : (uint64_t(b) << 32) | a;
    }
    std::unordered_map<uint64_t, uint32_t> pairs_;  // joints holding each pair
};

// The joints of a world, in ascending id. Owned by the world thread.
struct Joint {
    JointDesc desc;
    JPH::Ref<JPH::Constraint> constraint;  // none for MOTOR
    JPH::BodyID a, b;                      // invalid for the world
    JointState state = JointState::Active;
    // MOTOR: the axes in a's frame (world axes where a is the world).
    JPH::Vec3 axes_a[3];
    // DISTANCE: each end in its body's centre-of-mass frame.
    JPH::Vec3 local_a, local_b;
    float force = 0, torque = 0;  // carried in the last step, N and N m
};

}  // namespace pj
