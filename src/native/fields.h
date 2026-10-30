// Fields: gravity zones, quadratic drag and kinematic tracks.
//
// Gravity zones follow OMI_physics_gravity. A zone is a region (a box, a
// sphere or a registered shape, at a pose) and a field: directional, along a
// direction, or point, towards a centre, of some magnitude in m/s2. A body
// whose centre of mass is inside zones takes the world's gravity, then each
// zone's in ascending priority: a `replace` zone sets the gravity to its own,
// a `stop` zone sets it to zero, any other adds its own. The body's gravity
// factor scales the result. The world keeps Jolt's own gravity and adds the
// difference as a force, before every collision step, in a step listener.
//
// Quadratic drag is a force of -k m |v| v on a body with coefficient k
// (1/m), the omi_physics extension `quadraticDrag`.
//
// A kinematic track is a sampled pose (times, positions, rotations) a
// kinematic body follows: each tick the world moves the body to the track's
// pose at the end of the tick (MoveKinematic), so the bodies it carries ride
// it. Positions are interpolated linearly and rotations by slerp; past the
// last sample a track holds its last pose, or loops.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/PhysicsStepListener.h>

#include <unordered_map>
#include <vector>

namespace JPH {
class PhysicsSystem;
class BodyInterface;
}  // namespace JPH

namespace pj {

// Codes shared with pyjoltphysics.ZoneShape and pyjoltphysics.ZoneField.
enum class ZoneShape : uint8_t { Box = 0, Sphere = 1, Shape = 2 };
enum class ZoneField : uint8_t { Directional = 0, Point = 1 };

struct Zone {
    uint8_t shape = uint8_t(ZoneShape::Box);
    uint8_t field = uint8_t(ZoneField::Directional);
    double position[3] = {0, 0, 0};
    double rotation[4] = {0, 0, 0, 1};
    double half[3] = {0.5, 0.5, 0.5};  // box half extents; sphere radius in [0]
    JPH::RefConst<JPH::Shape> region;  // ZoneShape::Shape
    double gravity = 9.81;             // m/s2
    double direction[3] = {0, -1, 0};  // directional
    double center[3] = {0, 0, 0};      // point
    int32_t priority = 0;
    bool replace = false;
    bool stop = false;
};

struct SetGravityZones {
    std::vector<Zone> zones;
};

struct SetDrag {
    std::vector<Uid> uids;
    std::vector<float> drag;  // 1/m
};

struct Track {
    std::vector<double> times;      // (k,) seconds, ascending from 0
    std::vector<double> positions;  // (k, 3)
    std::vector<double> rotations;  // (k, 4)
    bool loop = false;
};

// Starts each body on the track from the tick that applies the batch; an
// empty track takes the body off its track.
struct SetTracks {
    std::vector<Uid> uids;
    Track track;
};

// Kinematic targets for the end of this tick, as Body.drive's callables give them.
struct MoveKinematic {
    std::vector<Uid> uids;
    std::vector<double> positions;  // (n, 3)
    std::vector<double> rotations;  // (n, 4)
};

// The gravity at `point` that the zones leave from `base`, m/s2.
JPH::Vec3 zone_gravity(const std::vector<Zone> &zones, JPH::RVec3Arg point, JPH::Vec3Arg base);
// The pose of `track` at `time` seconds after its start.
void track_pose(const Track &track, double time, JPH::RVec3 &position, JPH::Quat &rotation);

// The step listener: zone gravity and drag on awake dynamic bodies, and the
// torques of MOTOR joints. Written only by the world thread with the world
// lock held exclusively; Jolt calls OnStep during Update.
class FieldListener final : public JPH::PhysicsStepListener {
public:
    std::vector<Zone> zones;                       // in ascending priority
    std::unordered_map<uint32_t, float> drag;      // by BodyID index
    // MOTOR joints: (body a, body b, torque in a's frame) a step.
    struct Torque {
        JPH::BodyID a, b;
        JPH::Vec3 local;
    };
    std::vector<Torque> torques;

    void OnStep(const JPH::PhysicsStepListenerContext &context) override;
    bool any() const { return !zones.empty() || !drag.empty() || !torques.empty(); }
};

}  // namespace pj
