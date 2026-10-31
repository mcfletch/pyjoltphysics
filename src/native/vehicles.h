// Vehicles: Jolt's VehicleConstraint over a dynamic chassis body, with a
// wheeled, motorcycle or tracked controller.
//
// A vehicle is described in the chassis body's own frame and added by a
// batch; its id is assigned when the batch is submitted, from 1, and never
// reused. Inputs (throttle, steer, brake, handbrake) are batches too and hold
// until the next input batch changes them. On the world thread, before each
// PhysicsSystem::Update, every vehicle turns its inputs into the controller's
// driver input: the steering lock falls with speed and the steer angle moves
// at a limited rate; an idle vehicle that comes down to its holding speed
// holds on its brakes until it is driven again; aerodynamic drag, downforce and rolling resistance are forces on
// the chassis for the step. Jolt then casts the wheels against the world (a
// step listener per vehicle, run on its job system) and solves the
// suspension, tyre and drivetrain constraints with every other constraint.
//
// The wheel casts use the chassis's collision layer, so the world's groups,
// masks and filter table decide what a wheel stands on; sensors and character
// proxies are never stood on. A wheel's grip and rolling resistance come from
// a per-wheel surface set by command, or else from the world's surface table
// indexed by the material index of the body under the wheel.
//
// After the step the contact records naming a chassis give each vehicle the
// heaviest blow it took in the tick: the closing speed, the impulse and the
// other body, in the snapshot every tick and as a VEHICLE event when it passes
// the vehicle's reporting speed.
#pragma once

#include "common.h"
#include "events.h"

#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace JPH {
class PhysicsSystem;
}

namespace pj {

struct SnapshotBuffer;

using VehicleId = uint32_t;

// Codes shared with pyjoltphysics.VehicleKind.
enum class VehicleKind : uint8_t { Wheeled = 0, Motorcycle = 1, Tracked = 2 };
// Codes shared with pyjoltphysics.WheelCast.
enum class WheelCast : uint8_t { Ray = 0, Sphere = 1, Cylinder = 2 };

// The scalar settings of a vehicle, in the order pyjoltphysics.vehicles sends them.
enum VehicleParam : uint32_t {
    kForwardX, kForwardY, kForwardZ, kUpX, kUpY, kUpZ,
    kMaxPitchRoll,          // rad; pi for no limit
    kMaxSlope,              // rad: the steepest ground a wheel stands on
    kMaxTorque, kMinRpm, kMaxRpm, kEngineInertia, kEngineDamping,
    kAutomatic,             // 1 auto, 0 manual
    kSwitchTime, kClutchReleaseTime, kSwitchLatency, kShiftUpRpm, kShiftDownRpm, kClutchStrength,
    kDiffLimitedSlip,       // ratio between differentials; inf for open
    kMaxSteer,              // rad: the lock of the steered wheels at rest
    kSteerFalloffSpeed,     // m/s at which the lock has halved; 0 keeps it
    kSteerRate,             // rad/s the steer angle may move; 0 for at once
    kDrag,                  // N at 1 m/s, rising with the square of speed
    kDownforce,             // N at 1 m/s, rising with the square of speed, while any wheel is on the ground
    kRollingResistance,     // fraction of each wheel's load
    kHolding,               // brake input an idle vehicle holds itself with, 0 to 1
    kHoldingSpeed,          // m/s: an idle vehicle this slow holds until it is driven again
    kBodyFriction,          // 1: the ground body's friction scales the tyre's, as Jolt combines it
    kImpactSpeed,           // m/s: blows closing at least this fast are reported as events
    kCrashSpeed,            // m/s: blows closing at least this fast are reported as crashes
    kMaxLeanAngle, kLeanSpring, kLeanDamping, kLeanIntegration, kLeanIntegrationDecay, kLeanSmoothing,
    kCasterRadius,          // sphere cast radius, m; 0 takes each wheel's radius
    kParamCount,
};

// The per-wheel settings, in the order pyjoltphysics.vehicles sends them.
enum WheelParam : uint32_t {
    kWheelX, kWheelY, kWheelZ,  // suspension attachment, body frame
    kRadius, kWidth,
    kSuspensionMin, kSuspensionMax, kPreload,  // m
    kStiffness, kDamping,       // N/m, N s/m
    kSteered,                   // 1: turns with the steering
    kBrakeTorque, kHandbrakeTorque,  // N m
    kWheelInertia, kWheelDamping,
    kGrip,                      // multiplies the friction curves
    kTrackSide,                 // tracked: 0 left, 1 right; -1 none
    kWheelParamCount,
};

struct DifferentialDesc {
    int left = -1, right = -1;
    float ratio = 3.42f, split = 0.5f, limited_slip = 1.4f, torque_ratio = 1.0f;
};

struct AntiRollBarDesc {
    int left = 0, right = 1;
    float stiffness = 1000.0f;
};

struct TrackDesc {
    int driven = -1;
    float inertia = 10.0f, damping = 0.5f, brake_torque = 15000.0f, ratio = 6.0f;
};

struct VehicleDesc {
    VehicleId id = 0;  // assigned when the batch is submitted
    Uid chassis = kNoBody;
    uint8_t kind = uint8_t(VehicleKind::Wheeled);
    uint8_t cast = uint8_t(WheelCast::Ray);
    std::vector<double> params;         // kParamCount
    std::vector<double> wheels;         // (W, kWheelParamCount)
    std::vector<float> torque_curve;    // (x, y) pairs: x is rpm / max rpm
    std::vector<float> longitudinal;    // (slip ratio, friction) pairs
    std::vector<float> lateral;         // (slip angle in degrees, friction) pairs
    std::vector<float> gears, reverse_gears;
    std::vector<DifferentialDesc> differentials;
    std::vector<AntiRollBarDesc> anti_roll_bars;
    TrackDesc tracks[2];
    size_t wheel_count() const { return wheels.size() / kWheelParamCount; }
    double wheel(size_t i, WheelParam p) const { return wheels[i * kWheelParamCount + p]; }
};

struct AddVehicle {
    VehicleDesc desc;
};

struct RemoveVehicles {
    std::vector<VehicleId> ids;
};

struct VehicleInputs {
    std::vector<VehicleId> ids;
    std::vector<float> values;  // (n, 4): throttle -1..1, steer -1..1 (positive left), brake 0..1, handbrake 0..1
};

// A surface under some of a vehicle's wheels; wheel -1 is every wheel. A NaN
// grip gives the wheel back to the world's surface table.
struct SetVehicleSurfaces {
    std::vector<VehicleId> ids;
    std::vector<int32_t> wheel;
    std::vector<float> grip;
    std::vector<float> rolling;
};

// Grip and rolling resistance by material index, for wheels with no surface of their own.
struct SetSurfaceTable {
    std::vector<float> grip;
    std::vector<float> rolling;
};

// A wheel stands on bodies, never on sensors, character proxies or its own chassis.
class WheelBodyFilter final : public JPH::BodyFilter {
public:
    explicit WheelBodyFilter(const JPH::BodyID &chassis) : chassis_(chassis) {}
    bool ShouldCollide(const JPH::BodyID &id) const override { return id != chassis_; }
    bool ShouldCollideLocked(const JPH::Body &body) const override;

private:
    JPH::BodyID chassis_;
};

struct Vehicle {
    VehicleDesc desc;
    JPH::BodyID body;
    JPH::Ref<JPH::VehicleConstraint> constraint;
    JPH::Ref<JPH::VehicleCollisionTester> tester;
    std::unique_ptr<WheelBodyFilter> filter;
    float throttle = 0, steer = 0, brake = 0, handbrake = 0;
    float steer_angle = 0;  // rad, positive left, as last handed to the controller
    bool holding = false;   // idle and held on the brakes since it came down to the holding speed
    std::vector<float> grip, rolling;  // per wheel; NaN grip: the surface table's
    std::vector<uint8_t> steered;
    // The heaviest blow of the tick.
    float impact_speed = 0, impact_impulse = 0;
    Uid impact_uid = kNoBody;
    double impact_point[3] = {0, 0, 0};
    float impact_normal[3] = {0, 0, 0};
};

class VehicleSystem {
public:
    VehicleSystem(JPH::PhysicsSystem &system, double dt) : system_(system), dt_(float(dt)) {}
    ~VehicleSystem();
    VehicleSystem(const VehicleSystem &) = delete;
    VehicleSystem &operator=(const VehicleSystem &) = delete;

    // On the world thread with the world lock held exclusively. False where
    // the chassis is no dynamic body.
    bool add(const VehicleDesc &desc, const JPH::BodyID &chassis);
    void remove(const std::vector<VehicleId> &ids);
    // Removes the vehicles whose chassis is among `uids`, as the bodies go.
    void remove_chassis(const std::vector<Uid> &uids);
    // False for an id naming no vehicle.
    bool input(VehicleId id, const float values[4]);
    bool surface(VehicleId id, int32_t wheel, float grip, float rolling);
    void set_table(std::vector<float> grip, std::vector<float> rolling);
    // Driver inputs and the chassis forces for the coming step of dt.
    void before_step(double dt);
    // Each vehicle's heaviest blow from the tick's contact records, and its events.
    void after_step(uint32_t tick, std::vector<Event> &events);
    void write(SnapshotBuffer &buffer) const;
    void remove_all();
    bool empty() const { return vehicles_.empty(); }
    size_t size() const { return vehicles_.size(); }

private:
    Vehicle *find(VehicleId id);
    void detach(Vehicle &vehicle);
    float table_grip(uint16_t material) const;
    float table_rolling(uint16_t material) const;
    float wheel_grip(const Vehicle &vehicle, size_t wheel, const JPH::Body &ground) const;

    JPH::PhysicsSystem &system_;
    float dt_;
    std::vector<std::unique_ptr<Vehicle>> vehicles_;  // ascending id
    std::unordered_map<Uid, Vehicle *> by_chassis_;
    std::vector<float> table_grip_, table_rolling_;
};

}  // namespace pj
