#include "vehicles.h"

#include "snapshot.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Vehicle/MotorcycleController.h>
#include <Jolt/Physics/Vehicle/TrackedVehicleController.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>

#include <algorithm>

namespace pj {

namespace {

JPH::Vec3 vec(const std::vector<double> &p, uint32_t at) {
    return JPH::Vec3(float(p[at]), float(p[at + 1]), float(p[at + 2]));
}

void fill_curve(JPH::LinearCurve &curve, const std::vector<float> &pairs) {
    curve.Clear();
    curve.Reserve(JPH::uint(pairs.size() / 2));
    for (size_t i = 0; i + 1 < pairs.size(); i += 2)
        curve.AddPoint(pairs[i], pairs[i + 1]);
}

void fill_engine(JPH::VehicleEngineSettings &engine, const VehicleDesc &d) {
    const auto &p = d.params;
    engine.mMaxTorque = float(p[kMaxTorque]);
    engine.mMinRPM = float(p[kMinRpm]);
    engine.mMaxRPM = float(p[kMaxRpm]);
    engine.mInertia = float(p[kEngineInertia]);
    engine.mAngularDamping = float(p[kEngineDamping]);
    if (!d.torque_curve.empty())
        fill_curve(engine.mNormalizedTorque, d.torque_curve);
}

void fill_transmission(JPH::VehicleTransmissionSettings &t, const VehicleDesc &d) {
    const auto &p = d.params;
    t.mMode = p[kAutomatic] != 0.0 ? JPH::ETransmissionMode::Auto : JPH::ETransmissionMode::Manual;
    if (!d.gears.empty())
        t.mGearRatios.assign(d.gears.begin(), d.gears.end());
    if (!d.reverse_gears.empty())
        t.mReverseGearRatios.assign(d.reverse_gears.begin(), d.reverse_gears.end());
    t.mSwitchTime = float(p[kSwitchTime]);
    t.mClutchReleaseTime = float(p[kClutchReleaseTime]);
    t.mSwitchLatency = float(p[kSwitchLatency]);
    t.mShiftUpRPM = float(p[kShiftUpRpm]);
    t.mShiftDownRPM = float(p[kShiftDownRpm]);
    t.mClutchStrength = float(p[kClutchStrength]);
}

void fill_wheel(JPH::WheelSettings &w, const VehicleDesc &d, size_t i) {
    const JPH::Vec3 up = vec(d.params, kUpX).Normalized();
    const JPH::Vec3 forward = vec(d.params, kForwardX).Normalized();
    w.mPosition = JPH::Vec3(float(d.wheel(i, kWheelX)), float(d.wheel(i, kWheelY)), float(d.wheel(i, kWheelZ)));
    w.mSuspensionDirection = -up;
    w.mSteeringAxis = up;
    w.mWheelUp = up;
    w.mWheelForward = forward;
    w.mSuspensionMinLength = float(d.wheel(i, kSuspensionMin));
    w.mSuspensionMaxLength = float(d.wheel(i, kSuspensionMax));
    w.mSuspensionPreloadLength = float(d.wheel(i, kPreload));
    w.mSuspensionSpring.mMode = JPH::ESpringMode::StiffnessAndDamping;
    w.mSuspensionSpring.mStiffness = float(d.wheel(i, kStiffness));
    w.mSuspensionSpring.mDamping = float(d.wheel(i, kDamping));
    w.mRadius = float(d.wheel(i, kRadius));
    w.mWidth = float(d.wheel(i, kWidth));
}

}  // namespace

bool WheelBodyFilter::ShouldCollideLocked(const JPH::Body &body) const {
    if (body.GetID() == chassis_ || body.IsSensor())
        return false;
    return (flags_of(body.GetUserData()) & (flag::kSensor | flag::kCharacter)) == 0;
}

VehicleSystem::~VehicleSystem() { remove_all(); }

Vehicle *VehicleSystem::find(VehicleId id) {
    auto it = std::lower_bound(vehicles_.begin(), vehicles_.end(), id,
                               [](const std::unique_ptr<Vehicle> &v, VehicleId key) { return v->desc.id < key; });
    return it != vehicles_.end() && (*it)->desc.id == id ? it->get() : nullptr;
}

float VehicleSystem::table_grip(uint16_t material) const {
    return material < table_grip_.size() ? table_grip_[material] : 1.0f;
}

float VehicleSystem::table_rolling(uint16_t material) const {
    return material < table_rolling_.size() ? table_rolling_[material] : 0.0f;
}

float VehicleSystem::wheel_grip(const Vehicle &vehicle, size_t wheel, const JPH::Body &ground) const {
    const float own = vehicle.grip[wheel];
    return std::isnan(own) ? table_grip(material_bits_of(ground.GetUserData())) : own;
}

bool VehicleSystem::add(const VehicleDesc &desc, const JPH::BodyID &chassis) {
    JPH::Body *body = system_.GetBodyLockInterfaceNoLock().TryGetBody(chassis);
    // A soft body is dynamic too, and a vehicle constraint on one breaks Jolt's island builder.
    if (body == nullptr || !body->IsRigidBody() || !body->IsDynamic() || desc.wheels.empty() ||
        by_chassis_.count(desc.chassis) != 0)
        return false;
    const auto &p = desc.params;
    const size_t count = desc.wheel_count();
    const auto kind = static_cast<VehicleKind>(desc.kind);

    JPH::VehicleConstraintSettings settings;
    settings.mUp = vec(p, kUpX).Normalized();
    settings.mForward = vec(p, kForwardX).Normalized();
    settings.mMaxPitchRollAngle = float(p[kMaxPitchRoll]);
    for (const AntiRollBarDesc &bar : desc.anti_roll_bars) {
        JPH::VehicleAntiRollBar &out = settings.mAntiRollBars.emplace_back();
        out.mLeftWheel = bar.left;
        out.mRightWheel = bar.right;
        out.mStiffness = bar.stiffness;
    }
    if (kind == VehicleKind::Tracked) {
        auto *controller = new JPH::TrackedVehicleControllerSettings;
        settings.mController = controller;
        fill_engine(controller->mEngine, desc);
        fill_transmission(controller->mTransmission, desc);
        for (int side = 0; side < 2; ++side) {
            JPH::VehicleTrackSettings &track = controller->mTracks[side];
            track.mDrivenWheel = JPH::uint(std::max(desc.tracks[side].driven, 0));
            track.mInertia = desc.tracks[side].inertia;
            track.mAngularDamping = desc.tracks[side].damping;
            track.mMaxBrakeTorque = desc.tracks[side].brake_torque;
            track.mDifferentialRatio = desc.tracks[side].ratio;
            track.mWheels.clear();
        }
        for (size_t i = 0; i < count; ++i) {
            auto *wheel = new JPH::WheelSettingsTV;
            fill_wheel(*wheel, desc, i);
            wheel->mLongitudinalFriction = float(desc.wheel(i, kGrip)) * 4.0f;
            wheel->mLateralFriction = float(desc.wheel(i, kGrip)) * 2.0f;
            settings.mWheels.push_back(wheel);
            const int side = int(desc.wheel(i, kTrackSide));
            if (side == 0 || side == 1)
                controller->mTracks[side].mWheels.push_back(JPH::uint(i));
        }
        for (int side = 0; side < 2; ++side)
            if (controller->mTracks[side].mWheels.empty())
                return false;
    } else {
        JPH::WheeledVehicleControllerSettings *controller;
        if (kind == VehicleKind::Motorcycle) {
            auto *bike = new JPH::MotorcycleControllerSettings;
            bike->mMaxLeanAngle = float(p[kMaxLeanAngle]);
            bike->mLeanSpringConstant = float(p[kLeanSpring]);
            bike->mLeanSpringDamping = float(p[kLeanDamping]);
            bike->mLeanSpringIntegrationCoefficient = float(p[kLeanIntegration]);
            bike->mLeanSpringIntegrationCoefficientDecay = float(p[kLeanIntegrationDecay]);
            bike->mLeanSmoothingFactor = float(p[kLeanSmoothing]);
            controller = bike;
        } else {
            controller = new JPH::WheeledVehicleControllerSettings;
        }
        settings.mController = controller;
        fill_engine(controller->mEngine, desc);
        fill_transmission(controller->mTransmission, desc);
        controller->mDifferentialLimitedSlipRatio = float(p[kDiffLimitedSlip]);
        controller->mDifferentials.clear();
        for (const DifferentialDesc &diff : desc.differentials) {
            JPH::VehicleDifferentialSettings &out = controller->mDifferentials.emplace_back();
            out.mLeftWheel = diff.left;
            out.mRightWheel = diff.right;
            out.mDifferentialRatio = diff.ratio;
            out.mLeftRightSplit = diff.split;
            out.mLimitedSlipRatio = diff.limited_slip;
            out.mEngineTorqueRatio = diff.torque_ratio;
        }
        for (size_t i = 0; i < count; ++i) {
            auto *wheel = new JPH::WheelSettingsWV;
            fill_wheel(*wheel, desc, i);
            wheel->mMaxSteerAngle = desc.wheel(i, kSteered) != 0.0 ? float(p[kMaxSteer]) : 0.0f;
            wheel->mMaxBrakeTorque = float(desc.wheel(i, kBrakeTorque));
            wheel->mMaxHandBrakeTorque = float(desc.wheel(i, kHandbrakeTorque));
            wheel->mInertia = float(desc.wheel(i, kWheelInertia));
            wheel->mAngularDamping = float(desc.wheel(i, kWheelDamping));
            const float grip = float(desc.wheel(i, kGrip));
            std::vector<float> longitudinal = desc.longitudinal, lateral = desc.lateral;
            for (size_t k = 1; k < longitudinal.size(); k += 2)
                longitudinal[k] *= grip;
            for (size_t k = 1; k < lateral.size(); k += 2)
                lateral[k] *= grip;
            if (!longitudinal.empty())
                fill_curve(wheel->mLongitudinalFriction, longitudinal);
            if (!lateral.empty())
                fill_curve(wheel->mLateralFriction, lateral);
            settings.mWheels.push_back(wheel);
        }
    }

    auto vehicle = std::make_unique<Vehicle>();
    vehicle->desc = desc;
    vehicle->body = chassis;
    vehicle->grip.assign(count, std::numeric_limits<float>::quiet_NaN());
    vehicle->rolling.assign(count, std::numeric_limits<float>::quiet_NaN());
    vehicle->steered.resize(count);
    for (size_t i = 0; i < count; ++i)
        vehicle->steered[i] = desc.wheel(i, kSteered) != 0.0 ? 1 : 0;
    vehicle->constraint = new JPH::VehicleConstraint(*body, settings);
    const JPH::ObjectLayer layer = body->GetObjectLayer();
    vehicle->layer = layer;
    layers_.retain(layer);
    const JPH::Vec3 up = settings.mUp;
    const float slope = float(p[kMaxSlope]);
    switch (static_cast<WheelCast>(desc.cast)) {
    case WheelCast::Sphere: {
        float radius = float(p[kCasterRadius]);
        if (!(radius > 0.0f)) {
            radius = std::numeric_limits<float>::max();
            for (size_t i = 0; i < count; ++i)
                radius = std::min(radius, float(desc.wheel(i, kRadius)));
        }
        vehicle->tester = new JPH::VehicleCollisionTesterCastSphere(layer, radius, up, slope);
        break;
    }
    case WheelCast::Cylinder:
        vehicle->tester = new JPH::VehicleCollisionTesterCastCylinder(layer);
        break;
    default:
        vehicle->tester = new JPH::VehicleCollisionTesterRay(layer, up, slope);
    }
    vehicle->filter = std::make_unique<WheelBodyFilter>(chassis);
    vehicle->tester->SetBodyFilter(vehicle->filter.get());
    vehicle->constraint->SetVehicleCollisionTester(vehicle->tester);
    const bool body_friction = p[kBodyFriction] != 0.0;
    Vehicle *raw = vehicle.get();
    vehicle->constraint->SetCombineFriction([this, raw, body_friction](JPH::uint wheel, float &longitudinal, float &lateral,
                                                                       const JPH::Body &ground, const JPH::SubShapeID &) {
        if (body_friction) {
            const float friction = ground.GetFriction();
            longitudinal = JPH::Sqrt(longitudinal * friction);
            lateral = JPH::Sqrt(lateral * friction);
        }
        const float grip = wheel_grip(*raw, wheel, ground);
        longitudinal *= grip;
        lateral *= grip;
    });
    system_.AddConstraint(vehicle->constraint);
    system_.AddStepListener(vehicle->constraint);
    system_.GetBodyInterfaceNoLock().ActivateBody(chassis);
    by_chassis_[desc.chassis] = raw;
    auto at = std::lower_bound(vehicles_.begin(), vehicles_.end(), desc.id,
                               [](const std::unique_ptr<Vehicle> &v, VehicleId key) { return v->desc.id < key; });
    vehicles_.insert(at, std::move(vehicle));
    return true;
}

void VehicleSystem::detach(Vehicle &vehicle) {
    system_.RemoveStepListener(vehicle.constraint);
    system_.RemoveConstraint(vehicle.constraint);
    by_chassis_.erase(vehicle.desc.chassis);
    layers_.release(vehicle.layer);
}

void VehicleSystem::remove(const std::vector<VehicleId> &ids) {
    for (VehicleId id : ids) {
        Vehicle *vehicle = find(id);
        if (vehicle == nullptr)
            continue;
        detach(*vehicle);
        vehicles_.erase(std::find_if(vehicles_.begin(), vehicles_.end(),
                                     [&](const std::unique_ptr<Vehicle> &v) { return v.get() == vehicle; }));
    }
}

void VehicleSystem::remove_chassis(const std::vector<Uid> &uids) {
    if (vehicles_.empty())
        return;
    std::vector<VehicleId> gone;
    for (Uid uid : uids) {
        auto it = by_chassis_.find(uid);
        if (it != by_chassis_.end())
            gone.push_back(it->second->desc.id);
    }
    remove(gone);
}

void VehicleSystem::remove_all() {
    for (auto &vehicle : vehicles_)
        detach(*vehicle);
    vehicles_.clear();
}

bool VehicleSystem::input(VehicleId id, const float values[4]) {
    Vehicle *vehicle = find(id);
    if (vehicle == nullptr)
        return false;
    vehicle->throttle = std::clamp(values[0], -1.0f, 1.0f);
    vehicle->steer = std::clamp(values[1], -1.0f, 1.0f);
    vehicle->brake = std::clamp(values[2], 0.0f, 1.0f);
    vehicle->handbrake = std::clamp(values[3], 0.0f, 1.0f);
    if (vehicle->throttle != 0.0f || vehicle->steer != 0.0f || vehicle->brake != 0.0f || vehicle->handbrake != 0.0f)
        system_.GetBodyInterfaceNoLock().ActivateBody(vehicle->body);
    return true;
}

bool VehicleSystem::surface(VehicleId id, int32_t wheel, float grip, float rolling) {
    Vehicle *vehicle = find(id);
    if (vehicle == nullptr || wheel >= int32_t(vehicle->grip.size()))
        return false;
    const size_t first = wheel < 0 ? 0 : size_t(wheel);
    const size_t last = wheel < 0 ? vehicle->grip.size() : size_t(wheel) + 1;
    for (size_t i = first; i < last; ++i) {
        vehicle->grip[i] = grip;
        vehicle->rolling[i] = std::isnan(grip) ? std::numeric_limits<float>::quiet_NaN() : rolling;
    }
    system_.GetBodyInterfaceNoLock().ActivateBody(vehicle->body);
    return true;
}

void VehicleSystem::set_table(std::vector<float> grip, std::vector<float> rolling) {
    table_grip_ = std::move(grip);
    table_rolling_ = std::move(rolling);
}

void VehicleSystem::before_step(double dt) {
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const auto step = float(dt);
    for (auto &owned : vehicles_) {
        Vehicle &v = *owned;
        JPH::Body *body = locks.TryGetBody(v.body);
        if (body == nullptr)
            continue;
        const auto &p = v.desc.params;
        const JPH::Quat rotation = body->GetRotation();
        const JPH::Vec3 forward = rotation * v.constraint->GetLocalForward();
        const JPH::Vec3 velocity = body->GetLinearVelocity();
        const float forward_speed = velocity.Dot(forward);
        const float speed = velocity.Length();

        // The lock falls with the square of speed against the falloff speed.
        const float max_steer = float(p[kMaxSteer]);
        const float falloff = float(p[kSteerFalloffSpeed]);
        float lock = max_steer;
        if (falloff > 0.0f) {
            const float share = std::abs(forward_speed) / falloff;
            lock = max_steer / (1.0f + share * share);
        }
        const float wanted = v.steer * lock;
        const float rate = float(p[kSteerRate]);
        if (rate > 0.0f)
            v.steer_angle = std::clamp(wanted, v.steer_angle - rate * step, v.steer_angle + rate * step);
        else
            v.steer_angle = wanted;

        // An idle vehicle that has come down to its holding speed holds until it is driven again.
        const bool idle = v.throttle == 0.0f && v.brake == 0.0f;
        if (!idle)
            v.holding = false;
        else if (std::abs(forward_speed) < float(p[kHoldingSpeed]))
            v.holding = true;
        const float brake = v.holding ? std::max(v.brake, float(p[kHolding])) : v.brake;

        const auto kind = static_cast<VehicleKind>(v.desc.kind);
        if (kind == VehicleKind::Tracked) {
            auto *controller = static_cast<JPH::TrackedVehicleController *>(v.constraint->GetController());
            // Steering slows the inside track; past half a turn it runs backwards.
            const float turn = max_steer > 0.0f ? v.steer_angle / max_steer : v.steer;
            auto ratio = [](float r) { return std::abs(r) < 0.01f ? 0.01f : r; };
            const float left = turn > 0.0f ? ratio(1.0f - 2.0f * turn) : 1.0f;
            const float right = turn < 0.0f ? ratio(1.0f + 2.0f * turn) : 1.0f;
            controller->SetDriverInput(v.throttle, left, right, std::max(brake, v.handbrake));
        } else {
            auto *controller = static_cast<JPH::WheeledVehicleController *>(v.constraint->GetController());
            const float right = max_steer > 0.0f ? -v.steer_angle / max_steer : 0.0f;
            controller->SetDriverInput(v.throttle, right, brake, v.handbrake);
        }

        // The forces below never wake the chassis: a parked car falls asleep like any body.
        if (!body->IsActive())
            continue;
        JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
        constexpr JPH::EActivation keep = JPH::EActivation::DontActivate;
        const float drag = float(p[kDrag]);
        if (drag > 0.0f && speed > 1e-3f)
            bodies.AddForce(v.body, -velocity * (drag * speed), keep);
        bool grounded = false;
        const float rolling = float(p[kRollingResistance]);
        const JPH::Wheels &wheels = v.constraint->GetWheels();
        for (size_t i = 0; i < wheels.size(); ++i) {
            const JPH::Wheel *wheel = wheels[i];
            if (!wheel->HasContact())
                continue;
            grounded = true;
            float extra = v.rolling[i];
            if (std::isnan(extra)) {
                const JPH::Body *ground = locks.TryGetBody(wheel->GetContactBodyID());
                extra = ground != nullptr ? table_rolling(material_bits_of(ground->GetUserData())) : 0.0f;
            }
            const float resistance = rolling + extra;
            if (resistance <= 0.0f)
                continue;
            const JPH::Vec3 along = wheel->GetContactLongitudinal();
            const float rolling_speed = (body->GetPointVelocity(wheel->GetContactPosition()) -
                                         wheel->GetContactPointVelocity())
                                            .Dot(along);
            if (std::abs(rolling_speed) <= 0.15f)
                continue;
            const float load = std::max(0.0f, wheel->GetSuspensionLambda()) / step;
            bodies.AddForce(v.body, along * (-std::copysign(resistance * load, rolling_speed)),
                            wheel->GetContactPosition(), keep);
        }
        const float downforce = float(p[kDownforce]);
        if (downforce > 0.0f && grounded && speed > 1e-3f)
            bodies.AddForce(v.body, -v.constraint->GetWorldUp() * (downforce * speed * speed), keep);
    }
}

void VehicleSystem::after_step(uint32_t tick, std::vector<Event> &events) {
    if (vehicles_.empty())
        return;
    for (auto &owned : vehicles_) {
        Vehicle &v = *owned;
        v.impact_speed = 0.0f;
        v.impact_impulse = 0.0f;
        v.impact_uid = kNoBody;
    }
    const size_t contacts = events.size();
    for (size_t i = 0; i < contacts; ++i) {
        const Event &e = events[i];
        if (e.kind != uint8_t(EventKind::Contact) ||
            (e.phase != uint8_t(EventPhase::Begin) && e.phase != uint8_t(EventPhase::Persist)))
            continue;
        for (int side = 0; side < 2; ++side) {
            auto it = by_chassis_.find(side == 0 ? e.a : e.b);
            if (it == by_chassis_.end())
                continue;
            Vehicle &v = *it->second;
            if (e.approach <= v.impact_speed)
                continue;
            v.impact_speed = e.approach;
            v.impact_impulse = e.impulse;
            v.impact_uid = side == 0 ? e.b : e.a;
            for (int k = 0; k < 3; ++k) {
                v.impact_point[k] = e.point[k];
                // The normal points away from the chassis, into what it struck.
                v.impact_normal[k] = side == 0 ? e.normal[k] : -e.normal[k];
            }
        }
    }
    for (auto &owned : vehicles_) {
        const Vehicle &v = *owned;
        const float report = float(v.desc.params[kImpactSpeed]);
        const float crash = float(v.desc.params[kCrashSpeed]);
        if (v.impact_uid == kNoBody || !(v.impact_speed >= report || v.impact_speed >= crash))
            continue;
        Event e{};
        e.kind = uint8_t(EventKind::Vehicle);
        e.phase = uint8_t(v.impact_speed >= crash ? EventPhase::Crash : EventPhase::Impact);
        e.tick = tick;
        e.a = v.desc.id;
        e.b = v.impact_uid;
        for (int k = 0; k < 3; ++k) {
            e.point[k] = v.impact_point[k];
            e.normal[k] = v.impact_normal[k];
        }
        e.approach = v.impact_speed;
        e.impulse = v.impact_impulse;
        events.push_back(e);
    }
}

void VehicleSystem::write(SnapshotBuffer &out) const {
    size_t wheels = 0;
    for (const auto &v : vehicles_)
        wheels += v->desc.wheel_count();
    out.resize_vehicles(uint32_t(vehicles_.size()), uint32_t(wheels));
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    uint32_t w = 0;
    for (size_t r = 0; r < vehicles_.size(); ++r) {
        const Vehicle &v = *vehicles_[r];
        const JPH::Body *body = locks.TryGetBody(v.body);
        out.veh_id[r] = v.desc.id;
        out.veh_chassis[r] = v.desc.chassis;
        out.veh_kind[r] = v.desc.kind;
        out.veh_first_wheel[r] = w;
        out.veh_wheels[r] = uint32_t(v.desc.wheel_count());
        const JPH::VehicleController *controller = v.constraint->GetController();
        const JPH::VehicleEngine *engine;
        const JPH::VehicleTransmission *transmission;
        if (static_cast<VehicleKind>(v.desc.kind) == VehicleKind::Tracked) {
            auto *tracked = static_cast<const JPH::TrackedVehicleController *>(controller);
            engine = &tracked->GetEngine();
            transmission = &tracked->GetTransmission();
        } else {
            auto *wheeled = static_cast<const JPH::WheeledVehicleController *>(controller);
            engine = &wheeled->GetEngine();
            transmission = &wheeled->GetTransmission();
        }
        out.veh_rpm[r] = engine->GetCurrentRPM();
        out.veh_gear[r] = int8_t(transmission->GetCurrentGear());
        float forward_speed = 0.0f;
        if (body != nullptr)
            forward_speed = body->GetLinearVelocity().Dot(body->GetRotation() * v.constraint->GetLocalForward());
        out.veh_forward_speed[r] = forward_speed;
        out.veh_steer[r] = v.steer_angle;
        out.veh_impact_speed[r] = v.impact_speed;
        out.veh_impact_impulse[r] = v.impact_impulse;
        out.veh_impact_uid[r] = v.impact_uid;
        float inputs[4] = {v.throttle, v.steer, v.brake, v.handbrake};
        std::copy(inputs, inputs + 4, &out.veh_input[r * 4]);

        const JPH::Wheels &list = v.constraint->GetWheels();
        for (size_t i = 0; i < list.size(); ++i, ++w) {
            const JPH::Wheel *wheel = list[i];
            const JPH::WheelSettings *settings = wheel->GetSettings();
            const JPH::RMat44 transform = v.constraint->GetWheelWorldTransform(JPH::uint(i), JPH::Vec3::sAxisX(),
                                                                               JPH::Vec3::sAxisY());
            const JPH::RVec3 centre = transform.GetTranslation();
            const JPH::Quat q = transform.GetQuaternion().Normalized();
            double *position = &out.wheel_position[size_t(w) * 3];
            position[0] = centre.GetX();
            position[1] = centre.GetY();
            position[2] = centre.GetZ();
            float *orientation = &out.wheel_orientation[size_t(w) * 4];
            orientation[0] = q.GetX();
            orientation[1] = q.GetY();
            orientation[2] = q.GetZ();
            orientation[3] = q.GetW();
            out.wheel_vehicle[w] = uint32_t(r);
            out.wheel_length[w] = wheel->GetSuspensionLength();
            out.wheel_compression[w] = settings->mSuspensionMaxLength - wheel->GetSuspensionLength();
            out.wheel_steer[w] = wheel->GetSteerAngle();
            out.wheel_angle[w] = wheel->GetRotationAngle();
            out.wheel_spin[w] = wheel->GetAngularVelocity();
            out.wheel_load[w] = std::max(0.0f, wheel->GetSuspensionLambda()) / dt_;
            float slip_long = 0.0f, slip_lat = 0.0f;
            if (static_cast<VehicleKind>(v.desc.kind) != VehicleKind::Tracked) {
                auto *wv = static_cast<const JPH::WheelWV *>(wheel);
                slip_long = wv->mLongitudinalSlip;
                slip_lat = wv->mLateralSlip;
            }
            out.wheel_slip[size_t(w) * 2] = slip_long;
            out.wheel_slip[size_t(w) * 2 + 1] = slip_lat;
            const JPH::Body *ground = wheel->HasContact() ? locks.TryGetBody(wheel->GetContactBodyID()) : nullptr;
            double *point = &out.wheel_contact_point[size_t(w) * 3];
            float *normal = &out.wheel_contact_normal[size_t(w) * 3];
            if (ground == nullptr) {
                out.wheel_contact[w] = 0;
                out.wheel_ground[w] = kNoBody;
                out.wheel_material[w] = 0;
                out.wheel_triangle[w] = -1;
                out.wheel_grip[w] = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    point[k] = 0.0;
                    normal[k] = 0.0f;
                }
                continue;
            }
            out.wheel_contact[w] = 1;
            const uint64_t user = ground->GetUserData();
            out.wheel_ground[w] = uid_of(user);
            out.wheel_material[w] = material_bits_of(user);
            out.wheel_grip[w] = wheel_grip(v, i, *ground);
            JPH::SubShapeID remainder;
            const JPH::Shape *leaf = ground->GetShape()->GetLeafShape(wheel->GetContactSubShapeID(), remainder);
            out.wheel_triangle[w] = leaf != nullptr && leaf->GetSubType() == JPH::EShapeSubType::Mesh
                                        ? int32_t(static_cast<const JPH::MeshShape *>(leaf)->GetTriangleUserData(remainder))
                                        : -1;
            const JPH::RVec3 c = wheel->GetContactPosition();
            const JPH::Vec3 n = wheel->GetContactNormal();
            point[0] = c.GetX();
            point[1] = c.GetY();
            point[2] = c.GetZ();
            normal[0] = n.GetX();
            normal[1] = n.GetY();
            normal[2] = n.GetZ();
        }
    }
}

}  // namespace pj
