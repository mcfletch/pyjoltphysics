// Joints, on the world thread with the world lock held exclusively.
#include "world.h"

#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>

#include <algorithm>
#include <cmath>

namespace pj {

// -- JointPairFilter -------------------------------------------------------------

bool JointPairFilter::CanCollide(const JPH::CollisionGroup &a, const JPH::CollisionGroup &b) const {
    return pairs_.find(key(a.GetGroupID(), b.GetGroupID())) == pairs_.end();
}

void JointPairFilter::add(Uid a, Uid b) { ++pairs_[key(a, b)]; }

void JointPairFilter::remove(Uid a, Uid b) {
    auto found = pairs_.find(key(a, b));
    if (found != pairs_.end() && --found->second == 0)
        pairs_.erase(found);
}

namespace {

JPH::Vec3 vec(const double v[3]) { return JPH::Vec3(float(v[0]), float(v[1]), float(v[2])); }
JPH::RVec3 rvec(const double v[3]) { return JPH::RVec3(v[0], v[1], v[2]); }

JPH::Vec3 unit(const double v[3], JPH::Vec3Arg fallback) {
    const JPH::Vec3 d = vec(v);
    return d.LengthSq() > 1e-18f ? d.Normalized() : JPH::Vec3(fallback);
}

float limit(double value) {
    if (value == kInf)
        return FLT_MAX;
    if (value == -kInf)
        return -FLT_MAX;
    return float(value);
}

JPH::SpringSettings spring(const float values[2]) {
    return JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping, values[0], values[1]);
}

void motor_settings(JPH::MotorSettings &motor, const JointDesc &desc, int axis) {
    motor.mSpringSettings = spring(desc.motor_spring);
    const float most = limit(desc.motor_max[axis]);
    motor.SetForceLimit(most);
    motor.SetTorqueLimit(most);
}

JPH::EMotorState motor_state(uint8_t mode) { return static_cast<JPH::EMotorState>(mode); }

// The joint's frame for SIXDOF: x along `axis`, y along `axis2` made square to it.
void frame(const JointDesc &desc, JPH::Vec3 &x, JPH::Vec3 &y) {
    x = unit(desc.axis, JPH::Vec3::sAxisX());
    y = unit(desc.axis2, x.GetNormalizedPerpendicular());
    y = (y - x * x.Dot(y));
    y = y.LengthSq() > 1e-12f ? y.Normalized() : x.GetNormalizedPerpendicular();
}

// Sets the motor state and target of one axis of a made constraint.
void drive(JPH::Constraint &constraint, JointKind kind, int axis, uint8_t mode, double target) {
    switch (kind) {
    case JointKind::Hinge: {
        auto &hinge = static_cast<JPH::HingeConstraint &>(constraint);
        hinge.SetMotorState(motor_state(mode));
        if (mode == uint8_t(MotorMode::Velocity))
            hinge.SetTargetAngularVelocity(float(target));
        else if (mode == uint8_t(MotorMode::Position))
            hinge.SetTargetAngle(float(target));
        break;
    }
    case JointKind::Slider: {
        auto &slider = static_cast<JPH::SliderConstraint &>(constraint);
        slider.SetMotorState(motor_state(mode));
        if (mode == uint8_t(MotorMode::Velocity))
            slider.SetTargetVelocity(float(target));
        else if (mode == uint8_t(MotorMode::Position))
            slider.SetTargetPosition(float(target));
        break;
    }
    case JointKind::SixDOF: {
        auto &six = static_cast<JPH::SixDOFConstraint &>(constraint);
        const auto which = static_cast<JPH::SixDOFConstraintSettings::EAxis>(axis);
        six.SetMotorState(which, motor_state(mode));
        if (axis < 3) {
            JPH::Vec3 velocity = six.GetTargetVelocityCS(), position = six.GetTargetPositionCS();
            velocity.SetComponent(uint(axis), float(target));
            position.SetComponent(uint(axis), float(target));
            if (mode == uint8_t(MotorMode::Velocity))
                six.SetTargetVelocityCS(velocity);
            else if (mode == uint8_t(MotorMode::Position))
                six.SetTargetPositionCS(position);
        } else {
            JPH::Vec3 velocity = six.GetTargetAngularVelocityCS();
            velocity.SetComponent(uint(axis - 3), float(target));
            if (mode == uint8_t(MotorMode::Velocity)) {
                six.SetTargetAngularVelocityCS(velocity);
            } else if (mode == uint8_t(MotorMode::Position)) {
                JPH::Vec3 angles = six.GetTargetOrientationCS().GetEulerAngles();
                angles.SetComponent(uint(axis - 3), float(target));
                six.SetTargetOrientationCS(JPH::Quat::sEulerAngles(angles));
            }
        }
        break;
    }
    default:
        break;
    }
}

JPH::Ref<JPH::TwoBodyConstraintSettings> settings_for(const JointDesc &desc) {
    switch (static_cast<JointKind>(desc.kind)) {
    case JointKind::Fixed: {
        auto *s = new JPH::FixedConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mAutoDetectPoint = true;
        return s;
    }
    case JointKind::Ball: {
        auto *s = new JPH::PointConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mPoint1 = s->mPoint2 = rvec(desc.anchor);
        return s;
    }
    case JointKind::Hinge: {
        auto *s = new JPH::HingeConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mPoint1 = s->mPoint2 = rvec(desc.anchor);
        s->mHingeAxis1 = s->mHingeAxis2 = unit(desc.axis, JPH::Vec3::sAxisY());
        s->mNormalAxis1 = s->mNormalAxis2 = s->mHingeAxis1.GetNormalizedPerpendicular();
        s->mLimitsMin = float(std::clamp(desc.lower[0], -double(JPH::JPH_PI), 0.0));
        s->mLimitsMax = float(std::clamp(desc.upper[0], 0.0, double(JPH::JPH_PI)));
        s->mLimitsSpringSettings = spring(desc.spring);
        s->mMaxFrictionTorque = desc.friction;
        motor_settings(s->mMotorSettings, desc, 0);
        return s;
    }
    case JointKind::Slider: {
        auto *s = new JPH::SliderConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mAutoDetectPoint = false;
        s->mPoint1 = s->mPoint2 = rvec(desc.anchor);
        s->SetSliderAxis(unit(desc.axis, JPH::Vec3::sAxisX()));
        s->mLimitsMin = limit(std::min(desc.lower[0], 0.0));
        s->mLimitsMax = limit(std::max(desc.upper[0], 0.0));
        s->mLimitsSpringSettings = spring(desc.spring);
        s->mMaxFrictionForce = desc.friction;
        motor_settings(s->mMotorSettings, desc, 0);
        return s;
    }
    case JointKind::Distance: {
        auto *s = new JPH::DistanceConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mPoint1 = rvec(desc.to_anchor);
        s->mPoint2 = rvec(desc.anchor);
        s->mMinDistance = std::isfinite(desc.lower[0]) ? float(desc.lower[0]) : -1.0f;
        s->mMaxDistance = std::isfinite(desc.upper[0]) ? float(desc.upper[0]) : -1.0f;
        s->mLimitsSpringSettings = spring(desc.spring);
        return s;
    }
    case JointKind::SixDOF: {
        auto *s = new JPH::SixDOFConstraintSettings;
        s->mSpace = JPH::EConstraintSpace::WorldSpace;
        s->mPosition1 = s->mPosition2 = rvec(desc.anchor);
        JPH::Vec3 x, y;
        frame(desc, x, y);
        s->mAxisX1 = s->mAxisX2 = x;
        s->mAxisY1 = s->mAxisY2 = y;
        for (int i = 0; i < 6; ++i) {
            const auto axis = static_cast<JPH::SixDOFConstraintSettings::EAxis>(i);
            if (desc.lower[i] == 0.0 && desc.upper[i] == 0.0 && !(i < 3 && desc.spring[0] > 0.0f))
                s->MakeFixedAxis(axis);
            else if (desc.lower[i] == -kInf && desc.upper[i] == kInf)
                s->MakeFreeAxis(axis);
            else
                s->SetLimitedAxis(axis, limit(desc.lower[i]), limit(desc.upper[i]));
            if (i < 3)
                s->mLimitsSpringSettings[i] = spring(desc.spring);
            s->mMaxFriction[i] = desc.friction;
            motor_settings(s->mMotorSettings[i], desc, i);
        }
        return s;
    }
    case JointKind::Motor:
        break;
    }
    return nullptr;
}

}  // namespace

void WorldCore::apply(AddJoints &add, uint64_t) {
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    if (joint_filter_ == nullptr)
        joint_filter_ = new JointPairFilter;
    for (JointDesc &desc : add.joints) {
        const Row *ra = desc.a == kNoBody ? nullptr : find(desc.a);
        const Row *rb = find(desc.b);
        if (rb == nullptr || (desc.a != kNoBody && ra == nullptr)) {
            ++rejected_;
            continue;
        }
        JPH::Body *a = ra ? locks.TryGetBody(ra->id) : &JPH::Body::sFixedToWorld;
        JPH::Body *b = locks.TryGetBody(rb->id);
        if (a == nullptr || b == nullptr) {
            ++rejected_;
            continue;
        }
        Joint joint;
        joint.desc = desc;
        joint.a = ra ? ra->id : JPH::BodyID();
        joint.b = rb->id;
        const auto kind = static_cast<JointKind>(desc.kind);
        if (kind == JointKind::Motor) {
            const JPH::Quat to_a = ra ? a->GetRotation().Conjugated() : JPH::Quat::sIdentity();
            joint.axes_a[0] = to_a * unit(desc.axis, JPH::Vec3::sAxisX());
            joint.axes_a[1] = to_a * unit(desc.axis2, JPH::Vec3::sAxisY());
            joint.axes_a[2] = to_a * unit(desc.axis3, JPH::Vec3::sAxisZ());
        } else {
            if (kind == JointKind::Slider && std::isnan(desc.anchor[0])) {
                const JPH::RVec3 origin = b->GetPosition();
                for (int c = 0; c < 3; ++c)
                    desc.anchor[c] = origin[c];
                joint.desc = desc;
            }
            JPH::Ref<JPH::TwoBodyConstraintSettings> settings = settings_for(desc);
            joint.constraint = settings->Create(*a, *b);
            if (kind == JointKind::Distance) {
                joint.local_a = JPH::Vec3(a->GetInverseCenterOfMassTransform() * rvec(desc.to_anchor));
                joint.local_b = JPH::Vec3(b->GetInverseCenterOfMassTransform() * rvec(desc.anchor));
            }
            for (int i = 0; i < 6; ++i)
                if (desc.motor_mode[i] != uint8_t(MotorMode::Off))
                    drive(*joint.constraint, kind, i, desc.motor_mode[i], desc.motor_target[i]);
            system_.AddConstraint(joint.constraint);
        }
        if (!desc.collide && ra != nullptr) {
            joint_filter_->add(desc.a, desc.b);
            for (const Row *row : {ra, rb})
                bodies.SetCollisionGroup(row->id, JPH::CollisionGroup(joint_filter_, row->uid, 0));
        }
        if (!(b->IsActive()))
            bodies.ActivateBody(rb->id);
        joints_.push_back(std::move(joint));
    }
    rebuild_torques();
}

void WorldCore::remove_joint(Joint &joint, uint64_t index, uint16_t reason) {
    if (joint.constraint != nullptr && joint.state != JointState::Broken)
        system_.RemoveConstraint(joint.constraint);
    if (!joint.desc.collide && joint.desc.a != kNoBody && joint_filter_ != nullptr)
        joint_filter_->remove(joint.desc.a, joint.desc.b);
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const JPH::BodyID &id : {joint.a, joint.b})
        if (!id.IsInvalid() && bodies.IsAdded(id) && bodies.GetMotionType(id) == JPH::EMotionType::Dynamic)
            bodies.ActivateBody(id);
    Event event{};
    event.kind = uint8_t(EventKind::Joint);
    event.phase = uint8_t(EventPhase::End);
    event.reason = reason;
    event.tick = uint32_t(index + 1);
    event.a = joint.desc.id;
    tick_events_.push_back(event);
}

void WorldCore::apply(RemoveJoints &remove, uint64_t index) {
    std::vector<JointId> ids = remove.ids;
    std::sort(ids.begin(), ids.end());
    auto keep = std::remove_if(joints_.begin(), joints_.end(), [&](Joint &joint) {
        if (!std::binary_search(ids.begin(), ids.end(), joint.desc.id))
            return false;
        remove_joint(joint, index, uint16_t(EndReason::None));
        return true;
    });
    joints_.erase(keep, joints_.end());
    rebuild_torques();
}

void WorldCore::remove_joints_of(const std::vector<Uid> &uids, uint64_t index) {
    if (joints_.empty())
        return;
    std::vector<Uid> gone = uids;
    std::sort(gone.begin(), gone.end());
    auto keep = std::remove_if(joints_.begin(), joints_.end(), [&](Joint &joint) {
        if (!std::binary_search(gone.begin(), gone.end(), joint.desc.b) &&
            !(joint.desc.a != kNoBody && std::binary_search(gone.begin(), gone.end(), joint.desc.a)))
            return false;
        remove_joint(joint, index, uint16_t(EndReason::Removed));
        return true;
    });
    joints_.erase(keep, joints_.end());
    rebuild_torques();
}

static Joint *find_joint(std::vector<Joint> &joints, JointId id) {
    auto found = std::lower_bound(joints.begin(), joints.end(), id,
                                  [](const Joint &joint, JointId value) { return joint.desc.id < value; });
    return found != joints.end() && found->desc.id == id ? &*found : nullptr;
}

void WorldCore::apply(SetJointMotors &set, uint64_t) {
    for (size_t i = 0; i < set.ids.size(); ++i) {
        Joint *joint = find_joint(joints_, set.ids[i]);
        const int axis = set.axis[i];
        if (joint == nullptr || axis > 5 || joint->state == JointState::Broken) {
            ++rejected_;
            continue;
        }
        JointDesc &desc = joint->desc;
        if (!std::isnan(set.target[i]))
            desc.motor_target[axis] = set.target[i];
        if (!std::isnan(set.max[i]))
            desc.motor_max[axis] = set.max[i];
        if (set.mode[i] != 0xff)
            desc.motor_mode[axis] = set.mode[i];
        const auto kind = static_cast<JointKind>(desc.kind);
        if (joint->constraint != nullptr) {
            const float most = limit(desc.motor_max[axis]);
            if (kind == JointKind::Hinge) {
                static_cast<JPH::HingeConstraint &>(*joint->constraint).GetMotorSettings().SetTorqueLimit(most);
            } else if (kind == JointKind::Slider) {
                static_cast<JPH::SliderConstraint &>(*joint->constraint).GetMotorSettings().SetForceLimit(most);
            } else if (kind == JointKind::SixDOF) {
                auto &motor = static_cast<JPH::SixDOFConstraint &>(*joint->constraint)
                                  .GetMotorSettings(static_cast<JPH::SixDOFConstraintSettings::EAxis>(axis));
                motor.SetForceLimit(most);
                motor.SetTorqueLimit(most);
            }
            drive(*joint->constraint, kind, axis, desc.motor_mode[axis], desc.motor_target[axis]);
        }
        JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
        for (const JPH::BodyID &id : {joint->a, joint->b})
            if (!id.IsInvalid() && bodies.GetMotionType(id) == JPH::EMotionType::Dynamic)
                bodies.ActivateBody(id);
    }
    rebuild_torques();
}

void WorldCore::apply(SetJointsEnabled &set, uint64_t) {
    for (size_t i = 0; i < set.ids.size(); ++i) {
        Joint *joint = find_joint(joints_, set.ids[i]);
        if (joint == nullptr || joint->state == JointState::Broken) {
            ++rejected_;
            continue;
        }
        const bool on = set.enabled[i] != 0;
        joint->state = on ? JointState::Active : JointState::Disabled;
        if (joint->constraint != nullptr)
            joint->constraint->SetEnabled(on);
        JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
        for (const JPH::BodyID &id : {joint->a, joint->b})
            if (!id.IsInvalid() && bodies.GetMotionType(id) == JPH::EMotionType::Dynamic)
                bodies.ActivateBody(id);
    }
    rebuild_torques();
}

void WorldCore::rebuild_torques() {
    fields_.torques.clear();
    for (const Joint &joint : joints_) {
        if (joint.desc.kind != uint8_t(JointKind::Motor) || joint.state != JointState::Active)
            continue;
        JPH::Vec3 local = JPH::Vec3::sZero();
        for (int i = 0; i < 3; ++i)
            local += joint.axes_a[i] * float(joint.desc.motor_target[i]);
        if (local != JPH::Vec3::sZero())
            fields_.torques.push_back({joint.a, joint.b, local});
    }
}

void WorldCore::activate_motor_joints() {
    if (fields_.torques.empty())
        return;
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const FieldListener::Torque &t : fields_.torques)
        for (const JPH::BodyID &id : {t.a, t.b})
            if (!id.IsInvalid() && !bodies.IsActive(id) && bodies.GetMotionType(id) == JPH::EMotionType::Dynamic)
                bodies.ActivateBody(id);
}

void WorldCore::joints_after_step(uint64_t tick) {
    if (joints_.empty())
        return;
    const float per = float(1.0 / dt_);
    bool broke = false;
    for (Joint &joint : joints_) {
        if (joint.constraint == nullptr || joint.state != JointState::Active)
            continue;
        JPH::Vec3 linear = JPH::Vec3::sZero(), angular = JPH::Vec3::sZero();
        JPH::Constraint &c = *joint.constraint;
        switch (static_cast<JointKind>(joint.desc.kind)) {
        case JointKind::Fixed: {
            auto &fixed = static_cast<JPH::FixedConstraint &>(c);
            linear = fixed.GetTotalLambdaPosition();
            angular = fixed.GetTotalLambdaRotation();
            break;
        }
        case JointKind::Ball:
            linear = static_cast<JPH::PointConstraint &>(c).GetTotalLambdaPosition();
            break;
        case JointKind::Hinge: {
            auto &hinge = static_cast<JPH::HingeConstraint &>(c);
            linear = hinge.GetTotalLambdaPosition();
            const auto r = hinge.GetTotalLambdaRotation();
            angular = JPH::Vec3(r[0], r[1], hinge.GetTotalLambdaRotationLimits() + hinge.GetTotalLambdaMotor());
            break;
        }
        case JointKind::Slider: {
            auto &slider = static_cast<JPH::SliderConstraint &>(c);
            const auto p = slider.GetTotalLambdaPosition();
            linear = JPH::Vec3(p[0], p[1], slider.GetTotalLambdaPositionLimits() + slider.GetTotalLambdaMotor());
            angular = slider.GetTotalLambdaRotation();
            break;
        }
        case JointKind::Distance:
            linear = JPH::Vec3(static_cast<JPH::DistanceConstraint &>(c).GetTotalLambdaPosition(), 0, 0);
            break;
        case JointKind::SixDOF: {
            auto &six = static_cast<JPH::SixDOFConstraint &>(c);
            linear = six.GetTotalLambdaPosition() + six.GetTotalLambdaMotorTranslation();
            angular = six.GetTotalLambdaRotation() + six.GetTotalLambdaMotorRotation();
            break;
        }
        case JointKind::Motor:
            break;
        }
        joint.force = linear.Length() * per;
        joint.torque = angular.Length() * per;
        const bool over = (joint.desc.break_force > 0.0f && joint.force > joint.desc.break_force) ||
                          (joint.desc.break_torque > 0.0f && joint.torque > joint.desc.break_torque);
        if (!over)
            continue;
        system_.RemoveConstraint(joint.constraint);
        joint.state = JointState::Broken;
        if (!joint.desc.collide && joint.desc.a != kNoBody && joint_filter_ != nullptr)
            joint_filter_->remove(joint.desc.a, joint.desc.b);
        Event event{};
        event.kind = uint8_t(EventKind::Joint);
        event.phase = uint8_t(EventPhase::Break);
        event.tick = uint32_t(tick);
        event.a = joint.desc.id;
        event.b = joint.desc.b;
        for (int k = 0; k < 3; ++k)
            event.point[k] = joint.desc.anchor[k];
        event.approach = joint.force;
        event.impulse = joint.torque;
        tick_events_.push_back(event);
        broke = true;
    }
    (void)broke;
}

void WorldCore::write_joints(SnapshotBuffer &out) const {
    const size_t n = joints_.size();
    out.resize_joints(uint32_t(n));
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (size_t i = 0; i < n; ++i) {
        const Joint &joint = joints_[i];
        out.joint_id[i] = joint.desc.id;
        out.joint_kind[i] = joint.desc.kind;
        out.joint_state[i] = uint8_t(joint.state);
        out.joint_force[i] = joint.force;
        out.joint_torque[i] = joint.torque;
        float value = 0.0f, value2 = 0.0f;
        if (joint.constraint != nullptr) {
            JPH::Constraint &c = *joint.constraint;
            switch (static_cast<JointKind>(joint.desc.kind)) {
            case JointKind::Hinge:
                value = static_cast<JPH::HingeConstraint &>(c).GetCurrentAngle();
                break;
            case JointKind::Slider:
                value = static_cast<JPH::SliderConstraint &>(c).GetCurrentPosition();
                break;
            case JointKind::Distance: {
                const JPH::Body *a = joint.a.IsInvalid() ? &JPH::Body::sFixedToWorld : locks.TryGetBody(joint.a);
                const JPH::Body *b = locks.TryGetBody(joint.b);
                if (a != nullptr && b != nullptr) {
                    const JPH::RVec3 pa = a->GetCenterOfMassTransform() * joint.local_a;
                    const JPH::RVec3 pb = b->GetCenterOfMassTransform() * joint.local_b;
                    value = float((pb - pa).Length());
                }
                break;
            }
            case JointKind::SixDOF: {
                JPH::Quat swing, twist;
                static_cast<JPH::SixDOFConstraint &>(c).GetRotationInConstraintSpace().GetSwingTwist(swing, twist);
                if (swing.GetW() < 0.0f)
                    swing = -swing;
                value = 2.0f * std::atan2(swing.GetY(), swing.GetW());
                value2 = 2.0f * std::atan2(swing.GetZ(), swing.GetW());
                break;
            }
            default:
                break;
            }
        }
        out.joint_value[i] = value;
        out.joint_value2[i] = value2;
    }
}

}  // namespace pj
