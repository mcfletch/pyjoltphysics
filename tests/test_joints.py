"""Joints: hinges, balls, sliders, distances, fixed and six-axis joints, motors, limits, breaking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    EndReason,
    EventKind,
    EventPhase,
    Joint,
    JointKind,
    JointState,
    Motor,
    MotorMode,
    Snapshot,
    World,
)
from pyjoltphysics.testing import MakeWorld, latest


def _box(
    world: World,
    position: tuple[float, float, float],
    size: tuple[float, float, float] = (0.2, 0.2, 0.2),
    mass: float = 1.0,
    **values: Any,
) -> int:
    uids, _ = world.add_bodies(BodyBatch.boxes([position], size=size, mass=mass, **values))
    return int(uids[0])


def _row(snapshot: Snapshot, uid: int) -> int:
    return snapshot.rows[uid]


def test_hinge_pendulum_keeps_its_anchor_and_reads_its_angle(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    bob = _box(world, (1.0, 2.0, 0.0))
    joint = world.add_joint(Joint.hinge(bob, anchor=(0.0, 2.0, 0.0), axis=(0.0, 0.0, 1.0)))
    for _ in range(6):
        world.run_ticks(20)
        snap = latest(world)
        p = snap.position[_row(snap, bob)]
        assert abs(np.linalg.norm(p - (0.0, 2.0, 0.0)) - 1.0) < 2e-3
        joints = snap.joints
        angle = float(joints.value[joints.rows[joint]])
        assert abs(angle - math.atan2(p[1] - 2.0, p[0])) < 1e-2
    assert joints.kind[0] == JointKind.HINGE and joints.state[0] == JointState.ACTIVE


def test_hinge_pendulum_period(make_world: MakeWorld) -> None:
    # A small sphere on a 1 m massless arm: T = 2 pi sqrt(L / g) for small swings.
    world = make_world(threads=1, allow_sleeping=False, tick_hz=240.0)
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(math.sin(0.1), 2.0 - math.cos(0.1), 0.0)], radius=0.01)
    )
    bob = int(uids[0])
    joint = world.add_joint(Joint.hinge(bob, anchor=(0.0, 2.0, 0.0), axis=(0.0, 0.0, 1.0)))
    angles = []
    for _ in range(960):
        world.run_ticks(1)
        snap = latest(world)
        angles.append(float(snap.joints.value[snap.joints.rows[joint]]))
    # The angle counts from the starting pose, so the swing is about -0.1.
    a = np.array(angles) - np.mean(angles)
    crossings = np.nonzero((a[:-1] < 0) & (a[1:] >= 0))[0]
    period = float(np.mean(np.diff(crossings))) / 240.0
    assert abs(period - 2 * math.pi * math.sqrt(1.0 / 9.81)) < 0.02 * period


def test_hinge_limits_stop_the_swing(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    bob = _box(world, (1.0, 2.0, 0.0))
    joint = world.add_joint(
        Joint.hinge(bob, anchor=(0.0, 2.0, 0.0), axis=(0.0, 0.0, 1.0), limits=(-0.5, 0.5))
    )
    world.run_ticks(600)
    snap = latest(world)
    assert abs(float(snap.joints.value[snap.joints.rows[joint]]) + 0.5) < math.radians(1.0)


def test_hinge_velocity_motor_turns_at_its_speed(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    wheel = _box(world, (0.0, 1.0, 0.0), size=(1.0, 0.1, 1.0), mass=5.0)
    world.add_joint(
        Joint.hinge(
            wheel,
            anchor=(0.0, 1.0, 0.0),
            axis=(0.0, 1.0, 0.0),
            motor=Motor(MotorMode.VELOCITY, 3.0, max=200.0),
        )
    )
    world.run_ticks(120)
    snap = latest(world)
    assert abs(float(snap.angular_velocity[_row(snap, wheel), 1]) - 3.0) < 0.06


def test_hinge_position_motor_reaches_its_angle(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    door = _box(world, (0.5, 1.0, 0.0), size=(1.0, 2.0, 0.1), mass=10.0)
    joint = world.add_joint(Joint.hinge(door, anchor=(0.0, 1.0, 0.0), axis=(0.0, 1.0, 0.0)))
    world.set_joint_motors([joint], MotorMode.POSITION, 1.0, max=500.0)
    world.run_ticks(600)
    snap = latest(world)
    assert abs(float(snap.joints.value[0]) - 1.0) < math.radians(1.0)


def test_slider_stops_at_its_limit(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    car = _box(world, (0.0, 2.0, 0.0))
    joint = world.add_joint(Joint.slider(car, axis=(0.0, 1.0, 0.0), limits=(-1.0, 0.5)))
    world.run_ticks(240)
    snap = latest(world)
    assert abs(float(snap.position[_row(snap, car), 1]) - 1.0) < 1e-3
    assert abs(float(snap.joints.value[snap.joints.rows[joint]]) + 1.0) < 1e-3
    assert abs(float(snap.position[_row(snap, car), 0])) < 1e-4


def test_slider_velocity_motor(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    car = _box(world, (0.0, 1.0, 0.0))
    world.add_joint(
        Joint.slider(car, axis=(1.0, 0.0, 0.0), motor=Motor(MotorMode.VELOCITY, 2.0, max=100.0))
    )
    world.run_ticks(120)
    snap = latest(world)
    assert abs(float(snap.linear_velocity[_row(snap, car), 0]) - 2.0) < 0.02


def test_distance_joint_holds_its_length(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    bob = _box(world, (2.0, 3.0, 0.0))
    joint = world.add_joint(Joint.distance(bob, anchor=(2.0, 3.0, 0.0), to_anchor=(0.0, 3.0, 0.0)))
    for _ in range(5):
        world.run_ticks(30)
        snap = latest(world)
        assert abs(np.linalg.norm(snap.position[_row(snap, bob)] - (0.0, 3.0, 0.0)) - 2.0) < 5e-3
        assert abs(float(snap.joints.value[snap.joints.rows[joint]]) - 2.0) < 5e-3


def test_distance_range_lets_the_body_move_within_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    bob = _box(world, (0.0, 2.0, 0.0))
    world.add_joint(
        Joint.distance(bob, anchor=(0.0, 2.0, 0.0), to_anchor=(0.0, 3.0, 0.0), length=(0.5, 1.5))
    )
    world.run_ticks(240)
    snap = latest(world)
    assert abs(float(snap.position[_row(snap, bob), 1]) - 1.5) < 5e-3


def test_fixed_joint_holds_two_bodies_together_as_they_fall(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    a = _box(world, (0.0, 5.0, 0.0))
    b = _box(world, (0.5, 5.2, 0.0))
    world.add_joint(Joint.fixed(b, to=a))
    world.set_velocities([a], angular=[(0.0, 0.0, 2.0)])
    world.run_ticks(60)
    snap = latest(world)
    pa, pb = snap.position[_row(snap, a)], snap.position[_row(snap, b)]
    assert abs(np.linalg.norm(pb - pa) - math.hypot(0.5, 0.2)) < 2e-3
    assert np.allclose(snap.orientation[_row(snap, a)], snap.orientation[_row(snap, b)], atol=2e-3)


def test_ball_joint_chain_keeps_its_links(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    links = [_box(world, (0.5 * (i + 1), 5.0, 0.0), size=(0.4, 0.1, 0.1)) for i in range(6)]
    joints = [Joint.ball(links[0], anchor=(0.25, 5.0, 0.0))]
    joints += [
        Joint.ball(links[i + 1], to=links[i], anchor=(0.5 * (i + 1) + 0.25, 5.0, 0.0))
        for i in range(5)
    ]
    ids, _ = world.add_joints(joints)
    assert list(ids) == list(range(1, 7))
    world.run_ticks(240)
    snap = latest(world)

    def end(uid: int, sign: float) -> np.ndarray:
        x, y, z, w = (float(v) for v in snap.orientation[_row(snap, uid)])
        # The link's local x axis, turned by its quaternion.
        axis = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        found: np.ndarray = snap.position[_row(snap, uid)] + sign * 0.25 * axis
        return found

    gaps = [np.linalg.norm(end(links[i], 1.0) - end(links[i + 1], -1.0)) for i in range(5)]
    assert max(gaps) < 0.01
    assert np.linalg.norm(end(links[0], -1.0) - (0.25, 5.0, 0.0)) < 0.01


def test_joint_breaks_past_its_force(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    weight = _box(world, (1.0, 2.0, 0.0), mass=50.0)
    joint = world.add_joint(
        Joint.hinge(weight, anchor=(0.0, 2.0, 0.0), axis=(0.0, 0.0, 1.0), break_force=100.0)
    )
    world.run_ticks(60)
    events = world.drain_events()
    broke = events[(events['kind'] == EventKind.JOINT) & (events['phase'] == EventPhase.BREAK)]
    assert len(broke) == 1 and broke['a'][0] == joint and broke['b'][0] == weight
    assert broke['approach'][0] > 100.0
    snap = latest(world)
    assert snap.joints.state[0] == JointState.BROKEN
    assert float(snap.position[_row(snap, weight), 1]) < 1.0


def test_joint_force_reads_the_weight_it_carries(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    weight = _box(world, (0.0, 1.0, 0.0), mass=10.0)
    world.add_joint(Joint.ball(weight, anchor=(0.0, 1.0, 0.0)))
    world.run_ticks(120)
    snap = latest(world)
    assert abs(float(snap.joints.force[0]) - 98.1) < 1.0


def test_removing_a_joint_frees_its_body(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    weight = _box(world, (0.0, 2.0, 0.0))
    joint = world.add_joint(Joint.ball(weight, anchor=(0.0, 2.0, 0.0)))
    world.run_ticks(60)
    world.drain_events()
    world.remove_joints([joint])
    world.run_ticks(60)
    snap = latest(world)
    assert float(snap.position[_row(snap, weight), 1]) < 1.0
    assert snap.joints.count == 0
    events = world.drain_events()
    ended = events[(events['kind'] == EventKind.JOINT) & (events['phase'] == EventPhase.END)]
    assert list(ended['a']) == [joint] and ended['reason'][0] == EndReason.NONE


def test_removing_a_body_removes_its_joints(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    a = _box(world, (0.0, 2.0, 0.0))
    joint = world.add_joint(Joint.ball(a, anchor=(0.0, 2.0, 0.0)))
    world.run_ticks(2)
    world.remove_bodies([a])
    world.run_ticks(2)
    assert latest(world).joints.count == 0
    events = world.drain_events()
    ended = events[(events['kind'] == EventKind.JOINT) & (events['phase'] == EventPhase.END)]
    assert list(ended['a']) == [joint] and ended['reason'][0] == EndReason.REMOVED


def test_disabled_joint_lets_go_and_holds_again_when_enabled(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    car = _box(world, (0.0, 1.0, 0.0))
    joint = world.add_joint(Joint.slider(car, axis=(1.0, 0.0, 0.0)))
    world.set_joints_enabled([joint], False)
    world.set_velocities([car], linear=[(0.0, 1.0, 0.0)])
    world.run_ticks(60)
    snap = latest(world)
    assert snap.joints.state[0] == JointState.DISABLED
    assert float(snap.position[_row(snap, car), 1]) > 1.4
    world.set_joints_enabled([joint], True)
    world.run_ticks(2)
    assert latest(world).joints.state[0] == JointState.ACTIVE


def test_jointed_bodies_do_not_collide_unless_asked(make_world: MakeWorld) -> None:
    for collide, apart in ((False, False), (True, True)):
        world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
        a = _box(world, (0.0, 1.0, 0.0), size=(1.0, 1.0, 1.0), motion=0)
        b = _box(world, (0.0, 1.0, 0.0), size=(0.5, 0.5, 0.5))
        world.add_joint(Joint.slider(b, to=a, axis=(1.0, 0.0, 0.0), collide=collide))
        world.run_ticks(30)
        events = world.drain_events()
        touched = bool(np.any(events['kind'] == EventKind.CONTACT))
        assert touched == apart


def test_universal_joint_carries_spin_across(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    shaft = _box(world, (0.0, 1.0, -0.5), size=(0.1, 0.1, 1.0), mass=1.0)
    out = _box(world, (0.0, 1.0, 0.5), size=(0.1, 0.1, 1.0), mass=1.0)
    world.add_joint(
        Joint.hinge(
            shaft,
            anchor=(0.0, 1.0, -0.5),
            axis=(0.0, 0.0, 1.0),
            motor=Motor(MotorMode.VELOCITY, 2.0, max=100.0),
        )
    )
    world.add_joint(
        Joint.universal(
            out, to=shaft, anchor=(0.0, 1.0, 0.0), axis1=(1.0, 0.0, 0.0), axis2=(0.0, 1.0, 0.0)
        )
    )
    world.run_ticks(120)
    snap = latest(world)
    assert abs(float(snap.angular_velocity[_row(snap, out), 2]) - 2.0) < 0.1


def test_two_axis_hinge_steers_within_its_stops(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    frame = _box(world, (0.0, 1.0, 0.0), size=(1.0, 0.1, 0.1), motion=0)
    wheel = _box(world, (1.0, 1.0, 0.0), size=(0.1, 0.6, 0.6))
    joint = world.add_joint(
        Joint.two_axis_hinge(
            wheel,
            to=frame,
            anchor=(1.0, 1.0, 0.0),
            axis1=(0.0, 1.0, 0.0),
            axis2=(1.0, 0.0, 0.0),
            limits1=(-0.3, 0.3),
            motor1=Motor(MotorMode.VELOCITY, 1.0, max=50.0),
            motor2=Motor(MotorMode.VELOCITY, 5.0, max=50.0),
        )
    )
    world.run_ticks(240)
    snap = latest(world)
    row = snap.joints.rows[joint]
    assert snap.joints.kind[row] == JointKind.SIX_DOF
    assert abs(float(snap.joints.value[row]) - 0.3) < math.radians(2.0)


def test_motor_joint_turns_a_body_by_its_torque(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    spinner = _box(world, (0.0, 1.0, 0.0), size=(1.0, 1.0, 1.0), mass=6.0)
    world.add_joint(Joint.motor(spinner, axes=[(0.0, 1.0, 0.0)], torques=[1.0]))
    world.run_ticks(120)
    snap = latest(world)
    # I = m (a^2 + b^2) / 12 = 1 kg m2; one second of 1 N m.
    assert abs(float(snap.angular_velocity[_row(snap, spinner), 1]) - 1.0) < 0.02


def test_joint_naming_no_body_is_rejected(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_joint(Joint.ball(999, anchor=(0.0, 0.0, 0.0)))
    world.run_ticks(1)
    assert world.stats()['rejected'] == 1
    assert latest(world).joints.count == 0


def test_joint_values_are_checked_when_submitted(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError):
        world.add_joint(Joint.hinge(1, anchor=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        world.add_joint(Joint.hinge(1, anchor=(0.0, 0.0, 0.0), axis=(0, 1, 0), limits=(1.0, -1.0)))
    with pytest.raises(ValueError):
        world.add_joint(Joint.ball(1, to=1, anchor=(0.0, 0.0, 0.0)))
