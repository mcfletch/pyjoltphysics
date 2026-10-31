"""Ragdolls: a skeleton of bodies joined by swing-twist joints, kinematic, powered or limp."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pyjoltphysics import (
    BodyBatch,
    Bone,
    MotionType,
    Ragdoll,
    RagdollMode,
    RagdollSettings,
    Shape,
    forward_kinematics,
)
from pyjoltphysics.testing import MakeWorld, floor, latest

HZ = 120.0


def _quat(axis: tuple[float, float, float], angle: float) -> npt.NDArray[np.float64]:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.array([*(a * math.sin(angle / 2)), math.cos(angle / 2)])


def _qmul(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    out: npt.NDArray[np.float64] = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return out


def _conj(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = q * np.array([-1.0, -1.0, -1.0, 1.0])
    return out


def _angle(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation angle of each quaternion, 0 to pi."""
    w = np.clip(np.abs(q[..., 3]) / np.linalg.norm(q, axis=-1), 0.0, 1.0)
    out: npt.NDArray[np.float64] = 2.0 * np.arccos(w)
    return out


def _local_rotations(
    settings: RagdollSettings, orientation: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.float64]:
    """Each bone's rotation relative to its parent's, from world orientations; the root's own."""
    out = orientation.astype(np.float64).copy()
    for i, bone in enumerate(settings.bones):
        if bone.parent >= 0:
            out[i] = _qmul(_conj(orientation[bone.parent].astype(np.float64)), orientation[i])
    return out


def _humanoid(**changes: Any) -> RagdollSettings:
    return RagdollSettings.humanoid(**changes)


def test_humanoid_skeleton_is_parents_first() -> None:
    settings = _humanoid()
    assert settings.bones[0].parent == -1
    for i, bone in enumerate(settings.bones):
        assert bone.parent < i
        assert bone.shape is not None
        assert bone.mass > 0
    assert settings.bone_index('head') > settings.bone_index('chest')
    assert sum(b.mass for b in settings.bones) == pytest.approx(70.0)


def test_from_skeleton_capsules_reach_the_child_joint() -> None:
    settings = RagdollSettings.from_skeleton(
        names=['root', 'mid', 'tip'],
        parents=[-1, 0, 1],
        translations=[(0.0, 1.0, 0.0), (0.0, 0.5, 0.0), (0.4, 0.0, 0.0)],
        radius=0.05,
        mass=3.0,
    )
    mid = settings.bones[1]
    assert mid.shape is not None
    # Mid runs from its joint to tip's, 0.4 m along +x.
    assert np.allclose(mid.shape_offset, (0.2, 0.0, 0.0))
    assert mid.shape.params[1] == pytest.approx(0.4 - 0.1)
    assert mid.twist_axis is not None
    assert np.allclose(mid.twist_axis, (1.0, 0.0, 0.0))
    assert sum(b.mass for b in settings.bones) == pytest.approx(3.0)


def test_bad_skeletons_are_refused() -> None:
    with pytest.raises(ValueError, match='parent'):
        RagdollSettings(bones=[Bone('a', 0, (0.0, 0.0, 0.0), shape=Shape.sphere(0.1))])
    with pytest.raises(ValueError, match='shape'):
        RagdollSettings(bones=[Bone('a', -1, (0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match='mass'):
        RagdollSettings(bones=[Bone('a', -1, (0.0, 0.0, 0.0), shape=Shape.sphere(0.1), mass=0.0)])


def test_bones_are_bodies_in_the_snapshot(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid()
    doll = world.add_ragdoll(settings, position=(0.0, 0.0, 0.0), mode=RagdollMode.KINEMATIC)
    assert isinstance(doll, Ragdoll)
    assert len(doll.bones) == len(settings.bones)
    world.run_ticks(2)
    snap = latest(world)
    rows = snap.ragdolls
    assert len(rows) == 1
    assert rows.id[0] == doll.id
    assert rows.mode[0] == RagdollMode.KINEMATIC
    first, count = int(rows.first_bone[0]), int(rows.bone_count[0])
    assert count == len(settings.bones)
    assert list(rows.bone_uid[first : first + count]) == list(doll.bones)
    body_rows = snap.rows_of(doll.bones)
    assert (body_rows >= 0).all()
    assert np.allclose(snap.position[body_rows], rows.bone_position[first : first + count])
    assert np.allclose(snap.orientation[body_rows], rows.bone_orientation[first : first + count])
    # At rest the bones stand where the skeleton's rest pose puts them.
    positions, _ = forward_kinematics(settings, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(rows.bone_position[first : first + count], positions, atol=1e-5)
    positions, orientations = doll.pose()
    assert positions.shape == (count, 3) and orientations.shape == (count, 4)


def test_limp_ragdoll_comes_to_rest_within_its_limits(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2)
    world.add_bodies(floor())
    settings = _humanoid()
    # Tipped forward so it falls onto its front, limbs folding.
    doll = world.add_ragdoll(
        settings, position=(0.0, 0.5, 0.0), rotation=_quat((1.0, 0.0, 0.0), 0.6)
    )
    world.run_ticks(int(6 * HZ))
    snap = latest(world)
    rows = snap.ragdolls
    first, count = int(rows.first_bone[0]), int(rows.bone_count[0])
    body_rows = snap.rows_of(doll.bones)
    assert np.linalg.norm(snap.linear_velocity[body_rows], axis=1).max() < 0.05
    assert rows.bone_position[first : first + count, 1].min() > 0.0
    assert rows.bone_position[first : first + count, 1].max() < 0.6
    local = _local_rotations(settings, rows.bone_orientation[first : first + count])
    for i, bone in enumerate(settings.bones):
        if bone.parent < 0:
            continue
        rest = np.asarray(bone.rotation, dtype=np.float64)
        offset = _qmul(_conj(rest), local[i])  # rotation away from the rest pose
        axis = np.asarray(bone.twist_axis, dtype=np.float64)
        # Swing-twist split about the twist axis.
        projection = np.dot(offset[:3], axis) * axis
        twist = np.array([*projection, offset[3]])
        twist /= np.linalg.norm(twist)
        swing = _qmul(offset, _conj(twist))
        twist_angle = 2 * math.atan2(float(np.dot(twist[:3], axis)), float(twist[3]))
        assert bone.twist[0] - 0.05 <= twist_angle <= bone.twist[1] + 0.05, bone.name
        assert _angle(swing) <= abs(bone.bend) + max(bone.swing) + 0.05, bone.name
        # Knees bend only backwards and elbows only forwards: a turn about +x of their sign.
        about_x = 2.0 * math.atan2(float(swing[0]), float(swing[3]))
        if bone.name.startswith('shin'):
            assert about_x > -0.05, bone.name
        if bone.name.startswith('lower_arm'):
            assert about_x < 0.05, bone.name


def test_kinematic_ragdoll_reproduces_the_animated_pose(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid()
    n = len(settings.bones)
    rest = np.array([b.rotation for b in settings.bones], dtype=np.float64)
    times = np.array([0.0, 0.5, 1.0])
    rotations = np.repeat(rest[None], 3, axis=0)
    elbow = settings.bone_index('lower_arm_l')
    knee = settings.bone_index('shin_r')
    rotations[1, elbow] = _qmul(rest[elbow], _quat((0.0, 0.0, 1.0), 1.2))
    rotations[1, knee] = _qmul(rest[knee], _quat((1.0, 0.0, 0.0), -0.9))
    translations = np.repeat(
        np.array([b.translation for b in settings.bones], dtype=np.float64)[None], 3, axis=0
    )
    translations[:, 0] += [[0.0, 0.0, 0.0], [0.0, 0.1, 0.3], [0.0, 0.0, 0.6]]
    doll = world.add_ragdoll(settings, position=(1.0, 0.0, 2.0), mode=RagdollMode.KINEMATIC)
    doll.play(times, rotations, translations, loop=True)
    for _ in range(5):
        world.run_ticks(23)
        snap = latest(world)
        t = (snap.time - 0.0) % 1.0
        # The clip evaluated at the snapshot's time, by linear and spherical interpolation.
        k = min(int(t / 0.5), 1)
        s = (t - times[k]) / 0.5
        trans = translations[k] * (1 - s) + translations[k + 1] * s
        rot = np.empty((n, 4))
        for j in range(n):
            a, b = rotations[k, j], rotations[k + 1, j]
            if np.dot(a, b) < 0:
                b = -b
            omega = math.acos(min(1.0, float(np.dot(a, b))))
            if omega < 1e-9:
                rot[j] = a
            else:
                rot[j] = (math.sin((1 - s) * omega) * a + math.sin(s * omega) * b) / math.sin(omega)
        want, _ = forward_kinematics(settings, rot, (1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0), trans)
        rows = snap.ragdolls
        first = int(rows.first_bone[0])
        got = rows.bone_position[first : first + n]
        assert np.abs(got - want).max() < 1e-3
        assert (snap.motion_type[snap.rows_of(doll.bones)] == MotionType.KINEMATIC).all()


def test_kinematic_ragdoll_follows_a_held_pose_and_its_placement(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    settings = _humanoid()
    doll = world.add_ragdoll(settings, mode=RagdollMode.KINEMATIC)
    rotations = np.array([b.rotation for b in settings.bones], dtype=np.float64)
    head = settings.bone_index('head')
    rotations[head] = _qmul(rotations[head], _quat((0.0, 1.0, 0.0), 0.7))
    turn = _quat((0.0, 1.0, 0.0), 1.0)
    doll.set_pose(rotations, position=(3.0, 0.0, -1.0), rotation=turn)
    world.run_ticks(3)
    want, wanted_q = forward_kinematics(settings, rotations, (3.0, 0.0, -1.0), turn)
    got, got_q = doll.pose()
    assert np.abs(got - want).max() < 1e-4
    assert np.abs(np.abs(np.sum(got_q * wanted_q, axis=1)) - 1).max() < 1e-5
    doll.place((4.0, 0.0, -1.0), turn)
    world.run_ticks(2)
    got, _ = doll.pose()
    assert np.abs(got - want - (1.0, 0.0, 0.0)).max() < 1e-4


def _deviation(settings: RagdollSettings, doll: Ragdoll) -> float:
    """The largest angle between a joint's rotation relative to its parent and its rest rotation."""
    _, orientation = doll.pose()
    local = _local_rotations(settings, orientation)
    rest = np.array([b.rotation for b in settings.bones], dtype=np.float64)
    angles = _angle(_qmul(_conj(rest[1:]), local[1:]))
    return float(angles.max())


def test_powered_ragdoll_struck_recovers_its_pose(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2, gravity=(0.0, -9.81, 0.0))
    world.add_bodies(floor())
    settings = _humanoid(root_force=4000.0, root_torque=1500.0)
    powered = world.add_ragdoll(settings, position=(0.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    limp_settings = _humanoid(root_force=4000.0, root_torque=1500.0)
    limp = world.add_ragdoll(limp_settings, position=(3.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    world.run_ticks(int(1.0 * HZ))
    assert _deviation(settings, powered) < 0.08
    limp.set_mode(RagdollMode.LIMP)
    arm = settings.bone_index('lower_arm_l')
    head = settings.bone_index('head')
    for doll in (powered, limp):
        doll.hit(arm, (0.0, 0.0, 40.0))
        # Above the head's joint, so the blow turns the head as well as pushing it.
        joints, _ = doll.pose()
        doll.hit(head, (25.0, 0.0, 0.0), point=joints[head] + (0.0, 0.15, 0.0))
    world.run_ticks(int(0.1 * HZ))
    struck = _deviation(settings, powered)
    assert struck > 0.15
    world.run_ticks(int(2.0 * HZ))
    assert _deviation(settings, powered) < 0.1
    assert _deviation(settings, limp) > 0.3
    # The powered one still stands where its root is driven.
    pelvis, _ = powered.pose()
    assert pelvis[0, 1] > 0.9


def test_mode_blends_kinematic_to_limp_and_powered_to_limp(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid(root_force=4000.0, root_torque=1500.0)
    kin = world.add_ragdoll(settings, position=(0.0, 0.5, 0.0), mode=RagdollMode.KINEMATIC)
    powered = world.add_ragdoll(settings, position=(3.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    world.run_ticks(30)
    before, _ = kin.pose()
    kin.set_mode(RagdollMode.LIMP)
    powered.set_mode(RagdollMode.LIMP, blend=0.5)
    world.run_ticks(int(0.25 * HZ))
    rows = latest(world).ragdolls
    strength = dict(zip(rows.id.tolist(), rows.strength.tolist(), strict=True))
    modes = dict(zip(rows.id.tolist(), rows.mode.tolist(), strict=True))
    assert modes[kin.id] == RagdollMode.LIMP
    assert strength[kin.id] == 0.0
    assert 0.3 < strength[powered.id] < 0.7
    snap = latest(world)
    assert (snap.motion_type[snap.rows_of(kin.bones)] == MotionType.DYNAMIC).all()
    after, _ = kin.pose()
    assert after[:, 1].mean() < before[:, 1].mean() - 0.1
    world.run_ticks(int(0.5 * HZ))
    rows = latest(world).ragdolls
    assert rows.strength[list(rows.id).index(powered.id)] == 0.0


def test_kinematic_blend_moves_from_the_physical_pose(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid()
    doll = world.add_ragdoll(
        settings, position=(0.0, 0.3, 0.0), rotation=_quat((1.0, 0.0, 0.0), 0.6)
    )
    world.run_ticks(int(2.0 * HZ))
    lying, _ = doll.pose()
    assert lying[:, 1].max() < 0.6
    doll.set_pose(None, position=(0.0, 0.0, 0.0))
    doll.set_mode(RagdollMode.KINEMATIC, blend=1.0)
    world.run_ticks(int(0.5 * HZ))
    halfway, _ = doll.pose()
    want, _ = forward_kinematics(settings, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert np.abs(halfway - want).max() > 0.05
    assert np.abs(halfway - lying).max() > 0.05
    world.run_ticks(int(0.6 * HZ))
    standing, _ = doll.pose()
    assert np.abs(standing - want).max() < 1e-3


def test_ragdoll_bones_honour_collision_groups(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor(group=2))
    passes = world.add_ragdoll(_humanoid(mask=1), position=(0.0, 0.3, 0.0))
    stops = world.add_ragdoll(_humanoid(), position=(3.0, 0.3, 0.0))
    world.run_ticks(int(1.0 * HZ))
    fell, _ = passes.pose()
    rests, _ = stops.pose()
    assert fell[:, 1].max() < -1.0
    assert rests[:, 1].min() > 0.0


def test_ragdoll_bones_meet_other_bodies(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    doll = world.add_ragdoll(_humanoid(), position=(0.0, 0.0, 0.0), mode=RagdollMode.KINEMATIC)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 3.0, 0.0)], radius=0.15, mass=2.0))
    world.run_ticks(int(1.0 * HZ))
    snap = latest(world)
    ball = snap.position[snap.rows[int(uids[0])]]
    head, _ = doll.pose()
    # The ball came down on the kinematic ragdoll and rolled off, or rests on it.
    assert ball[1] > 0.1
    assert not np.allclose(ball[[0, 2]], 0.0, atol=1e-3) or ball[1] > head[:, 1].max()


def test_self_collision_off_lets_limbs_cross(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    settings = _humanoid(self_collision=False)
    doll = world.add_ragdoll(settings, position=(0.0, 0.0, 0.0), mode=RagdollMode.KINEMATIC)
    world.run_ticks(2)
    assert latest(world).ragdolls.self_collision[0] == 0
    assert doll.self_collision is False


def test_removed_ragdoll_leaves_no_bodies(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    first = world.add_ragdoll(_humanoid(), position=(0.0, 0.3, 0.0))
    world.run_ticks(10)
    second = world.add_ragdoll(_humanoid(), position=(2.0, 0.3, 0.0))
    world.run_ticks(10)
    assert len(latest(world).ragdolls) == 2
    first.remove()
    world.run_ticks(1)
    snap = latest(world)
    assert len(snap.ragdolls) == 1
    assert snap.ragdolls.id[0] == second.id
    assert (snap.rows_of(first.bones) == -1).all()
    assert (snap.rows_of(second.bones) >= 0).all()
    assert world.stats()['ragdolls'] == 1
    # Removing a bone as a body is refused: the ragdoll owns it.
    rejected = world.stats()['rejected']
    world.remove_bodies(second.bones[:1])
    world.run_ticks(1)
    assert world.stats()['rejected'] == rejected + 1
    assert (latest(world).rows_of(second.bones) >= 0).all()


def test_strength_per_bone(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid(root_force=4000.0, root_torque=1500.0)
    doll = world.add_ragdoll(settings, position=(0.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    arm = settings.bone_index('upper_arm_l')
    strength = np.array([b.strength for b in settings.bones])
    strength[arm] = 0.0
    strength[settings.bone_index('lower_arm_l')] = 0.0
    doll.set_strength(strength)
    world.run_ticks(int(0.5 * HZ))
    doll.hit(settings.bone_index('lower_arm_l'), (0.0, 0.0, 30.0))
    world.run_ticks(int(1.0 * HZ))
    _, orientation = doll.pose()
    local = _local_rotations(settings, orientation)
    rest = np.array([b.rotation for b in settings.bones], dtype=np.float64)
    angles = _angle(_qmul(_conj(rest), local))
    limp = [arm, settings.bone_index('lower_arm_l')]
    held = [i for i in range(1, len(settings.bones)) if i not in limp]
    # The arm swings free after the blow; every other joint holds its pose.
    assert angles[limp].max() > 0.2
    assert angles[held].max() < 0.1


def test_ragdoll_settings_round_trip_as_dicts() -> None:
    settings = _humanoid(root_force=10.0)
    again = RagdollSettings.from_dict(settings.as_dict())
    assert again.as_dict() == settings.as_dict()
    assert again.names == settings.names


def test_bad_ragdoll_commands_are_refused(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    settings = _humanoid()
    doll = world.add_ragdoll(settings)
    with pytest.raises(ValueError):
        doll.set_pose(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        doll.play(np.array([0.0, 0.0]), np.zeros((2, len(settings.bones), 4)) + (0, 0, 0, 1))
    with pytest.raises(ValueError):
        doll.set_mode(RagdollMode.LIMP, blend=-1.0)


def test_still_ragdolls_sleep_and_wake_for_a_new_pose(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    settings = _humanoid(root_force=4000.0, root_torque=1500.0)
    powered = world.add_ragdoll(settings, position=(0.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    kinematic = world.add_ragdoll(settings, position=(3.0, 0.0, 0.0), mode=RagdollMode.KINEMATIC)
    world.run_ticks(int(4 * HZ))
    snap = latest(world)
    for doll in (powered, kinematic):
        assert not snap.awake[snap.rows_of(doll.bones)].any()
    standing, _ = powered.pose()
    assert standing[0, 1] > 0.9
    world.reset_stats()
    world.run_ticks(10)
    assert world.stats()['quiet_ticks'] > 0
    rotations = settings.rest_rotations
    rotations[settings.bone_index('head')] = _quat((0.0, 1.0, 0.0), 0.5)
    for doll in (powered, kinematic):
        doll.set_pose(rotations)
    world.run_ticks(int(1.0 * HZ))
    for doll in (powered, kinematic):
        _, orientation = doll.pose()
        local = _local_rotations(settings, orientation)
        head = settings.bone_index('head')
        assert _angle(_qmul(_conj(rotations[head]), local[head])) < 0.05


def test_bend_swings_a_joint_one_way(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1, gravity=(0.0, 0.0, 0.0))
    world.add_bodies(floor())
    settings = RagdollSettings(
        bones=[
            Bone('base', -1, (0.0, 0.1, 0.0), shape=Shape.box((1.0, 0.2, 1.0)), mass=200.0),
            Bone(
                'arm',
                0,
                (0.0, 0.1, 0.0),
                shape=Shape.capsule(0.05, 0.5),
                shape_offset=(0.0, 0.3, 0.0),
                mass=1.0,
                twist_axis=(0.0, 1.0, 0.0),
                plane_axis=(0.0, 0.0, 1.0),
                twist=(-0.1, 0.1),
                swing=(0.6, 0.05),
                bend=0.6,
                friction=0.0,
            ),
        ],
        angular_damping=0.0,
    )
    doll = world.add_ragdoll(settings)
    angles = []
    for push in (3.0, -3.0, 3.0):
        doll.hit(1, (push, 0.0, 0.0), point=None)
        for _ in range(int(0.6 * HZ)):
            world.run_ticks(1)
            _, orientation = doll.pose()
            local = _qmul(
                _conj(orientation[0].astype(np.float64)), orientation[1].astype(np.float64)
            )
            # A turn about the plane axis, +z, tips the arm towards -x.
            angles.append(2.0 * math.atan2(float(local[2]), float(local[3])))
    assert min(angles) > -0.05
    assert max(angles) == pytest.approx(1.2, abs=0.05)
