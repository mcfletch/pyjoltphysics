"""Joints, materials, collision filters, gravity zones and tracks: Jolt against omi_physics.

Each scenario builds the same bodies in both engines at 120 Hz and compares
what the plan's differential table names, within its tolerances
(openglcontext/plans/NATIVE-PHYSICS.md, "P3 as built"). omi_physics' side is
its recorded answer (:func:`differential.engines.reference`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from differential.engines import reference
from pyjoltphysics import (
    BodyBatch,
    EventKind,
    EventPhase,
    GravityZone,
    Joint,
    MotionType,
    Motor,
    MotorMode,
    Track,
    World,
    WorldSettings,
    ZoneField,
)
from pyjoltphysics.testing import latest

pytestmark = pytest.mark.differential

DT = 1.0 / 120.0
G = 9.81


def omi_world(gravity: float = G) -> Any:
    from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in
    from omi_physics import world as omi  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

    world = omi.PhysicsWorld(
        gravity=model.Gravity(gravity=gravity, direction=(0, -1, 0)), fixed_dt=DT
    )
    world.report_persist = False
    return world


def omi_box(
    world: Any,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    mass: float = 1.0,
    motion: str = 'dynamic',
    material: int = -1,
    collision_filter: int = -1,
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> int:
    from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

    shape = world.add_shape(model.Shape.box(size))
    index: int = world.add_body(
        model.Motion(type=motion, mass=mass),
        collider=model.Collider(
            shape=shape, physicsMaterial=material, collisionFilter=collision_filter
        ),
        position=position,
        orientation=orientation,
    )
    return index


def jolt_box(
    world: World,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    **values: Any,
) -> int:
    uids, _ = world.add_bodies(BodyBatch.boxes([position], size=size, **values))
    return int(uids[0])


def period(angles: list[float], hz: float) -> float:
    a = np.asarray(angles) - np.mean(angles)
    up = np.nonzero((a[:-1] < 0) & (a[1:] >= 0))[0]
    return float(np.mean(np.diff(up))) / hz


# -- joints ------------------------------------------------------------------------


def test_hinge_pendulum_period() -> None:
    """Period within 2% of each other and of 2 pi sqrt(L / g)."""
    start = (math.sin(0.15), 2.0 - math.cos(0.15), 0.0)
    ticks = 600

    def omi_period() -> float:
        from omi_physics.joints import HingeJoint  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        bob = omi_box(o, start, (0.1, 0.1, 0.1))
        hinge = o.add_joint_constraint(HingeJoint(-1, bob, (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)))
        omi_angles = []
        for _ in range(ticks):
            o.step(DT)
            omi_angles.append(hinge.angle(o))
        return period(omi_angles, 120.0)

    po = reference('p3/pendulum', omi_period)
    with World(WorldSettings(threads=1, allow_sleeping=False)) as j:
        uid = jolt_box(j, start, (0.1, 0.1, 0.1))
        j.add_joint(Joint.hinge(uid, anchor=(0.0, 2.0, 0.0), axis=(0, 0, 1)))
        jolt_angles = []
        for _ in range(ticks):
            j.run_ticks(1)
            jolt_angles.append(float(latest(j).joints.value[0]))
    expected = 2 * math.pi * math.sqrt(1.0 / G)
    pj = period(jolt_angles, 120.0)
    assert abs(pj - expected) < 0.02 * expected
    assert abs(po - pj) < 0.02 * expected


def test_ball_chain_link_distance() -> None:
    """Every link end stays within 1 cm of the next link's start in both engines."""
    n = 6

    def ends(position: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z, w = (float(v) for v in quat)
        axis = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        return position - 0.25 * axis, position + 0.25 * axis

    def worst(positions: list[np.ndarray], quats: list[np.ndarray]) -> float:
        pairs = [ends(p, q) for p, q in zip(positions, quats, strict=True)]
        return max(float(np.linalg.norm(pairs[i][1] - pairs[i + 1][0])) for i in range(n - 1))

    def omi_gap() -> float:
        from omi_physics.joints import BallJoint  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        links = [omi_box(o, (0.5 * (i + 1), 5.0, 0.0), (0.4, 0.1, 0.1)) for i in range(n)]
        o.add_joint_constraint(BallJoint(-1, links[0], (0.25, 5.0, 0.0)))
        for i in range(n - 1):
            o.add_joint_constraint(
                BallJoint(links[i], links[i + 1], (0.5 * (i + 1) + 0.25, 5.0, 0.0))
            )
        for _ in range(240):
            o.step(DT)
        return worst(
            [np.array(o.position[i]) for i in links], [np.array(o.orientation[i]) for i in links]
        )

    omi_gap_m = reference('p3/chain', omi_gap)
    with World(WorldSettings(threads=1, allow_sleeping=False)) as j:
        uids = [jolt_box(j, (0.5 * (i + 1), 5.0, 0.0), (0.4, 0.1, 0.1)) for i in range(n)]
        joints = [Joint.ball(uids[0], anchor=(0.25, 5.0, 0.0))]
        joints += [
            Joint.ball(uids[i + 1], to=uids[i], anchor=(0.5 * (i + 1) + 0.25, 5.0, 0.0))
            for i in range(n - 1)
        ]
        j.add_joints(joints)
        j.run_ticks(240)
        s = latest(j)
        rows = s.rows_of(uids)
        jolt_gap = worst(list(s.position[rows]), list(s.orientation[rows].astype(np.float64)))
    assert jolt_gap < 0.01
    assert omi_gap_m < 0.01


def test_hinge_limit_angle() -> None:
    """A box falling against a 0.5 rad stop rests within 1 degree of it in both engines.

    The box is 0.4 m: omi_physics' stops throw a 0.2 m box past them and a
    0.1 m box away (a divergence the plan records); Jolt holds all three.
    """

    def omi_angle() -> float:
        from omi_physics.joints import HingeJoint  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        bar = omi_box(o, (1.0, 2.0, 0.0), (0.4, 0.4, 0.4))
        hinge = o.add_joint_constraint(
            HingeJoint(-1, bar, (0.0, 2.0, 0.0), (0, 0, 1), lower=-0.5, upper=0.5)
        )
        for _ in range(600):
            o.step(DT)
        return float(hinge.angle(o))

    omi_rest = reference('p3/hinge_limit', omi_angle)
    jolt_angles = []
    for size in (0.1, 0.2, 0.4):
        with World(WorldSettings(threads=1)) as j:
            uid = jolt_box(j, (1.0, 2.0, 0.0), (size, size, size))
            j.add_joint(
                Joint.hinge(uid, anchor=(0.0, 2.0, 0.0), axis=(0, 0, 1), limits=(-0.5, 0.5))
            )
            j.run_ticks(600)
            jolt_angles.append(float(latest(j).joints.value[0]))
    assert max(abs(a + 0.5) for a in jolt_angles) < math.radians(1.0)
    assert abs(omi_rest + 0.5) < math.radians(1.0)


def test_hinge_motor_speed() -> None:
    """A wheel driven at 3 rad/s turns within 2% of it in both engines."""

    def omi_spin() -> float:
        from omi_physics.joints import HingeJoint  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world(gravity=0.0)
        wheel = omi_box(o, (0.0, 1.0, 0.0), (1.0, 0.1, 1.0), mass=5.0)
        o.add_joint_constraint(
            HingeJoint(-1, wheel, (0.0, 1.0, 0.0), (0, 1, 0), motor_speed=3.0, motor_torque=200.0)
        )
        for _ in range(120):
            o.step(DT)
        return float(o.angular_velocity[wheel][1])

    omi_speed = reference('p3/motor', omi_spin)
    with World(WorldSettings(threads=1, gravity=(0.0, 0.0, 0.0))) as j:
        uid = jolt_box(j, (0.0, 1.0, 0.0), (1.0, 0.1, 1.0), mass=5.0)
        j.add_joint(
            Joint.hinge(
                uid,
                anchor=(0.0, 1.0, 0.0),
                axis=(0, 1, 0),
                motor=Motor(MotorMode.VELOCITY, 3.0, 200.0),
            )
        )
        j.run_ticks(120)
        jolt_speed = float(latest(j).angular_velocity[0, 1])
    assert abs(jolt_speed - 3.0) < 0.06
    assert abs(omi_speed - 3.0) < 0.06


def test_slider_and_distance_travel_at_their_limits() -> None:
    """A slider stops at its lower limit, a rigid distance holds its length: 1 mm."""

    def omi_travel() -> tuple[float, float]:
        from omi_physics.joints import DistanceConstraint, SliderJoint  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        car = omi_box(o, (0.0, 2.0, 0.0), (0.2, 0.2, 0.2))
        o.add_joint_constraint(SliderJoint(-1, car, (0, 1, 0), lower=-1.0, upper=0.5))
        bob = omi_box(o, (4.0, 3.0, 0.0), (0.2, 0.2, 0.2))
        o.add_joint_constraint(DistanceConstraint(-1, bob, np.array([2.0, 3.0, 0.0]), np.zeros(3)))
        for _ in range(240):
            o.step(DT)
        return float(o.position[car][1]), float(
            np.linalg.norm(np.array(o.position[bob]) - (2.0, 3.0, 0.0))
        )

    omi_car, omi_len = reference('p3/slider', omi_travel)
    with World(WorldSettings(threads=1)) as j:
        c = jolt_box(j, (0.0, 2.0, 0.0), (0.2, 0.2, 0.2))
        b = jolt_box(j, (4.0, 3.0, 0.0), (0.2, 0.2, 0.2))
        j.add_joints(
            [
                Joint.slider(c, axis=(0, 1, 0), limits=(-1.0, 0.5)),
                Joint.distance(b, anchor=(4.0, 3.0, 0.0), to_anchor=(2.0, 3.0, 0.0)),
            ]
        )
        j.run_ticks(240)
        s = latest(j)
        jolt_car = float(s.position[s.rows[c], 1])
        jolt_len = float(np.linalg.norm(s.position[s.rows[b]] - (2.0, 3.0, 0.0)))
    assert abs(jolt_car - 1.0) < 1e-3 and abs(omi_car - 1.0) < 1e-3
    assert abs(jolt_len - 2.0) < 1e-3 and abs(omi_len - 2.0) < 1e-3


# -- materials ---------------------------------------------------------------------

SLOPE = math.radians(20.0)


def slides(
    key: str,
    make_jolt_materials: Callable[[World], list[int]],
    omi_materials: Callable[[Any], list[int]],
    ramp: int,
    boxes: list[int],
) -> tuple[list[float], list[float]]:
    """How far boxes of materials ``boxes`` slide down a 20 degree ramp of ``ramp`` in 1 s, by engine;
    omi_physics' recorded under ``key``."""
    turn = (0.0, 0.0, math.sin(SLOPE / 2), math.cos(SLOPE / 2))
    normal = np.array([-math.sin(SLOPE), math.cos(SLOPE), 0.0])
    down = np.array([-math.cos(SLOPE), -math.sin(SLOPE), 0.0])
    starts = [
        normal * 0.751 + np.array([0.0, 0.0, 1.2 * k - 1.2 * (len(boxes) - 1) / 2])
        for k in range(len(boxes))
    ]

    def omi_slides() -> list[float]:
        o = omi_world()
        om = omi_materials(o)
        omi_box(
            o,
            (0.0, 0.0, 0.0),
            (40.0, 1.0, 8.0),
            motion='static',
            material=om[ramp],
            orientation=turn,
        )
        obodies = [
            omi_box(o, tuple(p), (0.5, 0.5, 0.5), material=om[m], orientation=turn)
            for p, m in zip(starts, boxes, strict=True)
        ]
        for _ in range(120):
            o.step(DT)
        return [
            float(np.dot(np.array(o.position[i]) - p, down))
            for i, p in zip(obodies, starts, strict=True)
        ]

    omi_d = reference(key, omi_slides)
    with World(WorldSettings(threads=1)) as j:
        jm = make_jolt_materials(j)
        jolt_box(
            j,
            (0.0, 0.0, 0.0),
            (40.0, 1.0, 8.0),
            motion=MotionType.STATIC,
            orientations=[turn],
            material=jm[ramp],
        )
        uids = [
            jolt_box(j, tuple(p), (0.5, 0.5, 0.5), orientations=[turn], material=jm[m])
            for p, m in zip(starts, boxes, strict=True)
        ]
        j.run_ticks(120)
        s = latest(j)
        jolt_d = [
            float(np.dot(s.position[s.rows[u]] - p, down))
            for u, p in zip(uids, starts, strict=True)
        ]
    return omi_d, jolt_d


def test_pair_friction_grid() -> None:
    """marble-demo's grid: four pair frictions on one ramp; same order, distances within 15%."""
    mus = [0.05, 0.15, 0.25, 0.32]

    def jolt_materials(world: World) -> list[int]:
        ramp = world.materials.define(friction=0.9)
        made = [world.materials.define(friction=0.9) for _ in mus]
        for m, mu in zip(made, mus, strict=True):
            world.materials.pair(ramp, m, friction=mu)
        return [ramp, *made]

    def omi_materials(world: Any) -> list[int]:
        from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        ramp = world.add_material(model.Material(staticFriction=0.9, dynamicFriction=0.9))
        made = [
            world.add_material(model.Material(staticFriction=0.9, dynamicFriction=0.9)) for _ in mus
        ]
        for m, mu in zip(made, mus, strict=True):
            world.set_pair_friction(ramp, m, mu, mu)
        return [ramp, *made]

    omi_d, jolt_d = slides('p3/pair_grid', jolt_materials, omi_materials, 0, [1, 2, 3, 4])
    assert np.argsort(omi_d).tolist() == np.argsort(jolt_d).tolist()
    for a, b in zip(omi_d, jolt_d, strict=True):
        assert abs(a - b) < 0.15 * max(a, b)


def test_combine_average_friction() -> None:
    """Friction 0.4 and 0.1 averaged to 0.25 in both engines: within 10% or 3 cm."""

    def jolt_materials(world: World) -> list[int]:
        return [world.materials.define(friction=0.4), world.materials.define(friction=0.1)]

    def omi_materials(world: Any) -> list[int]:
        from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        return [
            world.add_material(model.Material(staticFriction=0.4, dynamicFriction=0.4)),
            world.add_material(model.Material(staticFriction=0.1, dynamicFriction=0.1)),
        ]

    (omi_d,), (jolt_d,) = slides('p3/average', jolt_materials, omi_materials, 0, [1])
    expected = 0.5 * G * (math.sin(SLOPE) - 0.25 * math.cos(SLOPE))
    for d in (omi_d, jolt_d):
        assert abs(d - expected) < max(0.03, 0.1 * expected)


def bounce_ratio(apexes: list[float]) -> float:
    return apexes[1] / apexes[0]


def test_material_restitution_bounce() -> None:
    """Restitution 0.7 by material: second apex over first within 0.05 of each other."""
    drop = 2.0

    def apexes(h: list[float]) -> list[float]:
        a = np.asarray(h)
        peaks = [
            float(a[i])
            for i in range(1, len(a) - 1)
            if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > 0.2
        ]
        return [drop] + peaks

    def omi_ratio() -> float:
        from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        m = o.add_material(model.Material(restitution=0.7, restitutionCombine=model.AVERAGE))
        omi_box(o, (0.0, -0.5, 0.0), (20.0, 1.0, 20.0), motion='static', material=m)
        s = o.add_shape(model.Shape.sphere(0.1))
        ball = o.add_body(
            model.Motion(),
            collider=model.Collider(shape=s, physicsMaterial=m),
            position=(0.0, drop, 0.0),
        )
        heights_o = []
        for _ in range(600):
            o.step(DT)
            heights_o.append(float(o.position[ball][1]))
        return bounce_ratio(apexes(heights_o)[1:])

    ro = reference('p3/restitution', omi_ratio)
    with World(WorldSettings(threads=1)) as j:
        jm = j.materials.define(friction=0.6, restitution=0.7)
        jolt_box(j, (0.0, -0.5, 0.0), (20.0, 1.0, 20.0), motion=MotionType.STATIC, material=jm)
        uids, _ = j.add_bodies(BodyBatch.spheres([(0.0, drop, 0.0)], radius=0.1, material=jm))
        heights_j = []
        for _ in range(600):
            j.run_ticks(1)
            snap = latest(j)
            heights_j.append(float(snap.position[snap.rows[int(uids[0])], 1]))

    rj = bounce_ratio(apexes(heights_j)[1:])
    assert abs(ro - rj) < 0.05


# -- collision filters ---------------------------------------------------------------


def test_collision_filters_touch_the_same_pairs() -> None:
    """Three filter systems: the set of pairs that touched is the same in both engines."""
    systems = [('ball',), ('shard',), ('wall',)]
    collide = {('ball', 'shard'): False}

    def meets(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        return collide.get((a[0], b[0]), collide.get((b[0], a[0]), True))

    which = [0, 1, 0, 1, 0, 1]

    def omi_touched() -> list[tuple[int, int]]:
        from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        filters = [
            o.add_filter(
                model.CollisionFilter(collisionSystems=('ball',), notCollideWithSystems=('shard',))
            ),
            o.add_filter(
                model.CollisionFilter(collisionSystems=('shard',), notCollideWithSystems=('ball',))
            ),
            o.add_filter(model.CollisionFilter(collisionSystems=('wall',))),
        ]
        floor_o = omi_box(
            o, (0.0, -0.5, 0.0), (20.0, 1.0, 20.0), motion='static', collision_filter=filters[2]
        )
        bodies_o = [
            omi_box(o, (0.0, 0.5 + 0.6 * k, 0.0), (0.4, 0.4, 0.4), collision_filter=filters[f])
            for k, f in enumerate(which)
        ]
        touched_o: set[tuple[int, int]] = set()
        names_o = {floor_o: 0, **{b: k + 1 for k, b in enumerate(bodies_o)}}

        def heard(event: Any) -> None:
            if event.phase == 'begin':
                first, second = sorted((names_o[event.a.index], names_o[event.b.index]))
                touched_o.add((first, second))

        o.add_contact_listener(heard)
        for _ in range(240):
            o.step(DT)
        return sorted(touched_o)

    touched_o = {(int(a), int(b)) for a, b in reference('p3/filters', omi_touched)}
    table = [[meets(systems[a], systems[b]) for b in range(3)] for a in range(3)]
    with World(WorldSettings(threads=1)) as j:
        j.set_filter_table(table)
        floor_j = jolt_box(
            j, (0.0, -0.5, 0.0), (20.0, 1.0, 20.0), motion=MotionType.STATIC, filter=2
        )
        bodies_j = [
            jolt_box(j, (0.0, 0.5 + 0.6 * k, 0.0), (0.4, 0.4, 0.4), filter=f)
            for k, f in enumerate(which)
        ]
        names_j = {floor_j: 0, **{b: k + 1 for k, b in enumerate(bodies_j)}}
        j.run_ticks(240)
        events = j.drain_events()
        begins = events[
            (events['kind'] == EventKind.CONTACT) & (events['phase'] == EventPhase.BEGIN)
        ]
        touched_j = {
            tuple(sorted((names_j[int(a)], names_j[int(b)])))
            for a, b in zip(begins['a'], begins['b'], strict=True)
        }
    assert touched_j == touched_o
    assert touched_j  # some pairs met
    assert not any(which[a - 1] != which[b - 1] for a, b in touched_j if a and b)


# -- gravity zones -----------------------------------------------------------------------


def test_gravity_zones_position_at_two_seconds() -> None:
    """Directional, point and replacing zones: positions at 2 s within 1% of the distance fallen."""
    starts = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (40.0, 0.0, 0.0)]

    def omi_positions() -> list[list[float]]:
        from omi_physics import model  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in
        from omi_physics.gravity import BoxRegion, GravityVolume, SphereRegion  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        o.add_gravity_volume(
            GravityVolume(
                model.Gravity(gravity=3.0, direction=(1, 1, 0)), BoxRegion((0, 0, 0), (5, 50, 5))
            )
        )
        o.add_gravity_volume(
            GravityVolume(
                model.Gravity(type=model.POINT, gravity=4.0, center=(25.0, 0.0, 0.0)),
                SphereRegion((20, 0, 0), 30.0),
            )
        )
        o.add_gravity_volume(
            GravityVolume(
                model.Gravity(gravity=1.5, direction=(0, 1, 0), replace=True, priority=3),
                SphereRegion((40, 0, 0), 30.0),
            )
        )
        bodies_o = [omi_box(o, p, (0.1, 0.1, 0.1)) for p in starts]
        for _ in range(240):
            o.step(DT)
        return [list(map(float, o.position[b])) for b in bodies_o]

    omi_p = reference('p3/zones', omi_positions)
    with World(WorldSettings(threads=1, allow_sleeping=False)) as j:
        uids = [jolt_box(j, p, (0.1, 0.1, 0.1)) for p in starts]
        j.set_gravity_zones(
            [
                GravityZone.box((0, 0, 0), (10, 100, 10), gravity=3.0, direction=(1, 1, 0)),
                GravityZone.sphere(
                    (20, 0, 0), 30.0, field=ZoneField.POINT, gravity=4.0, center=(25.0, 0.0, 0.0)
                ),
                GravityZone.sphere(
                    (40, 0, 0), 30.0, gravity=1.5, direction=(0, 1, 0), replace=True, priority=3
                ),
            ]
        )
        j.run_ticks(240)
        s = latest(j)
        jolt_p = [s.position[s.rows[u]].copy() for u in uids]
    for found, p, start in zip(omi_p, jolt_p, starts, strict=True):
        po = np.array(found)
        travelled = np.linalg.norm(po - start)
        assert np.linalg.norm(po - p) < 0.01 * travelled


# -- kinematic tracks ---------------------------------------------------------------------


def test_kinematic_track_pose_each_tick() -> None:
    """A platform on a circle: Jolt's track and drive within 1e-6 m of the pose each tick; omi's animator too."""

    def pose(t: float) -> tuple[float, float, float]:
        return (2.0 * math.cos(t), 1.0 + 0.5 * t, 2.0 * math.sin(t))

    def omi_worst() -> float:
        from omi_physics.kinematic import KinematicAnimator  # noqa: PLC0415 omi_physics is optional: the recorded answers stand in

        o = omi_world()
        plat = omi_box(o, pose(0.0), (1.0, 0.1, 1.0), motion='kinematic')
        animator = KinematicAnimator(o, plat, pose)
        worst = 0.0
        for tick in range(1, 241):
            animator.update(DT)
            o.step(DT)
            worst = max(worst, float(np.abs(np.array(o.position[plat]) - pose(tick * DT)).max()))
        return worst

    omi_err = reference('p3/kinematic', omi_worst)
    with World(WorldSettings(threads=1)) as j:
        driven = jolt_box(j, pose(0.0), (1.0, 0.1, 1.0), motion=MotionType.KINEMATIC)
        j.drive(driven, pose)
        jolt_err = 0.0
        for tick in range(1, 241):
            j.run_ticks(1)
            t = tick * DT
            s = latest(j)
            jolt_err = max(jolt_err, float(np.abs(s.position[s.rows[driven]] - pose(t)).max()))
    assert jolt_err < 1e-6
    assert omi_err < 1e-6


def test_sampled_track_matches_linear_interpolation() -> None:
    times = np.linspace(0.0, 2.0, 9)
    positions = np.stack([times, np.sin(times), np.zeros_like(times)], axis=1)
    with World(WorldSettings(threads=1)) as j:
        uid = jolt_box(j, tuple(positions[0]), (1.0, 0.1, 1.0), motion=MotionType.KINEMATIC)
        j.set_track([uid], Track(times, positions))
        worst = 0.0
        for tick in range(1, 241):
            j.run_ticks(1)
            t = min(tick * DT, 2.0)
            expected = [np.interp(t, times, positions[:, k]) for k in range(3)]
            s = latest(j)
            worst = max(worst, float(np.abs(s.position[s.rows[uid]] - expected).max()))
    assert worst < 1e-6
