"""Hostile values through the public API: each is refused with an exception or leaves every pose finite."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from test_vehicles import car_settings

from pyjoltphysics import (
    BodyBatch,
    CharacterCapabilities,
    Joint,
    MotionType,
    RagdollMode,
    RagdollSettings,
    Shape,
    SoftBodySettings,
    VehicleKind,
    VehicleTrack,
    World,
    WorldError,
    WorldSettings,
)
from pyjoltphysics.testing import MakeWorld, floor, latest
from pyjoltphysics.vehicles import Differential

Case = Callable[[World], object]
REFUSED = (ValueError, TypeError, OverflowError, RuntimeError)


def finite(world: World) -> bool:
    snapshot = world.latest()
    if snapshot is None:
        return True
    return all(
        np.isfinite(getattr(snapshot, name)).all()
        for name in ('position', 'orientation', 'linear_velocity', 'angular_velocity')
    )


def sphere(world: World, **values: Any) -> int:
    (uid,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)], radius=0.5, **values))
    return int(uid)


def car(world: World) -> int:
    (uid,), _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.0, 0.0)], size=(1.8, 0.6, 4.0), mass=1000.0)
    )
    return int(uid)


def cloth(**changes: Any) -> SoftBodySettings:
    return dataclasses.replace(
        SoftBodySettings.cloth(width=1.0, depth=1.0, columns=6, rows=6, mass=1.0), **changes
    )


def doll(world: World) -> Any:
    return world.add_ragdoll(
        RagdollSettings.humanoid(), position=(3.0, 1.0, 0.0), mode=RagdollMode.LIMP
    )


def removed_handle_ops(world: World) -> None:
    uid = sphere(world)
    world.remove_bodies([uid])
    world.remove_bodies([uid, uid, uid])
    world.run_ticks(1)
    world.set_transforms([uid], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)])
    world.add_forces([uid], [(1.0, 0.0, 0.0)])
    world.set_motion([uid], mass=[2.0])
    world.set_drag([uid], [1.0])
    world.activate([uid])
    world.remove_bodies([uid])


def vehicle_chassis_made_static(world: World) -> None:
    uid = car(world)
    world.add_vehicle(uid, car_settings())
    world.run_ticks(2)
    world.set_motion_types([uid], [MotionType.STATIC])
    world.run_ticks(5)
    world.set_motion_types([uid], [MotionType.KINEMATIC])


def ops_on_soft_body(world: World) -> None:
    uid = int(world.add_soft_body(cloth(), position=(0.0, 2.0, 0.0)).uid)
    world.run_ticks(2)
    world.set_motion_types([uid], [MotionType.KINEMATIC])
    world.set_motion([uid], mass=[5.0], inertia=np.eye(3))
    world.add_torques([uid], [(1.0, 1.0, 1.0)])
    world.add_impulses([uid], [(1.0, 1.0, 1.0)])
    world.set_velocities([uid], linear=[(1.0, 0.0, 0.0)], angular=[(1.0, 0.0, 0.0)])
    world.set_transforms([uid], [(0.0, 5.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)])
    world.set_drag([uid], [1.0])
    world.move_kinematic([uid], [(0.0, 1.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)])


def bone_ops(world: World) -> None:
    d = doll(world)
    world.run_ticks(2)
    bones = [int(b) for b in d.bones[:3]]
    world.set_motion_types(bones, [MotionType.STATIC] * 3)
    world.run_ticks(3)
    world.set_motion_types(bones, [MotionType.KINEMATIC] * 3)
    world.set_drag(bones, [1.0] * 3)
    world.remove_bodies(bones)
    world.run_ticks(3)
    world.remove_ragdolls([d.id, d.id])
    world.run_ticks(2)
    world.remove_ragdolls([d.id])


def vehicle_double_remove(world: World) -> None:
    uid = car(world)
    first = world.add_vehicle(uid, car_settings())
    second = world.add_vehicle(uid, car_settings())
    world.run_ticks(2)
    world.remove_vehicles([first.id, first.id, second.id])
    world.remove_bodies([uid])
    world.remove_vehicles([first.id])
    world.run_ticks(3)
    third = world.add_vehicle(uid, car_settings())
    world.vehicle_inputs([third.id, first.id], [[1.0, 0.0, 0.0, 0.0]] * 2)


def vehicle(change: dict[str, Any]) -> Case:
    def run(world: World) -> None:
        added = world.add_vehicle(car(world), dataclasses.replace(car_settings(), **change))
        world.run_ticks(5)
        world.vehicle_inputs([added.id], [[1.0, 1.0, 0.0, 0.0]])
        world.run_ticks(30)

    return run


def character_handles(world: World) -> None:
    character = world.add_character((0.0, 1.0, 0.0))
    (uid,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)], size=(2.0, 0.2, 2.0), motion=2))
    world.run_ticks(5)
    world.remove_bodies([uid])
    world.run_ticks(3)
    world.remove_characters([character.id])
    world.remove_characters([character.id, character.id])
    character.jump()
    character.set_ignore([2**32 - 1, int(uid)])


def shape(make: Callable[[], Shape]) -> Case:
    def run(world: World) -> None:
        sid = world.add_shape(make())
        world.add_bodies(BodyBatch(positions=[(0.0, 3.0, 0.0)], shape=sid, motion=0))
        world.add_bodies(BodyBatch.spheres([(0.0, 6.0, 0.0)], radius=0.3))

    return run


CASES: dict[str, Case] = {
    # bodies
    'far position': lambda w: w.add_bodies(BodyBatch.spheres([(1e30, 1e30, 1e30)])),
    'huge radius': lambda w: w.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)], radius=1e30)),
    'tiny radius': lambda w: w.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)], radius=1e-30)),
    'negative radius': lambda w: w.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)], radius=-1.0)),
    'huge mass': lambda w: sphere(w, mass=1e38),
    'tiny mass': lambda w: sphere(w, mass=1e-45),
    'huge velocity': lambda w: sphere(w, linear_velocity=(1e30, 0.0, 0.0)),
    'velocity past a float': lambda w: sphere(w, linear_velocity=(1e300, 0.0, 0.0)),
    'huge gravity factor': lambda w: sphere(w, gravity_factor=1e38),
    'zero box': lambda w: w.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(0.0, 0.0, 0.0))),
    'huge damping': lambda w: sphere(w, linear_damping=1e38, angular_damping=1e38),
    'far transform': lambda w: w.set_transforms(
        [sphere(w)], [(1e300, 0.0, 0.0)], [(0.0, 0.0, 0.0, 1.0)]
    ),
    'huge velocity set': lambda w: w.set_velocities([sphere(w)], linear=[(1e38, 1e38, 1e38)]),
    'velocity set past a float': lambda w: w.set_velocities(
        [sphere(w)], linear=[(1e300, 0.0, 0.0)]
    ),
    'force past a float': lambda w: w.add_forces([sphere(w)], [(1e300, 0.0, 0.0)]),
    'huge force': lambda w: w.add_forces([sphere(w)], [(1e38, 0.0, 0.0)]),
    'gravity past a float': lambda w: (sphere(w), w.set_gravity((0.0, -1e300, 0.0))),
    'huge gravity': lambda w: (sphere(w), w.set_gravity((0.0, -1e38, 0.0))),
    'removed handles': removed_handle_ops,
    'huge uids': lambda w: (w.remove_bodies([2**32 - 1, 2**31]), w.bodies_state([2**32 - 1, 0])),
    'remove a pending add': lambda w: w.remove_bodies(
        w.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)] * 10), at_tick=5)[0]
    ),
    # arrays
    'non-contiguous forces': lambda w: w.add_forces([sphere(w)], np.zeros((1, 3))[:, ::2]),
    'forces of the wrong length': lambda w: w.add_forces([sphere(w)], np.zeros((2, 3))),
    'int8 forces': lambda w: w.add_forces([sphere(w)], np.zeros((1, 3), np.int8)),
    'uid wrapped past 32 bits': lambda w: w.add_forces(
        np.array([sphere(w)], np.int64) - 2**33, np.zeros((1, 3))
    ),
    'negative uid': lambda w: w.add_forces([-1], np.zeros((1, 3))),
    'fractional uid': lambda w: w.add_forces(
        np.array([sphere(w)], np.float64) + 0.5, np.zeros((1, 3))
    ),
    '2-D uids': lambda w: w.add_forces(np.array([[sphere(w)]]), np.zeros((1, 3))),
    'NaN forces': lambda w: w.add_forces([sphere(w)], np.full((1, 3), np.nan)),
    # queries
    'ray of tiny direction': lambda w: w.raycast((0.0, 10.0, 0.0), (0.0, -1e-300, 0.0), 100.0),
    'ray of huge length': lambda w: w.raycast((0.0, 10.0, 0.0), (0.0, -1.0, 0.0), 1e300),
    'ray from far away': lambda w: w.raycast((1e300, 0.0, 0.0), (0.0, -1.0, 0.0), 1.0),
    'huge sweep': lambda w: w.sweep_sphere(0.5, (0.0, 10.0, 0.0), (0.0, -1e300, 0.0)),
    # shapes
    'empty hull': shape(lambda: Shape.convex(np.zeros((0, 3)))),
    'hull of a line': shape(
        lambda: Shape.convex(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float))
    ),
    'huge hull': shape(lambda: Shape.convex(np.random.default_rng(1).normal(size=(50, 3)) * 1e30)),
    'mesh past its vertices': shape(
        lambda: Shape.mesh(np.zeros((3, 3)), np.array([[0, 1, 7]], np.uint32))
    ),
    'degenerate mesh': shape(
        lambda: Shape.mesh(np.zeros((3, 3)), np.array([[0, 0, 0]], np.uint32))
    ),
    'NaN height field': shape(lambda: Shape.heightfield(np.full((4, 4), np.nan, np.float32))),
    'huge height field': shape(lambda: Shape.heightfield(np.full((4, 4), 1e38, np.float32))),
    'empty compound': shape(lambda: Shape.compound([])),
    'negative box': shape(lambda: Shape.box((-1.0, 1.0, 1.0))),
    # vehicles
    'vehicle chassis made static': vehicle_chassis_made_static,
    'vehicle double remove': vehicle_double_remove,
    'vehicle huge drag': vehicle({'drag': 1e38}),
    'vehicle huge downforce': vehicle({'downforce': 1e38}),
    'vehicle huge steer': vehicle({'max_steer': 1e38}),
    'vehicle NaN differential': vehicle(
        {'differentials': (Differential(0, 1, ratio=float('nan')),)}
    ),
    'vehicle huge differential': vehicle(
        {'differentials': (Differential(0, 1, ratio=1e38, torque_ratio=1e38),)}
    ),
    'vehicle differential past the wheels': vehicle({'differentials': (Differential(10**12, 1),)}),
    'tracks without inertia': vehicle(
        {
            'kind': VehicleKind.TRACKED,
            'tracks': (VehicleTrack(0, inertia=0.0), VehicleTrack(1, inertia=0.0)),
        }
    ),
    'track of NaN ratio': vehicle(
        {
            'kind': VehicleKind.TRACKED,
            'tracks': (VehicleTrack(0, ratio=float('nan')), VehicleTrack(1)),
        }
    ),
    'motorcycle of a car': vehicle({'kind': VehicleKind.MOTORCYCLE}),
    # ragdolls
    'bone ops': bone_ops,
    'tiny ragdoll': lambda w: w.add_ragdoll(
        RagdollSettings.humanoid(height=1e-6), position=(0.0, 1.0, 0.0)
    ),
    'huge ragdoll': lambda w: w.add_ragdoll(
        RagdollSettings.humanoid(height=1e30), position=(0.0, 1.0, 0.0)
    ),
    'light ragdoll': lambda w: w.add_ragdoll(
        RagdollSettings.humanoid(mass=1e-30), position=(0.0, 1.0, 0.0)
    ),
    'heavy ragdoll': lambda w: w.add_ragdoll(
        RagdollSettings.humanoid(mass=1e30), position=(0.0, 1.0, 0.0)
    ),
    'ragdoll of group -1': lambda w: w.add_ragdoll(
        dataclasses.replace(RagdollSettings.humanoid(), group=-1), position=(0.0, 1.0, 0.0)
    ),
    'ragdoll of filter 70000': lambda w: w.add_ragdoll(
        dataclasses.replace(RagdollSettings.humanoid(), filter=70000), position=(0.0, 1.0, 0.0)
    ),
    # soft bodies
    'ops on a soft body': ops_on_soft_body,
    'soft huge vertex radius': lambda w: w.add_soft_body(
        cloth(vertex_radius=1e30), position=(0.0, 2.0, 0.0)
    ),
    'soft huge max velocity': lambda w: w.add_soft_body(
        cloth(max_velocity=1e38), position=(0.0, 2.0, 0.0)
    ),
    'soft huge pressure': lambda w: w.add_soft_body(cloth(pressure=1e38), position=(0.0, 2.0, 0.0)),
    'soft huge damping': lambda w: w.add_soft_body(
        cloth(linear_damping=1e38), position=(0.0, 2.0, 0.0)
    ),
    'soft huge friction': lambda w: w.add_soft_body(cloth(friction=1e38), position=(0.0, 2.0, 0.0)),
    'soft huge gravity factor': lambda w: w.add_soft_body(
        cloth(gravity_factor=1e38), position=(0.0, 2.0, 0.0)
    ),
    'soft huge stretch': lambda w: w.add_soft_body(
        cloth(long_range_stretch=1e30), position=(0.0, 2.0, 0.0)
    ),
    'soft group -1': lambda w: w.add_soft_body(cloth(group=-1), position=(0.0, 2.0, 0.0)),
    'soft filter 70000': lambda w: w.add_soft_body(cloth(filter=70000), position=(0.0, 2.0, 0.0)),
    'soft material 70000': lambda w: w.add_soft_body(
        cloth(material=70000), position=(0.0, 2.0, 0.0)
    ),
    'soft 2**31 iterations': lambda w: w.add_soft_body(
        cloth(iterations=2**31), position=(0.0, 2.0, 0.0)
    ),
    'soft 1e12 iterations': lambda w: w.add_soft_body(
        cloth(iterations=10**12), position=(0.0, 2.0, 0.0)
    ),
    'soft degenerate edges': lambda w: w.add_soft_body(
        SoftBodySettings(
            positions=np.eye(4, 3), edges=np.array([[0, 0], [1, 1]]), edge_compliance=np.zeros(2)
        ),
        position=(0.0, 2.0, 0.0),
    ),
    'soft repeated faces': lambda w: w.add_soft_body(
        SoftBodySettings(positions=np.eye(4, 3), faces=np.array([[0, 1, 2]] * 100)),
        position=(0.0, 2.0, 0.0),
    ),
    'joint to a soft body': lambda w: w.add_joint(
        Joint.fixed(int(w.add_soft_body(cloth(), position=(0.0, 2.0, 0.0)).uid), sphere(w))
    ),
    # characters
    'character handles': character_handles,
    'huge character': lambda w: w.add_character(
        (0.0, 1.0, 0.0), CharacterCapabilities(radius=1e30, crouchHeight=3e30, standHeight=3e30)
    ),
    'light character': lambda w: w.add_character(
        (0.0, 1.0, 0.0), CharacterCapabilities(mass=1e-40)
    ),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_hostile_input_is_refused_or_harmless(make_world: MakeWorld, name: str) -> None:
    world = make_world(threads=4)
    world.add_bodies(floor(50.0))
    try:
        CASES[name](world)
        world.run_ticks(10)
    except REFUSED:
        return
    assert finite(world), name


@pytest.mark.parametrize(
    'settings',
    [
        {'tick_hz': 1e-300},
        {'threads': 100000},
        {'max_bodies': 2**32 - 1},
        {'temp_allocator_mb': 2**31},
        {'temp_allocator_mb': 0},
        {'max_body_pairs': 1, 'max_contact_constraints': 1},
    ],
    ids=str,
)
def test_extreme_world_settings_are_refused_or_harmless(settings: dict[str, Any]) -> None:
    try:
        world = World(WorldSettings(**settings))
    except REFUSED:
        return
    with world:
        world.add_bodies(floor(50.0))
        world.add_bodies(BodyBatch.spheres([(i * 0.4, 0.2, 0.0) for i in range(100)], radius=0.25))
        world.run_ticks(5)
        assert finite(world)


@pytest.mark.parametrize(
    'call',
    [
        pytest.param(lambda w, u: w.set_gravity((0.0, -1e300, 0.0)), id='gravity'),
        pytest.param(lambda w, u: w.add_forces([u], [(1e300, 0.0, 0.0)]), id='force'),
        pytest.param(lambda w, u: w.set_velocities([u], linear=[(1e300, 0.0, 0.0)]), id='velocity'),
        pytest.param(
            lambda w, u: w.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)], radius=1e30)),
            id='radius',
        ),
        pytest.param(lambda w, u: sphere(w, mass=1e-45), id='mass'),
        pytest.param(lambda w, u: w.set_wind([u], [(1e300, 0.0, 0.0)]), id='wind'),
    ],
)
def test_finite_doubles_past_a_float_raise(
    make_world: MakeWorld, call: Callable[[World, int], object]
) -> None:
    world = make_world(threads=2)
    uid = sphere(world)
    with pytest.raises(ValueError):
        call(world, uid)
    world.run_ticks(3)
    assert finite(world)


def test_a_body_that_overflows_stops_the_world_naming_it(make_world: MakeWorld) -> None:
    """A gravity factor that fits a float and overflows its velocity stops the world before NaN spreads."""
    world = make_world(threads=2)
    world.add_bodies(floor(50.0))
    others, _ = world.add_bodies(
        BodyBatch.spheres([(i * 1.0, 0.5, 0.0) for i in range(8)], radius=0.4)
    )
    uid = sphere(world, gravity_factor=1e38)
    with pytest.raises(WorldError, match='body %d' % uid):
        world.run_ticks(60)
    assert isinstance(world.error, RuntimeError)
    snapshot = latest(world)
    rows = [snapshot.rows[int(u)] for u in others]
    assert np.isfinite(snapshot.position[rows]).all()


@pytest.mark.parametrize(
    'uids',
    [[-1], [2**32], [1.5], np.array([1], np.int64) - 2**33, [[1]]],
    ids=['negative', '2**32', 'fraction', 'wrapped', '2-D'],
)
def test_uids_a_uint32_cannot_hold_raise(make_world: MakeWorld, uids: Any) -> None:
    world = make_world()
    with pytest.raises(ValueError, match='uids'):
        world.remove_bodies(uids)


@pytest.mark.parametrize('iterations', [0, 1001, 2**31, 10**12])
def test_soft_body_iterations_are_bounded(iterations: int) -> None:
    with pytest.raises(ValueError, match='iterations'):
        cloth(iterations=iterations)


@pytest.mark.parametrize(
    'changes', [{'group': -1}, {'group': 2**32}, {'filter': 70000}, {'material': 70000}], ids=str
)
def test_soft_and_ragdoll_collision_ids_fit_their_types(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        cloth(**changes)
    with pytest.raises(ValueError):
        dataclasses.replace(RagdollSettings.humanoid(), **changes)


@pytest.mark.parametrize(
    'values', [{'group': -1}, {'mask': 2**32}, {'filter': 70000}, {'material': -2}], ids=str
)
def test_body_collision_ids_fit_their_types(make_world: MakeWorld, values: dict[str, int]) -> None:
    world = make_world()
    with pytest.raises(ValueError):
        world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], **values))
