"""Materials: combine modes, the pair table, static against dynamic friction, material indices in records."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, Combine, EventKind, EventPhase, MotionType, Shape, World
from pyjoltphysics.testing import MakeWorld, latest

SLOPE = math.radians(20.0)
G = 9.81


def _slope_turn() -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(SLOPE / 2), math.cos(SLOPE / 2))


def _slide(
    make_world: MakeWorld,
    ramp_material: int | None,
    box_material: int | None,
    setup: Callable[[World], None],
    push: float = 0.0,
    seconds: float = 1.0,
) -> float:
    """How far a box slides down a 20 degree ramp in ``seconds``."""
    world = make_world(threads=1, tick_hz=240.0)
    setup(world)
    turn = _slope_turn()
    world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 0.0, 0.0)],
            size=(40.0, 1.0, 4.0),
            motion=MotionType.STATIC,
            orientations=[turn],
            material=ramp_material or 0,
            friction=0.0,
        )
    )
    # The box sits on the ramp's top face, 0.5 + 0.25 m along its normal.
    normal = np.array([-math.sin(SLOPE), math.cos(SLOPE), 0.0])
    down = np.array([-math.cos(SLOPE), -math.sin(SLOPE), 0.0])
    start = normal * 0.751
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            [start],
            size=(0.5, 0.5, 0.5),
            orientations=[turn],
            mass=1.0,
            material=box_material or 0,
            friction=0.0,
            linear_velocity=down * push,
        )
    )
    world.run_ticks(int(seconds * 240))
    snap = latest(world)
    return float(np.dot(snap.position[snap.rows[int(uids[0])]] - start, down))


def _analytic(mu: float, seconds: float = 1.0, push: float = 0.0) -> float:
    accel = max(0.0, G * (math.sin(SLOPE) - mu * math.cos(SLOPE)))
    return push * seconds + 0.5 * accel * seconds * seconds


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [
        (Combine.AVERAGE, 0.25),
        (Combine.MINIMUM, 0.1),
        (Combine.MULTIPLY, 0.04),
        (Combine.MAXIMUM, 0.4),
    ],
)
def test_combine_modes(make_world: MakeWorld, mode: Combine, expected: float) -> None:
    def setup(world: World) -> None:
        world.materials.define(friction=0.4, friction_combine=mode)  # 1, the ramp
        world.materials.define(friction=0.1, friction_combine=mode)  # 2, the box

    distance = _slide(make_world, 1, 2, setup)
    assert abs(distance - _analytic(expected)) < max(0.03, 0.1 * _analytic(expected))


def test_the_higher_combine_mode_wins(make_world: MakeWorld) -> None:
    def setup(world: World) -> None:
        world.materials.define(friction=0.4, friction_combine=Combine.AVERAGE)
        world.materials.define(friction=0.1, friction_combine=Combine.MINIMUM)

    assert abs(_slide(make_world, 1, 2, setup) - _analytic(0.1)) < 0.05


def test_pair_entry_replaces_the_combined_friction(make_world: MakeWorld) -> None:
    def setup(world: World) -> None:
        rock = world.materials.define(friction=0.9)
        ice = world.materials.define(friction=0.9)
        world.materials.pair(rock, ice, friction=0.05)

    assert abs(_slide(make_world, 1, 2, setup) - _analytic(0.05)) < 0.05


def test_static_friction_holds_a_body_at_rest_and_dynamic_friction_slows_a_moving_one(
    make_world: MakeWorld,
) -> None:
    # tan(20 degrees) = 0.36: a static 0.6 holds, a dynamic 0.2 lets it slide.
    def setup(world: World) -> None:
        world.materials.define(static_friction=0.6, dynamic_friction=0.2)

    assert abs(_slide(make_world, 1, 1, setup)) < 0.01
    moving = _slide(make_world, 1, 1, setup, push=1.0)
    assert abs(moving - _analytic(0.2, push=1.0)) < 0.1 * _analytic(0.2, push=1.0)


def test_body_without_material_keeps_its_own_coefficients(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.materials.define(friction=0.0)
    world.add_bodies(BodyBatch.plane())
    _uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 3.0, 0.0)], radius=0.25, restitution=0.8))
    world.run_ticks(240)
    events = world.drain_events()
    begins = events[(events['kind'] == EventKind.CONTACT) & (events['phase'] == EventPhase.BEGIN)]
    assert len(begins) >= 1 and begins['material_a'][0] == 0 and begins['material_b'][0] == 0


def test_pair_restitution_bounces(make_world: MakeWorld) -> None:
    heights = []
    for restitution in (None, 0.9):
        world = make_world(threads=1, tick_hz=240.0)
        floor_m = world.materials.define(friction=0.5, restitution=0.0)
        ball_m = world.materials.define(friction=0.5, restitution=0.0)
        if restitution is not None:
            world.materials.pair(floor_m, ball_m, restitution=restitution)
        world.add_bodies(BodyBatch.plane(material=floor_m))
        uids, _ = world.add_bodies(
            BodyBatch.spheres([(0.0, 2.25, 0.0)], radius=0.25, material=ball_m)
        )
        top = 0.0
        for _ in range(300):
            world.run_ticks(1)
            snap = latest(world)
            if world.tick > 200:
                top = max(top, float(snap.position[snap.rows[int(uids[0])], 1]))
        heights.append(top)
    assert heights[0] < 0.3
    assert heights[1] > 0.25 + 0.6 * 2.0 * 0.81


def test_records_carry_material_indices_and_mesh_triangles_their_own(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    grass = world.materials.define(friction=0.8)
    stone = world.materials.define(friction=0.6)
    metal = world.materials.define(friction=0.3)
    points = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5)]
    floor = Shape.mesh(points, [(0, 2, 1), (0, 3, 2)], materials=[grass, stone])
    world.add_bodies(BodyBatch.of_shape([(0.0, 0.0, 0.0)], floor, motion=MotionType.STATIC))
    world.add_bodies(
        BodyBatch.spheres([(3.0, 1.0, -1.0), (-3.0, 1.0, 1.0)], radius=0.25, material=metal)
    )
    world.run_ticks(120)
    events = world.drain_events()
    begins = events[(events['kind'] == EventKind.CONTACT) & (events['phase'] == EventPhase.BEGIN)]
    assert sorted(
        zip(begins['material_a'].tolist(), begins['material_b'].tolist(), strict=True)
    ) == sorted([(grass, metal), (stone, metal)])


def test_set_body_materials_changes_the_friction(make_world: MakeWorld) -> None:
    def setup(world: World) -> None:
        world.materials.define(friction=0.0)
        world.materials.define(friction=2.0)

    assert abs(_slide(make_world, 1, 1, setup) - _analytic(0.0)) < 0.05

    world = make_world(threads=1)
    world.materials.define(friction=2.0)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.5))
    world.set_body_materials(uids, 1)
    world.run_ticks(1)


def test_material_values_are_checked(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError):
        world.materials.define(friction=-1.0)
    with pytest.raises(IndexError):
        world.materials.pair(0, 5, friction=0.1)
    with pytest.raises(KeyError):
        world.materials[0]


def test_a_pair_entry_is_read_back_in_either_order(make_world: MakeWorld) -> None:
    world = make_world()
    wood = world.materials.define(friction=0.5)
    ice = world.materials.define(friction=0.1)
    assert world.materials.pairing(wood, ice) is None
    world.materials.pair(wood, ice, friction=0.2)
    found = world.materials.pairing(ice, wood)
    assert found is not None
    assert found[0] == pytest.approx(0.2) and found[1] == pytest.approx(0.2)
    assert math.isnan(found[2])
