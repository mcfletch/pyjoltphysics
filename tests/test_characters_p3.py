"""Character rows and records for the service: previous rows, eyes, stuck teleports, touches and knockback."""

from __future__ import annotations

import numpy as np

from pyjoltphysics import BodyBatch, CharacterCapabilities, EventKind, EventPhase, MotionType, Shape
from pyjoltphysics.testing import MakeWorld, floor, latest


def test_previous_row_is_the_centre_a_tick_before(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(20.0))
    hero = world.add_character((0.0, 0.9, 0.0), CharacterCapabilities(walkSpeed=3.0))
    world.run_ticks(10)
    hero.set_move((1.0, 0.0, 0.0))
    world.run_ticks(5)
    before = latest(world).characters.position.copy()
    world.run_ticks(1)
    rows = latest(world).characters
    assert np.allclose(rows.previous, before)
    assert rows.position[0, 0] > before[0, 0]


def test_eye_is_eye_height_above_the_feet_and_lower_crouched(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(20.0))
    hero = world.add_character(
        (0.0, 0.9, 0.0), CharacterCapabilities(eyeHeight=1.6, standHeight=1.8)
    )
    world.run_ticks(10)
    rows = latest(world).characters
    assert abs(float(rows.eye[0, 1]) - 1.6) < 0.01
    hero.set_crouch(True)
    world.run_ticks(2)
    rows = latest(world).characters
    assert abs(float(rows.eye[0, 1]) - 1.6 * 1.0 / 1.8) < 0.01


def test_teleport_into_a_wall_is_stuck(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(20.0))
    world.add_bodies(
        BodyBatch.boxes([(5.0, 1.0, 0.0)], size=(2.0, 2.0, 2.0), motion=MotionType.STATIC)
    )
    hero = world.add_character((0.0, 0.9, 0.0))
    world.run_ticks(2)
    assert not latest(world).characters.stuck[0]
    hero.teleport((5.0, 1.0, 0.0), search=0.0)
    world.run_ticks(1)
    assert latest(world).characters.stuck[0]
    hero.teleport((0.0, 0.9, 0.0))
    world.run_ticks(1)
    assert not latest(world).characters.stuck[0]


def test_characters_touching_each_other_make_records(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(20.0))
    a = world.add_character((0.0, 0.9, 0.0))
    b = world.add_character((2.0, 0.9, 0.0))
    world.run_ticks(5)
    world.drain_events()
    a.set_move((1.0, 0.0, 0.0))
    world.run_ticks(60)
    events = world.drain_events()
    touches = events[
        (events['kind'] == EventKind.CHARACTER) & (events['phase'] == EventPhase.BEGIN)
    ]
    assert (a.id, b.id) in set(zip(touches['a'].tolist(), touches['b'].tolist(), strict=True))


def test_body_striking_a_character_records_its_momentum(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(20.0))
    hero = world.add_character((0.0, 0.9, 0.0))
    world.run_ticks(5)
    world.drain_events()
    crate, _ = world.add_bodies(
        BodyBatch.boxes(
            [(-3.0, 0.9, 0.0)], size=(0.4, 0.4, 0.4), mass=10.0, linear_velocity=(8.0, 0.0, 0.0)
        )
    )
    world.run_ticks(60)
    events = world.drain_events()
    struck = events[
        (events['kind'] == EventKind.CHARACTER)
        & (events['phase'] == EventPhase.BEGIN)
        & (events['b'] == crate[0])
    ]
    assert len(struck) == 1 and struck['a'][0] == hero.id
    assert 30.0 < float(struck['impulse'][0]) < 90.0


def test_ground_material_under_a_character(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    grass = world.materials.define(friction=0.8)
    stone = world.materials.define(friction=0.6)
    points = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5), (15, 0, -5), (15, 0, 5)]
    tris = [(0, 2, 1), (0, 3, 2), (1, 5, 4), (1, 2, 5)]
    shape = Shape.mesh(points, tris, materials=[grass, grass, stone, stone])
    world.add_bodies(BodyBatch.of_shape([(0.0, 0.0, 0.0)], shape, motion=MotionType.STATIC))
    hero = world.add_character((0.0, 0.95, 0.0))
    world.run_ticks(60)
    assert int(latest(world).characters.ground_material[0]) == grass
    hero.teleport((10.0, 0.95, 0.0))
    world.run_ticks(60)
    assert int(latest(world).characters.ground_material[0]) == stone
