"""Collision filters, gravity zones, quadratic drag and kinematic tracks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    CharacterCapabilities,
    EventKind,
    GravityZone,
    MotionType,
    Shape,
    Track,
    World,
    ZoneField,
)
from pyjoltphysics.testing import MakeWorld, latest


def _height(world: World, uid: int) -> float:
    snap = latest(world)
    return float(snap.position[snap.rows[uid], 1])


# -- collision filters ------------------------------------------------------------


def test_filter_table_parts_two_filters(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_filter_table([[True, False], [False, True]])
    world.add_bodies(BodyBatch.plane(filter=0))
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0), (2.0, 1.0, 0.0)], radius=0.25, filter=[0, 1])
    )
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.25) < 0.01
    assert _height(world, int(uids[1])) < -1.0


def test_filter_table_changes_at_run_time(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.plane(filter=0))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, filter=1))
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.25) < 0.01
    world.set_filter_table([[True, False], [False, True]])
    world.run_ticks(60)
    assert _height(world, int(uids[0])) < 0.0


def test_filter_index_outside_the_table_meets_every_filter(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_filter_table([[True, False], [False, True]])
    world.add_bodies(BodyBatch.plane(filter=1))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, filter=7))
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.25) < 0.01


def test_set_body_filters_moves_a_body_to_another_filter(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_filter_table([[True, False], [False, True]])
    world.add_bodies(BodyBatch.plane(filter=0))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, filter=0))
    world.run_ticks(120)
    world.set_body_filters(uids, 1)
    world.run_ticks(60)
    assert _height(world, int(uids[0])) < 0.0


def test_filter_table_shape_is_checked(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError):
        world.set_filter_table([[True, False]])


# -- gravity zones -------------------------------------------------------------------


def test_directional_zone_replaces_gravity_inside_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], radius=0.1))
    world.set_gravity_zones(
        [
            GravityZone.box(
                (0.0, 0.0, 0.0), (4.0, 100.0, 4.0), gravity=2.0, direction=(0, 1, 0), replace=True
            )
        ]
    )
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.5 * 2.0 * 1.0) < 0.02
    assert abs(_height(world, int(uids[1])) + 0.5 * 9.81) < 0.05


def test_point_zone_pulls_towards_its_centre(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0), allow_sleeping=False)
    _uids, _ = world.add_bodies(BodyBatch.spheres([(5.0, 0.0, 0.0)], radius=0.1))
    world.set_gravity_zones(
        [
            GravityZone.sphere(
                (0.0, 0.0, 0.0), 20.0, field=ZoneField.POINT, center=(0, 0, 0), gravity=3.0
            )
        ]
    )
    world.run_ticks(120)
    snap = latest(world)
    assert abs(float(snap.position[0, 0]) - (5.0 - 1.5)) < 0.03


def test_zones_add_in_priority_and_stop_zeroes(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)], radius=0.1))
    world.set_gravity_zones(
        [
            # Higher priority applies last: the stop wins at the first sphere.
            GravityZone.sphere((0.0, 0.0, 0.0), 50.0, priority=5, stop=True),
            GravityZone.box(
                (0.0, 0.0, 0.0), (100.0, 100.0, 100.0), gravity=9.81, direction=(0, 1, 0)
            ),
            GravityZone.sphere((20.0, 0.0, 0.0), 5.0, priority=9, gravity=4.0, direction=(0, 1, 0)),
        ]
    )
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0]))) < 1e-3
    # The second sphere is outside the stop: world plus the box cancel, the small sphere lifts at 4.
    assert abs(_height(world, int(uids[1])) - 2.0) < 0.03


def test_shape_zone_and_gravity_factor(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    hull = world.add_shape(Shape.box((4.0, 100.0, 4.0)))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)], radius=0.1, gravity_factor=0.5))
    world.set_gravity_zones(
        [GravityZone.of_shape(hull, (0.0, 0.0, 0.0), gravity=0.0, replace=True)]
    )
    world.run_ticks(60)
    assert abs(_height(world, int(uids[0]))) < 1e-4


def test_character_falls_by_the_zone_it_is_in(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    hero = world.add_character((0.0, 10.0, 0.0), CharacterCapabilities())
    world.set_gravity_zones(
        [GravityZone.box((0.0, 10.0, 0.0), (10.0, 10.0, 10.0), gravity=2.0, replace=True)]
    )
    world.run_ticks(60)
    assert abs(hero.position[1] - (10.0 - 0.5 * 2.0 * 0.5**2)) < 0.02


def test_quadratic_drag_reaches_terminal_speed(make_world: MakeWorld) -> None:
    # Terminal speed sqrt(g / k).
    world = make_world(threads=1, allow_sleeping=False)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)], radius=0.1, quadratic_drag=0.1))
    world.run_ticks(1200)
    snap = latest(world)
    assert abs(-float(snap.linear_velocity[0, 1]) - math.sqrt(9.81 / 0.1)) < 0.05
    world.set_drag(uids, 0.0)
    world.run_ticks(120)
    assert -float(latest(world).linear_velocity[0, 1]) > math.sqrt(9.81 / 0.1) + 5.0


# -- kinematic tracks ---------------------------------------------------------------------


def test_track_poses_each_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1, tick_hz=60.0)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.0, 0.0)], size=(1, 0.2, 1), motion=MotionType.KINEMATIC)
    )
    uid = int(uids[0])
    times = [0.0, 1.0, 2.0]
    positions = [(0.0, 1.0, 0.0), (2.0, 1.0, 0.0), (2.0, 3.0, 0.0)]
    world.set_track(uids, Track(times, positions))
    worst = 0.0
    for tick in range(1, 150):
        world.run_ticks(1)
        t = min(tick / 60.0, 2.0)
        expected = np.array([2.0 * min(t, 1.0), 1.0 + 2.0 * max(t - 1.0, 0.0), 0.0])
        snap = latest(world)
        worst = max(worst, float(np.abs(snap.position[snap.rows[uid]] - expected).max()))
    assert worst < 1e-5


def test_track_loops_and_turns(make_world: MakeWorld) -> None:
    world = make_world(threads=1, tick_hz=60.0)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.0, 0.0)], size=(1, 0.2, 1), motion=MotionType.KINEMATIC)
    )
    half = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    world.set_track(
        uids, Track([0.0, 1.0], [(0, 1, 0), (1, 1, 0)], [(0, 0, 0, 1), half], loop=True)
    )
    world.run_ticks(90)
    snap = latest(world)
    assert abs(float(snap.position[0, 0]) - 0.5) < 1e-5
    angle = 2.0 * math.atan2(float(snap.orientation[0, 1]), float(snap.orientation[0, 3]))
    assert abs(angle - math.pi / 4) < 1e-4


def test_track_carries_a_rider(make_world: MakeWorld) -> None:
    world = make_world(threads=1, tick_hz=120.0)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.0, 0.0)], size=(4, 0.2, 4), motion=MotionType.KINEMATIC)
    )
    rider, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.35, 0.0)], size=(0.5, 0.5, 0.5), friction=1.0)
    )
    world.run_ticks(60)
    world.set_track(uids, Track([0.0, 2.0], [(0, 0, 0), (0, 2, 0)]))
    world.run_ticks(240)
    assert abs(_height(world, int(rider[0])) - 2.35) < 0.02


def test_drive_follows_a_pose_function(make_world: MakeWorld) -> None:
    world = make_world(threads=1, tick_hz=60.0)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(1.0, 1.0, 0.0)], size=(0.2, 0.2, 0.2), motion=MotionType.KINEMATIC)
    )
    uid = int(uids[0])
    world.drive(uid, lambda t: (math.cos(t), 1.0, math.sin(t)))
    for tick in range(1, 61):
        world.run_ticks(1)
        snap = latest(world)
        t = tick / 60.0
        assert np.allclose(
            snap.position[snap.rows[uid]], (math.cos(t), 1.0, math.sin(t)), atol=1e-5
        )
    world.drive(uid, None)
    world.run_ticks(5)
    assert world.stats()['rejected'] == 0


def test_track_on_a_dynamic_body_is_rejected(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.2))
    world.set_track(uids, Track([0.0], [(0, 0, 0)]))
    world.run_ticks(1)
    assert world.stats()['rejected'] == 1
    with pytest.raises(ValueError):
        world.set_track(uids, Track([0.0, 0.0], [(0, 0, 0), (1, 0, 0)]))


def test_zone_events_are_not_contacts(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_gravity_zones([GravityZone.sphere((0.0, 0.0, 0.0), 2.0)])
    world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)], radius=0.1))
    world.run_ticks(10)
    events = world.drain_events()
    assert not np.any(events['kind'] == EventKind.CONTACT)
