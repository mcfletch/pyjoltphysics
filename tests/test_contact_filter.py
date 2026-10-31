"""The Python contact filter: asked on Jolt's job threads, once per pair, for pairs that opted in."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, BodyFlag, EventKind, MotionType, World, WorldError
from pyjoltphysics.testing import MakeWorld, floor, grid, latest


def _height(world: World, uid: int) -> float:
    snap = latest(world)
    return float(snap.position[snap.rows[uid], 1])


def test_refused_pair_passes_through_and_is_asked_once(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    asked: list[tuple[int, int]] = []

    def verdict(a: int, b: int, point: object, normal: object, approach: float) -> bool:
        asked.append((a, b))
        return False

    world.set_contact_filter(verdict)
    (ground,), _ = world.add_bodies(floor(5.0, flags=BodyFlag.FILTER_CONTACTS))
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS)
    )
    world.run_ticks(60)
    assert _height(world, int(uids[0])) < 0.0
    assert asked == [(int(ground), int(uids[0]))]
    assert not np.any(world.drain_events()['kind'] == EventKind.CONTACT)


def test_solved_pair_collides(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    seen: list[float] = []

    def note(a: int, b: int, point: object, normal: object, approach: float) -> None:
        seen.append(approach)

    world.set_contact_filter(note)
    world.add_bodies(floor(5.0, flags=BodyFlag.FILTER_CONTACTS))
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS)
    )
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.25) < 0.01
    assert len(seen) == 1 and seen[0] > 2.0


def test_only_pairs_that_both_opted_in_are_asked(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_contact_filter(lambda *pair: False)
    world.add_bodies(floor(5.0))
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS)
    )
    world.run_ticks(120)
    assert abs(_height(world, int(uids[0])) - 0.25) < 0.01


def test_refusal_ends_when_the_pair_parts(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    asked: list[tuple[int, int]] = []

    def refuse(a: int, b: int, *rest: object) -> bool:
        asked.append((a, b))
        return False

    world.set_contact_filter(refuse)
    _a, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 0.0, 0.0)], radius=0.5, flags=BodyFlag.FILTER_CONTACTS, motion=0)
    )
    b, _ = world.add_bodies(
        BodyBatch.spheres(
            [(-3.0, 0.0, 0.0)],
            radius=0.25,
            flags=BodyFlag.FILTER_CONTACTS,
            linear_velocity=(5.0, 0.0, 0.0),
        )
    )
    world.run_ticks(240)  # through the first, out the far side
    world.set_velocities(b, linear=[(-5.0, 0.0, 0.0)])
    world.run_ticks(240)  # and back through it
    assert len(asked) == 2


def test_a_raising_filter_stops_the_world(make_world: MakeWorld) -> None:
    world = make_world(threads=4)

    def broken(*pair: object) -> bool:
        raise ValueError('no')

    world.set_contact_filter(broken)
    world.add_bodies(floor(5.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS)
    )
    with pytest.raises(WorldError):
        world.run_ticks(120)
    assert isinstance(world.error, ValueError)


def test_a_filter_cannot_query_the_world(make_world: MakeWorld) -> None:
    world = make_world(threads=2)

    def asks(*pair: object) -> bool:
        return world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0)) is None

    world.set_contact_filter(asks)
    world.add_bodies(floor(5.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS)
    )
    with pytest.raises(WorldError):
        world.run_ticks(120)
    assert isinstance(world.error, RuntimeError)


def test_many_pairs_on_many_job_threads_beside_a_busy_python_thread(make_world: MakeWorld) -> None:
    world = make_world(threads=8)
    counted: list[int] = []

    def alternate(a: int, b: int, *rest: object) -> bool:
        counted.append(a)
        return (a + b) % 2 == 0

    world.set_contact_filter(alternate)
    world.add_bodies(floor(20.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.boxes(grid(400, 0.6, 1.0), size=(0.5, 0.5, 0.5), flags=BodyFlag.FILTER_CONTACTS)
    )
    stop = threading.Event()

    def spin() -> None:
        x = 0
        while not stop.is_set():
            x += 1

    busy = threading.Thread(target=spin)
    busy.start()
    try:
        world.start()
        assert world.wait_tick(240, timeout=60.0)
        world.stop()
    finally:
        stop.set()
        busy.join()
    assert len(counted) > 100
    assert world.error is None


def test_free_running_world_with_a_filter_stops_and_restarts(make_world: MakeWorld) -> None:
    world = make_world(threads=4)
    world.set_contact_filter(lambda *pair: True)
    world.add_bodies(floor(10.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.spheres(grid(50, 0.6, 1.0), radius=0.2, flags=BodyFlag.FILTER_CONTACTS)
    )
    for _ in range(3):
        world.start()
        time.sleep(0.05)
        world.stop()
    world.run_ticks(10)
    assert world.error is None


def test_either_asks_where_one_body_carries_the_flag(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, gravity=(0.0, 0.0, 0.0))
    asked: list[tuple[int, int]] = []

    def verdict(a: int, b: int, point: object, normal: object, approach: float) -> bool:
        asked.append((a, b))
        return False

    world.set_contact_filter(verdict, either=True)
    (wall,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.0, 0.0)], size=(0.2, 4.0, 4.0),
                                                  motion=MotionType.STATIC,
                                                  flags=BodyFlag.FILTER_CONTACTS))
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(-2.0, 0.0, 0.0)], radius=0.25,
                                                    linear_velocity=(5.0, 0.0, 0.0)))
    world.run_ticks(60)
    assert asked == [(wall, ball)] or asked == [(min(wall, ball), max(wall, ball))]
    snap = latest(world)
    assert snap.position[snap.rows[ball], 0] > 1.5


def test_recheck_asks_a_refused_pair_again(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    allowed = threading.Event()
    asked: list[int] = []

    def verdict(a: int, b: int, point: object, normal: object, approach: float) -> bool:
        asked.append(a)
        return allowed.is_set()

    world.set_contact_filter(verdict, recheck=True)
    world.add_bodies(floor(10.0, flags=BodyFlag.FILTER_CONTACTS))
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.5, 0.0)], radius=0.5,
                                                    flags=BodyFlag.FILTER_CONTACTS))
    world.run_ticks(5)
    assert len(asked) >= 4
    allowed.set()
    world.run_ticks(120)
    snap = latest(world)
    assert snap.position[snap.rows[ball], 1] > -0.5
