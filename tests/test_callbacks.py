"""Python callbacks on the world thread: order, commands, errors and the GIL."""

from __future__ import annotations

import logging
import sys
import threading
import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, EventKind, EventPhase, World, WorldError
from pyjoltphysics.testing import MakeWorld, floor, grid

GIL = getattr(sys, '_is_gil_enabled', lambda: True)()


def test_callbacks_run_in_order_on_the_world_thread(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes([(0.0, 0.7, 0.0)]))
    calls: list[tuple[str, int]] = []
    threads: set[int] = set()

    def before(tick: int, now: float) -> None:
        calls.append(('before', tick))
        threads.add(threading.get_ident())
        assert now == pytest.approx(tick / 120.0)

    def after(tick: int, now: float) -> None:
        calls.append(('after', tick))
        assert world.latest() is not None and world.latest().tick == tick  # type: ignore[union-attr]  # checked first

    def events(records: np.ndarray, tick: int, now: float) -> None:
        calls.append(('events', tick))
        assert (records['tick'] == tick).all()

    world.set_before_step(before)
    world.set_after_step(after)
    world.set_on_events(events)
    world.at_tick(2, lambda tick, now: calls.append(('at', tick)))
    world.run_ticks(3)
    steps = [c for c in calls if c[0] != 'events']
    assert steps == [
        ('before', 0),
        ('after', 1),
        ('before', 1),
        ('after', 2),
        ('at', 2),
        ('before', 2),
        ('after', 3),
    ]
    assert threading.get_ident() not in threads
    assert len(threads) == 1


def test_a_command_from_a_callback_applies_at_the_next_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)]))
    seqs: dict[int, int] = {}

    def push(tick: int, now: float) -> None:
        if tick == 5:
            seqs[tick] = world.set_velocities([ball], linear=(1.0, 0.0, 0.0))

    world.set_before_step(push)
    world.run_ticks(10)
    applied = dict(map(tuple, world.drain_applied().tolist()))
    assert applied[seqs[5]] == 6


def test_a_raising_callback_stops_the_world_and_fails_its_waiters(
    make_world: MakeWorld, caplog: pytest.LogCaptureFixture
) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.spheres(grid(3)))

    def explode(tick: int, now: float) -> None:
        if tick == 4:
            raise KeyError('boom')

    world.set_after_step(explode)
    with (
        caplog.at_level(logging.ERROR, logger='pyjoltphysics'),
        pytest.raises(WorldError) as caught,
    ):
        world.run_ticks(10)
    assert isinstance(caught.value.__cause__, KeyError)
    assert isinstance(world.error, KeyError)
    assert world.tick == 4
    assert [r.message for r in caplog.records] == ['a world callback raised; the world has stopped']
    with pytest.raises(WorldError):
        world.run_ticks(1)
    with pytest.raises(WorldError):
        world.wait_tick(100, timeout=1.0)
    with pytest.raises(WorldError):
        world.start()


def test_a_free_running_world_stops_on_a_callback_error(make_world: MakeWorld) -> None:
    world = make_world(threads=1)

    def explode(tick: int, now: float) -> None:
        raise ValueError(tick)

    world.set_before_step(explode)
    world.start()
    with pytest.raises(WorldError):
        world.wait_tick(1000, timeout=10.0)
    assert isinstance(world.error, ValueError)


def test_stop_from_a_callback_ends_after_that_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1)

    def halt(tick: int, now: float) -> None:
        if tick == 3:
            world.stop()

    world.set_after_step(halt)
    world.start()
    assert not world.wait_tick(1000, timeout=10.0)
    assert world.tick == 3
    world.stop()
    world.set_after_step(None)
    world.run_ticks(2)
    assert world.tick == 5


def test_run_ticks_from_a_callback_is_refused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    seen: list[BaseException] = []

    def nested(tick: int, now: float) -> None:
        try:
            world.run_ticks(1)
        except RuntimeError as error:
            seen.append(error)

    world.set_before_step(nested)
    world.run_ticks(1)
    assert len(seen) == 1 and 'world thread' in str(seen[0])


def test_stop_while_a_callback_waits_for_the_gil(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.set_before_step(lambda tick, now: None)
    world.start()
    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        pass  # holds the GIL on a GIL build while the world wants it
    world.stop()
    assert world.tick > 0


def test_a_world_with_no_callback_never_waits_for_the_gil() -> None:
    """Ticks keep their rate while this thread holds the GIL for 0.5 s."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1.0)  # a GIL holder keeps it for up to 1 s
    try:
        with World() as world:
            world.add_bodies(BodyBatch.spheres([(0.0, 1e6, 0.0)]))  # falls for the whole test
            world.start()
            assert world.wait_tick(2, timeout=10.0)
            start = world.tick
            deadline = time.perf_counter() + 0.5
            while time.perf_counter() < deadline:
                pass
            ticks = world.tick - start
    finally:
        sys.setswitchinterval(previous)
    assert ticks >= 0.8 * 0.5 * 120, ticks


@pytest.mark.skipif(not GIL, reason='a free-threaded build has no GIL to wait for')
def test_a_callback_waits_for_the_gil_on_a_gil_build() -> None:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    try:
        with World() as world:
            world.add_bodies(BodyBatch.spheres([(0.0, 1e6, 0.0)]))
            world.set_before_step(lambda tick, now: None)
            world.start()
            assert world.wait_tick(2, timeout=10.0)
            start = world.tick
            deadline = time.perf_counter() + 0.5
            while time.perf_counter() < deadline:
                pass
            ticks = world.tick - start
    finally:
        sys.setswitchinterval(previous)
    assert ticks < 0.5 * 0.5 * 120, ticks


def test_on_events_receives_the_ticks_records(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes([(0.0, 1.0, 0.0)]))
    seen: list[np.ndarray] = []
    world.set_on_events(lambda records, tick, now: seen.append(records.copy()))
    world.run_ticks(60)
    begins = [
        r
        for batch in seen
        for r in batch
        if r['kind'] == EventKind.CONTACT and r['phase'] == EventPhase.BEGIN
    ]
    assert len(begins) == 1
    world.set_on_events(None)
