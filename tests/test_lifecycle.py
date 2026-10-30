"""The world thread's life: idle, start and stop, collection, fork and interpreter exit."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time
import weakref

import pytest

from pyjoltphysics import BodyBatch, World, WorldSettings
from pyjoltphysics.testing import MakeWorld, floor, grid

LINUX = sys.platform.startswith('linux')


def thread_count() -> int:
    return len(os.listdir('/proc/self/task'))


def run_python(source: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match='tick_hz'):
        WorldSettings(tick_hz=0.0)
    with pytest.raises(ValueError, match='threads'):
        WorldSettings(threads=0)
    assert WorldSettings().dt == pytest.approx(1.0 / 120.0)


def test_an_idle_world_runs_no_ticks_and_uses_no_cpu(make_world: MakeWorld) -> None:
    world = make_world(threads=4)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.start()
    deadline = time.monotonic() + 10.0
    while not world.stats()['idle'] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert world.stats()['idle']
    ticks = world.tick
    cpu = time.process_time()
    time.sleep(0.5)
    assert world.tick == ticks
    assert time.process_time() - cpu < 0.05
    # A command wakes it.
    world.set_velocities(world.latest().uid[1:], linear=(0.0, 2.0, 0.0))  # type: ignore[union-attr]  # published above
    assert world.wait_tick(ticks + 1, timeout=5.0)


def test_start_and_stop_are_idempotent(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.stop()
    assert not world.stats()['running']
    world.start()
    world.start()
    assert world.stats()['running']
    world.stop()
    world.stop()
    assert not world.stats()['running']
    world.start()
    assert world.stats()['running']
    world.stop()
    assert not world.stats()['running']


def test_a_world_is_a_context_manager() -> None:
    with World(WorldSettings(threads=1)) as world:
        world.add_bodies(BodyBatch.spheres(grid(2)))
        world.run_ticks(3)
    assert not world.stats()['running']
    assert repr(world) == '<World tick=3 lockstep>'


def test_worlds_are_made_stopped_and_made_again() -> None:
    for _ in range(5):
        with World(WorldSettings(threads=4)) as world:
            world.add_bodies(BodyBatch.spheres(grid(50)))
            world.start()
            assert world.wait_tick(3, timeout=10.0)
        del world
        gc.collect()


@pytest.mark.skipif(not LINUX, reason='counts threads in /proc')
def test_a_dropped_world_joins_its_threads() -> None:
    before = thread_count()
    world = World(WorldSettings(threads=4))
    world.add_bodies(BodyBatch.spheres(grid(20)))
    world.start()
    assert world.wait_tick(2, timeout=10.0)
    assert thread_count() == before + 4
    del world
    gc.collect()
    assert thread_count() == before


def closed_over_world() -> weakref.ref[World]:
    """Return a weak reference to a world that its own callbacks hold."""
    world = World(WorldSettings(threads=1))
    world.set_after_step(lambda tick, now: world.latest())
    world.set_before_step(lambda tick, now: world.tick)
    world.at_tick(10**9, lambda tick, now: world)
    world.run_ticks(2)
    return weakref.ref(world)


def test_a_callback_closing_over_its_world_is_collected() -> None:
    ref = closed_over_world()
    assert ref() is not None
    gc.collect()
    assert ref() is None


def test_a_world_collected_on_its_own_thread_stops() -> None:
    result = run_python(
        """
        import gc
        from pyjoltphysics import World, WorldSettings
        world = World(WorldSettings(threads=1))
        world.set_after_step(lambda tick, now: gc.collect() if tick == 3 else None)
        world.at_tick(10**9, lambda tick, now: world)
        world.start()
        world.wait_tick(2, timeout=10)
        del world
        print('done', flush=True)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'done'


@pytest.mark.parametrize('ending', ['exit', 'raise', 'callbacks'])
def test_interpreter_exit_stops_running_worlds(ending: str) -> None:
    result = run_python(
        f"""
        import numpy as np, sys
        from pyjoltphysics import BodyBatch, World, WorldSettings
        from pyjoltphysics.testing import floor, grid
        world = World(WorldSettings(threads=4, tick_hz=240.0))
        world.add_bodies(floor())
        world.add_bodies(BodyBatch.boxes(grid(500, spacing=0.6, y0=2.0), size=(0.5, 0.5, 0.5)))
        if {ending!r} == 'callbacks':
            world.set_before_step(lambda tick, now: world.latest())
            world.set_after_step(lambda tick, now: sum(range(50)))
        world.start()
        world.wait_tick(10, timeout=10)
        if {ending!r} == 'raise':
            raise SystemExit(3)
        print('end', flush=True)
        """
    )
    assert result.returncode == (3 if ending == 'raise' else 0), result.stderr
    assert 'Fatal' not in result.stderr


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='needs os.fork')
def test_a_forked_child_cannot_use_its_parents_world() -> None:
    result = run_python(
        """
        import os, warnings
        warnings.simplefilter('ignore', DeprecationWarning)
        from pyjoltphysics import BodyBatch, World, WorldSettings
        from pyjoltphysics.testing import grid
        world = World(WorldSettings(threads=4))
        world.add_bodies(BodyBatch.spheres(grid(100, y0=1e5)))
        world.start()
        world.wait_tick(5, timeout=10)
        pid = os.fork()
        if pid == 0:
            failures = 0
            for call in (world.latest, lambda: world.run_ticks(1), world.start,
                         lambda: world.add_bodies(BodyBatch.spheres(grid(1)))):
                try:
                    call()
                except RuntimeError as error:
                    failures += 'fork' in str(error)
            child = World(WorldSettings(threads=2))
            child.add_bodies(BodyBatch.spheres(grid(3)))
            child.run_ticks(5)
            ok = failures == 4 and child.latest().tick == 5
            child.stop()
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        ticks = world.tick
        world.wait_tick(ticks + 6, timeout=10)
        print(os.waitstatus_to_exitcode(status), world.tick > ticks, flush=True)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['0', 'True']
