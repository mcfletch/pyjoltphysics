"""Calls into a world from its own callbacks and from other threads: each returns or raises, none hangs."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable

import pytest
from test_lifecycle import run_python

from pyjoltphysics import BodyBatch, BodyFlag, World, WorldSettings
from pyjoltphysics.testing import MakeWorld, floor, sanitizer_slowdown


def test_a_dropped_world_is_freed_at_once() -> None:
    """No reference cycle holds a World: it goes on the dropping thread, not in a later collection."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        world = World()
        world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)]))
        world.run_ticks(2)
        assert world.materials is not None
        ref = weakref.ref(world)
        del world
        assert ref() is None
    finally:
        if enabled:
            gc.enable()


def test_a_material_table_outliving_its_world_raises() -> None:
    world = World()
    table = world.materials
    del world
    with pytest.raises(RuntimeError, match='gone'):
        table.define(friction=0.5)


def test_a_world_collected_on_its_own_thread_while_a_callback_raises() -> None:
    """The collector frees worlds from their own callbacks; the callbacks' errors are logged, not kept."""
    result = run_python(
        """
        import gc, time
        from pyjoltphysics import World, WorldSettings, BodyBatch, BodyFlag
        from pyjoltphysics.testing import floor
        gc.set_threshold(1, 1, 1)
        for i in range(20):
            w = World(WorldSettings(threads=2))
            w.add_bodies(floor(50.0, flags=BodyFlag.REPORT))
            w.add_bodies(BodyBatch.spheres([(j * 0.6, 0.2, 0) for j in range(16)], radius=0.25,
                                           flags=BodyFlag.REPORT))
            holder = {'w': w}
            def churn(t, tm, holder=holder):
                [object() for _ in range(100)]; holder.get('x'); gc.collect()
            def ev(arr, tick, tm, holder=holder):
                holder['w']
            holder['cb'] = ev
            w.set_after_step(churn); w.set_on_events(ev)
            w.start(); time.sleep(0.01)
            del w, holder
            time.sleep(0.02)
        gc.collect(); time.sleep(0.2)
        print('done')
        """,
        timeout=120 * sanitizer_slowdown(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith('done')
    # A callable the collector has cleared loses its defaults; none is called.
    assert 'missing 1 required positional argument' not in result.stderr


@pytest.mark.parametrize('drops', [1, 3])
def test_a_world_dropped_while_its_callback_sleeps_then_raises(drops: int) -> None:
    """World.__del__ joins while the callback runs; the process exits after the last."""
    result = run_python(
        f"""
        import time
        from pyjoltphysics import World, WorldSettings
        from pyjoltphysics.testing import floor
        for i in range({drops}):
            w = World(WorldSettings(threads=1))
            w.add_bodies(floor(5.0))
            def late(tick, t):
                time.sleep(0.2)
                raise ZeroDivisionError('raised after the world was dropped')
            w.set_after_step(late)
            w.start()
            time.sleep(0.05)
            del w, late
        print('survived')
        """,
        timeout=10 * sanitizer_slowdown(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert 'survived' in result.stdout


def test_interpreter_exit_while_a_callback_raises() -> None:
    result = run_python(
        """
        import time
        from pyjoltphysics import World, WorldSettings
        w = World(WorldSettings(threads=1))
        def late(tick, t):
            time.sleep(0.2)
            raise ZeroDivisionError('late')
        w.set_after_step(late)
        w.start()
        time.sleep(0.05)
        print('script ends')
        """,
        timeout=10 * sanitizer_slowdown(),
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_stop_racing_run_ticks_never_deadlocks() -> None:
    """20000 stops from one thread against run_ticks on another, under a bound on the whole."""
    result = run_python(
        """
        import threading
        from pyjoltphysics import World, WorldSettings, BodyBatch
        from pyjoltphysics.testing import floor
        w = World(WorldSettings(threads=2))
        w.add_bodies(floor(5.0))
        w.add_bodies(BodyBatch.spheres([(0, 1, 0)], radius=0.25))
        def stopper():
            for _ in range(20000):
                w.stop()
        t = threading.Thread(target=stopper)
        t.start()
        while t.is_alive():
            try:
                w.run_ticks(1)
            except RuntimeError:
                pass
        t.join()
        w.run_ticks(3)
        print('returned', w.tick)
        """,
        timeout=120 * sanitizer_slowdown(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert 'returned' in result.stdout


def test_start_from_a_callback_ends_run_ticks(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(5.0))
    world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25))
    world.set_after_step(lambda tick, now: world.start() if tick == 2 else None)
    with pytest.raises(RuntimeError, match='free-running'):
        world.run_ticks(5)
    assert not world.lockstep


def test_start_from_another_thread_ends_run_ticks(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(5.0))
    world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25))
    world.set_after_step(lambda tick, now: time.sleep(0.001))
    timer = threading.Timer(0.2, world.start)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match='free-running'):
            world.run_ticks(10**6)
    finally:
        timer.join()


def test_save_state_from_a_step_callback_reads_the_tick(make_world: MakeWorld) -> None:
    world = make_world()
    world.add_bodies(floor(5.0))
    world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], radius=0.25))
    saved: list[bytes] = []
    world.set_after_step(lambda tick, now: saved.append(world.save_state()))
    world.run_ticks(3)
    assert len(saved) == 3
    world.set_after_step(None)
    world.restore_state(saved[0])
    world.run_ticks(1)
    assert world.error is None


def blocking_calls(world: World) -> dict[str, Callable[[], object]]:
    return {
        'save_state': world.save_state,
        'flush': world.flush,
        'wait_tick': lambda: world.wait_tick(world.tick + 1),
        'run_ticks': lambda: world.run_ticks(1),
    }


@pytest.mark.parametrize('name', ['save_state', 'flush', 'wait_tick', 'run_ticks'])
def test_blocking_calls_from_the_contact_filter_raise(make_world: MakeWorld, name: str) -> None:
    """The filter runs on job threads inside the step; a call that waits for the step raises at once."""
    world = make_world(threads=8)
    world.add_bodies(floor(50.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.spheres(
            [(x * 0.6, 0.2, z * 0.6) for x in range(-8, 8) for z in range(-8, 8)],
            radius=0.25,
            flags=BodyFlag.FILTER_CONTACTS,
        )
    )
    call = blocking_calls(world)[name]
    raised: list[str] = []

    def contact_filter(*_args: object) -> bool:
        try:
            call()
        except RuntimeError as error:
            raised.append(str(error))
        return True

    world.set_contact_filter(contact_filter)
    started = time.monotonic()
    world.run_ticks(3)
    assert time.monotonic() - started < 20 * sanitizer_slowdown()
    assert raised and all('contact filter' in message for message in raised)
    assert world.error is None


@pytest.mark.parametrize('name', ['wait_tick', 'run_ticks'])
def test_blocking_calls_from_a_step_callback_raise(make_world: MakeWorld, name: str) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(5.0))
    call = blocking_calls(world)[name]
    raised: list[str] = []

    def after(tick: int, now: float) -> None:
        try:
            call()
        except RuntimeError as error:
            raised.append(str(error))

    world.set_after_step(after)
    world.run_ticks(2)
    assert len(raised) == 2 and 'world thread' in raised[0]


def test_stop_and_start_from_the_contact_filter_return(make_world: MakeWorld) -> None:
    """From the filter, stop asks the world to stop after the tick, and start leaves that stop standing."""
    world = make_world(threads=8)
    world.add_bodies(floor(50.0, flags=BodyFlag.FILTER_CONTACTS))
    world.add_bodies(
        BodyBatch.spheres(
            [(x * 0.6, 0.2, 0.0) for x in range(16)], radius=0.25, flags=BodyFlag.FILTER_CONTACTS
        )
    )

    def contact_filter(*_args: object) -> bool:
        world.stop()
        world.start()
        return True

    world.set_contact_filter(contact_filter)
    with pytest.raises(RuntimeError):
        world.run_ticks(50)
    assert world.tick < 50
    world.set_contact_filter(None)
    world.run_ticks(2)
    assert world.error is None


def test_settings_of_a_world_thread_are_bounded() -> None:
    with pytest.raises(ValueError, match='tick_hz'):
        WorldSettings(tick_hz=1e-300)
    with pytest.raises(ValueError, match='tick_hz'):
        WorldSettings(tick_hz=1e6)
    with pytest.raises(ValueError, match='temp_allocator_mb'):
        WorldSettings(temp_allocator_mb=4096)
    with pytest.raises(ValueError, match='gravity'):
        WorldSettings(gravity=(0.0, -1e300, 0.0))
