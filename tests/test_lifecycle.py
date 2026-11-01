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


def steady_thread_count(want: int | None = None, timeout: float = 5.0) -> int:
    """Return the thread count once it is ``want``, or once it holds for 0.1 s with ``want`` None.

    A joined thread can stay listed in ``/proc/self/task`` for a moment after
    ``pthread_join`` returns: the kernel wakes the joiner before it removes the
    task. A thread that was never joined stays listed, and the count returned
    after ``timeout`` seconds says so.
    """
    deadline = time.monotonic() + timeout
    count = thread_count()
    held = time.monotonic()
    while time.monotonic() < deadline:
        if want is not None and count == want:
            return count
        time.sleep(0.01)
        now = thread_count()
        if now != count:
            count, held = now, time.monotonic()
        elif want is None and time.monotonic() - held >= 0.1:
            return count
    return count


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


def test_a_future_at_tick_leaves_a_free_running_world_idle(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.run_ticks(1)
    calls: list[int] = []
    world.at_tick(50, lambda tick, time: calls.append(tick))
    world.start()
    time.sleep(0.3)
    assert world.stats()['idle']
    assert world.tick == 1
    # Lockstep brings the world to the tick, and the callable runs there.
    world.run_ticks(60)
    assert calls == [50]
    # One due now runs at once in a free-running world.
    world.at_tick(world.tick, lambda tick, time: calls.append(tick))
    world.start()
    deadline = time.monotonic() + 5.0
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [50, 61]


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
    # A world an earlier test left in a reference cycle joins its threads when the
    # collector frees it, so it is freed before the count is taken.
    gc.collect()
    before = steady_thread_count()
    world = World(WorldSettings(threads=4))
    world.add_bodies(BodyBatch.spheres(grid(20)))
    world.start()
    assert world.wait_tick(2, timeout=10.0)
    assert steady_thread_count(before + 4) == before + 4
    del world
    gc.collect()
    assert steady_thread_count(before) == before


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


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='needs os.fork')
def test_a_fork_waits_for_each_world_thread_to_finish_its_tick() -> None:
    # The before_step callback is half way through its tick, inside a lock,
    # when the main thread forks. A child that copied the process then would
    # inherit the lock held by a thread it does not have; the fork waits for
    # the tick to end, so every child finds it free.
    result = run_python(
        """
        import os, threading, time, warnings
        warnings.simplefilter('ignore', DeprecationWarning)
        from pyjoltphysics import BodyBatch, World, WorldSettings
        from pyjoltphysics.testing import grid
        held = threading.Lock()
        inside = threading.Event()
        def before(tick, now):
            with held:
                inside.set()
                time.sleep(0.004)
                inside.clear()
        world = World(WorldSettings(threads=2, tick_hz=1000.0))
        world.add_bodies(BodyBatch.spheres(grid(8, y0=1e5)))
        world.set_before_step(before)
        world.start()
        world.wait_tick(3, timeout=10)
        blocked = 0
        for _ in range(50):
            inside.wait(10)
            pid = os.fork()
            if pid == 0:
                os._exit(0 if held.acquire(blocking=False) else 1)
            _, status = os.waitpid(pid, 0)
            blocked += os.waitstatus_to_exitcode(status)
        world.stop()
        print(blocked, flush=True)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['0']


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='needs os.fork')
@pytest.mark.parametrize('where', ['before_step', 'contact_filter'])
def test_a_fork_from_a_worlds_own_callback_does_not_wait_for_it(where: str) -> None:
    result = run_python(
        f"""
        import os, warnings
        warnings.simplefilter('ignore', DeprecationWarning)
        from pyjoltphysics import BodyBatch, BodyFlag, World, WorldSettings
        from pyjoltphysics.testing import floor
        statuses = []
        def fork_once():
            if statuses:
                return
            pid = os.fork()
            if pid == 0:
                os._exit(7)
            statuses.append(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
        world = World(WorldSettings(threads=2))
        world.add_bodies(floor())
        world.add_bodies(BodyBatch.spheres([(0.0, 0.45, 0.0)], flags=BodyFlag.FILTER_CONTACTS))
        if {where!r} == 'before_step':
            world.set_before_step(lambda tick, now: fork_once())
        else:
            def accept(a, b, point, normal, approach):
                fork_once()
                return True
            world.set_contact_filter(accept, either=True)
        world.run_ticks(5)
        world.stop()
        print(statuses, flush=True)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[7]'


@pytest.mark.skipif(not LINUX, reason='reads thread stacks through glibc')
def test_callbacks_run_on_eight_mebibyte_stacks_whatever_the_default() -> None:
    # RLIMIT_STACK sets glibc's default thread stack, as musl's 128 KiB is set
    # for every thread there. The callbacks recurse through C code to a depth
    # that overflows 1 MiB of stack.
    source = textwrap.dedent(
        """
        import ctypes, threading
        from pyjoltphysics import BodyBatch, BodyFlag, World, WorldSettings
        from pyjoltphysics.testing import floor, grid
        libc = ctypes.CDLL(None)
        libc.pthread_self.restype = ctypes.c_void_p
        libc.pthread_getattr_np.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libc.pthread_attr_getstacksize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libc.pthread_attr_destroy.argtypes = [ctypes.c_void_p]
        def stack_size():
            attr = (ctypes.c_char * 128)()
            libc.pthread_getattr_np(libc.pthread_self(), attr)
            size = ctypes.c_size_t()
            libc.pthread_attr_getstacksize(attr, ctypes.byref(size))
            libc.pthread_attr_destroy(attr)
            return size.value
        def deep():
            nested = []
            for _ in range(20_000):
                nested = [nested]
            return len(repr(nested))
        sizes = {}
        def record(where):
            thread = threading.get_native_id()
            if (where, thread) not in sizes:
                sizes[where, thread] = (stack_size(), deep())
        def accept(a, b, point, normal, approach):
            record('filter')
            return True
        world = World(WorldSettings(threads=4))
        world.add_bodies(floor())
        world.add_bodies(BodyBatch.spheres(grid(64, spacing=1.2, y0=0.45), flags=BodyFlag.FILTER_CONTACTS))
        world.set_before_step(lambda tick, now: record('world'))
        world.set_contact_filter(accept, either=True)
        world.run_ticks(5)
        world.stop()
        print(sorted({(where, size >> 20, length) for (where, _), (size, length) in sizes.items()}), flush=True)
        """
    )

    def small_stack() -> None:
        import resource  # noqa: PLC0415 resource is POSIX only, and this test runs on Linux alone

        resource.setrlimit(
            resource.RLIMIT_STACK, (1 << 20, resource.getrlimit(resource.RLIMIT_STACK)[1])
        )

    result = subprocess.run(
        [sys.executable, '-c', source],
        capture_output=True,
        text=True,
        timeout=60,
        preexec_fn=small_stack,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[('filter', 8, 40002), ('world', 8, 40002)]"
