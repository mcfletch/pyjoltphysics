"""Lockstep and replay: a recorded free-running run replays bitwise in lockstep."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, BodyFlag, World, WorldSettings
from pyjoltphysics.testing import MakeWorld, floor, grid, latest, pose_digest

TICKS = 1000
BODIES = 500


def scene(world: World) -> np.ndarray:
    world.add_bodies(floor())
    uids, _ = world.add_bodies(
        BodyBatch.boxes(grid(BODIES, spacing=0.7, y0=1.0), size=(0.5, 0.5, 0.5))
    )
    return uids


def kicks(seed: int, uids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(40):
        chosen = rng.choice(uids, size=25, replace=False)
        out.append((chosen, rng.uniform(-2.0, 6.0, (25, 3))))
    return out


def run_lockstep(threads: int, plan: list[tuple[int, np.ndarray, np.ndarray]]) -> str:
    """Return the digest of every tick's poses, the kicks applied at the recorded ticks."""
    with World(WorldSettings(threads=threads, tick_hz=960.0)) as world:
        scene(world)
        for tick, chosen, impulses in plan:
            world.add_impulses(chosen, impulses, at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(TICKS):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            pose_digest(snapshot, digest)
        return digest.hexdigest()


def record_free_running(seed: int) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], str]:
    """Run free at 960 Hz with kicks at wall-clock moments; return (tick, batch) and the digest."""
    with World(WorldSettings(threads=8, tick_hz=960.0)) as world:
        uids = scene(world)
        batches = kicks(seed, uids)
        digest = hashlib.sha256()
        seqs = {}

        def record(tick: int, now: float) -> None:
            snapshot = world.latest()
            if tick <= TICKS and snapshot is not None:
                pose_digest(snapshot, digest)

        world.set_after_step(record)
        world.start()
        rng = np.random.default_rng(seed + 1)
        moments = np.sort(rng.uniform(0.0, 0.8, len(batches)))
        start = time.perf_counter()
        for moment, (chosen, impulses) in zip(moments, batches, strict=True):
            while time.perf_counter() - start < moment:
                time.sleep(0.0005)
            seqs[world.add_impulses(chosen, impulses)] = (chosen, impulses)
        assert world.wait_tick(TICKS, timeout=60.0)
        world.stop()
        applied = {int(seq): int(tick) for seq, tick in world.drain_applied().tolist()}
    plan = [(applied[seq], chosen, impulses) for seq, (chosen, impulses) in sorted(seqs.items())]
    assert len({tick for tick, _, _ in plan}) > 5
    return plan, digest.hexdigest()


def test_lockstep_runs_exactly_the_ticks_asked_for(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(BodyBatch.spheres(grid(10)))
    world.run_ticks(7)
    assert world.tick == 7
    assert world.lockstep
    time.sleep(0.05)
    assert world.tick == 7
    world.run_ticks(3)
    assert world.tick == 10
    world.run_ticks(0)
    assert world.tick == 10


def test_start_after_lockstep_free_runs(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.spheres([(0.0, 1e6, 0.0)]))
    world.run_ticks(5)
    world.start()
    assert not world.lockstep
    assert world.wait_tick(30, timeout=10.0)
    world.run_ticks(2)
    assert world.lockstep


def test_at_tick_batches_apply_at_their_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)]))
    world.set_velocities([ball], linear=(120.0, 0.0, 0.0), at_tick=10)
    world.run_ticks(20)
    snapshot = world.latest()
    assert snapshot is not None
    # Moving from the step of tick 10 on: 10 steps of 1 m.
    assert snapshot.position[0, 0] == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize('repeat', range(16))
def test_a_free_running_record_replays_bitwise_in_lockstep(repeat: int) -> None:
    plan, free = record_free_running(seed=repeat)
    assert run_lockstep(1, plan) == free
    assert run_lockstep(8, plan) == free


def settled_heap(world: World, report: bool = True) -> np.ndarray:
    """A heap of boxes on a floor, run in lockstep until every box is asleep; the boxes' uids."""
    flags = BodyFlag.REPORT if report else BodyFlag(0)
    world.add_bodies(floor(flags=flags))
    uids, _ = world.add_bodies(
        BodyBatch.boxes(grid(27, spacing=0.6, y0=0.3), size=(0.5, 0.5, 0.5), flags=flags)
    )
    for _ in range(4000):
        world.run_ticks(10)
        if not latest(world).awake.any():
            return uids
    raise AssertionError('the heap never settled')


def recorded(world: World, ticks: int) -> tuple[str, list[tuple[int, ...]]]:
    """The digest of each tick's poses and every event, over ``ticks`` ticks run one at a time."""
    digest = hashlib.sha256()
    events: list[tuple[int, ...]] = []
    for _ in range(ticks):
        world.run_ticks(1)
        pose_digest(latest(world), digest)
        for record in world.drain_events().tolist():
            events.append(tuple(record[:6]))
    return digest.hexdigest(), events


def test_a_settled_world_counts_its_ticks_without_stepping(make_world: MakeWorld) -> None:
    world = make_world(threads=2, tick_hz=120.0)
    settled_heap(world)
    # The tick after the heap falls asleep reports its contacts' removal; the two
    # after it step and report nothing, which makes the world quiet.
    world.run_ticks(3)
    start = world.tick
    world.reset_stats()
    world.run_ticks(100)
    snapshot = latest(world)
    assert world.tick == snapshot.tick == start + 100
    assert snapshot.time == pytest.approx((start + 100) / 120.0)
    assert len(snapshot.changed) == 0
    assert world.stats()['quiet_ticks'] == 100


def test_quiet_ticks_match_stepped_ticks_bitwise(make_world: MakeWorld) -> None:
    """A settled world counted through ticks then woken plays out as one stepped through them."""
    results = []
    for stepped in (True, False):
        world = make_world(threads=2)
        uids = settled_heap(world)
        world.drain_events()
        if stepped:
            # A step callback keeps every tick a real step.
            world.set_before_step(lambda tick, now: None)
        world.reset_stats()
        for _ in range(30):
            world.run_ticks(10)
        quiet = world.stats()['quiet_ticks']
        world.set_before_step(None)
        assert (quiet == 0) if stepped else (quiet >= 280)
        world.add_impulses(uids[:1], [(0.0, 4.0, 0.5)])
        world.add_impulses(uids[13:14], [(1.0, 0.0, 0.0)], at_tick=world.tick + 40)
        settled_at = world.tick - 300
        results.append((settled_at, *recorded(world, 400), world.drain_applied().tolist()))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]
    assert results[0][2] == results[1][2]
    assert len(results[0][2]) > 0
    applied = [[tick for _seq, tick in log] for *_, log in results]
    assert applied[0] == applied[1]


def test_a_batch_due_inside_a_run_wakes_a_quiet_world_at_its_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids = settled_heap(world, report=False)
    world.run_ticks(10)
    due = world.tick + 25
    world.add_impulses(uids[:1], [(0.0, 3.0, 0.0)], at_tick=due)
    before = latest(world).position[latest(world).rows_of(uids[:1])][0].copy()
    world.run_ticks(24)
    assert np.array_equal(latest(world).position[latest(world).rows_of(uids[:1])][0], before)
    world.run_ticks(2)
    assert latest(world).position[latest(world).rows_of(uids[:1])][0][1] > before[1]
    assert [tick for _seq, tick in world.drain_applied().tolist()][-1] == due


def test_a_callable_due_inside_a_run_runs_at_its_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    settled_heap(world, report=False)
    calls = []
    world.at_tick(world.tick + 50, lambda tick, now: calls.append(tick))
    world.reset_stats()
    world.run_ticks(60)
    assert calls == [world.tick - 10]
    assert world.stats()['quiet_ticks'] <= 50
