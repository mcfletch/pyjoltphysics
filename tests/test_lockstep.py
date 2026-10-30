"""Lockstep and replay: a recorded free-running run replays bitwise in lockstep."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, World, WorldSettings
from pyjoltphysics.testing import MakeWorld, floor, grid, pose_digest

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
