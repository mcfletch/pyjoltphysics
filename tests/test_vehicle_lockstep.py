"""Vehicles replay bitwise: a free-running run with inputs at wall-clock moments, in lockstep."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
from test_vehicles import car_settings

from pyjoltphysics import BodyBatch, World, WorldSettings
from pyjoltphysics.testing import floor, vehicle_digest

TICKS = 1000
COUNT = 10

Input = tuple[float, float, float, float]


def scene(world: World) -> list[int]:
    """A floor, crates and ten cars among them; return the vehicle ids."""
    world.add_bodies(floor(500.0))
    world.add_bodies(
        BodyBatch.boxes(
            [(3.0 * i - 15.0, 0.5, -25.0) for i in range(10)], size=(1.0, 1.0, 1.0), mass=60.0
        )
    )
    ids = []
    for i in range(COUNT):
        uids, _ = world.add_bodies(
            BodyBatch.boxes(
                [(4.0 * i - 18.0, 0.62, 0.0)],
                size=(1.85, 0.62, 4.2),
                mass=1180.0,
                angular_damping=0.4,
            )
        )
        ids.append(world.add_vehicle(int(uids[0]), car_settings()).id)
    return ids


def inputs(seed: int, ids: list[int]) -> list[tuple[int, Input]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, Input]] = []
    for _ in range(60):
        throttle, steer, brake, handbrake = (float(v) for v in rng.uniform(-1.0, 1.0, 4))
        out.append(
            (int(rng.choice(ids)), (throttle, steer, abs(brake) * 0.5, abs(handbrake) * 0.3))
        )
    return out


def record_free_running(seed: int) -> tuple[list[tuple[int, int, Input]], str]:
    with World(WorldSettings(threads=8, tick_hz=480.0)) as world:
        ids = scene(world)
        plan = inputs(seed, ids)
        digest = hashlib.sha256()

        def record(tick: int, now: float) -> None:
            snapshot = world.latest()
            if tick <= TICKS and snapshot is not None:
                vehicle_digest(snapshot, digest)

        world.set_after_step(record)
        world.start()
        rng = np.random.default_rng(seed + 1)
        moments = np.sort(rng.uniform(0.0, 1.5, len(plan)))
        seqs = {}
        start = time.perf_counter()
        for moment, (vid, values) in zip(moments, plan, strict=True):
            while time.perf_counter() - start < moment:
                time.sleep(0.0005)
            seqs[world.vehicle_inputs([vid], [values])] = (vid, values)
        assert world.wait_tick(TICKS, timeout=60.0)
        world.stop()
        applied = {int(seq): int(tick) for seq, tick in world.drain_applied().tolist()}
    replay = [(applied[seq], vid, values) for seq, (vid, values) in sorted(seqs.items())]
    assert len({tick for tick, *_ in replay}) > 5
    return replay, digest.hexdigest()


def run_lockstep(threads: int, plan: list[tuple[int, int, Input]]) -> str:
    with World(WorldSettings(threads=threads, tick_hz=480.0)) as world:
        scene(world)
        for tick, vid, values in plan:
            world.vehicle_inputs([vid], [values], at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(TICKS):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            vehicle_digest(snapshot, digest)
        return digest.hexdigest()


@pytest.mark.parametrize('repeat', range(4))
def test_vehicles_replay_bitwise_in_lockstep_on_1_and_8_threads(repeat: int) -> None:
    plan, free = record_free_running(seed=repeat)
    assert run_lockstep(1, plan) == free
    assert run_lockstep(8, plan) == free
