"""Characters replay bitwise: a free-running run with inputs at wall-clock moments, in lockstep."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, CharacterCapabilities, MotionType, World, WorldSettings
from pyjoltphysics.testing import character_digest, floor, grid

TICKS = 1000
COUNT = 24


def scene(world: World) -> list[int]:
    """A floor, posts, light crates and characters among them; return the character ids."""
    world.add_bodies(floor())
    world.add_bodies(
        BodyBatch.boxes(
            grid(36, spacing=3.0, y0=1.0), size=(0.4, 2.0, 0.4), motion=MotionType.STATIC
        )
    )
    world.add_bodies(
        BodyBatch.boxes(
            grid(16, spacing=2.5, y0=0.4) + (0.7, 0.0, 0.9), size=(0.6, 0.6, 0.6), mass=15.0
        )
    )
    starts = grid(COUNT, spacing=2.2, y0=1.0) + (1.4, 0.0, 1.3)
    caps = CharacterCapabilities()
    return [
        world.add_character(tuple(p), caps, pass_group=1 + (k % 3)).id
        for k, p in enumerate(starts.tolist())
    ]


def inputs(seed: int, ids: list[int]) -> list[tuple[str, int, tuple[float, ...]]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[str, int, tuple[float, ...]]] = []
    for _ in range(60):
        cid = int(rng.choice(ids))
        kind = str(rng.choice(['move', 'move', 'move', 'jump', 'slide', 'knockback']))
        angle = float(rng.uniform(0, 2 * np.pi))
        out.append((kind, cid, (np.cos(angle), 0.0, np.sin(angle), float(rng.uniform(1.0, 7.0)))))
    return out


def send(world: World, kind: str, cid: int, values: tuple[float, ...], at_tick: int = 0) -> int:
    character = world.character(cid)
    if kind == 'move':
        return character.set_move(values[:3], 'run', speed=values[3], at_tick=at_tick)
    if kind == 'jump':
        return character.jump(at_tick=at_tick)
    if kind == 'slide':
        return character.slide(0.6, at_tick=at_tick)
    return character.knockback((values[0] * 200.0, 100.0, values[2] * 200.0), at_tick=at_tick)


def record_free_running(seed: int) -> tuple[list[tuple[int, str, int, tuple[float, ...]]], str]:
    with World(WorldSettings(threads=8, tick_hz=960.0)) as world:
        ids = scene(world)
        plan = inputs(seed, ids)
        digest = hashlib.sha256()

        def record(tick: int, now: float) -> None:
            snapshot = world.latest()
            if tick <= TICKS and snapshot is not None:
                character_digest(snapshot, digest)

        world.set_after_step(record)
        world.start()
        rng = np.random.default_rng(seed + 1)
        moments = np.sort(rng.uniform(0.0, 0.8, len(plan)))
        seqs = {}
        start = time.perf_counter()
        for moment, entry in zip(moments, plan, strict=True):
            while time.perf_counter() - start < moment:
                time.sleep(0.0005)
            seqs[send(world, *entry)] = entry
        assert world.wait_tick(TICKS, timeout=60.0)
        world.stop()
        applied = {int(seq): int(tick) for seq, tick in world.drain_applied().tolist()}
    replay = [(applied[seq], *entry) for seq, entry in sorted(seqs.items())]
    assert len({tick for tick, *_ in replay}) > 5
    return replay, digest.hexdigest()


def run_lockstep(threads: int, plan: list[tuple[int, str, int, tuple[float, ...]]]) -> str:
    with World(WorldSettings(threads=threads, tick_hz=960.0)) as world:
        scene(world)
        for tick, kind, cid, values in plan:
            send(world, kind, cid, values, at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(TICKS):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            character_digest(snapshot, digest)
        return digest.hexdigest()


@pytest.mark.parametrize('repeat', range(16))
def test_characters_replay_bitwise_in_lockstep_on_1_and_8_threads(repeat: int) -> None:
    plan, free = record_free_running(seed=repeat)
    assert run_lockstep(1, plan) == free
    assert run_lockstep(8, plan) == free
