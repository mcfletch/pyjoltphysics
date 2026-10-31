"""Ragdolls and soft bodies replay bitwise: commands at wall-clock moments, replayed in lockstep."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    MotionType,
    RagdollMode,
    RagdollSettings,
    SoftBodySettings,
    World,
    WorldSettings,
)
from pyjoltphysics.testing import floor, ragdoll_digest, soft_digest

TICKS = 600
HZ = 240.0

Command = Callable[[World, dict[str, list[int]]], int]


def scene(world: World) -> dict[str, list[int]]:
    """A floor, a sphere, three ragdolls in three modes, a cloth, a rope and a ball."""
    world.add_bodies(floor(50.0))
    world.add_bodies(BodyBatch.spheres([(0.0, 0.5, 4.0)], radius=0.5, motion=MotionType.STATIC))
    settings = RagdollSettings.humanoid(root_force=4000.0, root_torque=1500.0)
    limp = world.add_ragdoll(settings, position=(-2.0, 0.4, 0.0))
    powered = world.add_ragdoll(settings, position=(0.0, 0.05, 0.0), mode=RagdollMode.POWERED)
    kinematic = world.add_ragdoll(settings, position=(2.0, 0.0, 0.0), mode=RagdollMode.KINEMATIC)
    rest = settings.rest_rotations
    swing = np.repeat(rest[None], 3, axis=0)
    swing[1, settings.bone_index('upper_arm_l')] = (0.0, 0.0, 0.5, 0.866)
    kinematic.play([0.0, 0.5, 1.0], swing, loop=True)
    cloth = SoftBodySettings.cloth(2.0, 2.0, columns=16, rows=16, mass=1.0, vertex_radius=0.01)
    sheet = world.add_soft_body(cloth, position=(0.0, 1.5, 4.0))
    rope = SoftBodySettings.rope([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], segments=12, mass=0.2)
    line = world.add_soft_body(rope, position=(4.0, 2.0, 0.0), pins=([0], [(4.0, 2.0, 0.0)]))
    ball = world.add_soft_body(
        SoftBodySettings.ball(0.3, 1, pressure=2000.0), position=(-4.0, 1.5, 0.0)
    )
    return {
        'ragdolls': [limp.id, powered.id, kinematic.id],
        'bones': [int(u) for u in limp.bones] + [int(u) for u in powered.bones],
        'soft': [sheet.uid, line.uid, ball.uid],
    }


def commands(seed: int) -> list[tuple[str, int, tuple[float, ...]]]:
    """Hits on bones, wind on soft bodies and mode changes, as (kind, index, values)."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, int, tuple[float, ...]]] = []
    for _ in range(30):
        kind = str(rng.choice(['hit', 'wind', 'mode']))
        values = tuple(float(v) for v in rng.uniform(-20.0, 20.0, 3))
        out.append((kind, int(rng.integers(0, 1000)), values))
    return out


def submit(
    world: World,
    ids: dict[str, list[int]],
    command: tuple[str, int, tuple[float, ...]],
    at_tick: int = 0,
) -> int:
    kind, index, values = command
    if kind == 'hit':
        bones = ids['bones']
        return world.add_impulses([bones[index % len(bones)]], [values], at_tick)
    if kind == 'wind':
        soft = ids['soft']
        return world.set_wind([soft[index % len(soft)]], [values], at_tick)
    ragdolls = ids['ragdolls'][:2]
    mode = RagdollMode.POWERED if index % 2 else RagdollMode.LIMP
    return world.set_ragdoll_modes([ragdolls[index % 2]], mode, blend=0.25, at_tick=at_tick)


def digest_of(world: World, digest: hashlib._Hash) -> None:
    snapshot = world.latest()
    assert snapshot is not None
    ragdoll_digest(snapshot, digest)
    soft_digest(snapshot, digest)


def record_free_running(
    seed: int,
) -> tuple[list[tuple[int, tuple[str, int, tuple[float, ...]]]], str]:
    with World(WorldSettings(threads=8, tick_hz=HZ)) as world:
        ids = scene(world)
        plan = commands(seed)
        digest = hashlib.sha256()

        def record(tick: int, now: float) -> None:
            if tick <= TICKS:
                digest_of(world, digest)

        world.set_after_step(record)
        world.start()
        rng = np.random.default_rng(seed + 1)
        moments = np.sort(rng.uniform(0.0, 0.8 * TICKS / HZ, len(plan)))
        seqs = {}
        start = time.perf_counter()
        for moment, command in zip(moments, plan, strict=True):
            while time.perf_counter() - start < moment:
                time.sleep(0.0005)
            seqs[submit(world, ids, command)] = command
        assert world.wait_tick(TICKS, timeout=120.0)
        world.stop()
        applied = {int(seq): int(tick) for seq, tick in world.drain_applied().tolist()}
    replay = [(applied[seq], command) for seq, command in sorted(seqs.items())]
    assert len({tick for tick, _ in replay}) > 5
    return replay, digest.hexdigest()


def run_lockstep(threads: int, plan: list[tuple[int, tuple[str, int, tuple[float, ...]]]]) -> str:
    with World(WorldSettings(threads=threads, tick_hz=HZ)) as world:
        ids = scene(world)
        for tick, command in plan:
            submit(world, ids, command, at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(TICKS):
            world.run_ticks(1)
            digest_of(world, digest)
        return digest.hexdigest()


@pytest.mark.parametrize('repeat', range(2))
def test_ragdolls_and_soft_bodies_replay_bitwise_on_1_and_8_threads(repeat: int) -> None:
    plan, free = record_free_running(seed=repeat)
    assert run_lockstep(1, plan) == free
    assert run_lockstep(8, plan) == free
