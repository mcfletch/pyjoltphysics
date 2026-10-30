"""Replay with joints, materials, filters, gravity zones, drag, tracks and the contact filter in the scene."""

from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    GravityZone,
    Joint,
    MotionType,
    Motor,
    MotorMode,
    Snapshot,
    Track,
    World,
    WorldSettings,
)
from pyjoltphysics.testing import Digest, floor, grid, pose_digest

TICKS = 600


def joint_digest(snapshot: Snapshot, digest: Digest) -> None:
    joints = snapshot.joints
    for name in ('id', 'state', 'value', 'value2', 'force', 'torque'):
        digest.update(np.ascontiguousarray(getattr(joints, name)).tobytes())


def scene(world: World) -> np.ndarray:
    world.set_filter_table([[True, True, False], [True, True, True], [False, True, True]])
    rubber = world.materials.define(static_friction=1.0, dynamic_friction=0.8, restitution=0.6)
    ice = world.materials.define(friction=0.05)
    world.materials.pair(rubber, ice, friction=0.3)
    world.set_contact_filter(lambda a, b, *rest: (a * 7 + b) % 5 != 0)
    world.add_bodies(floor(30.0, material=ice, flags=BodyFlag.FILTER_CONTACTS))
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            grid(200, spacing=0.7, y0=1.0),
            size=(0.5, 0.5, 0.5),
            material=np.arange(200) % 3,
            filter=np.arange(200) % 3,
            quadratic_drag=0.02,
            flags=BodyFlag.FILTER_CONTACTS,
        )
    )
    links, _ = world.add_bodies(
        BodyBatch.boxes([(10.0 + 0.5 * i, 8.0, 0.0) for i in range(8)], size=(0.4, 0.1, 0.1))
    )
    chain = [Joint.hinge(int(links[0]), anchor=(9.75, 8.0, 0.0), axis=(0, 0, 1))]
    chain += [
        Joint.ball(
            int(links[i + 1]),
            to=int(links[i]),
            anchor=(10.25 + 0.5 * i, 8.0, 0.0),
            break_force=400.0,
        )
        for i in range(7)
    ]
    wheel, _ = world.add_bodies(
        BodyBatch.boxes([(-10.0, 2.0, 0.0)], size=(1.0, 1.0, 0.2), mass=3.0)
    )
    chain.append(
        Joint.hinge(
            int(wheel[0]),
            anchor=(-10.0, 2.0, 0.0),
            axis=(0, 0, 1),
            motor=Motor(MotorMode.VELOCITY, 4.0, 30.0),
        )
    )
    spinner, _ = world.add_bodies(BodyBatch.boxes([(-14.0, 2.0, 0.0)], size=(0.6, 0.6, 0.6)))
    chain.append(Joint.motor(int(spinner[0]), axes=[(0, 1, 0)], torques=[0.5]))
    world.add_joints(chain)
    platform, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.2, 12.0)], size=(3.0, 0.2, 3.0), motion=MotionType.KINEMATIC)
    )
    world.set_track(
        platform, Track([0.0, 1.0, 2.0], [(0, 0.2, 12), (0, 3, 12), (0, 0.2, 12)], loop=True)
    )
    world.set_gravity_zones(
        [
            GravityZone.box((3.0, 5.0, 0.0), (4.0, 10.0, 4.0), gravity=3.0, direction=(0, 1, 0)),
            GravityZone.sphere((-3.0, 2.0, 0.0), 3.0, gravity=0.0, replace=True, priority=2),
        ]
    )
    return uids


def kicks(seed: int, uids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        (rng.choice(uids, size=20, replace=False), rng.uniform(-2.0, 6.0, (20, 3)))
        for _ in range(30)
    ]


def run_lockstep(threads: int, plan: list[tuple[int, np.ndarray, np.ndarray]]) -> str:
    with World(WorldSettings(threads=threads, tick_hz=480.0)) as world:
        scene(world)
        for tick, chosen, impulses in plan:
            world.add_impulses(chosen, impulses, at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(TICKS):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            pose_digest(snapshot, digest)
            joint_digest(snapshot, digest)
        return digest.hexdigest()


def record_free_running(seed: int) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], str]:
    with World(WorldSettings(threads=8, tick_hz=480.0)) as world:
        uids = scene(world)
        batches = kicks(seed, uids)
        digest = hashlib.sha256()
        seqs = {}

        def record(tick: int, now: float) -> None:
            snapshot = world.latest()
            if tick <= TICKS and snapshot is not None:
                pose_digest(snapshot, digest)
                joint_digest(snapshot, digest)

        world.set_after_step(record)
        world.start()
        rng = np.random.default_rng(seed + 1)
        moments = np.sort(rng.uniform(0.0, 0.9, len(batches)))
        start = time.perf_counter()
        for moment, (chosen, impulses) in zip(moments, batches, strict=True):
            while time.perf_counter() - start < moment:
                time.sleep(0.0005)
            seqs[world.add_impulses(chosen, impulses)] = (chosen, impulses)
        assert world.wait_tick(TICKS, timeout=120.0)
        world.stop()
        assert world.error is None
        applied = {int(seq): int(tick) for seq, tick in world.drain_applied().tolist()}
    plan = [(applied[seq], chosen, impulses) for seq, (chosen, impulses) in sorted(seqs.items())]
    assert len({tick for tick, _, _ in plan}) > 5
    return plan, digest.hexdigest()


@pytest.mark.parametrize('repeat', range(4))
def test_a_free_running_record_with_joints_zones_and_tracks_replays_bitwise(repeat: int) -> None:
    plan, free = record_free_running(seed=repeat)
    assert run_lockstep(1, plan) == free
    assert run_lockstep(8, plan) == free


def test_the_scene_moves_its_joints_and_track() -> None:
    with World(WorldSettings(threads=4, tick_hz=480.0)) as world:
        scene(world)
        world.run_ticks(240)
        snapshot = world.latest()
        assert snapshot is not None
        assert snapshot.joints.count == 10
        assert np.any(np.abs(snapshot.joints.value) > math.radians(5.0))
