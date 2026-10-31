"""Ragdolls and cloth on the world thread; one JSON line per run.

usage: python benchmarks/ragdolls_cloth.py [--ragdolls N] [--cloths N] [--side N] [--threads N ...]

Each run steps, in lockstep at 60 Hz, ``--ragdolls`` 11-bone humanoids and
``--cloths`` cloths of ``--side`` by ``--side`` vertices. Half the ragdolls
are powered, their root held and a looping arm swing as the pose target, so
they never sleep; the other half are dropped limp onto a field of boxes and
each other. Each cloth falls onto a sphere with a wind across it, so it stays
awake. Reported in milliseconds per tick, from the world's own timings after
half a second of warm-up: ``step_ms`` is ``PhysicsSystem::Update``,
``tick_ms`` the whole tick (apply, drive, step, events, publish).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pyjoltphysics import (
    BodyBatch,
    MotionType,
    RagdollMode,
    RagdollSettings,
    SoftBodySettings,
    World,
    WorldSettings,
)
from pyjoltphysics.testing import floor


def run(
    ragdolls: int, cloths: int, side: int, threads: int, ticks: int
) -> dict[str, float | int | str]:
    world = World(WorldSettings(tick_hz=60.0, threads=threads))
    try:
        world.add_bodies(floor(100.0))
        rng = np.random.default_rng(7)
        world.add_bodies(
            BodyBatch.boxes(
                rng.uniform((-12.0, 0.2, -12.0), (12.0, 0.2, -2.0), (30, 3)),
                size=(0.6, 0.4, 0.6),
                motion=MotionType.STATIC,
            )
        )
        settings = RagdollSettings.humanoid(root_force=4000.0, root_torque=1500.0)
        swing = np.repeat(settings.rest_rotations[None], 3, axis=0)
        for arm in ('upper_arm_l', 'upper_arm_r'):
            swing[1, settings.bone_index(arm)] = (0.0, 0.0, 0.5, 0.866)
        for i in range(ragdolls):
            x = -12.0 + 24.0 * (i // 2) / max(1, ragdolls // 2)
            if i % 2:
                doll = world.add_ragdoll(
                    settings, position=(x, 0.05, 6.0), mode=RagdollMode.POWERED
                )
                doll.play([0.0, 0.6, 1.2], swing, loop=True)
            else:
                world.add_ragdoll(
                    settings, position=(x, 1.5, -7.0), rotation=(0.3, 0.0, 0.0, 0.954)
                )
        cloth = SoftBodySettings.cloth(
            2.0, 2.0, columns=side, rows=side, mass=1.0, vertex_radius=0.01
        )
        for i in range(cloths):
            x = -10.0 + 5.0 * i
            world.add_bodies(
                BodyBatch.spheres([(x, 0.6, 14.0)], radius=0.6, motion=MotionType.STATIC)
            )
            body = world.add_soft_body(cloth, position=(x, 1.6, 14.0))
            body.set_wind((1.5, 0.0, 0.5))
        world.run_ticks(30)
        world.reset_stats()
        world.run_ticks(ticks)
        stats = world.stats()
        return {
            'case': 'ragdolls%d_cloths%dx%d' % (ragdolls, cloths, side * side),
            'threads': threads,
            'ticks': ticks,
            'step_ms': round(float(stats['step_ms']), 4),
            'tick_ms': round(float(stats['tick_ms']), 4),
            'publish_ms': round(float(stats['publish_ms']), 4),
            'characters_ms': round(float(stats['characters_ms']), 4),
        }
    finally:
        world.stop()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ragdolls', type=int, default=20)
    parser.add_argument('--cloths', type=int, default=5)
    parser.add_argument('--side', type=int, default=32)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--ticks', type=int, default=180)
    args = parser.parse_args(argv)
    for threads in args.threads:
        print(
            json.dumps(run(args.ragdolls, args.cloths, args.side, threads, args.ticks)), flush=True
        )


if __name__ == '__main__':
    main(sys.argv[1:])
