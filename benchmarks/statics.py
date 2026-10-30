"""Bulk static bodies and a height field world; one JSON line per case and thread count.

usage: python benchmarks/statics.py [--threads N ...] [--frames F]

``posts``: 1,000 dynamic 0.5 m boxes falling among 10,000 static posts (0.4 x
2 x 0.4 m, one shared shape) on a 600 m box floor. The posts are added in one
batch while the boxes fall, free-running at 120 Hz: ``add_ms`` is the
caller's time for ``add_bodies``, ``add_tick_ms`` the world thread's time for
the tick that applied the batch, and ``frame_ms`` the caller's time for
``run_ticks(2)`` afterwards.

``heightfield``: 300 spheres of 0.3 m falling onto a 257 x 257 height field
of 1 m cells with 3 m hills, ``frame_ms`` over the frames they fall and roll.

Reported in milliseconds; ``publish_ms`` is the snapshot's cost a tick.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pyjoltphysics import BodyBatch, MotionType, Shape, World, WorldSettings
from pyjoltphysics.testing import floor, grid


def frames(world: World, count: int) -> np.ndarray:
    walls = []
    for _ in range(count):
        start = time.perf_counter()
        world.run_ticks(2)
        walls.append(time.perf_counter() - start)
        world.drain_events()
    return np.array(walls) * 1e3


def posts(threads: int, count: int) -> dict[str, object]:
    with World(WorldSettings(threads=threads)) as world:
        world.add_bodies(floor(300.0))
        world.add_bodies(BodyBatch.boxes(grid(1000, spacing=2.0, y0=6.0), size=(0.5, 0.5, 0.5)))
        world.run_ticks(10)
        rng = np.random.default_rng(1)
        where = rng.uniform(-280.0, 280.0, (10_000, 3)) * (1.0, 0.0, 1.0) + (0.0, 1.0, 0.0)
        post = world.add_shape(Shape.box((0.4, 2.0, 0.4)))
        world.reset_stats()
        start = time.perf_counter()
        world.add_bodies(BodyBatch.of_shape(where, post, motion=MotionType.STATIC))
        add_ms = (time.perf_counter() - start) * 1e3
        world.run_ticks(1)
        stats = world.stats()
        add_tick = float(stats['tick_ms'])
        add_apply = float(stats['apply_ms'])
        world.run_ticks(1)  # the broad phase is optimised before this tick
        world.reset_stats()
        wall = frames(world, count)
        stats = world.stats()
        return {
            'case': 'posts10000_dynamic1000',
            'threads': threads,
            'add_ms': round(add_ms, 3),
            'add_tick_ms': round(add_tick, 3),
            'add_apply_ms': round(add_apply, 3),
            'frame_ms': round(float(wall.mean()), 3),
            'frame_p95_ms': round(float(np.percentile(wall, 95)), 3),
            'publish_ms': round(float(stats['publish_ms']), 4),
            'optimizations': int(stats['broadphase_optimizations']),
        }


def heightfield(threads: int, count: int) -> dict[str, object]:
    with World(WorldSettings(threads=threads)) as world:
        x = np.arange(257.0)
        hills = 3.0 * np.sin(x[None, :] / 15.0) * np.cos(x[:, None] / 11.0)
        start = time.perf_counter()
        land = world.add_shape(Shape.heightfield(hills, spacing=1.0))
        build_ms = (time.perf_counter() - start) * 1e3
        world.add_bodies(
            BodyBatch.of_shape([(-128.0, 0.0, -128.0)], land, motion=MotionType.STATIC)
        )
        rng = np.random.default_rng(2)
        drops = rng.uniform((-100.0, 6.0, -100.0), (100.0, 20.0, 100.0), (300, 3))
        world.add_bodies(BodyBatch.spheres(drops, radius=0.3))
        world.run_ticks(2)
        world.reset_stats()
        wall = frames(world, count)
        stats = world.stats()
        snapshot = world.latest()
        assert snapshot is not None
        return {
            'case': 'heightfield257_spheres300',
            'threads': threads,
            'build_ms': round(build_ms, 3),
            'frame_ms': round(float(wall.mean()), 3),
            'frame_p95_ms': round(float(np.percentile(wall, 95)), 3),
            'publish_ms': round(float(stats['publish_ms']), 4),
            'awake': int(snapshot.awake.sum()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--frames', type=int, default=240)
    args = parser.parse_args()
    for case in (posts, heightfield):
        for threads in args.threads:
            result = case(threads, args.frames)
            result['python'] = sys.version.split()[0]
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
