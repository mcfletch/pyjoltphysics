"""The 2026-10-26 spike's 5,000-active-box scene on the world thread; one JSON line per run.

usage: python benchmarks/boxes5000.py [--threads N ...] [--frames F] [--ground mesh|rough|box]

The scene: 1,000 static posts (0.4 x 2 x 0.4 m), 5,000 dynamic 0.5 m boxes of
1 kg on a grid, and every 20 frames an upward impulse of 3 N s to a quarter
of them. A frame is 1/60 s: two 120 Hz ticks, run in lockstep with
``run_ticks(2)``. The ground is the spike's: a 200 m square mesh of 20,000
triangles with a 5 cm ripple, with Jolt's enhanced internal edge removal
(the default for a static mesh); ``--ground rough`` is the same mesh with
``BodyFlag.NO_SMOOTH_EDGES`` and ``--ground box`` a static box.

Reported per frame, in milliseconds: ``frame_ms``, the caller's wall time for
``run_ticks(2)``; ``ticks_ms``, the world thread's time for the two ticks; and
the parts of a tick: ``step_ms`` (``PhysicsSystem::Update``), ``publish_ms``
(writing the snapshot) and ``events_ms`` (folding contacts into events).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pyjoltphysics import BodyBatch, BodyFlag, MotionType, Shape, World, WorldSettings

SIZE = 200.0
BOX = 0.5
POST = (0.4, 2.0, 0.4)
SPIKE_MS = {1: 2.71, 8: 1.20}


def posts(count: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-SIZE * 0.45, SIZE * 0.45, (count, 3))
    p[:, 1] = 1.0
    return p


def dynamic_grid(count: int) -> np.ndarray:
    side = max(1, int(np.ceil(np.sqrt(count))))
    pitch = min(3.0, SIZE * 0.8 / side)
    i = np.arange(count)
    return np.stack(
        [(i % side - side / 2) * pitch, 1.1 + 0.5 * (i % 3), (i // side - side / 2) * pitch], axis=1
    )


def ripple_mesh(cells: int = 100) -> Shape:
    """Return the spike's ground: ``2 * cells**2`` triangles over SIZE m with a 5 cm ripple."""
    edge = np.linspace(-SIZE / 2, SIZE / 2, cells + 1)
    x, z = np.meshgrid(edge, edge, indexing='xy')
    y = 0.05 * np.sin(x * 0.7) * np.cos(z * 0.6)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    k = cells + 1
    a = (np.arange(cells)[:, None] * k + np.arange(cells)[None, :]).ravel()
    triangles = np.concatenate(
        [np.stack([a, a + k, a + 1], axis=1), np.stack([a + 1, a + k, a + k + 1], axis=1)]
    )
    return Shape.mesh(points, triangles)


def run(threads: int, frames: int, ground: str = 'mesh') -> dict[str, object]:
    with World(WorldSettings(threads=threads)) as world:
        if ground in ('mesh', 'rough'):
            flags = BodyFlag.NO_SMOOTH_EDGES if ground == 'rough' else BodyFlag.NONE
            world.add_bodies(
                BodyBatch.of_shape(
                    [(0.0, 0.0, 0.0)], ripple_mesh(), motion=MotionType.STATIC, flags=flags
                )
            )
        else:
            world.add_bodies(
                BodyBatch.boxes(
                    [(0.0, -0.5, 0.0)], size=(SIZE, 1.0, SIZE), motion=MotionType.STATIC
                )
            )
        world.add_bodies(BodyBatch.boxes(posts(1000), size=POST, motion=MotionType.STATIC))
        uids, _ = world.add_bodies(BodyBatch.boxes(dynamic_grid(5000), size=(BOX, BOX, BOX)))

        def kick(frame: int) -> None:
            if frame % 20 == 0:
                chosen = uids[frame % 4 :: 4]
                world.add_impulses(chosen, (0.0, 3.0, 0.0))

        for frame in range(30):
            kick(frame)
            world.run_ticks(2)
            world.drain_events()
        world.reset_stats()
        walls = []
        for frame in range(frames):
            kick(frame)
            start = time.perf_counter()
            world.run_ticks(2)
            walls.append(time.perf_counter() - start)
            world.drain_events()
        stats = world.stats()
        snapshot = world.latest()
        assert snapshot is not None
        wall = np.array(walls) * 1e3
        return {
            'case': 'boxes5000_p1000_active',
            'ground': ground,
            'threads': threads,
            'python': sys.version.split()[0],
            'free_threaded': not getattr(sys, '_is_gil_enabled', lambda: True)(),
            'frames': frames,
            'frame_ms': round(float(wall.mean()), 3),
            'frame_p95_ms': round(float(np.percentile(wall, 95)), 3),
            'ticks_ms': round(2 * float(stats['tick_ms']), 3),
            'step_ms': round(float(stats['step_ms']), 3),
            'publish_ms': round(float(stats['publish_ms']), 4),
            'events_ms': round(float(stats['events_ms']), 4),
            'apply_ms': round(float(stats['apply_ms']), 4),
            'awake': int(snapshot.awake.sum()),
            'spike_frame_ms': SPIKE_MS.get(threads),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--ground', choices=('mesh', 'rough', 'box'), default='mesh')
    args = parser.parse_args()
    for threads in args.threads:
        for _ in range(args.repeats):
            print(json.dumps(run(threads, args.frames, args.ground)), flush=True)


if __name__ == '__main__':
    main()
