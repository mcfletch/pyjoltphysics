"""Characters and queries on the world thread; one JSON line per run.

usage: python benchmarks/characters.py [--case NAME ...] [--threads N ...] [--frames F]

Cases, after the 2026-10-26 spike's ``native/scenes.py``:

``chars300_hf``
    300 characters on a 256 x 256 height field (1 m cells, a few metres of
    relief), each running at 4 m/s in a heading that changes every 30 frames.
``chars30_posts``
    30 such characters on the spike's 200 m ripple mesh among 10,000 static
    posts (0.4 x 2 x 0.4 m).
``rays100k``
    100,000 rays straight down from scattered points over the posts world, in
    one ``raycast_batch`` call.
``stream``
    10 characters sprinting across a 1 km square height field cut into tiles of
    32 cells; each frame the tiles within 48 m of a character are kept and
    the rest removed, one ``swap_bodies`` batch for the frame's changes. The
    frames with a swap are timed apart from the others.
``query_wait``
    single rays from the caller while the world free-runs at 120 Hz with
    5,000 active boxes; the wait for the world lock, mean, p99 and max.

A frame is 1/60 s: ``--tick-hz 60`` runs one tick a frame, 120 two. Reported in
milliseconds per frame: ``frame_ms``, the caller's wall time for its inputs
and ``run_ticks``; ``characters_ms``, the world thread's time moving the
characters; ``step_ms``, ``PhysicsSystem::Update``. ``--rough`` turns enhanced
internal edge removal off for the ground and the characters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from boxes5000 import POST, SIZE, dynamic_grid, posts, ripple_mesh

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    CharacterCapabilities,
    MotionType,
    Shape,
    World,
    WorldSettings,
    heightfield_tiles,
)

SPIKE_MS = {'chars300_hf': 3.08, 'rays100k': 54.0}


def heightfield(n: int = 256) -> np.ndarray:
    xs = (np.arange(n) - n / 2).astype(np.float64)
    x, z = np.meshgrid(xs, xs)
    return (2.0 * np.sin(x * 0.05) * np.cos(z * 0.07) + 0.5 * np.sin(x * 0.21 + z * 0.13)).astype(
        np.float32
    )


def starts(count: int, span: float = 60.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-span, span, (count, 3))
    p[:, 1] = 4.0
    return p


def headings(count: int, frame: int, seed: int = 5) -> np.ndarray:
    """Each character's run direction at ``frame``: a new one every 30 frames."""
    out = np.zeros((count, 3))
    for k in range(count):
        epoch = (frame + k) // 30
        a = np.random.default_rng((seed, k, epoch)).uniform(0, 2 * np.pi)
        out[k] = (np.cos(a), 0.0, np.sin(a))
    return out


def summary(walls: list[float]) -> dict[str, float]:
    wall = np.array(walls) * 1e3
    return {
        'frame_ms': round(float(wall.mean()), 3),
        'frame_p95_ms': round(float(np.percentile(wall, 95)), 3),
        'frame_max_ms': round(float(wall.max()), 3),
    }


def characters_case(
    name: str, threads: int, frames: int, tick_hz: float, rough: bool
) -> dict[str, object]:
    count = 300 if name == 'chars300_hf' else 30
    flags = BodyFlag.NO_SMOOTH_EDGES if rough else BodyFlag.NONE
    with World(WorldSettings(threads=threads, tick_hz=tick_hz)) as world:
        if name == 'chars300_hf':
            world.add_bodies(
                BodyBatch.heightfield(heightfield(), origin=(-128.0, 0.0, -128.0), flags=flags)
            )
        else:
            world.add_bodies(
                BodyBatch.of_shape(
                    [(0.0, 0.0, 0.0)], ripple_mesh(), motion=MotionType.STATIC, flags=flags
                )
            )
            world.add_bodies(BodyBatch.boxes(posts(10_000), size=POST, motion=MotionType.STATIC))
        caps = CharacterCapabilities(smoothEdges=not rough)
        made = [world.add_character(tuple(p), caps) for p in starts(count).tolist()]
        ids = np.array([c.id for c in made], np.uint32)
        ticks = round(tick_hz / 60.0)
        codes = np.zeros(count, np.uint8)
        moves = np.zeros(count, np.uint8)

        def steer(frame: int) -> None:
            if frame % 30 == 0 or frame < 2:
                values = np.concatenate([headings(count, frame), np.full((count, 1), 4.0)], axis=1)
                world.character_ops(ids, moves, codes, values)

        for frame in range(30):
            steer(frame)
            world.run_ticks(ticks)
        world.reset_stats()
        walls = []
        for frame in range(30, 30 + frames):
            start = time.perf_counter()
            steer(frame)
            world.run_ticks(ticks)
            walls.append(time.perf_counter() - start)
        stats = world.stats()
        snapshot = world.latest()
        assert snapshot is not None
        rows = snapshot.characters
        return {
            'case': name,
            'threads': threads,
            'tick_hz': tick_hz,
            'rough': rough,
            'characters': len(rows),
            'grounded': int(rows.grounded.sum()),
            **summary(walls),
            'characters_ms': round(ticks * float(stats['characters_ms']), 3),
            'step_ms': round(ticks * float(stats['step_ms']), 3),
            'spike_ms': SPIKE_MS.get(name),
        }


def stream_case(threads: int, frames: int) -> dict[str, object]:
    n, tile, reach = 1025, 32, 48.0
    x = np.arange(n, dtype=np.float64)
    heights = (3.0 * np.sin(x[None, :] * 0.02) * np.cos(x[:, None] * 0.015)).astype(np.float32)
    origin = (-512.0, 0.0, -512.0)
    every = heightfield_tiles(heights, origin=origin, tile=tile)
    corners = np.asarray(every.positions)
    shapes = list(every.shape)  # type: ignore[arg-type]  # heightfield_tiles names a list of Shape
    centres = corners[:, [0, 2]] + tile / 2
    with World(WorldSettings(threads=threads, tick_hz=60.0)) as world:
        held: dict[int, int] = {}

        def stream(foci: np.ndarray) -> int:
            distance = np.linalg.norm(centres[:, None, :] - foci[None, :, [0, 2]], axis=2).min(
                axis=1
            )
            near = set(np.flatnonzero(distance < reach).tolist())
            leaving = [held.pop(k) for k in list(held) if k not in near]
            coming = sorted(near - set(held))
            if not coming and not leaving:
                return 0
            chosen: list[Shape | int] = [shapes[k] for k in coming]
            batch = BodyBatch.of_shape(
                corners[coming].reshape(-1, 3), chosen, motion=MotionType.STATIC
            )
            uids, _ = world.swap_bodies(leaving, batch)
            held.update(zip(coming, uids.tolist(), strict=True))
            return len(coming) + len(leaving)

        rng = np.random.default_rng(4)
        foci = np.column_stack(
            [rng.uniform(-300, -200, 10), np.full(10, 8.0), rng.uniform(-100, 100, 10)]
        )
        stream(foci)
        made = [world.add_character(tuple(p), CharacterCapabilities()) for p in foci.tolist()]
        world.run_ticks(60)
        for c in made:
            c.set_move((1.0, 0.0, 0.1), 'sprint')
        quiet, swapping, changed, held_max = [], [], 0, 0
        for _ in range(frames):
            snapshot = world.latest()
            assert snapshot is not None
            start = time.perf_counter()
            count = stream(np.array(snapshot.characters.position))
            world.run_ticks(1)
            (swapping if count else quiet).append(time.perf_counter() - start)
            changed += count
            held_max = max(held_max, len(held))
        snapshot = world.latest()
        assert snapshot is not None
        rows = snapshot.characters
        return {
            'case': 'stream',
            'threads': threads,
            'tiles_changed': changed,
            'tiles_held_max': held_max,
            'frames_swapping': len(swapping),
            'swap_frame_ms': round(float(np.mean(swapping)) * 1e3, 3) if swapping else None,
            'swap_frame_max_ms': round(float(np.max(swapping)) * 1e3, 3) if swapping else None,
            'quiet_frame_ms': round(float(np.mean(quiet)) * 1e3, 3) if quiet else None,
            'grounded': int(rows.grounded.sum()),
            'lowest_feet': round(float((rows.position[:, 1] - 0.5 * rows.height).min()), 2),
        }


def rays_case(threads: int, repeats: int) -> dict[str, object]:
    with World(WorldSettings(threads=threads)) as world:
        world.add_bodies(
            BodyBatch.of_shape([(0.0, 0.0, 0.0)], ripple_mesh(), motion=MotionType.STATIC)
        )
        world.add_bodies(BodyBatch.boxes(posts(10_000), size=POST, motion=MotionType.STATIC))
        world.run_ticks(1)
        rng = np.random.default_rng(7)
        origins = rng.uniform(-SIZE * 0.45, SIZE * 0.45, (100_000, 3))
        origins[:, 1] = 50.0
        directions = np.tile((0.0, -1.0, 0.0), (100_000, 1))
        world.raycast_batch(origins[:1000], directions[:1000], 100.0)
        walls = []
        for _ in range(repeats):
            start = time.perf_counter()
            hits = world.raycast_batch(origins, directions, 100.0)
            walls.append(time.perf_counter() - start)
        best = min(walls)
        return {
            'case': 'rays100k',
            'threads': threads,
            'call_ms': round(best * 1e3, 2),
            'call_mean_ms': round(float(np.mean(walls)) * 1e3, 2),
            'us_per_ray': round(best * 1e6 / 100_000, 3),
            'hit_posts': int((hits.distance < 49.5).sum()),
            'spike_ms': SPIKE_MS['rays100k'],
        }


def query_wait_case(threads: int, seconds: float) -> dict[str, object]:
    with World(WorldSettings(threads=threads, tick_hz=120.0)) as world:
        world.add_bodies(
            BodyBatch.of_shape([(0.0, 0.0, 0.0)], ripple_mesh(), motion=MotionType.STATIC)
        )
        world.add_bodies(BodyBatch.boxes(posts(1000), size=POST, motion=MotionType.STATIC))
        uids, _ = world.add_bodies(BodyBatch.boxes(dynamic_grid(5000), size=(0.5, 0.5, 0.5)))
        world.start()
        world.wait_tick(30)
        rng = np.random.default_rng(9)
        latencies = []
        deadline = time.perf_counter() + seconds
        kicks = 0
        while time.perf_counter() < deadline:
            o = rng.uniform(-80, 80, 3)
            o[1] = 30.0
            start = time.perf_counter()
            world.raycast(o, (0.0, -1.0, 0.0), 100.0)
            latencies.append(time.perf_counter() - start)
            if len(latencies) % 2000 == 0:
                world.add_impulses(uids[kicks % 4 :: 4], (0.0, 3.0, 0.0))
                kicks += 1
            time.sleep(0.0002)
        stats = world.stats()
        world.stop()
        lat = np.array(latencies) * 1e6
        return {
            'case': 'query_wait',
            'threads': threads,
            'queries': len(latencies),
            'ray_us_p50': round(float(np.percentile(lat, 50)), 2),
            'ray_us_p99': round(float(np.percentile(lat, 99)), 2),
            'ray_us_max': round(float(lat.max()), 1),
            'wait_ms_mean': round(float(stats['query_wait_ms']), 4),
            'wait_ms_max': round(float(stats['query_wait_max_ms']), 3),
            'ticks': int(stats['ticks']),
            'tick_ms': round(float(stats['tick_ms']), 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--case',
        nargs='+',
        default=['chars300_hf', 'chars30_posts', 'rays100k', 'query_wait'],
    )
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--tick-hz', type=float, nargs='+', default=[60.0, 120.0])
    parser.add_argument('--rough', action='store_true')
    args = parser.parse_args()
    for case in args.case:
        for threads in args.threads:
            if case == 'stream':
                result = stream_case(threads, args.frames * 4)
            elif case == 'rays100k':
                result = rays_case(threads, 5)
            elif case == 'query_wait':
                result = query_wait_case(threads, 5.0)
            else:
                for tick_hz in args.tick_hz:
                    result = characters_case(case, threads, args.frames, tick_hz, args.rough)
                    result['python'] = sys.version.split()[0]
                    print(json.dumps(result), flush=True)
                continue
            result['python'] = sys.version.split()[0]
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
