"""Bodies in bulk: one batch of thousands, and the broad phase rebuilt once after a large static add."""

from __future__ import annotations

import time

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, MotionType, Shape, World
from pyjoltphysics.testing import MakeWorld, floor, grid
from pyjoltphysics.testing import sanitizer_slowdown as slowdown


def optimizations_reach(world: World, count: int, timeout: float = 5.0) -> int:
    """Return the broad phase rebuilds once they are ``count``, or the number after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while world.stats()['broadphase_optimizations'] != count and time.monotonic() < deadline:
        time.sleep(0.01)
    return int(world.stats()['broadphase_optimizations'])


def posts(count: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    p = rng.uniform(-250.0, 250.0, (count, 3))
    p[:, 1] = 1.0
    return p


def apply_ten_thousand_posts(make_world: MakeWorld) -> float:
    """Return the milliseconds a world of 500 boxes takes to apply a batch of 10,000 static posts."""
    world = make_world(threads=4)
    world.add_bodies(floor(300.0))
    world.add_bodies(BodyBatch.boxes(grid(500, spacing=1.2), size=(0.5, 0.5, 0.5)))
    world.run_ticks(10)
    post = world.add_shape(Shape.box((0.4, 2.0, 0.4)))
    world.add_bodies(BodyBatch.of_shape(posts(10_000), post, motion=MotionType.STATIC))
    world.reset_stats()
    world.run_ticks(1)
    stats = world.stats()
    assert stats['bodies'] == 10_501
    world.stop()
    return float(stats['apply_ms'])


@pytest.mark.serial
def test_ten_thousand_static_posts_apply_within_one_tick(make_world: MakeWorld) -> None:
    # The fastest of three worlds: other load on the machine slows one apply, not all three.
    applies = [apply_ten_thousand_posts(make_world) for _ in range(3)]
    # Creating and adding the batch takes under three ticks at 120 Hz: 6 ms where this was
    # written and 14 ms on a three-core Intel runner.
    assert min(applies) < slowdown() * 3 * 1000.0 / 120.0, applies


def test_the_broad_phase_is_rebuilt_once_after_a_large_static_add(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(300.0))
    world.run_ticks(1)
    assert world.stats()['broadphase_optimizations'] == 0
    world.add_bodies(BodyBatch.boxes(posts(2000), size=(0.4, 2.0, 0.4), motion=MotionType.STATIC))
    world.run_ticks(1)
    world.run_ticks(5)
    # The world thread rebuilds the trees once it has no tick due, after run_ticks returns.
    assert optimizations_reach(world, 1) == 1
    # Fewer than 256 bodies of any kind later do not ask for another.
    world.add_bodies(BodyBatch.boxes(grid(100, spacing=1.5, y0=5.0), size=(0.5, 0.5, 0.5)))
    world.run_ticks(5)
    assert optimizations_reach(world, 1, timeout=0.5) == 1
    # Bodies past 256 since the last rebuild ask for one, dynamic or not.
    world.add_bodies(BodyBatch.boxes(grid(300, spacing=1.5, y0=8.0), size=(0.5, 0.5, 0.5)))
    world.run_ticks(5)
    assert optimizations_reach(world, 2) == 2


def test_five_thousand_dynamic_boxes_land_on_ten_thousand_posts(make_world: MakeWorld) -> None:
    world = make_world(threads=4)
    world.add_bodies(floor(300.0))
    world.add_bodies(BodyBatch.boxes(posts(10_000), size=(0.4, 2.0, 0.4), motion=MotionType.STATIC))
    uids, _ = world.add_bodies(
        BodyBatch.boxes(grid(5000, spacing=3.0, y0=4.0), size=(0.5, 0.5, 0.5))
    )
    world.run_ticks(240)
    snapshot = world.latest()
    assert snapshot is not None
    heights = snapshot.position[snapshot.rows_of(uids), 1]
    # No box passed through the floor or a post.
    assert heights.min() > 0.2
