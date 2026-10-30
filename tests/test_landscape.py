"""Ground streamed around characters: tile bodies swapped in one tick, and walked across."""

from __future__ import annotations

from typing import cast

import numpy as np

from pyjoltphysics import BodyBatch, CharacterCapabilities, Shape, heightfield_tiles
from pyjoltphysics.testing import MakeWorld


def landscape(size: int = 129) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    return (1.5 * np.sin(x[None, :] * 0.08) * np.cos(x[:, None] * 0.06)).astype(np.float32)


def test_a_swap_removes_and_adds_in_the_same_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    old, _ = world.add_bodies(heightfield_tiles(landscape(65), origin=(-32, 0, -32), tile=32))
    world.run_ticks(1)
    world.drain_applied()
    new, seq = world.swap_bodies(
        old, heightfield_tiles(landscape(65) + 1.0, origin=(-32, 0, -32), tile=32)
    )
    world.run_ticks(1)
    assert world.drain_applied().tolist() == [[seq, 1]]
    snapshot = world.latest()
    assert snapshot is not None
    assert sorted(snapshot.uid.tolist()) == sorted(new.tolist())
    assert world.stats()['bodies'] == len(new)


def test_a_character_walks_across_tiles_swapped_in_around_it(make_world: MakeWorld) -> None:
    """Tiles of 16 cells within 24 m of the walker are kept; the rest are removed, one batch a tick."""
    world = make_world(threads=4, tick_hz=60.0)
    heights = landscape(257)
    tile = 16
    origin = np.array([-128.0, 0.0, -128.0])
    every = heightfield_tiles(heights, origin=tuple(origin), tile=tile)
    corners = np.asarray(every.positions)
    shapes = cast(list[Shape], every.shape)
    held: dict[int, int] = {}

    def stream(around: np.ndarray) -> None:
        centres = corners[:, [0, 2]] + tile / 2
        near = set(np.flatnonzero(np.linalg.norm(centres - around[[0, 2]], axis=1) < 24.0).tolist())
        leaving = [held.pop(k) for k in list(held) if k not in near]
        coming = sorted(near - set(held))
        if not coming and not leaving:
            return
        batch = BodyBatch.of_shape(
            corners[coming].reshape(-1, 3), [shapes[k] for k in coming], motion=0
        )
        uids, _ = world.swap_bodies(leaving, batch)
        held.update(zip(coming, uids.tolist(), strict=True))

    stream(np.array([-60.0, 0.0, 0.0]))
    walker = world.add_character((-60.0, 4.0, 0.0), CharacterCapabilities(), gravity=9.81)
    world.run_ticks(90)
    assert walker.grounded
    walker.set_move((1.0, 0.0, 0.0), 'sprint')
    for _ in range(14 * 60 // 4):
        world.run_ticks(4)
        stream(np.array(walker.position))
        assert walker.position[1] > -5.0
    assert walker.position[0] > 50.0
    assert walker.grounded
    assert len(held) < 20
