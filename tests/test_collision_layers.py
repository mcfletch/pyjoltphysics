"""Collision layers: one per distinct (group, mask, filter, motion, role) in use, freed with its last user."""

from __future__ import annotations

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, MotionType
from pyjoltphysics.testing import MakeWorld, latest

#: Jolt names a layer in 16 bits and keeps one value for "no layer".
LAYERS = 65535


def statics(count: int, first_mask: int) -> BodyBatch:
    masks = np.arange(first_mask, first_mask + count, dtype=np.uint32)
    return BodyBatch.spheres(np.zeros((count, 3)), radius=0.5, motion=MotionType.STATIC, mask=masks)


def test_layers_of_removed_bodies_are_used_again(make_world: MakeWorld) -> None:
    """70 batches of 1000 distinct masks, each removed after a tick: 70000 keys over the life, 1000 at once."""
    world = make_world(threads=2)
    for batch in range(70):
        uids, _ = world.add_bodies(statics(1000, 1 + batch * 1000))
        world.run_ticks(1)
        world.remove_bodies(uids)
        world.run_ticks(1)
    assert world.stats()['rejected'] == 0
    assert world.error is None
    (uid,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)]))
    world.run_ticks(2)
    assert latest(world).position[latest(world).rows[int(uid)], 1] < 5.0


def test_a_batch_past_the_layers_raises_on_the_callers_thread(make_world: MakeWorld) -> None:
    world = make_world(threads=2, max_bodies=LAYERS + 2000)
    world.add_bodies(statics(LAYERS - 10, 1))
    world.run_ticks(1)
    with pytest.raises(ValueError, match='collision layers'):
        world.add_bodies(statics(11, LAYERS + 100))
    world.add_bodies(statics(10, LAYERS + 100))
    world.run_ticks(1)
    assert world.stats()['bodies'] == LAYERS
    assert world.stats()['rejected'] == 0
    assert world.error is None


def test_rows_a_queued_batch_left_no_layer_for_are_rejected(make_world: MakeWorld) -> None:
    """The submit check counts layers in use; two batches queued together can pass it and overrun."""
    world = make_world(threads=2, max_bodies=LAYERS + 2000)
    world.add_bodies(statics(LAYERS - 10, 1))
    world.add_bodies(statics(20, LAYERS + 100))
    world.run_ticks(1)
    assert world.stats()['bodies'] == LAYERS
    assert world.stats()['rejected'] == 10
    assert world.stats()['queued_bodies'] == 0
    assert world.error is None


def test_a_change_the_layers_cannot_hold_is_rejected(make_world: MakeWorld) -> None:
    """A filter change needs a new layer; with none free it is counted and the world steps on."""
    world = make_world(threads=2, max_bodies=LAYERS + 10)
    uids, _ = world.add_bodies(statics(LAYERS, 1))
    world.run_ticks(1)
    world.set_body_filters(uids[:1], [7])
    world.run_ticks(1)
    assert world.stats()['rejected'] == 1
    # Removing a body frees its layer for the new key; the changed body's old layer is then freed.
    world.remove_bodies(uids[1:2])
    world.set_body_filters(uids[:1], [7])
    world.run_ticks(1)
    assert world.stats()['rejected'] == 1
    assert world.error is None


def test_bodies_sharing_a_key_share_its_layer_until_the_last_goes(make_world: MakeWorld) -> None:
    world = make_world(threads=2, max_bodies=LAYERS + 10)
    shared, _ = world.add_bodies(
        BodyBatch.spheres(np.zeros((3, 3)), radius=0.5, motion=MotionType.STATIC, mask=5)
    )
    world.add_bodies(statics(LAYERS - 1, 100))
    world.run_ticks(1)
    world.remove_bodies(shared[:2])
    world.run_ticks(1)
    with pytest.raises(ValueError, match='collision layers'):
        world.add_bodies(statics(1, 10**6))
    world.remove_bodies(shared[2:])
    world.run_ticks(1)
    world.add_bodies(statics(1, 10**6))
    world.run_ticks(1)
    assert world.stats()['rejected'] == 0
