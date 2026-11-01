"""Snapshots: zero-copy read-only views whose buffers are reused only once released."""

from __future__ import annotations

import gc
import threading
import tracemalloc

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, MotionType, Snapshot, World
from pyjoltphysics.testing import MakeWorld, floor, grid


def latest(world: World) -> Snapshot:
    snapshot = world.latest()
    assert snapshot is not None
    return snapshot


def test_no_snapshot_before_the_first_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    assert world.latest() is None


def test_snapshot_records_tick_time_and_applied_seq(make_world: MakeWorld) -> None:
    world = make_world(threads=1, tick_hz=100.0)
    _, seq = world.add_bodies(BodyBatch.spheres(grid(3)))
    world.run_ticks(7)
    snapshot = latest(world)
    assert snapshot.tick == 7
    assert snapshot.time == pytest.approx(0.07, abs=1e-12)
    assert snapshot.applied_seq == seq
    assert repr(snapshot) == '<Snapshot tick=7 bodies=3 applied_seq=1>'


def test_views_are_read_only_and_typed(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.spheres(grid(4)))
    world.run_ticks(1)
    snapshot = latest(world)
    for name, dtype, shape in [
        ('uid', np.uint32, (4,)),
        ('position', np.float64, (4, 3)),
        ('orientation', np.float32, (4, 4)),
        ('linear_velocity', np.float32, (4, 3)),
        ('angular_velocity', np.float32, (4, 3)),
        ('awake', np.bool_, (4,)),
        ('motion_type', np.uint8, (4,)),
    ]:
        array = getattr(snapshot, name)
        assert array.dtype == dtype, name
        assert array.shape == shape, name
        assert not array.flags.writeable, name
        with pytest.raises(ValueError, match='read-only'):
            array[0] = 0


def test_views_share_the_buffer_without_copying(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.spheres(grid(4)))
    world.run_ticks(1)
    snapshot = latest(world)
    a = snapshot.position
    b = snapshot.position
    assert a.ctypes.data == b.ctypes.data
    assert not a.flags.owndata


def test_a_held_view_keeps_its_values_while_the_world_runs_on(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(BodyBatch.spheres(grid(50, y0=20.0)))
    world.run_ticks(1)
    view = latest(world).position  # the Snapshot is released; the view keeps it alive
    copy = view.copy()
    world.run_ticks(60)
    gc.collect()
    np.testing.assert_array_equal(view, copy)
    assert not np.array_equal(latest(world).position, copy)


def test_released_buffers_are_reused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.spheres(grid(20, y0=50.0)))
    world.run_ticks(5)
    steady = world.stats()['pool']
    assert steady <= 2
    held = []
    for _ in range(5):
        world.run_ticks(1)
        held.append(latest(world))
    ticks = [s.tick for s in held]
    positions = [s.position.copy() for s in held]
    world.run_ticks(20)
    # The five held, the published one and one being written.
    grown = world.stats()['pool']
    assert grown == 7
    for snapshot, tick, position in zip(held, ticks, positions, strict=True):
        assert snapshot.tick == tick
        np.testing.assert_array_equal(snapshot.position, position)
    del held, snapshot
    world.run_ticks(50)
    assert world.stats()['pool'] == grown


def test_changed_lists_moved_and_newly_asleep_rows(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    resting, _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.run_ticks(1)
    assert latest(world).rows[int(resting[0])] in latest(world).changed
    asleep_at = None
    for _ in range(400):
        world.run_ticks(1)
        snapshot = latest(world)
        row = snapshot.rows[int(resting[0])]
        if not snapshot.awake[row]:
            asleep_at = snapshot.tick
            assert row in snapshot.changed
            break
    assert asleep_at is not None
    world.run_ticks(1)
    assert latest(world).changed.size == 0


def test_a_reused_buffer_is_brought_up_to_date(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(
        BodyBatch.spheres(grid(10, spacing=3.0), linear_velocity=(1.0, 0.0, 0.0))
    )
    world.run_ticks(1)
    first = latest(world)
    start = {int(u): float(first.position[r, 0]) for r, u in enumerate(first.uid)}
    for tick in range(2, 60):
        if tick % 7 == 0:
            world.remove_bodies(uids[tick // 7])
        world.run_ticks(1)
        snapshot = latest(world)
        moved = np.array(
            [snapshot.position[r, 0] - start[int(u)] for r, u in enumerate(snapshot.uid)]
        )
        np.testing.assert_allclose(moved, (tick - 1) / 120.0, atol=1e-5)
    assert latest(world).count == 10 - 8


def test_rows_of_names_missing_uids(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres(grid(3)))
    world.run_ticks(1)
    snapshot = latest(world)
    assert snapshot.rows_of([int(uids[1]), 999]).tolist() == [snapshot.rows[int(uids[1])], -1]


def test_rows_of_allocates_by_live_bodies_not_by_the_highest_uid(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    streamed, _ = world.add_bodies(
        BodyBatch.spheres(grid(60_000, y0=1e4), motion=MotionType.STATIC)
    )
    world.run_ticks(1)
    world.remove_bodies(streamed)
    kept, _ = world.add_bodies(BodyBatch.spheres(grid(5)))
    world.run_ticks(1)
    snapshot = latest(world)
    assert int(snapshot.uid.max()) > 60_000
    tracemalloc.start()
    try:
        rows = snapshot.rows_of(kept)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert rows.tolist() == [snapshot.rows[int(uid)] for uid in kept]
    # A table indexed by uid would be 480 kB here.
    assert peak < 16_384
    gone = [int(streamed[0]), int(streamed[-1]), 0, -1, int(kept[-1]) + 1]
    assert snapshot.rows_of(gone).tolist() == [-1] * 5
    assert snapshot.rows_of([[int(kept[0])], [int(streamed[7])]]).tolist() == [
        [snapshot.rows[int(kept[0])]],
        [-1],
    ]


def test_rows_of_keeps_the_shape_of_its_input(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres(grid(4)))
    world.run_ticks(1)
    snapshot = latest(world)
    rows = snapshot.rows
    a, b, c, d = (int(uid) for uid in uids)
    assert snapshot.rows_of([[a, b], [c, d]]).tolist() == [[rows[a], rows[b]], [rows[c], rows[d]]]
    assert snapshot.rows_of([[a, 999], [-1, 2**40]]).tolist() == [[rows[a], -1], [-1, -1]]
    assert snapshot.rows_of(b).tolist() == [rows[b]]
    assert snapshot.rows_of([]).shape == (0,)
    assert snapshot.rows_of(np.array(uids, dtype=np.uint32)).tolist() == [
        rows[u] for u in (a, b, c, d)
    ]


def test_rows_of_rejects_uids_that_are_not_integers(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres(grid(3)))
    world.run_ticks(1)
    snapshot = latest(world)
    with pytest.raises(TypeError, match='integer'):
        snapshot.rows_of([float(uids[1]) + 0.9])


def test_rows_of_in_an_empty_world(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.run_ticks(1)
    assert latest(world).rows_of([1, 2]).tolist() == [-1, -1]


def test_many_readers_while_the_world_free_runs(make_world: MakeWorld) -> None:
    world = make_world(threads=2, tick_hz=1000.0)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes(grid(300, spacing=0.6, y0=2.0), size=(0.5, 0.5, 0.5)))
    world.start()
    stop = threading.Event()
    problems: list[str] = []

    def read() -> None:
        kept: list[tuple[np.ndarray, np.ndarray]] = []
        while not stop.is_set():
            snapshot = world.latest()
            if snapshot is None:
                continue
            view = snapshot.position
            kept.append((view, view.copy()))
            if len(kept) > 8:
                old, copy = kept.pop(0)
                if not np.array_equal(old, copy):
                    problems.append('a held view changed')

    readers = [threading.Thread(target=read) for _ in range(4)]
    for reader in readers:
        reader.start()
    assert world.wait_tick(200, timeout=30.0)
    stop.set()
    for reader in readers:
        reader.join()
    assert problems == []
