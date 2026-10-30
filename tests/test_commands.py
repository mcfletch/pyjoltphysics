"""Command batches: bodies added in bulk, moved, pushed, changed and removed."""

from __future__ import annotations

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, BodyFlag, MotionType, ShapeKind
from pyjoltphysics.testing import MakeWorld, floor, grid

G = 9.81


def free_fall_heights(y0: float, ticks: int, dt: float) -> np.ndarray:
    """Heights after each tick of Jolt's integrator in float32: v += g dt, then y += v dt."""
    g = np.float32(-G)
    step = np.float32(dt)
    v = np.float32(0.0)
    y = y0
    out = []
    for _ in range(ticks):
        v = np.float32(v + g * step)
        y = y + float(np.float32(v * step))
        out.append(y)
    return np.array(out)


def test_a_body_falls_as_the_integrator_predicts(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 100.0, 0.0)], radius=0.5))
    heights = []
    for _ in range(30):
        world.run_ticks(1)
        snapshot = world.latest()
        assert snapshot is not None
        heights.append(snapshot.position[snapshot.rows[int(uids[0])], 1])
    expected = free_fall_heights(100.0, 30, 1.0 / 120.0)
    np.testing.assert_allclose(heights, expected, rtol=0, atol=1e-9)
    # The same, against the closed form of the integrator in double precision.
    n = np.arange(1, 31)
    closed = 100.0 - G * (1.0 / 120.0) ** 2 * n * (n + 1) / 2
    np.testing.assert_allclose(heights, closed, rtol=0, atol=1e-5)


def test_boxes_fall_onto_a_floor_and_rest_on_it(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor())
    uids, _ = world.add_bodies(BodyBatch.boxes(grid(16, spacing=1.5, y0=2.0), size=(1, 1, 1)))
    world.run_ticks(600)
    snapshot = world.latest()
    assert snapshot is not None
    rows = snapshot.rows_of(uids)
    np.testing.assert_allclose(snapshot.position[rows, 1], 0.5, atol=0.03)
    assert not snapshot.awake[rows].any()


def test_five_thousand_boxes_are_added_in_one_batch(make_world: MakeWorld) -> None:
    world = make_world(threads=4)
    world.add_bodies(floor(200.0))
    uids, seq = world.add_bodies(BodyBatch.boxes(grid(5000, spacing=1.2), size=(0.5, 0.5, 0.5)))
    assert len(uids) == 5000
    assert len(np.unique(uids)) == 5000
    world.run_ticks(2)
    snapshot = world.latest()
    assert snapshot is not None
    assert snapshot.count == 5001
    assert snapshot.includes(seq)
    assert set(snapshot.uid.tolist()) >= set(uids.tolist())


def test_uids_are_assigned_at_submission_and_never_reused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    first, _ = world.add_bodies(BodyBatch.spheres(grid(3)))
    world.remove_bodies(first)
    second, _ = world.add_bodies(BodyBatch.spheres(grid(3)))
    assert first.tolist() == [1, 2, 3]
    assert second.tolist() == [4, 5, 6]


def test_applied_seq_tells_a_reader_whether_its_write_is_in_the_snapshot(
    make_world: MakeWorld,
) -> None:
    world = make_world(threads=1)
    uids, add_seq = world.add_bodies(BodyBatch.spheres([(0.0, 5.0, 0.0)]))
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None and snapshot.includes(add_seq)
    later = world.set_velocities(uids, linear=(1.0, 0.0, 0.0), at_tick=5)
    after = world.set_velocities(uids, angular=(0.0, 1.0, 0.0))
    assert after == later + 1
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None
    # `after` is applied, but `later` waits for tick 5: the contiguous sequence stops below it.
    assert snapshot.applied_seq == later - 1
    assert not snapshot.includes(after)
    world.run_ticks(4)
    snapshot = world.latest()
    assert snapshot is not None
    assert snapshot.applied_seq == after
    applied = world.drain_applied()
    assert applied.tolist() == [[add_seq, 0], [after, 1], [later, 5]]


def test_set_transforms_moves_bodies(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(BodyBatch.boxes(grid(2)))
    half = np.sqrt(0.5)
    world.set_transforms(uids, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], (0.0, half, 0.0, half))
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None
    rows = snapshot.rows_of(uids)
    np.testing.assert_allclose(snapshot.position[rows], [(1, 2, 3), (4, 5, 6)], atol=1e-9)
    np.testing.assert_allclose(snapshot.orientation[rows[0]], (0, half, 0, half), atol=1e-6)


def test_velocities_impulses_forces_and_torques(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(BodyBatch.spheres(grid(4, spacing=5.0), mass=2.0))
    a, b, c, d = (int(u) for u in uids)
    world.set_velocities([a], linear=(1.0, 0.0, 0.0))
    world.add_impulses([b], (0.0, 4.0, 0.0))  # 4 N s on 2 kg: 2 m/s
    world.add_forces([c], (0.0, 0.0, 240.0))  # 240 N for 1/120 s on 2 kg: 1 m/s
    world.add_torques([d], (0.0, 1.0, 0.0))
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None
    v = snapshot.linear_velocity
    rows = snapshot.rows
    np.testing.assert_allclose(v[rows[a]], (1, 0, 0), atol=1e-6)
    np.testing.assert_allclose(v[rows[b]], (0, 2, 0), atol=1e-6)
    np.testing.assert_allclose(v[rows[c]], (0, 0, 1), atol=1e-5)
    assert snapshot.angular_velocity[rows[d], 1] > 0.0


def test_set_gravity_wakes_sleeping_bodies(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    uids, _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.run_ticks(240)
    snapshot = world.latest()
    assert snapshot is not None and not snapshot.awake[snapshot.rows[int(uids[0])]]
    world.set_gravity((0.0, 9.81, 0.0))
    world.run_ticks(30)
    snapshot = world.latest()
    assert snapshot is not None
    assert snapshot.position[snapshot.rows[int(uids[0])], 1] > 0.6


def test_activate_wakes_a_body(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    uids, _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.run_ticks(240)
    world.activate(uids)
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None and snapshot.awake[snapshot.rows[int(uids[0])]]


def test_motion_types_change_where_the_body_allows_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    movable, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 5.0, 0.0)], motion=MotionType.STATIC, flags=BodyFlag.MOVABLE)
    )
    fixed, _ = world.add_bodies(BodyBatch.boxes([(5.0, 5.0, 0.0)], motion=MotionType.STATIC))
    world.set_motion_types(np.concatenate([movable, fixed]), MotionType.DYNAMIC)
    world.run_ticks(30)
    snapshot = world.latest()
    assert snapshot is not None
    rows = snapshot.rows
    assert snapshot.motion_type[rows[int(movable[0])]] == MotionType.DYNAMIC
    assert snapshot.position[rows[int(movable[0])], 1] < 5.0
    assert snapshot.motion_type[rows[int(fixed[0])]] == MotionType.STATIC
    assert world.stats()['rejected'] == 1


def test_set_materials_changes_the_bounce(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.plane(restitution=1.0))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 3.0, 0.0)], radius=0.5))
    world.set_materials(uids, 0.5, 1.0)
    peak = 0.0
    fell = False
    for _ in range(120):
        world.run_ticks(2)
        snapshot = world.latest()
        assert snapshot is not None
        y = float(snapshot.position[0 if snapshot.uid[0] == uids[0] else 1, 1])
        fell = fell or y < 0.7
        if fell:
            peak = max(peak, y)
    assert peak > 2.0


def test_removed_bodies_leave_the_snapshot(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    uids, _ = world.add_bodies(BodyBatch.spheres(grid(5, spacing=3.0)))
    world.run_ticks(1)
    world.remove_bodies(uids[[0, 2]])
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None
    assert sorted(snapshot.uid.tolist()) == sorted(uids[[1, 3, 4]].tolist())
    assert world.stats()['bodies'] == 3
    # The moved rows are in `changed`, so a reader keeping state by row sees them.
    for row in range(snapshot.count):
        if row in snapshot.changed:
            continue
        pytest.fail('row %d was not rewritten after the removal' % row)


def test_every_primitive_shape_comes_to_rest_on_a_plane(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(BodyBatch.plane())
    batches = [
        (BodyBatch.boxes([(0.0, 2.0, 0.0)], size=(1.0, 0.4, 1.0)), 0.2),
        (BodyBatch.spheres([(3.0, 2.0, 0.0)], radius=0.3), 0.3),
        (
            BodyBatch.capsules(
                [(6.0, 2.0, 0.0)],
                radius=0.2,
                height=1.0,
                orientations=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
            ),
            0.2,
        ),
        (BodyBatch.cylinders([(9.0, 2.0, 0.0)], radius=0.5, height=0.6), 0.3),
    ]
    uids = [int(world.add_bodies(batch)[0][0]) for batch, _ in batches]
    world.run_ticks(360)
    snapshot = world.latest()
    assert snapshot is not None
    for uid, (_, rest) in zip(uids, batches, strict=True):
        assert snapshot.position[snapshot.rows[uid], 1] == pytest.approx(rest, abs=0.03)


def test_density_gives_the_mass_where_mass_is_zero(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.0, 0.0)], size=(1.0, 1.0, 1.0), mass=0.0, density=500.0)
    )
    world.add_impulses(uids, (0.0, 500.0, 0.0))  # 500 N s on 500 kg: 1 m/s
    world.run_ticks(1)
    snapshot = world.latest()
    assert snapshot is not None
    assert snapshot.linear_velocity[0, 1] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize(
    ('batch', 'message'),
    [
        (BodyBatch(np.zeros((1, 3)), ShapeKind.BOX, (1.0, -1.0, 1.0, 0.0)), 'box'),
        (BodyBatch(np.zeros((1, 3)), ShapeKind.SPHERE, (0.0, 0, 0, 0)), 'sphere'),
        (BodyBatch(np.zeros((1, 3)), 99, (1.0, 1, 1, 0)), 'unknown shape'),
        (BodyBatch.plane(motion=MotionType.DYNAMIC), 'plane is static'),
        (BodyBatch.spheres([(0.0, 0.0, 0.0)], mass=0.0, density=0.0), 'mass or density'),
        (BodyBatch.spheres([(np.nan, 0.0, 0.0)]), 'not finite'),
        (BodyBatch.spheres([(0.0, 0.0, 0.0)], orientations=(0, 0, 0, 0)), 'zero quaternion'),
        (BodyBatch.spheres([(0.0, 0.0, 0.0)], friction=-1.0), 'not negative'),
        (BodyBatch.spheres([(0.0, 0.0, 0.0)], motion=7), 'motion type'),
        (BodyBatch.spheres([(0.0, 0.0, 0.0)], flags=1 << 20), 'flags'),
    ],
)
def test_a_bad_batch_is_refused_at_submission(
    make_world: MakeWorld, batch: BodyBatch, message: str
) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match=message):
        world.add_bodies(batch)
    assert world.submitted_seq == 0


def test_a_batch_past_max_bodies_is_refused(make_world: MakeWorld) -> None:
    world = make_world(threads=1, max_bodies=10)
    world.add_bodies(BodyBatch.spheres(grid(8)))
    with pytest.raises(ValueError, match='max_bodies'):
        world.add_bodies(BodyBatch.spheres(grid(3)))
    world.add_bodies(BodyBatch.spheres(grid(2)))


def test_mismatched_lengths_are_refused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match='shape'):
        world.set_velocities([1, 2], linear=np.zeros((3, 3)))
    with pytest.raises(ValueError, match='shape'):
        world.add_bodies(BodyBatch.spheres(np.zeros((2, 3)), radius=[0.5, 0.5, 0.5]))
