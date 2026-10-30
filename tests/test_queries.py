"""Queries on the caller's thread: rays, ray batches, shape sweeps and overlaps."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    MotionType,
    QueryFilter,
    RayHit,
    RayHits,
    Shape,
    World,
)
from pyjoltphysics.testing import MakeWorld, floor


def hit_uid(hit: RayHit | None) -> int:
    """The uid a ray met; the ray has to meet something."""
    assert hit is not None
    return hit.uid


def boxes_world(make_world: MakeWorld, **settings: object) -> tuple[World, int, np.ndarray]:
    """A floor and three 1 m static boxes on it at x = 0, 3 and 6; return the floor and box uids."""
    world = make_world(**settings)
    (ground,), _ = world.add_bodies(floor())
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 0.5, 0.0), (3.0, 0.5, 0.0), (6.0, 0.5, 0.0)], motion=MotionType.STATIC
        )
    )
    world.run_ticks(1)
    return world, int(ground), uids


def test_a_ray_down_hits_the_top_of_a_box(make_world: MakeWorld) -> None:
    world, _, uids = boxes_world(make_world)
    hit = world.raycast((3.0, 10.0, 0.0), (0.0, -1.0, 0.0), 100.0)
    assert hit is not None
    assert hit.uid == uids[1]
    assert hit.distance == pytest.approx(9.0, abs=1e-5)
    assert hit.fraction == pytest.approx(0.09, abs=1e-6)
    np.testing.assert_allclose(hit.point, (3.0, 1.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(hit.normal, (0.0, 1.0, 0.0), atol=1e-5)


def test_a_ray_that_reaches_nothing_returns_none(make_world: MakeWorld) -> None:
    world, _, _ = boxes_world(make_world)
    assert world.raycast((0.0, 10.0, 0.0), (0.0, 1.0, 0.0), 100.0) is None
    assert world.raycast((3.0, 10.0, 0.0), (0.0, -1.0, 0.0), 5.0) is None


def test_a_ray_direction_need_not_be_a_unit_vector(make_world: MakeWorld) -> None:
    world, _, _ = boxes_world(make_world)
    hit = world.raycast((3.0, 10.0, 0.0), (0.0, -7.0, 0.0), 100.0)
    assert hit is not None and hit.distance == pytest.approx(9.0, abs=1e-5)


def test_a_zero_direction_is_refused(make_world: MakeWorld) -> None:
    world, _, _ = boxes_world(make_world)
    with pytest.raises(ValueError, match='direction'):
        world.raycast((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 10.0)


def test_a_ray_batch_answers_every_ray_in_arrays(make_world: MakeWorld) -> None:
    world, ground, uids = boxes_world(make_world)
    origins = np.array([(0.0, 5.0, 0.0), (3.0, 5.0, 0.0), (1.5, 5.0, 0.0), (0.0, -5.0, 0.0)])
    directions = np.tile((0.0, -1.0, 0.0), (4, 1))
    hits = world.raycast_batch(origins, directions, 50.0)
    assert isinstance(hits, RayHits)
    assert len(hits) == 4
    np.testing.assert_array_equal(hits.hit, [True, True, True, False])
    np.testing.assert_array_equal(hits.uid, [uids[0], uids[1], ground, 0])
    np.testing.assert_allclose(hits.distance[:3], [4.0, 4.0, 5.0], atol=1e-5)
    assert np.isinf(hits.distance[3])
    np.testing.assert_allclose(hits.point[2], (1.5, 0.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(hits.normal[:3], np.tile((0.0, 1.0, 0.0), (3, 1)), atol=1e-5)
    assert hits.sub_shape.dtype == np.uint32
    assert hits.fraction.dtype == np.float32


def test_a_ray_into_a_mesh_names_the_triangle_it_hits(make_world: MakeWorld) -> None:
    world = make_world()
    points = np.array([(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)], np.float64)
    triangles = np.array([(0, 2, 1), (0, 3, 2)], np.uint32)
    (mesh,), _ = world.add_bodies(BodyBatch.meshes([(0.0, 0.0, 0.0)], points, triangles))
    world.run_ticks(1)
    hits = world.raycast_batch([(0.5, 1.0, -0.5), (-0.5, 1.0, 0.5)], [(0, -1, 0), (0, -1, 0)], 5.0)
    assert hits.hit.all() and (hits.uid == mesh).all()
    assert hits.sub_shape[0] != hits.sub_shape[1]


def test_a_filter_excludes_uids_and_masks_groups(make_world: MakeWorld) -> None:
    world = make_world()
    world.add_bodies(floor())
    (low,), _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.5, 0.0)], motion=MotionType.STATIC, group=2)
    )
    (high,), _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 2.5, 0.0)], motion=MotionType.STATIC, group=4)
    )
    world.run_ticks(1)
    down = ((0.0, 10.0, 0.0), (0.0, -1.0, 0.0), 20.0)
    assert hit_uid(world.raycast(*down)) == high
    assert hit_uid(world.raycast(*down, filter=QueryFilter(exclude=[high]))) == low
    assert hit_uid(world.raycast(*down, filter=QueryFilter(mask=2))) == low
    assert world.raycast(*down, filter=QueryFilter(mask=8)) is None


def test_sensors_are_seen_only_when_the_filter_asks_for_triggers(make_world: MakeWorld) -> None:
    world, _, uids = boxes_world(make_world)
    (zone,), _ = world.add_bodies(
        BodyBatch.boxes([(3.0, 3.0, 0.0)], motion=MotionType.STATIC, flags=BodyFlag.SENSOR)
    )
    world.run_ticks(1)
    ray = ((3.0, 10.0, 0.0), (0.0, -1.0, 0.0), 100.0)
    assert hit_uid(world.raycast(*ray)) == uids[1]
    assert hit_uid(world.raycast(*ray, filter=QueryFilter(triggers=True))) == zone
    assert set(world.overlap(Shape.box((1, 1, 1)), (3.0, 3.0, 0.0)).tolist()) == set()
    found = world.overlap(Shape.box((1, 1, 1)), (3.0, 3.0, 0.0), filter=QueryFilter(triggers=True))
    assert found.tolist() == [zone]


def test_a_sphere_swept_down_stops_on_the_box_top(make_world: MakeWorld) -> None:
    world, _, uids = boxes_world(make_world)
    hit = world.sweep_sphere(0.25, (3.0, 5.0, 0.0), (3.0, -5.0, 0.0))
    assert hit is not None
    assert hit.uid == uids[1]
    assert hit.fraction == pytest.approx((5.0 - 1.25) / 10.0, abs=1e-4)
    assert hit.distance == pytest.approx(3.75, abs=1e-3)
    np.testing.assert_allclose(hit.normal, (0.0, 1.0, 0.0), atol=1e-4)
    assert hit.point[1] == pytest.approx(1.0, abs=1e-3)


def test_a_sweep_with_all_hits_returns_every_body_in_order(make_world: MakeWorld) -> None:
    world, ground, uids = boxes_world(make_world)
    hits = world.sweep_sphere(0.2, (-2.0, 0.5, 0.0), (8.0, 0.5, 0.0), all_hits=True)
    assert [h.uid for h in hits] == uids.tolist()
    fractions = [h.fraction for h in hits]
    assert fractions == sorted(fractions)
    assert ground not in [h.uid for h in hits]
    first = world.sweep_sphere(0.2, (-2.0, 0.5, 0.0), (8.0, 0.5, 0.0))
    assert first is not None and first.uid == uids[0]


def test_capsule_and_box_sweeps(make_world: MakeWorld) -> None:
    world, _, uids = boxes_world(make_world)
    capsule = world.sweep_capsule(0.3, 1.2, (-3.0, 1.5, 0.0), (9.0, 1.5, 0.0))
    assert capsule is not None and capsule.uid == uids[0]
    assert capsule.distance == pytest.approx(3.0 - 0.5 - 0.3, abs=2e-3)
    box = world.sweep_box((0.5, 0.5, 0.5), (3.0, 6.0, 0.0), (3.0, 0.0, 0.0))
    assert box is not None and box.uid == uids[1]
    assert box.distance == pytest.approx(6.0 - 1.0 - 0.25, abs=2e-3)


def test_an_overlap_returns_the_uids_of_the_bodies_touching_the_volume(
    make_world: MakeWorld,
) -> None:
    world, ground, uids = boxes_world(make_world)
    found = world.overlap(Shape.sphere(2.0), (1.5, 0.5, 0.0))
    assert found.dtype == np.uint32
    assert sorted(found.tolist()) == sorted([ground, int(uids[0]), int(uids[1])])
    above = world.overlap(Shape.box((1.0, 1.0, 1.0)), (6.0, 3.0, 0.0))
    assert above.tolist() == []


def test_a_shape_whose_centre_of_mass_is_off_its_origin_overlaps_where_it_is_placed(
    make_world: MakeWorld,
) -> None:
    world, ground, uids = boxes_world(make_world)
    corner = Shape.convex([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    assert world.overlap(corner, (2.6, -0.2, -0.5)).tolist() == [ground, int(uids[1])]
    assert world.overlap(corner, (1.4, -0.2, -0.5)).tolist() == [ground]


def test_queries_see_a_body_from_the_tick_that_adds_it(make_world: MakeWorld) -> None:
    world = make_world()
    world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)], motion=MotionType.STATIC))
    assert world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0) is None
    world.run_ticks(1)
    assert world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0) is not None


def test_queries_from_many_threads_beside_a_free_running_world(make_world: MakeWorld) -> None:
    world = make_world(threads=4, tick_hz=240.0)
    world.add_bodies(floor())
    rng = np.random.default_rng(3)
    world.add_bodies(BodyBatch.boxes(rng.uniform(-10, 10, (500, 3)) + (0, 12, 0), size=(0.4,) * 3))
    world.start()
    errors: list[BaseException] = []
    counts = [0] * 6

    def hammer(k: int) -> None:
        try:
            origins = rng.uniform(-10, 10, (200, 3)) + (0, 30, 0)
            directions = np.tile((0.0, -1.0, 0.0), (200, 1))
            deadline = time.perf_counter() + 0.5
            while time.perf_counter() < deadline:
                hits = world.raycast_batch(origins, directions, 100.0)
                assert hits.hit.all()
                assert (hits.point[:, 1] >= -1e-3).all()
                world.overlap(Shape.sphere(1.0), (0.0, 0.5, 0.0))
                counts[k] += 1
        except BaseException as error:  # noqa: BLE001 reported by the main thread below
            errors.append(error)

    threads = [threading.Thread(target=hammer, args=(k,)) for k in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    world.stop()
    assert not errors, errors[0]
    assert min(counts) > 3
    assert world.tick > 20


def test_a_query_inside_a_world_callback_runs_without_the_lock(make_world: MakeWorld) -> None:
    world, _, uids = boxes_world(make_world)
    seen: list[int] = []

    def look(tick: int, time: float) -> None:
        hit = world.raycast((6.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)
        seen.append(hit.uid if hit else 0)

    world.set_before_step(look)
    world.set_after_step(look)
    world.run_ticks(3)
    world.set_before_step(None)
    world.set_after_step(None)
    assert seen == [int(uids[2])] * 6


def test_a_hundred_thousand_rays_take_well_under_a_second(make_world: MakeWorld) -> None:
    world, _, _ = boxes_world(make_world)
    rng = np.random.default_rng(1)
    origins = rng.uniform(-50, 50, (100_000, 3))
    origins[:, 1] = 20.0
    directions = np.tile((0.0, -1.0, 0.0), (100_000, 1))
    world.raycast_batch(origins[:1000], directions[:1000], 50.0)
    start = time.perf_counter()
    hits = world.raycast_batch(origins, directions, 50.0)
    elapsed = time.perf_counter() - start
    assert hits.hit.all()
    assert elapsed < 0.5
