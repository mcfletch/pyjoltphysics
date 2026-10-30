"""Shapes beyond the primitives: hulls, meshes, height fields, compounds, decorations, sharing."""

from __future__ import annotations

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    MotionType,
    Shape,
    ShapeKind,
    World,
    heightfield_side,
    heightfield_tiles,
)
from pyjoltphysics.testing import MakeWorld, floor


def rest_heights(world: World, uids: np.ndarray, ticks: int = 360) -> np.ndarray:
    world.run_ticks(ticks)
    snapshot = world.latest()
    assert snapshot is not None
    heights: np.ndarray = snapshot.position[snapshot.rows_of(uids), 1]
    return heights


def cube_points(half: float = 0.5) -> np.ndarray:
    return np.array(
        [(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)]
    )


def ramp(width: float = 20.0, slope: float = 0.0, cells: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return a square grid mesh in x, z of ``cells`` squares a side, rising ``slope`` m per m of x."""
    edge = np.linspace(-width / 2, width / 2, cells + 1)
    x, z = np.meshgrid(edge, edge, indexing='xy')
    points = np.stack([x.ravel(), (x * slope).ravel(), z.ravel()], axis=1)
    k = cells + 1
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j * k + i
            triangles += [(a, a + k, a + 1), (a + 1, a + k, a + k + 1)]
    return points, np.array(triangles, dtype=np.uint32)


# -- convex hulls --------------------------------------------------------------


def test_a_hull_rests_on_its_lowest_points(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    tetra = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)], dtype=float) - 0.25
    uids, _ = world.add_bodies(BodyBatch.hulls([(0.0, 2.0, 0.0)], tetra, mass=1.0))
    # The origin sits 0.25 m above the tetrahedron's base.
    assert rest_heights(world, uids)[0] == pytest.approx(0.25, abs=0.01)


def test_hulls_take_points_per_body_and_mass_from_density(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    points = np.stack([cube_points(0.25), cube_points(0.5)])  # (2, 8, 3)
    uids, _ = world.add_bodies(
        BodyBatch.hulls([(-2.0, 2.0, 0.0), (2.0, 2.0, 0.0)], points, mass=0.0, density=1000.0)
    )
    np.testing.assert_allclose(rest_heights(world, uids), (0.25, 0.5), atol=0.01)
    assert world.stats()['shapes'] == 2


@pytest.mark.parametrize(
    ('points', 'reason'),
    [
        (np.zeros((3, 3)), 'at least 4 points'),
        (np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2)], float), 'no volume'),
        (np.array([(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)], float), 'no volume'),
        (np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, np.nan)]), 'finite'),
    ],
)
def test_a_degenerate_hull_is_refused(
    make_world: MakeWorld, points: np.ndarray, reason: str
) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match=reason):
        world.add_shape(Shape.convex(points))


# -- meshes --------------------------------------------------------------------


def test_a_sphere_rests_on_a_static_mesh(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    points, triangles = ramp()
    world.add_bodies(BodyBatch.meshes([(0.0, 0.0, 0.0)], points, triangles))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.3, 2.0, 0.7)], radius=0.5))
    assert rest_heights(world, uids)[0] == pytest.approx(0.5, abs=0.01)


def test_a_dynamic_mesh_is_refused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    points, triangles = ramp()
    mesh = world.add_shape(Shape.mesh(points, triangles))
    with pytest.raises(ValueError, match='static or kinematic body only'):
        world.add_bodies(BodyBatch.of_shape([(0.0, 0.0, 0.0)], mesh, mass=1.0))


def test_a_kinematic_mesh_carries_what_rests_on_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    points, triangles = ramp(width=4.0, cells=2)
    (deck,), _ = world.add_bodies(
        BodyBatch.meshes([(0.0, 0.0, 0.0)], points, triangles, motion=MotionType.KINEMATIC)
    )
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.6, 0.0)], size=(0.5, 0.5, 0.5)))
    world.run_ticks(60)
    world.set_velocities([deck], linear=(0.0, 1.0, 0.0))
    world.run_ticks(60)
    snapshot = world.latest()
    assert snapshot is not None
    # Resting at 0.25 m, then carried up 0.5 m.
    assert snapshot.position[snapshot.rows[int(box)], 1] == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize(
    ('triangles', 'reason'),
    [
        (np.array([(0, 1, 9)], np.uint32), 'names vertex 9'),
        (np.array([(0, 0, 1)], np.uint32), 'degenerate'),
        (np.zeros((0, 3), np.uint32), 'at least one triangle'),
    ],
)
def test_a_bad_mesh_is_refused(make_world: MakeWorld, triangles: np.ndarray, reason: str) -> None:
    world = make_world(threads=1)
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1)], float)
    with pytest.raises(ValueError, match=reason):
        world.add_shape(Shape.mesh(points, triangles))


# -- height fields -------------------------------------------------------------


def test_a_rectangular_heightfield_holds_spheres_at_its_heights(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    rows, columns = 5, 9  # 4 m along z, 16 m along x at 2 m spacing
    heights = np.tile(np.linspace(0.0, 1.6, columns), (rows, 1)).astype(np.float32)
    world.add_bodies(
        BodyBatch.heightfield(heights, origin=(-8.0, 0.0, -2.0), spacing=(2.0, 1.0), friction=1.0)
    )
    xs = np.array([-6.0, 0.0, 6.0])
    uids, _ = world.add_bodies(
        BodyBatch.spheres(np.stack([xs, np.full(3, 3.0), np.zeros(3)], axis=1), radius=0.25)
    )
    world.run_ticks(120)
    snapshot = world.latest()
    assert snapshot is not None
    rows_ = snapshot.rows_of(uids)
    # On a slope of 0.1 the sphere's centre is r / cos(atan 0.1) above the surface.
    ground = 0.1 * (snapshot.position[rows_, 0] + 8.0)
    lift = 0.25 * np.sqrt(1 + 0.1**2)
    np.testing.assert_allclose(snapshot.position[rows_, 1], ground + lift, atol=0.01)


def test_heightfield_holes_let_bodies_through(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    heights = np.zeros((9, 9), np.float32)
    heights[4, 4] = np.nan
    mask = np.zeros((9, 9), bool)
    mask[4, 1] = True
    world.add_bodies(BodyBatch.heightfield(heights, origin=(-4.0, 0.0, -4.0), holes=mask))
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(0.0, 1.0, 0.0), (-3.0, 1.0, 0.0), (3.0, 1.0, 0.0)], radius=0.2)
    )
    world.run_ticks(120)
    snapshot = world.latest()
    assert snapshot is not None
    y = snapshot.position[snapshot.rows_of(uids), 1]
    assert y[0] < -1.0 and y[1] < -1.0
    assert y[2] == pytest.approx(0.2, abs=0.01)


def test_heightfield_side_rounds_to_square_blocks() -> None:
    assert heightfield_side(5, 9, 4) == 12
    assert heightfield_side(2, 2, 2) == 4
    assert heightfield_side(257, 129, 4) == 260


@pytest.mark.parametrize(
    ('heights', 'named', 'reason'),
    [
        (np.zeros((1, 5)), {}, 'at least 2 rows'),
        (np.full((4, 4), np.nan), {}, 'every sample'),
        (np.full((4, 4), np.inf), {}, 'finite'),
        (np.zeros((4, 4)), {'block_size': 9}, 'block size'),
        (np.zeros((4, 4)), {'spacing': 0.0}, 'spacing'),
        (np.zeros((4, 4)), {'bits_per_sample': 0}, 'bits per sample'),
    ],
)
def test_a_bad_heightfield_is_refused(
    make_world: MakeWorld, heights: np.ndarray, named: dict[str, object], reason: str
) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match=reason):
        world.add_shape(Shape.heightfield(heights, **named))  # type: ignore[arg-type]  # each case names its own keywords


def test_a_dynamic_heightfield_is_refused(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match='static or kinematic'):
        world.add_bodies(BodyBatch.heightfield(np.zeros((4, 4)), motion=MotionType.DYNAMIC))


def test_heightfield_tiles_share_their_edge_samples() -> None:
    rng = np.random.default_rng(3)
    heights = rng.uniform(0.0, 2.0, (11, 21)).astype(np.float32)
    batch = heightfield_tiles(heights, origin=(100.0, 5.0, -50.0), spacing=2.0, tile=8)
    # 20 cells by 10 cells in tiles of 8: 3 by 2 tiles.
    assert len(batch) == 6
    positions = np.asarray(batch.positions)
    np.testing.assert_allclose(positions[0], (100.0, 5.0, -50.0))
    np.testing.assert_allclose(positions[1], (116.0, 5.0, -50.0))
    shapes = batch.shape
    assert isinstance(shapes, list)
    first, second = shapes[0], shapes[1]
    assert isinstance(first, Shape) and isinstance(second, Shape)
    assert first.heights is not None and second.heights is not None
    np.testing.assert_array_equal(first.heights[:, -1], second.heights[:, 0])


def test_a_sphere_rolls_across_tile_seams_as_across_one_field(make_world: MakeWorld) -> None:
    traces = []
    for tile in (32, 4):
        world = make_world(threads=1)
        heights = np.zeros((9, 33), np.float32)
        world.add_bodies(
            heightfield_tiles(heights, origin=(-16.0, 0.0, -4.0), spacing=1.0, tile=tile)
        )
        (ball,), _ = world.add_bodies(
            BodyBatch.spheres([(-14.0, 0.25, 0.0)], radius=0.25, linear_velocity=(6.0, 0.0, 0.0))
        )
        trace = []
        for _ in range(240):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            trace.append(snapshot.position[snapshot.rows[int(ball)]])
        traces.append(np.array(trace))
    one, tiled = traces
    # Seams lie every 4 m from x = -12; the ball crosses at least two.
    assert tiled[-1, 0] > -7.0
    # A ball rolling at 6 m/s over Jolt's triangles rides up to about 5 mm off
    # them at cell edges, on one field as on a mesh; the seams add no more.
    lift_one = np.abs(one[:, 1] - 0.25).max()
    lift_tiled = np.abs(tiled[:, 1] - 0.25).max()
    assert lift_one < 0.01
    assert lift_tiled <= lift_one + 0.001


# -- compounds -----------------------------------------------------------------


def test_a_compound_rests_on_its_lowest_child(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    dumbbell = Shape.compound(
        [
            (Shape.sphere(0.5), (-1.0, 0.0, 0.0)),
            (Shape.box((2.0, 0.2, 0.2)), (0.0, 0.0, 0.0)),
            (Shape.sphere(0.5), (1.0, 0.0, 0.0)),
        ]
    )
    uids, _ = world.add_bodies(BodyBatch.of_shape([(0.0, 3.0, 0.0)], dumbbell, mass=2.0))
    world.run_ticks(480)
    snapshot = world.latest()
    assert snapshot is not None
    row = snapshot.rows[int(uids[0])]
    assert snapshot.position[row, 1] == pytest.approx(0.5, abs=0.01)
    assert not snapshot.awake[row]


def test_a_mutable_compound_behaves_as_a_static_one(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    parts = [(Shape.box((1.0, 1.0, 1.0)), (0.0, 0.0, 0.0)), (Shape.box((1, 1, 1)), (0.0, 1.0, 0.0))]
    fixed, changing = Shape.compound(parts), Shape.compound(parts, mutable=True)
    uids, _ = world.add_bodies(
        BodyBatch(
            [(-3.0, 2.0, 0.0), (3.0, 2.0, 0.0)],
            shape=[fixed, changing],
            mass=1.0,
        )
    )
    np.testing.assert_allclose(rest_heights(world, uids), 0.5, atol=0.01)
    assert world.shape_info(world.add_shape(changing)).kind == ShapeKind.COMPOUND


def test_a_compound_refuses_a_heightfield_and_a_dynamic_mesh(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    land = Shape.heightfield(np.zeros((4, 4)))
    with pytest.raises(ValueError, match='height field'):
        world.add_shape(Shape.compound([(land, (0.0, 0.0, 0.0))]))
    points, triangles = ramp(cells=2)
    holder = Shape.compound([(Shape.mesh(points, triangles), (0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match='static or kinematic'):
        world.add_bodies(BodyBatch.of_shape([(0.0, 0.0, 0.0)], holder, mass=1.0))


# -- decorations ---------------------------------------------------------------


def test_scale_offset_and_centre_of_mass_are_body_parameters(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            [(-3.0, 3.0, 0.0), (0.0, 3.0, 0.0)],
            size=(1.0, 1.0, 1.0),
            scale=[(1.0, 2.0, 1.0), (1.0, 1.0, 1.0)],
            shape_offset=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
    )
    # Twice as tall: the origin is 1 m up. Placed 1 m above the origin: it rests at -0.5 + 1.
    np.testing.assert_allclose(rest_heights(world, uids), (1.0, -0.5), atol=0.01)


def test_a_low_centre_of_mass_rights_a_tilted_body(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    tilt = (0.0, 0.0, np.sin(0.35), np.cos(0.35))  # 40 degrees about z
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            [(-3.0, 1.2, 0.0), (3.0, 1.2, 0.0)],
            size=(0.6, 2.0, 0.6),
            orientations=tilt,
            center_of_mass=[(0.0, -0.9, 0.0), (0.0, 0.0, 0.0)],
        )
    )
    world.run_ticks(600)
    snapshot = world.latest()
    assert snapshot is not None
    upright = np.abs(snapshot.orientation[snapshot.rows_of(uids), 3])
    assert upright[0] > 0.999  # stood back up
    assert upright[1] < 0.9  # fell over


def test_a_sphere_refuses_a_non_uniform_scale(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    with pytest.raises(ValueError, match='cannot take the scale'):
        world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0)], scale=(1.0, 2.0, 1.0)))
    with pytest.raises(ValueError, match='not zero'):
        world.add_bodies(BodyBatch.boxes([(0.0, 1.0, 0.0)], scale=(1.0, 0.0, 1.0)))


# -- sharing -------------------------------------------------------------------


def test_ten_thousand_posts_share_one_registered_shape(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    post = world.add_shape(Shape.convex(cube_points(0.2) * (1.0, 5.0, 1.0)))
    rng = np.random.default_rng(1)
    positions = rng.uniform(-200.0, 200.0, (10_000, 3)) * (1.0, 0.0, 1.0) + (0.0, 1.0, 0.0)
    world.add_bodies(BodyBatch.of_shape(positions, post, motion=MotionType.STATIC))
    world.run_ticks(1)
    stats = world.stats()
    assert stats['bodies'] == 10_000
    assert stats['shapes'] == 1
    info = world.shape_info(post)
    assert info.volume == pytest.approx(0.4 * 2.0 * 0.4, rel=0.05)
    np.testing.assert_allclose(info.bounds_max, (0.2, 1.0, 0.2), atol=1e-3)
    # A Shape object given to add_bodies is registered once per world.
    hull = Shape.convex(cube_points(0.3))
    world.add_bodies(BodyBatch.of_shape([(0.0, 5.0, 0.0)], hull))
    world.add_bodies(BodyBatch.of_shape([(1.0, 5.0, 0.0)], hull))
    assert world.stats()['shapes'] == 2
    assert world.add_shape(hull) == world.add_shape(hull)


def test_a_removed_shape_id_names_nothing_but_its_bodies_stay(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    box = world.add_shape(Shape.box((1.0, 1.0, 1.0)))
    uids, _ = world.add_bodies(BodyBatch.of_shape([(0.0, 2.0, 0.0)], box))
    assert world.remove_shape(box)
    assert not world.remove_shape(box)
    with pytest.raises(ValueError, match='names no shape'):
        world.add_bodies(BodyBatch.of_shape([(0.0, 2.0, 0.0)], box))
    assert rest_heights(world, uids)[0] == pytest.approx(0.5, abs=0.01)
