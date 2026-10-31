"""Soft bodies: cloth, ropes and pressure balls through Jolt's soft-body simulation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pyjoltphysics import (
    BodyBatch,
    LongRange,
    MotionType,
    Snapshot,
    SoftBody,
    SoftBodySettings,
    Track,
)
from pyjoltphysics.testing import MakeWorld, floor, latest

HZ = 120.0


def _vertices(snapshot: Snapshot, body: SoftBody) -> npt.NDArray[np.float64]:
    rows = snapshot.soft_bodies
    i = list(rows.uid).index(body.uid)
    out: npt.NDArray[np.float64] = rows.vertices(i) + rows.origin[i]
    return out


def test_cloth_grid_layout() -> None:
    cloth = SoftBodySettings.cloth(width=2.0, depth=1.0, columns=5, rows=3, mass=1.5)
    assert cloth.positions.shape == (15, 3)
    assert cloth.faces is not None and cloth.faces.shape == (2 * 4 * 2, 3)
    assert np.allclose(cloth.positions[:, 1], 0.0)
    assert cloth.positions[:, 0].min() == pytest.approx(-1.0)
    assert cloth.positions[:, 2].max() == pytest.approx(0.5)
    assert cloth.vertex_masses().sum() == pytest.approx(1.5)
    assert cloth.corner_indices() == (0, 4, 10, 14)


def test_rope_and_ball_layout() -> None:
    rope = SoftBodySettings.rope([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], segments=10, mass=0.5)
    assert rope.positions.shape == (11, 3)
    assert rope.faces is None
    assert rope.edges is not None and len(rope.edges) == 10
    stiff = SoftBodySettings.rope(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], segments=10, bend_compliance=1e-4
    )
    assert stiff.edges is not None and len(stiff.edges) == 10 + 9
    ball = SoftBodySettings.ball(radius=0.5, subdivisions=2, pressure=1000.0)
    radii = np.linalg.norm(ball.positions, axis=1)
    assert np.allclose(radii, 0.5)
    assert ball.faces is not None and len(ball.faces) == 20 * 4**2


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError, match='face'):
        SoftBodySettings(np.zeros((3, 3)), faces=np.array([[0, 1, 5]]))
    with pytest.raises(ValueError, match='mass'):
        SoftBodySettings.cloth(1.0, 1.0, 3, 3, mass=-1.0)
    with pytest.raises(ValueError, match='vertices'):
        SoftBodySettings(np.zeros((0, 3)))


def test_cloth_pinned_at_two_corners_hangs(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2)
    # Damped so that its swing dies out within the run.
    cloth = SoftBodySettings.cloth(1.0, 1.0, columns=11, rows=11, mass=0.5, linear_damping=2.0)
    a, b, _, _ = cloth.corner_indices()
    pins = cloth.positions[[a, b]] + (0.0, 2.0, 0.0)
    body = world.add_soft_body(cloth, position=(0.0, 2.0, 0.0), pins=([a, b], pins))
    world.run_ticks(int(6 * HZ))
    snap = latest(world)
    vertices = _vertices(snap, body)
    assert np.abs(vertices[[a, b]] - pins).max() < 1e-4
    # The free edge hangs about a cloth's depth below the pinned one.
    assert vertices[:, 1].min() < 2.0 - 0.8
    assert vertices[:, 1].max() < 2.0 + 1e-3
    # Edges stretch by little: the free corners are within 10% of the cloth's depth from their pins.
    assert np.linalg.norm(vertices[-1] - vertices[b]) < 1.1
    rows = snap.soft_bodies
    i = list(rows.uid).index(body.uid)
    normals = rows.normals_of(i)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)


def test_cloth_drapes_over_a_sphere(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.spheres([(0.0, 0.5, 0.0)], radius=0.5, motion=MotionType.STATIC))
    radius = 0.01
    cloth = SoftBodySettings.cloth(
        2.0, 2.0, columns=21, rows=21, mass=1.0, vertex_radius=radius, friction=0.6
    )
    body = world.add_soft_body(cloth, position=(0.0, 1.3, 0.0))
    world.run_ticks(int(3 * HZ))
    vertices = _vertices(latest(world), body)
    distance = np.linalg.norm(vertices - (0.0, 0.5, 0.0), axis=1)
    assert distance.min() > 0.5 - radius
    centre = vertices[len(vertices) // 2]
    assert centre[1] == pytest.approx(1.0 + radius, abs=0.03)
    assert vertices[:, 1].min() < 0.3


def test_pressure_ball_bounces(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2)
    world.add_bodies(floor())
    ball = SoftBodySettings.ball(
        radius=0.4, subdivisions=2, pressure=4000.0, mass=1.0, restitution=0.0
    )
    rest_volume = ball.volume()
    body = world.add_soft_body(ball, position=(0.0, 2.0, 0.0))
    heights = []
    bottoms = []
    volumes = []
    for _ in range(int(2.5 * HZ)):
        world.run_ticks(1)
        snap = latest(world)
        vertices = _vertices(snap, body)
        heights.append(vertices[:, 1].mean())
        bottoms.append(vertices[:, 1].min())
        volumes.append(ball.volume(vertices - vertices.mean(axis=0)))
    heights_ = np.array(heights)
    lowest = int(np.argmin(heights_[: int(1.2 * HZ)]))
    assert bottoms[lowest] < 0.02
    # Back up off the floor after the first contact.
    assert heights_[lowest:].max() > heights_[lowest] + 0.2
    assert min(bottoms[lowest + int(0.2 * HZ) : lowest + int(0.5 * HZ)]) > 0.05
    assert min(volumes) > 0.6 * rest_volume


def test_rope_swings(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    rope = SoftBodySettings.rope([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], segments=20, mass=0.2)
    body = world.add_soft_body(rope, position=(0.0, 2.0, 0.0), pins=([0], [(0.0, 2.0, 0.0)]))
    ends = []
    for _ in range(int(3 * HZ)):
        world.run_ticks(1)
        ends.append(_vertices(latest(world), body)[-1])
    path = np.array(ends)
    assert path[:, 1].min() < 2.0 - 0.9
    # It swings through to the other side of the pin and back again.
    left = int(np.argmax(path[:, 0] < -0.5))
    assert left > 0
    assert path[left:, 0].max() > 0.5
    lengths = np.linalg.norm(np.diff(_vertices(latest(world), body), axis=0), axis=1).sum()
    assert lengths == pytest.approx(1.0, rel=0.05)


def test_cloth_holds_a_dropped_box(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=2)
    cloth = SoftBodySettings.cloth(2.0, 2.0, columns=15, rows=15, mass=1.0, vertex_radius=0.01)
    corners = list(cloth.corner_indices())
    pins = cloth.positions[corners] + (0.0, 1.0, 0.0)
    world.add_soft_body(cloth, position=(0.0, 1.0, 0.0), pins=(corners, pins))
    uids, _ = world.add_bodies(BodyBatch.boxes([(0.0, 2.0, 0.0)], size=(0.3, 0.3, 0.3), mass=0.5))
    world.run_ticks(int(3 * HZ))
    snap = latest(world)
    box = snap.position[snap.rows[int(uids[0])]]
    assert 0.3 < box[1] < 1.0


def test_wind_pushes_a_hanging_cloth(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    cloth = SoftBodySettings.cloth(1.0, 1.0, columns=9, rows=9, mass=0.3)
    a, b, _, _ = cloth.corner_indices()
    still = world.add_soft_body(
        cloth, position=(0.0, 2.0, 0.0), pins=([a, b], cloth.positions[[a, b]] + (0, 2, 0))
    )
    blown = world.add_soft_body(
        cloth, position=(3.0, 2.0, 0.0), pins=([a, b], cloth.positions[[a, b]] + (3, 2, 0))
    )
    world.set_wind([blown.uid], [(0.0, 0.0, 2.0)])
    world.run_ticks(int(3 * HZ))
    snap = latest(world)
    calm = _vertices(snap, still)
    windy = _vertices(snap, blown)
    assert windy[:, 2].mean() > calm[:, 2].mean() + 0.1


def test_pin_follows_a_moving_body(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 2.0, 0.0)], size=(0.1, 0.1, 0.1), motion=MotionType.KINEMATIC)
    )
    carrier = int(uids[0])
    world.set_track(
        [carrier],
        Track(
            times=np.array([0.0, 2.0]),
            positions=np.array([(0.0, 2.0, 0.0), (2.0, 2.0, 0.0)]),
        ),
    )
    rope = SoftBodySettings.rope([(0.0, 0.0, 0.0), (0.0, -1.0, 0.0)], segments=10, mass=0.1)
    body = world.add_soft_body(rope, position=(0.0, 2.0, 0.0))
    body.pin([0], [(0.0, 0.0, 0.0)], bodies=[carrier])
    world.run_ticks(int(1.5 * HZ))
    snap = latest(world)
    tip = _vertices(snap, body)[0]
    carried = snap.position[snap.rows[carrier]]
    assert np.abs(tip - carried).max() < 1e-3
    body.unpin()
    world.run_ticks(int(1.0 * HZ))
    assert _vertices(latest(world), body)[0, 1] < 1.5


def test_snapshot_vertices_and_versions(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1, time_before_sleep=0.2)
    world.add_bodies(floor())
    cloth = SoftBodySettings.cloth(1.0, 1.0, columns=5, rows=5, mass=0.2)
    body = world.add_soft_body(cloth, position=(0.0, 0.5, 0.0))
    world.run_ticks(1)
    first = latest(world)
    rows = first.soft_bodies
    assert len(rows) == 1 and rows.uid[0] == body.uid
    assert int(rows.vertex_count[0]) == 25
    assert rows.version[0] == first.tick
    # The body's own row carries the soft body's origin.
    assert np.allclose(first.position[first.rows[body.uid]], rows.origin[0])
    held = rows.vertices(0).copy()
    world.run_ticks(int(4 * HZ))
    settled = latest(world)
    srows = settled.soft_bodies
    assert not settled.awake[settled.rows[body.uid]]
    asleep_at = int(srows.version[0])
    assert asleep_at < settled.tick
    # The first snapshot still holds what it held: a buffer a reader holds is never written.
    assert np.array_equal(first.soft_bodies.vertices(0), held)
    world.run_ticks(5)
    assert latest(world).soft_bodies.version[0] == asleep_at
    world.activate([body.uid])
    world.set_wind([body.uid], [(1.0, 0.0, 0.0)])
    world.run_ticks(2)
    assert latest(world).soft_bodies.version[0] > asleep_at


def test_soft_bodies_add_and_remove_at_run_time(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor())
    rope = SoftBodySettings.rope([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], segments=4)
    cloth = SoftBodySettings.cloth(1.0, 1.0, 4, 4)
    a = world.add_soft_body(rope, position=(0.0, 1.0, 0.0))
    world.run_ticks(3)
    b = world.add_soft_body(cloth, position=(0.0, 2.0, 0.0))
    world.run_ticks(3)
    rows = latest(world).soft_bodies
    assert list(rows.uid) == [a.uid, b.uid]
    assert list(rows.vertex_count) == [5, 16]
    assert list(rows.first_vertex) == [0, 5]
    assert world.stats()['soft_bodies'] == 2
    world.remove_bodies([a.uid])
    world.run_ticks(1)
    snap = latest(world)
    rows = snap.soft_bodies
    assert list(rows.uid) == [b.uid]
    assert list(rows.first_vertex) == [0]
    assert snap.rows_of([a.uid])[0] == -1
    assert world.stats()['soft_bodies'] == 1


def test_soft_body_honours_collision_groups(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    world.add_bodies(floor(group=2))
    cloth = SoftBodySettings.cloth(1.0, 1.0, 5, 5, mask=1)
    body = world.add_soft_body(cloth, position=(0.0, 0.5, 0.0))
    world.run_ticks(int(1.0 * HZ))
    assert _vertices(latest(world), body)[:, 1].max() < -1.0


def test_settings_round_trip_as_dicts() -> None:
    cloth = SoftBodySettings.cloth(1.0, 1.0, 4, 3, mass=0.7, pressure=0.0, bend_compliance=1e-3)
    again = SoftBodySettings.from_dict(cloth.as_dict())
    assert np.array_equal(again.positions, cloth.positions)
    assert again.bend_compliance == cloth.bend_compliance


def test_massless_vertices_stay_where_they_are_placed(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    cloth = SoftBodySettings.cloth(1.0, 1.0, columns=6, rows=6, mass=0.4)
    masses = cloth.vertex_masses().copy()
    masses[:6] = 0.0  # the -z edge
    held = SoftBodySettings.cloth(1.0, 1.0, columns=6, rows=6, masses=masses, linear_damping=2.0)
    body = world.add_soft_body(held, position=(0.0, 2.0, 0.0))
    world.run_ticks(int(3 * HZ))
    vertices = _vertices(latest(world), body)
    assert np.abs(vertices[:6] - (held.positions[:6] + (0.0, 2.0, 0.0))).max() < 1e-5
    assert vertices[6:, 1].min() < 1.3


def test_long_range_attachments_limit_the_stretch(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=HZ, threads=1)
    reach = {}
    for x, long_range in ((0.0, LongRange.NONE), (3.0, LongRange.GEODESIC)):
        cloth = SoftBodySettings.cloth(
            1.0,
            1.0,
            columns=11,
            rows=11,
            mass=2.0,
            compliance=1e-2,
            linear_damping=2.0,
            long_range=long_range,
        )
        a, b, c, _ = cloth.corner_indices()
        pins = cloth.positions[[a, b]] + (x, 2.0, 0.0)
        body = world.add_soft_body(cloth, position=(x, 2.0, 0.0), pins=([a, b], pins))
        reach[long_range] = (body, c, pins[0])
    world.run_ticks(int(4 * HZ))
    snap = latest(world)
    stretched = {
        mode: float(np.linalg.norm(_vertices(snap, body)[c] - pin))
        for mode, (body, c, pin) in reach.items()
    }
    # Soft edges let the free corner sag past the cloth's depth; the attachments hold it near 1 m.
    assert stretched[LongRange.NONE] > 1.05
    assert stretched[LongRange.GEODESIC] < 1.005
