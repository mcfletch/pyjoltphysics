"""Jolt against omi_physics on three scenes, compared on behaviour.

Two solvers never agree to the bit, so each scene names what is compared and
the tolerance: rest heights to 1 cm, a stack's sideways drift under 5 cm and
its top height to 5 cm, and a bounce's rebound ratios to 0.05.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, World, WorldSettings

omi = pytest.importorskip('omi_physics.world', reason='omi_physics is the reference engine')
model = pytest.importorskip('omi_physics.model')

pytestmark = pytest.mark.differential

DT = 1.0 / 120.0
FRICTION = 0.6


def omi_ground(restitution: float = 0.0) -> tuple[Any, int]:
    world = omi.PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)), fixed_dt=DT)
    material = world.add_material(
        model.Material(staticFriction=FRICTION, dynamicFriction=FRICTION, restitution=restitution)
    )
    ground = world.add_shape(model.Shape.box((40, 1, 40)))
    world.add_body(
        model.Motion(type=model.STATIC),
        collider=model.Collider(shape=ground, physicsMaterial=material),
        position=(0, -0.5, 0),
    )
    return world, material


def omi_heights(restitution: float, ticks: int) -> np.ndarray:
    world, material = omi_ground(restitution)
    sphere = world.add_shape(model.Shape.sphere(0.5))
    body = world.add_body(
        model.Motion(type=model.DYNAMIC, mass=1.0),
        collider=model.Collider(shape=sphere, physicsMaterial=material),
        position=(0, 3, 0),
    )
    heights = []
    for _ in range(ticks):
        world.step(DT)
        heights.append(float(world.position[body][1]))
    return np.array(heights)


def jolt_heights(restitution: float, ticks: int) -> np.ndarray:
    with World(WorldSettings(threads=1)) as world:
        world.add_bodies(BodyBatch.plane(friction=FRICTION, restitution=restitution))
        (ball,), _ = world.add_bodies(
            BodyBatch.spheres(
                [(0.0, 3.0, 0.0)], radius=0.5, friction=FRICTION, restitution=restitution
            )
        )
        heights = []
        for _ in range(ticks):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            heights.append(float(snapshot.position[snapshot.rows[int(ball)], 1]))
    return np.array(heights)


def apexes(heights: np.ndarray) -> np.ndarray:
    inner = heights[1:-1]
    return inner[(inner > heights[:-2]) & (inner >= heights[2:])]


def test_a_sphere_dropped_on_a_plane_rests_at_its_radius() -> None:
    omi_rest = omi_heights(0.0, 600)[-1]
    jolt_rest = jolt_heights(0.0, 600)[-1]
    assert jolt_rest == pytest.approx(0.5, abs=0.01)
    assert jolt_rest == pytest.approx(omi_rest, abs=0.01)


def test_a_stack_of_ten_boxes_settles_upright() -> None:
    world, material = omi_ground()
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    bodies = [
        world.add_body(
            model.Motion(type=model.DYNAMIC, mass=1.0),
            collider=model.Collider(shape=box, physicsMaterial=material),
            position=(0, 0.5 + k * 1.01, 0),
        )
        for k in range(10)
    ]
    for _ in range(240):
        world.step(DT)
    omi_poses = np.array([world.position[b] for b in bodies])
    with World(WorldSettings(threads=1)) as jolt:
        jolt.add_bodies(BodyBatch.plane(friction=FRICTION))
        uids, _ = jolt.add_bodies(
            BodyBatch.boxes(
                [(0.0, 0.5 + k * 1.01, 0.0) for k in range(10)], size=(1, 1, 1), friction=FRICTION
            )
        )
        jolt.run_ticks(240)
        snapshot = jolt.latest()
        assert snapshot is not None
        jolt_poses = snapshot.position[snapshot.rows_of(uids)]
    for poses in (omi_poses, jolt_poses):
        assert np.abs(poses[:, [0, 2]]).max() < 0.05
    assert jolt_poses[-1, 1] == pytest.approx(omi_poses[-1, 1], abs=0.05)


@pytest.mark.parametrize('restitution', [0.5, 0.8])
def test_a_bounce_rebounds_as_high(restitution: float) -> None:
    rest = 0.5
    ratios = []
    for heights in (omi_heights(restitution, 600), jolt_heights(restitution, 600)):
        peaks = apexes(heights)[:2] - rest
        ratios.append((peaks[0] / (3.0 - rest), peaks[1] / peaks[0]))
    (omi_first, omi_second), (jolt_first, jolt_second) = ratios
    assert jolt_first == pytest.approx(omi_first, abs=0.05)
    assert jolt_second == pytest.approx(omi_second, abs=0.05)
