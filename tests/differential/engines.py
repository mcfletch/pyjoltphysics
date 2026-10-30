"""One scene description run through Jolt and through omi_physics, for the differential suite.

A scene is a list of :class:`Body`, each with a shape spec: ``('box', size)``,
``('sphere', r)``, ``('capsule', r, h)``, ``('cylinder', r, h)``, ``('hull',
points)``, ``('mesh', points, triangles)``, ``('heightfield', heights, origin,
spacing)`` or ``('compound', [(spec, position), ...])``. Both engines step it
at 120 Hz from the same initial conditions; :class:`Trace` holds each tick's
positions and orientations, and the trigger enters and exits by tick.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from omi_physics import model
from omi_physics import world as omi

from pyjoltphysics import BodyBatch, BodyFlag, MotionType, Shape, World, WorldSettings

DT = 1.0 / 120.0
G = 9.81
Spec = tuple[Any, ...]


@dataclasses.dataclass
class Body:
    """One body of a scene, in terms both engines take."""

    shape: Spec
    position: tuple[float, float, float]
    motion: str = 'dynamic'
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    sensor: bool = False
    continuous: bool = False


@dataclasses.dataclass
class Trace:
    """Per tick (ticks, n, 3) positions and (ticks, n, 4) orientations; trigger (tick, phase, a, b)."""

    position: np.ndarray
    orientation: np.ndarray
    triggers: list[tuple[int, str, int, int]]
    awake: np.ndarray  # (n,) at the last tick

    def last(self) -> np.ndarray:
        last: np.ndarray = self.position[-1]
        return last


# -- Jolt ----------------------------------------------------------------------


def jolt_shape(spec: Spec) -> Shape:
    kind = spec[0]
    if kind == 'box':
        return Shape.box(spec[1])
    if kind == 'sphere':
        return Shape.sphere(spec[1])
    if kind == 'capsule':
        return Shape.capsule(spec[1], spec[2])
    if kind == 'cylinder':
        return Shape.cylinder(spec[1], spec[2])
    if kind == 'hull':
        return Shape.convex(spec[1])
    if kind == 'mesh':
        return Shape.mesh(spec[1], spec[2])
    if kind == 'heightfield':
        return Shape.heightfield(spec[1], origin=spec[2], spacing=spec[3], bits_per_sample=16)
    if kind == 'compound':
        return Shape.compound([(jolt_shape(child), position) for child, position in spec[1]])
    raise ValueError(kind)


_MOTION = {
    'static': MotionType.STATIC,
    'kinematic': MotionType.KINEMATIC,
    'dynamic': MotionType.DYNAMIC,
}


def run_jolt(bodies: list[Body], ticks: int, threads: int = 1, **settings: Any) -> Trace:
    with World(WorldSettings(threads=threads, **settings)) as world:
        uids = []
        for body in bodies:
            flags = (BodyFlag.SENSOR if body.sensor else BodyFlag.NONE) | (
                BodyFlag.CONTINUOUS if body.continuous else BodyFlag.NONE
            )
            (uid,), _ = world.add_bodies(
                BodyBatch.of_shape(
                    [body.position],
                    jolt_shape(body.shape),
                    motion=_MOTION[body.motion],
                    mass=body.mass,
                    friction=body.friction,
                    restitution=body.restitution,
                    linear_velocity=body.velocity,
                    orientations=body.orientation,
                    flags=flags,
                )
            )
            uids.append(int(uid))
        index = {uid: k for k, uid in enumerate(uids)}
        positions, orientations, triggers = [], [], []
        for _ in range(ticks):
            world.run_ticks(1)
            snapshot = world.latest()
            assert snapshot is not None
            rows = snapshot.rows_of(uids)
            positions.append(snapshot.position[rows].copy())
            orientations.append(snapshot.orientation[rows].astype(np.float64))
            for event in world.drain_events():
                if event['kind'] == 1 and event['phase'] in (0, 2):
                    phase = 'enter' if event['phase'] == 0 else 'exit'
                    triggers.append(
                        (int(event['tick']), phase, index[int(event['a'])], index[int(event['b'])])
                    )
        snapshot = world.latest()
        assert snapshot is not None
        awake = snapshot.awake[snapshot.rows_of(uids)].copy()
    return Trace(np.array(positions), np.array(orientations), triggers, awake)


# -- omi_physics ---------------------------------------------------------------


def omi_shape(spec: Spec) -> Any:
    kind = spec[0]
    if kind == 'box':
        return model.Shape.box(tuple(spec[1]))
    if kind == 'sphere':
        return model.Shape.sphere(spec[1])
    if kind == 'capsule':
        return model.Shape.capsule(height=spec[2], radius=spec[1])
    if kind == 'cylinder':
        return model.Shape.cylinder(height=spec[2], radius=spec[1])
    if kind == 'hull':
        return model.Shape.convex(spec[1])
    if kind == 'mesh':
        return model.Shape.trimesh(spec[1], spec[2])
    if kind == 'heightfield':
        return model.Shape.heightfield(spec[1], spec[2], spec[3])
    if kind == 'compound':
        return model.Shape.compound([(omi_shape(child), position) for child, position in spec[1]])
    raise ValueError(kind)


def run_omi(bodies: list[Body], ticks: int) -> Trace:
    world = omi.PhysicsWorld(gravity=model.Gravity(gravity=G, direction=(0, -1, 0)), fixed_dt=DT)
    world.report_persist = False
    indices = []
    for body in bodies:
        material = world.add_material(
            model.Material(
                staticFriction=body.friction,
                dynamicFriction=body.friction,
                restitution=body.restitution,
            )
        )
        shape = world.add_shape(omi_shape(body.shape))
        motion = model.Motion(type=body.motion, mass=body.mass, linearVelocity=body.velocity)
        if body.sensor:
            index = world.add_body(
                motion,
                trigger=model.Trigger(shape=shape),
                position=body.position,
                orientation=body.orientation,
            )
        else:
            index = world.add_body(
                motion,
                collider=model.Collider(shape=shape, physicsMaterial=material),
                position=body.position,
                orientation=body.orientation,
            )
        indices.append(index)
    index_of = {index: k for k, index in enumerate(indices)}
    triggers: list[tuple[int, str, int, int]] = []
    tick = [0]

    def heard(phase: str, trigger: int, other: int) -> None:
        if phase in ('enter', 'exit'):
            triggers.append((tick[0], phase, index_of[trigger], index_of[other]))

    world.add_trigger_listener(heard)
    positions, orientations = [], []
    for t in range(ticks):
        tick[0] = t + 1
        world.step(DT)
        positions.append(np.array([world.position[i] for i in indices], dtype=np.float64))
        orientations.append(np.array([world.orientation[i] for i in indices], dtype=np.float64))
    awake = np.array([world.is_awake_mover(i) for i in indices])
    return Trace(np.array(positions), np.array(orientations), triggers, awake)


# -- scenery -------------------------------------------------------------------


def ground(friction: float = 0.5, restitution: float = 0.0, half: float = 20.0) -> Body:
    """A static box ground whose top face is y = 0."""
    return Body(
        ('box', (2 * half, 1.0, 2 * half)), (0.0, -0.5, 0.0), 'static', 0.0, friction, restitution
    )


def grid_mesh(width: float, cells: int, slope: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A square grid of triangles in x, z, rising ``slope`` metres a metre of x, facing up."""
    edge = np.linspace(-width / 2, width / 2, cells + 1)
    x, z = np.meshgrid(edge, edge, indexing='xy')
    points = np.stack([x.ravel(), (x * slope).ravel(), z.ravel()], axis=1)
    k = cells + 1
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j * k + i
            triangles += [(a, a + k, a + 1), (a + 1, a + k, a + k + 1)]
    return points, np.array(triangles, dtype=np.int64)


def mesh_ground(slope: float = 0.0, friction: float = 0.5, restitution: float = 0.0) -> Body:
    points, triangles = grid_mesh(40.0, 20, slope)
    return Body(('mesh', points, triangles), (0.0, 0.0, 0.0), 'static', 0.0, friction, restitution)


def heightfield_ground(slope: float = 0.0, friction: float = 0.5, restitution: float = 0.0) -> Body:
    samples = 41
    x = np.linspace(-20.0, 20.0, samples)
    heights = np.tile(x * slope, (samples, 1))
    return Body(
        ('heightfield', heights, (-20.0, 0.0, -20.0), 1.0),
        (0.0, 0.0, 0.0),
        'static',
        0.0,
        friction,
        restitution,
    )


def tetra_hull(size: float = 1.0) -> np.ndarray:
    """A regular-ish tetrahedron's points with its base at y = -size/4 and centroid at the origin."""
    p = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)], dtype=np.float64) * size
    return p - p.mean(axis=0)
