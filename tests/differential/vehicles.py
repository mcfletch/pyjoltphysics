"""glisteel's car run through omi_physics' RaycastVehicle and through a Jolt vehicle.

The car is glisteel's ``CarSpec``: 1180 kg, a 1.85 x 0.62 x 4.2 m chassis box,
four 0.33 m wheels on a 2.55 m wheelbase and a 1.70 m track, rear drive, and
the tuning of an electric supercar. Both engines step at 120 Hz. The Jolt car
is :meth:`VehicleSettings.from_raycast` of the same tuning, so the suite
measures that mapping. A scenario is a ground and a schedule of inputs by time;
:class:`Drive` holds the chassis's state each tick.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from omi_physics import model
from omi_physics import vehicle as omi_vehicle
from omi_physics import world as omi

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    MotionType,
    VehicleSettings,
    World,
    WorldSettings,
)

DT = 1.0 / 120.0
MASS = 1180.0
CHASSIS = (1.85, 0.62, 4.2)
#: Where glisteel hangs the wheels: half the body down, the ride height and the mass drop up.
WHEEL_HEIGHT = -0.31 + 0.14 + 0.15
Inputs = Callable[[float], tuple[float, float, float]]


def tuning() -> Any:
    """glisteel's ``CarSpec.tuning``."""
    return omi_vehicle.VehicleTuning(
        engine_force=20000.0,
        brake_force=16000.0,
        maximum_steer=0.52,
        steer_speed=3.2,
        rolling_resistance=0.02,
        downforce=6.0,
        steer_falloff_speed=10.0,
        base_speed=21.0,
        drag=0.9,
    )


def wheels() -> list[Any]:
    """glisteel's ``CarSpec.wheels()``."""
    return omi_vehicle.car_wheels(
        wheelbase=2.55,
        track=1.70,
        height=WHEEL_HEIGHT,
        drive='rear',
        radius=0.33,
        suspension_travel=0.22,
        suspension_stiffness=26.0,
        suspension_damping=0.55,
        grip=1.9,
    )


def settings(**changes: Any) -> VehicleSettings:
    """The Jolt car: the mapping of glisteel's tuning and wheels."""
    return VehicleSettings.from_raycast(tuning(), wheels(), MASS, **changes)


@dataclasses.dataclass(frozen=True)
class Piece:
    """A static box of the ground: centre, size, and a rotation about X then Z (rad)."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    tilt_x: float = 0.0
    tilt_z: float = 0.0

    def orientation(self) -> tuple[float, float, float, float]:
        qx = (math.sin(self.tilt_x / 2), 0.0, 0.0, math.cos(self.tilt_x / 2))
        qz = (0.0, 0.0, math.sin(self.tilt_z / 2), math.cos(self.tilt_z / 2))
        x1, y1, z1, w1 = qz
        x2, y2, z2, w2 = qx
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )


@dataclasses.dataclass(frozen=True)
class Ground:
    """Static boxes, and optionally a height field ``(heights, origin, spacing)``."""

    pieces: tuple[Piece, ...] = ()
    heightfield: tuple[np.ndarray, tuple[float, float, float], float] | None = None


FLAT = Ground((Piece((0.0, -1.0, 0.0), (4000.0, 2.0, 4000.0)),))


def wall(distance: float) -> Ground:
    """The flat ground with a 1 m thick wall across the road ``distance`` metres ahead (-Z)."""
    return Ground(FLAT.pieces + (Piece((0.0, 1.5, -distance - 0.5), (40.0, 3.0, 1.0)),))


def ramp(distance: float, angle: float = math.radians(15.0), length: float = 8.0) -> Ground:
    """The flat ground with a ramp rising toward -Z, its foot ``distance`` ahead, and a sheer end."""
    rise = length * math.sin(angle)
    centre_z = -distance - length * math.cos(angle) / 2.0
    # A thick slab tilted about X, its top surface the ramp: lowered by half its thickness.
    thickness = 2.0
    drop = thickness / 2.0 / math.cos(angle)
    return Ground(
        FLAT.pieces
        + (Piece((0.0, rise / 2.0 - drop, centre_z), (6.0, thickness, length), tilt_x=angle),)
    )


def bank(angle: float) -> Ground:
    """Ground tilted about Z: rising toward -X by ``angle``, a road banked for a left-hand turn."""
    return Ground((Piece((0.0, -1.0, 0.0), (4000.0, 2.0, 4000.0), tilt_z=-angle),))


def bumpy(
    amplitude: float = 0.04, wavelength: float = 3.0, size: int = 257, spacing: float = 1.0
) -> Ground:
    """A height field of crossing sine waves, ``amplitude`` m high every ``wavelength`` m."""
    coordinate = np.arange(size) * spacing
    xs, zs = np.meshgrid(coordinate, coordinate)
    heights = amplitude * (
        np.sin(2 * math.pi * xs / wavelength) * np.sin(2 * math.pi * zs / (wavelength * 1.3))
    )
    half = (size - 1) * spacing / 2.0
    return Ground((), (heights.astype(np.float32), (-half, 0.0, -half), spacing))


@dataclasses.dataclass
class Drive:
    """Each tick: time, chassis position, velocity, forward and up axes, wheels on the ground, impact speed."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    grounded: np.ndarray
    impact: np.ndarray

    def speed(self) -> np.ndarray:
        result: np.ndarray = np.linalg.norm(self.velocity, axis=1)
        return result

    def forward_speed(self) -> np.ndarray:
        result: np.ndarray = np.einsum('ij,ij->i', self.velocity, self.forward)
        return result

    def heading(self) -> np.ndarray:
        """Yaw of the forward axis about +Y, rad, 0 facing -Z, positive left; unwrapped."""
        result: np.ndarray = np.unwrap(np.arctan2(-self.forward[:, 0], -self.forward[:, 2]))
        return result

    def at(self, seconds: float) -> int:
        return int(np.searchsorted(self.time, seconds - 1e-9))


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    u = np.array([x, y, z])
    result: np.ndarray = (
        2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)
    )
    return result


def _start(heading: float) -> tuple[float, float, float, float]:
    return (0.0, math.sin(heading / 2.0), 0.0, math.cos(heading / 2.0))


def run_omi(
    ground: Ground,
    inputs: Inputs,
    seconds: float,
    start: tuple[float, float, float] = (0.0, 0.62, 0.0),
    heading: float = 0.0,
) -> Drive:
    """Drive the omi_physics car."""
    world = omi.PhysicsWorld()
    for piece in ground.pieces:
        shape = world.add_shape(model.Shape.box(piece.size))
        world.add_body(
            model.Motion(type=model.STATIC),
            collider=model.Collider(shape=shape),
            position=piece.centre,
            orientation=piece.orientation(),
        )
    if ground.heightfield is not None:
        heights, origin, spacing = ground.heightfield
        count, cols = heights.shape
        xs = origin[0] + np.arange(cols) * spacing
        zs = origin[2] + np.arange(count) * spacing
        gx, gz = np.meshgrid(xs, zs)
        points = np.stack(
            [gx.ravel(), heights.ravel().astype(np.float64) + origin[1], gz.ravel()], axis=1
        )
        faces = []
        for r in range(count - 1):
            for c in range(cols - 1):
                a = r * cols + c
                faces += [(a, a + cols, a + 1), (a + 1, a + cols, a + cols + 1)]
        shape = world.add_shape(model.Shape.trimesh(points, np.asarray(faces, dtype='i')))
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    chassis = world.add_shape(model.Shape.box(CHASSIS))
    body = world.add_body(
        model.Motion(type=model.DYNAMIC, mass=MASS, linearDamping=0.0, angularDamping=0.4),
        collider=model.Collider(shape=chassis),
        position=start,
    )
    car = omi_vehicle.RaycastVehicle(world, body, wheels(), tuning())
    car.place(start, heading)
    rows: dict[str, list[Any]] = {
        k: [] for k in ('time', 'position', 'velocity', 'forward', 'up', 'grounded', 'impact')
    }
    for tick in range(round(seconds / DT)):
        throttle, steer, brake = inputs(tick * DT)
        car.control(throttle=throttle, steer=steer, brake=brake)
        car.update(DT)
        world.step(DT)
        found = world.impact_on(body)
        rows['time'].append((tick + 1) * DT)
        rows['position'].append(car.position().copy())
        rows['velocity'].append(car.velocity().copy())
        rows['forward'].append(car.forward())
        rows['up'].append(car.up())
        rows['grounded'].append(sum(w.grounded for w in car.wheels))
        rows['impact'].append(0.0 if found is None else float(found[1]))
    return Drive(**{k: np.asarray(v) for k, v in rows.items()})


def jolt_world(ground: Ground, threads: int = 1) -> World:
    """A lockstep Jolt world holding ``ground``."""
    world = World(WorldSettings(tick_hz=1.0 / DT, threads=threads))
    for piece in ground.pieces:
        world.add_bodies(
            BodyBatch.boxes(
                [piece.centre],
                size=piece.size,
                motion=MotionType.STATIC,
                orientations=[piece.orientation()],
            )
        )
    if ground.heightfield is not None:
        heights, origin, spacing = ground.heightfield
        world.add_bodies(BodyBatch.heightfield(heights, origin=origin, spacing=spacing))
    return world


def add_jolt_car(
    world: World,
    start: tuple[float, float, float] = (0.0, 0.62, 0.0),
    heading: float = 0.0,
    vehicle_settings: VehicleSettings | None = None,
) -> tuple[int, Any]:
    """Add the chassis and its vehicle; return the chassis uid and the :class:`Vehicle`."""
    uids, _ = world.add_bodies(
        BodyBatch.boxes(
            [start],
            size=CHASSIS,
            mass=MASS,
            linear_damping=0.0,
            angular_damping=0.4,
            orientations=[_start(heading)],
            flags=BodyFlag.REPORT,
        )
    )
    chassis = int(uids[0])
    return chassis, world.add_vehicle(chassis, vehicle_settings or settings())


def run_jolt(
    ground: Ground,
    inputs: Inputs,
    seconds: float,
    start: tuple[float, float, float] = (0.0, 0.62, 0.0),
    heading: float = 0.0,
    vehicle_settings: VehicleSettings | None = None,
    threads: int = 1,
) -> Drive:
    """Drive the Jolt car."""
    with jolt_world(ground, threads) as world:
        chassis, car = add_jolt_car(world, start, heading, vehicle_settings)
        rows: dict[str, list[Any]] = {
            k: [] for k in ('time', 'position', 'velocity', 'forward', 'up', 'grounded', 'impact')
        }
        last: tuple[float, float, float] | None = None
        for tick in range(round(seconds / DT)):
            given = inputs(tick * DT)
            if given != last:
                car.set_input(throttle=given[0], steer=given[1], brake=given[2])
                last = given
            world.run_ticks(1)
            snap = world.latest()
            assert snap is not None
            row = snap.rows[chassis]
            q = np.asarray(snap.orientation[row], dtype=np.float64)
            vehicles = snap.vehicles
            wheel_rows = vehicles.wheels
            rows['time'].append((tick + 1) * DT)
            rows['position'].append(np.array(snap.position[row]))
            rows['velocity'].append(np.array(snap.linear_velocity[row], dtype=np.float64))
            rows['forward'].append(_rotate(q, np.array([0.0, 0.0, -1.0])))
            rows['up'].append(_rotate(q, np.array([0.0, 1.0, 0.0])))
            rows['grounded'].append(int(np.count_nonzero(wheel_rows.contact)))
            rows['impact'].append(float(vehicles.impact_speed[0]))
    return Drive(**{k: np.asarray(v) for k, v in rows.items()})


def constant(throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0) -> Inputs:
    """The same inputs throughout."""
    return lambda _t: (throttle, steer, brake)


def phases(*spans: tuple[float, tuple[float, float, float]]) -> Inputs:
    """Inputs by span: each (until seconds, (throttle, steer, brake)); the last holds after."""

    def at(t: float) -> tuple[float, float, float]:
        for until, given in spans:
            if t < until:
                return given
        return spans[-1][1]

    return at
