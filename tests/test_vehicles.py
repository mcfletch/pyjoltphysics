"""Vehicles: wheeled cars, motorcycles and tracked vehicles over Jolt's vehicle constraint."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import types
from typing import Any

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    Engine,
    EventKind,
    EventPhase,
    Lean,
    MotionType,
    Shape,
    Vehicle,
    VehicleKind,
    VehicleSettings,
    VehicleTrack,
    WheelSettings,
    World,
    WorldSettings,
)
from pyjoltphysics.testing import MakeWorld, floor, latest, vehicle_digest

MASS = 1180.0
CHASSIS = (1.85, 0.62, 4.2)


def _tuning(**changes: Any) -> types.SimpleNamespace:
    """A raycast vehicle's tuning, as omi_physics' VehicleTuning carries it."""
    values = {
        'engine_force': 20000.0,
        'brake_force': 16000.0,
        'reverse_fraction': 0.4,
        'maximum_steer': 0.52,
        'steer_speed': 3.2,
        'rolling_resistance': 0.02,
        'downforce': 6.0,
        'base_speed': 21.0,
        'drag': 0.9,
        'steer_falloff_speed': 10.0,
        'holding': 6000.0,
    }
    values.update(changes)
    return types.SimpleNamespace(**values)


def _wheels() -> list[types.SimpleNamespace]:
    """glisteel's four wheels, as omi_physics' WheelSpec carries them."""
    out = []
    for front in (True, False):
        for side in (-1.0, 1.0):
            out.append(
                types.SimpleNamespace(
                    position=(side * 0.85, -0.02, -1.275 if front else 1.275),
                    radius=0.33,
                    suspension_travel=0.22,
                    suspension_stiffness=26.0,
                    suspension_damping=0.55,
                    steering=front,
                    driven=not front,
                    braked=True,
                    grip=1.9,
                )
            )
    return out


def car_settings(**changes: Any) -> VehicleSettings:
    return VehicleSettings.from_raycast(_tuning(), _wheels(), MASS, **changes)


def _chassis(
    world: World, position: tuple[float, float, float] = (0.0, 0.62, 0.0), **values: Any
) -> int:
    values.setdefault('angular_damping', 0.4)
    values.setdefault('linear_damping', 0.0)
    uids, _ = world.add_bodies(BodyBatch.boxes([position], size=CHASSIS, mass=MASS, **values))
    return int(uids[0])


def _car(
    world: World, position: tuple[float, float, float] = (0.0, 0.62, 0.0), **changes: Any
) -> Vehicle:
    return world.add_vehicle(_chassis(world, position), car_settings(**changes))


def _world(make_world: MakeWorld, **settings: Any) -> World:
    settings.setdefault('threads', 1)
    world = make_world(**settings)
    world.add_bodies(floor(500.0))
    return world


def _heading(world: World, uid: int) -> float:
    snap = latest(world)
    x, y, z, w = (float(c) for c in snap.orientation[snap.rows[uid]])
    # The chassis's -Z axis rotated, as a yaw about +Y: 0 facing -Z, positive left.
    fx = -(2 * (x * z + w * y))
    fz = -(1 - 2 * (x * x + y * y))
    return math.atan2(-fx, -fz)


def _upright(world: World, uid: int) -> float:
    """The world Y component of the chassis's up axis: 1 upright, 0 on its side."""
    snap = latest(world)
    x, _y, z, _w = (float(c) for c in snap.orientation[snap.rows[uid]])
    return 1 - 2 * (x * x + z * z)


def _drive(world: World, car: Vehicle, seconds: float, **inputs: Any) -> None:
    car.set_input(**inputs)
    world.run_ticks(round(seconds * world.settings.tick_hz))


# -- standing ------------------------------------------------------------------


def test_a_car_settles_on_its_wheels_at_its_ride_height(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(240)
    snap = latest(world)
    row = snap.rows[car.chassis]
    wheels = snap.vehicles.wheels
    assert len(snap.vehicles) == 1 and len(wheels) == 4
    assert wheels.contact.all()
    # The springs carry the car: 26 car-weights per metre, so a quarter of the
    # weight each is 1 / (4 * 26) m of compression.
    assert wheels.compression == pytest.approx(np.full(4, 1.0 / 104.0), abs=2e-3)
    assert float(wheels.load.sum()) == pytest.approx(MASS * 9.81, rel=0.02)
    # Hub at -0.02, droop 0.22 less the compression, radius 0.33: the chassis's origin above the floor.
    assert float(snap.position[row, 1]) == pytest.approx(0.02 + 0.22 - 1.0 / 104.0 + 0.33, abs=3e-3)
    assert np.linalg.norm(snap.linear_velocity[row]) < 0.01
    assert (wheels.ground != 0).all()


def test_wheel_poses_hang_below_their_mounts_with_the_axle_across_the_car(
    make_world: MakeWorld,
) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(120)
    snap = latest(world)
    wheels = snap.vehicles.wheels
    chassis = snap.position[snap.rows[car.chassis]]
    for i, spec in enumerate(_wheels()):
        centre = wheels.position[i]
        assert centre[0] - chassis[0] == pytest.approx(spec.position[0], abs=1e-3)
        assert centre[2] - chassis[2] == pytest.approx(spec.position[2], abs=1e-3)
        assert centre[1] == pytest.approx(0.33, abs=2e-3)
    # The axle (model X) stays along world X on a straight car.
    x, y, z, w = (float(c) for c in wheels.orientation[0])
    axle = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    assert abs(axle[0]) == pytest.approx(1.0, abs=1e-3)


def test_a_wheel_off_the_edge_of_the_world_touches_nothing(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    car = _car(world, (0.0, 5.0, 0.0))
    world.run_ticks(2)
    wheels = latest(world).vehicles.wheels
    assert not wheels.contact.any()
    assert (wheels.ground == 0).all() and (wheels.triangle == -1).all()
    assert wheels.compression == pytest.approx(np.zeros(4), abs=1e-6)
    assert car.impact() is None


# -- going, stopping and turning -----------------------------------------------


def test_throttle_drives_it_forward_in_first_gear(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 2.0, throttle=1.0)
    snap = latest(world)
    assert car.forward_speed > 15.0
    assert car.gear == 1 and car.rpm > 0.0
    assert float(snap.position[snap.rows[car.chassis], 2]) < -10.0
    assert (snap.vehicles.wheels.spin[2:] > 0.0).all()
    assert snap.vehicles.input[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def _yaw_rate(world: World, uid: int) -> float:
    snap = latest(world)
    return float(snap.angular_velocity[snap.rows[uid]][1])


def _crowned(make_world: MakeWorld, angle: float) -> World:
    """A road mesh falling ``angle`` rad from a crown along x = 0 to either side, as a drained road does.

    Triangles every 2 m along it and 1 m across, so the wheels cross edges as
    they do on a road swept from a centreline.
    """
    world = make_world(threads=1, tick_hz=120.0)
    across = np.arange(-6.0, 6.5, 1.0)
    along = np.arange(-1000.0, 20.0, 2.0)
    xs, zs = np.meshgrid(across, along)
    ys = -np.abs(xs) * math.tan(angle)
    points = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    n = len(across)
    rows = np.arange(len(along) - 1)[:, None] * n + np.arange(n - 1)[None, :]
    a, b, c, d = rows, rows + 1, rows + n, rows + n + 1
    triangles = np.concatenate([np.stack([a, c, b], -1), np.stack([b, c, d], -1)]).reshape(-1, 3)
    mesh = world.add_shape(Shape.mesh(points, triangles))
    world.add_bodies(BodyBatch.of_shape([(0.0, 0.0, 0.0)], mesh, motion=MotionType.STATIC))
    return world


def test_driven_straight_it_holds_its_heading_without_chatter(make_world: MakeWorld) -> None:
    """The tyres hold a car with no slip angle: its yaw rate stays still from tick to tick.

    On a crowned road each side's wheels stand on ground tilted the other way,
    so each pair pulls outward and the tyres hold that without a slip angle.
    """
    world = _crowned(make_world, math.radians(1.5))
    car = _car(world)
    _drive(world, car, 1.0)
    car.set_input(throttle=0.6)
    rates = []
    for _ in range(round(4.0 * 120.0)):
        world.run_ticks(1)
        rates.append(_yaw_rate(world, car.chassis))
    late = np.abs(np.asarray(rates[240:]))
    assert car.forward_speed > 25.0
    assert late.max() < 0.002, 'yaw rate up to %.4f rad/s' % late.max()
    assert abs(_heading(world, car.chassis)) < 0.002


def test_negative_throttle_reverses(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 2.0, throttle=-1.0)
    assert car.forward_speed < -3.0
    assert car.gear == -1


def test_the_power_falls_away_above_the_base_speed(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world, drag=0.0, downforce=0.0, rolling_resistance=0.0)
    world.run_ticks(60)
    speeds = []
    car.set_input(throttle=1.0)
    for _ in range(8):
        world.run_ticks(120)
        speeds.append(car.forward_speed)
    gains = np.diff(speeds)
    # Constant power: the speed gained each second shrinks as the car gets going.
    assert gains[-1] < 0.6 * gains[1]


def test_drag_sets_a_top_speed(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(2000.0))
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 18.0, throttle=1.0)
    top = car.forward_speed
    world.run_ticks(120)
    # Power 420 kW against 0.9 v^2 of drag and the rolling resistance balances below 77 m/s.
    assert 68.0 < top < 77.0
    assert car.forward_speed - top < 0.6


def test_the_brake_stops_it_and_holds_it(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 2.0, throttle=1.0)
    _drive(world, car, 4.0, brake=1.0)
    assert abs(car.forward_speed) < 0.05


def test_steering_left_turns_it_left_and_only_the_steered_wheels_turn(
    make_world: MakeWorld,
) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 1.0, throttle=0.5)
    before = _heading(world, car.chassis)
    _drive(world, car, 1.0, throttle=0.3, steer=1.0)
    assert _heading(world, car.chassis) > before + 0.3
    wheels = latest(world).vehicles.wheels
    assert (wheels.steer[:2] > 0.0).all() and (wheels.steer[2:] == 0.0).all()


def test_the_lock_eases_off_with_speed(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    world.run_ticks(60)
    _drive(world, car, 0.5, steer=1.0)
    standing = car.steer_angle
    assert standing == pytest.approx(0.52, abs=1e-3)
    _drive(world, car, 2.5, throttle=1.0)
    _drive(world, car, 0.5, throttle=1.0, steer=1.0)
    speed = car.forward_speed
    assert car.steer_angle == pytest.approx(0.52 / (1 + (speed / 10.0) ** 2), rel=0.1)


def test_the_steering_moves_at_its_rate(make_world: MakeWorld) -> None:
    world = _world(make_world, tick_hz=100.0)
    car = _car(world)
    world.run_ticks(10)
    _drive(world, car, 0.1, steer=1.0)
    # steer_speed 3.2 of the 0.52 rad lock a second: 0.1664 rad in 0.1 s.
    assert car.steer_angle == pytest.approx(3.2 * 0.52 * 0.1, rel=0.02)


def test_an_idle_car_holds_its_place_on_a_slope(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    tilt = math.radians(8.0)
    world.add_bodies(
        BodyBatch.boxes(
            [(0.0, -1.0, 0.0)],
            size=(200.0, 2.0, 200.0),
            motion=MotionType.STATIC,
            orientations=[(math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))],
        )
    )
    held = _car(world, (0.0, 0.75, 0.0))
    world.run_ticks(120)
    start = latest(world).position[latest(world).rows[held.chassis]].copy()
    world.run_ticks(240)
    end = latest(world).position[latest(world).rows[held.chassis]]
    assert np.linalg.norm(end - start) < 0.05


def test_a_car_that_does_not_hold_rolls_down_the_slope(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    tilt = math.radians(8.0)
    world.add_bodies(
        BodyBatch.boxes(
            [(0.0, -1.0, 0.0)],
            size=(200.0, 2.0, 200.0),
            motion=MotionType.STATIC,
            orientations=[(math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))],
        )
    )
    rolling = _car(world, (0.0, 0.75, 0.0), holding=0.0)
    world.run_ticks(360)
    assert abs(rolling.forward_speed) > 1.0


# -- surfaces ------------------------------------------------------------------


def test_less_grip_means_less_acceleration(make_world: MakeWorld) -> None:
    speeds = []
    for grip in (None, 0.3):
        world = _world(make_world)
        car = _car(world)
        world.run_ticks(60)
        if grip is not None:
            car.set_surface(grip)
        _drive(world, car, 1.5, throttle=1.0)
        speeds.append(car.forward_speed)
        assert latest(world).vehicles.wheels.grip.tolist() == pytest.approx([grip or 1.0] * 4)
    assert speeds[1] < 0.5 * speeds[0]


def test_soft_going_bogs_a_coasting_car_down(make_world: MakeWorld) -> None:
    speeds = []
    for rolling in (0.0, 0.3):
        world = _world(make_world)
        car = _car(world)
        world.run_ticks(60)
        _drive(world, car, 2.0, throttle=1.0)
        car.set_surface(1.0, rolling=rolling)
        _drive(world, car, 2.0)
        speeds.append(car.forward_speed)
    assert speeds[1] < speeds[0] - 3.0


def test_the_surface_table_serves_wheels_by_the_material_under_them(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(500.0, material=3))
    world.set_surface_table([1.0, 1.0, 1.0, 0.25], [0.0, 0.0, 0.0, 0.1])
    car = _car(world)
    world.run_ticks(60)
    wheels = latest(world).vehicles.wheels
    assert wheels.material.tolist() == [3, 3, 3, 3]
    assert wheels.grip.tolist() == pytest.approx([0.25] * 4)
    car.set_surface(0.8, wheel=1)
    world.run_ticks(1)
    assert latest(world).vehicles.wheels.grip.tolist() == pytest.approx([0.25, 0.8, 0.25, 0.25])
    car.set_surface(None)
    world.run_ticks(1)
    assert latest(world).vehicles.wheels.grip.tolist() == pytest.approx([0.25] * 4)


def test_a_wheel_on_a_mesh_reports_the_triangle_under_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    points = np.array([(-50, 0, -50), (50, 0, -50), (50, 0, 50), (-50, 0, 50)], dtype=np.float64)
    shape = world.add_shape(
        Shape.mesh(
            points,
            np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32),
            ids=np.array([70, 71], dtype=np.uint32),
        )
    )
    world.add_bodies(BodyBatch(positions=[(0.0, 0.0, 0.0)], shape=shape, motion=MotionType.STATIC))
    _car(world)
    world.run_ticks(60)
    wheels = latest(world).vehicles.wheels
    assert wheels.contact.all()
    assert set(wheels.triangle.tolist()) <= {70, 71} and len(set(wheels.triangle.tolist())) == 2


# -- what a wheel stands on ------------------------------------------------------


def test_wheels_stand_only_on_what_the_chassis_collides_with(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor(50.0, group=2))
    car = world.add_vehicle(_chassis(world, (0.0, 0.62, 0.0), mask=1), car_settings())
    world.run_ticks(30)
    assert not latest(world).vehicles.wheels.contact.any()
    assert car.forward_speed == 0.0


def test_wheels_pass_through_sensors(make_world: MakeWorld) -> None:
    world = _world(make_world)
    world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 0.3, 0.0)],
            size=(10.0, 0.6, 10.0),
            motion=MotionType.STATIC,
            flags=BodyFlag.SENSOR,
        )
    )
    _car(world, (0.0, 1.5, 0.0))
    world.run_ticks(240)
    wheels = latest(world).vehicles.wheels
    assert wheels.contact.all()
    assert wheels.contact_point[:, 1] == pytest.approx(np.zeros(4), abs=1e-3)


def test_a_wheel_rolling_over_a_dynamic_body_presses_on_it(make_world: MakeWorld) -> None:
    world = _world(make_world)
    plank, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.1, 1.275)], size=(3.0, 0.2, 1.0), mass=50.0)
    )
    car = _car(world, (0.0, 0.85, 0.0))
    world.run_ticks(120)
    wheels = latest(world).vehicles.wheels
    assert int(plank[0]) in wheels.ground.tolist()
    assert car.forward_speed == pytest.approx(0.0, abs=0.05)


# -- blows -----------------------------------------------------------------------


def test_driving_into_a_wall_is_an_impact_with_the_closing_speed(make_world: MakeWorld) -> None:
    world = _world(make_world)
    walls, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.5, -30.5)], size=(20.0, 3.0, 1.0), motion=MotionType.STATIC)
    )
    car = _car(world)
    world.run_ticks(60)
    world.drain_events()
    car.set_input(throttle=1.0)
    best = 0.0
    speed_before = 0.0
    for _ in range(360):
        speed = car.forward_speed
        world.run_ticks(1)
        found = car.impact()
        if found is not None and found[1] > best:
            best = found[1]
            speed_before = speed
            assert found[0] == int(walls[0]) and found[2] > 0.0
    assert best > 15.0
    assert best == pytest.approx(speed_before, rel=0.1)
    events = world.drain_events()
    mine = events[events['kind'] == EventKind.VEHICLE]
    assert len(mine) >= 1
    crash = mine[mine['phase'] == EventPhase.CRASH]
    assert len(crash) == 1 and crash['a'][0] == car.id and crash['b'][0] == walls[0]
    assert float(crash['approach'][0]) == pytest.approx(best, rel=1e-5)
    # The normal points from the chassis into the wall: -Z.
    assert float(crash['normal'][0][2]) < -0.9


def test_a_gentle_touch_is_no_event(make_world: MakeWorld) -> None:
    world = _world(make_world)
    world.add_bodies(
        BodyBatch.boxes([(0.0, 1.5, -3.0)], size=(20.0, 3.0, 1.0), motion=MotionType.STATIC)
    )
    car = _car(world, impact_speed=5.0)
    world.run_ticks(60)
    world.drain_events()
    _drive(world, car, 2.0, throttle=0.05)
    events = world.drain_events()
    assert not (events['kind'] == EventKind.VEHICLE).any()


# -- life cycle ------------------------------------------------------------------


def test_vehicles_come_and_go_while_the_world_runs(make_world: MakeWorld) -> None:
    world = _world(make_world, threads=4)
    world.start()
    cars = [_car(world, (4.0 * i, 0.62, 0.0)) for i in range(3)]
    assert world.wait_tick(30, 5.0)
    world.flush()
    world.wait_tick(world.tick + 2, 5.0)
    assert len(latest(world).vehicles) == 3
    assert world.stats()['vehicles'] == 3
    cars[1].remove()
    world.remove_bodies([cars[2].chassis])
    world.wait_tick(world.tick + 3, 5.0)
    snap = latest(world)
    assert snap.vehicles.id.tolist() == [cars[0].id]
    assert len(snap.vehicles.wheels) == 4
    assert cars[1].chassis in snap.rows and cars[2].chassis not in snap.rows
    assert world.stats()['vehicles'] == 1
    more = _car(world, (20.0, 0.62, 0.0))
    world.wait_tick(world.tick + 3, 5.0)
    assert latest(world).vehicles.id.tolist() == [cars[0].id, more.id]
    assert more.id > cars[2].id
    world.stop()
    assert not world.error


def test_a_vehicle_needs_a_dynamic_chassis(make_world: MakeWorld) -> None:
    world = _world(make_world)
    static, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 2.0, 0.0)], size=CHASSIS, motion=MotionType.STATIC)
    )
    world.add_vehicle(int(static[0]), car_settings())
    world.add_vehicle(9999, car_settings())
    world.run_ticks(1)
    assert len(latest(world).vehicles) == 0
    assert world.stats()['rejected'] == 2


def test_inputs_for_a_removed_vehicle_are_rejected(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    car.remove()
    car.set_input(throttle=1.0)
    world.run_ticks(1)
    assert world.stats()['rejected'] == 1
    with pytest.raises(LookupError):
        _ = car.forward_speed


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'wheels': ()}, 'at least one wheel'),
        ({'wheels': (WheelSettings((0.0, 0.0, 0.0), radius=0.0),)}, 'radius'),
        (
            {'wheels': (WheelSettings((0.0, 0.0, 0.0), suspension_min=0.5, suspension_max=0.2),)},
            'suspension',
        ),
        ({'wheels': (WheelSettings((0.0, 0.0, 0.0), stiffness=0.0),)}, 'stiffness'),
        ({'lateral_friction': ((0.0, 0.0), (0.0, 1.0))}, 'ascend'),
        ({'drag': math.nan}, 'not finite'),
    ],
)
def test_bad_settings_are_refused(
    make_world: MakeWorld, change: dict[str, Any], message: str
) -> None:
    world = _world(make_world)
    chassis = _chassis(world)
    with pytest.raises(ValueError, match=message):
        world.add_vehicle(chassis, dataclasses.replace(car_settings(), **change))


def test_settings_survive_a_round_trip_through_builtin_types() -> None:
    settings = car_settings(anti_roll_bars=())
    plain = settings.as_dict()
    assert isinstance(plain['wheels'][0]['position'], list)
    assert VehicleSettings.from_dict(plain) == settings


def test_differentials_pair_the_driven_wheels_by_axle() -> None:
    settings = car_settings()
    diffs = settings.resolved_differentials()
    assert [(d.left, d.right) for d in diffs] == [(2, 3)]
    assert diffs[0].limited_slip == math.inf and diffs[0].ratio == 1.0
    assert settings.wheelbase() == pytest.approx(2.55)


def test_the_raycast_mapping_is_the_documented_one() -> None:
    settings = car_settings()
    wheel = settings.wheels[0]
    assert wheel.stiffness == pytest.approx(26.0 * MASS * 9.81)
    assert wheel.damping == pytest.approx(0.55 * 2.0 * math.sqrt(wheel.stiffness * MASS / 4))
    assert wheel.brake_torque == pytest.approx(16000.0 / 4 * 0.33)
    assert (
        settings.wheels[0].handbrake_torque == 0.0
        and settings.wheels[2].handbrake_torque == wheel.brake_torque
    )
    assert settings.engine.max_torque == pytest.approx(20000.0 * 0.33)
    assert settings.transmission.gears == (1.0,) and settings.transmission.reverse_gears == (-0.4,)
    assert settings.steer_rate == pytest.approx(3.2 * 0.52)
    assert settings.holding == pytest.approx(6000.0 / 16000.0)
    knee = 21.0 / (settings.engine.max_rpm / 60.0 * 2 * math.pi * 0.33)
    xs, ys = np.array(settings.engine.torque_curve).T
    assert ys[0] == 1.0 and xs[1] == pytest.approx(knee) and ys[1] == 1.0
    # Constant power above the knee: torque times speed is flat.
    assert xs[1:] * ys[1:] == pytest.approx(np.full(len(xs) - 1, knee))


# -- determinism -------------------------------------------------------------------


def _replay(threads: int, inputs: list[tuple[int, int, tuple[float, float, float, float]]]) -> str:
    world = World(WorldSettings(threads=threads, tick_hz=120.0))
    try:
        world.add_bodies(floor(500.0))
        world.add_bodies(
            BodyBatch.boxes(
                [(3.0 * i - 15.0, 0.5, -40.0) for i in range(10)], size=(1.0, 1.0, 1.0), mass=60.0
            )
        )
        cars = [_car(world, (4.0 * i - 18.0, 0.62, 0.0)) for i in range(10)]
        for tick, index, values in inputs:
            cars[index].set_input(*values, at_tick=tick)
        digest = hashlib.sha256()
        for _ in range(40):
            world.run_ticks(15)
            vehicle_digest(latest(world), digest)
        return digest.hexdigest()
    finally:
        world.stop()


def test_ten_vehicles_replay_bitwise_on_1_and_8_threads() -> None:
    rng = np.random.default_rng(7)
    inputs: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for _ in range(60):
        throttle, steer, brake, handbrake = (float(v) for v in rng.uniform(-1, 1, 4))
        values = (throttle, steer, abs(brake) * 0.5, abs(handbrake) * 0.2)
        inputs.append((int(rng.integers(1, 500)), int(rng.integers(0, 10)), values))
    inputs.sort()
    one = _replay(1, inputs)
    assert one == _replay(8, inputs)
    assert one == _replay(1, inputs)


def test_a_saved_state_restores_the_vehicles(make_world: MakeWorld) -> None:
    world = _world(make_world)
    car = _car(world)
    car.set_input(throttle=1.0, steer=0.4)
    world.run_ticks(120)
    state = world.save_state()
    world.run_ticks(60)
    later = latest(world).position[latest(world).rows[car.chassis]].copy()
    world.restore_state(state)
    world.run_ticks(60)
    again = latest(world).position[latest(world).rows[car.chassis]]
    assert np.linalg.norm(again - later) < 0.05


# -- motorcycles and tracked vehicles -----------------------------------------------


def bike_settings() -> VehicleSettings:
    wheels = (
        WheelSettings(
            (0.0, -0.1, -0.75),
            radius=0.31,
            width=0.1,
            suspension_max=0.25,
            stiffness=30000.0,
            damping=2500.0,
            steered=True,
            driven=False,
            brake_torque=500.0,
            inertia=0.6,
        ),
        WheelSettings(
            (0.0, -0.1, 0.75),
            radius=0.31,
            width=0.12,
            suspension_max=0.25,
            stiffness=30000.0,
            damping=2500.0,
            driven=True,
            brake_torque=300.0,
            handbrake_torque=600.0,
            inertia=0.6,
        ),
    )
    return VehicleSettings(
        wheels=wheels,
        kind=VehicleKind.MOTORCYCLE,
        max_steer=math.radians(30.0),
        steer_rate=2.0,
        lean=Lean(),
        differential_ratio=1.93 * 40.0 / 16.0,
    )


def test_a_motorcycle_rides_upright_and_turns(make_world: MakeWorld) -> None:
    world = _world(make_world, tick_hz=120.0)
    uids, _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.8, 0.0)], size=(0.4, 0.6, 1.6), mass=240.0))
    bike = world.add_vehicle(int(uids[0]), bike_settings())
    world.run_ticks(60)
    _drive(world, bike, 3.0, throttle=0.6)
    assert _upright(world, bike.chassis) > 0.95  # within 18 degrees of vertical
    snap = latest(world)
    assert bike.forward_speed > 5.0
    assert snap.vehicles.kind[0] == VehicleKind.MOTORCYCLE and len(snap.vehicles.wheels) == 2
    before = _heading(world, bike.chassis)
    _drive(world, bike, 1.5, throttle=0.4, steer=0.5)
    assert _heading(world, bike.chassis) > before + 0.2
    assert _upright(world, bike.chassis) > 0.6  # leaning into the turn, not lying down


def tank_settings() -> VehicleSettings:
    wheels = []
    for side, x in ((0, -1.4), (1, 1.4)):
        for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
            wheels.append(
                WheelSettings(
                    (x, -0.5, z),
                    radius=0.4,
                    width=0.4,
                    suspension_max=0.4,
                    stiffness=60000.0,
                    damping=6000.0,
                    track=side,
                    inertia=1.0,
                )
            )
    return VehicleSettings(
        wheels=tuple(wheels),
        kind=VehicleKind.TRACKED,
        tracks=(VehicleTrack(driven_wheel=4), VehicleTrack(driven_wheel=9)),
        engine=Engine(max_torque=1000.0),
        max_steer=1.0,
    )


def test_a_tracked_vehicle_drives_turns_on_the_spot_and_stays_upright(
    make_world: MakeWorld,
) -> None:
    world = _world(make_world, tick_hz=120.0)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.2, 0.0)], size=(3.0, 1.0, 5.0), mass=4000.0)
    )
    tank = world.add_vehicle(int(uids[0]), tank_settings())
    world.run_ticks(120)
    snap = latest(world)
    assert snap.vehicles.kind[0] == VehicleKind.TRACKED and len(snap.vehicles.wheels) == 10
    assert snap.vehicles.wheels.contact.all()
    _drive(world, tank, 3.0, throttle=1.0)
    assert tank.forward_speed > 1.0
    _drive(world, tank, 3.0, brake=1.0)
    before = _heading(world, tank.chassis)
    _drive(world, tank, 1.0, throttle=0.6, steer=1.0)
    assert _heading(world, tank.chassis) > before + 0.2
    assert _upright(world, tank.chassis) > 0.98


def test_a_tracked_vehicle_needs_wheels_on_both_tracks(make_world: MakeWorld) -> None:
    world = _world(make_world)
    uids, _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.2, 0.0)], size=(3.0, 1.0, 5.0), mass=4000.0)
    )
    settings = tank_settings()
    one_sided = dataclasses.replace(
        settings,
        wheels=tuple(dataclasses.replace(w, track=0) for w in settings.wheels),
        tracks=(VehicleTrack(driven_wheel=4), VehicleTrack(driven_wheel=4)),
    )
    world.add_vehicle(int(uids[0]), one_sided)
    world.run_ticks(1)
    assert len(latest(world).vehicles) == 0 and world.stats()['rejected'] == 1
