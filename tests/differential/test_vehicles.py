"""glisteel's car on Jolt against omi_physics' RaycastVehicle, scenario by scenario.

The two models differ: omi_physics pushes the chassis with forces at ray
contacts and holds the tyres to a friction circle at once; Jolt solves the
suspension, a slip-curve tyre and an engine, clutch and differential as
constraints. The comparisons are of behaviour glisteel's handling and feel
tests rely on, with the tolerances in openglcontext/plans/NATIVE-PHYSICS.md
(P6a as built).

Each scenario reduces a drive to a few numbers (its summary). omi_physics'
summary is its recorded answer (:func:`differential.vehicles.reference`),
checked against a live run where omi_physics is installed; Jolt's is computed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from differential import engines
from differential import vehicles as V
from pyjoltphysics import VehicleSettings

pytestmark = pytest.mark.differential

LAUNCH = V.phases((0.3, (0.0, 0.0, 0.0)), (60.0, (1.0, 0.0, 0.0)))


def _settled(drive: V.Drive) -> list[float]:
    """Ride height, speed and wheels on the ground at the end."""
    return [float(drive.position[-1, 1]), float(drive.speed()[-1]), int(drive.grounded[-1])]


def test_standing_still_it_settles_at_the_same_ride_height() -> None:
    omi = V.reference('settle', _settled, V.FLAT, V.constant(), 2.0)
    jolt = _settled(V.run_jolt(V.FLAT, V.constant(), 2.0))
    assert jolt[0] == pytest.approx(omi[0], abs=1e-3)
    assert jolt[1] < 0.01 and omi[1] < 0.01
    assert jolt[2] == omi[2] == 4


def _launch_speeds(drive: V.Drive) -> list[float]:
    """Speed at 2.3, 4.3 and 8.3 s."""
    return [float(drive.speed()[drive.at(t) - 1]) for t in (2.3, 4.3, 8.3)]


def test_it_accelerates_to_speed_in_a_straight_line() -> None:
    omi = V.reference('launch', _launch_speeds, V.FLAT, LAUNCH, 8.3)
    drive = V.run_jolt(V.FLAT, LAUNCH, 8.3)
    assert _launch_speeds(drive) == pytest.approx(omi, rel=0.04)
    # Straight: a road's width is 7 m; it stays within half a metre of the line.
    assert abs(drive.position[-1, 0]) < 0.5 and abs(drive.heading()[-1]) < 0.01


BRAKE = V.phases((0.3, (0.0, 0.0, 0.0)), (3.3, (1.0, 0.0, 0.0)), (60.0, (0.0, 0.0, 1.0)))


def _braking(drive: V.Drive) -> list[float]:
    """Stopping distance over the square of the speed at the brake, and the speed at the end."""
    i = drive.at(3.3)
    v0 = drive.speed()[i]
    stop = i + int(np.argmax(drive.speed()[i:] < 0.2))
    return [
        float(abs(drive.position[stop, 2] - drive.position[i, 2]) / v0**2),
        float(drive.speed()[-1]),
    ]


def test_it_brakes_to_a_stop_over_the_same_distance_per_speed() -> None:
    omi = V.reference('brake', _braking, V.FLAT, BRAKE, 8.0)
    jolt = _braking(V.run_jolt(V.FLAT, BRAKE, 8.0))
    assert omi[1] < 0.05 and jolt[1] < 0.05
    assert jolt[0] == pytest.approx(omi[0], rel=0.08)


@pytest.mark.parametrize(('accelerate', 'steer'), [(1.0, 0.5), (2.0, 0.3), (2.0, 1.0)])
def test_steady_cornering_pulls_the_same_lateral_g(accelerate: float, steer: float) -> None:
    inputs = V.phases(
        (0.3, (0.0, 0.0, 0.0)), (0.3 + accelerate, (1.0, 0.0, 0.0)), (60.0, (0.0, steer, 0.0))
    )

    def cornering(drive: V.Drive) -> list[float]:
        """Yaw rate and lateral g over the turn, and the lowest the up axis tilted."""
        i, j = drive.at(accelerate + 1.8), len(drive.time) - 1
        yaw = (drive.heading()[j] - drive.heading()[i]) / (drive.time[j] - drive.time[i])
        return [
            float(yaw),
            float(drive.speed()[i:].mean() * yaw / 9.81),
            float(drive.up[:, 1].min()),
        ]

    seconds = accelerate + 3.3
    omi_yaw, omi_g, omi_up = V.reference(
        'corner/%g/%g' % (accelerate, steer), cornering, V.FLAT, inputs, seconds
    )
    jolt_yaw, jolt_g, jolt_up = cornering(V.run_jolt(V.FLAT, inputs, seconds))
    assert omi_up > 0.99 and jolt_up > 0.99
    assert jolt_g == pytest.approx(omi_g, rel=0.06)
    assert jolt_yaw == pytest.approx(omi_yaw, rel=0.10)


def _flight(drive: V.Drive) -> tuple[int, int]:
    air = drive.grounded == 0
    best = (0, 0)
    start = None
    for k, up in enumerate(air):
        if up and start is None:
            start = k
        if not up and start is not None:
            if k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    return best


def _jump(drive: V.Drive) -> list[float]:
    """Highest point, time in the air and where it lands (z)."""
    take_off, landing = _flight(drive)
    return [
        float(drive.position[:, 1].max()),
        float(drive.time[landing] - drive.time[take_off]),
        float(drive.position[landing, 2]),
    ]


def test_a_jump_flies_as_far_and_lands_upright() -> None:
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0)), (60.0, (0.2, 0.0, 0.0)))
    omi = V.reference('jump', _jump, V.ramp(20.0), inputs, 5.0)
    drive = V.run_jolt(V.ramp(20.0), inputs, 5.0)
    jolt = _jump(drive)
    assert jolt[0] == pytest.approx(omi[0], abs=0.15)
    assert jolt[1] == pytest.approx(omi[1], rel=0.12)
    assert jolt[2] == pytest.approx(omi[2], rel=0.08)
    assert drive.up[:, 1].min() > 0.9 and drive.up[-1, 1] > 0.999
    assert drive.grounded[-1] == 4


def _wall(drive: V.Drive) -> list[float]:
    """The hardest blow, the speed two ticks before it, and the speed and up axis at the end."""
    k = int(np.argmax(drive.impact))
    return [
        float(drive.impact[k]),
        float(drive.speed()[k - 2]),
        float(drive.speed()[-1]),
        float(drive.up[-1, 1]),
    ]


def test_a_wall_stops_it_with_the_blow_of_its_closing_speed() -> None:
    omi = V.reference('wall', _wall, V.wall(40.0), LAUNCH, 4.0)
    jolt = _wall(V.run_jolt(V.wall(40.0), LAUNCH, 4.0))
    for blow, closing, speed, up in (omi, jolt):
        assert blow == pytest.approx(closing, rel=0.05)
        assert speed < 1.0 and up > 0.99
    assert jolt[0] == pytest.approx(omi[0], rel=0.08)


def _upright(drive: V.Drive) -> list[float]:
    """The up axis's height at the end."""
    return [float(drive.up[-1, 1])]


def _final_speed(drive: V.Drive) -> list[float]:
    """The speed at the end."""
    return [float(drive.speed()[-1])]


def test_on_a_banked_road_it_stands_and_drives() -> None:
    ground = V.bank(math.radians(10.0))
    omi = V.reference('bank/stand', _upright, ground, V.constant(), 3.0)
    drive = V.run_jolt(ground, V.constant(), 3.0)
    assert np.linalg.norm(drive.position[-1] - drive.position[0]) < 0.25
    assert _upright(drive)[0] == pytest.approx(omi[0], abs=1e-3)
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (60.0, (0.3, 0.0, 0.0)))
    omi = V.reference('bank/drive', _final_speed, ground, inputs, 6.0)
    drive = V.run_jolt(ground, inputs, 6.0)
    assert _final_speed(drive)[0] == pytest.approx(omi[0], rel=0.08)
    # Held on the slope by the tyres' friction at no slip: within omi's 0.5 m of the line.
    assert abs(drive.position[-1, 0]) < 0.6 and abs(drive.heading()[-1]) < 0.01


def _ride(drive: V.Drive) -> list[float]:
    """The chassis's root-mean-square vertical speed after the first second, and the speed at the end."""
    return [
        float(np.sqrt((drive.velocity[drive.at(1.0) :, 1] ** 2).mean())),
        float(drive.speed()[-1]),
    ]


def test_over_bumps_it_rides_like_the_raycast_car() -> None:
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (60.0, (0.3, 0.0, 0.0)))
    start = (0.0, 0.7, 60.0)
    omi = V.reference('bumps', _ride, V.bumpy(), inputs, 6.0, start=start)
    drive = V.run_jolt(V.bumpy(), inputs, 6.0, start=start)
    jolt = _ride(drive)
    assert jolt[0] == pytest.approx(omi[0], rel=0.15)
    assert jolt[1] == pytest.approx(omi[1], rel=0.10)
    assert drive.up[:, 1].min() > 0.99
    assert abs(drive.position[-1, 0]) < 2.0


def test_the_car_is_the_one_omi_physics_builds_from_glisteel_s_spec() -> None:
    """The plain tuning and wheels map to the same Jolt car as omi_physics' own records."""
    if not engines.have_omi():
        pytest.skip('omi_physics is not installed; the plain records are the reference')
    from omi_physics import vehicle as omi_vehicle  # noqa: PLC0415 omi_physics is optional

    given = V.settings()
    omi = VehicleSettings.from_raycast(
        omi_vehicle.VehicleTuning(**V.TUNING), omi_vehicle.car_wheels(**V.WHEELS), V.MASS
    )
    assert _same(given.as_dict(), omi.as_dict())


def _same(one: object, two: object) -> bool:
    """Whether two settings dictionaries hold the same values, numbers within 1e-12."""
    if isinstance(one, dict) and isinstance(two, dict):
        return one.keys() == two.keys() and all(_same(one[k], two[k]) for k in one)
    if isinstance(one, (list, tuple)) and isinstance(two, (list, tuple)):
        return len(one) == len(two) and all(_same(a, b) for a, b in zip(one, two, strict=True))
    if isinstance(one, (int, float)) and isinstance(two, (int, float)):
        return math.isclose(one, two, rel_tol=1e-12, abs_tol=1e-12)
    return one == two
