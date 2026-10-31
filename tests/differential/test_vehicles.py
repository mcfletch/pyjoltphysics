"""glisteel's car on Jolt against omi_physics' RaycastVehicle, scenario by scenario.

The two models differ: omi_physics pushes the chassis with forces at ray
contacts and holds the tyres to a friction circle at once; Jolt solves the
suspension, a slip-curve tyre and an engine, clutch and differential as
constraints. The comparisons are of behaviour glisteel's handling and feel
tests rely on, with the tolerances in openglcontext/plans/NATIVE-PHYSICS.md
(P6a as built).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from differential import vehicles as V

pytestmark = pytest.mark.differential

LAUNCH = V.phases((0.3, (0.0, 0.0, 0.0)), (60.0, (1.0, 0.0, 0.0)))


def _both(
    ground: V.Ground, inputs: V.Inputs, seconds: float, **kwargs: Any
) -> tuple[V.Drive, V.Drive]:
    return V.run_omi(ground, inputs, seconds, **kwargs), V.run_jolt(
        ground, inputs, seconds, **kwargs
    )


def test_standing_still_it_settles_at_the_same_ride_height() -> None:
    omi, jolt = _both(V.FLAT, V.constant(), 2.0)
    assert jolt.position[-1, 1] == pytest.approx(omi.position[-1, 1], abs=1e-3)
    assert jolt.speed()[-1] < 0.01 and omi.speed()[-1] < 0.01
    assert jolt.grounded[-1] == omi.grounded[-1] == 4


def test_it_accelerates_to_speed_in_a_straight_line() -> None:
    omi, jolt = _both(V.FLAT, LAUNCH, 8.3)
    for t in (2.3, 4.3, 8.3):
        assert jolt.speed()[jolt.at(t) - 1] == pytest.approx(omi.speed()[omi.at(t) - 1], rel=0.12)
    # Straight: a road's width is 7 m; both stay within half a metre of the line.
    assert abs(jolt.position[-1, 0]) < 0.5 and abs(jolt.heading()[-1]) < 0.01


def test_it_brakes_to_a_stop_over_the_same_distance_per_speed() -> None:
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (3.3, (1.0, 0.0, 0.0)), (60.0, (0.0, 0.0, 1.0)))
    found = []
    for drive in _both(V.FLAT, inputs, 8.0):
        i = drive.at(3.3)
        v0 = drive.speed()[i]
        stop = i + int(np.argmax(drive.speed()[i:] < 0.2))
        found.append(abs(drive.position[stop, 2] - drive.position[i, 2]) / v0**2)
        assert drive.speed()[-1] < 0.05
    assert found[1] == pytest.approx(found[0], rel=0.08)


@pytest.mark.parametrize(('accelerate', 'steer'), [(1.0, 0.5), (2.0, 0.3), (2.0, 1.0)])
def test_steady_cornering_pulls_the_same_lateral_g(accelerate: float, steer: float) -> None:
    inputs = V.phases(
        (0.3, (0.0, 0.0, 0.0)), (0.3 + accelerate, (1.0, 0.0, 0.0)), (60.0, (0.0, steer, 0.0))
    )
    found = []
    for drive in _both(V.FLAT, inputs, accelerate + 3.3):
        i, j = drive.at(accelerate + 1.8), len(drive.time) - 1
        yaw = (drive.heading()[j] - drive.heading()[i]) / (drive.time[j] - drive.time[i])
        found.append((yaw, drive.speed()[i:].mean() * yaw / 9.81))
        assert drive.up[:, 1].min() > 0.99
    (omi_yaw, omi_g), (jolt_yaw, jolt_g) = found
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


def test_a_jump_flies_as_far_and_lands_upright() -> None:
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0)), (60.0, (0.2, 0.0, 0.0)))
    omi, jolt = _both(V.ramp(20.0), inputs, 5.0)
    (o0, o1), (j0, j1) = _flight(omi), _flight(jolt)
    assert jolt.position[:, 1].max() == pytest.approx(omi.position[:, 1].max(), abs=0.15)
    assert jolt.time[j1] - jolt.time[j0] == pytest.approx(omi.time[o1] - omi.time[o0], rel=0.12)
    assert jolt.position[j1, 2] == pytest.approx(omi.position[o1, 2], rel=0.08)
    assert jolt.up[:, 1].min() > 0.9 and jolt.up[-1, 1] > 0.999
    assert jolt.grounded[-1] == 4


def test_a_wall_stops_it_with_the_blow_of_its_closing_speed() -> None:
    omi, jolt = _both(V.wall(40.0), LAUNCH, 4.0)
    for drive in (omi, jolt):
        k = int(np.argmax(drive.impact))
        assert drive.impact[k] == pytest.approx(drive.speed()[k - 2], rel=0.05)
        assert drive.speed()[-1] < 1.0 and drive.up[-1, 1] > 0.99
    assert jolt.impact.max() == pytest.approx(omi.impact.max(), rel=0.08)


def test_on_a_banked_road_it_stands_and_drives() -> None:
    angle = math.radians(10.0)
    omi, jolt = _both(V.bank(angle), V.constant(), 3.0)
    assert np.linalg.norm(jolt.position[-1] - jolt.position[0]) < 0.25
    assert jolt.up[-1, 1] == pytest.approx(omi.up[-1, 1], abs=1e-3)
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (60.0, (0.3, 0.0, 0.0)))
    omi, jolt = _both(V.bank(angle), inputs, 6.0)
    assert jolt.speed()[-1] == pytest.approx(omi.speed()[-1], rel=0.08)
    # The slip-curve tyre takes 0.5 degrees of slip to hold the slope: 2.8 m downhill over 200 m.
    assert abs(jolt.position[-1, 0]) < 3.5 and abs(jolt.heading()[-1]) < 0.03


def test_over_bumps_it_rides_like_the_raycast_car() -> None:
    inputs = V.phases((0.3, (0.0, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (60.0, (0.3, 0.0, 0.0)))
    omi, jolt = _both(V.bumpy(), inputs, 6.0, start=(0.0, 0.7, 60.0))
    rms = [float(np.sqrt((d.velocity[d.at(1.0) :, 1] ** 2).mean())) for d in (omi, jolt)]
    assert rms[1] == pytest.approx(rms[0], rel=0.25)
    assert jolt.speed()[-1] == pytest.approx(omi.speed()[-1], rel=0.10)
    assert jolt.up[:, 1].min() > 0.99
    assert abs(jolt.position[-1, 0]) < 4.0
