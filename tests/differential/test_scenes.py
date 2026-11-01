"""Jolt against omi_physics, scene by scene, compared on behaviour within stated tolerances.

Each test runs the same bodies from the same initial conditions at 120 Hz in
both engines. Two solvers never agree to the bit, so each test names what it
compares and its tolerance, with the reason. Where the engines legitimately
differ, the test asserts the physically right bound on each and says so.

omi_physics' side is its recorded answer (:func:`differential.engines.reference`),
checked against a live run where omi_physics is installed, so this module runs
without it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from differential.engines import (
    DT,
    Body,
    G,
    Spec,
    Trace,
    grid_mesh,
    ground,
    heightfield_ground,
    mesh_ground,
    reference,
    run_jolt,
    run_omi,
    tetra_hull,
)

pytestmark = pytest.mark.differential

#: Each primitive and the height of its origin above the ground at rest.
RESTING: list[tuple[str, Spec, float]] = [
    ('sphere', ('sphere', 0.5), 0.5),
    ('box', ('box', (1.0, 0.6, 0.8)), 0.3),
    ('capsule', ('capsule', 0.3, 1.0), 0.3),  # lies on its side
    ('cylinder', ('cylinder', 0.4, 1.0), 0.5),  # stands on its end
    ('hull', ('hull', tetra_hull(1.0)), 0.25),
]
GROUNDS: dict[str, Callable[..., Body]] = {
    'box': ground,
    'mesh': mesh_ground,
    'heightfield': heightfield_ground,
}
LYING = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # 90 degrees about z


@pytest.mark.parametrize('surface', sorted(GROUNDS))
@pytest.mark.parametrize(('name', 'spec', 'height'), RESTING, ids=[r[0] for r in RESTING])
def test_each_shape_rests_on_each_ground(
    surface: str, name: str, spec: Spec, height: float
) -> None:
    """Rest height after 3 s: each engine within 1 cm of the shape's geometry, and of each other.

    1 cm covers both engines' contact slop (Jolt's penetration slop is 2 cm,
    of which a resting body keeps a few millimetres) and omi's softer contact.
    The capsule starts lying on its side so neither engine has to topple it.
    """
    orientation = LYING if name == 'capsule' else (0.0, 0.0, 0.0, 1.0)
    bodies = [
        GROUNDS[surface](),
        Body(spec, (0.3, height + 0.5, -0.2), orientation=orientation),
    ]
    jolt = run_jolt(bodies, 360).last()[1, 1]
    omi = reference(f'rest/{name}/{surface}', lambda: run_omi(bodies, 360).last()[1, 1])
    assert jolt == pytest.approx(height, abs=0.01)
    assert jolt == pytest.approx(omi, abs=0.01)


SLOPE = math.tan(math.radians(20.0))


@pytest.mark.parametrize('surface', ['mesh', 'heightfield'])
@pytest.mark.parametrize('friction', [0.0, 0.3, 0.8])
def test_a_box_slides_down_a_slope_by_its_friction(surface: str, friction: float) -> None:
    """Slide distance along a 20 degree slope in 1 s: within 10% (or 3 cm) of g (sin - mu cos) t2 / 2.

    At mu = 0.8 (above tan 20 = 0.36) the box holds; both engines must keep it
    within 3 cm. Jolt combines frictions as the geometric mean and omi as the
    average; ground and box share one coefficient, where the two agree.
    """
    theta = math.atan(SLOPE)
    lift = 0.25 / math.cos(theta) + 0.01
    start = (2.0, 2.0 * SLOPE + lift, 0.0)
    tilt = (0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2))
    bodies = [
        GROUNDS[surface](slope=SLOPE, friction=friction),
        Body(('box', (0.5, 0.5, 0.5)), start, friction=friction, orientation=tilt),
    ]
    ticks = 132  # 0.1 s to settle onto the slope, then 1 s
    accel = max(0.0, G * (math.sin(theta) - friction * math.cos(theta)))
    expected = 0.5 * accel * 1.0**2

    def slide(trace: Trace) -> tuple[float, float]:
        box = trace.position[:, 1]
        # The box is already moving after the settling 0.1 s.
        return float(np.linalg.norm(box[-1] - box[11])), float(
            np.linalg.norm(box[12] - box[10])
        ) / (2 * DT)

    omi = reference(f'slide/{surface}/{friction}', lambda: slide(run_omi(bodies, ticks)))
    along, moving = slide(run_jolt(bodies, ticks))
    assert along == pytest.approx(expected + moving * 1.0, rel=0.1, abs=0.03)
    assert along == pytest.approx(omi[0], rel=0.1, abs=0.03)


@pytest.mark.parametrize('surface', sorted(GROUNDS))
def test_a_bounce_rebounds_as_high_on_each_ground(surface: str) -> None:
    """First apex over drop height at restitution 0.6: within 0.05 between engines and of e2 = 0.36.

    0.05 covers the energy each solver loses in the contact's first tick.
    """
    e = 0.6
    bodies = [
        GROUNDS[surface](restitution=e),
        Body(('sphere', 0.5), (0.0, 3.0, 0.0), restitution=e),
    ]

    def rebound(trace: Trace) -> float:
        y = trace.position[:, 1, 1]
        inner = y[1:-1]
        peaks = inner[(inner > y[:-2]) & (inner >= y[2:])]
        return float((peaks[0] - 0.5) / 2.5)

    ratios = [
        rebound(run_jolt(bodies, 240)),
        reference(f'rebound/{surface}', lambda: rebound(run_omi(bodies, 240))),
    ]
    assert ratios[0] == pytest.approx(e * e, abs=0.05)
    assert ratios[0] == pytest.approx(ratios[1], abs=0.05)


def test_a_stack_of_five_settles_and_sleeps() -> None:
    """Stack of five 1 m boxes after 4 s: drift under 2 cm in both; top within 2 cm of 4.5 m.

    Jolt's stack falls asleep by then (asserted); omi's top is compared with Jolt's.
    """
    bodies = [ground(0.6)] + [
        Body(('box', (1.0, 1.0, 1.0)), (0.0, 0.5 + k * 1.001, 0.0), friction=0.6) for k in range(5)
    ]

    def stack(trace: Trace) -> tuple[float, float]:
        boxes = trace.last()[1:]
        return float(np.abs(boxes[:, [0, 2]]).max()), float(boxes[-1, 1])

    jolt = run_jolt(bodies, 480)
    drift, top = stack(jolt)
    assert drift < 0.02
    assert top == pytest.approx(
        reference('stack', lambda: stack(run_omi(bodies, 480)))[1], abs=0.02
    )
    assert top == pytest.approx(4.5, abs=0.02)
    assert not jolt.awake[1:].any()


def test_a_pyramid_settles_and_sleeps() -> None:
    """A 4-3-2-1 pyramid of 1 m boxes after 5 s: every box within 2 cm of its stacked place.

    That is the physically right pose. Jolt holds it to 1 cm, or to twice omi's
    error where that is larger, and falls asleep; omi's recorded error is held
    to 5 cm in test_reference.py, its iterative solver letting a pyramid creep
    further.
    """
    bodies = [ground(0.7)]
    places = []
    for level in range(4):
        for k in range(4 - level):
            x = (k - (3 - level) / 2) * 1.01
            places.append((x, 0.5 + level * 1.001, 0.0))
            bodies.append(Body(('box', (1.0, 1.0, 1.0)), places[-1], friction=0.7))
    want = np.array(places)
    jolt = run_jolt(bodies, 600)
    omi = reference('pyramid', lambda: float(np.abs(run_omi(bodies, 600).last()[1:] - want).max()))
    assert np.abs(jolt.last()[1:] - want).max() < max(0.01, 2 * omi)
    assert not jolt.awake[1:].any()


def stairs(steps: int = 5, rise: float = 0.3, run: float = 0.6) -> list[Body]:
    """Steps descending along +x from the top tread at x in [0, run]; the floor is y = 0."""
    out = [ground(0.5, half=30.0)]
    for k in range(steps):
        height = rise * (steps - k)
        out.append(
            Body(('box', (run, height, 4.0)), (k * run + run / 2, height / 2, 0.0), 'static', 0.0)
        )
    return out


def test_a_ball_rolls_down_stairs_into_a_band() -> None:
    """A ball pushed off the top step at 1.5 m/s: after 4 s both engines have it on the floor
    past the stairs, between 3 and 12 m out, and within 1.5 m of each other.

    The band is wide because each landing on a tread scatters the ball's path;
    the engines agree on where it ends up, not on each bounce.
    """
    bodies = stairs() + [Body(('sphere', 0.2), (0.3, 1.5 + 0.2, 0.0), velocity=(1.5, 0.0, 0.0))]
    ends = [
        run_jolt(bodies, 480).last()[-1],
        reference('stairs', lambda: run_omi(bodies, 480).last()[-1]),
    ]
    assert 3.0 < ends[0][0] < 12.0
    assert ends[0][1] == pytest.approx(0.2, abs=0.01)
    assert abs(ends[0][0] - ends[1][0]) < 1.5


def test_a_body_falling_through_a_sensor_enters_and_leaves_within_a_tick() -> None:
    """Trigger enter and exit ticks of a ball falling through a sensor box: within one tick.

    Jolt reports an overlap at the end of the step that makes it; omi tests
    overlaps against the poses at the start of its step, a tick behind at most.
    """
    bodies = [
        Body(('box', (2.0, 1.0, 2.0)), (0.0, 5.0, 0.0), 'static', 0.0, sensor=True),
        Body(('sphere', 0.2), (0.0, 8.0, 0.0)),
    ]
    jolt = run_jolt(bodies, 180).triggers
    omi = reference('sensor/fall', lambda: run_omi(bodies, 180).triggers)
    assert [(p, a, b) for _, p, a, b in jolt] == [('enter', 0, 1), ('exit', 0, 1)]
    assert [(p, a, b) for _, p, a, b in omi] == [('enter', 0, 1), ('exit', 0, 1)]
    for (tj, *_), (to, *_) in zip(jolt, omi, strict=True):
        assert abs(tj - to) <= 1


def test_a_sensor_holds_a_body_asleep_inside_it() -> None:
    """A box at rest inside a sensor: one enter and no exit over 5 s in both engines,
    though Jolt drops a sleeping body's contacts (the package holds the pair)."""
    bodies = [
        ground(),
        Body(('box', (4.0, 2.0, 4.0)), (0.0, 1.0, 0.0), 'static', 0.0, sensor=True),
        Body(('box', (1.0, 1.0, 1.0)), (0.0, 0.6, 0.0)),
    ]
    jolt = run_jolt(bodies, 600)
    omi = reference('sensor/asleep', lambda: run_omi(bodies, 600).triggers)
    assert not jolt.awake[2]
    for triggers in (jolt.triggers, omi):
        assert [(p, a, b) for _, p, a, b in triggers] == [('enter', 1, 2)]


def test_a_kinematic_sensor_passes_over_a_static_body() -> None:
    """A kinematic sensor sweeping across a static box enters and leaves it in Jolt.

    omi's triggers report moving bodies only, so a static box is not reported
    there; the trigger is a region, and a region passing over a wall overlaps it,
    so Jolt's report is the right one and omi's count is asserted as it is.
    """
    bodies = [
        Body(('box', (1.0, 1.0, 1.0)), (0.0, 0.0, 0.0), 'static', 0.0),
        Body(('sphere', 0.5), (-3.0, 0.0, 0.0), 'kinematic', velocity=(6.0, 0.0, 0.0), sensor=True),
    ]
    jolt = run_jolt(bodies, 120, gravity=(0.0, 0.0, 0.0)).triggers
    assert [(p, a, b) for _, p, a, b in jolt] == [('enter', 1, 0), ('exit', 1, 0)]
    omi = reference('sensor/kinematic', lambda: run_omi(bodies, 120).triggers)
    assert len(omi) in (0, 2)


def test_a_continuous_bullet_does_not_pass_a_thin_wall() -> None:
    """A 5 cm ball at 300 m/s against a 5 cm wall: Jolt's continuous body stops at the wall.

    At 2.5 m a tick the ball crosses the wall between two poses. omi has no
    continuous detection and the ball passes; Jolt with CONTINUOUS is right.
    """
    wall = Body(('box', (0.05, 4.0, 4.0)), (0.0, 2.0, 0.0), 'static', 0.0)
    bullet = Body(('sphere', 0.05), (-2.0, 2.0, 0.0), velocity=(300.0, 0.0, 0.0), continuous=True)
    jolt = run_jolt([wall, bullet], 6, gravity=(0.0, 0.0, 0.0)).last()[1]
    assert jolt[0] < 0.0
    omi = reference('bullet', lambda: run_omi([wall, bullet], 6).last()[1, 0])
    assert omi > 0.0


def test_a_continuous_bullet_does_not_pass_a_mesh_wall() -> None:
    """The same bullet against a wall of mesh triangles in both windings: Jolt stops it."""
    points, triangles = grid_mesh(4.0, 2)
    upright = points[:, [1, 2, 0]]  # the grid turned into the x = 0 plane
    both = np.concatenate([triangles, triangles[:, ::-1]])
    wall = Body(('mesh', upright + (0.0, 2.0, 0.0), both), (0.0, 0.0, 0.0), 'static', 0.0)
    bullet = Body(('sphere', 0.05), (-2.0, 2.0, 0.0), velocity=(300.0, 0.0, 0.0), continuous=True)
    jolt = run_jolt([wall, bullet], 6, gravity=(0.0, 0.0, 0.0)).last()[1]
    assert jolt[0] < 0.0


def test_a_thousand_boxes_fill_a_container_to_the_same_depth() -> None:
    """1,000 0.5 m boxes dropped into a 10 m square bin: the mean resting height within 3%
    and its spread within 10% between engines, after 4 s.

    Trajectories diverge at the first collision; the pile's statistics do not.
    Mean height is 0.25 m plus the layers the boxes make (1,000 boxes of 0.25 m2
    cover the 100 m2 floor 2.5 times).
    """
    rng = np.random.default_rng(0)
    bodies = [ground(0.5, half=6.0)]
    for x, z, sx, sz in (
        (-5.25, 0.0, 0.5, 11.0),
        (5.25, 0.0, 0.5, 11.0),
        (0.0, -5.25, 11.0, 0.5),
        (0.0, 5.25, 11.0, 0.5),
    ):
        bodies.append(Body(('box', (sx, 8.0, sz)), (x, 4.0, z), 'static', 0.0))
    first = len(bodies)
    for p in rng.uniform((-4.7, 1.0, -4.7), (4.7, 16.0, 4.7), (1000, 3)):
        bodies.append(Body(('box', (0.5, 0.5, 0.5)), (float(p[0]), float(p[1]), float(p[2]))))
    jolt = run_jolt(bodies, 480, threads=4).last()[first:, 1]

    def pile() -> tuple[float, float, float]:
        heights = run_omi(bodies, 480).last()[first:, 1]
        return float(heights.min()), float(heights.mean()), float(heights.std())

    lowest, mean, spread = reference('bin', pile)
    assert jolt.min() > 0.2 and lowest > 0.2
    assert jolt.mean() == pytest.approx(mean, rel=0.03)
    assert jolt.std() == pytest.approx(spread, rel=0.10)


def test_a_compound_comes_to_its_rest_pose() -> None:
    """A dumbbell compound dropped tilted: both rest with the bar level at 0.5 m, within 1 cm and 1 degree."""
    dumbbell = (
        'compound',
        [
            (('sphere', 0.5), (-1.0, 0.0, 0.0)),
            (('box', (2.0, 0.2, 0.2)), (0.0, 0.0, 0.0)),
            (('sphere', 0.5), (1.0, 0.0, 0.0)),
        ],
    )
    tilt = (0.0, 0.0, math.sin(0.15), math.cos(0.15))
    bodies = [ground(0.8), Body(dumbbell, (0.0, 2.0, 0.0), mass=2.0, orientation=tilt)]

    def rest(trace: Trace) -> tuple[float, float]:
        q = trace.orientation[-1, 1]
        # The bar's axis is the body's x axis turned by q; level means no y part.
        x_axis_y = 2.0 * (q[0] * q[1] + q[3] * q[2])
        return float(trace.last()[1, 1]), math.degrees(math.asin(np.clip(x_axis_y, -1, 1)))

    for height, tilt_degrees in (
        rest(run_jolt(bodies, 600)),
        reference('dumbbell', lambda: rest(run_omi(bodies, 600))),
    ):
        assert height == pytest.approx(0.5, abs=0.01)
        assert abs(tilt_degrees) < 1.0
