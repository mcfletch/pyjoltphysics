"""Characters through Jolt and through omi_physics' controller on the same ground.

Each scene runs a character through both engines at 60 Hz and compares where
it goes: across a height field for 10 s, up a flight of stairs, up a walkable
slope and against a steep one, a jump's flight at gravity 42 and a fall's
landing. The two collide differently (Jolt's capsule keeps a 2 cm skin and
meets a height field's triangles; omi's meets its bilinear surface), so each
test states its band.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from omi_physics import model
from omi_physics import world as omi
from omi_physics.character import CharacterCapabilities as OmiCaps
from omi_physics.character import CharacterController

from pyjoltphysics import (
    BodyBatch,
    CharacterCapabilities,
    MotionType,
    Shape,
    World,
    WorldSettings,
)

pytestmark = pytest.mark.differential

DT = 1.0 / 60.0


@dataclasses.dataclass
class Ground:
    """A static scene: boxes (size, centre) and at most one height field (heights, origin, spacing)."""

    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = dataclasses.field(
        default_factory=list
    )
    field: tuple[np.ndarray, tuple[float, float, float], float] | None = None
    mesh: tuple[np.ndarray, np.ndarray] | None = None


@dataclasses.dataclass
class Track:
    """The character's feet at every tick, and whether it was grounded."""

    feet: np.ndarray
    grounded: np.ndarray


def run_jolt(
    ground: Ground,
    start: tuple[float, float, float],
    seconds: float,
    move: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: str = 'walk',
    gravity: float = 9.81,
    jump_at: int | None = None,
    snap: bool = True,
    **caps: Any,
) -> Track:
    capabilities = CharacterCapabilities(**caps)
    with World(WorldSettings(tick_hz=1.0 / DT, threads=1, gravity=(0.0, -gravity, 0.0))) as world:
        for size, centre in ground.boxes:
            world.add_bodies(BodyBatch.boxes([centre], size=size, motion=MotionType.STATIC))
        if ground.field is not None:
            heights, origin, spacing = ground.field
            world.add_bodies(
                BodyBatch.heightfield(heights, origin=origin, spacing=spacing, bits_per_sample=16)
            )
        if ground.mesh is not None:
            world.add_bodies(
                BodyBatch.of_shape(
                    [(0.0, 0.0, 0.0)], Shape.mesh(*ground.mesh), motion=MotionType.STATIC
                )
            )
        world.run_ticks(1)
        centre = (start[0], start[1] + 0.5 * capabilities.standHeight, start[2])
        walker = world.add_character(centre, capabilities, gravity=gravity)
        if snap:
            walker.teleport(centre, search=4.0)
        walker.set_move(move, mode)
        feet, grounded = [], []
        for tick in range(round(seconds / DT)):
            if tick == jump_at:
                walker.jump()
            world.run_ticks(1)
            feet.append(walker.feet)
            grounded.append(walker.grounded)
        return Track(np.array(feet), np.array(grounded))


def run_omi(
    ground: Ground,
    start: tuple[float, float, float],
    seconds: float,
    move: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: str = 'walk',
    gravity: float = 9.81,
    jump_at: int | None = None,
    snap: bool = True,
    **caps: Any,
) -> Track:
    world = omi.PhysicsWorld(gravity=model.Gravity(gravity=gravity, direction=(0, -1, 0)))
    for size, centre in ground.boxes:
        shape = world.add_shape(model.Shape.box(size))
        world.add_body(
            model.Motion(type=model.STATIC), collider=model.Collider(shape=shape), position=centre
        )
    if ground.field is not None:
        heights, origin, spacing = ground.field
        shape = world.add_shape(model.Shape.heightfield(heights, origin, spacing))
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    if ground.mesh is not None:
        shape = world.add_shape(model.Shape.trimesh(*ground.mesh))
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    capabilities = OmiCaps(**caps)
    centre = (start[0], start[1] + 0.5 * capabilities.standHeight, start[2])
    walker = CharacterController(world, capabilities, position=centre, gravity=gravity)
    if snap:
        walker.safe_bind(centre)
    walker.set_move(move, mode)
    feet, grounded = [], []
    for tick in range(round(seconds / DT)):
        if tick == jump_at:
            walker.jump()
        walker.update(DT)
        feet.append(walker.base())
        grounded.append(walker.grounded)
    return Track(np.array(feet), np.array(grounded))


def both(*args: Any, **kwargs: Any) -> tuple[Track, Track]:
    return run_jolt(*args, **kwargs), run_omi(*args, **kwargs)


def landscape() -> Ground:
    x = np.arange(129, dtype=np.float64)
    heights = 1.5 * np.sin(x[None, :] * 0.08) * np.cos(x[:, None] * 0.06) + 0.3 * np.sin(
        x[None, :] * 0.3
    )
    return Ground(field=(heights.astype(np.float32), (-64.0, 0.0, -64.0), 1.0))


def ramp(degrees: float) -> Ground:
    rise = math.tan(math.radians(degrees))
    points = np.array(
        [(2.0, 0.0, -5.0), (12.0, 10.0 * rise, -5.0), (12.0, 10.0 * rise, 5.0), (2.0, 0.0, 5.0)]
    )
    floor = ((40.0, 1.0, 40.0), (0.0, -0.5, 0.0))
    return Ground(boxes=[floor], mesh=(points, np.array([(0, 2, 1), (0, 3, 2)], np.uint32)))


def test_a_run_across_a_height_field_for_ten_seconds_ends_in_one_place() -> None:
    ground = landscape()
    assert ground.field is not None
    start = (-50.0, float(ground.field[0][64, 14]), 0.0)
    jolt, omi_ = both(ground, start, 10.0, move=(1.0, 0.0, 0.3), mode='run')
    # 6 m/s along the ground for 10 s; both stay on the surface throughout.
    assert jolt.grounded[30:].all() and omi_.grounded[30:].all()
    assert np.linalg.norm(jolt.feet[-1] - omi_.feet[-1]) < 0.1
    assert abs(jolt.feet[-1, 1] - omi_.feet[-1, 1]) < 0.02
    travelled = np.linalg.norm(np.diff(jolt.feet, axis=0), axis=1).sum()
    assert travelled == pytest.approx(60.0, rel=0.01)


def test_a_flight_of_stairs_is_climbed_by_both_at_walking_pace() -> None:
    floor = ((40.0, 1.0, 40.0), (0.0, -0.5, 0.0))
    steps = [
        ((0.4, 0.2 * (k + 1), 4.0), (2.0 + 0.4 * k + 0.2, 0.1 * (k + 1), 0.0)) for k in range(6)
    ]
    top = ((10.0, 1.2, 4.0), (2.0 + 2.4 + 5.0, 0.6, 0.0))
    ground = Ground(boxes=[floor, *steps, top])
    jolt, omi_ = both(ground, (0.0, 0.0, 0.0), 3.0, move=(1.0, 0.0, 0.0), stepHeight=0.35)
    assert jolt.feet[-1, 1] == pytest.approx(1.2, abs=0.05)
    assert omi_.feet[-1, 1] == pytest.approx(1.2, abs=0.05)
    assert jolt.feet[-1, 0] == pytest.approx(omi_.feet[-1, 0], abs=0.6)


@pytest.mark.parametrize(('degrees', 'climbs'), [(30.0, True), (60.0, False)])
def test_a_slope_is_climbed_or_refused_by_both(degrees: float, climbs: bool) -> None:
    jolt, omi_ = both(ramp(degrees), (0.0, 0.0, 0.0), 3.0, move=(1.0, 0.0, 0.0), maxSlope=50.0)
    for track in (jolt, omi_):
        assert bool(track.feet[-1, 1] > 2.0) is climbs
    if climbs:
        # 9 m at 3 m/s along the path: 2 m of floor, then 7 m up the slope.
        along = 7.0
        expected = (
            2.0 + along * math.cos(math.radians(degrees)),
            along * math.sin(math.radians(degrees)),
        )
        assert jolt.feet[-1, 0] == pytest.approx(expected[0], abs=0.15)
        assert jolt.feet[-1, 1] == pytest.approx(expected[1], abs=0.1)
        # omi_physics loses about 0.6 m at the foot of the slope.
        assert omi_.feet[-1, 0] == pytest.approx(jolt.feet[-1, 0], abs=0.7)
    else:
        assert abs(jolt.feet[-1, 0] - omi_.feet[-1, 0]) < 0.1


def test_a_jump_at_gravity_42_flies_the_same_parabola_in_both() -> None:
    floor = Ground(boxes=[((40.0, 1.0, 40.0), (0.0, -0.5, 0.0))])
    kw: dict[str, Any] = {
        'gravity': 42.0,
        'jump_at': 10,
        'jumpHeight': 1.5,
        'coyoteTime': 0.0,
        'jumpBuffer': 0.0,
    }
    jolt, omi_ = both(floor, (0.0, 0.0, 0.0), 1.0, **kw)
    j = jolt.feet[:, 1] - jolt.feet[9, 1]
    o = omi_.feet[:, 1] - omi_.feet[9, 1]
    flight = ~jolt.grounded & ~omi_.grounded
    flight[:10] = False
    assert flight.sum() > 10
    assert np.abs(j[flight] - o[flight]).max() < 1e-6
    assert j.max() == pytest.approx(1.5, abs=42.0 * DT * DT)
    # Jolt's capsule stops at its contact a tick before it reports the ground.
    assert (~jolt.grounded).sum() - (~omi_.grounded).sum() in (0, 1)


def test_a_fall_from_five_metres_lands_at_the_same_tick_in_both() -> None:
    floor = Ground(boxes=[((40.0, 1.0, 40.0), (0.0, -0.5, 0.0))])
    jolt, omi_ = both(floor, (0.0, 5.0, 0.0), 1.5, snap=False)
    assert int(np.argmax(jolt.grounded)) == pytest.approx(int(np.argmax(omi_.grounded)), abs=1)
    assert jolt.feet[-1, 1] == pytest.approx(omi_.feet[-1, 1], abs=0.03)
    falling = ~jolt.grounded & ~omi_.grounded
    assert np.abs(jolt.feet[falling, 1] - omi_.feet[falling, 1]).max() < 1e-6


def test_a_ledge_twice_the_step_height_is_refused_by_jolt_and_climbed_by_omi() -> None:
    """A divergence: running into a 0.6 m ledge with a 0.35 m step height.

    Jolt's character is stopped at the face, as the step height says. omi's
    controller is lifted onto the ledge by its depenetration over the edge.
    """
    floor = ((60.0, 1.0, 60.0), (0.0, -0.5, 0.0))
    ledge = ((40.0, 0.6, 40.0), (21.5, 0.3, 0.0))
    jolt, omi_ = both(
        Ground(boxes=[floor, ledge]), (0.0, 0.0, 0.0), 3.0, move=(1.0, 0.0, 0.0), mode='run'
    )
    assert jolt.feet[-1, 1] == pytest.approx(0.0, abs=0.02)
    assert jolt.feet[-1, 0] < 1.5 - 0.3 + 0.05
    assert omi_.feet[-1, 1] == pytest.approx(0.6, abs=0.02)
