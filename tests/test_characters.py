"""Characters: the behaviour of omi_physics' controller, on Jolt's CharacterVirtual.

The assertions follow omi_physics' own character tests (tests/test_character*.py
there) wherever they are about physics: speeds, steps, slopes, the jump's
parabola, landings, knock-downs, slides, pushing and platforms. A character is
commanded by batches applied at the next tick and read from the snapshot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyjoltphysics import (
    BodyBatch,
    BodyFlag,
    Character,
    CharacterCapabilities,
    CharacterState,
    EventKind,
    EventPhase,
    MotionType,
    RayHit,
    Shape,
    World,
)
from pyjoltphysics.testing import MakeWorld, floor


def hit_uid(hit: RayHit | None) -> int:
    """The uid a ray met; the ray has to meet something."""
    assert hit is not None
    return hit.uid


G = 9.81


class Rig:
    """A lockstep world at ``tick_hz`` with a floor, and characters added to it."""

    def __init__(
        self, make_world: MakeWorld, tick_hz: float = 60.0, gravity: float = G, **settings: object
    ) -> None:
        self.world: World = make_world(tick_hz=tick_hz, gravity=(0.0, -gravity, 0.0), **settings)
        self.dt = 1.0 / tick_hz
        self.gravity = gravity
        (self.floor,), _ = self.world.add_bodies(floor())

    def character(
        self, at: tuple[float, float, float] = (0.0, 0.0, 0.0), settle: bool = True, **caps: object
    ) -> Character:
        capabilities = CharacterCapabilities(**caps)  # type: ignore[arg-type]  # test keywords
        centre = (at[0], at[1] + 0.5 * capabilities.standHeight + 0.05, at[2])
        made = self.world.add_character(centre, capabilities, gravity=self.gravity)
        if settle:
            self.step(0.5)
            assert made.grounded
        return made

    def box(
        self, size: tuple[float, float, float], centre: tuple[float, float, float], **values: object
    ) -> int:
        values.setdefault('motion', MotionType.STATIC)
        (uid,), _ = self.world.add_bodies(BodyBatch.boxes([centre], size=size, **values))
        return int(uid)

    def step(self, seconds: float) -> None:
        self.world.run_ticks(round(seconds / self.dt))

    def ticks(self, count: int = 1) -> None:
        self.world.run_ticks(count)

    def x_of(self, uid: int) -> float:
        snapshot = self.world.latest()
        assert snapshot is not None
        return float(snapshot.position[snapshot.rows[uid], 0])


def state_of(character: Character) -> CharacterState:
    """The character's state, read afresh (mypy narrows a property read twice)."""
    return character.state


def ramp_mesh(degrees: float, length: float = 30.0, width: float = 20.0) -> Shape:
    """A plane through the origin rising along +x at ``degrees``, as two triangles."""
    rise = math.tan(math.radians(degrees))
    h = length / 2
    points = np.array(
        [
            (-h, -h * rise, -width / 2),
            (h, h * rise, -width / 2),
            (h, h * rise, width / 2),
            (-h, -h * rise, width / 2),
        ]
    )
    return Shape.mesh(points, [(0, 2, 1), (0, 3, 2)])


# -- standing and walking -------------------------------------------------------


def test_a_character_dropped_above_the_floor_lands_and_stands(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    assert state_of(walker) is CharacterState.STANDING
    assert walker.feet[1] == pytest.approx(0.0, abs=0.05)
    assert walker.ground_uid == rig.floor
    np.testing.assert_allclose(walker.ground_normal, (0.0, 1.0, 0.0), atol=1e-4)


def test_speed_tiers_walk_run_sprint(make_world: MakeWorld) -> None:
    covered = {}
    for mode in ('walk', 'run', 'sprint'):
        rig = Rig(make_world)
        walker = rig.character()
        start = walker.position[0]
        walker.set_move((1.0, 0.0, 0.0), mode)
        rig.step(1.0)
        covered[mode] = walker.position[0] - start
    assert covered['walk'] == pytest.approx(3.0, rel=0.02)
    assert covered['run'] == pytest.approx(6.0, rel=0.02)
    assert covered['sprint'] == pytest.approx(9.0, rel=0.02)


def test_a_move_given_a_speed_goes_at_it_whatever_the_mode(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character(walkSpeed=1.5)
    walker.set_move((1.0, 0.0, 0.0), 'run', speed=2.5)
    rig.ticks(1)
    start = walker.position[0]
    rig.ticks(60)
    assert walker.position[0] - start == pytest.approx(2.5, abs=1e-3)
    assert walker.velocity[0] == pytest.approx(2.5, abs=1e-3)


def test_acceleration_reaches_the_walking_speed_in_speed_over_acceleration(
    make_world: MakeWorld,
) -> None:
    rig = Rig(make_world)
    walker = rig.character(acceleration=6.0)
    walker.set_move((0.0, 0.0, 1.0))
    rig.ticks(15)
    assert walker.velocity[2] == pytest.approx(1.5, abs=0.11)
    rig.ticks(30)
    assert walker.velocity[2] == pytest.approx(3.0, abs=1e-3)


def test_steps_up_a_small_ledge(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    rig.box((40.0, 0.3, 40.0), (21.0, 0.15, 0.0))
    walker = rig.character(stepHeight=0.4)
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(2.0)
    assert walker.position[0] > 2.0
    assert walker.feet[1] == pytest.approx(0.3, abs=0.05)
    assert walker.grounded


def test_blocked_by_a_step_taller_than_the_step_height(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    rig.box((40.0, 0.6, 40.0), (21.5, 0.3, 0.0))
    walker = rig.character(stepHeight=0.35)
    walker.set_move((1.0, 0.0, 0.0), 'run')
    rig.step(3.0)
    assert walker.position[0] < 1.5 - 0.3 + 0.05
    assert walker.feet[1] == pytest.approx(0.0, abs=0.05)


def test_a_walkable_slope_is_climbed_and_a_steeper_one_is_not(make_world: MakeWorld) -> None:
    for degrees, climbs in ((30.0, True), (60.0, False)):
        rig = Rig(make_world)
        rig.world.add_bodies(
            BodyBatch.of_shape([(6.0, 0.0, 0.0)], ramp_mesh(degrees), motion=MotionType.STATIC)
        )
        walker = rig.character(at=(0.0, 0.0, 0.0), maxSlope=50.0)
        walker.set_move((1.0, 0.0, 0.0))
        rig.step(4.0)
        if climbs:
            assert walker.feet[1] > 2.0
        else:
            assert walker.feet[1] < 0.5


def test_the_pace_up_a_walkable_slope_is_the_pace_on_the_flat(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    rig.world.add_bodies(
        BodyBatch.of_shape(
            [(0.0, 0.0, 0.0)], ramp_mesh(20.0, length=60.0), motion=MotionType.STATIC
        )
    )
    walker = rig.character(at=(-2.0, -2.0 * math.tan(math.radians(20.0)), 0.0))
    walker.set_move((1.0, 0.0, 0.0))
    rig.ticks(10)
    start = np.array(walker.feet)
    rig.step(1.0)
    travelled = float(np.linalg.norm(np.array(walker.feet) - start))
    assert travelled == pytest.approx(3.0, rel=0.03)


def test_a_steep_slope_slides_the_character_down(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    rig.world.add_bodies(
        BodyBatch.of_shape(
            [(0.0, 0.0, 0.0)], ramp_mesh(60.0, length=20.0), motion=MotionType.STATIC
        )
    )
    walker = rig.world.add_character(
        (4.0, 4.0 * math.tan(math.radians(60.0)) + 1.5, 0.0),
        CharacterCapabilities(maxSlope=50.0),
        gravity=G,
    )
    rig.step(0.3)
    x0 = walker.position[0]
    rig.step(0.5)
    assert walker.position[0] < x0 - 0.3
    assert not walker.grounded
    assert state_of(walker) is CharacterState.AIRBORNE


# -- jumping and falling ----------------------------------------------------------


@pytest.mark.parametrize('tick_hz', [60.0, 120.0, 240.0])
def test_a_jump_follows_the_closed_form_parabola_at_gravity_42(
    make_world: MakeWorld, tick_hz: float
) -> None:
    gravity, height = 42.0, 1.5
    rig = Rig(make_world, tick_hz=tick_hz, gravity=gravity)
    walker = rig.character(jumpHeight=height, coyoteTime=0.0, jumpBuffer=0.0)
    y0 = walker.feet[1]
    walker.jump()
    launch = math.sqrt(2.0 * gravity * height)
    heights = []
    for _ in range(int(2.0 * launch / gravity * tick_hz) + 5):
        rig.ticks(1)
        if walker.grounded:
            break
        heights.append(walker.feet[1])
    t = np.arange(1, len(heights) + 1) / tick_hz
    expected = y0 + launch * t - 0.5 * gravity * t * t
    assert np.max(np.abs(np.array(heights) - expected)) < 1e-6
    assert max(heights) - y0 == pytest.approx(height, abs=0.5 * gravity / tick_hz**2 * 2 + 1e-6)
    flight = (len(heights) + 1) / tick_hz
    assert flight == pytest.approx(2.0 * launch / gravity, abs=1.0 / tick_hz)
    assert walker.landing == pytest.approx(launch, abs=1.01 * gravity / tick_hz)


def test_a_fall_covers_the_distance_gravity_gives_it(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.world.add_character((0.0, 30.0, 0.0), CharacterCapabilities(), gravity=G)
    rig.ticks(1)
    start = walker.position[1]
    for k in range(1, 31):
        rig.ticks(1)
        assert start - walker.position[1] == pytest.approx(
            0.5 * G * (k * rig.dt) ** 2 + G * rig.dt * rig.dt * k, abs=1e-6
        )


def test_a_fall_stops_gaining_speed_at_terminal_velocity(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.world.add_character(
        (0.0, 900.0, 0.0), CharacterCapabilities(terminalVelocity=10.0), gravity=G
    )
    rig.ticks(120)
    assert walker.velocity[1] == pytest.approx(-10.0, abs=1e-4)


def test_a_constant_gravity_of_42_lands_a_fall_at_its_closed_form_speed(
    make_world: MakeWorld,
) -> None:
    rig = Rig(make_world, tick_hz=120.0, gravity=42.0)
    walker = rig.world.add_character(
        (0.0, 0.9 + 0.05 + 5.0, 0.0), CharacterCapabilities(), gravity=42.0
    )
    landed = None
    for _ in range(240):
        rig.ticks(1)
        if walker.landing is not None:
            landed = walker.landing
            break
    assert landed == pytest.approx(math.sqrt(2.0 * 42.0 * 5.0), rel=0.03)


def test_landing_is_reported_once_and_as_an_event(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.world.add_character((0.0, 3.0, 0.0), CharacterCapabilities(), gravity=G)
    reports = []
    for _ in range(90):
        rig.ticks(1)
        reports.append(walker.landing)
    assert sum(r is not None for r in reports) == 1
    events = rig.world.drain_events()
    lands = events[(events['kind'] == EventKind.CHARACTER) & (events['phase'] == EventPhase.LAND)]
    assert len(lands) == 1
    assert lands['a'][0] == walker.id
    assert lands['b'][0] == rig.floor
    assert lands['approach'][0] == pytest.approx(
        next(r for r in reports if r is not None), abs=1e-5
    )


def test_no_double_jump(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character(coyoteTime=0.0, jumpBuffer=0.0)
    walker.jump()
    rig.ticks(5)
    rising = walker.velocity[1]
    walker.jump()
    rig.ticks(1)
    assert walker.velocity[1] < rising


def test_a_jump_asked_for_just_before_landing_fires_on_landing(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    # Feet 0.35 m up: 0.27 s to land, the jump asked for 0.15 s in.
    walker = rig.world.add_character(
        (0.0, 1.25, 0.0), CharacterCapabilities(jumpBuffer=0.2), gravity=G
    )
    rig.ticks(9)
    assert not walker.grounded
    walker.jump()
    for _ in range(60):
        rig.ticks(1)
        if walker.velocity[1] > 1.0:
            break
    assert walker.velocity[1] > 1.0


def test_jump_follows_slope_rises_jump_height_above_a_falling_slope(make_world: MakeWorld) -> None:
    def apex_drop(follows: bool) -> float:
        rig = Rig(make_world)
        rig.world.add_bodies(
            BodyBatch.of_shape(
                [(0.0, 0.0, 0.0)], ramp_mesh(-20.0, length=80.0), motion=MotionType.STATIC
            )
        )
        walker = rig.character(
            at=(-20.0, 20.0 * math.tan(math.radians(20.0)), 0.0),
            jumpFollowsSlope=follows,
            coyoteTime=0.0,
        )
        walker.set_move((1.0, 0.0, 0.0), 'run')
        rig.step(0.5)
        start = walker.feet[1]
        walker.jump()
        top = start
        for _ in range(60):
            rig.ticks(1)
            top = max(top, float(walker.feet[1]))
        return float(top - start)

    assert apex_drop(True) < apex_drop(False) - 0.3


# -- states: slide, knock-down, crouch ----------------------------------------------


def test_a_slide_keeps_low_then_rises_and_carries_its_speed(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    walker.set_move((1.0, 0.0, 0.0), 'run')
    rig.step(0.5)
    walker.slide(0.8, 0.3)
    rig.ticks(1)
    assert state_of(walker) is CharacterState.SLIDING
    assert walker.height == pytest.approx(1.0, abs=1e-6)
    speeds, progress = [], []
    for _ in range(46):
        rig.ticks(1)
        speeds.append(walker.velocity[0])
        progress.append(walker.slide_progress)
    assert speeds[0] > 0.8 * 6.0
    assert speeds[-1] < speeds[0]
    assert progress == sorted(progress)
    assert progress[-1] == pytest.approx(1.0, abs=0.03)
    rig.ticks(2)
    assert state_of(walker) is CharacterState.STANDING
    assert walker.height == pytest.approx(1.8, abs=1e-6)
    events = rig.world.drain_events()
    assert (events['phase'][events['kind'] == EventKind.CHARACTER] == EventPhase.SLIDE).sum() == 1


def test_a_jump_asked_for_long_before_landing_is_forgotten(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.world.add_character(
        (0.0, 3.0, 0.0), CharacterCapabilities(jumpBuffer=0.12), gravity=G
    )
    rig.ticks(2)
    walker.jump()
    for _ in range(60):
        rig.ticks(1)
        assert walker.velocity[1] <= 1e-6


def test_a_slide_cannot_start_in_the_air(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.world.add_character((0.0, 20.0, 0.0), CharacterCapabilities(), gravity=G)
    rig.ticks(2)
    walker.slide(0.8)
    rig.ticks(1)
    assert state_of(walker) is CharacterState.AIRBORNE


def test_knocked_down_lies_low_ignores_input_and_gets_up(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    walker.knock_down()
    walker.set_move((0.0, 0.0, 1.0))
    rig.ticks(1)
    assert state_of(walker) is CharacterState.DOWN
    assert walker.height == pytest.approx(1.0, abs=1e-6)
    at = walker.position
    rig.step(0.5)
    assert float(np.linalg.norm(np.subtract(walker.position, at))) < 1e-4
    assert walker.feet[1] == pytest.approx(0.0, abs=0.05)
    walker.jump()
    rig.ticks(1)
    assert state_of(walker) is CharacterState.DOWN
    walker.get_up()
    rig.ticks(1)
    assert state_of(walker) is CharacterState.STANDING
    assert walker.height == pytest.approx(1.8, abs=1e-6)
    rig.ticks(30)
    assert walker.velocity[2] == pytest.approx(3.0, abs=0.05)
    phases = rig.world.drain_events()['phase']
    assert EventPhase.DOWN in phases and EventPhase.UP in phases


def test_get_up_restarts_at_a_start_speed_along_the_last_heading(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    walker.set_move((1.0, 0.0, 0.0))
    rig.ticks(5)
    walker.knock_down()
    walker.set_move((0.0, 0.0, 0.0))
    rig.ticks(10)
    walker.get_up(start_speed=4.0)
    rig.ticks(1)
    assert walker.velocity[0] == pytest.approx(4.0 * (1 - 6 * rig.dt), abs=0.1)
    assert walker.velocity[2] == pytest.approx(0.0, abs=1e-6)


def test_a_crouch_under_a_low_ceiling_cannot_stand(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    walker.set_crouch(True)
    rig.ticks(1)
    rig.box((10.0, 0.5, 10.0), (0.0, 1.45, 0.0))
    rig.ticks(1)
    walker.set_crouch(False)
    rig.ticks(2)
    assert state_of(walker) is CharacterState.CROUCHING
    assert walker.height == pytest.approx(1.0, abs=1e-6)


def test_knockback_changes_the_velocity_by_impulse_over_mass(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character(mass=80.0, pushFriction=0.0)
    walker.knockback((160.0, 0.0, 0.0))
    rig.ticks(2)
    assert walker.velocity[0] == pytest.approx(2.0, abs=0.02)
    lifted = rig.character(at=(5.0, 0.0, 0.0), mass=40.0)
    lifted.knockback((0.0, 200.0, 0.0))
    rig.ticks(1)
    assert not lifted.grounded
    assert lifted.velocity[1] == pytest.approx(5.0 - G * rig.dt / 2, abs=0.05)


# -- pushing, platforms, other characters ------------------------------------------


def test_a_walker_pushes_a_light_crate_at_walking_speed(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    crate = rig.box((0.8, 0.8, 0.8), (1.5, 0.4, 0.0), motion=MotionType.DYNAMIC, mass=20.0)
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(1.0)
    start = rig.x_of(crate)
    rig.step(2.0)
    assert (rig.x_of(crate) - start) / 2.0 == pytest.approx(3.0, rel=0.15)


def test_a_crate_heavier_than_push_mass_blocks_the_walker(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character(pushMass=100.0)
    crate = rig.box((0.8, 0.8, 0.8), (1.5, 0.4, 0.0), motion=MotionType.DYNAMIC, mass=200.0)
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(3.0)
    assert rig.x_of(crate) == pytest.approx(1.5, abs=0.02)
    assert walker.position[0] < 1.5 - 0.4 - 0.3 + 0.05


def test_a_thrown_crate_shoves_the_walker_and_is_slowed_by_it(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    crate = rig.box(
        (0.8, 0.8, 0.8),
        (3.0, 0.9, 0.0),
        motion=MotionType.DYNAMIC,
        mass=20.0,
        gravity_factor=0.0,
        linear_velocity=(-8.0, 0.0, 0.0),
    )
    rig.step(0.5)
    snapshot = rig.world.latest()
    assert snapshot is not None
    # Jolt's character contact moves the capsule out of the crate as it comes
    # in, and the character's pushes take the crate's speed.
    assert walker.position[0] < -0.2
    assert snapshot.linear_velocity[snapshot.rows[crate], 0] > -4.0


def test_a_character_enters_and_leaves_a_trigger(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    zone = rig.box((1.0, 2.0, 4.0), (2.0, 1.0, 0.0), flags=BodyFlag.SENSOR)
    walker = rig.character()
    rig.world.drain_events()
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(2.0)
    events = rig.world.drain_events()
    mine = events[(events['kind'] == EventKind.TRIGGER) & (events['a'] == zone)]
    assert mine['b'].tolist() == [walker.id, walker.id]
    assert mine['phase'].tolist() == [EventPhase.BEGIN, EventPhase.END]


def test_a_character_proxy_touches_no_solid_body(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    rig.box((0.8, 0.8, 0.8), (3.0, 0.4, 0.0), motion=MotionType.DYNAMIC, mass=20.0)
    walker = rig.character()
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(2.0)
    events = rig.world.drain_events()
    assert not (
        (events['kind'] == EventKind.CONTACT)
        & ((events['a'] == walker.id) | (events['b'] == walker.id))
    ).any()


def test_a_moving_wall_pushes_a_standing_character(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    wall = rig.box((0.4, 2.0, 4.0), (-1.0, 1.0, 0.0), motion=MotionType.KINEMATIC)
    walker = rig.character()
    rig.world.set_velocities([wall], linear=(1.0, 0.0, 0.0))
    rig.step(2.0)
    assert walker.position[0] == pytest.approx(rig.x_of(wall) + 0.2 + 0.3, abs=0.05)


def test_a_walker_on_a_moving_platform_is_carried(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    platform = rig.box((4.0, 0.4, 4.0), (0.0, 1.0, 0.0), motion=MotionType.KINEMATIC)
    walker = rig.world.add_character((0.0, 1.2 + 0.95, 0.0), CharacterCapabilities(), gravity=G)
    rig.step(0.5)
    assert walker.ground_uid == platform
    rig.world.set_velocities([platform], linear=(1.0, 0.0, 0.0))
    rig.ticks(2)
    x0 = walker.position[0]
    rig.step(1.0)
    assert walker.position[0] - x0 == pytest.approx(1.0, abs=0.03)
    assert walker.grounded


def test_a_lift_raises_the_walker_standing_on_it(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    lift = rig.box((4.0, 0.4, 4.0), (0.0, 1.0, 0.0), motion=MotionType.KINEMATIC)
    walker = rig.world.add_character((0.0, 1.2 + 0.95, 0.0), CharacterCapabilities(), gravity=G)
    rig.step(0.5)
    feet = walker.feet[1]
    rig.world.set_velocities([lift], linear=(0.0, 1.0, 0.0))
    rig.step(1.0)
    assert walker.feet[1] - feet == pytest.approx(1.0, abs=0.05)
    assert walker.grounded


def test_characters_of_one_pass_group_walk_through_each_other(make_world: MakeWorld) -> None:
    for group_b, passes in ((7, True), (8, False)):
        rig = Rig(make_world)
        a = rig.world.add_character(
            (0.0, 0.95, 0.0), CharacterCapabilities(), gravity=G, pass_group=7
        )
        b = rig.world.add_character(
            (3.0, 0.95, 0.0), CharacterCapabilities(), gravity=G, pass_group=group_b
        )
        rig.step(0.3)
        a.set_move((1.0, 0.0, 0.0))
        b.set_move((-1.0, 0.0, 0.0))
        rig.step(1.5)
        if passes:
            assert a.position[0] > b.position[0] + 1.0
        else:
            assert a.position[0] < b.position[0] - 0.55


def test_a_heavy_character_pushes_a_light_one(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    heavy = rig.world.add_character((0.0, 0.95, 0.0), CharacterCapabilities(mass=160.0), gravity=G)
    light = rig.world.add_character((0.7, 0.95, 0.0), CharacterCapabilities(mass=40.0), gravity=G)
    rig.step(0.3)
    heavy.set_move((1.0, 0.0, 0.0))
    rig.step(2.0)
    assert light.position[0] > 2.0
    assert heavy.position[0] > 1.5


def test_an_ignored_body_is_walked_through(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    wall = rig.box((0.4, 3.0, 6.0), (2.0, 1.5, 0.0))
    walker = rig.character()
    walker.set_ignore([wall])
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(2.0)
    assert walker.position[0] > 4.0


def test_walking_into_a_wall_begins_a_touch_and_walking_away_ends_it(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    wall = rig.box((0.4, 3.0, 6.0), (2.0, 1.5, 0.0))
    walker = rig.character()
    rig.world.drain_events()
    walker.set_move((1.0, 0.0, 0.0))
    rig.step(1.0)
    walker.set_move((-1.0, 0.0, 0.0))
    rig.step(1.0)
    events = rig.world.drain_events()
    mine = events[(events['kind'] == EventKind.CHARACTER) & (events['b'] == wall)]
    assert mine['phase'].tolist() == [EventPhase.BEGIN, EventPhase.END]


def test_a_ray_meets_the_character_proxy_unless_it_has_none(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    seen = rig.character()
    unseen = rig.world.add_character(
        (5.0, 0.95, 0.0), CharacterCapabilities(), gravity=G, proxy=False
    )
    rig.step(0.3)
    hit = rig.world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)
    assert hit is not None and hit.uid == seen.id
    hit = rig.world.raycast((5.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)
    assert hit is not None and hit.uid == rig.floor
    assert unseen.id != seen.id


def test_per_character_gravity(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    slow = rig.world.add_character((0.0, 50.0, 0.0), CharacterCapabilities(), gravity=5.0)
    fast = rig.world.add_character((3.0, 50.0, 0.0), CharacterCapabilities(), gravity=20.0)
    rig.ticks(1)
    s0, f0 = slow.position[1], fast.position[1]
    rig.ticks(30)
    t = 30 * rig.dt
    assert s0 - slow.position[1] == pytest.approx(5.0 * (t * t / 2 + rig.dt * t), abs=1e-6)
    assert f0 - fast.position[1] == pytest.approx(20.0 * (t * t / 2 + rig.dt * t), abs=1e-6)


def test_fly_ignores_gravity_and_collision(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    flyer = rig.character()
    flyer.set_fly(True)
    rig.ticks(1)
    y0 = flyer.position[1]
    rig.ticks(60)
    assert flyer.position[1] == pytest.approx(y0, abs=1e-9)
    assert state_of(flyer) is CharacterState.FLYING


def test_a_swimmer_sinks_slowly_and_stops_on_the_bottom(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    swimmer = rig.world.add_character((0.0, 5.0, 0.0), CharacterCapabilities(), gravity=G)
    swimmer.set_swim(True, buoyancy=0.0)
    rig.ticks(1)
    rig.step(1.0)
    assert state_of(swimmer) is CharacterState.SWIMMING
    assert 4.0 > swimmer.position[1] > 2.0
    rig.step(5.0)
    assert swimmer.feet[1] == pytest.approx(0.0, abs=0.05)


def test_teleport_seats_the_character_on_the_floor_below(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walker = rig.character()
    walker.teleport((10.0, 3.0, 0.0))
    rig.ticks(1)
    assert walker.position[0] == pytest.approx(10.0, abs=1e-6)
    assert walker.feet[1] == pytest.approx(0.0, abs=0.05)
    assert walker.grounded


def test_a_removed_character_leaves_the_snapshot(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    a = rig.character()
    b = rig.character(at=(4.0, 0.0, 0.0))
    a.remove()
    rig.ticks(1)
    snapshot = rig.world.latest()
    assert snapshot is not None
    assert snapshot.characters.id.tolist() == [b.id]
    assert hit_uid(rig.world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 10.0)) == rig.floor


def test_the_snapshot_holds_characters_as_read_only_arrays(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    walkers = [rig.character(at=(2.0 * k, 0.0, 0.0), settle=False) for k in range(3)]
    rig.step(0.5)
    snapshot = rig.world.latest()
    assert snapshot is not None
    rows = snapshot.characters
    assert rows.id.tolist() == [w.id for w in walkers]
    assert rows.position.shape == (3, 3) and rows.position.dtype == np.float64
    assert rows.velocity.shape == (3, 3)
    assert (rows.state == CharacterState.STANDING).all()
    assert rows.grounded.all()
    with pytest.raises(ValueError):
        rows.position[0, 0] = 1.0


def test_a_character_standing_on_another_names_it_as_its_ground(make_world: MakeWorld) -> None:
    rig = Rig(make_world)
    below = rig.character()
    above = rig.world.add_character((0.0, 3.0, 0.0), CharacterCapabilities(), gravity=G)
    rig.step(1.0)
    assert above.feet[1] == pytest.approx(1.8, abs=0.1)
    assert above.ground_uid == below.id
