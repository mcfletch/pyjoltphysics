"""Events: one begin and one end per pair, persist on request, sleep held as touching."""

from __future__ import annotations

import numpy as np
import pytest

from pyjoltphysics import (
    EVENT_DTYPE,
    BodyBatch,
    BodyFlag,
    EndReason,
    EventKind,
    EventPhase,
    MotionType,
    Shape,
    event_times,
)
from pyjoltphysics.testing import MakeWorld, floor, grid


def of(events: np.ndarray, kind: EventKind, phase: EventPhase) -> np.ndarray:
    chosen: np.ndarray = events[(events['kind'] == kind) & (events['phase'] == phase)]
    return chosen


def test_the_record_is_72_bytes() -> None:
    assert EVENT_DTYPE.itemsize == 72
    assert EVENT_DTYPE.fields is not None
    assert EVENT_DTYPE.fields['point'][0] == np.dtype(('<f8', (3,)))
    assert EVENT_DTYPE.names == (
        'kind',
        'phase',
        'reason',
        'tick',
        'a',
        'b',
        'point',
        'normal',
        'approach',
        'impulse',
        'material_a',
        'material_b',
        'depth',
        'reserved',
    )


def test_a_landing_is_one_begin_with_its_contact(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (ground,), _ = world.add_bodies(floor())
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 2.0, 0.0)], size=(1, 1, 1)))
    world.run_ticks(120)
    events = world.drain_events()
    begins = of(events, EventKind.CONTACT, EventPhase.BEGIN)
    assert len(begins) == 1
    begin = begins[0]
    assert (begin['a'], begin['b']) == (ground, box)
    np.testing.assert_allclose(begin['normal'], (0.0, 1.0, 0.0), atol=1e-3)
    assert abs(begin['point'][1]) < 0.05
    # About sqrt(2 g h) for a 1.5 m fall.
    assert 4.5 < begin['approach'] < 6.0
    assert begin['impulse'] > 0.0
    assert len(of(events, EventKind.CONTACT, EventPhase.END)) == 0
    assert len(of(events, EventKind.CONTACT, EventPhase.PERSIST)) == 0
    assert event_times(begins, 120.0)[0] == begin['tick'] / 120.0


def test_falling_asleep_does_not_end_a_resting_contact(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.6, 0.0)], size=(1, 1, 1)))
    world.run_ticks(400)
    events = world.drain_events()
    sleeps = of(events, EventKind.STATE, EventPhase.SLEEP)
    assert sleeps['a'].tolist() == [box]
    assert sleeps['b'].tolist() == [0]
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 1
    assert len(of(events, EventKind.CONTACT, EventPhase.END)) == 0
    # Woken while still resting: a wake, and neither an end nor a second begin.
    world.activate([box])
    world.run_ticks(5)
    events = world.drain_events()
    assert of(events, EventKind.STATE, EventPhase.WAKE)['a'].tolist() == [box]
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 0
    assert len(of(events, EventKind.CONTACT, EventPhase.END)) == 0


def test_a_sleeping_body_knocked_away_ends_its_contact(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (ground,), _ = world.add_bodies(floor())
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.6, 0.0)], size=(1, 1, 1)))
    world.run_ticks(400)
    world.drain_events()
    world.add_impulses([box], (0.0, 10.0, 0.0))
    world.run_ticks(10)
    events = world.drain_events()
    ends = of(events, EventKind.CONTACT, EventPhase.END)
    assert [(e['a'], e['b']) for e in ends] == [(ground, box)]
    assert ends[0]['reason'] == EndReason.NONE
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 0


def test_persist_only_for_bodies_that_report(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    world.add_bodies(floor())
    (quiet,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    (loud,), _ = world.add_bodies(BodyBatch.boxes([(3.0, 0.5, 0.0)], flags=BodyFlag.REPORT))
    world.run_ticks(30)
    persists = of(world.drain_events(), EventKind.CONTACT, EventPhase.PERSIST)
    assert set(persists['b'].tolist()) == {loud}
    assert quiet not in persists['b']
    # One record a tick for the pair.
    assert len(persists) == len(np.unique(persists['tick']))
    assert len(persists) >= 28


def test_a_sensor_reports_enter_and_exit(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (sensor,), _ = world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 5.0, 0.0)], size=(2, 1, 2), motion=MotionType.STATIC, flags=BodyFlag.SENSOR
        )
    )
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 8.0, 0.0)], radius=0.2))
    world.run_ticks(240)
    events = world.drain_events()
    enters = of(events, EventKind.TRIGGER, EventPhase.BEGIN)
    exits = of(events, EventKind.TRIGGER, EventPhase.END)
    assert [(e['a'], e['b']) for e in enters] == [(sensor, ball)]
    assert [(e['a'], e['b']) for e in exits] == [(sensor, ball)]
    assert exits[0]['tick'] > enters[0]['tick']
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 0


def test_a_body_asleep_inside_a_sensor_stays_inside(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    (sensor,), _ = world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 1.0, 0.0)], size=(4, 2, 4), motion=MotionType.STATIC, flags=BodyFlag.SENSOR
        )
    )
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.6, 0.0)]))
    world.run_ticks(400)
    events = world.drain_events()
    assert of(events, EventKind.STATE, EventPhase.SLEEP)['a'].tolist() == [box]
    assert [(e['a'], e['b']) for e in of(events, EventKind.TRIGGER, EventPhase.BEGIN)] == [
        (sensor, box)
    ]
    assert len(of(events, EventKind.TRIGGER, EventPhase.END)) == 0


def test_removing_a_body_ends_its_pairs(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (ground,), _ = world.add_bodies(floor())
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.run_ticks(10)
    world.drain_events()
    world.remove_bodies([box])
    world.run_ticks(3)
    events = world.drain_events()
    ends = of(events, EventKind.CONTACT, EventPhase.END)
    assert [(e['a'], e['b'], e['reason']) for e in ends] == [(ground, box, EndReason.REMOVED)]
    assert ends[0]['tick'] == 11
    assert len(of(events, EventKind.STATE, EventPhase.SLEEP)) == 0


def test_a_full_ring_drops_persist_records_first(make_world: MakeWorld) -> None:
    world = make_world(threads=1, event_capacity=50, allow_sleeping=False)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes(grid(30, spacing=2.0, y0=0.5), flags=BodyFlag.REPORT))
    world.run_ticks(60)
    events = world.drain_events()
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 30
    assert len(events) == 50
    assert world.stats()['events_dropped'] > 0


def test_stats_count_the_records_the_ring_holds_until_a_drain(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False)
    world.add_bodies(floor())
    world.add_bodies(BodyBatch.boxes(grid(4, spacing=2.0, y0=0.5), flags=BodyFlag.REPORT))
    assert world.stats()['events_held'] == 0
    world.run_ticks(30)
    held = world.stats()['events_held']
    assert held == len(world.drain_events()) > 0
    assert world.stats()['events_held'] == 0


def test_begin_and_end_records_exceed_the_capacity_rather_than_drop(
    make_world: MakeWorld,
) -> None:
    world = make_world(threads=1, event_capacity=5)
    world.add_bodies(floor())
    uids, _ = world.add_bodies(BodyBatch.boxes(grid(20, spacing=2.0, y0=0.5)))
    world.run_ticks(5)
    world.remove_bodies(uids)
    world.run_ticks(1)
    events = world.drain_events()
    assert len(of(events, EventKind.CONTACT, EventPhase.BEGIN)) == 20
    assert len(of(events, EventKind.CONTACT, EventPhase.END)) == 20


def test_event_order_is_the_same_with_any_thread_count(make_world: MakeWorld) -> None:
    runs = []
    for threads in (1, 8):
        world = make_world(threads=threads)
        world.add_bodies(floor())
        world.add_bodies(BodyBatch.boxes(grid(200, spacing=0.8, y0=1.0), size=(0.5, 0.5, 0.5)))
        world.run_ticks(300)
        runs.append(world.drain_events())
    assert runs[0].tobytes() == runs[1].tobytes()


def test_a_compound_touching_with_several_children_is_one_pair(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (ground,), _ = world.add_bodies(floor())
    legs = Shape.compound(
        [(Shape.box((0.2, 1.0, 0.2)), (x, 0.0, z)) for x in (-0.5, 0.5) for z in (-0.5, 0.5)]
        + [(Shape.box((1.2, 0.1, 1.2)), (0.0, 0.55, 0.0))]
    )
    (table,), _ = world.add_bodies(BodyBatch.of_shape([(0.0, 1.0, 0.0)], legs, mass=10.0))
    world.run_ticks(240)
    world.add_impulses([table], (0.0, 60.0, 0.0))
    world.run_ticks(240)
    events = world.drain_events()
    begins = of(events, EventKind.CONTACT, EventPhase.BEGIN)
    ends = of(events, EventKind.CONTACT, EventPhase.END)
    # Four legs land at once and again after the hop: one begin and one end per touch.
    assert [(e['a'], e['b']) for e in begins] == [(ground, table)] * 2
    assert [(e['a'], e['b']) for e in ends] == [(ground, table)]


def test_a_sensor_added_over_a_sleeping_body_reports_it(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    world.add_bodies(floor())
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.5, 0.0)]))
    world.run_ticks(300)
    snapshot = world.latest()
    assert snapshot is not None and not snapshot.awake[snapshot.rows[int(box)]]
    world.drain_events()
    (sensor,), _ = world.add_bodies(
        BodyBatch.boxes(
            [(0.0, 1.0, 0.0)], size=(3, 2, 3), motion=MotionType.STATIC, flags=BodyFlag.SENSOR
        )
    )
    world.run_ticks(300)
    events = world.drain_events()
    enters = of(events, EventKind.TRIGGER, EventPhase.BEGIN)
    assert [(e['a'], e['b']) for e in enters] == [(sensor, box)]
    assert enters[0]['tick'] <= 302
    # The box falls asleep again, held inside.
    assert len(of(events, EventKind.TRIGGER, EventPhase.END)) == 0
    snapshot = world.latest()
    assert snapshot is not None and not snapshot.awake[snapshot.rows[int(box)]]


def test_a_kinematic_sensor_reports_static_bodies(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    (wall,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.0, 0.0)], motion=MotionType.STATIC))
    (sensor,), _ = world.add_bodies(
        BodyBatch.spheres(
            [(-3.0, 0.0, 0.0)],
            radius=0.5,
            motion=MotionType.KINEMATIC,
            linear_velocity=(6.0, 0.0, 0.0),
            flags=BodyFlag.SENSOR,
        )
    )
    world.run_ticks(120)
    events = world.drain_events()
    pairs = [(e['a'], e['b']) for e in of(events, EventKind.TRIGGER, EventPhase.BEGIN)]
    assert pairs == [(sensor, wall)]
    assert len(of(events, EventKind.TRIGGER, EventPhase.END)) == 1


def test_a_continuous_bullet_stops_at_a_thin_mesh_wall(make_world: MakeWorld) -> None:
    world = make_world(threads=1, gravity=(0.0, 0.0, 0.0))
    # A single two-sided quad in the x = 0 plane.
    points = np.array([(0, -2, -2), (0, 2, -2), (0, 2, 2), (0, -2, 2)], float)
    quad = np.array([(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)], np.uint32)
    world.add_bodies(BodyBatch.meshes([(0.0, 0.0, 0.0)], points, quad))
    uids, _ = world.add_bodies(
        BodyBatch.spheres(
            [(-2.0, 0.0, 0.5), (-2.0, 0.0, -0.5)],
            radius=0.05,
            linear_velocity=(400.0, 0.0, 0.0),
            flags=[BodyFlag.CONTINUOUS, BodyFlag.NONE],
        )
    )
    world.run_ticks(4)
    snapshot = world.latest()
    assert snapshot is not None
    x = snapshot.position[snapshot.rows_of(uids), 0]
    assert x[0] < 0.0  # stopped
    assert x[1] > 0.0  # tunnelled


def test_event_points_keep_millimetres_five_kilometres_out(make_world: MakeWorld) -> None:
    world = make_world(threads=1)
    far = 5000.123456
    world.add_bodies(BodyBatch.plane(position=(far, 0.0, far)))
    world.add_bodies(BodyBatch.spheres([(far, 1.0, far)], radius=0.5))
    world.run_ticks(120)
    begin = of(world.drain_events(), EventKind.CONTACT, EventPhase.BEGIN)[0]
    assert begin['point'].dtype == np.float64
    np.testing.assert_allclose(begin['point'], (far, 0.0, far), atol=2e-3)


def test_persist_records_carry_the_impulse_estimated_for_their_tick(make_world: MakeWorld) -> None:
    world = make_world(threads=1, allow_sleeping=False, tick_hz=60.0)
    world.add_bodies(floor())
    (box,), _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 0.5, 0.0)], mass=2.0, flags=BodyFlag.REPORT)
    )
    world.run_ticks(60)
    world.drain_events()
    world.run_ticks(10)
    resting = of(world.drain_events(), EventKind.CONTACT, EventPhase.PERSIST)['impulse']
    # Holding 2 kg up for a tick of 1/60 s is its weight times the tick.
    assert len(resting) == 10
    np.testing.assert_allclose(resting, 2.0 * 9.81 / 60.0, rtol=0.25)
    world.set_velocities([box], [(0.0, -3.0, 0.0)], [(0.0, 0.0, 0.0)])
    world.run_ticks(1)
    pressed = of(world.drain_events(), EventKind.CONTACT, EventPhase.PERSIST)['impulse']
    # Stopping 2 kg closing at 3 m/s.
    assert pressed.max() > 2.0 * 3.0 * 0.9


def test_a_resting_contact_reports_how_far_the_bodies_overlap(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=120.0, threads=1, allow_sleeping=False)
    world.add_bodies(floor(10.0, flags=BodyFlag.REPORT))
    world.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(1.0, 1.0, 1.0), mass=2.0))
    world.run_ticks(600)
    world.drain_events()
    world.run_ticks(3)
    events = world.drain_events()
    persist = events[events['phase'] == EventPhase.PERSIST]
    # The box rests sunk by the penetration slop, 2 cm by default.
    assert len(persist) == 3
    assert persist['depth'] == pytest.approx(0.02, abs=2e-3)
