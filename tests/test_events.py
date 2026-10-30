"""Events: one begin and one end per pair, persist on request, sleep held as touching."""

from __future__ import annotations

import numpy as np

from pyjoltphysics import (
    EVENT_DTYPE,
    BodyBatch,
    BodyFlag,
    EndReason,
    EventKind,
    EventPhase,
    MotionType,
    event_times,
)
from pyjoltphysics.testing import MakeWorld, floor, grid


def of(events: np.ndarray, kind: EventKind, phase: EventPhase) -> np.ndarray:
    chosen: np.ndarray = events[(events['kind'] == kind) & (events['phase'] == phase)]
    return chosen


def test_the_record_is_48_bytes() -> None:
    assert EVENT_DTYPE.itemsize == 48
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
