"""Changing a body's mass, damping, gravity factor and flags, and saving and restoring the world."""

from __future__ import annotations

import numpy as np
import pytest

from pyjoltphysics import BodyBatch, BodyFlag, EventKind, EventPhase, MotionType, World
from pyjoltphysics.testing import MakeWorld, floor, latest


def _state(world: World, uid: int) -> tuple[np.ndarray, np.ndarray]:
    snap = latest(world)
    row = snap.rows[uid]
    return snap.position[row].copy(), snap.linear_velocity[row].copy()


def test_gravity_factor_zero_floats(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 10.0, 0.0)], radius=0.5))
    world.set_motion([ball], gravity_factor=0.0)
    world.run_ticks(60)
    position, velocity = _state(world, ball)
    assert position[1] == pytest.approx(10.0, abs=1e-6)
    assert np.allclose(velocity, 0.0)


def test_gravity_factor_halves_the_fall(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, allow_sleeping=False)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 10.0, 0.0), (5.0, 10.0, 0.0)], radius=0.5))
    world.set_motion(uids[:1], gravity_factor=0.5)
    world.run_ticks(30)
    half, _ = _state(world, int(uids[0]))
    full, _ = _state(world, int(uids[1]))
    assert (10.0 - half[1]) == pytest.approx((10.0 - full[1]) / 2.0, rel=1e-3)


def test_linear_damping_slows_a_body(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0), (0.0, 5.0, 0.0)], radius=0.5,
                                                 linear_velocity=(4.0, 0.0, 0.0)))
    world.set_motion(uids[:1], linear_damping=2.0)
    world.run_ticks(60)
    _, damped = _state(world, int(uids[0]))
    _, free = _state(world, int(uids[1]))
    assert free[0] == pytest.approx(4.0, rel=1e-4)
    assert 0.0 < damped[0] < 1.0


def test_angular_damping_slows_a_spin(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, gravity=(0.0, 0.0, 0.0))
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0)], radius=0.5,
                                                    angular_velocity=(0.0, 3.0, 0.0)))
    world.set_motion([ball], angular_damping=3.0)
    world.run_ticks(60)
    snap = latest(world)
    assert 0.0 < snap.angular_velocity[snap.rows[ball], 1] < 0.5


def test_mass_changes_the_answer_to_an_impulse(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, gravity=(0.0, 0.0, 0.0))
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 0.0, 0.0), (0.0, 5.0, 0.0)], radius=0.5,
                                                 mass=1.0))
    world.set_motion(uids[:1], mass=4.0)
    world.add_impulses(uids, (2.0, 0.0, 0.0))
    world.run_ticks(1)
    _, heavy = _state(world, int(uids[0]))
    _, light = _state(world, int(uids[1]))
    assert light[0] == pytest.approx(2.0, rel=1e-4)
    assert heavy[0] == pytest.approx(0.5, rel=1e-4)


def test_nan_keeps_a_value(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, allow_sleeping=False)
    (ball,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 10.0, 0.0)], radius=0.5,
                                                    gravity_factor=0.0))
    world.set_motion([ball], linear_damping=0.1)
    world.run_ticks(10)
    position, _ = _state(world, ball)
    assert position[1] == pytest.approx(10.0, abs=1e-6)


def test_set_motion_refuses_bad_values(make_world: MakeWorld) -> None:
    world = make_world()
    with pytest.raises(ValueError, match='mass'):
        world.set_motion([1], mass=-1.0)
    with pytest.raises(ValueError, match='damping'):
        world.set_motion([1], linear_damping=float('inf'))


def test_report_flag_turns_persist_records_on_and_off(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1, allow_sleeping=False)
    world.add_bodies(floor(10.0))
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.25, 0.0)], size=(0.5, 0.5, 0.5)))
    world.run_ticks(10)
    world.drain_events()
    world.run_ticks(5)
    persists = world.drain_events()
    assert not np.any(persists['phase'] == EventPhase.PERSIST)
    world.set_flags([box], BodyFlag.REPORT, True)
    world.run_ticks(5)
    events = world.drain_events()
    persist = events[(events['kind'] == EventKind.CONTACT) & (events['phase'] == EventPhase.PERSIST)]
    assert len(persist) >= 4
    world.set_flags([box], BodyFlag.REPORT, False)
    world.run_ticks(1)
    world.drain_events()
    world.run_ticks(5)
    assert not np.any(world.drain_events()['phase'] == EventPhase.PERSIST)


def test_filter_contacts_flag_set_later_asks_the_filter(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    asked: list[tuple[int, int]] = []

    def verdict(a: int, b: int, point: object, normal: object, approach: float) -> bool:
        asked.append((a, b))
        return False

    world.set_contact_filter(verdict)
    uids, _ = world.add_bodies(BodyBatch.spheres([(0.0, 1.0, 0.0), (0.0, 0.0, 0.0)], radius=0.5,
                                                 gravity_factor=0.0))
    world.set_velocities(uids[:1], linear=(0.0, -2.0, 0.0))
    world.set_flags(uids, BodyFlag.FILTER_CONTACTS, True)
    world.run_ticks(30)
    assert asked
    position, _ = _state(world, int(uids[0]))
    assert position[1] < 0.0


def test_set_flags_takes_only_report_and_no_sleep(make_world: MakeWorld) -> None:
    world = make_world()
    with pytest.raises(ValueError, match='REPORT'):
        world.set_flags([1], BodyFlag.SENSOR, True)


def test_no_sleep_flag_keeps_a_body_awake(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    world.add_bodies(floor(10.0))
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 0.25, 0.0)], size=(0.5, 0.5, 0.5)))
    world.set_flags([box], BodyFlag.NO_SLEEP, True)
    world.run_ticks(180)
    snap = latest(world)
    assert bool(snap.awake[snap.rows[box]])


def test_restore_puts_bodies_back(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    world.add_bodies(floor(10.0))
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(0.5, 0.5, 0.5)))
    world.run_ticks(20)
    saved = world.save_state()
    assert isinstance(saved, bytes) and len(saved) > 0
    at_save, moving = _state(world, box)
    world.run_ticks(40)
    later, _ = _state(world, box)
    assert later[1] < at_save[1]
    world.restore_state(saved)
    world.run_ticks(1)
    # One tick after the restore is one tick after the save.
    again = World(world.settings)
    try:
        again.add_bodies(floor(10.0))
        again.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(0.5, 0.5, 0.5)))
        again.run_ticks(21)
        expected, _ = _state(again, box)
    finally:
        again.stop()
    restored, _ = _state(world, box)
    assert np.array_equal(restored, expected)
    assert moving[1] < 0.0


def test_restore_refuses_a_state_of_another_layout(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    world.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(0.5, 0.5, 0.5)))
    world.run_ticks(1)
    saved = world.save_state()
    world.add_bodies(BodyBatch.boxes([(3.0, 5.0, 0.0)], size=(0.5, 0.5, 0.5)))
    world.run_ticks(1)
    world.restore_state(saved)
    world.run_ticks(1)
    assert world.stats()['rejected'] >= 1


@pytest.mark.parametrize('slop', [0.02, 0.005])
def test_penetration_slop_bounds_how_far_a_resting_box_sinks(make_world: MakeWorld,
                                                             slop: float) -> None:
    world = make_world(tick_hz=120.0, threads=1, penetration_slop=slop)
    world.add_bodies(floor(10.0))
    (box,), _ = world.add_bodies(BodyBatch.boxes([(0.0, 5.0, 0.0)], size=(1.0, 1.0, 1.0),
                                                 mass=2.0))
    world.run_ticks(480)
    position, _ = _state(world, box)
    assert 0.5 - slop - 1e-3 <= position[1] <= 0.5 + 1e-4


def test_penetration_slop_refuses_negative(make_world: MakeWorld) -> None:
    with pytest.raises(ValueError, match='penetration_slop'):
        make_world(penetration_slop=-1.0)


def test_flush_applies_batches_without_stepping(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=60.0, threads=1)
    world.run_ticks(1)
    (box,), seq = world.add_bodies(BodyBatch.boxes([(0.0, 0.0, 0.0)], size=(2.0, 2.0, 2.0),
                                                   motion=MotionType.STATIC))
    assert world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0)) is None
    world.flush()
    hit = world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0))
    assert hit is not None and hit.uid == box
    assert world.tick == 1
    world.remove_bodies([box])
    world.flush()
    assert world.raycast((0.0, 5.0, 0.0), (0.0, -1.0, 0.0)) is None
    applied = world.drain_applied()
    assert [int(s) for s, _t in applied] == [seq, seq + 1]


def test_flush_beside_a_free_running_world(make_world: MakeWorld) -> None:
    world = make_world(tick_hz=240.0, threads=2)
    world.add_bodies(BodyBatch.spheres(np.random.default_rng(1).uniform(-5, 5, (200, 3)), radius=0.2,
                                       gravity_factor=0.0, linear_velocity=(0.0, 0.0, 1.0)))
    world.start()
    for x in range(50):
        (post,), _ = world.add_bodies(BodyBatch.boxes([(x * 10.0 + 100.0, 0.0, 0.0)],
                                                      size=(1.0, 1.0, 1.0), motion=MotionType.STATIC))
        world.flush()
        hit = world.raycast((x * 10.0 + 100.0, 5.0, 0.0), (0.0, -1.0, 0.0))
        assert hit is not None and hit.uid == post
    world.stop()
