"""What leaves the world with a body: vehicles, pins, joints and drag, by remove_bodies or remove_ragdolls."""

from __future__ import annotations

import numpy as np
from test_vehicles import car_settings

from pyjoltphysics import (
    BodyBatch,
    Joint,
    RagdollMode,
    RagdollSettings,
    SoftBodySettings,
    World,
)
from pyjoltphysics.testing import MakeWorld, floor, latest


def chassis(world: World) -> int:
    (uid,), _ = world.add_bodies(
        BodyBatch.boxes([(0.0, 1.0, 0.0)], size=(1.8, 0.6, 4.0), mass=1000.0)
    )
    return int(uid)


def busy(world: World) -> None:
    """New bodies on the freed body slots, stepped: a constraint left on a freed body reads them."""
    world.add_bodies(BodyBatch.spheres([(i * 1.0, 3.0, 0.0) for i in range(40)], radius=0.3))
    world.run_ticks(30)


def test_a_second_vehicle_on_a_chassis_is_rejected(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(50.0))
    body = chassis(world)
    world.add_vehicle(body, car_settings())
    world.add_vehicle(body, car_settings())
    world.run_ticks(2)
    assert world.stats()['vehicles'] == 1
    assert world.stats()['rejected'] == 1
    world.remove_bodies([body])
    world.run_ticks(2)
    assert world.stats()['vehicles'] == 0
    busy(world)
    assert world.error is None


def test_a_vehicle_on_a_ragdoll_bone_goes_with_the_ragdoll(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(50.0))
    doll = world.add_ragdoll(
        RagdollSettings.humanoid(), position=(0.0, 1.0, 0.0), mode=RagdollMode.LIMP
    )
    world.run_ticks(2)
    world.add_vehicle(int(doll.bones[0]), car_settings())
    world.run_ticks(2)
    assert world.stats()['vehicles'] == 1
    world.remove_ragdolls([doll.id])
    world.run_ticks(30)
    assert world.stats()['vehicles'] == 0
    assert len(latest(world).vehicles.id) == 0
    busy(world)
    assert world.error is None


def cloth(world: World) -> int:
    settings = SoftBodySettings.cloth(width=1.0, depth=1.0, columns=6, rows=6, mass=1.0)
    return int(world.add_soft_body(settings, position=(0.0, 2.0, 0.0)).uid)


def test_a_soft_body_is_no_chassis(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(50.0))
    uid = cloth(world)
    world.run_ticks(2)
    world.add_vehicle(uid, car_settings())
    world.run_ticks(10)
    assert world.stats()['vehicles'] == 0
    assert world.stats()['rejected'] == 1
    assert world.error is None


def test_a_joint_to_a_soft_body_is_rejected(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    world.add_bodies(floor(50.0))
    uid = cloth(world)
    (sphere,), _ = world.add_bodies(BodyBatch.spheres([(0.0, 3.0, 0.0)], radius=0.3))
    world.run_ticks(1)
    world.add_joint(Joint.fixed(uid, int(sphere)))
    world.add_joint(Joint.fixed(int(sphere), uid))
    world.run_ticks(10)
    assert world.stats()['rejected'] == 2
    assert len(latest(world).joints.id) == 0


def fall_after_a_ragdoll(make_world: MakeWorld, drag: float) -> list[float]:
    world = make_world(threads=1)
    doll = world.add_ragdoll(
        RagdollSettings.humanoid(), position=(0.0, 100.0, 0.0), mode=RagdollMode.LIMP
    )
    world.run_ticks(1)
    if drag:
        world.set_drag(doll.bones, [drag] * len(doll.bones))
    world.run_ticks(1)
    world.remove_ragdolls([doll.id])
    world.run_ticks(1)
    uids, _ = world.add_bodies(
        BodyBatch.spheres([(i * 2.0, 100.0, 0.0) for i in range(len(doll.bones))], radius=0.2)
    )
    world.run_ticks(120)
    snapshot = latest(world)
    return [float(snapshot.linear_velocity[snapshot.rows[int(u)], 1]) for u in uids]


def test_bodies_on_a_removed_ragdolls_slots_fall_without_its_drag(make_world: MakeWorld) -> None:
    """The bones' drag goes with them; new bodies on their Jolt slots fall freely for 1 s."""
    assert np.allclose(fall_after_a_ragdoll(make_world, drag=5.0), -9.81, atol=0.01)


def test_a_soft_body_pinned_to_a_bone_lets_go_with_the_ragdoll(make_world: MakeWorld) -> None:
    world = make_world(threads=2)
    doll = world.add_ragdoll(
        RagdollSettings.humanoid(), position=(3.0, 1.0, 0.0), mode=RagdollMode.LIMP
    )
    settings = SoftBodySettings.cloth(width=1.0, depth=1.0, columns=6, rows=6, mass=1.0)
    world.add_soft_body(
        settings,
        position=(3.0, 2.0, 0.0),
        pins=((0,), [(0.0, 0.0, 0.0)]),
        pin_bodies=[int(doll.bones[0])],
    )
    world.run_ticks(2)
    world.remove_ragdolls([doll.id])
    busy(world)
    assert world.stats()['soft_bodies'] == 1
    assert world.error is None
