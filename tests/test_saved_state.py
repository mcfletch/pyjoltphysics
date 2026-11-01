"""save_state and restore_state: the saved bytes are checked before Jolt reads them."""

from __future__ import annotations

import random
import struct
from collections.abc import Callable

import pytest

from pyjoltphysics import BodyBatch, World
from pyjoltphysics.testing import MakeWorld, floor, latest

HEADER = 32


def resting_spheres(make_world: MakeWorld) -> World:
    """Eight spheres on a floor, five ticks in, so the state holds contacts."""
    world = make_world()
    world.add_bodies(floor(5.0))
    world.add_bodies(BodyBatch.spheres([(x * 0.6, 0.24, 0.0) for x in range(8)], radius=0.25))
    world.run_ticks(5)
    return world


def test_a_saved_state_carries_its_format_length_and_hash(make_world: MakeWorld) -> None:
    data = resting_spheres(make_world).save_state()
    magic, version, _jolt, length, _hash = struct.unpack_from('<4sIQQQ', data)
    assert magic == b'PJST'
    assert version == 1
    assert length == len(data) - HEADER


def test_a_state_restores_the_poses_it_saved(make_world: MakeWorld) -> None:
    world = resting_spheres(make_world)
    data = world.save_state()
    before = latest(world).position.copy()
    world.run_ticks(30)
    rejected = world.stats()['rejected']
    world.restore_state(data)
    world.flush()
    assert world.stats()['rejected'] == rejected
    assert (latest(world).position == before).all()


@pytest.mark.parametrize(
    'damage',
    [
        pytest.param(lambda d: d[:-1], id='truncated'),
        pytest.param(lambda d: d + b'\0', id='trailing byte'),
        pytest.param(lambda d: d[:HEADER] + d[HEADER + 1 :] + b'\0', id='shifted'),
        pytest.param(lambda d: d[:100] + bytes([d[100] ^ 1]) + d[101:], id='bit flip'),
        pytest.param(lambda d: b'XJST' + d[4:], id='magic'),
        pytest.param(lambda d: d[:4] + struct.pack('<I', 2) + d[8:], id='format version'),
        pytest.param(lambda d: d[:8] + struct.pack('<Q', 0) + d[16:], id='jolt version'),
        pytest.param(lambda d: d[:HEADER], id='header only'),
        pytest.param(lambda d: b'', id='empty'),
    ],
)
def test_a_damaged_state_is_refused_before_jolt_reads_it(
    make_world: MakeWorld, damage: Callable[[bytes], bytes]
) -> None:
    data = resting_spheres(make_world).save_state()
    world = resting_spheres(make_world)
    with pytest.raises(ValueError, match='state'):
        world.restore_state(damage(data))


def test_a_state_of_other_bodies_is_rejected_when_applied(make_world: MakeWorld) -> None:
    data = resting_spheres(make_world).save_state()
    world = resting_spheres(make_world)
    world.add_bodies(BodyBatch.spheres([(0.0, 3.0, 0.0)]))
    world.run_ticks(1)
    rejected = world.stats()['rejected']
    world.restore_state(data)
    world.run_ticks(1)
    assert world.stats()['rejected'] == rejected + 1


def corrupt(good: bytes, rng: random.Random, kind: int) -> bytes:
    """One of the review's damage classes, applied past the header."""
    data = bytearray(good)
    if kind == 0:
        return bytes(data[: rng.randrange(HEADER, len(data))])
    if kind == 1:
        for _ in range(rng.randrange(1, 8)):
            i = rng.randrange(HEADER, len(data))
            data[i] ^= 1 << rng.randrange(8)
        return bytes(data)
    if kind == 2:
        for _ in range(4):
            i = rng.randrange(HEADER, len(data) - 4)
            data[i : i + 4] = b'\xff\xff\xff\xff' if rng.random() < 0.5 else rng.randbytes(4)
        return bytes(data)
    return bytes(data[:HEADER]) + rng.randbytes(rng.randrange(0, 2 * len(good)))


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_corrupted_states_never_reach_jolt(make_world: MakeWorld, seed: int) -> None:
    """Truncations, bit flips, overwrites and random tails, 300 a seed: each is refused, the world steps on."""
    world = resting_spheres(make_world)
    good = world.save_state()
    rng = random.Random(seed)
    refused = 0
    for trial in range(300):
        data = corrupt(good, rng, trial % 4)
        if data == good:
            continue
        with pytest.raises(ValueError):
            world.restore_state(data)
        refused += 1
        if trial % 25 == 0:
            world.run_ticks(1)
    world.restore_state(good)
    world.run_ticks(3)
    assert refused >= 290
    assert world.error is None
