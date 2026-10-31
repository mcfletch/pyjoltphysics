"""Scenes as arrays, shared by the test suites and the benchmarks.

>>> from pyjoltphysics.testing import grid
>>> grid(4).shape
(4, 3)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import numpy as np

from pyjoltphysics.bodies import BodyBatch
from pyjoltphysics.enums import MotionType
from pyjoltphysics.world import Snapshot, World

__all__ = [
    'Digest',
    'MakeWorld',
    'character_digest',
    'floor',
    'grid',
    'latest',
    'pose_digest',
    'vehicle_digest',
]

#: The ``make_world(**settings)`` factory the test suites' fixture returns.
MakeWorld = Callable[..., World]


class Digest(Protocol):
    """A hashlib object: anything with ``update(bytes)``."""

    def update(self, data: bytes, /) -> None: ...


D = TypeVar('D', bound=Digest)


def latest(world: World) -> Snapshot:
    """Return the world's newest snapshot; an AssertionError before the first tick."""
    snapshot = world.latest()
    assert snapshot is not None, 'the world has run no tick'
    return snapshot


def floor(half: float = 100.0, **values: Any) -> BodyBatch:
    """Return a static box floor ``2 * half`` metres square whose top face is y = 0.

    ``values`` are further :class:`~pyjoltphysics.BodyBatch` fields.
    """
    return BodyBatch.boxes(
        [(0.0, -0.5, 0.0)], size=(2 * half, 1.0, 2 * half), motion=MotionType.STATIC, **values
    )


def grid(count: int, spacing: float = 1.1, y0: float = 1.0) -> np.ndarray:
    """Return ``count`` positions on a square grid, in layers ``spacing`` apart from ``y0`` up."""
    side = int(np.ceil(np.sqrt(count)))
    i = np.arange(count)
    return np.stack(
        [
            (i % side - side / 2) * spacing,
            y0 + (i // (side * side)) * spacing,
            (i // side % side - side / 2) * spacing,
        ],
        axis=1,
    ).astype(np.float64)


def pose_digest(snapshot: Snapshot, digest: D) -> D:
    """Return ``digest`` updated with the snapshot's uids, positions and orientations, by uid."""
    out = digest
    order = np.argsort(snapshot.uid, kind='stable')
    out.update(np.ascontiguousarray(snapshot.uid[order]).tobytes())
    out.update(np.ascontiguousarray(snapshot.position[order]).tobytes())
    out.update(np.ascontiguousarray(snapshot.orientation[order]).tobytes())
    return out


def character_digest(snapshot: Snapshot, digest: D) -> D:
    """Return ``digest`` updated with every character's id, position, velocity, state and ground.

    The body poses go in too, by :func:`pose_digest`.
    """
    out = pose_digest(snapshot, digest)
    rows = snapshot.characters
    for array in (
        rows.id,
        rows.position,
        rows.velocity,
        rows.state,
        rows.ground_uid,
        rows.slide_progress,
    ):
        out.update(np.ascontiguousarray(array).tobytes())
    return out


def vehicle_digest(snapshot: Snapshot, digest: D) -> D:
    """Return ``digest`` updated with every vehicle's engine, gear and wheels.

    The body poses go in too, by :func:`pose_digest`.
    """
    out = pose_digest(snapshot, digest)
    rows = snapshot.vehicles
    wheels = rows.wheels
    for array in (
        rows.id,
        rows.rpm,
        rows.gear,
        rows.steer,
        rows.impact_speed,
        wheels.position,
        wheels.orientation,
        wheels.length,
        wheels.spin,
        wheels.slip,
        wheels.ground,
    ):
        out.update(np.ascontiguousarray(array).tobytes())
    return out
