"""Scenes as arrays, shared by the test suites and the benchmarks.

>>> from pyjoltphysics.testing import grid
>>> grid(4).shape
(4, 3)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np

from pyjoltphysics.bodies import BodyBatch
from pyjoltphysics.enums import MotionType
from pyjoltphysics.world import Snapshot, World

__all__ = ['Digest', 'MakeWorld', 'floor', 'grid', 'pose_digest']

#: The ``make_world(**settings)`` factory the test suites' fixture returns.
MakeWorld = Callable[..., World]


class Digest(Protocol):
    """A hashlib object: anything with ``update(bytes)``."""

    def update(self, data: bytes, /) -> None: ...


D = TypeVar('D', bound=Digest)


def floor(half: float = 100.0) -> BodyBatch:
    """Return a static box floor ``2 * half`` metres square whose top face is y = 0."""
    return BodyBatch.boxes(
        [(0.0, -0.5, 0.0)], size=(2 * half, 1.0, 2 * half), motion=MotionType.STATIC
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
