"""Query filters and results: rays, ray batches, shape sweeps and overlaps.

Queries run on the calling thread and read the world as the last completed
tick left it. A query waits while the world thread applies batches, moves
characters or steps (the world lock's exclusive side), and runs beside the
rest of a tick. On the world thread, inside a callback, it takes no lock.
The GIL is released while Jolt answers it.

>>> from pyjoltphysics import QueryFilter
>>> QueryFilter(exclude=[3, 1]).exclude
(1, 3)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

__all__ = ['ALL_GROUPS', 'QueryFilter', 'RayHit', 'RayHits', 'ShapeHit']

#: Every collision group bit.
ALL_GROUPS = 0xFFFFFFFF


def _sorted_uids(uids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(u) for u in uids))


@dataclasses.dataclass(frozen=True)
class QueryFilter:
    """Which bodies a query meets.

    A body is met when its collision group shares a bit with ``mask`` and its
    mask shares a bit with ``group``, as two bodies meet each other. Bodies
    whose uid is in ``exclude`` are passed over. Sensor bodies are met only
    with ``triggers``; character proxies (the bodies characters stand in the
    world as) unless ``characters`` is false.
    """

    group: int = ALL_GROUPS
    mask: int = ALL_GROUPS
    exclude: Iterable[int] = ()
    triggers: bool = False
    characters: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, 'exclude', _sorted_uids(self.exclude))

    def native(self) -> tuple[int, int, npt.NDArray[np.uint32], bool, bool]:
        """Return the native call's ``group, mask, exclude, triggers, characters`` arguments."""
        exclude = np.asarray(self.exclude, dtype=np.uint32).reshape(-1)
        return (
            int(self.group),
            int(self.mask),
            exclude,
            bool(self.triggers),
            bool(self.characters),
        )


#: The filter a query with none uses.
DEFAULT_FILTER = QueryFilter()


@dataclasses.dataclass(frozen=True)
class RayHit:
    """The first body a ray meets.

    ``fraction`` is of the ray's ``max_distance``; ``distance`` is metres from
    the origin. ``normal`` is the surface's unit normal at ``point``.
    ``sub_shape`` is Jolt's sub-shape id of the part hit (a mesh triangle, a
    compound child), stable for one shape.
    """

    uid: int
    distance: float
    fraction: float
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    sub_shape: int


@dataclasses.dataclass(frozen=True)
class RayHits:
    """A ray batch's answers, one row a ray.

    ``hit`` (n,) bool; ``uid`` (n,) uint32, 0 for a miss; ``fraction`` (n,)
    float32 of ``max_distance``, 1 for a miss; ``distance`` (n,) float64
    metres, ``inf`` for a miss; ``point`` (n, 3) float64, NaN for a miss;
    ``normal`` (n, 3) float32, 0 for a miss; ``sub_shape`` (n,) uint32.
    """

    hit: npt.NDArray[np.bool_]
    uid: npt.NDArray[np.uint32]
    fraction: npt.NDArray[np.float32]
    distance: npt.NDArray[np.float64]
    point: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float32]
    sub_shape: npt.NDArray[np.uint32]

    def __len__(self) -> int:
        return int(self.hit.shape[0])


@dataclasses.dataclass(frozen=True)
class ShapeHit:
    """A body a swept shape meets.

    ``fraction`` is of the way from the start to the end and ``distance``
    metres along it. ``point`` is on the body hit and ``normal`` the unit
    vector from it towards the swept shape. ``depth`` is the penetration at
    the hit, positive where the shape overlaps the body at the start.
    """

    uid: int
    fraction: float
    distance: float
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    depth: float
    sub_shape: int
