"""Gravity zones and kinematic tracks.

A :class:`GravityZone` follows OMI_physics_gravity: a region (a box, a sphere
or a registered shape, at a pose) and a field, directional along
``direction`` or point towards ``center``, of ``gravity`` m/s2. A body whose
centre of mass is inside zones takes the world's gravity, then each zone's in
ascending ``priority``: a ``replace`` zone sets the gravity to its own, a
``stop`` zone sets it to zero, any other adds its own. The body's gravity
factor scales the result. A character's gravity is changed the same way, at
its capsule's centre, along the vertical.

A :class:`Track` is a sampled pose a kinematic body follows, evaluated on the
world thread each tick.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics.enums import ZoneField, ZoneShape

__all__ = ['GravityZone', 'Track', 'zone_arrays']

Vector = tuple[float, float, float]


def _vector(value: Sequence[float], name: str) -> Vector:
    x, y, z = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError('%s is not finite' % (name,))
    return (x, y, z)


@dataclasses.dataclass(frozen=True)
class GravityZone:
    """One gravity zone; :meth:`box`, :meth:`sphere` and :meth:`of_shape` make each region.

    ``half`` is a box's half extents, or a sphere's radius in its first
    entry; ``shape_id`` names a registered shape for ``ZoneShape.SHAPE``.
    """

    shape: ZoneShape = ZoneShape.BOX
    position: Vector = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    half: Vector = (0.5, 0.5, 0.5)
    shape_id: int = 0
    field: ZoneField = ZoneField.DIRECTIONAL
    gravity: float = 9.81
    direction: Vector = (0.0, -1.0, 0.0)
    center: Vector = (0.0, 0.0, 0.0)
    priority: int = 0
    replace: bool = False
    stop: bool = False

    @classmethod
    def box(cls, position: Sequence[float], size: Sequence[float], **values: Any) -> GravityZone:
        """A box of edge lengths ``size`` centred at ``position``; ``rotation`` turns it."""
        sx, sy, sz = _vector(size, 'size')
        return cls(
            ZoneShape.BOX, _vector(position, 'position'), half=(sx / 2, sy / 2, sz / 2), **values
        )

    @classmethod
    def sphere(cls, position: Sequence[float], radius: float, **values: Any) -> GravityZone:
        """A ball of ``radius`` about ``position``."""
        return cls(
            ZoneShape.SPHERE,
            _vector(position, 'position'),
            half=(float(radius), 0.0, 0.0),
            **values,
        )

    @classmethod
    def of_shape(
        cls,
        shape_id: int,
        position: Sequence[float],
        rotation: Sequence[float] = (0, 0, 0, 1),
        **values: Any,
    ) -> GravityZone:
        """The inside of a registered shape (convex shapes, and closed meshes) at a pose."""
        x, y, z, w = (float(v) for v in rotation)
        return cls(
            ZoneShape.SHAPE,
            _vector(position, 'position'),
            (x, y, z, w),
            shape_id=int(shape_id),
            **values,
        )


def zone_arrays(zones: Sequence[GravityZone]) -> dict[str, np.ndarray]:
    """Return the zones as the native module's arrays."""
    n = len(zones)
    return {
        'shape': np.array([int(z.shape) for z in zones], np.uint8),
        'field': np.array([int(z.field) for z in zones], np.uint8),
        'position': np.array([z.position for z in zones], np.float64).reshape(n, 3),
        'rotation': np.array([z.rotation for z in zones], np.float64).reshape(n, 4),
        'half': np.array([z.half for z in zones], np.float64).reshape(n, 3),
        'shape_id': np.array([z.shape_id for z in zones], np.uint32),
        'gravity': np.array([z.gravity for z in zones], np.float64),
        'direction': np.array([z.direction for z in zones], np.float64).reshape(n, 3),
        'center': np.array([z.center for z in zones], np.float64).reshape(n, 3),
        'priority': np.array([z.priority for z in zones], np.int32),
        'replace': np.array([z.replace for z in zones], np.uint8),
        'stop': np.array([z.stop for z in zones], np.uint8),
    }


@dataclasses.dataclass(frozen=True)
class Track:
    """A sampled pose: ``times`` (k,) seconds ascending from the start, ``positions`` (k, 3), ``rotations`` (k, 4).

    Positions are interpolated linearly and rotations by slerp. Past the last
    sample the body holds the last pose, or with ``loop`` starts again.
    ``rotations`` of None holds the identity.
    """

    times: npt.ArrayLike
    positions: npt.ArrayLike
    rotations: npt.ArrayLike | None = None
    loop: bool = False

    def native(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the samples as contiguous float64 arrays."""
        times = np.ascontiguousarray(self.times, dtype=np.float64).reshape(-1)
        k = len(times)
        if k == 0:
            raise ValueError('a track has at least one sample')
        positions = np.ascontiguousarray(self.positions, dtype=np.float64).reshape(k, 3)
        rotations = np.ascontiguousarray(
            np.broadcast_to((0.0, 0.0, 0.0, 1.0), (k, 4))
            if self.rotations is None
            else self.rotations,
            dtype=np.float64,
        ).reshape(k, 4)
        return times, positions, rotations
