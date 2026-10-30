"""BodyBatch: many bodies to add in one command, as arrays.

Every per-body value is either one value for the whole batch or an array with
one entry (or one row) per body; :meth:`BodyBatch.arrays` broadcasts them.

>>> import numpy as np
>>> from pyjoltphysics import BodyBatch, MotionType
>>> batch = BodyBatch.boxes(np.zeros((3, 3)), size=(0.5, 0.5, 0.5), mass=2.0)
>>> len(batch)
3
>>> floor = BodyBatch.plane(motion=MotionType.STATIC)
>>> int(floor.arrays()['motion'][0])
0
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics.enums import BodyFlag, MotionType, ShapeKind

__all__ = ['BodyBatch']

ArrayLike = npt.ArrayLike
_ALL_GROUPS = 0xFFFFFFFF


def _rows(value: ArrayLike, count: int, width: int, dtype: type, name: str) -> np.ndarray:
    array: np.ndarray = np.asarray(value, dtype=dtype)
    if array.ndim == 1 and width > 1:
        array = np.broadcast_to(array, (count, width))
    elif array.ndim == 0 and width == 1:
        array = np.broadcast_to(array, (count,))
    expected = (count,) if width == 1 else (count, width)
    if array.shape != expected:
        raise ValueError('%s has shape %s where the batch needs %s' % (name, array.shape, expected))
    return np.ascontiguousarray(array)


@dataclasses.dataclass
class BodyBatch:
    """Bodies to add with :meth:`~pyjoltphysics.World.add_bodies`.

    ``positions`` is (n, 3) metres and fixes the batch's length. ``shape_kind``
    is a :class:`ShapeKind` and ``shape_params`` its four parameters, per batch
    or per body. ``orientations`` are unit quaternions ``(x, y, z, w)``. A
    moving body takes its mass from ``mass`` (kg) or, where that is 0, from
    ``density`` (kg/m3) and its shape's volume. Damping of None takes the
    world's :class:`~pyjoltphysics.WorldSettings` default. ``group`` and
    ``mask`` are 32-bit collision bit sets: two bodies collide when each
    one's group shares a bit with the other's mask.
    """

    positions: ArrayLike
    shape_kind: ShapeKind | ArrayLike = ShapeKind.BOX
    shape_params: ArrayLike = (1.0, 1.0, 1.0, 0.0)
    motion: MotionType | ArrayLike = MotionType.DYNAMIC
    orientations: ArrayLike | None = None
    mass: ArrayLike = 1.0
    density: ArrayLike = 0.0
    friction: ArrayLike = 0.5
    restitution: ArrayLike = 0.0
    linear_damping: ArrayLike | None = None
    angular_damping: ArrayLike | None = None
    gravity_factor: ArrayLike = 1.0
    linear_velocity: ArrayLike = (0.0, 0.0, 0.0)
    angular_velocity: ArrayLike = (0.0, 0.0, 0.0)
    group: ArrayLike = 1
    mask: ArrayLike = _ALL_GROUPS
    flags: BodyFlag | ArrayLike = BodyFlag.NONE

    def __len__(self) -> int:
        return int(np.shape(self.positions)[0])

    # -- builders, one per primitive ---------------------------------------

    @classmethod
    def boxes(
        cls,
        positions: ArrayLike,
        size: Sequence[float] | ArrayLike = (1.0, 1.0, 1.0),
        **values: Any,
    ) -> BodyBatch:
        """Return boxes of edge lengths ``size`` (x, y, z), one per position."""
        size = np.asarray(size, dtype=np.float64)
        params = np.concatenate([size, np.zeros(size.shape[:-1] + (1,))], axis=-1)
        return cls(positions, ShapeKind.BOX, params, **values)

    @classmethod
    def spheres(cls, positions: ArrayLike, radius: ArrayLike = 0.5, **values: Any) -> BodyBatch:
        """Return spheres of ``radius``, one per position."""
        return cls(positions, ShapeKind.SPHERE, _params(radius), **values)

    @classmethod
    def capsules(
        cls, positions: ArrayLike, radius: ArrayLike = 0.5, height: ArrayLike = 1.0, **values: Any
    ) -> BodyBatch:
        """Return Y-axis capsules; ``height`` is the cylinder between the hemisphere centres."""
        return cls(positions, ShapeKind.CAPSULE, _params(radius, height), **values)

    @classmethod
    def cylinders(
        cls, positions: ArrayLike, radius: ArrayLike = 0.5, height: ArrayLike = 1.0, **values: Any
    ) -> BodyBatch:
        """Return Y-axis cylinders of ``radius`` and total ``height``."""
        return cls(positions, ShapeKind.CYLINDER, _params(radius, height), **values)

    @classmethod
    def plane(
        cls,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        half_extent: float = 0.0,
        **values: Any,
    ) -> BodyBatch:
        """Return one static plane through ``position`` facing +Y (or ``orientations``' +Y)."""
        values.setdefault('motion', MotionType.STATIC)
        return cls([position], ShapeKind.PLANE, (half_extent, 0.0, 0.0, 0.0), **values)

    # -- the native arrays -------------------------------------------------

    def arrays(
        self, linear_damping: float = 0.0, angular_damping: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Return every field broadcast to one contiguous array per field, by name.

        Damping left as None takes ``linear_damping`` and ``angular_damping``.
        """
        positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('positions has shape %s where (n, 3) is needed' % (positions.shape,))
        n = positions.shape[0]
        orientations = (0.0, 0.0, 0.0, 1.0) if self.orientations is None else self.orientations
        damp_l = linear_damping if self.linear_damping is None else self.linear_damping
        damp_a = angular_damping if self.angular_damping is None else self.angular_damping
        return {
            'position': positions,
            'orientation': _rows(orientations, n, 4, np.float64, 'orientations'),
            'shape_kind': _rows(self.shape_kind, n, 1, np.uint8, 'shape_kind'),
            'shape_params': _rows(self.shape_params, n, 4, np.float64, 'shape_params'),
            'motion': _rows(self.motion, n, 1, np.uint8, 'motion'),
            'mass': _rows(self.mass, n, 1, np.float64, 'mass'),
            'density': _rows(self.density, n, 1, np.float64, 'density'),
            'friction': _rows(self.friction, n, 1, np.float32, 'friction'),
            'restitution': _rows(self.restitution, n, 1, np.float32, 'restitution'),
            'linear_damping': _rows(damp_l, n, 1, np.float32, 'linear_damping'),
            'angular_damping': _rows(damp_a, n, 1, np.float32, 'angular_damping'),
            'gravity_factor': _rows(self.gravity_factor, n, 1, np.float32, 'gravity_factor'),
            'linear_velocity': _rows(self.linear_velocity, n, 3, np.float64, 'linear_velocity'),
            'angular_velocity': _rows(self.angular_velocity, n, 3, np.float64, 'angular_velocity'),
            'group': _rows(self.group, n, 1, np.uint32, 'group'),
            'mask': _rows(self.mask, n, 1, np.uint32, 'mask'),
            'flags': _rows(self.flags, n, 1, np.uint32, 'flags'),
        }


def _params(*columns: ArrayLike) -> np.ndarray:
    """Return shape parameters with the given leading columns and zeros after them."""
    arrays = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in columns))
    zeros = [np.zeros_like(arrays[0])] * (4 - len(arrays))
    return np.stack(list(arrays) + zeros, axis=-1)
