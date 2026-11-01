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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics._checks import whole_numbers
from pyjoltphysics.enums import BodyFlag, MotionType, ShapeKind
from pyjoltphysics.shapes import Shape

__all__ = ['BodyBatch', 'ShapeRef', 'heightfield_tiles']

#: What a batch's ``shape`` field holds per body: a shape id of the world, or a
#: :class:`~pyjoltphysics.Shape` the world builds once when the batch is added.
ShapeRef = Shape | int

ArrayLike = npt.ArrayLike
_ALL_GROUPS = 0xFFFFFFFF


def _rows(value: ArrayLike, count: int, width: int, dtype: type, name: str) -> np.ndarray:
    array: np.ndarray = (
        whole_numbers(value, dtype, name)
        if np.issubdtype(dtype, np.integer)
        else np.asarray(value, dtype=dtype)
    )
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

    ``positions`` is (n, 3) metres and fixes the batch's length. A body's
    shape is ``shape``, one :class:`~pyjoltphysics.Shape` or shape id for the
    batch or a sequence of them per body; where ``shape`` is None it is the
    primitive ``shape_kind`` (a :class:`ShapeKind`) with its four
    ``shape_params``. Each body's shape is scaled by ``scale``, then placed at
    ``shape_offset`` turned by ``shape_rotation`` in the body's frame, and its
    centre of mass moved by ``center_of_mass`` (metres); bodies with equal
    shapes and decorations share one Jolt shape. ``orientations`` are unit
    quaternions ``(x, y, z, w)``. A
    moving body takes its mass from ``mass`` (kg) or, where that is 0, from
    ``density`` (kg/m3) and its shape's volume. Damping of None takes the
    world's :class:`~pyjoltphysics.WorldSettings` default. ``group`` and
    ``mask`` are 32-bit collision bit sets: two bodies collide when each
    one's group shares a bit with the other's mask, and the world's filter
    table lets their ``filter`` indices meet. ``material`` is a row of the
    world's :class:`~pyjoltphysics.MaterialTable`, 0 for the body's own
    ``friction`` and ``restitution``. ``quadratic_drag`` (1/m) is a force of
    ``-k m |v| v``.
    """

    positions: ArrayLike
    shape_kind: ShapeKind | ArrayLike = ShapeKind.BOX
    shape_params: ArrayLike = (1.0, 1.0, 1.0, 0.0)
    shape: ShapeRef | Sequence[ShapeRef] | ArrayLike | None = None
    scale: ArrayLike = (1.0, 1.0, 1.0)
    shape_offset: ArrayLike = (0.0, 0.0, 0.0)
    shape_rotation: ArrayLike = (0.0, 0.0, 0.0, 1.0)
    center_of_mass: ArrayLike = (0.0, 0.0, 0.0)
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
    material: ArrayLike = 0
    filter: ArrayLike = 0
    quadratic_drag: ArrayLike = 0.0

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

    # -- builders for shapes with arrays ------------------------------------

    @classmethod
    def of_shape(
        cls, positions: ArrayLike, shape: ShapeRef | Sequence[ShapeRef], **values: Any
    ) -> BodyBatch:
        """Return bodies of a registered shape id or :class:`~pyjoltphysics.Shape`, one per position."""
        return cls(positions, shape=shape, **values)

    @classmethod
    def hulls(
        cls,
        positions: ArrayLike,
        points: ArrayLike,
        max_convex_radius: float = 0.05,
        hull_tolerance: float = 1e-3,
        **values: Any,
    ) -> BodyBatch:
        """Return convex hulls: ``points`` (m, 3) for every body, or (n, m, 3) one cloud per body."""
        cloud = np.asarray(points, dtype=np.float64)
        if cloud.ndim == 2:
            shape: Shape | list[ShapeRef] = Shape.convex(cloud, max_convex_radius, hull_tolerance)
        elif cloud.ndim == 3:
            shape = [Shape.convex(c, max_convex_radius, hull_tolerance) for c in cloud]
        else:
            raise ValueError(
                'points has shape %s where (m, 3) or (n, m, 3) is needed' % (cloud.shape,)
            )
        return cls(positions, shape=shape, **values)

    @classmethod
    def meshes(
        cls, positions: ArrayLike, points: ArrayLike, triangles: ArrayLike, **values: Any
    ) -> BodyBatch:
        """Return static bodies of one triangle mesh, ``points`` (m, 3) and ``triangles`` (t, 3)."""
        values.setdefault('motion', MotionType.STATIC)
        return cls(positions, shape=Shape.mesh(points, triangles), **values)

    @classmethod
    def heightfield(
        cls,
        heights: ArrayLike,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        spacing: float | Sequence[float] = 1.0,
        holes: ArrayLike | None = None,
        block_size: int = 4,
        bits_per_sample: int = 8,
        height_range: tuple[float, float] | None = None,
        **values: Any,
    ) -> BodyBatch:
        """Return one static body at ``origin`` holding a :meth:`Shape.heightfield <pyjoltphysics.Shape.heightfield>`."""
        values.setdefault('motion', MotionType.STATIC)
        land = Shape.heightfield(
            heights,
            spacing=spacing,
            holes=holes,
            block_size=block_size,
            bits_per_sample=bits_per_sample,
            height_range=height_range,
        )
        return cls([tuple(float(v) for v in origin)], shape=land, **values)

    # -- the native arrays -------------------------------------------------

    def arrays(
        self,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        shape_id: Callable[[Shape], int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return every field broadcast to one contiguous array per field, by name.

        Damping left as None takes ``linear_damping`` and ``angular_damping``.
        ``shape_id`` turns a :class:`~pyjoltphysics.Shape` in ``shape`` into
        its id; :meth:`World.add_bodies <pyjoltphysics.World.add_bodies>`
        passes its :meth:`~pyjoltphysics.World.add_shape`.
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
            'shape_id': self._shape_ids(n, shape_id),
            'scale': _rows(self.scale, n, 3, np.float64, 'scale'),
            'shape_offset': _rows(self.shape_offset, n, 3, np.float64, 'shape_offset'),
            'shape_rotation': _rows(self.shape_rotation, n, 4, np.float64, 'shape_rotation'),
            'center_of_mass': _rows(self.center_of_mass, n, 3, np.float64, 'center_of_mass'),
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
            'material': _rows(self.material, n, 1, np.uint16, 'material'),
            'filter': _rows(self.filter, n, 1, np.uint16, 'filter'),
            'drag': _rows(self.quadratic_drag, n, 1, np.float32, 'quadratic_drag'),
        }

    def _shape_ids(self, n: int, shape_id: Callable[[Shape], int] | None) -> np.ndarray:
        """Return the (n,) uint32 shape id of each body, 0 where it is a primitive."""
        shape = self.shape
        if shape is None:
            return np.zeros(n, np.uint32)
        refs: list[object] = (
            [shape] * n if isinstance(shape, (Shape, int, np.integer)) else list(shape)  # type: ignore[arg-type]  # a sequence or array of refs
        )
        if len(refs) != n:
            raise ValueError('shape has %d entries where the batch has %d bodies' % (len(refs), n))
        # Each distinct Shape is registered once, in order of first use.
        ids: dict[Shape, int] = {}
        out = np.empty(n, np.uint32)
        for k, ref in enumerate(refs):
            if isinstance(ref, Shape):
                if shape_id is None:
                    raise ValueError('a Shape in a batch is registered by World.add_bodies')
                if ref not in ids:
                    ids[ref] = shape_id(ref)
                out[k] = ids[ref]
            elif isinstance(ref, (int, np.integer)):
                out[k] = int(ref)
            else:
                raise TypeError("a body's shape is a Shape or a shape id, not %r" % (ref,))
        return out


def _params(*columns: ArrayLike) -> np.ndarray:
    """Return shape parameters with the given leading columns and zeros after them."""
    arrays = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in columns))
    zeros = [np.zeros_like(arrays[0])] * (4 - len(arrays))
    return np.stack(list(arrays) + zeros, axis=-1)


def heightfield_tiles(
    heights: npt.ArrayLike,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    spacing: float | Sequence[float] = 1.0,
    tile: int = 64,
    holes: npt.ArrayLike | None = None,
    **values: Any,
) -> BodyBatch:
    """Return one static body a tile of ``tile`` by ``tile`` cells covering a height grid.

    Neighbouring tiles share the samples of their common edge, so their
    triangles meet exactly. Each body sits at its tile's corner, which keeps
    a tile's local coordinates small however far the landscape reaches.
    ``heights``, ``origin``, ``spacing`` and ``holes`` are as
    :meth:`Shape.heightfield` takes them; ``values`` are further
    :class:`~pyjoltphysics.BodyBatch` fields, and ``block_size``,
    ``bits_per_sample`` and ``height_range`` pass to every tile.
    """
    grid = np.array(heights, dtype=np.float32, copy=True)
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ValueError('heights has shape %s where at least (2, 2) is needed' % (grid.shape,))
    if tile < 1:
        raise ValueError('a tile is at least one cell')
    if holes is not None:
        grid[np.asarray(holes, dtype=bool)] = np.nan
    dx, dz = (float(v) for v in np.broadcast_to(np.asarray(spacing, dtype=np.float64), (2,)))
    ox, oy, oz = (float(v) for v in origin)
    options = {
        k: values.pop(k) for k in ('block_size', 'bits_per_sample', 'height_range') if k in values
    }
    rows, columns = grid.shape
    shapes: list[Shape | int] = []
    corners = []
    for j in range(0, rows - 1, tile):
        for i in range(0, columns - 1, tile):
            part = grid[j : min(j + tile, rows - 1) + 1, i : min(i + tile, columns - 1) + 1]
            if np.isnan(part).all():
                continue
            shapes.append(Shape.heightfield(part, spacing=(dx, dz), **options))
            corners.append((ox + i * dx, oy, oz + j * dz))
    values.setdefault('motion', MotionType.STATIC)
    return BodyBatch.of_shape(np.array(corners, dtype=np.float64).reshape(-1, 3), shapes, **values)
