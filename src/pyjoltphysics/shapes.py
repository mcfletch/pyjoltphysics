"""Shape: a collision shape built once by a world and named by its id.

A :class:`Shape` describes a shape; :meth:`World.add_shape
<pyjoltphysics.World.add_shape>` builds it on the calling thread and returns
the world's id for it, and a :class:`~pyjoltphysics.BodyBatch` names that id,
or the :class:`Shape` itself, for each of its bodies. Every body naming one id
shares the one Jolt shape.

>>> import numpy as np
>>> from pyjoltphysics import BodyBatch, MotionType, Shape, World, WorldSettings
>>> with World(WorldSettings(threads=1)) as world:
...     post = world.add_shape(Shape.box((0.2, 2.0, 0.2)))
...     posts = world.add_bodies(
...         BodyBatch.of_shape(np.random.uniform(-50, 50, (1000, 3)), post, motion=MotionType.STATIC)
...     )
...     world.stats()['shapes']
1

Heights of a :meth:`Shape.heightfield` are ``heights[j, i]`` at ``x = origin[0]
+ i * spacing[0]`` and ``z = origin[2] + j * spacing[1]``, as in the OMI
physics height field. Jolt takes a square grid whose side is a multiple of its
block size; any other grid is padded on its +x and +z sides with samples that
collide with nothing (:func:`heightfield_side` gives the side). A landscape
larger than one shape should hold is several tiles,
:func:`~pyjoltphysics.heightfield_tiles`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics.enums import ShapeKind

__all__ = ['Shape', 'ShapeInfo', 'heightfield_side']

Vector = Sequence[float]
#: A compound's child: ``(shape, position)`` or ``(shape, position, orientation)``,
#: the shape a :class:`Shape` or a shape id of the same world.
Part = tuple[Any, ...]


def _float_rows(values: npt.ArrayLike, width: int, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError('%s has shape %s where (n, %d) is needed' % (name, array.shape, width))
    array.flags.writeable = False
    return array


@dataclasses.dataclass(frozen=True, eq=False)
class Shape:
    """A collision shape to build with :meth:`World.add_shape <pyjoltphysics.World.add_shape>`.

    Make one with a class method. Shapes compare by identity, so a world
    builds a :class:`Shape` object once however many batches name it.
    """

    kind: ShapeKind
    params: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    points: np.ndarray | None = None
    triangles: np.ndarray | None = None
    heights: np.ndarray | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float] = (1.0, 1.0)
    block_size: int = 4
    bits_per_sample: int = 8
    height_range: tuple[float, float] | None = None
    parts: tuple[tuple[Shape | int, tuple[float, ...], tuple[float, ...]], ...] = ()
    mutable: bool = False
    max_convex_radius: float = 0.05
    hull_tolerance: float = 1e-3

    # -- primitives, for compounds and shared bodies -------------------------

    @classmethod
    def box(cls, size: Vector = (1.0, 1.0, 1.0)) -> Shape:
        """Return a box of edge lengths ``size`` (x, y, z)."""
        x, y, z = (float(v) for v in size)
        return cls(ShapeKind.BOX, (x, y, z, 0.0))

    @classmethod
    def sphere(cls, radius: float = 0.5) -> Shape:
        """Return a sphere of ``radius``."""
        return cls(ShapeKind.SPHERE, (float(radius), 0.0, 0.0, 0.0))

    @classmethod
    def capsule(cls, radius: float = 0.5, height: float = 1.0) -> Shape:
        """Return a Y-axis capsule; ``height`` is the cylinder between the hemisphere centres."""
        return cls(ShapeKind.CAPSULE, (float(radius), float(height), 0.0, 0.0))

    @classmethod
    def cylinder(cls, radius: float = 0.5, height: float = 1.0) -> Shape:
        """Return a Y-axis cylinder of ``radius`` and total ``height``."""
        return cls(ShapeKind.CYLINDER, (float(radius), float(height), 0.0, 0.0))

    # -- shapes with arrays --------------------------------------------------

    @classmethod
    def convex(
        cls, points: npt.ArrayLike, max_convex_radius: float = 0.05, hull_tolerance: float = 1e-3
    ) -> Shape:
        """Return the convex hull of ``points``, (m, 3) metres in the body's frame.

        Jolt rounds the hull's edges by a convex radius of at most
        ``max_convex_radius`` (smaller for a small hull) and lets points lie
        ``hull_tolerance`` metres outside it. Fewer than 4 points, or points on a
        plane or a line, raise ``ValueError`` when the shape is built.
        """
        return cls(
            ShapeKind.CONVEX,
            points=_float_rows(points, 3, 'points'),
            max_convex_radius=float(max_convex_radius),
            hull_tolerance=float(hull_tolerance),
        )

    @classmethod
    def mesh(cls, points: npt.ArrayLike, triangles: npt.ArrayLike) -> Shape:
        """Return a triangle mesh: (m, 3) vertices in the body's frame and (t, 3) vertex indices.

        A mesh makes static and kinematic bodies only. Degenerate and repeated
        triangles are dropped. Contacts with edges between faces that meet at
        under 5 degrees are treated as contacts with the faces (Jolt's active
        edges). A triangle collides on its front face, the side from which its
        vertices run anticlockwise; a wall met from both sides lists each
        triangle in both windings.
        """
        tris = np.array(triangles, dtype=np.int64, copy=True)
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError('triangles has shape %s where (t, 3) is needed' % (tris.shape,))
        if tris.size and tris.min() < 0:
            raise ValueError('triangle indices are not negative')
        indices = tris.astype(np.uint32)
        indices.flags.writeable = False
        return cls(ShapeKind.MESH, points=_float_rows(points, 3, 'points'), triangles=indices)

    @classmethod
    def heightfield(
        cls,
        heights: npt.ArrayLike,
        origin: Vector = (0.0, 0.0, 0.0),
        spacing: float | Vector = 1.0,
        holes: npt.ArrayLike | None = None,
        block_size: int = 4,
        bits_per_sample: int = 8,
        height_range: tuple[float, float] | None = None,
    ) -> Shape:
        """Return a height field: ``heights[j, i]`` at ``(origin + (i dx, h, j dz))`` in the body's frame.

        ``spacing`` is metres between samples, one number or ``(dx, dz)``. A NaN
        height, or a true entry of ``holes`` (a bool array of ``heights``'
        shape), is a hole: the cells touching that sample collide with nothing.
        Jolt stores each block of ``block_size`` by ``block_size`` samples (2 to
        8) with ``bits_per_sample`` bits (1 to 16) relative to the block's range;
        8 bits keep a sample within 1/255 of its block's range. ``height_range``
        is ``(lowest, highest)`` that later edits may reach. A height field
        makes static and kinematic bodies only.
        """
        grid = np.array(heights, dtype=np.float32, copy=True)
        if grid.ndim != 2:
            raise ValueError('heights has shape %s where (rows, columns) is needed' % (grid.shape,))
        if holes is not None:
            mask = np.asarray(holes, dtype=bool)
            if mask.shape != grid.shape:
                raise ValueError('holes has shape %s where %s is needed' % (mask.shape, grid.shape))
            grid[mask] = np.nan
        grid.flags.writeable = False
        dx, dz = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (2,))
        ox, oy, oz = (float(v) for v in origin)
        return cls(
            ShapeKind.HEIGHTFIELD,
            heights=grid,
            origin=(ox, oy, oz),
            spacing=(float(dx), float(dz)),
            block_size=int(block_size),
            bits_per_sample=int(bits_per_sample),
            height_range=None
            if height_range is None
            else (float(height_range[0]), float(height_range[1])),
        )

    @classmethod
    def compound(cls, parts: Sequence[Part], mutable: bool = False) -> Shape:
        """Return child shapes placed in one body's frame.

        Each part is ``(shape, position)`` or ``(shape, position, orientation)``:
        a :class:`Shape` or a shape id, its centre in metres from the body's
        origin, and a unit quaternion ``(x, y, z, w)``. A child may be any shape
        but a height field. ``mutable`` builds Jolt's ``MutableCompoundShape``,
        quicker to build and slower to collide, in place of the static one.
        """
        placed = []
        for part in parts:
            if len(part) not in (2, 3):
                raise ValueError('a compound part is (shape, position[, orientation])')
            shape, position = part[0], tuple(float(v) for v in part[1])
            orientation = (
                tuple(float(v) for v in part[2]) if len(part) == 3 else (0.0, 0.0, 0.0, 1.0)
            )
            if len(position) != 3 or len(orientation) != 4:
                raise ValueError('a part position has 3 components and an orientation 4')
            if not isinstance(shape, (Shape, int, np.integer)):
                raise TypeError('a compound part is a Shape or a shape id')
            placed.append(
                (shape if isinstance(shape, Shape) else int(shape), position, orientation)
            )
        return cls(ShapeKind.COMPOUND, parts=tuple(placed), mutable=bool(mutable))


@dataclasses.dataclass(frozen=True)
class ShapeInfo:
    """What a world built for a shape id: :meth:`World.shape_info <pyjoltphysics.World.shape_info>`.

    ``volume`` is m3 (0 for a mesh, height field or plane), ``center_of_mass``
    and the bounds are metres in the shape's frame, and ``dynamic`` is false
    for a shape only static and kinematic bodies take.
    """

    kind: ShapeKind
    volume: float
    center_of_mass: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    dynamic: bool


def heightfield_side(rows: int, columns: int, block_size: int = 4) -> int:
    """Return the side of the square sample grid Jolt is given for ``rows`` by ``columns`` samples.

    It is at least both, at least two blocks, and a multiple of ``block_size``.
    """
    wanted = max(rows, columns, 2 * block_size)
    return -(-wanted // block_size) * block_size
