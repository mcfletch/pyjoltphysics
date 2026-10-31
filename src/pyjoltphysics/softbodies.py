"""Soft bodies: cloth, ropes and pressure balls through Jolt's soft-body simulation.

A :class:`SoftBodySettings` is particles (``positions``, each with a mass)
held by constraints: edges with a ``compliance`` (the inverse of stiffness,
m/N; 0 is rigid), shear edges across each quad, bend constraints across each
pair of faces, volume constraints on tetrahedra, and long-range attachments
that keep each free particle within reach of the nearest pinned one. A closed
surface may hold a ``pressure`` (n R T, in Pa m3). The constructors build the
common cases: :meth:`~SoftBodySettings.cloth`, :meth:`~SoftBodySettings.rope`,
:meth:`~SoftBodySettings.ball` and :meth:`~SoftBodySettings.from_mesh`.

A soft body is a body (:meth:`World.add_soft_body
<pyjoltphysics.World.add_soft_body>` returns its uid): it collides with
rigid bodies by the world's groups, masks and filter table, sleeps and wakes,
and is removed by :meth:`World.remove_bodies <pyjoltphysics.World.remove_bodies>`.
Soft bodies do not collide with each other. Its vertices are in the
snapshot (:class:`SoftBodyRows`), float32 relative to its centre of mass.

>>> from pyjoltphysics import SoftBodySettings, World, WorldSettings
>>> cloth = SoftBodySettings.cloth(1.0, 1.0, columns=9, rows=9, mass=0.3, linear_damping=2.0)
>>> a, b, _, _ = cloth.corner_indices()
>>> with World(WorldSettings(threads=1)) as world:
...     body = world.add_soft_body(cloth, position=(0.0, 2.0, 0.0),
...                                pins=([a, b], cloth.positions[[a, b]] + (0.0, 2.0, 0.0)))
...     world.run_ticks(480)
...     lowest = float(body.vertices()[:, 1].min())
>>> lowest < 1.3
True
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native
from pyjoltphysics.enums import BodyFlag
from pyjoltphysics.queries import ALL_GROUPS

if TYPE_CHECKING:
    from pyjoltphysics.world import World

__all__ = ['Bend', 'LongRange', 'SoftBody', 'SoftBodyRows', 'SoftBodySettings']

#: A pin set: vertex indices and their targets, (n, 3) world points or points in ``bodies``' frames.
Pins = tuple[Sequence[int], npt.ArrayLike]


class Bend(enum.IntEnum):
    """The bend constraints across each pair of faces: none, a distance, or the dihedral angle."""

    NONE = 0
    DISTANCE = 1
    DIHEDRAL = 2


class LongRange(enum.IntEnum):
    """Long-range attachments to the nearest pinned vertex: none, by straight line, or along the edges."""

    NONE = 0
    EUCLIDEAN = 1
    GEODESIC = 2


def _empty(width: int, dtype: type = np.uint32) -> np.ndarray:
    return np.zeros((0, width), dtype=dtype)


@dataclasses.dataclass(eq=False)
class SoftBodySettings:
    """A soft body's particles, constraints and material.

    ``positions`` (V, 3) are metres in the body's frame. ``mass`` (kg) is
    shared among the vertices equally, or ``masses`` (V,) gives each its own;
    a vertex of mass 0 is fixed where it is placed. ``faces`` (F, 3) are
    counter-clockwise seen from the side their normal faces. Edges, shear and
    bend constraints come from the faces, unless ``edges`` (E, 2) are given
    with ``edge_compliance`` (E,). ``tetrahedra`` (T, 4) hold their volume
    with ``volume_compliance``. Compliances are m/N, 0 rigid and ``inf`` for
    none. ``iterations`` is the solver's per step. ``linear_damping`` is
    1/s, ``max_velocity`` m/s and ``vertex_radius`` (m) the distance the
    vertices keep from the surfaces they meet. ``group``, ``mask``,
    ``filter``, ``material`` and ``flags`` are a body's.
    """

    positions: np.ndarray
    faces: np.ndarray | None = None
    edges: np.ndarray | None = None
    edge_compliance: np.ndarray | None = None
    tetrahedra: np.ndarray | None = None
    mass: float = 1.0
    masses: np.ndarray | None = None
    compliance: float = 0.0
    shear_compliance: float = 0.0
    bend_compliance: float = math.inf
    bend: Bend = Bend.DISTANCE
    volume_compliance: float = 0.0
    pressure: float = 0.0
    long_range: LongRange = LongRange.NONE
    long_range_stretch: float = 1.0
    iterations: int = 5
    linear_damping: float = 0.1
    max_velocity: float = 500.0
    friction: float = 0.2
    restitution: float = 0.0
    gravity_factor: float = 1.0
    vertex_radius: float = 0.0
    allow_sleep: bool = True
    double_sided: bool = False
    group: int = 1
    mask: int = ALL_GROUPS
    filter: int = 0
    material: int = 0
    flags: BodyFlag = BodyFlag.NONE
    #: A :meth:`cloth`'s (columns, rows) of vertices; None for other shapes.
    grid: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64).reshape(-1, 3)
        if self.grid is not None:
            columns, rows = (int(v) for v in self.grid)
            if columns * rows != len(self.positions):
                raise ValueError(
                    'a grid of %d by %d is not the %d vertices'
                    % (columns, rows, len(self.positions))
                )
            self.grid = (columns, rows)
        count = len(self.positions)
        if count == 0:
            raise ValueError('a soft body has at least one of its vertices')
        for name, width in (('faces', 3), ('edges', 2), ('tetrahedra', 4)):
            value = getattr(self, name)
            if value is not None:
                array = np.ascontiguousarray(value, dtype=np.uint32).reshape(-1, width)
                if array.size and int(array.max()) >= count:
                    raise ValueError(
                        '%s name a vertex past the %d there are (a face, edge or tetrahedron '
                        'index is under the vertex count)' % (name, count)
                    )
                setattr(self, name, array)
        if self.edges is not None:
            compliance = (
                self.edge_compliance if self.edge_compliance is not None else self.compliance
            )
            self.edge_compliance = np.ascontiguousarray(
                np.broadcast_to(np.asarray(compliance, dtype=np.float32), (len(self.edges),))
            )
        if not (math.isfinite(self.mass) and self.mass >= 0.0):
            raise ValueError('mass is finite and not negative, not %r' % (self.mass,))
        if self.masses is not None:
            self.masses = np.ascontiguousarray(self.masses, dtype=np.float64).reshape(count)
            if not (np.isfinite(self.masses).all() and (self.masses >= 0).all()):
                raise ValueError('masses are finite and not negative')

    def __len__(self) -> int:
        return len(self.positions)

    def vertex_masses(self) -> npt.NDArray[np.float64]:
        """Return (V,) the mass of each vertex, kg."""
        if self.masses is not None:
            return self.masses
        return np.full(len(self.positions), self.mass / len(self.positions))

    def volume(self, positions: npt.ArrayLike | None = None) -> float:
        """Return the volume a closed surface's faces enclose, m3, at ``positions`` or the rest positions."""
        points = self.positions if positions is None else np.asarray(positions, dtype=np.float64)
        if self.faces is None:
            return 0.0
        a, b, c = (points[self.faces[:, k]] for k in range(3))
        return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)

    def corner_indices(self) -> tuple[int, int, int, int]:
        """Return a :meth:`cloth`'s corner vertices: (-x -z, +x -z, -x +z, +x +z)."""
        if self.grid is None:
            raise ValueError('only a cloth made by SoftBodySettings.cloth has corners')
        columns, rows = self.grid
        return (0, columns - 1, (rows - 1) * columns, rows * columns - 1)

    # -- builders ------------------------------------------------------------

    @classmethod
    def cloth(
        cls,
        width: float,
        depth: float,
        columns: int,
        rows: int,
        mass: float = 1.0,
        **values: Any,
    ) -> SoftBodySettings:
        """Return a flat cloth ``width`` (x) by ``depth`` (z) metres in the y = 0 plane, centred.

        ``columns`` by ``rows`` vertices, row major from -x -z; faces up (+y).
        ``values`` are further fields; the default bend is a distance
        constraint of compliance ``bend_compliance`` (none unless given).
        """
        if columns < 2 or rows < 2:
            raise ValueError('a cloth has at least 2 columns and 2 rows of vertices')
        x = np.linspace(-width / 2.0, width / 2.0, columns)
        z = np.linspace(-depth / 2.0, depth / 2.0, rows)
        gx, gz = np.meshgrid(x, z)
        positions = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
        i = np.arange(rows - 1)[:, None] * columns + np.arange(columns - 1)[None, :]
        i = i.ravel()
        # Two triangles a quad, counter-clockwise seen from +y.
        faces = np.concatenate(
            [
                np.stack([i, i + columns, i + 1], axis=1),
                np.stack([i + 1, i + columns, i + columns + 1], axis=1),
            ]
        )
        return cls(positions, faces=faces, mass=mass, grid=(columns, rows), **values)

    @classmethod
    def rope(
        cls,
        points: npt.ArrayLike,
        segments: int | None = None,
        mass: float = 0.1,
        bend_compliance: float = math.inf,
        **values: Any,
    ) -> SoftBodySettings:
        """Return a rope along the polyline ``points``, metres in the body's frame.

        ``segments`` resamples it to that many equal segments; None keeps the
        points as given. Each segment is an edge of compliance ``compliance``;
        a finite ``bend_compliance`` adds an edge across every second vertex,
        which resists bending.
        """
        polyline = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(polyline) < 2:
            raise ValueError('a rope has at least two points')
        if segments is not None:
            lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
            at = np.linspace(0.0, lengths[-1], segments + 1)
            polyline = np.stack([np.interp(at, lengths, polyline[:, k]) for k in range(3)], axis=1)
        count = len(polyline)
        stretch = values.get('compliance', 0.0)
        edges = [np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)]
        compliance = [np.full(count - 1, stretch)]
        if math.isfinite(bend_compliance) and count > 2:
            edges.append(np.stack([np.arange(count - 2), np.arange(2, count)], axis=1))
            compliance.append(np.full(count - 2, bend_compliance))
        return cls(
            polyline,
            edges=np.concatenate(edges),
            edge_compliance=np.concatenate(compliance).astype(np.float32),
            mass=mass,
            bend_compliance=bend_compliance,
            **values,
        )

    @classmethod
    def ball(
        cls,
        radius: float = 0.5,
        subdivisions: int = 2,
        pressure: float = 1000.0,
        mass: float = 1.0,
        **values: Any,
    ) -> SoftBodySettings:
        """Return a closed sphere of ``radius`` held round by ``pressure``.

        The surface is an icosahedron with each face split ``subdivisions``
        times into four (20 times 4 to that power faces).
        """
        positions, faces = _icosphere(subdivisions)
        values.setdefault('bend', Bend.NONE)
        return cls(positions * radius, faces=faces, pressure=pressure, mass=mass, **values)

    @classmethod
    def from_mesh(
        cls, positions: npt.ArrayLike, faces: npt.ArrayLike, mass: float = 1.0, **values: Any
    ) -> SoftBodySettings:
        """Return a soft body of a triangle mesh: its vertices, and constraints from its faces."""
        return cls(np.asarray(positions), faces=np.asarray(faces), mass=mass, **values)

    # -- files and the native module ------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as plain data, arrays as lists."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, enum.Enum):
                value = int(value.value)
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SoftBodySettings:
        """Return the settings :meth:`as_dict` wrote."""
        values = dict(data)
        if values.get('grid') is not None:
            values['grid'] = tuple(values['grid'])
        values['bend'] = Bend(values['bend'])
        values['long_range'] = LongRange(values['long_range'])
        values['flags'] = BodyFlag(values['flags'])
        return cls(**values)

    def native(
        self,
        position: Sequence[float],
        rotation: Sequence[float],
        pin_vertex: npt.NDArray[np.uint32],
        pin_body: npt.NDArray[np.uint32],
        pin_point: npt.NDArray[np.float64],
    ) -> dict[str, Any]:
        """Return the keyword arguments of the native ``add_soft_body``."""
        masses = self.vertex_masses()
        inv_mass = np.divide(1.0, masses, out=np.zeros_like(masses), where=masses > 0).astype(
            np.float32
        )
        params = [
            *(float(v) for v in position),
            *(float(v) for v in rotation),
            self.compliance,
            self.shear_compliance,
            self.bend_compliance,
            float(int(self.bend)),
            self.volume_compliance,
            self.pressure,
            float(int(self.long_range)),
            self.long_range_stretch,
            float(self.iterations),
            self.linear_damping,
            self.max_velocity,
            self.friction,
            self.restitution,
            self.gravity_factor,
            self.vertex_radius,
            1.0 if self.allow_sleep else 0.0,
            1.0 if self.double_sided else 0.0,
            float(self.group),
            float(self.mask),
            float(self.filter),
            float(self.material),
            float(int(self.flags)),
        ]
        return {
            'params': np.array(params, dtype=np.float64),
            'positions': self.positions,
            'inv_mass': inv_mass,
            'faces': self.faces if self.faces is not None else _empty(3),
            'edges': self.edges if self.edges is not None else _empty(2),
            'edge_compliance': (
                self.edge_compliance
                if self.edge_compliance is not None
                else np.zeros(0, np.float32)
            ),
            'tetrahedra': self.tetrahedra if self.tetrahedra is not None else _empty(4),
            'pin_vertex': pin_vertex,
            'pin_body': pin_body,
            'pin_point': pin_point,
        }


def pin_arrays(
    count: int, vertices: Sequence[int], points: npt.ArrayLike, bodies: Sequence[int] | None
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float64]]:
    """Return a pin set as the native module reads it: vertex, body uid (0 for the world), point."""
    vertex = np.ascontiguousarray(vertices, dtype=np.uint32).reshape(-1)
    n = len(vertex)
    point = np.ascontiguousarray(points, dtype=np.float64).reshape(n, 3)
    body = (
        np.zeros(n, np.uint32)
        if bodies is None
        else np.ascontiguousarray(bodies, dtype=np.uint32).reshape(n)
    )
    if count and n and int(vertex.max()) >= count:
        raise ValueError('pin vertex %d names no vertex of %d' % (int(vertex.max()), count))
    return vertex, body, point


class SoftBodyRows:
    """Every soft body of one snapshot, as read-only NumPy views, row ``i`` the body ``uid[i]``.

    ``origin`` (n, 3) float64 is each body's centre of mass, also its row's
    position among the bodies. Its vertices are rows ``first_vertex[i]`` to
    ``first_vertex[i] + vertex_count[i]`` of ``positions`` and ``normals``
    (float32), relative to the origin; a vertex on no face has a zero
    normal. ``version`` is the tick the vertices were taken at: a renderer
    uploads a body's vertices when it differs from the one it last uploaded.
    """

    __slots__ = ('_native', 'tick')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        #: The tick of the snapshot.
        self.tick = native.tick

    def __len__(self) -> int:
        return int(self._native.soft_count)

    @property
    def uid(self) -> npt.NDArray[np.uint32]:
        """(n,) body uids, ascending."""
        return self._native.soft_uid

    @property
    def first_vertex(self) -> npt.NDArray[np.uint32]:
        """(n,) row of each body's first vertex."""
        return self._native.soft_first_vertex

    @property
    def vertex_count(self) -> npt.NDArray[np.uint32]:
        """(n,) number of vertices of each body."""
        return self._native.soft_vertices

    @property
    def version(self) -> npt.NDArray[np.uint64]:
        """(n,) tick the body's vertices were taken at."""
        return self._native.soft_version

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        """(n, 3) centre of mass of each body, metres."""
        return self._native.soft_origin

    @property
    def positions(self) -> npt.NDArray[np.float32]:
        """(v, 3) every body's vertices, relative to its origin."""
        return self._native.soft_position

    @property
    def normals(self) -> npt.NDArray[np.float32]:
        """(v, 3) every body's vertex normals."""
        return self._native.soft_normal

    def span(self, i: int) -> slice:
        """Return the vertex rows of body row ``i``."""
        first = int(self.first_vertex[i])
        return slice(first, first + int(self.vertex_count[i]))

    def vertices(self, i: int) -> npt.NDArray[np.float32]:
        """Return body row ``i``'s vertices, relative to its origin (a view)."""
        return self.positions[self.span(i)]

    def normals_of(self, i: int) -> npt.NDArray[np.float32]:
        """Return body row ``i``'s vertex normals (a view)."""
        return self.normals[self.span(i)]

    def rows(self) -> dict[int, int]:
        """Return the row of each soft body uid."""
        return {int(v): i for i, v in enumerate(self.uid)}


class SoftBody:
    """The handle of one soft body of a world: commands queued, vertices read from the newest snapshot."""

    __slots__ = ('settings', 'uid', 'world')

    def __init__(self, world: World, uid: int, settings: SoftBodySettings) -> None:
        self.world = world
        #: The body uid.
        self.uid = int(uid)
        #: The settings the body was made from.
        self.settings = settings

    def __repr__(self) -> str:
        return '<SoftBody %d vertices=%d>' % (self.uid, len(self.settings))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SoftBody) and other.world is self.world and other.uid == self.uid

    def __hash__(self) -> int:
        return hash((id(self.world), self.uid))

    def pin(
        self,
        vertices: Sequence[int],
        points: npt.ArrayLike,
        bodies: Sequence[int] | None = None,
        at_tick: int = 0,
    ) -> int:
        """Queue replacing the pins: each vertex follows its point, in the world or in a body's frame."""
        return self.world.set_soft_pins(self.uid, vertices, points, bodies, at_tick)

    def unpin(self, at_tick: int = 0) -> int:
        """Queue removing every pin; the vertices fall free with their own masses."""
        return self.world.set_soft_pins(self.uid, [], np.zeros((0, 3)), None, at_tick)

    def set_wind(self, force: Sequence[float], at_tick: int = 0) -> int:
        """Queue the wind on this body, a force in N shared among its vertices every tick."""
        return self.world.set_wind([self.uid], [tuple(force)], at_tick)

    def remove(self, at_tick: int = 0) -> int:
        """Queue removing the body."""
        return self.world.remove_bodies([self.uid], at_tick)

    def vertices(self) -> npt.NDArray[np.float64]:
        """Return the newest snapshot's vertices in world coordinates, (V, 3) float64 (a copy)."""
        snapshot = self.world.latest()
        if snapshot is None:
            raise LookupError('no snapshot has been published yet')
        rows = snapshot.soft_bodies
        found = np.flatnonzero(rows.uid == self.uid)
        if not len(found):
            raise LookupError('soft body %d is not in the newest snapshot' % self.uid)
        i = int(found[0])
        out: npt.NDArray[np.float64] = rows.vertices(i).astype(np.float64) + rows.origin[i]
        return out


def _icosphere(subdivisions: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Return a unit icosphere's vertices and outward-facing faces."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    points = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in points]
    for _ in range(subdivisions):
        middle: dict[tuple[int, int], int] = {}
        finer = []
        for a, b, c in faces:
            ab, bc, ca = (_midpoint(vertices, middle, i, j) for i, j in ((a, b), (b, c), (c, a)))
            finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = finer
    return np.array(vertices), np.array(faces, dtype=np.uint32)


def _midpoint(
    vertices: list[npt.NDArray[np.float64]], middle: dict[tuple[int, int], int], a: int, b: int
) -> int:
    """Return the index of the unit vertex between ``a`` and ``b``, adding it once."""
    key = (min(a, b), max(a, b))
    if key not in middle:
        m = vertices[a] + vertices[b]
        vertices.append(m / np.linalg.norm(m))
        middle[key] = len(vertices) - 1
    return middle[key]
