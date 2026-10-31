"""World and Snapshot: a Jolt world stepped on its own native thread.

>>> import numpy as np
>>> from pyjoltphysics import BodyBatch, World, WorldSettings
>>> with World(WorldSettings(threads=1)) as world:
...     floor = world.add_bodies(BodyBatch.plane())
...     uids, seq = world.add_bodies(BodyBatch.spheres([(0.0, 2.0, 0.0)], radius=0.5))
...     world.run_ticks(240)
...     snapshot = world.latest()
...     height = float(snapshot.position[snapshot.rows[int(uids[0])], 1])
>>> round(height, 1)
0.5

The world thread applies command batches at the start of each tick, steps,
then publishes a :class:`Snapshot`. Commands may come from any thread,
including a callback on the world thread, and take effect at the next tick.
"""

from __future__ import annotations

import atexit
import dataclasses
import math
import os
import threading
import weakref
from collections.abc import Callable, Iterable, Sequence
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native
from pyjoltphysics.bodies import BodyBatch
from pyjoltphysics.characters import Character, CharacterCapabilities, CharacterRows
from pyjoltphysics.enums import BodyFlag, MotionType, MotorMode, ShapeKind
from pyjoltphysics.events import as_events
from pyjoltphysics.fields import GravityZone, Track, zone_arrays
from pyjoltphysics.joints import Joint, JointRows, joint_arrays
from pyjoltphysics.materials import MaterialTable
from pyjoltphysics.queries import ALL_GROUPS, DEFAULT_FILTER, QueryFilter, RayHit, RayHits, ShapeHit
from pyjoltphysics.settings import WorldSettings
from pyjoltphysics.shapes import Shape, ShapeInfo

__all__ = [
    'BodiesState',
    'ContactFilter',
    'EventsCallback',
    'PoseFunction',
    'Snapshot',
    'StepCallback',
    'World',
    'WorldError',
    'WorldStopped',
]

#: ``fn(tick, time)`` on the world thread. For ``before_step`` ``tick`` is the
#: number of ticks completed; for ``after_step`` it includes the one just run.
StepCallback = Callable[[int, float], object]
#: ``fn(events, tick, time)`` on the world thread with the tick's records.
EventsCallback = Callable[[npt.NDArray[np.void], int, float], object]

#: ``fn(a, b, point, normal, approach)`` on Jolt's job threads, for two bodies
#: about to touch: ``a < b`` uids, the point (m), the unit normal from ``a`` to
#: ``b`` and the closing speed (m/s). False lets them pass through each other
#: until they part; True or None solves them.
ContactFilter = Callable[
    [int, int, tuple[float, float, float], tuple[float, float, float], float], object
]
#: ``fn(time)`` for :meth:`World.drive`: a position, or ``(position, quaternion)``.
PoseFunction = Callable[[float], Any]

#: Raised by a wait on a world that was stopped before the wait was met.
WorldStopped = _native.WorldStopped


class WorldError(RuntimeError):
    """A callback raised on the world thread, which stopped the world.

    The callback's exception is ``__cause__`` and ``World.error``.
    """


_hooks_lock = threading.Lock()
_hooks_installed = False


def _install_process_hooks() -> None:
    """Register the atexit and fork hooks with the first world made."""
    global _hooks_installed
    with _hooks_lock:
        if _hooks_installed:
            return
        # atexit hooks run before the interpreter finalizes: each world thread
        # is joined while it can still attach to run a callback, or skip one.
        atexit.register(_native.stop_all_worlds)
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(
                before=_native.fork_prepare,
                after_in_parent=_native.fork_parent,
                after_in_child=_native.fork_child,
            )
        _hooks_installed = True


def _uids(uids: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return np.ascontiguousarray(np.atleast_1d(np.asarray(uids)), dtype=np.uint32)


def _vectors(values: npt.ArrayLike, count: int, width: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = np.broadcast_to(array, (count, width))
    if array.shape != (count, width):
        raise ValueError(
            '%s has shape %s where (%d, %d) is needed' % (name, array.shape, count, width)
        )
    return np.ascontiguousarray(array)


def _column(values: npt.ArrayLike, count: int, dtype: type, name: str) -> np.ndarray:
    array: np.ndarray = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = np.broadcast_to(array, (count,))
    if array.shape != (count,):
        raise ValueError('%s has shape %s where (%d,) is needed' % (name, array.shape, count))
    return np.ascontiguousarray(array)


def _triple(values: Sequence[float], name: str) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError('%s is not finite' % (name,))
    return (x, y, z)


def _quad(values: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in values)
    return (x, y, z, w)


def _as_triple(row: npt.NDArray[np.floating[Any]]) -> tuple[float, float, float]:
    x, y, z = row.tolist()
    return (float(x), float(y), float(z))


class Snapshot:
    """The state of every body after one tick, as read-only NumPy views.

    The views share memory with the world's snapshot buffer and keep it from
    being reused while any of them, or this object, is alive. Row ``i`` of
    every array is the body ``uid[i]``; :attr:`rows` maps uid to row. Rows of
    one snapshot are not the rows of another: removing a body moves the last
    row into its place.
    """

    __slots__ = ('_native', '_rows')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        self._rows: dict[int, int] | None = None

    def __repr__(self) -> str:
        return '<Snapshot tick=%d bodies=%d applied_seq=%d>' % (
            self.tick,
            self.count,
            self.applied_seq,
        )

    @property
    def tick(self) -> int:
        """Ticks the world had completed."""
        return self._native.tick

    @property
    def time(self) -> float:
        """Simulation time in seconds, ``tick / tick_hz``."""
        return self._native.time

    @property
    def applied_seq(self) -> int:
        """The highest batch sequence number at and below which every batch is applied."""
        return self._native.applied_seq

    @property
    def count(self) -> int:
        """The number of bodies, and of rows."""
        return self._native.count

    @property
    def uid(self) -> npt.NDArray[np.uint32]:
        """(n,) body uid of each row."""
        return self._native.uid

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """(n, 3) body origins in metres, float64."""
        return self._native.position

    @property
    def orientation(self) -> npt.NDArray[np.float32]:
        """(n, 4) unit quaternions ``(x, y, z, w)``."""
        return self._native.orientation

    @property
    def linear_velocity(self) -> npt.NDArray[np.float32]:
        """(n, 3) m/s of each centre of mass."""
        return self._native.linear_velocity

    @property
    def angular_velocity(self) -> npt.NDArray[np.float32]:
        """(n, 3) rad/s."""
        return self._native.angular_velocity

    @property
    def awake(self) -> npt.NDArray[np.bool_]:
        """(n,) true for a body Jolt is simulating, false for a sleeping or static one."""
        return self._native.awake.view(np.bool_)

    @property
    def motion_type(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`~pyjoltphysics.MotionType` codes."""
        return self._native.motion_type

    @property
    def changed(self) -> npt.NDArray[np.uint32]:
        """Ascending rows this tick wrote: bodies that moved, fell asleep, were added or changed.

        Every other row holds what it held in the snapshot of the tick before.
        """
        return self._native.changed

    @property
    def rows(self) -> dict[int, int]:
        """Body uid to row."""
        if self._rows is None:
            self._rows = {int(uid): row for row, uid in enumerate(self._native.uid.tolist())}
        return self._rows

    def rows_of(self, uids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return the row of each uid, -1 for a uid that is not a body in this snapshot."""
        table = self.rows
        return np.array([table.get(int(u), -1) for u in np.atleast_1d(np.asarray(uids))], np.int64)

    @property
    def characters(self) -> CharacterRows:
        """Every character, as read-only views; see :class:`~pyjoltphysics.CharacterRows`."""
        return CharacterRows(self._native)

    @property
    def joints(self) -> JointRows:
        """Every joint, as read-only views; see :class:`~pyjoltphysics.JointRows`."""
        return JointRows(self._native)

    def includes(self, seq: int) -> bool:
        """True when the batch with sequence number ``seq``, and every batch before it, is applied."""
        return self.applied_seq >= seq


@dataclasses.dataclass(frozen=True)
class BodiesState:
    """Bodies' state read from Jolt: ``position`` (n, 3) float64, ``orientation`` (n, 4),
    ``linear_velocity`` and ``angular_velocity`` (n, 3) float32, ``awake`` and ``found``
    (n,) bool; a row whose uid names no body is zero with ``found`` false."""

    position: npt.NDArray[np.float64]
    orientation: npt.NDArray[np.float32]
    linear_velocity: npt.NDArray[np.float32]
    angular_velocity: npt.NDArray[np.float32]
    awake: npt.NDArray[np.bool_]
    found: npt.NDArray[np.bool_]


class World:
    """A Jolt physics world, stepped on a native thread the world owns.

    :meth:`start` free-runs it at ``tick_hz``; :meth:`run_ticks` runs exactly
    the ticks asked for and leaves the world in lockstep until the next
    :meth:`start`. A free-running world with no pending batch, no awake body
    and no step callback runs no ticks and uses no CPU. :meth:`stop` joins the
    thread; a world is also stopped when it is collected, by its ``with``
    block, and at interpreter exit.

    Every command method returns the batch's sequence number and takes
    ``at_tick``: the batch is applied at the first tick whose index (ticks
    completed before it) is at least ``at_tick``.
    """

    def __init__(self, settings: WorldSettings | None = None) -> None:
        _install_process_hooks()
        self._settings = settings if settings is not None else WorldSettings()
        self._native = _native.World(**self._settings.native())
        # Each Shape object is built once per world; the entry goes with the Shape.
        self._shape_ids: weakref.WeakKeyDictionary[Shape, int] = weakref.WeakKeyDictionary()
        self._shapes_lock = threading.Lock()
        self._materials = MaterialTable(self._send_materials)
        self._drivers: dict[int, PoseFunction] = {}
        self._drivers_lock = threading.Lock()

    def __enter__(self) -> World:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def __repr__(self) -> str:
        return '<World tick=%d %s>' % (self.tick, 'lockstep' if self.lockstep else 'free-running')

    @property
    def settings(self) -> WorldSettings:
        """The settings the world was made with."""
        return self._settings

    # -- the thread --------------------------------------------------------

    def start(self) -> None:
        """Free-run the world at ``tick_hz``, starting its thread if it is not running."""
        self._raising(self._native.start)

    def stop(self) -> None:
        """Stop the world thread and join it; from a callback, stop after the current tick."""
        self._native.stop()

    def run_ticks(self, count: int) -> None:
        """Run exactly ``count`` ticks on the world thread, unpaced, and return when they are done."""
        self._raising(self._native.run_ticks, count)

    def wait_tick(self, tick: int, timeout: float | None = None) -> bool:
        """Wait until a snapshot of ``tick`` or later is published; False on timeout or stop."""
        return bool(
            self._raising(self._native.wait_tick, tick, -1.0 if timeout is None else timeout)
        )

    @property
    def tick(self) -> int:
        """Ticks completed."""
        return self._native.tick

    @property
    def lockstep(self) -> bool:
        """True unless :meth:`start` is free-running the world."""
        return self._native.lockstep

    @property
    def error(self) -> BaseException | None:
        """The exception a callback raised, which stopped the world; None otherwise."""
        error = self._native.error
        return error if isinstance(error, BaseException) else None

    @property
    def submitted_seq(self) -> int:
        """The sequence number of the last batch submitted."""
        return self._native.submitted_seq

    def stats(self) -> dict[str, int | float | bool]:
        """Return counters and mean milliseconds a tick since :meth:`reset_stats`, by name."""
        return self._native.stats()

    def reset_stats(self) -> None:
        """Start the timing means again."""
        self._native.reset_stats()

    # -- reading -----------------------------------------------------------

    def latest(self) -> Snapshot | None:
        """Return the newest snapshot without waiting; None before the first tick."""
        native = self._native.latest()
        return None if native is None else Snapshot(native)

    def drain_events(self) -> npt.NDArray[np.void]:
        """Return and remove every event record since the last drain, as :data:`EVENT_DTYPE`."""
        return as_events(self._native.drain_events())

    def drain_applied(self) -> npt.NDArray[np.uint64]:
        """Return and remove the (seq, tick) of every batch applied since the last drain.

        ``tick`` is the index of the tick that applied it; a lockstep replay
        submits the same batch with ``at_tick=tick``.
        """
        return self._native.drain_applied()

    # -- shapes ------------------------------------------------------------

    def add_shape(self, shape: Shape) -> int:
        """Build ``shape`` on the calling thread and return its id in this world.

        The same :class:`Shape` object returns the same id every time. Bodies
        naming the id share the one Jolt shape. A shape Jolt cannot build
        raises ``ValueError`` with the reason.
        """
        with self._shapes_lock:
            known = self._shape_ids.get(shape)
            if known is not None:
                return known
            made = self._build(shape)
            self._shape_ids[shape] = made
            return made

    def remove_shape(self, shape_id: int) -> bool:
        """Forget a shape id, True if it named a shape; bodies made with the shape keep it."""
        with self._shapes_lock:
            for shape, known in list(self._shape_ids.items()):
                if known == shape_id:
                    del self._shape_ids[shape]
            return bool(self._native.remove_shape(shape_id))

    def shape_info(self, shape_id: int) -> ShapeInfo:
        """Return the kind, volume, centre of mass and bounds of a shape id."""
        info = self._native.shape_info(shape_id)

        def vector(name: str) -> tuple[float, float, float]:
            x, y, z = np.asarray(info[name], dtype=np.float64).tolist()
            return (x, y, z)

        return ShapeInfo(
            kind=ShapeKind(int(info['kind'])),  # type: ignore[call-overload]  # an int
            volume=float(info['volume']),  # type: ignore[arg-type]  # a float
            center_of_mass=vector('center_of_mass'),
            bounds_min=vector('bounds_min'),
            bounds_max=vector('bounds_max'),
            dynamic=bool(info['dynamic']),
        )

    def _build(self, shape: Shape) -> int:
        native = self._native
        if shape.kind == ShapeKind.CONVEX:
            assert shape.points is not None
            return native.add_convex(
                np.ascontiguousarray(shape.points), shape.max_convex_radius, shape.hull_tolerance
            )
        if shape.kind == ShapeKind.MESH:
            assert shape.points is not None and shape.triangles is not None
            return native.add_mesh(
                np.ascontiguousarray(shape.points),
                np.ascontiguousarray(shape.triangles),
                None if shape.materials is None else np.ascontiguousarray(shape.materials),
                None if shape.ids is None else np.ascontiguousarray(shape.ids),
            )
        if shape.kind == ShapeKind.HEIGHTFIELD:
            assert shape.heights is not None
            low, high = (
                shape.height_range if shape.height_range is not None else (math.nan, math.nan)
            )
            return native.add_heightfield(
                np.ascontiguousarray(shape.heights),
                shape.origin,
                shape.spacing,
                shape.block_size,
                shape.bits_per_sample,
                low,
                high,
            )
        if shape.kind == ShapeKind.COMPOUND:
            children = np.array(
                [
                    self._build_locked(child) if isinstance(child, Shape) else child
                    for child, _, _ in shape.parts
                ],
                dtype=np.uint32,
            ).reshape(-1)
            positions = np.array([p for _, p, _ in shape.parts], np.float64).reshape(-1, 3)
            orientations = np.array([q for _, _, q in shape.parts], np.float64).reshape(-1, 4)
            return native.add_compound(children, positions, orientations, shape.mutable)
        return native.add_primitive(int(shape.kind), shape.params)

    def _build_locked(self, shape: Shape) -> int:
        """add_shape for a compound's child, with the shapes lock already held."""
        known = self._shape_ids.get(shape)
        if known is None:
            known = self._shape_ids[shape] = self._build(shape)
        return known

    # -- commands ----------------------------------------------------------

    def add_bodies(self, batch: BodyBatch, at_tick: int = 0) -> tuple[npt.NDArray[np.uint32], int]:
        """Queue the batch's bodies; return their uids, assigned now, and the sequence number.

        A :class:`Shape` the batch names is built by :meth:`add_shape` first.
        """
        arrays = batch.arrays(
            self._settings.linear_damping, self._settings.angular_damping, self.add_shape
        )
        return self._native.add_bodies(**arrays, at_tick=at_tick)

    def remove_bodies(self, uids: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue the removal of bodies; their pairs end with reason ``REMOVED``."""
        return self._native.remove_bodies(_uids(uids), at_tick)

    def set_transforms(
        self,
        uids: npt.ArrayLike,
        positions: npt.ArrayLike,
        orientations: npt.ArrayLike = (0.0, 0.0, 0.0, 1.0),
        at_tick: int = 0,
    ) -> int:
        """Queue moving bodies to positions (m) and orientations (unit quaternions x, y, z, w)."""
        ids = _uids(uids)
        n = len(ids)
        return self._native.set_transforms(
            ids,
            _vectors(positions, n, 3, 'positions'),
            _vectors(orientations, n, 4, 'orientations'),
            at_tick,
        )

    def set_velocities(
        self,
        uids: npt.ArrayLike,
        linear: npt.ArrayLike | None = None,
        angular: npt.ArrayLike | None = None,
        at_tick: int = 0,
    ) -> int:
        """Queue linear (m/s) and angular (rad/s) velocities; None leaves one as it is."""
        ids = _uids(uids)
        n = len(ids)
        return self._native.set_velocities(
            ids,
            None if linear is None else _vectors(linear, n, 3, 'linear'),
            None if angular is None else _vectors(angular, n, 3, 'angular'),
            at_tick,
        )

    def add_forces(self, uids: npt.ArrayLike, forces: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue forces (N) at the centres of mass, acting over the next step."""
        ids = _uids(uids)
        return self._native.add_forces(ids, _vectors(forces, len(ids), 3, 'forces'), at_tick)

    def add_impulses(self, uids: npt.ArrayLike, impulses: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue impulses (N s) at the centres of mass, changing velocity at once."""
        ids = _uids(uids)
        return self._native.add_impulses(ids, _vectors(impulses, len(ids), 3, 'impulses'), at_tick)

    def add_torques(self, uids: npt.ArrayLike, torques: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue torques (N m), acting over the next step."""
        ids = _uids(uids)
        return self._native.add_torques(ids, _vectors(torques, len(ids), 3, 'torques'), at_tick)

    def set_gravity(self, gravity: tuple[float, float, float], at_tick: int = 0) -> int:
        """Queue a new gravity (m/s2); every dynamic body is woken to feel it."""
        g = tuple(float(v) for v in gravity)
        if len(g) != 3:
            raise ValueError('gravity has three components')
        return self._native.set_gravity((g[0], g[1], g[2]), at_tick)

    def activate(self, uids: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue waking bodies."""
        return self._native.activate(_uids(uids), at_tick)

    def set_motion_types(
        self, uids: npt.ArrayLike, motion: MotionType | npt.ArrayLike, at_tick: int = 0
    ) -> int:
        """Queue new motion types; a static body becomes moving only if it was added ``MOVABLE``."""
        ids = _uids(uids)
        return self._native.set_motion_types(
            ids, _column(motion, len(ids), np.uint8, 'motion'), at_tick
        )

    def set_materials(
        self,
        uids: npt.ArrayLike,
        friction: npt.ArrayLike,
        restitution: npt.ArrayLike,
        at_tick: int = 0,
    ) -> int:
        """Queue new friction and restitution coefficients."""
        ids = _uids(uids)
        n = len(ids)
        return self._native.set_materials(
            ids,
            _column(friction, n, np.float32, 'friction'),
            _column(restitution, n, np.float32, 'restitution'),
            at_tick,
        )

    def swap_bodies(
        self, remove: npt.ArrayLike, batch: BodyBatch, at_tick: int = 0
    ) -> tuple[npt.NDArray[np.uint32], int]:
        """Queue removing ``remove`` and adding ``batch`` in one tick; return the new uids and the sequence number.

        Ground streamed around a moving focus swaps tiles this way: no tick
        has the old tiles gone and the new ones not yet there.
        """
        arrays = batch.arrays(
            self._settings.linear_damping, self._settings.angular_damping, self.add_shape
        )
        return self._native.swap_bodies(_uids(remove), **arrays, at_tick=at_tick)

    # -- queries -----------------------------------------------------------

    def raycast(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        max_distance: float = 1000.0,
        filter: QueryFilter | None = None,
    ) -> RayHit | None:
        """Return the first body a ray from ``origin`` along ``direction`` meets within ``max_distance`` m.

        ``direction`` need not be a unit vector. None when the ray meets nothing.
        """
        hit = self._native.raycast(
            _triple(origin, 'origin'),
            _triple(direction, 'direction'),
            float(max_distance),
            *(filter or DEFAULT_FILTER).native(),
        )
        if hit is None:
            return None
        uid, fraction, point, normal, sub_shape = hit
        px, py, pz = point
        nx, ny, nz = normal
        return RayHit(
            uid=uid,
            distance=fraction * float(max_distance),
            fraction=fraction,
            point=(px, py, pz),
            normal=(nx, ny, nz),
            sub_shape=sub_shape,
        )

    def raycast_batch(
        self,
        origins: npt.ArrayLike,
        directions: npt.ArrayLike,
        max_distance: float = 1000.0,
        filter: QueryFilter | None = None,
    ) -> RayHits:
        """Cast one ray a row of ``origins`` (n, 3) along ``directions`` (n, 3); return the answers as arrays.

        A row whose direction is zero is a miss.
        """
        o = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 3)
        hit, uid, fraction, point, normal, sub = self._native.raycast_batch(
            o, d, float(max_distance), *(filter or DEFAULT_FILTER).native()
        )
        hits = hit.view(np.bool_)
        distance = np.where(hits, fraction.astype(np.float64) * float(max_distance), np.inf)
        return RayHits(hits, uid, fraction, distance, point, normal, sub)

    def sweep(
        self,
        shape: Shape | int,
        start: Sequence[float],
        end: Sequence[float],
        orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        filter: QueryFilter | None = None,
        all_hits: bool = False,
    ) -> Any:
        """Sweep ``shape`` at ``orientation`` from ``start`` to ``end``; return the first :class:`ShapeHit` or None.

        With ``all_hits``, return the first hit of every body along the way,
        in order. ``shape`` is a :class:`Shape` or a shape id.
        """
        kind, params, shape_id = self._query_shape(shape)
        uid, sub, fraction, depth, point, normal = self._native.cast_shape(
            kind,
            params,
            shape_id,
            _triple(start, 'start'),
            _quad(orientation),
            _triple(end, 'end'),
            all_hits,
            *(filter or DEFAULT_FILTER).native(),
        )
        length = math.dist(_triple(start, 'start'), _triple(end, 'end'))
        hits = [
            ShapeHit(
                uid=int(uid[k]),
                fraction=float(fraction[k]),
                distance=float(fraction[k]) * length,
                point=_as_triple(point[k]),
                normal=_as_triple(normal[k]),
                depth=float(depth[k]),
                sub_shape=int(sub[k]),
            )
            for k in range(len(uid))
        ]
        if all_hits:
            return hits
        return hits[0] if hits else None

    def sweep_sphere(
        self,
        radius: float,
        start: Sequence[float],
        end: Sequence[float],
        filter: QueryFilter | None = None,
        all_hits: bool = False,
    ) -> Any:
        """Sweep a sphere of ``radius`` from ``start`` to ``end``; see :meth:`sweep`."""
        return self.sweep(Shape.sphere(radius), start, end, filter=filter, all_hits=all_hits)

    def sweep_capsule(
        self,
        radius: float,
        height: float,
        start: Sequence[float],
        end: Sequence[float],
        orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        filter: QueryFilter | None = None,
        all_hits: bool = False,
    ) -> Any:
        """Sweep a Y-axis capsule (``height`` between the hemisphere centres); see :meth:`sweep`."""
        return self.sweep(Shape.capsule(radius, height), start, end, orientation, filter, all_hits)

    def sweep_box(
        self,
        size: Sequence[float],
        start: Sequence[float],
        end: Sequence[float],
        orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        filter: QueryFilter | None = None,
        all_hits: bool = False,
    ) -> Any:
        """Sweep a box of edge lengths ``size``; see :meth:`sweep`."""
        return self.sweep(Shape.box(size), start, end, orientation, filter, all_hits)

    def overlap(
        self,
        shape: Shape | int,
        position: Sequence[float],
        orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        filter: QueryFilter | None = None,
    ) -> npt.NDArray[np.uint32]:
        """Return the ascending uids of the bodies ``shape`` touches at ``position`` and ``orientation``."""
        kind, params, shape_id = self._query_shape(shape)
        return self._native.overlap(
            kind,
            params,
            shape_id,
            _triple(position, 'position'),
            _quad(orientation),
            *(filter or DEFAULT_FILTER).native(),
        )

    def _query_shape(
        self, shape: Shape | int
    ) -> tuple[int, tuple[float, float, float, float], int]:
        if isinstance(shape, Shape):
            if shape.kind < ShapeKind.CONVEX:
                return int(shape.kind), shape.params, 0
            return 0, (0.0, 0.0, 0.0, 0.0), self.add_shape(shape)
        return 0, (0.0, 0.0, 0.0, 0.0), int(shape)

    # -- characters --------------------------------------------------------

    def add_character(
        self,
        position: Sequence[float],
        capabilities: CharacterCapabilities | None = None,
        gravity: float | None = None,
        pass_group: int = 0,
        group: int = 1,
        mask: int = ALL_GROUPS,
        ignore: Iterable[int] = (),
        proxy: bool = True,
        at_tick: int = 0,
    ) -> Character:
        """Queue a character with its capsule's centre at ``position``; return its :class:`Character`.

        ``gravity`` (m/s2, downward) defaults to the length of the world's.
        Characters of one ``pass_group`` (other than 0) do not block each
        other. ``group`` and ``mask`` are collision bits as bodies have them;
        ``ignore`` names bodies the capsule passes through. ``proxy`` puts a
        kinematic capsule in the world under the character's id, which rays
        and sensors meet and trigger events name, and against which no body
        is solved.
        """
        caps = capabilities if capabilities is not None else CharacterCapabilities()
        values, bits = caps.native()
        g = float(np.linalg.norm(self._settings.gravity)) if gravity is None else float(gravity)
        cid, _ = self._native.add_character(
            _triple(position, 'position'),
            values,
            bits,
            g,
            int(group),
            int(mask),
            int(pass_group),
            bool(proxy),
            _uids(list(ignore)),
            at_tick,
        )
        return Character(self, cid)

    def character(self, cid: int) -> Character:
        """Return the :class:`Character` handle of a character id."""
        return Character(self, cid)

    def remove_characters(self, ids: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue removing characters and their proxy bodies."""
        return self._native.remove_characters(_uids(ids), at_tick)

    def character_ops(
        self,
        ids: npt.ArrayLike,
        ops: npt.ArrayLike,
        codes: npt.ArrayLike,
        values: npt.ArrayLike,
        at_tick: int = 0,
    ) -> int:
        """Queue character inputs as arrays, applied in order at the next tick.

        One row an input: the character id, a :class:`~pyjoltphysics.CharacterOp`,
        its small code and four values; :class:`Character`'s methods build them.
        """
        ids_ = _uids(ids)
        n = len(ids_)
        return self._native.character_ops(
            ids_,
            _column(ops, n, np.uint8, 'ops'),
            _column(codes, n, np.uint8, 'codes'),
            _vectors(values, n, 4, 'values'),
            at_tick,
        )

    def set_character_ignore(self, cid: int, uids: Iterable[int], at_tick: int = 0) -> int:
        """Queue the set of bodies a character passes through."""
        return self._native.set_character_ignore(int(cid), _uids(list(uids)), at_tick)

    # -- joints ------------------------------------------------------------

    def add_joints(
        self, joints: Sequence[Joint], at_tick: int = 0
    ) -> tuple[npt.NDArray[np.uint32], int]:
        """Queue joints; return their ids, assigned now, and the batch's sequence number.

        A joint naming a body that is not in the world when the batch is
        applied is skipped and counted in ``stats()['rejected']``.
        """
        return self._native.add_joints(**joint_arrays(list(joints)), at_tick=at_tick)

    def add_joint(self, joint: Joint, at_tick: int = 0) -> int:
        """Queue one joint and return its id."""
        ids, _ = self.add_joints([joint], at_tick)
        return int(ids[0])

    def remove_joints(self, ids: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue taking joints out of the world; each makes an end record."""
        return self._native.remove_joints(_uids(ids), at_tick)

    def set_joint_motors(
        self,
        ids: npt.ArrayLike,
        mode: MotorMode | int | None,
        target: npt.ArrayLike = math.nan,
        max: npt.ArrayLike = math.nan,
        axis: npt.ArrayLike = 0,
        at_tick: int = 0,
    ) -> int:
        """Queue new motor settings for one axis of each joint; None or NaN leaves a value as it is.

        A MOTOR joint's axis 0 to 2 takes a new torque as its ``target``.
        """
        ids_ = _uids(ids)
        n = len(ids_)
        return self._native.set_joint_motors(
            ids_,
            _column(axis, n, np.uint8, 'axis'),
            _column(0xFF if mode is None else int(mode), n, np.uint8, 'mode'),
            _column(target, n, np.float64, 'target'),
            _column(max, n, np.float64, 'max'),
            at_tick,
        )

    def set_joints_enabled(
        self, ids: npt.ArrayLike, enabled: npt.ArrayLike, at_tick: int = 0
    ) -> int:
        """Queue switching joints off (False) or on (True); a broken joint stays broken."""
        ids_ = _uids(ids)
        return self._native.set_joints_enabled(
            ids_, _column(enabled, len(ids_), np.uint8, 'enabled'), at_tick
        )

    # -- materials and filters ---------------------------------------------

    @property
    def materials(self) -> MaterialTable:
        """The world's :class:`~pyjoltphysics.MaterialTable`; each change is sent as a batch."""
        return self._materials

    def _send_materials(self, table: MaterialTable) -> int:
        static, dynamic, restitution, fcombine, rcombine, pairs = table.native()
        return self._native.set_material_table(
            static,
            dynamic,
            restitution,
            fcombine,
            rcombine,
            pairs,
            self._settings.static_friction_speed,
        )

    def set_body_materials(
        self, uids: npt.ArrayLike, material: npt.ArrayLike, at_tick: int = 0
    ) -> int:
        """Queue new material indices for bodies; 0 is a body's own friction and restitution."""
        ids = _uids(uids)
        return self._native.set_body_materials(
            ids, _column(material, len(ids), np.uint16, 'material'), at_tick
        )

    def set_filter_table(self, collides: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue the (F, F) table of which collision filters meet; it is made symmetric.

        A body's ``filter`` index outside the table meets every filter. Every
        moving body is woken, so pairs the table parts or joins are met at once.
        """
        table = np.asarray(collides, dtype=bool)
        if table.ndim != 2 or table.shape[0] != table.shape[1]:
            raise ValueError('the filter table is (F, F), not %s' % (table.shape,))
        both = np.ascontiguousarray(table & table.T, dtype=np.uint8)
        return self._native.set_filter_table(both, at_tick)

    def set_body_filters(
        self, uids: npt.ArrayLike, filters: npt.ArrayLike, at_tick: int = 0
    ) -> int:
        """Queue new collision filter indices for bodies."""
        ids = _uids(uids)
        return self._native.set_body_filters(
            ids, _column(filters, len(ids), np.uint16, 'filters'), at_tick
        )

    def set_contact_filter(
        self, fn: ContactFilter | None, either: bool = False, recheck: bool = False
    ) -> ContactFilter | None:
        """Ask ``fn`` about pairs whose bodies both carry ``BodyFlag.FILTER_CONTACTS`` as they first touch.

        With ``either``, a pair is asked about where either body carries it.
        With ``recheck``, a refusal holds for the step that made it and the
        pair is asked again at each later step it meets, so a rule that
        changes from step to step lets a refused pair collide.

        ``fn(a, b, point, normal, approach)`` runs on the Jolt job thread that
        found the pair, inside the step, once per pair: the answer holds until
        the two part. False lets them pass through each other; True or None
        solves them. It may submit commands and may not query the world
        (``RuntimeError``) or wait for it. An exception stops the world, as
        from any callback. None removes it.
        """
        self._native.set_contact_filter(fn, bool(either), bool(recheck))
        return fn

    # -- gravity zones, drag, kinematic tracks -------------------------------

    def set_gravity_zones(self, zones: Sequence[GravityZone], at_tick: int = 0) -> int:
        """Queue the world's gravity zones, replacing any it has; every dynamic body is woken."""
        zones = list(zones)
        shape_ids = [z.shape_id for z in zones]
        if any(sid == 0 for z, sid in zip(zones, shape_ids, strict=True) if int(z.shape) == 2):
            raise ValueError('a shape zone names a shape id')
        return self._native.set_gravity_zones(**zone_arrays(zones), at_tick=at_tick)

    def set_drag(self, uids: npt.ArrayLike, drag: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue quadratic drag coefficients (1/m): a force of ``-k m |v| v``; 0 removes it."""
        ids = _uids(uids)
        return self._native.set_drag(ids, _column(drag, len(ids), np.float32, 'drag'), at_tick)

    def set_motion(
        self,
        uids: npt.ArrayLike,
        mass: npt.ArrayLike | None = None,
        linear_damping: npt.ArrayLike | None = None,
        angular_damping: npt.ArrayLike | None = None,
        gravity_factor: npt.ArrayLike | None = None,
        at_tick: int = 0,
    ) -> int:
        """Queue new motion properties for moving bodies; None, or NaN in an array, keeps a value.

        ``mass`` (kg, dynamic bodies) scales the inertia with it; damping is
        1/s; ``gravity_factor`` multiplies gravity. A static body is rejected.
        Each changed dynamic body is woken.
        """
        ids = _uids(uids)
        n = len(ids)

        def column(values: npt.ArrayLike | None, name: str) -> np.ndarray:
            return _column(np.nan if values is None else values, n, np.float32, name)

        return self._native.set_motion(
            ids,
            column(mass, 'mass'),
            column(linear_damping, 'linear_damping'),
            column(angular_damping, 'angular_damping'),
            column(gravity_factor, 'gravity_factor'),
            at_tick,
        )

    def set_flags(self, uids: npt.ArrayLike, flags: BodyFlag, on: bool, at_tick: int = 0) -> int:
        """Queue setting (``on``) or clearing ``flags`` on bodies: ``REPORT``, ``NO_SLEEP`` and ``FILTER_CONTACTS``."""
        bits = int(flags)
        if bits & ~int(BodyFlag.REPORT | BodyFlag.NO_SLEEP | BodyFlag.FILTER_CONTACTS):
            raise ValueError('only REPORT, NO_SLEEP and FILTER_CONTACTS change after a body is added')
        return self._native.set_flags(_uids(uids), bits, bool(on), at_tick)

    def bodies_state(self, uids: npt.ArrayLike) -> BodiesState:
        """The bodies of ``uids`` as Jolt holds them now, not as the last snapshot published them.

        From a world callback it reads the world as the tick stands (after its
        batches, before its step); from another thread it takes the world
        lock's shared side, as a query does.
        """
        position, orientation, linear, angular, awake, found = self._native.bodies_state(_uids(uids))
        return BodiesState(position, orientation, linear, angular, awake.astype(bool),
                           found.astype(bool))

    def triangle_of(self, uid: int, sub_shape: int) -> int:
        """The number of the mesh triangle a query's ``sub_shape`` names on body ``uid``; -1 for another part.

        The number is the triangle's row in the mesh's ``triangles``, or the
        ``ids`` the mesh was given; a compound or decorated body's mesh child
        answers the same.
        """
        return int(self._native.triangle_of(int(uid), int(sub_shape)))

    def flush(self) -> None:
        """Apply the batches due at the current tick now, on the calling thread, without stepping.

        Queries then meet the bodies those batches add and miss the ones they
        remove, and a snapshot of the current tick is published with what
        they changed (its ``applied_seq`` includes them). The world lock is
        taken exclusively for the apply, so a free-running world thread waits
        for it between its own phases; on the world thread, inside a callback,
        it does nothing.
        """
        self._native.flush()

    def save_state(self) -> bytes:
        """The bodies' and joints' state (Jolt's ``PhysicsSystem::SaveState``), read between ticks.

        Positions, velocities, sleep, contacts and constraint state; not the
        characters, the tick count or the batches still queued.
        """
        return bytes(self._native.save_state())

    def restore_state(self, data: bytes, at_tick: int = 0) -> int:
        """Queue reading a :meth:`save_state` back, at the start of the tick that applies it.

        The world must hold the bodies and joints it held when the state was
        saved; a state of another layout is rejected (``stats()['rejected']``).
        """
        return self._native.restore_state(bytes(data), at_tick)

    def set_track(self, uids: npt.ArrayLike, track: Track, at_tick: int = 0) -> int:
        """Queue kinematic bodies onto ``track``, from the tick that applies the batch.

        Each tick the world moves each body to the track's pose at the end of
        the tick, so what rides on it is carried. A body that is not kinematic
        is rejected.
        """
        times, positions, rotations = track.native()
        return self._native.set_tracks(
            _uids(uids), times, positions, rotations, track.loop, at_tick
        )

    def clear_tracks(self, uids: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue taking bodies off their tracks; they keep the pose they reached."""
        empty = np.zeros((0, 3))
        return self._native.set_tracks(
            _uids(uids), np.zeros(0), empty, np.zeros((0, 4)), False, at_tick
        )

    def move_kinematic(
        self,
        uids: npt.ArrayLike,
        positions: npt.ArrayLike,
        orientations: npt.ArrayLike = (0.0, 0.0, 0.0, 1.0),
        at_tick: int = 0,
    ) -> int:
        """Move kinematic bodies to poses by the end of a tick, at the velocity that reaches them.

        From a callback on the world thread the move belongs to the current
        tick (and the call returns 0); from any other thread it is a batch
        applied at the next tick.
        """
        ids = _uids(uids)
        n = len(ids)
        return self._native.move_kinematic(
            ids,
            _vectors(positions, n, 3, 'positions'),
            _vectors(orientations, n, 4, 'orientations'),
            at_tick,
        )

    def drive(self, uid: int, pose: PoseFunction | None) -> None:
        """Move kinematic body ``uid`` along ``pose(time)`` each tick; None stops.

        ``pose`` runs on the world thread before each step with the
        simulation time at the end of the tick, and returns a position or
        ``(position, quaternion)``. The body reaches that pose by the end of
        the tick. A world with a driven body keeps ticking while free-running.
        """
        with self._drivers_lock:
            if pose is None:
                self._drivers.pop(int(uid), None)
            else:
                self._drivers[int(uid)] = pose
            active = bool(self._drivers)
        self._native.set_drivers(self._drive_all if active else None)

    def _drive_all(self, _tick: int, time: float) -> None:
        with self._drivers_lock:
            drivers = sorted(self._drivers.items())
        end = time + self._settings.dt
        uids = np.empty(len(drivers), np.uint32)
        positions = np.empty((len(drivers), 3))
        rotations = np.empty((len(drivers), 4))
        for k, (uid, pose) in enumerate(drivers):
            got = pose(end)
            if len(got) == 2 and np.ndim(got[0]) == 1:
                positions[k], rotations[k] = got[0], got[1]
            else:
                positions[k], rotations[k] = got, (0.0, 0.0, 0.0, 1.0)
            uids[k] = uid
        self._native.move_kinematic(uids, positions, rotations, 0)

    # -- callbacks on the world thread -------------------------------------

    def set_before_step(self, fn: StepCallback | None) -> StepCallback | None:
        """Call ``fn(tick, time)`` each tick after its batches are applied; None removes it.

        Returns ``fn``, so the method can decorate it. A world with
        a step callback keeps ticking while free-running.
        """
        self._native.set_before_step(fn)
        return fn

    def set_after_step(self, fn: StepCallback | None) -> StepCallback | None:
        """Call ``fn(tick, time)`` each tick after its snapshot is published; None removes it."""
        self._native.set_after_step(fn)
        return fn

    def set_on_events(self, fn: EventsCallback | None) -> EventsCallback | None:
        """Call ``fn(events, tick, time)`` after each tick that made event records; None removes it."""
        if fn is None:
            self._native.set_on_events(None)
        else:
            self._native.set_on_events(_EventsAdapter(fn))
        return fn

    def at_tick(self, tick: int, fn: StepCallback) -> None:
        """Call ``fn(tick, time)`` once, before the step of the first tick at least ``tick``."""
        self._native.at_tick(tick, fn)

    def _raising(self, call: Callable[..., Any], *args: Any) -> Any:
        try:
            return call(*args)
        except _native.CallbackFailed as failed:
            raise WorldError(str(failed)) from self.error


class _EventsAdapter:
    """Views the native record bytes as EVENT_DTYPE for an ``on_events`` callable."""

    __slots__ = ('fn',)

    def __init__(self, fn: EventsCallback) -> None:
        self.fn = fn

    def __call__(self, raw: npt.NDArray[np.uint8], tick: int, time: float) -> object:
        return self.fn(as_events(raw), tick, time)
