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
import os
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native
from pyjoltphysics.bodies import BodyBatch
from pyjoltphysics.enums import MotionType
from pyjoltphysics.events import as_events
from pyjoltphysics.settings import WorldSettings

__all__ = ['Snapshot', 'StepCallback', 'EventsCallback', 'World', 'WorldError', 'WorldStopped']

#: ``fn(tick, time)`` on the world thread. For ``before_step`` ``tick`` is the
#: number of ticks completed; for ``after_step`` it includes the one just run.
StepCallback = Callable[[int, float], object]
#: ``fn(events, tick, time)`` on the world thread with the tick's records.
EventsCallback = Callable[[npt.NDArray[np.void], int, float], object]

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

    def includes(self, seq: int) -> bool:
        """True when the batch with sequence number ``seq``, and every batch before it, is applied."""
        return self.applied_seq >= seq


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

    # -- commands ----------------------------------------------------------

    def add_bodies(self, batch: BodyBatch, at_tick: int = 0) -> tuple[npt.NDArray[np.uint32], int]:
        """Queue the batch's bodies; return their uids, assigned now, and the sequence number."""
        arrays = batch.arrays(self._settings.linear_damping, self._settings.angular_damping)
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

    # -- callbacks on the world thread -------------------------------------

    def set_before_step(self, fn: StepCallback | None) -> StepCallback | None:
        """Call ``fn(tick, time)`` each tick after its batches are applied; None removes it.

        Returns ``fn``, so the method also serves as a decorator. A world with
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
