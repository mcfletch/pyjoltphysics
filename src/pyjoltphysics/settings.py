"""WorldSettings: how a world steps, how big it may grow, and its defaults."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

__all__ = ['WorldSettings']

#: The tick rates a world steps at, Hz.
MIN_TICK_HZ = 1.0
MAX_TICK_HZ = 10000.0
#: The most job threads a world starts, the world thread included.
MAX_THREADS = 256
#: Jolt sizes its scratch allocator in a 32-bit byte count.
MAX_TEMP_ALLOCATOR_MB = 4095
_FLOAT_MAX = 3.4028234663852886e38


@dataclasses.dataclass(frozen=True)
class WorldSettings:
    """The fixed settings of one :class:`~pyjoltphysics.World`.

    Units are SI: seconds, metres, kilograms. ``threads`` counts Jolt's job
    threads including the world thread, 1 to 256; 1 steps on the world thread alone.
    """

    #: Ticks a second, each one step of ``1 / tick_hz`` seconds; 1 to 10000.
    tick_hz: float = 120.0
    #: Seconds a free-running world may fall behind its schedule before it
    #: drops ticks (counted in ``stats()['dropped_ticks']``), so simulated time
    #: slows rather than the world running ticks back to back.
    max_catch_up: float = 0.1
    #: Gravity in m/s2.
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    threads: int = 4
    max_bodies: int = 65536
    max_body_pairs: int = 65536
    max_contact_constraints: int = 65536
    #: Event records held for :meth:`~pyjoltphysics.World.drain_events`; past
    #: it persist records are dropped first, then sleep and wake records.
    event_capacity: int = 65536
    allow_sleeping: bool = True
    #: Seconds a body stays below ``sleep_velocity`` before it falls asleep.
    time_before_sleep: float = 0.5
    #: m/s at every point of a body, below which it may fall asleep.
    sleep_velocity: float = 0.03
    #: m/s of closing speed below which a contact does not bounce.
    min_velocity_for_restitution: float = 1.0
    collision_steps: int = 1
    #: Megabytes Jolt allocates up front for one step's scratch memory, 0 to
    #: 4095; a step that needs more allocates the rest from the heap.
    temp_allocator_mb: int = 16
    #: m/s of sliding at a contact below which a material's static friction
    #: holds; above it the dynamic friction acts.
    static_friction_speed: float = 0.05
    #: Metres two bodies may sink into each other before the solver pushes
    #: them apart (Jolt's ``mPenetrationSlop``): a resting body sits up to this
    #: far into what it rests on.
    penetration_slop: float = 0.02
    #: Body damping where a :class:`~pyjoltphysics.BodyBatch` names none, 1/s.
    linear_damping: float = 0.0
    angular_damping: float = 0.0

    def __post_init__(self) -> None:
        if not (MIN_TICK_HZ <= self.tick_hz <= MAX_TICK_HZ):
            raise ValueError(
                'tick_hz is from %g to %g, not %r' % (MIN_TICK_HZ, MAX_TICK_HZ, self.tick_hz)
            )
        if not (0 <= self.temp_allocator_mb <= MAX_TEMP_ALLOCATOR_MB):
            raise ValueError(
                'temp_allocator_mb is from 0 to %d, not %r'
                % (MAX_TEMP_ALLOCATOR_MB, self.temp_allocator_mb)
            )
        if not 1 <= self.threads <= MAX_THREADS:
            raise ValueError('threads is from 1 to %d, not %r' % (MAX_THREADS, self.threads))
        if len(self.gravity) != 3:
            raise ValueError('gravity has three components')
        if not all(math.isfinite(g) and abs(g) <= _FLOAT_MAX for g in self.gravity):
            raise ValueError('gravity is finite as 32-bit floats, not %r' % (self.gravity,))

    @property
    def dt(self) -> float:
        """Seconds a tick steps the world."""
        return 1.0 / self.tick_hz

    def native(self) -> dict[str, Any]:
        """Return the keyword arguments of the native world's constructor."""
        values = dataclasses.asdict(self)
        del values['linear_damping'], values['angular_damping']
        values['gravity'] = tuple(float(g) for g in self.gravity)
        return values
