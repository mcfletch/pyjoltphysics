"""WorldSettings: how a world steps, how big it may grow, and its defaults."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

__all__ = ['WorldSettings']


@dataclasses.dataclass(frozen=True)
class WorldSettings:
    """The fixed settings of one :class:`~pyjoltphysics.World`.

    Units are SI: seconds, metres, kilograms. ``threads`` counts Jolt's job
    threads including the world thread; 1 steps on the world thread alone.
    """

    #: Ticks a second, each one step of ``1 / tick_hz`` seconds.
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
    #: Megabytes Jolt allocates up front for one step's scratch memory; a step
    #: that needs more allocates the rest from the heap.
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
        if not (math.isfinite(self.tick_hz) and self.tick_hz > 0):
            raise ValueError('tick_hz is positive, not %r' % (self.tick_hz,))
        if self.threads < 1:
            raise ValueError('threads is at least 1, not %r' % (self.threads,))
        if len(self.gravity) != 3:
            raise ValueError('gravity has three components')

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
