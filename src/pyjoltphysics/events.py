"""The event record, as a NumPy structured dtype.

``World.drain_events()`` returns every record since the last drain as one
array of :data:`EVENT_DTYPE`; an ``on_events`` callback receives the records
of one tick the same way. Each record is 64 bytes:

========== ========= ===============================================================
field      type      meaning
========== ========= ===============================================================
kind       uint8     :class:`~pyjoltphysics.EventKind`
phase      uint8     :class:`~pyjoltphysics.EventPhase`
reason     uint16    :class:`~pyjoltphysics.EndReason`, for end records
tick       uint32    the tick the event happened in, which is the tick of the first
                     snapshot that shows it; its time is ``tick / tick_hz`` seconds
a          uint32    body uid: the lower uid of a contact, the sensor of a trigger,
                     the body of a sleep or wake; the joint id of a joint record
b          uint32    the other body's uid; 0 for sleep and wake; a joint's body
point      3 float64 world position of the contact, metres (begin and persist)
normal     3 float32 unit vector from ``a`` to ``b`` (contacts, begin and persist)
approach   float32   closing speed of the two bodies along the normal at the point,
                     m/s, positive when they approach (begin and persist)
impulse    float32   estimated normal impulse of the collision, N s, in contact begin
                     records; 0 in every other record
material_a uint16    contact begin and persist: the material index of ``a``'s side,
                     a mesh triangle's own where it has one
material_b uint16    the same for ``b``
========== ========= ===============================================================

Jolt reports contacts to its listeners before its solver runs, so the impulse
of a begin record is Jolt's ``EstimateCollisionResponse`` for the two bodies
alone, and no record carries the impulse the solver applied.

A record has no time field. Simulation time is exactly ``tick / tick_hz``,
which :func:`event_times` computes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native

__all__ = ['EVENT_DTYPE', 'event_times']

EVENT_DTYPE = np.dtype(
    [
        ('kind', '<u1'),
        ('phase', '<u1'),
        ('reason', '<u2'),
        ('tick', '<u4'),
        ('a', '<u4'),
        ('b', '<u4'),
        ('point', '<f8', (3,)),
        ('normal', '<f4', (3,)),
        ('approach', '<f4'),
        ('impulse', '<f4'),
        ('material_a', '<u2'),
        ('material_b', '<u2'),
    ]
)
if EVENT_DTYPE.itemsize != _native.EVENT_SIZE:  # pragma: no cover - a build defect
    raise ImportError(
        'EVENT_DTYPE is %d bytes and the native record %d'
        % (EVENT_DTYPE.itemsize, _native.EVENT_SIZE)
    )


def as_events(raw: npt.NDArray[np.uint8]) -> npt.NDArray[np.void]:
    """Return the native module's record bytes as an array of :data:`EVENT_DTYPE`."""
    return raw.view(EVENT_DTYPE)


def event_times(events: npt.NDArray[np.void], tick_hz: float) -> npt.NDArray[np.float64]:
    """Return each record's simulation time in seconds, for a world stepping at ``tick_hz``."""
    return np.asarray(events['tick'], dtype=np.float64) / float(tick_hz)
