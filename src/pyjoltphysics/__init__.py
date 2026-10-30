"""Jolt Physics for Python, stepped on a native thread.

A :class:`World` owns a Jolt physics system and a thread that steps it at a
fixed rate. Bodies are added, changed and removed by command batches of
arrays (:class:`BodyBatch`); each tick publishes a :class:`Snapshot` read as
NumPy views, and contact, trigger, sleep and wake events as records of
:data:`EVENT_DTYPE`. Shapes are boxes, spheres, capsules, cylinders and
static planes.

>>> import pyjoltphysics
>>> pyjoltphysics.jolt_version()
'5.6.0'
>>> pyjoltphysics.build_options().double_precision
True
"""

from __future__ import annotations

import dataclasses

from pyjoltphysics import _native
from pyjoltphysics.bodies import BodyBatch
from pyjoltphysics.enums import BodyFlag, EndReason, EventKind, EventPhase, MotionType, ShapeKind
from pyjoltphysics.events import EVENT_DTYPE, event_times
from pyjoltphysics.settings import WorldSettings
from pyjoltphysics.world import (
    EventsCallback,
    Snapshot,
    StepCallback,
    World,
    WorldError,
    WorldStopped,
)

__version__ = '0.1.0a1'


@dataclasses.dataclass(frozen=True)
class BuildOptions:
    """The options the compiled module and its Jolt were built with.

    ``simd_level`` is the widest instruction set Jolt was compiled for:
    ``'sse4.2'``, ``'avx'``, ``'avx2'`` or ``'avx512'`` on x86-64, ``'neon'``
    on arm64, ``'none'`` elsewhere. ``debug`` is true when Jolt's assertions or
    the C++ debug checks are compiled in. ``compiler`` is the compiler's name
    and version, such as ``'gcc 13.3.0'``.
    """

    double_precision: bool
    cross_platform_deterministic: bool
    simd_level: str
    free_threaded: bool
    debug: bool
    compiler: str


def jolt_version() -> str:
    """Return the version of the Jolt Physics compiled into the module, such as ``'5.6.0'``."""
    return _native.jolt_version()


def build_options() -> BuildOptions:
    """Return the options the compiled module was built with."""
    native = _native.build_options()
    return BuildOptions(
        double_precision=bool(native['double_precision']),
        cross_platform_deterministic=bool(native['cross_platform_deterministic']),
        simd_level=str(native['simd_level']),
        free_threaded=bool(native['free_threaded']),
        debug=bool(native['debug']),
        compiler=str(native['compiler']),
    )


def selftest() -> bool:
    """Return True after stepping a one-body Jolt world once on the calling thread.

    The call exercises Jolt's allocator, type registry, physics system and
    single-threaded job system. A Jolt built with options that differ from the
    binding's raises ``RuntimeError``.
    """
    return _native.selftest()


__all__ = [
    # the world
    'World',
    'WorldSettings',
    'WorldError',
    'WorldStopped',
    'Snapshot',
    'StepCallback',
    'EventsCallback',
    # bodies
    'BodyBatch',
    'BodyFlag',
    'MotionType',
    'ShapeKind',
    # events
    'EVENT_DTYPE',
    'EndReason',
    'EventKind',
    'EventPhase',
    'event_times',
    # the build
    'BuildOptions',
    '__version__',
    'build_options',
    'jolt_version',
    'selftest',
]
