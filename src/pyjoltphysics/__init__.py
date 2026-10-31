"""Jolt Physics for Python, stepped on a native thread.

A :class:`World` owns a Jolt physics system and a thread that steps it at a
fixed rate. Bodies are added, changed and removed by command batches of
arrays (:class:`BodyBatch`); each tick publishes a :class:`Snapshot` read as
NumPy views, and contact, trigger, sleep and wake events as records of
:data:`EVENT_DTYPE`. Shapes are boxes, spheres, capsules, cylinders, static
planes, convex hulls, triangle meshes, height fields and compounds
(:class:`Shape`), built once and shared by id. Rays, sweeps and overlaps run
on the calling thread (:meth:`World.raycast`); characters
(:meth:`World.add_character`) move on the world thread every tick. Joints
(:class:`Joint`), materials (:attr:`World.materials`), collision filters,
gravity zones (:class:`GravityZone`) and kinematic tracks (:class:`Track`)
are commands like the rest.

>>> import pyjoltphysics
>>> pyjoltphysics.jolt_version()
'5.6.0'
>>> pyjoltphysics.build_options().double_precision
True
"""

from __future__ import annotations

import dataclasses

from pyjoltphysics import _native
from pyjoltphysics.bodies import BodyBatch, ShapeRef, heightfield_tiles
from pyjoltphysics.characters import (
    Character,
    CharacterCapabilities,
    CharacterOp,
    CharacterRows,
    CharacterState,
    MoveMode,
)
from pyjoltphysics.enums import (
    BodyFlag,
    Combine,
    EndReason,
    EventKind,
    EventPhase,
    JointKind,
    JointState,
    MotionType,
    MotorMode,
    ShapeKind,
    ZoneField,
    ZoneShape,
)
from pyjoltphysics.events import EVENT_DTYPE, event_times
from pyjoltphysics.fields import GravityZone, Track
from pyjoltphysics.joints import Joint, JointRows, Motor
from pyjoltphysics.materials import Material, MaterialTable
from pyjoltphysics.queries import ALL_GROUPS, QueryFilter, RayHit, RayHits, ShapeHit
from pyjoltphysics.settings import WorldSettings
from pyjoltphysics.shapes import Shape, ShapeInfo, heightfield_side
from pyjoltphysics.world import (
    BodiesState,
    ContactFilter,
    EventsCallback,
    PoseFunction,
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
    'BodiesState',
    # the world
    'World',
    'WorldSettings',
    'WorldError',
    'WorldStopped',
    'Snapshot',
    'StepCallback',
    'EventsCallback',
    'ContactFilter',
    'PoseFunction',
    # bodies
    'BodyBatch',
    'BodyFlag',
    'MotionType',
    'ShapeKind',
    # shapes
    'Shape',
    'ShapeInfo',
    'ShapeRef',
    'heightfield_side',
    'heightfield_tiles',
    # joints
    'Joint',
    'JointKind',
    'JointRows',
    'JointState',
    'Motor',
    'MotorMode',
    # materials
    'Combine',
    'Material',
    'MaterialTable',
    # gravity zones and kinematic tracks
    'GravityZone',
    'Track',
    'ZoneField',
    'ZoneShape',
    # queries
    'ALL_GROUPS',
    'QueryFilter',
    'RayHit',
    'RayHits',
    'ShapeHit',
    # characters
    'Character',
    'CharacterCapabilities',
    'CharacterOp',
    'CharacterRows',
    'CharacterState',
    'MoveMode',
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
