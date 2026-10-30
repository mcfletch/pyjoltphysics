"""Jolt Physics for Python, stepped on a native thread.

This release carries the build of Jolt and the binding toolchain. The world,
its bodies and its queries follow in later releases; see the project README.

>>> import pyjoltphysics
>>> pyjoltphysics.jolt_version()
'5.6.0'
>>> pyjoltphysics.build_options().double_precision
True
"""

from __future__ import annotations

import dataclasses

from pyjoltphysics import _native

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
    'BuildOptions',
    '__version__',
    'build_options',
    'jolt_version',
    'selftest',
]
