"""The compiled module imports, links Jolt and was built with the decided options."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import platform
import re
import subprocess
import sys
import sysconfig

import pytest

import pyjoltphysics
from pyjoltphysics import BuildOptions, _native

X86_64 = platform.machine().lower() in {'x86_64', 'amd64'}
ARM64 = platform.machine().lower() in {'arm64', 'aarch64'}
FREE_THREADED_PYTHON = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))
CHECKOUT = pathlib.Path(__file__).resolve().parent.parent


def test_the_package_version_is_a_pep_440_release() -> None:
    # release.yml reads this string to decide whether to publish, and the build
    # takes the distribution's version from it.
    assert re.fullmatch(
        r'\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?', pyjoltphysics.__version__
    )


def test_jolt_version_is_the_vendored_tag() -> None:
    assert pyjoltphysics.jolt_version() == '5.6.0'
    assert _native.jolt_version() == '5.6.0'


def test_build_options_is_a_frozen_record() -> None:
    options = pyjoltphysics.build_options()
    assert isinstance(options, BuildOptions)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.debug = True  # type: ignore[misc]  # assigning to a frozen field is the case under test


def test_build_options_carry_every_field_the_module_reports() -> None:
    native = _native.build_options()
    fields = {field.name for field in dataclasses.fields(BuildOptions)}
    assert set(native) == fields


def test_positions_are_double_precision() -> None:
    assert pyjoltphysics.build_options().double_precision is True


def test_the_build_is_cross_platform_deterministic() -> None:
    assert pyjoltphysics.build_options().cross_platform_deterministic is True


def test_release_wheels_have_assertions_off() -> None:
    assert pyjoltphysics.build_options().debug is False


@pytest.mark.skipif(not X86_64, reason='the SSE4.2 floor applies to x86-64')
def test_x86_64_simd_floor_is_sse42() -> None:
    assert pyjoltphysics.build_options().simd_level == 'sse4.2'


@pytest.mark.skipif(not ARM64, reason='NEON applies to arm64')
def test_arm64_uses_neon() -> None:
    assert pyjoltphysics.build_options().simd_level == 'neon'


def test_simd_level_is_a_known_name() -> None:
    assert pyjoltphysics.build_options().simd_level in {
        'sse4.2',
        'avx',
        'avx2',
        'avx512',
        'neon',
        'none',
    }


def test_compiler_is_named() -> None:
    compiler = pyjoltphysics.build_options().compiler
    assert compiler.split()[0] in {'gcc', 'clang', 'msvc'}


def test_free_threaded_flag_matches_the_interpreter() -> None:
    assert pyjoltphysics.build_options().free_threaded is FREE_THREADED_PYTHON


@pytest.mark.skipif(not FREE_THREADED_PYTHON, reason='needs a free-threaded interpreter')
def test_import_leaves_the_gil_off() -> None:
    # A fresh interpreter, so no other extension module the test run imported
    # can have turned the GIL on, and with PYTHON_GIL unset, since PYTHON_GIL=0
    # keeps the GIL off whatever a module declares.
    probe = 'import sys, pyjoltphysics; pyjoltphysics.selftest(); print(sys._is_gil_enabled())'
    result = subprocess.run(
        [sys.executable, '-W', 'error::RuntimeWarning', '-c', probe],
        env={name: value for name, value in os.environ.items() if name != 'PYTHON_GIL'},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == 'False'


def test_selftest_steps_one_body() -> None:
    assert pyjoltphysics.selftest() is True


def test_selftest_runs_repeatedly() -> None:
    assert all(_native.selftest() for _ in range(3))


def test_every_library_nanobind_compiles_in_ships_its_licence() -> None:
    nanobind = pytest.importorskip('nanobind', reason='nanobind is in the test extra')
    tomllib = pytest.importorskip('tomllib', reason='tomllib reads pyproject.toml from Python 3.11')
    project = tomllib.loads((CHECKOUT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    shipped = set(project['license-files'])
    notice = (CHECKOUT / 'NOTICE').read_text(encoding='utf-8')
    vendored = sorted(
        p.name for p in (pathlib.Path(nanobind.__file__).parent / 'ext').iterdir() if p.is_dir()
    )
    assert vendored
    for name in ['nanobind', *(n.replace('_', '-') for n in vendored)]:
        licence = f'LICENSES/{name}-LICENSE'
        assert licence in shipped
        assert (CHECKOUT / licence).is_file()
        assert licence in notice
