"""The committed ``_native.pyi`` is the stub nanobind generates from the built module."""

from __future__ import annotations

import pathlib

import pytest

from pyjoltphysics import _native

stubgen = pytest.importorskip('nanobind.stubgen', reason='nanobind is in the test extra')

CHECKOUT_STUB = (
    pathlib.Path(__file__).resolve().parent.parent / 'src' / 'pyjoltphysics' / '_native.pyi'
)
PATTERNS = pathlib.Path(__file__).resolve().parent.parent / 'tools' / 'stub-patterns.txt'
INSTALLED_STUB = pathlib.Path(_native.__file__).with_name('_native.pyi')
# A wheel carries the stub beside the module; an editable install leaves it in
# the checkout.
STUB = INSTALLED_STUB if INSTALLED_STUB.exists() else CHECKOUT_STUB


def generated_stub() -> str:
    """Return the stub text nanobind writes for the imported module."""
    generator = stubgen.StubGen(
        _native,
        quiet=True,
        include_docstrings=True,
        patterns=stubgen.load_pattern_file(str(PATTERNS)),
    )
    generator.put(_native)
    return str(generator.get())


def test_committed_stub_matches_the_module() -> None:
    assert STUB.read_text(encoding='utf-8') == generated_stub()


def test_stub_declares_every_public_function() -> None:
    stub = STUB.read_text(encoding='utf-8')
    for name in ('jolt_version', 'build_options', 'selftest', 'add_bodies', 'run_ticks', 'latest'):
        assert 'def %s(' % name in stub
    for name in ('World', 'Snapshot', 'CallbackFailed', 'WorldStopped'):
        assert 'class %s' % name in stub
