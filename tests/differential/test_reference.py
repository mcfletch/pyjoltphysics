"""The recorded omi_physics answers stand in for a live run and are held to one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from differential import engines


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    answers: dict[str, object] = {'one': 1.5, 'pairs': [[0, 1], [2, 3]]}
    monkeypatch.setattr(engines, '_recorded', answers)
    monkeypatch.delenv(engines.RECORD, raising=False)
    return answers


def test_without_omi_physics_the_recorded_answer_is_given(
    recorded: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engines, 'have_omi', lambda: False)
    assert engines.reference('one', lambda: pytest.fail('nothing runs')) == 1.5


def test_a_live_answer_within_the_tolerance_passes(
    recorded: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engines, 'have_omi', lambda: True)
    assert engines.reference('one', lambda: 1.5 + 1e-5) == 1.5
    assert engines.reference('pairs', lambda: [(0, 1), (2, 3)]) == [[0, 1], [2, 3]]


@pytest.mark.parametrize(('key', 'live'), [('one', 1.6), ('pairs', [(0, 1)])])
def test_a_live_answer_that_moved_fails(
    recorded: dict[str, object], monkeypatch: pytest.MonkeyPatch, key: str, live: object
) -> None:
    monkeypatch.setattr(engines, 'have_omi', lambda: True)
    with pytest.raises(AssertionError):
        engines.reference(key, lambda: live)


def test_an_unrecorded_key_says_how_to_record_it(recorded: dict[str, object]) -> None:
    with pytest.raises(KeyError, match=engines.RECORD):
        engines.reference('missing', lambda: 0.0)


def test_recording_writes_one_answer_a_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / 'answers.json'
    monkeypatch.setattr(engines, '_recorded', {})
    monkeypatch.setenv(engines.RECORD, str(out))
    assert engines.reference('b', lambda: (1.0, 2)) == [1.0, 2]
    engines.reference('a', lambda: True)
    assert json.loads(out.read_text()) == {'a': True, 'b': [1.0, 2]}
    assert out.read_text().count('\n') == 4


def test_every_recording_is_loaded_from_the_file() -> None:
    assert 'p3/pendulum' in json.loads(engines.GOLDEN.read_text())
