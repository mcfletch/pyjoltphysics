"""The recorded omi_physics answers stand in for a live run and are held to one."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


SLOPE = math.radians(20.0)


def _slides(friction: float) -> Callable[..., bool]:
    """A slide over 1 s of a 20 degree slope within 10% (or 3 cm) of g (sin - mu cos) t2 / 2."""
    accel = max(0.0, engines.G * (math.sin(SLOPE) - friction * math.cos(SLOPE)))
    return lambda along, moving: abs(along - (0.5 * accel + moving)) <= max(0.03, 0.1 * along)


def _rests_at(height: float) -> Callable[[float], bool]:
    return lambda y: abs(y - height) <= 0.01


#: What each recorded omi_physics answer has to satisfy, by key. The scenario
#: tests compare Jolt with these answers, and where omi_physics is absent
#: nothing else checks them: a re-recorded answer that breaks one fails here.
PHYSICAL: dict[str, Callable[..., bool]] = {
    'smoke/stack': lambda drift, top: drift < 0.05,
    'stack': lambda drift, top: drift < 0.02 and abs(top - 4.5) <= 0.02,
    'pyramid': lambda worst: worst < 0.05,
    'stairs': lambda x, y, z: 3.0 < x < 12.0 and abs(y - 0.2) <= 0.01,
    'vehicles/settle': lambda height, speed, wheels: speed < 0.01 and wheels == 4,
    'vehicles/brake': lambda per_speed, speed: speed < 0.05,
    'vehicles/wall': lambda blow, closing, speed, up: (
        abs(blow - closing) <= 0.05 * closing and speed < 1.0 and up > 0.99
    ),
    **{
        f'vehicles/corner/{corner}': lambda yaw, g, up: up > 0.99
        for corner in ('1/0.5', '2/0.3', '2/1')
    },
    **{
        f'slide/{surface}/{friction}': _slides(friction)
        for surface in ('mesh', 'heightfield')
        for friction in (0.0, 0.3, 0.8)
    },
    **{
        f'rest/{shape}/{surface}': _rests_at(height)
        for shape, height in (
            ('sphere', 0.5),
            ('box', 0.3),
            ('capsule', 0.3),
            ('cylinder', 0.5),
            ('hull', 0.25),
        )
        for surface in ('box', 'mesh', 'heightfield')
    },
}


@pytest.mark.parametrize('key', sorted(PHYSICAL))
def test_each_recorded_answer_is_physical(key: str) -> None:
    answer: Any = json.loads(engines.GOLDEN.read_text())[key]
    values = answer if isinstance(answer, list) else [answer]
    assert PHYSICAL[key](*values), (key, answer)
