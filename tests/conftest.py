"""Fixtures for the world suites: worlds that are stopped after each test."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pyjoltphysics import World, WorldSettings
from pyjoltphysics.testing import MakeWorld


@pytest.fixture
def make_world() -> Iterator[MakeWorld]:
    """Return a factory of worlds, ``make_world(**settings)``; each is stopped afterwards."""
    made: list[World] = []

    def make(**settings: Any) -> World:
        world = World(WorldSettings(**settings))
        made.append(world)
        return world

    yield make
    for world in made:
        world.stop()
