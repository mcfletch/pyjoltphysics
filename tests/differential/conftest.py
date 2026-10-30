"""The differential suite needs omi_physics, the reference engine; it is skipped without it."""

from __future__ import annotations

import pytest

pytest.importorskip('omi_physics.world', reason='omi_physics is the reference engine')
