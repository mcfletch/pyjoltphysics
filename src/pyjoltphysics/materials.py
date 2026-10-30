"""Materials: friction and restitution by index, with combine modes and a pair table.

>>> from pyjoltphysics import Combine, World, WorldSettings
>>> with World(WorldSettings(threads=1)) as world:
...     wood = world.materials.define(friction=0.5, restitution=0.2)
...     ice = world.materials.define(static_friction=0.05, dynamic_friction=0.02,
...                                  friction_combine=Combine.MINIMUM)
...     world.materials.pair(wood, ice, friction=0.1)
>>> (wood, ice)
(1, 2)

A body names a material by index (``BodyBatch.material``, or
:meth:`~pyjoltphysics.World.set_body_materials`); a mesh triangle may name its
own (``Shape.mesh(..., materials=)``). Index 0 is the body's own ``friction``
and ``restitution``, combined as Jolt combines them. Where either side of a
contact has another material, the world combines the two rows by the higher of
their combine modes, then a pair entry replaces any value it sets. The static
friction holds while the bodies slide past each other slower than the world's
``static_friction_speed`` at the contact, and the dynamic friction above it.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Callable

import numpy as np

from pyjoltphysics.enums import Combine

__all__ = ['Material', 'MaterialTable']


@dataclasses.dataclass(frozen=True)
class Material:
    """One row of a world's material table, with the OMI physics material's fields."""

    static_friction: float = 0.6
    dynamic_friction: float = 0.6
    restitution: float = 0.0
    friction_combine: Combine = Combine.AVERAGE
    restitution_combine: Combine = Combine.AVERAGE

    def __post_init__(self) -> None:
        for name in ('static_friction', 'dynamic_friction', 'restitution'):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError('%s is finite and not negative, not %r' % (name, value))


class MaterialTable:
    """A world's materials and pair entries; each change is sent to the world as a batch.

    Row 0 stands for a body's own coefficients and is not a :class:`Material`.
    """

    def __init__(self, send: Callable[[MaterialTable], int]) -> None:
        self._send = send
        self._lock = threading.Lock()
        self._rows: list[Material] = [Material()]
        self._pairs: dict[tuple[int, int], tuple[float, float, float]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Material:
        if index == 0:
            raise KeyError("material 0 is the body's own friction and restitution")
        return self._rows[index]

    def pairing(self, first: int, second: int) -> tuple[float, float, float] | None:
        """How two materials meet, in either order: (static friction, dynamic friction,
        restitution), NaN for a value left to the materials; None with no pair entry."""
        with self._lock:
            return self._pairs.get((first, second))

    def define(
        self,
        friction: float | None = None,
        restitution: float = 0.0,
        *,
        static_friction: float | None = None,
        dynamic_friction: float | None = None,
        friction_combine: Combine = Combine.AVERAGE,
        restitution_combine: Combine = Combine.AVERAGE,
    ) -> int:
        """Add a material and return its index.

        ``friction`` sets both coefficients; ``static_friction`` and
        ``dynamic_friction`` set one each (both default to 0.6).
        """
        default = 0.6 if friction is None else float(friction)
        material = Material(
            default if static_friction is None else float(static_friction),
            default if dynamic_friction is None else float(dynamic_friction),
            float(restitution),
            Combine(friction_combine),
            Combine(restitution_combine),
        )
        with self._lock:
            if len(self._rows) >= 0xFFFF:
                raise ValueError('a world holds at most 65535 materials')
            self._rows.append(material)
            index = len(self._rows) - 1
        self._send(self)
        return index

    def pair(
        self,
        first: int,
        second: int,
        friction: float | None = None,
        restitution: float | None = None,
        *,
        static_friction: float | None = None,
        dynamic_friction: float | None = None,
    ) -> None:
        """Set how two materials meet, in either order; None leaves a value to the materials.

        ``friction`` sets both coefficients. marble-demo's pair friction grid
        is a table of these.
        """
        with self._lock:
            for index in (first, second):
                if not 0 <= index < len(self._rows):
                    raise IndexError('material %d is not in the table' % (index,))
            nan = math.nan
            static = static_friction if static_friction is not None else friction
            dynamic = dynamic_friction if dynamic_friction is not None else friction
            entry = (
                nan if static is None else float(static),
                nan if dynamic is None else float(dynamic),
                nan if restitution is None else float(restitution),
            )
            self._pairs[(first, second)] = self._pairs[(second, first)] = entry
        self._send(self)

    def native(self) -> tuple[np.ndarray, ...]:
        """Return the table as the native module's arrays."""
        with self._lock:
            rows = list(self._rows)
            m = len(rows)
            pairs = (
                np.full((m, m, 3), np.nan, np.float32)
                if self._pairs
                else np.zeros((0, 0, 3), np.float32)
            )
            for (a, b), entry in self._pairs.items():
                pairs[a, b] = entry
        return (
            np.array([r.static_friction for r in rows], np.float32),
            np.array([r.dynamic_friction for r in rows], np.float32),
            np.array([r.restitution for r in rows], np.float32),
            np.array([int(r.friction_combine) for r in rows], np.uint8),
            np.array([int(r.restitution_combine) for r in rows], np.uint8),
            np.ascontiguousarray(pairs),
        )
