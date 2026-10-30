"""Characters: capsules moved by input, stepped on the world thread every tick.

A character is omi_physics' controller over Jolt's ``CharacterVirtual``: walk,
run, sprint and crouch speeds, a jump whose apex is ``jumpHeight`` exactly
under a constant gravity, stairs, slopes, slides, knock-downs, landings,
pushing light bodies and riding moving platforms. :meth:`World.add_character
<pyjoltphysics.World.add_character>` returns a :class:`Character`, whose
methods queue inputs applied at the start of the next tick and whose
properties read the newest snapshot.

>>> from pyjoltphysics import CharacterCapabilities, CharacterState
>>> CharacterCapabilities(walkSpeed=2.0).walkSpeed
2.0
>>> CharacterState.AIRBORNE.name
'AIRBORNE'
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native

if TYPE_CHECKING:
    from pyjoltphysics.world import World

__all__ = [
    'Character',
    'CharacterCapabilities',
    'CharacterOp',
    'CharacterRows',
    'CharacterState',
    'MoveMode',
]


class CharacterState(enum.IntEnum):
    """What a character is doing: the first that holds, in the order of the codes below."""

    STANDING = 0
    CROUCHING = 1
    SLIDING = 2
    AIRBORNE = 3
    DOWN = 4
    SWIMMING = 5
    FLYING = 6

    @classmethod
    def ordered(cls) -> tuple[CharacterState, ...]:
        """Return the states in the order they are decided: flying first, standing last."""
        return (
            cls.FLYING,
            cls.SWIMMING,
            cls.DOWN,
            cls.AIRBORNE,
            cls.SLIDING,
            cls.CROUCHING,
            cls.STANDING,
        )


class MoveMode(enum.IntEnum):
    """The speed a move goes at: ``walkSpeed``, ``runSpeed``, ``sprintSpeed`` or ``crouchSpeed``."""

    WALK = 0
    RUN = 1
    SPRINT = 2
    CROUCH = 3


class CharacterOp(enum.IntEnum):
    """The native input codes; :class:`Character`'s methods send them."""

    MOVE = 0
    JUMP = 1
    SLIDE = 2
    EXTEND_SLIDE = 3
    END_SLIDE = 4
    KNOCK_DOWN = 5
    GET_UP = 6
    KNOCKBACK = 7
    IMPULSE = 8
    CROUCH = 9
    FLY = 10
    FLY_MOVE = 11
    SWIM = 12
    TELEPORT = 13
    GRAVITY = 14
    PASS_GROUP = 15


#: The float capabilities in the order the native module reads them.
_FLOATS = (
    'walkSpeed',
    'runSpeed',
    'sprintSpeed',
    'crouchSpeed',
    'flySpeed',
    'swimSpeed',
    'swimDrag',
    'jumpHeight',
    'stepHeight',
    'maxSlope',
    'eyeHeight',
    'standHeight',
    'crouchHeight',
    'radius',
    'airControl',
    'pushFriction',
    'terminalVelocity',
    'coyoteTime',
    'jumpBuffer',
    'mass',
    'pushMass',
    'pushSpeed',
    'pushForce',
    'acceleration',
)
#: The bool capabilities, bit by bit.
_BITS = (
    'canSprint',
    'canCrouch',
    'canJump',
    'canFly',
    'jumpFollowsSlope',
    'suppressOverVoid',
    'stickToFloor',
    'smoothEdges',
)


@dataclasses.dataclass(frozen=True)
class CharacterCapabilities:
    """How a character moves: omi_physics' ``CharacterCapabilities`` and what Jolt adds.

    Units are metres, seconds, kilograms and degrees. The OMI field names are
    kept, since they are the glTF vocabulary. Heights are the whole capsule,
    feet to crown; ``radius`` is the capsule's.

    ``jumpHeight`` sets the launch speed ``sqrt(2 g jumpHeight)``, so the apex
    is ``jumpHeight`` above the launch at any tick rate. ``jumpFollowsSlope``
    adds the vertical speed of the run along the ground to the launch.
    ``airControl`` scales the input off the ground. ``pushFriction`` (1/s)
    bleeds carried speed (a knockback, a slide, a jump pad) while grounded.
    ``terminalVelocity`` caps a fall (0 for none). ``coyoteTime`` and
    ``jumpBuffer`` are seconds a jump is forgiven after walking off an edge
    and remembered before landing. ``mass`` divides a knockback and presses
    on what the character stands on. A dynamic body no heavier than
    ``pushMass`` is pushed by walking into it, with at most ``pushForce``
    newtons and at most ``pushSpeed`` (None: ``walkSpeed``); a heavier one
    blocks. ``acceleration`` (m/s2) brings the input speed up and down; inf
    reaches it at once, as omi_physics does. ``stickToFloor`` keeps a walker
    on the ground over drops up to ``stepHeight``. ``smoothEdges`` is Jolt's
    enhanced internal edge removal for the capsule.
    """

    walkSpeed: float = 3.0
    runSpeed: float = 6.0
    sprintSpeed: float = 9.0
    crouchSpeed: float = 1.5
    canSprint: bool = True
    canCrouch: bool = True
    canJump: bool = True
    canFly: bool = True
    flySpeed: float = 8.0
    swimSpeed: float = 2.0
    swimDrag: float = 4.0
    jumpHeight: float = 1.2
    jumpFollowsSlope: bool = False
    stepHeight: float = 0.35
    maxSlope: float = 50.0
    eyeHeight: float = 1.6
    standHeight: float = 1.8
    crouchHeight: float = 1.0
    radius: float = 0.3
    airControl: float = 0.4
    pushFriction: float = 6.0
    terminalVelocity: float = 55.0
    suppressOverVoid: bool = False
    coyoteTime: float = 0.12
    jumpBuffer: float = 0.12
    mass: float = 80.0
    pushMass: float = 200.0
    pushSpeed: float | None = None
    pushForce: float = 1000.0
    acceleration: float = math.inf
    stickToFloor: bool = True
    smoothEdges: bool = True

    def native(self) -> tuple[npt.NDArray[np.float64], int]:
        """Return the float capabilities in the native order and the bool ones as bits."""
        values = [
            math.nan
            if name == 'pushSpeed' and self.pushSpeed is None
            else float(getattr(self, name))
            for name in _FLOATS
        ]
        bits = sum(1 << k for k, name in enumerate(_BITS) if getattr(self, name))
        return np.array(values, dtype=np.float64), bits


class CharacterRows:
    """Every character of one snapshot, as read-only NumPy views, row ``i`` the character ``id[i]``.

    ``position`` (n, 3) float64 is the capsule's centre; ``velocity`` (n, 3)
    float32 is the distance moved over the last tick, per second. ``state``
    holds :class:`CharacterState` codes; ``grounded`` is true on walkable
    ground. ``ground_uid`` is the body underfoot (0 for none) and
    ``ground_normal`` its unit normal. ``slide_progress`` runs 0 to 1 through
    a slide. ``landing_speed`` (m/s) is the downward speed of the last
    landing, at ``landed_tick`` (0 before the first). ``height`` is the
    capsule's height. ``previous`` is the capsule's centre at the tick before,
    for drawing between the two. ``eye`` is the eye position: ``eyeHeight``
    above the feet, scaled by the capsule's height over ``standHeight``.
    ``ground_material`` is the material index underfoot (a mesh triangle's own
    where it has one). ``stuck`` is true where the last teleport left the
    capsule inside geometry.
    """

    __slots__ = ('_native', '_rows', 'tick')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        self._rows: dict[int, int] | None = None
        #: The tick of the snapshot.
        self.tick = native.tick

    def __len__(self) -> int:
        return int(self._native.character_count)

    @property
    def id(self) -> npt.NDArray[np.uint32]:
        """(n,) character ids, ascending."""
        return self._native.character_id

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """(n, 3) capsule centres, metres."""
        return self._native.character_position

    @property
    def velocity(self) -> npt.NDArray[np.float32]:
        """(n, 3) m/s over the last tick."""
        return self._native.character_velocity

    @property
    def state(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`CharacterState` codes."""
        return self._native.character_state

    @property
    def grounded(self) -> npt.NDArray[np.bool_]:
        """(n,) true on walkable ground."""
        return self._native.character_grounded.view(np.bool_)

    @property
    def ground_uid(self) -> npt.NDArray[np.uint32]:
        """(n,) uid of the body underfoot; 0 for none."""
        return self._native.character_ground

    @property
    def ground_normal(self) -> npt.NDArray[np.float32]:
        """(n, 3) unit normal of the ground, or of the steep slope a character slides on; 0 in the air."""
        return self._native.character_ground_normal

    @property
    def slide_progress(self) -> npt.NDArray[np.float32]:
        """(n,) 0 to 1 through a slide; 0 when not sliding."""
        return self._native.character_slide

    @property
    def landing_speed(self) -> npt.NDArray[np.float32]:
        """(n,) m/s of the last landing."""
        return self._native.character_landing

    @property
    def landed_tick(self) -> npt.NDArray[np.uint32]:
        """(n,) tick of the last landing; 0 before the first."""
        return self._native.character_landed_tick

    @property
    def height(self) -> npt.NDArray[np.float32]:
        """(n,) capsule height, metres."""
        return self._native.character_height

    @property
    def previous(self) -> npt.NDArray[np.float64]:
        """(n, 3) capsule centres at the tick before, metres."""
        return self._native.character_previous

    @property
    def eye(self) -> npt.NDArray[np.float64]:
        """(n, 3) eye positions, metres."""
        return self._native.character_eye

    @property
    def ground_material(self) -> npt.NDArray[np.uint16]:
        """(n,) material index underfoot; 0 in the air or on a body of no material."""
        return self._native.character_ground_material

    @property
    def stuck(self) -> npt.NDArray[np.bool_]:
        """(n,) true where the last teleport left the capsule inside geometry."""
        return (self._native.character_flags & 1).astype(np.bool_)

    @property
    def rows(self) -> dict[int, int]:
        """Character id to row."""
        if self._rows is None:
            self._rows = {int(cid): row for row, cid in enumerate(self.id.tolist())}
        return self._rows


def _mode(mode: str | MoveMode) -> MoveMode:
    if isinstance(mode, MoveMode):
        return mode
    try:
        return MoveMode[mode.upper()]
    except KeyError:
        raise ValueError('mode is walk, run, sprint or crouch, not %r' % (mode,)) from None


class Character:
    """One character of a world: inputs queued for the next tick, state read from the newest snapshot.

    Every input method returns the batch's sequence number and takes
    ``at_tick``, as the world's commands do. The properties read
    :meth:`World.latest <pyjoltphysics.World.latest>`; a character not yet
    applied, or removed, raises ``LookupError``.
    """

    __slots__ = ('_world', 'id')

    def __init__(self, world: World, cid: int) -> None:
        self._world = world
        #: The character's id: a uid from the bodies' sequence, and its proxy body's uid.
        self.id = int(cid)

    def __repr__(self) -> str:
        return '<Character %d>' % (self.id,)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Character) and other._world is self._world and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    # -- inputs ----------------------------------------------------------------

    def _op(
        self, op: CharacterOp, code: int = 0, values: Sequence[float] = (), at_tick: int = 0
    ) -> int:
        row = np.zeros((1, 4), np.float64)
        row[0, : len(values)] = values
        return self._world.character_ops(
            np.array([self.id], np.uint32),
            np.array([int(op)], np.uint8),
            np.array([code], np.uint8),
            row,
            at_tick=at_tick,
        )

    def set_move(
        self,
        direction: Sequence[float],
        mode: str | MoveMode = MoveMode.WALK,
        speed: float | None = None,
        at_tick: int = 0,
    ) -> int:
        """Move along ``direction`` flattened to the ground, at the mode's speed or ``speed`` m/s.

        A zero direction stands still; a crouch goes no faster than ``crouchSpeed``.
        """
        x, y, z = (float(v) for v in direction)
        pace = math.nan if speed is None else float(speed)
        return self._op(CharacterOp.MOVE, _mode(mode), (x, y, z, pace), at_tick)

    def jump(self, at_tick: int = 0) -> int:
        """Jump if grounded, or within ``coyoteTime`` of walking off; else hold it for ``jumpBuffer``."""
        return self._op(CharacterOp.JUMP, at_tick=at_tick)

    def slide(
        self, duration: float, rise: float = 0.3, steer: bool = False, at_tick: int = 0
    ) -> int:
        """Slide for ``duration`` seconds, from the ground: low, keeping its speed, rising over the last ``rise``.

        With ``steer`` the input goes on moving it at the mode's speed instead.
        """
        return self._op(CharacterOp.SLIDE, int(steer), (float(duration), float(rise)), at_tick)

    def extend_slide(self, duration: float, at_tick: int = 0) -> int:
        """Have a slide last ``duration`` seconds more, from then."""
        return self._op(CharacterOp.EXTEND_SLIDE, 0, (float(duration),), at_tick)

    def end_slide(self, at_tick: int = 0) -> int:
        """End a slide and stand, or crouch where there is no room."""
        return self._op(CharacterOp.END_SLIDE, at_tick=at_tick)

    def knock_down(self, at_tick: int = 0) -> int:
        """Lie low: input ignored, carried speed dropped; gravity still acts."""
        return self._op(CharacterOp.KNOCK_DOWN, at_tick=at_tick)

    def get_up(self, start_speed: float | None = None, at_tick: int = 0) -> int:
        """Stand again (crouched under a low ceiling), carrying ``start_speed`` m/s along the last heading."""
        return self._op(CharacterOp.GET_UP, 0, (float(start_speed or 0.0),), at_tick)

    def knockback(self, impulse: Sequence[float], at_tick: int = 0) -> int:
        """Change the velocity by ``impulse`` (N s) over ``mass``; an upward part leaves the ground."""
        return self._op(CharacterOp.KNOCKBACK, 0, tuple(float(v) for v in impulse), at_tick)

    def apply_impulse(self, velocity: Sequence[float], at_tick: int = 0) -> int:
        """Launch at ``velocity`` (m/s), replacing the motion: a jump pad or a blast."""
        return self._op(CharacterOp.IMPULSE, 0, tuple(float(v) for v in velocity), at_tick)

    def set_crouch(self, crouch: bool, at_tick: int = 0) -> int:
        """Crouch, or stand where there is room."""
        return self._op(CharacterOp.CROUCH, int(crouch), at_tick=at_tick)

    def set_fly(self, flying: bool, at_tick: int = 0) -> int:
        """Fly (no collision, no gravity) or walk again."""
        return self._op(CharacterOp.FLY, int(flying), at_tick=at_tick)

    def set_fly_move(self, direction: Sequence[float], at_tick: int = 0) -> int:
        """Fly along ``direction`` in three dimensions, at ``flySpeed``."""
        return self._op(CharacterOp.FLY_MOVE, 0, tuple(float(v) for v in direction), at_tick)

    def set_swim(self, swimming: bool, buoyancy: float = 0.9, at_tick: int = 0) -> int:
        """Swim: collide as walking, move as flying, pulled by the gravity ``buoyancy`` leaves."""
        return self._op(CharacterOp.SWIM, int(swimming), (float(buoyancy),), at_tick)

    def teleport(self, position: Sequence[float], search: float = 4.0, at_tick: int = 0) -> int:
        """Put the capsule's centre at ``position`` at rest, seated on a floor within ``search`` m below."""
        x, y, z = (float(v) for v in position)
        return self._op(CharacterOp.TELEPORT, 0, (x, y, z, float(search)), at_tick)

    def set_gravity(self, gravity: float, at_tick: int = 0) -> int:
        """Set this character's gravity, m/s2 downward."""
        return self._op(CharacterOp.GRAVITY, 0, (float(gravity),), at_tick)

    def set_pass_group(self, group: int, at_tick: int = 0) -> int:
        """Set the pass group: characters of one group (other than 0) do not block each other."""
        return self._op(CharacterOp.PASS_GROUP, 0, (float(group),), at_tick)

    def set_ignore(self, uids: Iterable[int], at_tick: int = 0) -> int:
        """Collide with nothing in ``uids``, in place of any earlier set."""
        return self._world.set_character_ignore(self.id, uids, at_tick=at_tick)

    def remove(self, at_tick: int = 0) -> int:
        """Remove the character and its proxy body."""
        return self._world.remove_characters([self.id], at_tick=at_tick)

    # -- reading the newest snapshot -------------------------------------------

    def _row(self) -> tuple[CharacterRows, int]:
        snapshot = self._world.latest()
        if snapshot is None:
            raise LookupError('the world has run no tick')
        rows = snapshot.characters
        row = rows.rows.get(self.id)
        if row is None:
            raise LookupError('character %d is not in the newest snapshot' % (self.id,))
        return rows, row

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """(3,) the capsule's centre, metres."""
        rows, row = self._row()
        return np.array(rows.position[row])

    @property
    def feet(self) -> npt.NDArray[np.float64]:
        """(3,) the bottom of the capsule, metres."""
        rows, row = self._row()
        return np.array(rows.position[row]) - (0.0, 0.5 * float(rows.height[row]), 0.0)

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        """(3,) m/s over the last tick."""
        rows, row = self._row()
        return np.array(rows.velocity[row], dtype=np.float64)

    @property
    def state(self) -> CharacterState:
        """What the character is doing."""
        rows, row = self._row()
        return CharacterState(int(rows.state[row]))

    @property
    def grounded(self) -> bool:
        """True on walkable ground."""
        rows, row = self._row()
        return bool(rows.grounded[row])

    @property
    def ground_uid(self) -> int:
        """The uid of the body underfoot; 0 for none."""
        rows, row = self._row()
        return int(rows.ground_uid[row])

    @property
    def ground_normal(self) -> npt.NDArray[np.float64]:
        """(3,) the unit normal underfoot; 0 in the air."""
        rows, row = self._row()
        return np.array(rows.ground_normal[row], dtype=np.float64)

    @property
    def slide_progress(self) -> float:
        """0 to 1 through a slide; 0 when not sliding."""
        rows, row = self._row()
        return float(rows.slide_progress[row])

    @property
    def landing(self) -> float | None:
        """The speed (m/s) the character landed at in the newest snapshot's tick, else None."""
        rows, row = self._row()
        tick = int(rows.landed_tick[row])
        return float(rows.landing_speed[row]) if tick and tick == rows.tick else None

    @property
    def height(self) -> float:
        """The capsule's height, metres."""
        rows, row = self._row()
        return float(rows.height[row])
