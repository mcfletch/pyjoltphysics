"""Ragdolls: skeletons of bodies joined by swing-twist joints.

A :class:`RagdollSettings` lists the bones parents first. Each bone is a body
whose frame is its joint's frame, so the poses the snapshot carries
(:class:`RagdollRows`) are joint poses, the matrices a renderer skins with.
A bone's ``translation`` and ``rotation`` are its rest pose in its parent
joint's frame (the root's in the ragdoll's own frame), as a glTF node's are;
its shape is placed in the joint's frame by ``shape_offset`` and
``shape_rotation``. A joint's limits are a twist about ``twist_axis`` and a
swing cone about it, ``swing`` half angles towards ``plane_axis``'s normal
and along it (radians).

A ragdoll is kinematic, powered or limp (:class:`RagdollMode`). Kinematic
bones follow the pose target exactly; powered bones are turned towards it by
motors of a torque per bone (``strength``, N m), and the root is pulled to its
target by a spring of limited force; limp bones fall. The pose target is a
held pose (:meth:`Ragdoll.set_pose`) or a clip of sampled poses
(:meth:`Ragdoll.play`) the world thread evaluates each tick.

>>> from pyjoltphysics import RagdollMode, RagdollSettings, World, WorldSettings
>>> from pyjoltphysics.testing import floor
>>> with World(WorldSettings(threads=1)) as world:
...     _ = world.add_bodies(floor())
...     doll = world.add_ragdoll(RagdollSettings.humanoid(), position=(0.0, 0.5, 0.0))
...     world.run_ticks(480)
...     positions, orientations = doll.pose()
>>> bool(positions[:, 1].max() < 0.5)
True
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native
from pyjoltphysics.enums import BodyFlag, ShapeKind
from pyjoltphysics.queries import ALL_GROUPS
from pyjoltphysics.shapes import Shape

if TYPE_CHECKING:
    from pyjoltphysics.world import World

__all__ = [
    'Bone',
    'Ragdoll',
    'RagdollMode',
    'RagdollRows',
    'RagdollSettings',
    'forward_kinematics',
]

Vector = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
#: A point or quaternion: a sequence of floats or a 1-D array.
Point = Sequence[float] | npt.NDArray[np.floating[Any]]
_IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


class RagdollMode(enum.IntEnum):
    """How a ragdoll's bones move: following the pose, driven towards it, or falling."""

    KINEMATIC = 0
    POWERED = 1
    LIMP = 2


@dataclasses.dataclass
class Bone:
    """One bone: a joint of the skeleton and the body that moves with it.

    ``parent`` is the index of the parent bone, -1 for the root. ``mass`` is
    kg. ``twist`` is the (min, max) twist about ``twist_axis`` and ``swing``
    the (normal, plane) half angles of the swing cone, radians. A
    ``twist_axis`` of None is the direction to the first child joint, or the
    bone's own translation for a bone without children; a ``plane_axis`` of
    None is a direction square to it. ``strength`` (N m) is the motors'
    torque limit when powered; ``friction`` (N m) holds the joint in every
    mode.
    """

    name: str
    parent: int
    translation: Vector
    rotation: Quaternion = _IDENTITY
    shape: Shape | None = None
    shape_offset: Vector = (0.0, 0.0, 0.0)
    shape_rotation: Quaternion = _IDENTITY
    mass: float = 1.0
    twist_axis: Vector | None = None
    plane_axis: Vector | None = None
    twist: tuple[float, float] = (-0.3, 0.3)
    swing: tuple[float, float] = (0.6, 0.6)
    strength: float = 300.0
    friction: float = 0.5


@dataclasses.dataclass
class RagdollSettings:
    """A skeleton of :class:`Bone` and the settings its bodies share.

    ``self_collision`` lets bones collide with each other, except a bone and
    its parent and bones whose shapes overlap in the rest pose.
    ``motor_frequency`` (Hz) and ``motor_damping`` (ratio) are the joint
    motors' spring. ``root_force`` (N) and ``root_torque`` (N m) limit the
    spring of ``root_frequency`` and ``root_damping`` that holds a powered
    ragdoll's root bone to its target pose; 0 leaves the root free. The rest
    are the bodies' material, damping (1/s), gravity factor and collision
    group, mask, filter, material index and :class:`~pyjoltphysics.BodyFlag`.
    """

    bones: list[Bone]
    self_collision: bool = True
    motor_frequency: float = 10.0
    motor_damping: float = 1.0
    root_frequency: float = 3.0
    root_damping: float = 1.0
    root_force: float = 0.0
    root_torque: float = 0.0
    friction: float = 0.6
    restitution: float = 0.0
    linear_damping: float = 0.05
    angular_damping: float = 0.3
    gravity_factor: float = 1.0
    group: int = 1
    mask: int = ALL_GROUPS
    filter: int = 0
    material: int = 0
    flags: BodyFlag = BodyFlag.NONE

    def __post_init__(self) -> None:
        if not self.bones:
            raise ValueError('a ragdoll has at least one bone')
        for i, bone in enumerate(self.bones):
            if (i == 0) != (bone.parent == -1) or bone.parent >= i:
                raise ValueError(
                    'bone %d (%s): a parent comes before its children, and only the first bone '
                    'has none' % (i, bone.name)
                )
            if bone.shape is None:
                raise ValueError('bone %d (%s) has no shape' % (i, bone.name))
            if not (math.isfinite(bone.mass) and bone.mass > 0.0):
                raise ValueError('bone %d (%s): mass is positive' % (i, bone.name))
        names = [bone.name for bone in self.bones]
        if len(set(names)) != len(names):
            raise ValueError('bone names are unique')

    def __len__(self) -> int:
        return len(self.bones)

    def bone_index(self, name: str) -> int:
        """Return the index of the bone called ``name``."""
        for i, bone in enumerate(self.bones):
            if bone.name == name:
                return i
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        """The bone names, in bone order."""
        return [bone.name for bone in self.bones]

    @property
    def parents(self) -> npt.NDArray[np.int32]:
        """(B,) parent indices, -1 for the root."""
        return np.array([bone.parent for bone in self.bones], dtype=np.int32)

    @property
    def rest_rotations(self) -> npt.NDArray[np.float64]:
        """(B, 4) rest rotations, each in its parent joint's frame."""
        return np.array([bone.rotation for bone in self.bones], dtype=np.float64)

    @property
    def rest_translations(self) -> npt.NDArray[np.float64]:
        """(B, 3) rest translations, each in its parent joint's frame."""
        return np.array([bone.translation for bone in self.bones], dtype=np.float64)

    # -- builders ------------------------------------------------------------

    @classmethod
    def from_skeleton(
        cls,
        names: Sequence[str],
        parents: Sequence[int],
        translations: npt.ArrayLike,
        rotations: npt.ArrayLike | None = None,
        radius: float | npt.ArrayLike = 0.05,
        mass: float = 70.0,
        **settings: Any,
    ) -> RagdollSettings:
        """Return capsule bones for a skeleton, as a character rig describes one.

        ``translations`` (B, 3) and ``rotations`` (B, 4) are the rest pose in
        each parent joint's frame, parents first. Each bone's capsule runs from
        its joint to its first child's, of ``radius`` (m, one or a bone each);
        a bone without children carries one 0.9 times as long as its own
        translation, in that direction. ``mass`` (kg) is shared among the bones
        by capsule volume. ``settings`` are further :class:`RagdollSettings`
        fields.
        """
        count = len(names)
        moves = np.asarray(translations, dtype=np.float64).reshape(count, 3)
        turns = (
            np.tile(np.array(_IDENTITY), (count, 1))
            if rotations is None
            else np.asarray(rotations, dtype=np.float64).reshape(count, 4)
        )
        radii = np.broadcast_to(np.asarray(radius, dtype=np.float64), (count,))
        children: list[list[int]] = [[] for _ in range(count)]
        for i, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(i)
        shapes: list[tuple[Shape, Vector, Quaternion, Vector, float]] = []
        for i in range(count):
            if children[i]:
                c = children[i][0]
                # The child's joint, in this joint's frame.
                reach = moves[c]
            else:
                own = moves[i]
                own_length = float(np.linalg.norm(own))
                # A leaf continues its own bone's direction, in its own frame.
                direction = (
                    _rotate(_conjugate(turns[i]), own / own_length)
                    if own_length
                    else np.array([0.0, 1.0, 0.0])
                )
                reach = direction * 0.9 * own_length
            length = float(np.linalg.norm(reach))
            axis = reach / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])
            r = float(radii[i])
            if length > 2.0 * r:
                shape = Shape.capsule(r, length - 2.0 * r)
                volume = math.pi * r * r * (length - 2.0 * r) + 4.0 / 3.0 * math.pi * r**3
            else:
                shape = Shape.sphere(max(r, length / 2.0))
                volume = 4.0 / 3.0 * math.pi * max(r, length / 2.0) ** 3
            offset = _vector(axis * length / 2.0)
            spin = _between((0.0, 1.0, 0.0), axis)
            shapes.append((shape, offset, spin, _vector(axis), volume))
        total = sum(entry[4] for entry in shapes)
        bones = []
        for i in range(count):
            shape, offset, spin, axis, volume = shapes[i]
            bones.append(
                Bone(
                    name=str(names[i]),
                    parent=int(parents[i]),
                    translation=_vector(moves[i]),
                    rotation=_quaternion(turns[i]),
                    shape=shape,
                    shape_offset=offset,
                    shape_rotation=spin,
                    mass=mass * volume / total,
                    twist_axis=axis,
                )
            )
        return cls(bones=bones, **settings)

    @classmethod
    def humanoid(cls, height: float = 1.8, mass: float = 70.0, **settings: Any) -> RagdollSettings:
        """Return an 11-bone standing figure ``height`` metres tall, Y up, facing +Z.

        Pelvis (the root, at 0.55 of the height), chest, head, upper and lower
        arms hanging at the sides, thighs and shins. Joint limits suit each:
        knees and elbows bend one way, the neck and shoulders move most.
        ``settings`` are further :class:`RagdollSettings` fields.
        """
        s = height / 1.8
        names = [
            'pelvis', 'chest', 'head',
            'upper_arm_l', 'lower_arm_l', 'upper_arm_r', 'lower_arm_r',
            'thigh_l', 'shin_l', 'thigh_r', 'shin_r',
        ]  # fmt: skip
        parents = [-1, 0, 1, 1, 3, 1, 5, 0, 7, 0, 9]
        translations = (
            np.array(
                [
                    (0.0, 0.99, 0.0),
                    (0.0, 0.30, 0.0),
                    (0.0, 0.32, 0.0),
                    (0.19, 0.24, 0.0),
                    (0.0, -0.30, 0.0),
                    (-0.19, 0.24, 0.0),
                    (0.0, -0.30, 0.0),
                    (0.10, -0.06, 0.0),
                    (0.0, -0.44, 0.0),
                    (-0.10, -0.06, 0.0),
                    (0.0, -0.44, 0.0),
                ]
            )
            * s
        )
        radius = (
            np.array([0.12, 0.13, 0.10, 0.05, 0.045, 0.05, 0.045, 0.07, 0.055, 0.07, 0.055]) * s
        )
        out = cls.from_skeleton(names, parents, translations, radius=radius, mass=mass, **settings)
        limits = {
            'chest': ((-0.3, 0.3), (0.4, 0.3)),
            'head': ((-0.8, 0.8), (0.6, 0.5)),
            'upper_arm_l': ((-0.6, 0.6), (1.4, 1.2)),
            'upper_arm_r': ((-0.6, 0.6), (1.4, 1.2)),
            'lower_arm_l': ((-0.6, 0.6), (1.2, 0.05)),
            'lower_arm_r': ((-0.6, 0.6), (1.2, 0.05)),
            'thigh_l': ((-0.4, 0.4), (1.0, 0.5)),
            'thigh_r': ((-0.4, 0.4), (1.0, 0.5)),
            'shin_l': ((-0.1, 0.1), (1.2, 0.05)),
            'shin_r': ((-0.1, 0.1), (1.2, 0.05)),
        }
        strength = {'chest': 600.0, 'head': 150.0, 'thigh_l': 600.0, 'thigh_r': 600.0}
        for bone in out.bones:
            if bone.name in limits:
                bone.twist, bone.swing = limits[bone.name]
            bone.strength = strength.get(bone.name, 250.0) * mass / 70.0
            # Elbows and knees swing towards the front (+z), square to the bone.
            if bone.name.startswith(('lower_arm', 'shin')):
                bone.plane_axis = (1.0, 0.0, 0.0)
        return out

    # -- files and the native module ------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as plain data (shapes as kind and parameters)."""
        out = {
            f.name: _plain(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.name not in ('bones', 'flags')
        }
        out['flags'] = int(self.flags)
        bones = []
        for bone in self.bones:
            entry = {f.name: _plain(getattr(bone, f.name)) for f in dataclasses.fields(bone)}
            assert bone.shape is not None
            if bone.shape.kind not in (
                ShapeKind.BOX,
                ShapeKind.SPHERE,
                ShapeKind.CAPSULE,
                ShapeKind.CYLINDER,
            ):
                raise ValueError('bone %s: only primitive shapes are written as data' % bone.name)
            entry['shape'] = {'kind': int(bone.shape.kind), 'params': list(bone.shape.params)}
            bones.append(entry)
        out['bones'] = bones
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RagdollSettings:
        """Return the settings :meth:`as_dict` wrote."""
        values = dict(data)
        bones = []
        for entry in values.pop('bones'):
            fields = dict(entry)
            shape = fields.pop('shape')
            fields['shape'] = Shape(ShapeKind(shape['kind']), tuple(shape['params']))
            for name in (
                'translation',
                'rotation',
                'shape_offset',
                'shape_rotation',
                'twist',
                'swing',
            ):
                fields[name] = tuple(fields[name])
            for name in ('twist_axis', 'plane_axis'):
                if fields[name] is not None:
                    fields[name] = tuple(fields[name])
            bones.append(Bone(**fields))
        values['flags'] = BodyFlag(values['flags'])
        return cls(bones=bones, **values)

    def native(
        self,
        shape_id: Any,
        position: Point,
        rotation: Point,
        mode: RagdollMode,
    ) -> dict[str, Any]:
        """Return the keyword arguments of the native ``add_ragdoll``; ``shape_id`` builds a shape."""
        params = [
            *(float(v) for v in position),
            *(float(v) for v in rotation),
            float(int(mode)),
            1.0 if self.self_collision else 0.0,
            self.motor_frequency,
            self.motor_damping,
            self.root_frequency,
            self.root_damping,
            self.root_force,
            self.root_torque,
            self.friction,
            self.restitution,
            self.linear_damping,
            self.angular_damping,
            self.gravity_factor,
            float(self.group),
            float(self.mask),
            float(self.filter),
            float(self.material),
            float(int(self.flags)),
        ]
        rows = []
        ids = []
        children: dict[int, int] = {}
        for i, bone in enumerate(self.bones):
            if bone.parent >= 0:
                children.setdefault(bone.parent, i)
        for i, bone in enumerate(self.bones):
            twist = np.asarray(
                bone.twist_axis if bone.twist_axis is not None else self._bone_axis(i, children),
                dtype=np.float64,
            )
            twist = twist / np.linalg.norm(twist)
            if bone.plane_axis is not None:
                plane = np.asarray(bone.plane_axis, dtype=np.float64)
            else:
                plane = np.cross(twist, (0.0, 0.0, 1.0) if abs(twist[2]) < 0.9 else (1.0, 0.0, 0.0))
            rows.append(
                [
                    float(bone.parent),
                    *bone.translation,
                    *bone.rotation,
                    *bone.shape_offset,
                    *bone.shape_rotation,
                    bone.mass,
                    *twist,
                    *plane,
                    bone.twist[0],
                    bone.twist[1],
                    bone.swing[0],
                    bone.swing[1],
                    bone.strength,
                    bone.friction,
                ]
            )
            assert bone.shape is not None
            ids.append(shape_id(bone.shape))
        return {
            'params': np.array(params, dtype=np.float64),
            'bones': np.array(rows, dtype=np.float64).reshape(len(rows), -1),
            'shape_id': np.array(ids, dtype=np.uint32),
        }

    def _bone_axis(self, i: int, children: Mapping[int, int]) -> npt.NDArray[np.float64]:
        child = children.get(i)
        if child is not None:
            reach = np.asarray(self.bones[child].translation, dtype=np.float64)
        else:
            reach = _rotate(
                _conjugate(np.asarray(self.bones[i].rotation, dtype=np.float64)),
                np.asarray(self.bones[i].translation, dtype=np.float64),
            )
        length = float(np.linalg.norm(reach))
        return reach / length if length > 1e-9 else np.array([0.0, 1.0, 0.0])


def forward_kinematics(
    settings: RagdollSettings,
    rotations: npt.ArrayLike | None,
    position: Point,
    rotation: Point,
    translations: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return each joint's world position (B, 3) and orientation (B, 4) for a pose.

    ``rotations`` (B, 4) and ``translations`` (B, 3) are in each parent
    joint's frame, None for the rest pose's; ``position`` and ``rotation``
    place the ragdoll's frame in the world. This is the arithmetic the world
    thread drives a kinematic ragdoll by.
    """
    count = len(settings.bones)
    turns = (
        settings.rest_rotations
        if rotations is None
        else np.asarray(rotations, dtype=np.float64).reshape(count, 4)
    )
    moves = (
        settings.rest_translations
        if translations is None
        else np.asarray(translations, dtype=np.float64).reshape(count, 3)
    )
    out_p = np.empty((count, 3))
    out_q = np.empty((count, 4))
    base_p = np.asarray(position, dtype=np.float64)
    base_q = np.asarray(rotation, dtype=np.float64)
    base_q = base_q / np.linalg.norm(base_q)
    for i, bone in enumerate(settings.bones):
        frame_p, frame_q = (
            (base_p, base_q) if bone.parent < 0 else (out_p[bone.parent], out_q[bone.parent])
        )
        out_p[i] = frame_p + _rotate(frame_q, moves[i])
        q = _multiply(frame_q, turns[i] / np.linalg.norm(turns[i]))
        out_q[i] = q / np.linalg.norm(q)
    return out_p, out_q


class RagdollRows:
    """Every ragdoll of one snapshot, as read-only NumPy views, row ``i`` the ragdoll ``id[i]``.

    ``mode`` is a :class:`RagdollMode` code and ``strength`` the share of
    their torque the motors have, 0 to 1. A ragdoll's bones are rows
    ``first_bone[i]`` to ``first_bone[i] + bone_count[i]`` of
    ``bone_uid``, ``bone_position`` (float64, metres) and
    ``bone_orientation`` (x, y, z, w), in bone order: one slice per ragdoll
    holds every joint's world pose.
    """

    __slots__ = ('_native', 'tick')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        #: The tick of the snapshot.
        self.tick = native.tick

    def __len__(self) -> int:
        return int(self._native.ragdoll_count)

    @property
    def id(self) -> npt.NDArray[np.uint32]:
        """(n,) ragdoll ids, ascending."""
        return self._native.ragdoll_id

    @property
    def mode(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`RagdollMode` codes."""
        return self._native.ragdoll_mode

    @property
    def strength(self) -> npt.NDArray[np.float32]:
        """(n,) the motors' share of their torque, 0 to 1."""
        return self._native.ragdoll_strength

    @property
    def self_collision(self) -> npt.NDArray[np.uint8]:
        """(n,) 1 where non-neighbouring bones collide."""
        return self._native.ragdoll_self_collision

    @property
    def first_bone(self) -> npt.NDArray[np.uint32]:
        """(n,) row of each ragdoll's first bone."""
        return self._native.ragdoll_first_bone

    @property
    def bone_count(self) -> npt.NDArray[np.uint32]:
        """(n,) number of bones of each ragdoll."""
        return self._native.ragdoll_bones

    @property
    def bone_uid(self) -> npt.NDArray[np.uint32]:
        """(b,) body uid of each bone."""
        return self._native.bone_uid

    @property
    def bone_position(self) -> npt.NDArray[np.float64]:
        """(b, 3) joint positions, metres."""
        return self._native.bone_position

    @property
    def bone_orientation(self) -> npt.NDArray[np.float32]:
        """(b, 4) joint orientations ``(x, y, z, w)``."""
        return self._native.bone_orientation

    def bones(self, i: int) -> slice:
        """Return the bone rows of ragdoll row ``i``."""
        first = int(self.first_bone[i])
        return slice(first, first + int(self.bone_count[i]))

    def rows(self) -> dict[int, int]:
        """Return the row of each ragdoll id."""
        return {int(v): i for i, v in enumerate(self.id)}


class Ragdoll:
    """The handle of one ragdoll of a world: commands queued, poses read from the newest snapshot."""

    __slots__ = ('bones', 'id', 'self_collision', 'settings', 'world')

    def __init__(
        self, world: World, rid: int, bones: npt.NDArray[np.uint32], settings: RagdollSettings
    ) -> None:
        self.world = world
        #: The ragdoll id.
        self.id = int(rid)
        #: (B,) body uid of each bone, in bone order.
        self.bones = bones
        #: The settings the ragdoll was made from.
        self.settings = settings
        #: Whether non-neighbouring bones collide.
        self.self_collision = settings.self_collision

    def __repr__(self) -> str:
        return '<Ragdoll %d bones=%d>' % (self.id, len(self.bones))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ragdoll) and other.world is self.world and other.id == self.id

    def __hash__(self) -> int:
        return hash((id(self.world), self.id))

    def bone(self, name: str) -> int:
        """Return the uid of the bone called ``name``."""
        return int(self.bones[self.settings.bone_index(name)])

    def set_mode(self, mode: RagdollMode, blend: float = 0.0, at_tick: int = 0) -> int:
        """Queue a change of mode over ``blend`` seconds; see :meth:`World.set_ragdoll_modes`."""
        return self.world.set_ragdoll_modes([self.id], mode, blend, at_tick)

    def set_pose(
        self,
        rotations: npt.ArrayLike | None,
        translations: npt.ArrayLike | None = None,
        position: Point | None = None,
        rotation: Point | None = None,
        at_tick: int = 0,
    ) -> int:
        """Queue the held pose target; see :meth:`World.set_ragdoll_pose`."""
        return self.world.set_ragdoll_pose(
            self, rotations, translations, position, rotation, at_tick
        )

    def place(self, position: Point, rotation: Point = _IDENTITY, at_tick: int = 0) -> int:
        """Queue where the ragdoll's frame stands, leaving the joints' pose as it is."""
        return self.world.set_ragdoll_pose(self, None, None, position, rotation, at_tick)

    def play(
        self,
        times: npt.ArrayLike,
        rotations: npt.ArrayLike,
        translations: npt.ArrayLike | None = None,
        loop: bool = False,
        at_tick: int = 0,
    ) -> int:
        """Queue a clip as the pose target; see :meth:`World.set_ragdoll_clip`."""
        return self.world.set_ragdoll_clip(self, times, rotations, translations, loop, at_tick)

    def stop(self, at_tick: int = 0) -> int:
        """Queue clearing the clip: the held pose is the target again."""
        return self.world.set_ragdoll_clip(
            self, (), np.zeros((0, len(self.bones), 4)), None, False, at_tick
        )

    def set_strength(self, strength: npt.ArrayLike, at_tick: int = 0) -> int:
        """Queue the motors' torque limit of each bone, N m (one for all, or one a bone)."""
        return self.world.set_ragdoll_strength(self, strength, at_tick)

    def hit(self, bone: int, impulse: Sequence[float], at_tick: int = 0) -> int:
        """Queue an impulse (N s) on bone index ``bone``, at its centre of mass."""
        return self.world.add_impulses([int(self.bones[bone])], [tuple(impulse)], at_tick)

    def remove(self, at_tick: int = 0) -> int:
        """Queue removing the ragdoll and its bodies."""
        return self.world.remove_ragdolls([self.id], at_tick)

    def pose(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32]]:
        """Return the newest snapshot's joint positions (B, 3) and orientations (B, 4)."""
        snapshot = self.world.latest()
        if snapshot is None:
            raise LookupError('no snapshot has been published yet')
        rows = snapshot.ragdolls
        found = np.flatnonzero(rows.id == self.id)
        if not len(found):
            raise LookupError('ragdoll %d is not in the newest snapshot' % self.id)
        span = rows.bones(int(found[0]))
        return rows.bone_position[span], rows.bone_orientation[span]


# -- quaternions, (x, y, z, w) ---------------------------------------------------


def _multiply(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _conjugate(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _rotate(q: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    u = q[:3]
    w = q[3]
    out: npt.NDArray[np.float64] = (
        2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)
    )
    return out


def _vector(values: npt.NDArray[np.float64]) -> Vector:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _quaternion(values: npt.NDArray[np.float64]) -> Quaternion:
    x, y, z, w = (float(v) for v in values)
    return (x, y, z, w)


def _between(a: Sequence[float], b: npt.NDArray[np.float64]) -> Quaternion:
    """The shortest rotation taking unit ``a`` to unit ``b``."""
    u = np.asarray(a, dtype=np.float64)
    d = float(np.dot(u, b))
    if d < -0.999999:
        axis = np.cross(u, (1.0, 0.0, 0.0))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, (0.0, 0.0, 1.0))
        axis /= np.linalg.norm(axis)
        return (float(axis[0]), float(axis[1]), float(axis[2]), 0.0)
    c = np.cross(u, b)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, enum.Enum):
        return int(value.value)
    return value
