"""Joints: what holds two bodies together, or a body to the world.

>>> from pyjoltphysics import BodyBatch, Joint, World, WorldSettings
>>> with World(WorldSettings(threads=1)) as world:
...     uids, _ = world.add_bodies(BodyBatch.boxes([(1.0, 2.0, 0.0)], size=(0.2, 0.2, 0.2)))
...     hinge = world.add_joint(Joint.hinge(int(uids[0]), anchor=(0.0, 2.0, 0.0), axis=(0, 0, 1)))
...     world.run_ticks(30)
...     angle = float(world.latest().joints.value[0])
>>> angle < 0.0
True

A :class:`Joint` is plain data: its kind, the two bodies and, in world
coordinates as the bodies stand when the batch is applied, its anchor and axes.
Each body carries them from then on. ``to`` is the body ``body`` is held to,
0 for the world. Angles are radians and distances metres, measured from the
pose the joint was made in.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native
from pyjoltphysics.enums import JointKind, MotorMode

__all__ = ['Joint', 'JointRows', 'Motor', 'joint_arrays']

Vector = tuple[float, float, float]
_INF = math.inf
_FREE = (-_INF, _INF)


def _vector(value: Iterable[float], name: str) -> Vector:
    x, y, z = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError('%s is not finite' % (name,))
    return (x, y, z)


def _axis(value: Iterable[float], name: str = 'axis') -> Vector:
    axis = _vector(value, name)
    if math.hypot(*axis) < 1e-9:
        raise ValueError('%s has no direction' % (name,))
    return axis


def _range(value: Sequence[float] | None, name: str) -> tuple[float, float]:
    if value is None:
        return _FREE
    low, high = (float(v) for v in value)
    if not low <= high:
        raise ValueError('%s is (lower, upper) with lower <= upper, not %r' % (name, value))
    return (low, high)


@dataclasses.dataclass(frozen=True)
class Motor:
    """A joint's motor on one axis.

    ``target`` is a speed (m/s, rad/s) for ``VELOCITY`` and a position (m,
    rad) for ``POSITION``. ``max`` is the most force (N) or torque (N m) it
    uses. A position motor is a spring of ``frequency`` Hz and ``damping``
    ratio towards its target.
    """

    mode: MotorMode = MotorMode.VELOCITY
    target: float = 0.0
    max: float = _INF
    frequency: float = 2.0
    damping: float = 1.0


@dataclasses.dataclass(frozen=True)
class Joint:
    """One joint, to add with :meth:`~pyjoltphysics.World.add_joint`; the builders make each kind.

    Axis ``i`` of ``lower``, ``upper`` and ``motors`` is the one axis of a
    hinge or slider at 0; for ``SIX_DOF`` the six axes translation x, y, z
    then rotation x, y, z of the joint's frame (x along ``axis``, y along
    ``axis2`` made square to it). A free axis has limits ``(-inf, inf)``
    and a locked one ``(0, 0)``. ``limit_spring`` (frequency in Hz, damping
    ratio) makes the limits soft; ``(0, 0)`` is rigid. ``friction`` (N or
    N m) holds the free axes. A joint whose force or torque in a step passes
    ``break_force`` (N) or ``break_torque`` (N m) breaks; 0 never breaks.
    The two bodies collide with each other only with ``collide``.
    """

    kind: JointKind
    body: int
    to: int = 0
    anchor: Vector = (0.0, 0.0, 0.0)
    to_anchor: Vector = (0.0, 0.0, 0.0)
    axis: Vector = (1.0, 0.0, 0.0)
    axis2: Vector = (0.0, 1.0, 0.0)
    axis3: Vector = (0.0, 0.0, 1.0)
    lower: tuple[float, ...] = (-_INF,) * 6
    upper: tuple[float, ...] = (_INF,) * 6
    limit_spring: tuple[float, float] = (0.0, 0.0)
    friction: float = 0.0
    motors: tuple[Motor | None, ...] = (None,) * 6
    break_force: float = 0.0
    break_torque: float = 0.0
    collide: bool = False

    # -- builders ------------------------------------------------------------

    @classmethod
    def fixed(cls, body: int, to: int = 0, **values: Any) -> Joint:
        """Hold ``body`` to ``to`` at the pose they have when the joint is made."""
        return cls(JointKind.FIXED, int(body), int(to), **values)

    @classmethod
    def ball(cls, body: int, to: int = 0, *, anchor: Iterable[float], **values: Any) -> Joint:
        """Hold a point of each body together at ``anchor``; they turn freely about it."""
        return cls(JointKind.BALL, int(body), int(to), anchor=_vector(anchor, 'anchor'), **values)

    @classmethod
    def hinge(
        cls,
        body: int,
        to: int = 0,
        *,
        anchor: Iterable[float],
        axis: Iterable[float],
        limits: Sequence[float] | None = None,
        motor: Motor | None = None,
        **values: Any,
    ) -> Joint:
        """Turn ``body`` against ``to`` about ``axis`` through ``anchor``.

        ``limits`` (lower, upper) stop the angle, within -pi to pi; the angle
        starts at 0.
        """
        low, high = _range(limits, 'limits')
        return cls(
            JointKind.HINGE,
            int(body),
            int(to),
            anchor=_vector(anchor, 'anchor'),
            axis=_axis(axis),
            lower=(low,) + (-_INF,) * 5,
            upper=(high,) + (_INF,) * 5,
            motors=(motor,) + (None,) * 5,
            **values,
        )

    @classmethod
    def slider(
        cls,
        body: int,
        to: int = 0,
        *,
        axis: Iterable[float],
        anchor: Iterable[float] | None = None,
        limits: Sequence[float] | None = None,
        motor: Motor | None = None,
        **values: Any,
    ) -> Joint:
        """Slide ``body`` against ``to`` along ``axis``, neither turning against the other.

        ``limits`` (lower <= 0 <= upper, m) stop the slide; the position
        starts at 0. ``anchor`` is where the slide is measured, by default the
        body's origin (pass it for a body whose centre of mass is elsewhere).
        """
        low, high = _range(limits, 'limits')
        return cls(
            JointKind.SLIDER,
            int(body),
            int(to),
            anchor=(math.nan, math.nan, math.nan) if anchor is None else _vector(anchor, 'anchor'),
            axis=_axis(axis),
            lower=(low,) + (-_INF,) * 5,
            upper=(high,) + (_INF,) * 5,
            motors=(motor,) + (None,) * 5,
            **values,
        )

    @classmethod
    def distance(
        cls,
        body: int,
        to: int = 0,
        *,
        anchor: Iterable[float],
        to_anchor: Iterable[float],
        length: float | Sequence[float] | None = None,
        **values: Any,
    ) -> Joint:
        """Hold ``anchor`` on ``body`` and ``to_anchor`` on ``to`` at a distance.

        ``length`` is the distance, or (shortest, longest) for a range; None
        is the distance between the two points as they stand.
        """
        if length is None:
            low, high = _FREE
        elif isinstance(length, (int, float)):
            low = high = float(length)
        else:
            low, high = _range(length, 'length')
        return cls(
            JointKind.DISTANCE,
            int(body),
            int(to),
            anchor=_vector(anchor, 'anchor'),
            to_anchor=_vector(to_anchor, 'to_anchor'),
            lower=(low,) + (-_INF,) * 5,
            upper=(high,) + (_INF,) * 5,
            **values,
        )

    @classmethod
    def six_dof(
        cls,
        body: int,
        to: int = 0,
        *,
        anchor: Iterable[float],
        axis: Iterable[float] = (1.0, 0.0, 0.0),
        axis2: Iterable[float] = (0.0, 1.0, 0.0),
        lower: Sequence[float] = (0.0,) * 6,
        upper: Sequence[float] = (0.0,) * 6,
        motors: Sequence[Motor | None] = (None,) * 6,
        **values: Any,
    ) -> Joint:
        """Lock, free or limit each of six axes of a frame at ``anchor``.

        The axes are translation x, y, z then rotation x, y, z, x along
        ``axis`` and y along ``axis2``. Every axis is locked by default.
        """
        low = tuple(float(v) for v in lower)
        high = tuple(float(v) for v in upper)
        if len(low) != 6 or len(high) != 6 or len(motors) != 6:
            raise ValueError('a six-axis joint has six lower and upper limits and six motors')
        return cls(
            JointKind.SIX_DOF,
            int(body),
            int(to),
            anchor=_vector(anchor, 'anchor'),
            axis=_axis(axis),
            axis2=_axis(axis2, 'axis2'),
            lower=low,
            upper=high,
            motors=tuple(motors),
            **values,
        )

    @classmethod
    def universal(
        cls,
        body: int,
        to: int = 0,
        *,
        anchor: Iterable[float],
        axis1: Iterable[float],
        axis2: Iterable[float],
        **values: Any,
    ) -> Joint:
        """X3D's UniversalJoint: ``axis1`` on ``to`` and ``axis2`` on ``body`` keep their angle.

        The bodies turn against each other about either axis, so a spin about
        the line square to both is carried across. ``value`` and ``value2`` in
        a snapshot are the turns about ``axis1`` and ``axis2``.
        """
        first, second = np.array(_axis(axis1, 'axis1')), np.array(_axis(axis2, 'axis2'))
        across = np.cross(first, second)
        if np.linalg.norm(across) < 1e-9:
            raise ValueError('a universal joint has two axes that are not parallel')
        return cls.six_dof(
            body,
            to,
            anchor=anchor,
            axis=tuple(across),
            axis2=tuple(first),
            lower=(0.0, 0.0, 0.0, 0.0, -_INF, -_INF),
            upper=(0.0, 0.0, 0.0, 0.0, _INF, _INF),
            **values,
        )

    @classmethod
    def two_axis_hinge(
        cls,
        body: int,
        to: int = 0,
        *,
        anchor: Iterable[float],
        axis1: Iterable[float],
        axis2: Iterable[float],
        limits1: Sequence[float] | None = None,
        motor1: Motor | None = None,
        motor2: Motor | None = None,
        suspension: tuple[float, float] | None = None,
        **values: Any,
    ) -> Joint:
        """X3D's DoubleAxisHingeJoint: a steered wheel, turning about ``axis1`` (steering) and ``axis2`` (axle).

        ``limits1`` stop the steering angle; ``motor1`` and ``motor2`` drive
        the steering and the axle. ``suspension`` (frequency in Hz, damping
        ratio) lets the anchor give along ``axis1`` as a spring.
        """
        first, second = np.array(_axis(axis1, 'axis1')), np.array(_axis(axis2, 'axis2'))
        across = np.cross(first, second)
        if np.linalg.norm(across) < 1e-9:
            raise ValueError('a two-axis hinge has two axes that are not parallel')
        low1, high1 = _range(limits1, 'limits1')
        spring = (0.0, 0.0) if suspension is None else (float(suspension[0]), float(suspension[1]))
        return cls.six_dof(
            body,
            to,
            anchor=anchor,
            axis=tuple(across),
            axis2=tuple(first),
            lower=(0.0, 0.0, 0.0, 0.0, low1, -_INF),
            upper=(0.0, 0.0, 0.0, 0.0, high1, _INF),
            motors=(None, None, None, None, motor1, motor2),
            limit_spring=spring,
            **values,
        )

    @classmethod
    def motor(
        cls,
        body: int,
        to: int = 0,
        *,
        axes: Sequence[Iterable[float]],
        torques: Sequence[float],
        **values: Any,
    ) -> Joint:
        """X3D's MotorJoint: torques (N m) about up to three axes, every step.

        ``to`` carries the axes (the world, where it is 0); ``body`` is turned
        forward about each and ``to`` back. ``set_joint_motors`` changes a
        torque by its axis, 0 to 2.
        """
        if not 1 <= len(axes) <= 3 or len(torques) != len(axes):
            raise ValueError('a motor joint has one to three axes and a torque for each')
        found = [_axis(a, 'axes') for a in axes]
        defaults = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        found += defaults[len(found) :]
        motors = tuple(Motor(MotorMode.VELOCITY, float(t)) for t in torques)
        return cls(
            JointKind.MOTOR,
            int(body),
            int(to),
            axis=found[0],
            axis2=found[1],
            axis3=found[2],
            motors=motors + (None,) * (6 - len(motors)),
            **values,
        )


def joint_arrays(joints: Sequence[Joint]) -> dict[str, np.ndarray]:
    """Return the joints as the native module's arrays, one row a joint."""
    n = len(joints)
    modes = np.zeros((n, 6), np.uint8)
    targets = np.zeros((n, 6), np.float64)
    maxes = np.full((n, 6), np.inf, np.float64)
    motor_spring = np.tile(np.array([2.0, 1.0], np.float32), (n, 1))
    for i, joint in enumerate(joints):
        for k, motor in enumerate(joint.motors):
            if motor is not None:
                modes[i, k] = int(motor.mode)
                targets[i, k] = motor.target
                maxes[i, k] = motor.max
                motor_spring[i] = (motor.frequency, motor.damping)
    return {
        'kind': np.array([int(j.kind) for j in joints], np.uint8),
        'a': np.array([j.to for j in joints], np.uint32),
        'b': np.array([j.body for j in joints], np.uint32),
        'anchor': np.array([j.anchor for j in joints], np.float64).reshape(n, 3),
        'to_anchor': np.array([j.to_anchor for j in joints], np.float64).reshape(n, 3),
        'axis': np.array([j.axis for j in joints], np.float64).reshape(n, 3),
        'axis2': np.array([j.axis2 for j in joints], np.float64).reshape(n, 3),
        'axis3': np.array([j.axis3 for j in joints], np.float64).reshape(n, 3),
        'lower': np.array([j.lower for j in joints], np.float64).reshape(n, 6),
        'upper': np.array([j.upper for j in joints], np.float64).reshape(n, 6),
        'spring': np.array([j.limit_spring for j in joints], np.float32).reshape(n, 2),
        'friction': np.array([j.friction for j in joints], np.float32),
        'motor_mode': modes,
        'motor_target': targets,
        'motor_max': maxes,
        'motor_spring': motor_spring,
        'break_force': np.array([j.break_force for j in joints], np.float32),
        'break_torque': np.array([j.break_torque for j in joints], np.float32),
        'collide': np.array([j.collide for j in joints], np.uint8),
    }


class JointRows:
    """Every joint of one snapshot, as read-only NumPy views, row ``i`` the joint ``id[i]``.

    ``kind`` holds :class:`~pyjoltphysics.JointKind` codes and ``state``
    :class:`~pyjoltphysics.JointState` codes. ``value`` is a hinge's angle
    (rad), a slider's position (m) or a distance joint's length (m); for a
    six-axis joint (universal, two-axis hinge) ``value`` and ``value2`` are
    its turns about its frame's y and z axes. ``force`` (N) and ``torque``
    (N m) are what the joint carried in the last step.
    """

    __slots__ = ('_native', '_rows')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        self._rows: dict[int, int] | None = None

    def __len__(self) -> int:
        return int(self._native.joint_count)

    @property
    def count(self) -> int:
        """The number of joints, and of rows."""
        return int(self._native.joint_count)

    @property
    def id(self) -> npt.NDArray[np.uint32]:
        """(n,) joint ids, ascending."""
        return self._native.joint_id

    @property
    def kind(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`~pyjoltphysics.JointKind` codes."""
        return self._native.joint_kind

    @property
    def state(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`~pyjoltphysics.JointState` codes."""
        return self._native.joint_state

    @property
    def value(self) -> npt.NDArray[np.float32]:
        """(n,) angle (rad), position (m) or length (m)."""
        return self._native.joint_value

    @property
    def value2(self) -> npt.NDArray[np.float32]:
        """(n,) a six-axis joint's turn about its frame's z axis (rad); 0 for other kinds."""
        return self._native.joint_value2

    @property
    def force(self) -> npt.NDArray[np.float32]:
        """(n,) N carried in the last step."""
        return self._native.joint_force

    @property
    def torque(self) -> npt.NDArray[np.float32]:
        """(n,) N m carried in the last step."""
        return self._native.joint_torque

    @property
    def rows(self) -> dict[int, int]:
        """Joint id to row."""
        if self._rows is None:
            self._rows = {int(j): row for row, j in enumerate(self._native.joint_id.tolist())}
        return self._rows
