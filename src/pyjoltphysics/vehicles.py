"""Vehicles: Jolt's vehicle constraint over a dynamic chassis body.

A vehicle is a dynamic body (the chassis) with wheels cast against the world
every tick, suspension springs, tyre friction curves, an engine, a gearbox and
differentials, solved with the rest of the world on its thread.
:meth:`World.add_vehicle <pyjoltphysics.World.add_vehicle>` takes the chassis
uid and a :class:`VehicleSettings` and returns a :class:`Vehicle`, whose
:meth:`~Vehicle.set_input` queues the driver's throttle, steer, brake and
handbrake (held until the next input) and whose properties read the newest
snapshot. :attr:`Snapshot.vehicles <pyjoltphysics.Snapshot.vehicles>` holds
every vehicle and wheel of a tick as arrays.

Three kinds share the settings: :attr:`VehicleKind.WHEELED` (any number of
wheels, differentials between them), :attr:`VehicleKind.MOTORCYCLE` (two
wheels and a lean controller) and :attr:`VehicleKind.TRACKED` (two tracks,
steered by running them at different speeds).

Settings are plain data in SI units, with nothing of any engine or file
format in them: :meth:`VehicleSettings.as_dict` and
:meth:`VehicleSettings.from_dict` turn them to and from builtin types for a
scene file or a service, and :meth:`VehicleSettings.from_raycast` maps the
tuning of omi_physics' ``RaycastVehicle`` onto them.

>>> from pyjoltphysics import VehicleSettings, WheelSettings
>>> settings = VehicleSettings(wheels=(WheelSettings((-0.8, 0.0, -1.3), steered=True),
...                                    WheelSettings((0.8, 0.0, -1.3), steered=True),
...                                    WheelSettings((-0.8, 0.0, 1.3)),
...                                    WheelSettings((0.8, 0.0, 1.3))))
>>> [(d.left, d.right) for d in settings.resolved_differentials()]
[(0, 1), (2, 3)]
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyjoltphysics import _native

if TYPE_CHECKING:
    from pyjoltphysics.world import World

__all__ = [
    'AntiRollBar',
    'Differential',
    'Engine',
    'Lean',
    'Transmission',
    'Vehicle',
    'VehicleKind',
    'VehicleRows',
    'VehicleSettings',
    'VehicleTrack',
    'WheelCast',
    'WheelRows',
    'WheelSettings',
]

Vector = tuple[float, float, float]
Curve = tuple[tuple[float, float], ...]


class VehicleKind(enum.IntEnum):
    """The controller a vehicle drives with."""

    WHEELED = 0
    MOTORCYCLE = 1
    TRACKED = 2


class WheelCast(enum.IntEnum):
    """How a wheel finds the ground: a ray, a swept sphere or a swept cylinder."""

    RAY = 0
    SPHERE = 1
    CYLINDER = 2


@dataclasses.dataclass(frozen=True)
class WheelSettings:
    """One wheel, in the chassis body's own frame.

    ``position`` is where the suspension is attached; the wheel's centre hangs
    between ``suspension_min`` and ``suspension_max`` metres below it along
    the vehicle's down axis, and the tyre meets the ground ``radius`` below
    that. The spring's natural length is ``suspension_max + preload``.
    ``stiffness`` (N/m) and ``damping`` (N s/m) are the spring's. A
    ``steered`` wheel turns to :attr:`VehicleSettings.max_steer`; a ``driven``
    wheel takes engine torque through the differentials built when
    :attr:`VehicleSettings.differentials` is None. ``brake_torque`` and
    ``handbrake_torque`` (N m) are the most each pedal holds the wheel with.
    ``inertia`` (kg m2) and ``angular_damping`` (1/s) are the wheel's own.
    ``grip`` multiplies the vehicle's friction curves at this wheel. ``track``
    is 0 (left) or 1 (right) on a tracked vehicle, -1 elsewhere.
    """

    position: Vector
    radius: float = 0.3
    width: float = 0.2
    suspension_min: float = 0.0
    suspension_max: float = 0.3
    preload: float = 0.0
    stiffness: float = 30000.0
    damping: float = 3000.0
    steered: bool = False
    driven: bool = True
    brake_torque: float = 1500.0
    handbrake_torque: float = 0.0
    inertia: float = 0.9
    angular_damping: float = 0.2
    grip: float = 1.0
    track: int = -1

    def native(self) -> list[float]:
        """Return the wheel as the native module reads it."""
        return [
            *(float(v) for v in self.position),
            self.radius,
            self.width,
            self.suspension_min,
            self.suspension_max,
            self.preload,
            self.stiffness,
            self.damping,
            1.0 if self.steered else 0.0,
            self.brake_torque,
            self.handbrake_torque,
            self.inertia,
            self.angular_damping,
            self.grip,
            float(self.track),
        ]


@dataclasses.dataclass(frozen=True)
class Engine:
    """The engine: ``max_torque`` (N m) scaled by ``torque_curve`` between ``min_rpm`` and ``max_rpm``.

    ``torque_curve`` is (x, fraction of ``max_torque``) points, x being the
    engine speed over ``max_rpm``. The engine gives no torque above
    ``max_rpm``. ``inertia`` (kg m2) and ``damping`` (1/s) are its own.
    """

    max_torque: float = 500.0
    min_rpm: float = 1000.0
    max_rpm: float = 6000.0
    torque_curve: Curve = ((0.0, 0.8), (0.66, 1.0), (1.0, 0.8))
    inertia: float = 0.5
    damping: float = 0.2


@dataclasses.dataclass(frozen=True)
class Transmission:
    """The gearbox: ratios of engine speed to gearbox output speed, and when an automatic shifts.

    ``automatic`` shifts up past ``shift_up_rpm`` and down below
    ``shift_down_rpm``, taking ``switch_time`` seconds with no drive and
    ``clutch_release_time`` to engage, then waiting ``switch_latency`` before
    the next shift. ``clutch_strength`` (kg m2/s) is the torque the engaged
    clutch passes per rad/s of slip. A negative throttle selects reverse.
    """

    automatic: bool = True
    gears: tuple[float, ...] = (2.66, 1.78, 1.3, 1.0, 0.74)
    reverse_gears: tuple[float, ...] = (-2.90,)
    switch_time: float = 0.5
    clutch_release_time: float = 0.3
    switch_latency: float = 0.5
    shift_up_rpm: float = 4000.0
    shift_down_rpm: float = 2000.0
    clutch_strength: float = 10.0


@dataclasses.dataclass(frozen=True)
class Differential:
    """Engine torque to a pair of wheels; -1 names no wheel on that side.

    ``ratio`` is gearbox speed over wheel speed. ``split`` is the share of the
    torque to the right wheel. Past ``limited_slip`` (fastest over slowest
    wheel speed; inf for an open differential) the torque goes to the slower
    wheel. ``torque_ratio`` is this differential's share of the engine; the
    shares of a vehicle add to 1.
    """

    left: int
    right: int
    ratio: float = 3.42
    split: float = 0.5
    limited_slip: float = 1.4
    torque_ratio: float = 1.0


@dataclasses.dataclass(frozen=True)
class AntiRollBar:
    """A bar between two wheels pushing their suspension lengths together with ``stiffness`` N/m."""

    left: int
    right: int
    stiffness: float = 1000.0


@dataclasses.dataclass(frozen=True)
class VehicleTrack:
    """One track of a tracked vehicle; its wheels are the ones whose ``track`` names this side.

    ``driven_wheel`` takes the engine through ``ratio``; ``inertia`` (kg m2)
    and ``damping`` (1/s) are the track's and its wheels' as seen there;
    ``brake_torque`` (N m) is the brake on the driven wheel.
    """

    driven_wheel: int
    inertia: float = 10.0
    damping: float = 0.5
    brake_torque: float = 15000.0
    ratio: float = 6.0


@dataclasses.dataclass(frozen=True)
class Lean:
    """A motorcycle's lean controller: a spring (N m/rad) and damper (N m s/rad) toward the lean a turn needs.

    ``max_angle`` (rad) also limits the steering at speed. ``integration``
    and ``integration_decay`` add an integral term; ``smoothing`` (0 to 1)
    filters the target lean.
    """

    max_angle: float = math.radians(45.0)
    spring: float = 5000.0
    damping: float = 1000.0
    integration: float = 0.0
    integration_decay: float = 4.0
    smoothing: float = 0.8


@dataclasses.dataclass(frozen=True)
class VehicleSettings:
    """A vehicle, in SI units and the chassis body's own frame.

    ``forward`` and ``up`` are the chassis's axes; the defaults face -Z with
    +Y up. ``cast`` and ``max_slope`` (rad) decide how a wheel finds the
    ground (``cast_radius`` is the sphere's, 0 taking the smallest wheel
    radius). ``max_pitch_roll`` (rad) keeps the chassis within that angle of
    upright; pi leaves it free.

    Tyres: ``longitudinal_friction`` is (slip ratio, friction) points and
    ``lateral_friction`` (slip angle in degrees, friction) points, scaled by
    each wheel's ``grip`` and by the surface under it. With ``body_friction``
    the ground body's friction scales them as well, as
    ``sqrt(curve * friction)``.

    Steering: the steered wheels turn to ``max_steer`` (rad) at rest; with
    ``steer_falloff_speed`` (m/s) the lock falls to half at that speed, as
    ``max_steer / (1 + (v / falloff)**2)``. ``steer_rate`` (rad/s) limits how
    fast the steer angle moves; 0 moves it at once.

    Forces on the chassis each step: ``drag`` (N at 1 m/s) against the
    velocity, rising with its square; ``downforce`` (N at 1 m/s) down the
    vehicle's world up, rising with the square of speed, while any wheel is on
    the ground; ``rolling_resistance`` (a fraction of each grounded wheel's
    load, plus the surface's own) against the rolling direction above
    0.15 m/s. A vehicle with no throttle and no brake that comes down to
    ``holding_speed`` (m/s) brakes with input ``holding`` (0 to 1) until it
    is given throttle or brake again.

    Blows: the heaviest contact on the chassis in a tick is in the snapshot;
    one closing at ``impact_speed`` (m/s) or faster is also an event of kind
    VEHICLE, phase IMPACT, or CRASH at ``crash_speed`` or faster.

    ``differentials`` None gives one differential per axle of driven wheels
    (wheels sharing a forward coordinate, the left one the lower X), sharing
    the engine equally, with ``differential_ratio`` and
    ``differential_limited_slip``. ``tracks`` are a tracked vehicle's left
    and right tracks; ``lean`` is a motorcycle's.
    """

    wheels: tuple[WheelSettings, ...]
    kind: VehicleKind = VehicleKind.WHEELED
    engine: Engine = Engine()
    transmission: Transmission = Transmission()
    differentials: tuple[Differential, ...] | None = None
    differential_ratio: float = 3.42
    differential_limited_slip: float = 1.4
    anti_roll_bars: tuple[AntiRollBar, ...] = ()
    tracks: tuple[VehicleTrack, VehicleTrack] | None = None
    lean: Lean = Lean()
    forward: Vector = (0.0, 0.0, -1.0)
    up: Vector = (0.0, 1.0, 0.0)
    cast: WheelCast = WheelCast.RAY
    cast_radius: float = 0.0
    max_slope: float = math.radians(80.0)
    max_pitch_roll: float = math.pi
    longitudinal_friction: Curve = ((0.0, 0.0), (0.06, 1.2), (0.2, 1.0))
    lateral_friction: Curve = ((0.0, 0.0), (3.0, 1.2), (20.0, 1.0))
    body_friction: bool = True
    max_steer: float = math.radians(30.0)
    steer_falloff_speed: float = 0.0
    steer_rate: float = 0.0
    drag: float = 0.0
    downforce: float = 0.0
    rolling_resistance: float = 0.0
    holding: float = 0.0
    holding_speed: float = 0.5
    impact_speed: float = 2.0
    crash_speed: float = 10.0

    # -- the shape of it ---------------------------------------------------

    def wheelbase(self) -> float:
        """Front axle to rear axle along ``forward``, in metres: steered wheels to the rest where some steer."""
        forward = np.asarray(self.forward, dtype=np.float64)
        forward = forward / np.linalg.norm(forward)
        along = [float(np.dot(w.position, forward)) for w in self.wheels]
        steered = [a for a, w in zip(along, self.wheels, strict=True) if w.steered]
        rest = [a for a, w in zip(along, self.wheels, strict=True) if not w.steered]
        if steered and rest:
            return abs(sum(steered) / len(steered) - sum(rest) / len(rest))
        return max(along) - min(along) if along else 0.0

    def resolved_differentials(self) -> tuple[Differential, ...]:
        """Return the differentials the vehicle drives with: the given ones, or one per driven axle."""
        if self.differentials is not None:
            return self.differentials
        axles: dict[float, list[int]] = {}
        for index, wheel in enumerate(self.wheels):
            if wheel.driven:
                axles.setdefault(round(float(wheel.position[2]), 6), []).append(index)
        out = []
        share = 1.0 / max(len(axles), 1)
        for key in sorted(axles):
            members = sorted(axles[key], key=lambda i: float(self.wheels[i].position[0]))
            left = members[0]
            right = members[-1] if len(members) > 1 else -1
            out.append(
                Differential(
                    left,
                    right,
                    ratio=self.differential_ratio,
                    limited_slip=self.differential_limited_slip,
                    torque_ratio=share,
                )
            )
        return tuple(out)

    # -- the native form ---------------------------------------------------

    def native(self) -> dict[str, Any]:
        """Return the settings as the native ``add_vehicle`` reads them."""
        e, t, lean = self.engine, self.transmission, self.lean
        params = [
            *self.forward,
            *self.up,
            self.max_pitch_roll,
            self.max_slope,
            e.max_torque,
            e.min_rpm,
            e.max_rpm,
            e.inertia,
            e.damping,
            1.0 if t.automatic else 0.0,
            t.switch_time,
            t.clutch_release_time,
            t.switch_latency,
            t.shift_up_rpm,
            t.shift_down_rpm,
            t.clutch_strength,
            self.differential_limited_slip,
            self.max_steer,
            self.steer_falloff_speed,
            self.steer_rate,
            self.drag,
            self.downforce,
            self.rolling_resistance,
            self.holding,
            self.holding_speed,
            1.0 if self.body_friction else 0.0,
            self.impact_speed,
            self.crash_speed,
            lean.max_angle,
            lean.spring,
            lean.damping,
            lean.integration,
            lean.integration_decay,
            lean.smoothing,
            self.cast_radius,
        ]
        tracks = np.zeros((2, 5), dtype=np.float64)
        if self.tracks is not None:
            for side, track in enumerate(self.tracks):
                tracks[side] = (
                    track.driven_wheel,
                    track.inertia,
                    track.damping,
                    track.brake_torque,
                    track.ratio,
                )
        differentials = (
            ()
            if self.kind == VehicleKind.TRACKED
            else [
                (d.left, d.right, d.ratio, d.split, d.limited_slip, d.torque_ratio)
                for d in self.resolved_differentials()
            ]
        )
        return {
            'kind': int(self.kind),
            'cast': int(self.cast),
            'params': np.array(params, dtype=np.float64),
            'wheels': np.array([w.native() for w in self.wheels], dtype=np.float64).reshape(-1, 17),
            'torque_curve': _pairs(e.torque_curve),
            'longitudinal': _pairs(self.longitudinal_friction),
            'lateral': _pairs(self.lateral_friction),
            'gears': np.array(t.gears, dtype=np.float32),
            'reverse_gears': np.array(t.reverse_gears, dtype=np.float32),
            'differentials': np.array(differentials, dtype=np.float64).reshape(-1, 6),
            'anti_roll_bars': np.array(
                [(b.left, b.right, b.stiffness) for b in self.anti_roll_bars], dtype=np.float64
            ).reshape(-1, 3),
            'tracks': tracks,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as builtin types: dicts, lists, floats, ints and bools."""
        plain: dict[str, Any] = _plain(dataclasses.asdict(self))
        return plain

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VehicleSettings:
        """Return the settings :meth:`as_dict` wrote; missing keys take their defaults."""
        given = dict(values)
        given['wheels'] = tuple(_from(WheelSettings, w) for w in given['wheels'])
        for key, kind in (('engine', Engine), ('transmission', Transmission), ('lean', Lean)):
            if key in given:
                given[key] = _from(kind, given[key])
        if given.get('differentials') is not None:
            given['differentials'] = tuple(_from(Differential, d) for d in given['differentials'])
        if 'anti_roll_bars' in given:
            given['anti_roll_bars'] = tuple(_from(AntiRollBar, b) for b in given['anti_roll_bars'])
        if given.get('tracks') is not None:
            left, right = (_from(VehicleTrack, t) for t in given['tracks'])
            given['tracks'] = (left, right)
        if 'kind' in given:
            given['kind'] = VehicleKind(given['kind'])
        if 'cast' in given:
            given['cast'] = WheelCast(given['cast'])
        for key in ('forward', 'up'):
            if key in given:
                given[key] = _vector(given[key])
        for key in ('longitudinal_friction', 'lateral_friction'):
            if key in given:
                given[key] = tuple((float(x), float(y)) for x, y in given[key])
        return cls(**given)

    # -- omi_physics' raycast vehicle ---------------------------------------

    @classmethod
    def from_raycast(
        cls,
        tuning: Any,
        wheels: Sequence[Any],
        mass: float,
        gravity: float = 9.81,
        **changes: Any,
    ) -> VehicleSettings:
        """Return settings that drive like omi_physics' ``RaycastVehicle`` with this tuning.

        ``tuning`` is a ``VehicleTuning`` and ``wheels`` its ``WheelSpec``
        list (any objects with those attributes); ``mass`` is the chassis's
        (kg) and ``gravity`` the world's (m/s2). ``changes`` replace fields
        of the result. The mapping, field by field:

        - ``WheelSpec.position``, ``radius`` and ``grip`` carry over;
          ``suspension_travel`` is ``suspension_max`` with ``suspension_min``
          0. ``suspension_stiffness`` (car weights per metre) is
          ``stiffness = s * mass * gravity`` N/m; ``suspension_damping`` (a
          fraction of critical) is ``damping = z * 2 sqrt(stiffness * mass /
          wheels)`` N s/m.
        - ``engine_force`` (N at the wheels) and ``base_speed`` become a
          one-gear electric drivetrain: gear 1, differential ratio 1, an
          engine of ``engine_force * radius`` N m whose torque curve is flat
          to ``base_speed`` and falls as ``base_speed / v`` above it
          (constant power), with ``max_rpm`` at three times the speed the
          power and ``drag`` balance at. ``reverse_fraction`` is the reverse
          gear's ratio. Open differentials split the torque equally between
          the driven wheels, as the raycast vehicle shares ``engine_force``.
        - ``brake_force`` (N, shared by the braked wheels) is each braked
          wheel's ``brake_torque = brake_force / n * radius``; the rear
          (unsteered) braked wheels take as much again as a handbrake.
        - ``maximum_steer``, ``steer_falloff_speed`` and ``steer_speed``
          give ``max_steer``, ``steer_falloff_speed`` and ``steer_rate =
          steer_speed * maximum_steer``.
        - ``rolling_resistance``, ``drag`` and ``downforce`` carry over.
          ``holding`` (N) becomes the brake input ``holding / brake_force``
          below 0.5 m/s.
        - The tyre curves saturate at full grip: longitudinal friction 1 from
          a slip ratio of 0.05, lateral friction 1 from half a degree, flat beyond,
          scaled by each wheel's grip; the ground body's friction is not
          used (``body_friction`` False), as the raycast vehicle uses only
          its ``Surface``.
        """
        count = len(wheels)
        braked = sum(1 for w in wheels if getattr(w, 'braked', True)) or 1
        share = mass / max(count, 1)
        brake_force = float(tuning.brake_force)
        settings_wheels = []
        for spec in wheels:
            stiffness = float(spec.suspension_stiffness) * mass * gravity
            damping = float(spec.suspension_damping) * 2.0 * math.sqrt(stiffness * share)
            radius = float(spec.radius)
            brake = brake_force / braked * radius if getattr(spec, 'braked', True) else 0.0
            settings_wheels.append(
                WheelSettings(
                    position=_vector(spec.position),
                    radius=radius,
                    width=0.2,
                    suspension_min=0.0,
                    suspension_max=float(spec.suspension_travel),
                    stiffness=stiffness,
                    damping=damping,
                    steered=bool(spec.steering),
                    driven=bool(spec.driven),
                    brake_torque=brake,
                    handbrake_torque=0.0 if spec.steering else brake,
                    inertia=0.5 * 15.0 * radius * radius,
                    angular_damping=0.0,
                    grip=float(spec.grip),
                )
            )
        radius = sum(w.radius for w in settings_wheels) / max(count, 1)
        force = float(tuning.engine_force)
        base = float(getattr(tuning, 'base_speed', 0.0))
        drag = float(getattr(tuning, 'drag', 0.0))
        # The speed where the power meets the drag; three times it is the engine's limit.
        power = force * base if base > 0.0 else force * 30.0
        balance = (power / drag) ** (1.0 / 3.0) if drag > 0.0 else 100.0
        top = 3.0 * max(balance, base, 10.0)
        max_rpm = top / radius * 60.0 / (2.0 * math.pi)
        curve: list[tuple[float, float]] = [(0.0, 1.0)]
        if base > 0.0:
            knee = base / top
            curve.append((knee, 1.0))
            for k in range(1, 33):
                x = knee + (1.0 - knee) * k / 32.0
                curve.append((x, knee / x))
        else:
            curve.append((1.0, 1.0))
        holding = float(getattr(tuning, 'holding', 0.0))
        maximum_steer = float(tuning.maximum_steer)
        result = cls(
            wheels=tuple(settings_wheels),
            engine=Engine(
                max_torque=force * radius,
                min_rpm=0.0,
                max_rpm=max_rpm,
                torque_curve=tuple(curve),
                inertia=0.05,
                damping=0.0,
            ),
            transmission=Transmission(
                automatic=True,
                gears=(1.0,),
                reverse_gears=(-float(tuning.reverse_fraction),),
                switch_time=0.0,
                clutch_release_time=0.0,
                switch_latency=0.0,
                shift_up_rpm=max_rpm * 2.0,
                shift_down_rpm=0.0,
                clutch_strength=5000.0,
            ),
            differential_ratio=1.0,
            differential_limited_slip=math.inf,
            longitudinal_friction=((0.0, 0.0), (0.05, 1.0), (1.0, 1.0)),
            lateral_friction=((0.0, 0.0), (0.5, 1.0), (90.0, 1.0)),
            body_friction=False,
            max_steer=maximum_steer,
            steer_falloff_speed=float(getattr(tuning, 'steer_falloff_speed', 0.0)),
            steer_rate=float(tuning.steer_speed) * maximum_steer,
            drag=drag,
            downforce=float(getattr(tuning, 'downforce', 0.0)),
            rolling_resistance=float(tuning.rolling_resistance),
            holding=min(1.0, holding / brake_force) if brake_force > 0.0 else 0.0,
            holding_speed=0.5,
        )
        return dataclasses.replace(result, **changes) if changes else result


def _vector(values: Sequence[float]) -> Vector:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _pairs(curve: Iterable[tuple[float, float]]) -> npt.NDArray[np.float32]:
    return np.array([(float(x), float(y)) for x, y in curve], dtype=np.float32).reshape(-1, 2)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, enum.IntEnum):
        return int(value)
    return value


def _from(kind: Any, values: Any) -> Any:
    if isinstance(values, kind):
        return values
    given = dict(values)
    for field in dataclasses.fields(kind):
        if field.name in given and isinstance(given[field.name], list):
            item = given[field.name]
            given[field.name] = tuple(tuple(v) if isinstance(v, list) else v for v in item)
    return kind(**given)


class VehicleRows:
    """Every vehicle of one snapshot, as read-only NumPy views, row ``i`` the vehicle ``id[i]``.

    ``chassis`` is the chassis body's uid and ``kind`` a :class:`VehicleKind`
    code. ``rpm`` is the engine's speed and ``gear`` the gear (-1 reverse, 0
    neutral, 1 first). ``forward_speed`` (m/s) is the chassis's velocity
    along its forward axis. ``steer`` (rad, positive left) is the angle the
    steering handed the wheels. ``impact_speed`` (m/s), ``impact_impulse``
    (N s) and ``impact_uid`` describe the heaviest blow of the tick (uid 0 for
    none). ``input`` (n, 4) holds the throttle, steer, brake and handbrake in
    force. A vehicle's wheels are rows ``first_wheel[i]`` to ``first_wheel[i]
    + wheel_count[i]`` of :attr:`wheels`.
    """

    __slots__ = ('_native', 'tick')

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native
        #: The tick of the snapshot.
        self.tick = native.tick

    def __len__(self) -> int:
        return int(self._native.vehicle_count)

    @property
    def id(self) -> npt.NDArray[np.uint32]:
        """(n,) vehicle ids, ascending."""
        return self._native.vehicle_id

    @property
    def chassis(self) -> npt.NDArray[np.uint32]:
        """(n,) chassis body uids."""
        return self._native.vehicle_chassis

    @property
    def kind(self) -> npt.NDArray[np.uint8]:
        """(n,) :class:`VehicleKind` codes."""
        return self._native.vehicle_kind

    @property
    def first_wheel(self) -> npt.NDArray[np.uint32]:
        """(n,) row of each vehicle's first wheel in :attr:`wheels`."""
        return self._native.vehicle_first_wheel

    @property
    def wheel_count(self) -> npt.NDArray[np.uint32]:
        """(n,) number of wheels of each vehicle."""
        return self._native.vehicle_wheels

    @property
    def rpm(self) -> npt.NDArray[np.float32]:
        """(n,) engine speed, revolutions a minute."""
        return self._native.vehicle_rpm

    @property
    def gear(self) -> npt.NDArray[np.int8]:
        """(n,) gear: -1 reverse, 0 neutral, 1 first."""
        return self._native.vehicle_gear

    @property
    def forward_speed(self) -> npt.NDArray[np.float32]:
        """(n,) speed along the chassis's forward axis, m/s; negative reversing."""
        return self._native.vehicle_forward_speed

    @property
    def steer(self) -> npt.NDArray[np.float32]:
        """(n,) steer angle handed to the steered wheels, rad, positive left."""
        return self._native.vehicle_steer

    @property
    def impact_speed(self) -> npt.NDArray[np.float32]:
        """(n,) closing speed of the tick's heaviest blow on the chassis, m/s; 0 for none."""
        return self._native.vehicle_impact_speed

    @property
    def impact_impulse(self) -> npt.NDArray[np.float32]:
        """(n,) estimated impulse of that blow, N s."""
        return self._native.vehicle_impact_impulse

    @property
    def impact_uid(self) -> npt.NDArray[np.uint32]:
        """(n,) uid of the body struck; 0 for none."""
        return self._native.vehicle_impact_uid

    @property
    def input(self) -> npt.NDArray[np.float32]:
        """(n, 4) throttle, steer, brake and handbrake in force."""
        return self._native.vehicle_input

    @property
    def wheels(self) -> WheelRows:
        """Every wheel of every vehicle; see :class:`WheelRows`."""
        return WheelRows(self._native)

    def rows(self) -> dict[int, int]:
        """Return the row of each vehicle id."""
        return {int(v): i for i, v in enumerate(self.id)}


class WheelRows:
    """Every wheel of one snapshot, in vehicle order, as read-only NumPy views.

    ``vehicle`` is the row of the wheel's vehicle in :class:`VehicleRows`.
    ``position`` (w, 3) float64 is the wheel's centre and ``orientation``
    (w, 4) the rotation taking a wheel model with its axle along X and up
    along Y to the world, steer and spin included. ``length`` (m) is the
    suspension's length and ``compression`` (m) its maximum length less that.
    ``steer`` (rad, positive left), ``angle`` (rad about the axle) and
    ``spin`` (rad/s, positive rolling forward). ``load`` (N) is the
    suspension's force. ``slip`` (w, 2) is the longitudinal slip ratio and
    the lateral slip angle (rad), zero on tracks. ``contact`` is true on the
    ground; ``ground`` is the uid under the wheel (0 for none), ``material``
    that body's material index, ``triangle`` the mesh triangle's number (-1
    off a mesh), ``grip`` what the surface gives the tyre, and
    ``contact_point`` and ``contact_normal`` where and how it touches.
    """

    __slots__ = ('_native',)

    def __init__(self, native: _native.Snapshot) -> None:
        self._native = native

    def __len__(self) -> int:
        return int(self._native.wheel_count)

    @property
    def vehicle(self) -> npt.NDArray[np.uint32]:
        """(w,) row of each wheel's vehicle."""
        return self._native.wheel_vehicle

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """(w, 3) wheel centres, metres."""
        return self._native.wheel_position

    @property
    def orientation(self) -> npt.NDArray[np.float32]:
        """(w, 4) wheel rotations, x, y, z, w."""
        return self._native.wheel_orientation

    @property
    def length(self) -> npt.NDArray[np.float32]:
        """(w,) suspension lengths, m."""
        return self._native.wheel_length

    @property
    def compression(self) -> npt.NDArray[np.float32]:
        """(w,) suspension compression from full droop, m."""
        return self._native.wheel_compression

    @property
    def steer(self) -> npt.NDArray[np.float32]:
        """(w,) steer angles, rad, positive left."""
        return self._native.wheel_steer

    @property
    def angle(self) -> npt.NDArray[np.float32]:
        """(w,) rotation about the axle, rad."""
        return self._native.wheel_angle

    @property
    def spin(self) -> npt.NDArray[np.float32]:
        """(w,) angular velocity about the axle, rad/s."""
        return self._native.wheel_spin

    @property
    def load(self) -> npt.NDArray[np.float32]:
        """(w,) suspension force, N."""
        return self._native.wheel_load

    @property
    def slip(self) -> npt.NDArray[np.float32]:
        """(w, 2) longitudinal slip ratio and lateral slip angle, rad."""
        return self._native.wheel_slip

    @property
    def contact(self) -> npt.NDArray[np.bool_]:
        """(w,) true where the wheel is on the ground."""
        return self._native.wheel_contact.view(np.bool_)

    @property
    def ground(self) -> npt.NDArray[np.uint32]:
        """(w,) uid of the body under the wheel; 0 for none."""
        return self._native.wheel_ground

    @property
    def material(self) -> npt.NDArray[np.uint16]:
        """(w,) material index of the body under the wheel."""
        return self._native.wheel_material

    @property
    def triangle(self) -> npt.NDArray[np.int32]:
        """(w,) number of the mesh triangle under the wheel; -1 off a mesh."""
        return self._native.wheel_triangle

    @property
    def grip(self) -> npt.NDArray[np.float32]:
        """(w,) grip the surface gives the tyre; 0 off the ground."""
        return self._native.wheel_grip

    @property
    def contact_point(self) -> npt.NDArray[np.float64]:
        """(w, 3) where each tyre touches, metres."""
        return self._native.wheel_contact_point

    @property
    def contact_normal(self) -> npt.NDArray[np.float32]:
        """(w, 3) the ground's normal there."""
        return self._native.wheel_contact_normal


class Vehicle:
    """The handle of one vehicle of a world: inputs queued, state read from the newest snapshot."""

    __slots__ = ('chassis', 'id', 'world')

    def __init__(self, world: World, vid: int, chassis: int) -> None:
        self.world = world
        #: The vehicle id.
        self.id = int(vid)
        #: The chassis body's uid.
        self.chassis = int(chassis)

    def __repr__(self) -> str:
        return '<Vehicle %d chassis=%d>' % (self.id, self.chassis)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vehicle) and other.world is self.world and other.id == self.id

    def __hash__(self) -> int:
        return hash((id(self.world), self.id))

    def set_input(
        self,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        handbrake: float = 0.0,
        at_tick: int = 0,
    ) -> int:
        """Queue the driver's inputs, held from the next tick until changed; return the sequence number.

        ``throttle`` -1 to 1 (negative reverses), ``steer`` -1 to 1 (positive
        turns left, the way a positive yaw about +Y does), ``brake`` and
        ``handbrake`` 0 to 1.
        """
        return self.world.vehicle_inputs([self.id], [(throttle, steer, brake, handbrake)], at_tick)

    def set_surface(
        self,
        grip: float | None,
        rolling: float = 0.0,
        wheel: int | None = None,
        at_tick: int = 0,
    ) -> int:
        """Queue the surface under one wheel, or under all of them (``wheel`` None).

        ``grip`` scales the tyre's friction there and ``rolling`` adds to the
        rolling resistance, as a fraction of the wheel's load. ``grip`` None
        gives the wheels back to the world's surface table
        (:meth:`World.set_surface_table <pyjoltphysics.World.set_surface_table>`).
        """
        return self.world.set_vehicle_surfaces(
            [self.id], [-1 if wheel is None else int(wheel)], [grip], [rolling], at_tick
        )

    def remove(self, at_tick: int = 0) -> int:
        """Queue removing the vehicle; the chassis body stays."""
        return self.world.remove_vehicles([self.id], at_tick)

    def _row(self) -> tuple[VehicleRows, int]:
        snapshot = self.world.latest()
        if snapshot is None:
            raise LookupError('no snapshot has been published yet')
        rows = snapshot.vehicles
        found = np.flatnonzero(rows.id == self.id)
        if not len(found):
            raise LookupError('vehicle %d is not in the newest snapshot' % self.id)
        return rows, int(found[0])

    @property
    def forward_speed(self) -> float:
        """Speed along the chassis's forward axis, m/s; negative reversing."""
        rows, i = self._row()
        return float(rows.forward_speed[i])

    @property
    def rpm(self) -> float:
        """Engine speed, revolutions a minute."""
        rows, i = self._row()
        return float(rows.rpm[i])

    @property
    def gear(self) -> int:
        """Gear: -1 reverse, 0 neutral, 1 first."""
        rows, i = self._row()
        return int(rows.gear[i])

    @property
    def steer_angle(self) -> float:
        """Steer angle handed to the steered wheels, rad, positive left."""
        rows, i = self._row()
        return float(rows.steer[i])

    def impact(self) -> tuple[int, float, float] | None:
        """Return the heaviest blow of the newest tick: (uid struck, closing speed m/s, impulse N s), or None."""
        rows, i = self._row()
        uid = int(rows.impact_uid[i])
        if uid == 0:
            return None
        return uid, float(rows.impact_speed[i]), float(rows.impact_impulse[i])

    def wheel_slice(self) -> slice:
        """Return this vehicle's rows of :attr:`VehicleRows.wheels`."""
        rows, i = self._row()
        first = int(rows.first_wheel[i])
        return slice(first, first + int(rows.wheel_count[i]))
