"""The codes the native world uses for motion types, shapes, flags and events."""

from __future__ import annotations

import enum

__all__ = [
    'BodyFlag',
    'Combine',
    'EndReason',
    'EventKind',
    'EventPhase',
    'JointKind',
    'JointState',
    'MotionType',
    'MotorMode',
    'ShapeKind',
    'ZoneField',
    'ZoneShape',
]


class MotionType(enum.IntEnum):
    """How a body moves: Jolt's ``EMotionType`` codes."""

    #: Never moves; collides with moving bodies only.
    STATIC = 0
    #: Moved by commands only; pushes dynamic bodies and is not pushed.
    KINEMATIC = 1
    #: Moved by forces, gravity and contacts.
    DYNAMIC = 2


class ShapeKind(enum.IntEnum):
    """A collision shape's kind; for a primitive, the meaning of its four parameters.

    Sizes follow the OMI physics shape model, in metres. Capsules and
    cylinders run along the body's Y axis. The kinds from ``CONVEX`` on carry
    arrays and are built by :meth:`~pyjoltphysics.World.add_shape`.
    """

    #: Edge lengths x, y, z.
    BOX = 1
    #: Radius.
    SPHERE = 2
    #: Radius, then the height of the cylinder between the hemisphere centres.
    CAPSULE = 3
    #: Radius, then the total height.
    CYLINDER = 4
    #: Half extent of the plane's bounds (0 for 1 km). The plane passes
    #: through the body's origin with the body's +Y as its normal; static only.
    PLANE = 5
    #: The convex hull of a point cloud.
    CONVEX = 6
    #: Indexed triangles; static and kinematic bodies only.
    MESH = 7
    #: A grid of heights; static and kinematic bodies only.
    HEIGHTFIELD = 8
    #: Child shapes placed in the body's frame.
    COMPOUND = 9


class BodyFlag(enum.IntFlag):
    """Per-body options, combined with ``|``."""

    NONE = 0
    #: Reports overlaps as trigger events and makes no contact response.
    SENSOR = 1 << 0
    #: Never falls asleep.
    NO_SLEEP = 1 << 1
    #: Continuous collision detection (Jolt's linear cast motion quality),
    #: for small fast bodies that would pass through thin ones in one tick.
    CONTINUOUS = 1 << 2
    #: Persist events, once a tick, for every pair this body touches.
    REPORT = 1 << 3
    #: A static body that a later ``set_motion_types`` may make kinematic
    #: or dynamic; it needs a mass or density.
    MOVABLE = 1 << 4
    #: Jolt's enhanced internal edge removal: contacts with the inner edges of
    #: a mesh, height field or compound are merged with the faces beside them,
    #: so a body sliding across them does not catch on an edge. A static mesh
    #: or height field has it without the flag; on a moving body it applies
    #: to every pair the body is in, and to the seams between static bodies
    #: laid flush beside each other.
    SMOOTH_EDGES = 1 << 5
    #: A static mesh or height field without enhanced internal edge removal.
    NO_SMOOTH_EDGES = 1 << 6
    #: The world's contact filter (:meth:`~pyjoltphysics.World.set_contact_filter`)
    #: is asked about pairs where both bodies carry this flag.
    FILTER_CONTACTS = 1 << 7


class EventKind(enum.IntEnum):
    """What an event record is about."""

    #: Two solid bodies touching.
    CONTACT = 0
    #: A body overlapping a sensor; ``a`` is the sensor.
    TRIGGER = 1
    #: A body falling asleep or waking; ``a`` is the body and ``b`` is 0.
    STATE = 2
    #: A character's record; ``a`` is the character.
    CHARACTER = 3
    #: A joint's record; ``a`` is the joint id.
    JOINT = 4
    #: A vehicle's record; ``a`` is the vehicle id.
    VEHICLE = 5


class EventPhase(enum.IntEnum):
    """Where in a pair's or a body's life an event record falls."""

    BEGIN = 0
    PERSIST = 1
    END = 2
    SLEEP = 3
    WAKE = 4
    #: A character landed on body ``b``; ``approach`` is the landing speed.
    LAND = 5
    #: A character jumped; ``approach`` is the launch speed.
    JUMP = 6
    #: A character began a slide.
    SLIDE = 7
    #: A character was knocked down.
    DOWN = 8
    #: A character got up.
    UP = 9
    #: A joint broke; ``approach`` is the force (N) and ``impulse`` the torque (N m) it carried.
    BREAK = 10
    #: A vehicle's heaviest blow of a tick: ``b`` the body struck, ``approach`` the closing
    #: speed (m/s), ``impulse`` its estimate (N s), ``normal`` from the chassis into ``b``.
    IMPACT = 11
    #: An impact at or above the vehicle's ``crash_speed``.
    CRASH = 12


class EndReason(enum.IntEnum):
    """Why an end record ended its pair."""

    #: The bodies parted, or an end record of another phase.
    NONE = 0
    #: A body of the pair was removed.
    REMOVED = 1


class JointKind(enum.IntEnum):
    """What a joint holds: Jolt's constraint of each kind."""

    #: Holds the two bodies at the pose they had when it was made.
    FIXED = 0
    #: Holds a point of each body together; they turn freely about it.
    BALL = 1
    #: Turns about one axis through an anchor.
    HINGE = 2
    #: Slides along one axis, without turning.
    SLIDER = 3
    #: Holds two points at a distance, or within a range of distances.
    DISTANCE = 4
    #: Each of six axes (translation x, y, z, rotation x, y, z of its frame)
    #: locked, free or limited; universal and two-axis hinges are made of it.
    SIX_DOF = 5
    #: X3D's MotorJoint: torques about up to three axes, every step.
    MOTOR = 6


class MotorMode(enum.IntEnum):
    """What a joint's motor drives towards: Jolt's ``EMotorState``."""

    OFF = 0
    #: A speed: m/s or rad/s.
    VELOCITY = 1
    #: A position: m or rad, reached as a spring of the motor's frequency and damping.
    POSITION = 2


class JointState(enum.IntEnum):
    """A joint in a snapshot."""

    ACTIVE = 0
    #: Switched off by :meth:`~pyjoltphysics.World.set_joints_enabled`.
    DISABLED = 1
    #: Broken past its force or torque limit and out of the world.
    BROKEN = 2


class Combine(enum.IntEnum):
    """How two materials' friction or restitution make one; the higher code wins.

    ``DEFAULT`` is Jolt's rule (the geometric mean of the frictions and the
    larger restitution), which a body without a material uses. The others are
    the OMI physics material's combine modes.
    """

    DEFAULT = 0
    AVERAGE = 1
    MINIMUM = 2
    MULTIPLY = 3
    MAXIMUM = 4


class ZoneShape(enum.IntEnum):
    """The region of a gravity zone."""

    BOX = 0
    SPHERE = 1
    #: A registered shape at a pose.
    SHAPE = 2


class ZoneField(enum.IntEnum):
    """The field of a gravity zone (OMI_physics_gravity's types)."""

    #: Along a direction.
    DIRECTIONAL = 0
    #: Towards a centre.
    POINT = 1
