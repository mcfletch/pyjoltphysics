"""The compiled pyjoltphysics module over Jolt Physics."""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def jolt_version() -> str:
    """
    Return the version of the Jolt Physics compiled into the module, such as '5.6.0'.
    """

def build_options() -> dict[str, bool | str]:
    """Return the options the module and its Jolt were built with, by name."""

def selftest() -> bool:
    """
    Return True after stepping a one-body Jolt world once on the calling thread.
    """

class CallbackFailed(RuntimeError):
    pass

class WorldStopped(RuntimeError):
    pass

EVENT_SIZE: int = 72

class Snapshot:
    """The state of every body after one tick."""

    @property
    def tick(self) -> int: ...

    @property
    def time(self) -> float: ...

    @property
    def applied_seq(self) -> int: ...

    @property
    def count(self) -> int: ...

    @property
    def uid(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def position(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def orientation(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 4), writable=False)]: ...

    @property
    def linear_velocity(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 3), writable=False)]: ...

    @property
    def angular_velocity(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 3), writable=False)]: ...

    @property
    def awake(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def motion_type(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def character_count(self) -> int: ...

    @property
    def character_id(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_position(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def character_velocity(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 3), writable=False)]: ...

    @property
    def character_state(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def character_grounded(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def character_ground(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_ground_normal(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 3), writable=False)]: ...

    @property
    def character_slide(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_landing(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_landed_tick(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_height(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def character_previous(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def character_eye(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def character_ground_material(self) -> Annotated[NDArray[numpy.uint16], dict(shape=(None,), writable=False)]: ...

    @property
    def character_flags(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_count(self) -> int: ...

    @property
    def joint_id(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_kind(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_state(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_value(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_value2(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_force(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def joint_torque(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_count(self) -> int: ...

    @property
    def wheel_count(self) -> int: ...

    @property
    def vehicle_id(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_chassis(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_kind(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_first_wheel(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_wheels(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_rpm(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_gear(self) -> Annotated[NDArray[numpy.int8], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_forward_speed(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_steer(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_impact_speed(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_impact_impulse(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_impact_uid(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def vehicle_input(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 4), writable=False)]: ...

    @property
    def wheel_vehicle(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_position(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def wheel_orientation(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 4), writable=False)]: ...

    @property
    def wheel_length(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_compression(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_steer(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_angle(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_spin(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_load(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_slip(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 2), writable=False)]: ...

    @property
    def wheel_contact(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_ground(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_material(self) -> Annotated[NDArray[numpy.uint16], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_triangle(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_grip(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,), writable=False)]: ...

    @property
    def wheel_contact_point(self) -> Annotated[NDArray[numpy.float64], dict(shape=(None, 3), writable=False)]: ...

    @property
    def wheel_contact_normal(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, 3), writable=False)]: ...

    @property
    def changed(self) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,), writable=False)]: ...

class World:
    """A Jolt physics world stepped on its own native thread."""

    def __init__(self, *, tick_hz: float = 120.0, max_catch_up: float = 0.1, gravity: Sequence[float] = (0.0, -9.81, 0.0), threads: int = 4, max_bodies: int = 65536, max_body_pairs: int = 65536, max_contact_constraints: int = 65536, event_capacity: int = 65536, allow_sleeping: bool = True, time_before_sleep: float = 0.5, sleep_velocity: float = 0.03, min_velocity_for_restitution: float = 1.0, collision_steps: int = 1, temp_allocator_mb: int = 16, static_friction_speed: float = 0.05, penetration_slop: float = 0.02) -> None: ...

    def add_bodies(self, position: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], orientation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], shape_kind: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], shape_params: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], shape_id: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], scale: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], shape_offset: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], shape_rotation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], center_of_mass: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], motion: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], mass: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], density: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], restitution: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], angular_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], gravity_factor: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear_velocity: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], angular_velocity: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], group: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], mask: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], flags: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], material: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], filter: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], drag: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> tuple[Annotated[NDArray[numpy.uint32], dict(shape=(None,))], int]:
        """
        Queue bodies to add; return their uids and the batch's sequence number.
        """

    def swap_bodies(self, remove: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], position: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], orientation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], shape_kind: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], shape_params: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], shape_id: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], scale: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], shape_offset: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], shape_rotation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], center_of_mass: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], motion: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], mass: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], density: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], restitution: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], angular_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], gravity_factor: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear_velocity: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], angular_velocity: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], group: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], mask: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], flags: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], material: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], filter: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], drag: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> tuple[Annotated[NDArray[numpy.uint32], dict(shape=(None,))], int]:
        """
        Queue removing bodies and adding others in one tick; return the new uids and the sequence number.
        """

    def raycast(self, origin: Sequence[float], direction: Sequence[float], max_distance: float, group: int, mask: int, exclude: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], triggers: bool, characters: bool) -> tuple[int, float, list[float], list[float], int] | None:
        """
        Cast one ray; return (uid, fraction, point, normal, sub_shape) of the first hit, or None.
        """

    def raycast_batch(self, origins: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], directions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], max_distance: float, group: int, mask: int, exclude: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], triggers: bool, characters: bool) -> tuple[Annotated[NDArray[numpy.uint8], dict(shape=(None,))], Annotated[NDArray[numpy.uint32], dict(shape=(None,))], Annotated[NDArray[numpy.float32], dict(shape=(None,))], Annotated[NDArray[numpy.float64], dict(shape=(None, 3))], Annotated[NDArray[numpy.float32], dict(shape=(None, 3))], Annotated[NDArray[numpy.uint32], dict(shape=(None,))]]:
        """
        Cast rays; return (hit, uid, fraction, point, normal, sub_shape) arrays.
        """

    def cast_shape(self, kind: int, params: Sequence[float], shape_id: int, start: Sequence[float], rotation: Sequence[float], end: Sequence[float], all_hits: bool, group: int, mask: int, exclude: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], triggers: bool, characters: bool) -> tuple[Annotated[NDArray[numpy.uint32], dict(shape=(None,))], Annotated[NDArray[numpy.uint32], dict(shape=(None,))], Annotated[NDArray[numpy.float32], dict(shape=(None,))], Annotated[NDArray[numpy.float32], dict(shape=(None,))], Annotated[NDArray[numpy.float64], dict(shape=(None, 3))], Annotated[NDArray[numpy.float32], dict(shape=(None, 3))]]:
        """
        Sweep a shape; return (uid, sub_shape, fraction, depth, point, normal) arrays, one row a body hit.
        """

    def overlap(self, kind: int, params: Sequence[float], shape_id: int, position: Sequence[float], rotation: Sequence[float], group: int, mask: int, exclude: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], triggers: bool, characters: bool) -> Annotated[NDArray[numpy.uint32], dict(shape=(None,))]:
        """Return the ascending uids of the bodies a shape at a pose touches."""

    def add_character(self, position: Sequence[float], caps: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], bits: int, gravity: float, group: int, mask: int, pass_group: int, proxy: bool, ignore: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> tuple[int, int]:
        """Queue a character; return its id and the batch's sequence number."""

    def remove_characters(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def character_ops(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], ops: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], codes: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int:
        """Queue character inputs, applied in order at the next tick."""

    def set_character_ignore(self, id: int, ignore: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def add_convex(self, points: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], max_convex_radius: float, hull_tolerance: float) -> int:
        """Build and register a convex hull; return its shape id."""

    def add_mesh(self, points: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], triangles: Annotated[NDArray[numpy.uint32], dict(shape=(None, 3), order='C', device='cpu', writable=False)], materials: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> int:
        """Build and register a triangle mesh; return its shape id."""

    def add_heightfield(self, heights: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], origin: Sequence[float], spacing: Sequence[float], block_size: int, bits_per_sample: int, min_height: float, max_height: float) -> int:
        """Build and register a height field; return its shape id."""

    def add_compound(self, children: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], positions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], orientations: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], mutable: bool) -> int:
        """
        Build and register a compound of registered shapes; return its shape id.
        """

    def add_primitive(self, kind: int, params: Sequence[float]) -> int:
        """Register a primitive shape; return its shape id."""

    def remove_shape(self, shape_id: int) -> bool:
        """Forget a shape id; bodies made with the shape keep it."""

    def shape_info(self, shape_id: int) -> dict[str, object]: ...

    def remove_bodies(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_transforms(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], position: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], orientation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_velocities(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)] | None, angular: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)] | None, at_tick: int = 0) -> int: ...

    def add_forces(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def add_impulses(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def add_torques(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_gravity(self, gravity: Sequence[float], at_tick: int = 0) -> int: ...

    def activate(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_motion_types(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], motion: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_materials(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], restitution: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def add_joints(self, kind: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], a: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], b: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], anchor: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], to_anchor: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], axis: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], axis2: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], axis3: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], lower: Annotated[NDArray[numpy.float64], dict(shape=(None, 6), order='C', device='cpu', writable=False)], upper: Annotated[NDArray[numpy.float64], dict(shape=(None, 6), order='C', device='cpu', writable=False)], spring: Annotated[NDArray[numpy.float32], dict(shape=(None, 2), order='C', device='cpu', writable=False)], friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], motor_mode: Annotated[NDArray[numpy.uint8], dict(shape=(None, 6), order='C', device='cpu', writable=False)], motor_target: Annotated[NDArray[numpy.float64], dict(shape=(None, 6), order='C', device='cpu', writable=False)], motor_max: Annotated[NDArray[numpy.float64], dict(shape=(None, 6), order='C', device='cpu', writable=False)], motor_spring: Annotated[NDArray[numpy.float32], dict(shape=(None, 2), order='C', device='cpu', writable=False)], break_force: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], break_torque: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], collide: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> tuple[Annotated[NDArray[numpy.uint32], dict(shape=(None,))], int]:
        """Queue joints; return their ids and the batch's sequence number."""

    def remove_joints(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_joint_motors(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], axis: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], mode: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], target: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], max: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_joints_enabled(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], enabled: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def add_vehicle(self, chassis: int, kind: int, cast: int, params: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], wheels: Annotated[NDArray[numpy.float64], dict(shape=(None, 17), order='C', device='cpu', writable=False)], torque_curve: Annotated[NDArray[numpy.float32], dict(shape=(None, 2), order='C', device='cpu', writable=False)], longitudinal: Annotated[NDArray[numpy.float32], dict(shape=(None, 2), order='C', device='cpu', writable=False)], lateral: Annotated[NDArray[numpy.float32], dict(shape=(None, 2), order='C', device='cpu', writable=False)], gears: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], reverse_gears: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], differentials: Annotated[NDArray[numpy.float64], dict(shape=(None, 6), order='C', device='cpu', writable=False)], anti_roll_bars: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], tracks: Annotated[NDArray[numpy.float64], dict(shape=(2, 5), order='C', device='cpu', writable=False)], at_tick: int = 0) -> tuple[int, int]:
        """
        Queue a vehicle on a dynamic chassis; return its id and the batch's sequence number.
        """

    def remove_vehicles(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def vehicle_inputs(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float32], dict(shape=(None, 4), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int:
        """
        Queue vehicle inputs (throttle, steer, brake, handbrake), held from the next tick.
        """

    def set_vehicle_surfaces(self, ids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], wheel: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C', device='cpu', writable=False)], grip: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], rolling: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_surface_table(self, grip: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], rolling: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_material_table(self, static_friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], dynamic_friction: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], restitution: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], friction_combine: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], restitution_combine: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], pairs: Annotated[NDArray[numpy.float32], dict(shape=(None, None, 3), order='C', device='cpu', writable=False)], static_speed: float, at_tick: int = 0) -> int: ...

    def set_body_materials(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], material: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_filter_table(self, table: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_body_filters(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], filter: Annotated[NDArray[numpy.uint16], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_gravity_zones(self, shape: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], field: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], position: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], rotation: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], half: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], shape_id: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], gravity: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], center: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], priority: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C', device='cpu', writable=False)], replace: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], stop: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_motion(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], mass: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], linear_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], angular_damping: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], gravity_factor: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], inertia: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_flags(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], bits: int, on: bool, at_tick: int = 0) -> int: ...

    def bodies_state(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> tuple[Annotated[NDArray[numpy.float64], dict(shape=(None, 3))], Annotated[NDArray[numpy.float32], dict(shape=(None, 4))], Annotated[NDArray[numpy.float32], dict(shape=(None, 3))], Annotated[NDArray[numpy.float32], dict(shape=(None, 3))], Annotated[NDArray[numpy.uint8], dict(shape=(None,))], Annotated[NDArray[numpy.uint8], dict(shape=(None,))]]:
        """
        Each body's position, orientation, velocities, awake and found, as Jolt holds it now.
        """

    def triangle_of(self, uid: int, sub_shape: int) -> int:
        """
        The number of the mesh triangle sub_shape names on body uid, or -1 for a part that is no mesh triangle.
        """

    def flush(self) -> None: ...

    def save_state(self) -> bytes: ...

    def restore_state(self, data: bytes, at_tick: int = 0) -> int: ...

    def set_drag(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], drag: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int: ...

    def set_tracks(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], times: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], positions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], rotations: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], loop: bool, at_tick: int = 0) -> int: ...

    def move_kinematic(self, uids: Annotated[NDArray[numpy.uint32], dict(shape=(None,), order='C', device='cpu', writable=False)], positions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], rotations: Annotated[NDArray[numpy.float64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], at_tick: int = 0) -> int:
        """
        Move kinematic bodies to poses by the end of a tick: this one from a world callback, else the next.
        """

    def set_drivers(self, fn: object | None) -> None: ...

    def set_contact_filter(self, fn: object | None, either: bool = False, recheck: bool = False) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def run_ticks(self, count: int) -> None: ...

    def wait_tick(self, tick: int, timeout: float = -1.0) -> bool: ...

    def latest(self) -> Snapshot | None: ...

    def drain_events(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

    def drain_applied(self) -> Annotated[NDArray[numpy.uint64], dict(shape=(None, 2))]: ...

    def set_before_step(self, fn: object | None) -> None: ...

    def set_after_step(self, fn: object | None) -> None: ...

    def set_on_events(self, fn: object | None) -> None: ...

    def at_tick(self, tick: int, fn: object) -> None: ...

    @property
    def error(self) -> object: ...

    @property
    def tick(self) -> int: ...

    @property
    def lockstep(self) -> bool: ...

    @property
    def forked(self) -> bool: ...

    @property
    def submitted_seq(self) -> int: ...

    @property
    def on_world_thread(self) -> bool: ...

    def stats(self) -> dict[str, int | float | bool]: ...

    def reset_stats(self) -> None: ...

def stop_all_worlds() -> None:
    """
    Stop every world; from then on no world thread runs a callback. For atexit.
    """

def fork_prepare() -> None:
    """The os.register_at_fork hook run before a fork."""

def fork_parent() -> None:
    """The os.register_at_fork hook run in the parent."""

def fork_child() -> None:
    """
    The os.register_at_fork hook run in the child: every world is marked forked.
    """
