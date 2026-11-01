# pyjoltphysics API

The package's public names, all in `pyjoltphysics`. Units are SI throughout:
metres, seconds, kilograms, radians.

## The world

```python
import numpy as np
import pyjoltphysics as pj

world = pj.World(pj.WorldSettings(tick_hz=120.0, threads=4))
world.add_bodies(pj.BodyBatch.plane())
uids, seq = world.add_bodies(pj.BodyBatch.boxes(np.random.uniform(-5, 5, (100, 3)) + (0, 10, 0),
                                                size=(0.5, 0.5, 0.5), mass=1.0))
world.start()                       # free-runs at tick_hz on the world thread
world.wait_tick(60)
snapshot = world.latest()           # read-only NumPy views, no copy
rows = snapshot.rows_of(uids)
heights = snapshot.position[rows, 1]
events = world.drain_events()       # structured array of pj.EVENT_DTYPE
world.stop()
```

A `World` owns one Jolt `PhysicsSystem`, a Jolt job system and a native world
thread. The world thread runs one tick at a time:

1. apply the command batches due at this tick (the exclusive world lock)
2. call the `at_tick` callables due, then `before_step`
3. move the characters, then step the world by `1 / tick_hz` seconds (`PhysicsSystem::Update`); both under the exclusive world lock
4. fold the step's contact reports into event records and append them to the event ring
5. publish the snapshot
6. call `after_step`, then `on_events` when the tick made records

The thread either free-runs or runs in lockstep:

| Call | Effect |
|---|---|
| `start()` | free-run at `tick_hz`, starting the thread if it is not running; a `run_ticks` waiting on another thread raises `WorldStopped` for the ticks it had not run; from a callback or the contact filter, free-run from the next tick, leaving a stop asked for earlier in place |
| `run_ticks(n)` | run exactly `n` ticks, unpaced, and return when they are done; the world stays in lockstep until the next `start()`; raises `WorldStopped` when `stop()` or `start()` ends the run first |
| `stop()` | stop and join the thread; idempotent; from a callback or the contact filter, the thread ends after the current tick and nothing is joined |
| `flush()` | apply the batches due at the current tick on the calling thread, under the world lock, without stepping, and publish a snapshot of the current tick with what they changed; queries then meet what they add; nothing from a step callback |
| `with World(...) as world:` | `stop()` at the end of the block |

A free-running world with no pending batch, no awake body, no step callback
(`before_step` or `after_step`) and no `at_tick` due at its current tick
waits on a condition variable: it runs no ticks and uses no CPU, and its tick
count and simulation time stand still. A command wakes it at once. An
`at_tick` for a later tick does not keep it ticking: the callable runs when
other work, or `run_ticks`, brings the world to its tick.

A lockstep world counts ticks without stepping once it is quiet: no awake
body, character, kinematic track or target, motor torque or step callback,
and two ticks in a row that applied no batch and made no contact report.
Jolt's state then does not change from one step to the next, so
`run_ticks(n)` advances the tick count and simulation time by `n` and
publishes one snapshot of the new tick (with no changed rows) on the calling
thread, without waking the world thread. The decision is made per call: a
call that a waiting batch, a due `at_tick` or an awake body would act on
steps every tick, and a woken world plays out bitwise as one stepped through
the quiet ticks. `stats()['quiet_ticks']` counts the ticks counted this way.

After a tick that adds 256 or more static bodies, the world thread rebuilds
the broad phase's trees (`PhysicsSystem::OptimizeBroadPhase`) once, the next
time it has no tick due: before waiting in lockstep, or in the time before
the next scheduled tick when free-running. `stats()['broadphase_optimizations']`
counts the rebuilds. A free-running world
that falls more than `max_catch_up` seconds behind its schedule drops the
ticks it is behind by (`stats()['dropped_ticks']`), so simulated time slows.

A world is also stopped when it is garbage collected and at interpreter exit
(an `atexit` hook registered with the first world). A world is never stepped
inline on the calling thread and never in another process.

### `WorldSettings`

A frozen dataclass; every field has a default.

| Field | Default | Meaning |
|---|---|---|
| `tick_hz` | 120.0 | ticks a second, 1 to 10000 |
| `max_catch_up` | 0.1 | seconds behind schedule before a free-running world drops ticks |
| `gravity` | (0, -9.81, 0) | m/s2, each finite as a 32-bit float |
| `threads` | 4 | Jolt job threads, the world thread included, 1 to 256; 1 steps on the world thread alone |
| `max_bodies` | 65536 | bodies the world holds; an add batch past it raises `ValueError` |
| `max_body_pairs`, `max_contact_constraints` | 65536 | Jolt's pair and contact capacities |
| `event_capacity` | 65536 | event records held for `drain_events` |
| `allow_sleeping` | True | bodies at rest fall asleep |
| `time_before_sleep` | 0.5 | seconds below `sleep_velocity` before a body falls asleep |
| `sleep_velocity` | 0.03 | m/s at every point of the body |
| `min_velocity_for_restitution` | 1.0 | m/s closing speed below which a contact does not bounce |
| `collision_steps` | 1 | Jolt collision sub-steps a tick |
| `temp_allocator_mb` | 16 | step scratch memory allocated up front, 0 to 4095; more comes from the heap |
| `linear_damping`, `angular_damping` | 0.0 | 1/s, for bodies whose batch names none |
| `static_friction_speed` | 0.05 | m/s of sliding at a contact below which a material's static friction holds |
| `penetration_slop` | 0.02 | m two bodies may sink into each other before the solver separates them (Jolt's `mPenetrationSlop`); a resting body sits up to this far into what it rests on |

`settings.dt` is `1 / tick_hz`.

### Reading

| Call | Returns |
|---|---|
| `latest()` | the newest `Snapshot`, or None before the first tick; never waits |
| `bodies_state(uids)` | a `BodiesState` of each body's `position`, `orientation`, `linear_velocity`, `angular_velocity`, `awake` and `found` as Jolt holds it now: inside a callback, the tick as it stands; elsewhere under the world lock's shared side |
| `wait_tick(tick, timeout=None)` | True once a snapshot of `tick` or later is published; False on timeout or when the world stops |
| `drain_events()` | every event record since the last drain, as an array of `EVENT_DTYPE` |
| `drain_applied()` | (k, 2) uint64 `(seq, tick)`: each batch applied since the last drain and the index of the tick that applied it |
| `tick` | ticks completed |
| `lockstep` | True unless `start()` is free-running the world |
| `error` | the exception that stopped the world (a callback's, or a `RuntimeError` naming what failed on the world thread), or None |
| `last_rejection` | why the world thread could not apply the last batch that raised while it applied it (out of memory), or None; the batch is counted in `rejected` and the world steps on |
| `submitted_seq` | the sequence number of the last batch submitted |
| `stats()` | counters (`ticks`, `dropped_ticks`, `events_dropped`, `events_held` in the ring until the next drain, `rejected`, `pool`, `bodies` in the world, `queued_bodies` submitted and not yet applied, `shapes` registered, `broadphase_optimizations`, `idle`, `running`) `characters`, `vehicles`, `ragdolls`, `soft_bodies`, `queries` that waited for the world lock, and mean milliseconds a tick since `reset_stats()` (`apply_ms`, `characters_ms` (kinematic targets, characters, vehicles, ragdolls and soft bodies before the step), `ragdolls_ms` (the ragdolls' pose targets and motors before the step, a part of `characters_ms`), `soft_bodies_ms` (the soft bodies' pins and wind before the step and their vertex copies after it), `step_ms`, `events_ms`, `publish_ms`, `callbacks_ms`, `tick_ms`), `quiet_ticks` counted without stepping since `reset_stats()`, with `query_wait_ms` and `query_wait_max_ms`, the mean and longest wait of a query for the world lock |

## Commands

Every command takes arrays for many bodies, may be called from any thread
(a callback on the world thread included), and returns the batch's sequence
number. Batches are applied in sequence order at the start of a tick, before
the step: the first tick whose index (ticks completed before it) is at least
the batch's `at_tick`, 0 by default. A command naming a uid that is not a body
is skipped; one the body cannot take (a velocity for a static body) is
skipped and counted in `stats()['rejected']`. A value the world stores as a
32-bit float (velocities, forces, gravity, masses, vehicle, ragdoll and soft
body settings) raises `ValueError` where it is finite as a double and not as
a float; a uid, group, mask, filter or material index outside its integer
type raises `ValueError` and is never wrapped. A body whose position or
velocity stops being finite in a step (a gravity factor of 1e38, say) stops
the world with a `WorldError` naming the body, before the NaN reaches the
bodies it meets.

| Call | Change |
|---|---|
| `add_bodies(batch)` | add a `BodyBatch`; returns `(uids, seq)`, the uids assigned at once |
| `remove_bodies(uids)` | remove bodies; their pairs end with `EndReason.REMOVED` |
| `swap_bodies(remove, batch)` | remove bodies and add a `BodyBatch` in one batch, so in one tick; returns `(uids, seq)` |
| `set_transforms(uids, positions, orientations=(0, 0, 0, 1))` | teleport; waking moving bodies |
| `set_velocities(uids, linear=None, angular=None)` | m/s and rad/s; None leaves one as it is |
| `add_forces(uids, forces, points=None)` | N at the centre of mass, or at world `points` (n, 3), acting over the next step |
| `add_impulses(uids, impulses, points=None)` | N s, changing the velocity at once; at world `points` they turn the body too |
| `add_torques(uids, torques)` | N m, over the next step |
| `set_gravity(gravity)` | m/s2; wakes every dynamic body |
| `activate(uids)` | wake bodies |
| `set_motion_types(uids, motion)` | `MotionType` per body; a static body becomes moving only if added with `BodyFlag.MOVABLE` |
| `set_materials(uids, friction, restitution)` | new coefficients |
| `set_body_materials(uids, material)` | material indices (below) |
| `set_body_filters(uids, filters)` | collision filter indices (below) |
| `set_drag(uids, drag)` | quadratic drag coefficients, 1/m |
| `set_motion(uids, mass=None, linear_damping=None, angular_damping=None, gravity_factor=None, inertia=None)` | moving bodies' motion properties; None, or NaN in an array, keeps a value; a new mass scales the inertia with it; `inertia` is a 3x3 tensor in kg m2 about the centre of mass in the body's frame, one for every body or (n, 3, 3), applied after the mass, its diagonal above 0; changed dynamic bodies wake |
| `set_flags(uids, flags, on)` | set or clear `BodyFlag.REPORT`, `NO_SLEEP` and `FILTER_CONTACTS`, the flags that change after a body is added |
| `restore_state(data)` | read a `save_state()` back into the bodies and joints (below) |

Vectors are (n, 3) arrays, or one (3,) vector for every body; quaternions are
`(x, y, z, w)`. Values that are not finite, zero quaternions, and arrays
whose lengths differ raise `ValueError` when the batch is submitted.

### Uids and `applied_seq`

A body's uid is assigned when its add batch is submitted, counting from 1,
and is never reused within a world; 0 names no body. The body exists from the
tick that applies the batch. A snapshot's `applied_seq` is the highest
sequence number at and below which every batch is applied, so
`snapshot.includes(seq)` tells a reader whether its own write is in the poses
it is looking at. A batch held back by `at_tick` holds `applied_seq` below it
until its tick.

### `BodyBatch`

Bodies to add, as arrays. Every per-body field is one value for the batch or
an array with one entry (or row) per body.

| Field | Default | Meaning |
|---|---|---|
| `positions` | | (n, 3) body origins; fixes the batch's length |
| `shape` | None | a `Shape` or shape id for the batch, or a sequence of them per body; None takes `shape_kind` |
| `shape_kind`, `shape_params` | box, (1, 1, 1, 0) | a primitive `ShapeKind` and its four parameters |
| `scale` | (1, 1, 1) | scales the shape in its own frame |
| `shape_offset`, `shape_rotation` | (0, 0, 0), identity | place the scaled shape in the body's frame, m and a quaternion |
| `center_of_mass` | (0, 0, 0) | moves the centre of mass from where the shape puts it, m |
| `motion` | `DYNAMIC` | `MotionType` |
| `orientations` | identity | (n, 4) unit quaternions |
| `mass` | 1.0 | kg; 0 takes the mass from `density` |
| `density` | 0.0 | kg/m3, used where `mass` is 0 |
| `friction`, `restitution` | 0.5, 0.0 | Jolt combines a pair's as the geometric mean of the frictions and the larger restitution |
| `linear_damping`, `angular_damping` | None | 1/s; None takes the `WorldSettings` default |
| `gravity_factor` | 1.0 | multiplies gravity for the body |
| `linear_velocity`, `angular_velocity` | 0 | initial m/s and rad/s |
| `group`, `mask` | 1, 0xFFFFFFFF | 32-bit collision sets: two bodies collide when each one's group shares a bit with the other's mask |
| `filter` | 0 | collision filter index into the world's filter table |
| `material` | 0 | row of the world's material table; 0 is the body's own `friction` and `restitution` |
| `quadratic_drag` | 0.0 | 1/m: a force of `-k m |v| v` |
| `flags` | `NONE` | `BodyFlag` bits |

Builders, each returning a batch, with every other field as a keyword:
`BodyBatch.boxes(positions, size=(x, y, z))`, `spheres(positions, radius)`,
`capsules(positions, radius, height)`, `cylinders(positions, radius, height)`,
`plane(position=(0, 0, 0), half_extent=0.0)`, `of_shape(positions, shape)`,
`hulls(positions, points)` with `points` (m, 3) for every body or (n, m, 3)
one cloud per body, `meshes(positions, points, triangles)` (static by
default) and `heightfield(heights, origin, spacing, ...)`, one static body at
`origin`. `heightfield_tiles(heights, origin, spacing, tile=64)` returns one
static body a tile.

A batch is checked when it is submitted: a bad shape, a shape id that names
nothing, a scale the shape cannot take (a sphere and a capsule take a uniform
scale, a cylinder the same scale in x and z), and a dynamic body of a mesh or
height field raise `ValueError` naming the row. Shapes are resolved then too,
so the world thread only creates the bodies.

`add_bodies` adds a batch with Jolt's `AddBodiesPrepare` and
`AddBodiesFinalize`, in the tick that applies it. Measured on this machine
(benchmarks/statics.py), 10,000 static posts of one shared shape are created
and added in 3.5 to 4.6 ms of that tick (4.2 to 5.8 ms for the whole tick,
under the 8.3 ms of a 120 Hz tick), and `add_bodies` itself costs the
caller 3.7 to 4.5 ms.

### Shapes

`ShapeKind`, with sizes as in the OMI physics shape model:

| Kind | Parameters | Notes |
|---|---|---|
| `BOX` | edge lengths x, y, z | |
| `SPHERE` | radius | |
| `CAPSULE` | radius, height of the cylinder between the hemisphere centres | along the body's Y axis |
| `CYLINDER` | radius, total height | along the body's Y axis |
| `PLANE` | half extent of its bounds, 0 for 1 km | through the body's origin, normal the body's +Y; static only |
| `CONVEX` | a `Shape` | the hull of a point cloud |
| `MESH` | a `Shape` | static and kinematic bodies only |
| `HEIGHTFIELD` | a `Shape` | static and kinematic bodies only |
| `COMPOUND` | a `Shape` | children of any kind but a height field |

Equal primitive kinds and parameters share one Jolt shape. Boxes and
cylinders are rounded by a convex radius of 5 cm or half the smallest
dimension, whichever is smaller.

#### `Shape` and shape ids

```python
post = world.add_shape(pj.Shape.box((0.4, 2.0, 0.4)))        # an int
world.add_bodies(pj.BodyBatch.of_shape(positions, post, motion=pj.MotionType.STATIC))
rock = pj.Shape.convex(points)                              # built by the first add_bodies naming it
world.add_bodies(pj.BodyBatch.of_shape(more_positions, rock, mass=0.0, density=2600.0))
```

A `Shape` describes a shape; `World.add_shape(shape)` builds it on the
calling thread, with the GIL released, and returns the world's id for it.
Every body naming the id shares the one Jolt shape, so 10,000 posts hold one.
The same `Shape` object returns the same id every time, and a batch naming a
`Shape` registers it through `add_shape`. `remove_shape(id)` forgets the id;
bodies made with it keep the shape. `shape_info(id)` returns a `ShapeInfo`:
`kind`, `volume` (m3, 0 for a mesh, height field or plane), `center_of_mass`,
`bounds_min` and `bounds_max` in the shape's frame, and `dynamic`.

| Constructor | Builds |
|---|---|
| `Shape.box(size, rounding=None)`, `sphere(radius)`, `capsule(radius, height)`, `cylinder(radius, height)` | a primitive, as a shape a compound or `of_shape` can name; a box's `rounding` (m) rounds its edges, by default 5 cm or half its smallest half extent |
| `Shape.convex(points, max_convex_radius=0.05, hull_tolerance=1e-3)` | Jolt's `ConvexHullShape` of (m, 3) points: edges rounded by up to `max_convex_radius` (less for a small hull), points allowed `hull_tolerance` m outside. Fewer than 4 points, or points on a plane, a line or one point, raise `ValueError` |
| `Shape.mesh(points, triangles, materials=None, ids=None)` | Jolt's `MeshShape` of (m, 3) vertices and (t, 3) indices. Degenerate and repeated triangles are dropped; an index past the vertices raises `ValueError`. Edges between faces meeting at under 5 degrees are inactive (Jolt's active edges). A triangle collides on its front face, from which its vertices run anticlockwise; a wall met from both sides lists each triangle in both windings |
| `Shape.heightfield(heights, origin, spacing, holes=None, block_size=4, bits_per_sample=8, height_range=None)` | Jolt's `HeightFieldShape`, below |
| `Shape.compound(parts, mutable=False)` | Jolt's `StaticCompoundShape`, or with `mutable` its `MutableCompoundShape` (quicker to build, slower to collide). Each part is `(shape, position)` or `(shape, position, orientation)`, the shape a `Shape` or an id |

Jolt computes no mass properties for a mesh, a height field or a plane, so
it gives them no dynamic body: such a batch row raises `ValueError`, as does
a compound holding one. A kinematic body of one takes `mass` (1 kg when 0) as
a unit box's mass and inertia; a kinematic body meets no solver, so they do
not move it.

A dynamic body's mass is `mass`, or `density` times the shape's volume where
`mass` is 0, for every dynamic kind; its inertia comes from the shape, scaled
to that mass.

#### Height fields

`heights[j, i]` is the height at `x = origin[0] + i * spacing[0]` and `z =
origin[2] + j * spacing[1]`, added to `origin[1]`, in the body's frame;
`spacing` is one number or `(dx, dz)`. Jolt takes a square grid whose side is
a multiple of `block_size` and at least two blocks; any other grid is padded on
its +x and +z sides with samples that collide with nothing, so the padding
adds memory (one byte a sample at 8 bits) and no surface.
`heightfield_side(rows, columns, block_size)` gives the side. A NaN height,
or a true entry of `holes` (a bool array of the grid's shape), is a hole: the
cells touching that sample collide with nothing. Jolt stores each block of
`block_size` by `block_size` samples in `bits_per_sample` bits (1 to 16)
relative to the block's lowest and highest sample: 8 bits keep a sample within
1/255 of its block's range. `height_range=(low, high)` widens the range Jolt
quantises over, for edits that will reach past the samples.

`heightfield_tiles(heights, origin, spacing, tile=64, holes=None, **values)`
cuts a large grid into one static body a tile of `tile` by `tile` cells.
Neighbouring tiles share their common edge's samples, so their triangles meet
exactly and a body crossing a seam meets the same surface as on one field
(tests/test_shapes.py rolls a ball across seams and compares). Each body sits
at its tile's corner, so a tile's local coordinates stay small anywhere in a
kilometre-scale world.

Jolt tests each triangle of a mesh or height field apart from its
neighbours, so a body crossing an inner edge can meet the edge: a ball rolling
at 6 m/s rides up to about 5 mm off it, and one landing beside an inner edge
can be pushed sideways. A static body whose shape holds a mesh or a height
field has Jolt's enhanced internal edge removal, which merges those contacts
with the faces beside them; `BodyFlag.NO_SMOOTH_EDGES` turns it off for such a
body, and `BodyFlag.SMOOTH_EDGES` turns it on for any other (it then applies to
every pair the body is in). On a moving body it also works across the seams
between static bodies laid flush: the pair's contacts are made together with
those of every static body beside the moving one that it collides with, and an
edge lying on a face it rests on, of whichever body, is no contact; an edge
contact that remains on an edge shared with a face of another body in the same
plane takes that plane's normal. A ball of 0.5 m crossing 4 m static boxes laid
flush is kicked up at their seams at up to 3.4 m/s at 28 m/s without the flag,
and at up to 1.1 m/s with it (1.1 and 0.6 m/s at 14 m/s). Without the flag on the moving body the ball rides up to
6 mm at a seam between two height field tile bodies. Its cost did not show in the benchmarks: 5,000 boxes on the 20,000
triangle mesh took 3.12 ms a frame with it and 3.24 ms without on 1 job
thread, and 300 characters on a height field 1.49 and 1.51 ms.

#### Decorations

A body's `scale`, `shape_offset`, `shape_rotation` and `center_of_mass`
decorate its shape with Jolt's `ScaledShape`, `RotatedTranslatedShape` and
`OffsetCenterOfMassShape`, in that order: the shape is scaled, then placed in
the body's frame, then its centre of mass moved. Bodies with the same shape
and decoration share one decorated shape. The body's origin stays where
`positions` puts it; a snapshot's `position` is the origin, not the centre of
mass.

### `BodyFlag`

| Flag | Meaning |
|---|---|
| `SENSOR` | reports overlaps as trigger events; no contact response |
| `NO_SLEEP` | never falls asleep |
| `CONTINUOUS` | continuous collision detection (Jolt's linear cast), for small fast bodies |
| `REPORT` | persist events, once a tick, for every pair the body touches |
| `MOVABLE` | a static body a later `set_motion_types` may make kinematic or dynamic; needs a mass or density |
| `SMOOTH_EDGES` | Jolt's enhanced internal edge removal: contacts with the inner edges of a mesh, height field or compound merge with the faces beside them; a static mesh or height field has it without the flag; on a moving body it also removes the seams between static bodies laid flush |
| `NO_SMOOTH_EDGES` | a static mesh or height field without enhanced internal edge removal |
| `FILTER_CONTACTS` | the world's contact filter is asked about pairs where both bodies carry it |

## Snapshots

A `Snapshot` holds the state of every body after one tick. Its arrays are
read-only NumPy views of a buffer in the world's pool; they are not copies.
A buffer is reused only when no `Snapshot` and no view of one refers to it,
so a view stays valid, with the values of its tick, for as long as it is
held. Holding snapshots grows the pool and never blocks the world thread.

| Attribute | Type | Meaning |
|---|---|---|
| `tick`, `time` | int, float | ticks completed and `tick / tick_hz` seconds |
| `applied_seq` | int | see above |
| `count` | int | bodies, and rows |
| `uid` | (n,) uint32 | the body of each row |
| `position` | (n, 3) float64 | body origins, m |
| `orientation` | (n, 4) float32 | unit quaternions `(x, y, z, w)` |
| `linear_velocity` | (n, 3) float32 | m/s |
| `angular_velocity` | (n, 3) float32 | rad/s |
| `awake` | (n,) bool | false for a sleeping or static body |
| `motion_type` | (n,) uint8 | `MotionType` |
| `changed` | (k,) uint32 | ascending rows this tick wrote; every other row holds what it held at the tick before |
| `rows` | dict | uid to row |
| `rows_of(uids)` | (m,) int64 | the row of each uid, -1 where it is not a body |
| `includes(seq)` | bool | `applied_seq >= seq` |

Rows of one snapshot are not the rows of another: removing a body moves the
last row into its place, and both rows are in `changed`. A body that falls
asleep is in `changed` on that tick, with its final pose.

## Events

`EVENT_DTYPE` is a 72-byte record:

| Field | Type | Meaning |
|---|---|---|
| `kind` | uint8 | `EventKind`: `CONTACT`, `TRIGGER`, `STATE`, `CHARACTER`, `JOINT` or `VEHICLE` |
| `phase` | uint8 | `EventPhase`: `BEGIN`, `PERSIST`, `END`, `SLEEP` or `WAKE`; for a character also `LAND`, `JUMP`, `SLIDE`, `DOWN` and `UP`; for a joint `END` and `BREAK`; for a vehicle `IMPACT` and `CRASH` |
| `reason` | uint16 | `EndReason` of an end: `NONE`, or `REMOVED` when a body of the pair was removed |
| `tick` | uint32 | the tick the event happened in: the tick of the first snapshot that shows it |
| `a`, `b` | uint32 | uids: for a contact the lower uid first; for a trigger the sensor first; for sleep and wake the body, and 0; for a character the character, and the body or character it touched or landed on; for a joint the joint id, and for a break the joint's body |
| `point` | 3 float64 | world contact point, m (begin and persist) |
| `normal` | 3 float32 | unit vector from `a` to `b` (contact begin and persist) |
| `approach` | float32 | closing speed along the normal at the point, m/s, positive when approaching |
| `impulse` | float32 | estimated normal impulse, N s, in contact begin and persist records; in a character begin, the momentum of a dynamic body striking it (N s); a joint break's torque, N m; 0 in every other record |
| `material_a`, `material_b` | uint16 | contact begin and persist: the material index of each side, a mesh triangle's own where it has one |
| `depth` | float32 | contact begin and persist: how far the two bodies overlap, m |
| `reserved` | uint32 | 0 |

A record's simulation time is `tick / tick_hz`; `event_times(events, tick_hz)`
computes it. The impulse is Jolt's `EstimateCollisionResponse` for the two
bodies alone, at the start of the record's tick: on a persist record, the
impulse that stops the bodies closing in that tick (a resting body's weight
times the tick). Jolt calls its contact listeners before its solver runs, so no
record carries the impulse the solver applied. The point is float64, as body positions are, so it keeps the
same precision at any distance from the origin.

The world folds Jolt's per-sub-shape reports into one record per body pair:
one begin when the pair first touches and one end when it parts, however
many children of a compound or triangles of a mesh touch. Persist
records come once a tick for pairs where either body has `BodyFlag.REPORT`.
A sensor's begin and end are a body entering and leaving it.

Sleep. Jolt drops a sleeping body's contacts and adds them again when it
wakes. The world holds a pair whose contacts go while none of its bodies is
awake as touching: no end is recorded when it falls asleep and no begin when
it wakes still touching. A body that wakes and moves away ends its pairs in
the tick their contacts fail to return. A body asleep inside a sensor stays
inside it.

Sensors. Jolt tests a pair only where one of its bodies is awake, so on its
own a static sensor added over a sleeping body would not see it. Adding a
sensor, or moving one with `set_transforms`, wakes the moving bodies whose
bounds meet the sensor's (one broad-phase query); the next step reports
those inside, and those at rest fall asleep again and are held as inside. A
kinematic sensor also meets static and kinematic bodies (Jolt's
`mCollideKinematicVsNonDynamic`). Two static bodies never meet, so a static
sensor does not report a static body.

Continuous detection. A body with `BodyFlag.CONTINUOUS` is cast along its
motion each step and stops at the first surface it would pass, a thin box
or a mesh triangle (tests/test_events.py, a 5 cm ball at 400 m/s).

Each tick's records are sorted, so a run makes the same records in the same
order with any number of job threads.

The ring holds `event_capacity` records. When it is full, persist records go
first, oldest first, then sleep and wake records; begin and end records are
never dropped, so that each pair's state stays balanced, and a ring holding
only those grows past its capacity until it is drained. Dropped records are
counted in `stats()['events_dropped']`. `stats()['events_held']` is how many it holds now.

Character records (`kind` `CHARACTER`, `a` the character's id):

| Phase | Meaning |
|---|---|
| `BEGIN`, `END` | the capsule began or stopped touching body or character `b` (within 2 cm); a begin has the point, the normal from the character to the body, the closing speed as `approach` (against the character's velocity of the tick before) and, for a dynamic body, the body's mass times that speed as `impulse` |
| `LAND` | it landed on `b` (a body's uid or another character's id); `approach` is the landing speed, m/s |
| `JUMP` | it jumped; `approach` is the launch speed, m/s |
| `SLIDE`, `DOWN`, `UP` | it began a slide, was knocked down, got up |

They are never dropped from a full ring, as begin and end records are not.

## Queries

```python
hit = world.raycast((0, 10, 0), (0, -1, 0), 100.0)        # RayHit or None
hits = world.raycast_batch(origins, directions, 100.0)     # RayHits of (n,) and (n, 3) arrays
first = world.sweep_sphere(0.25, start, end)               # ShapeHit or None
every = world.sweep_capsule(0.3, 1.2, start, end, all_hits=True)
touching = world.overlap(pj.Shape.box((2, 2, 2)), (0, 1, 0))   # ascending uids
seen = world.raycast(eye, ahead, 50.0, filter=pj.QueryFilter(exclude=[me], mask=0b0110))
```

A query runs on the calling thread and reads the world as the last completed
tick left it: a body added by a batch is met from the tick that applies it. It
takes the world lock's shared side, so it waits while the world thread holds
the lock exclusively (applying batches, moving characters and stepping) and
runs beside the rest of the tick and beside other queries. The GIL is released
while Jolt answers. On the world thread, inside a callback, a query takes no
lock: the tick already holds the shared side.

| Call | Returns |
|---|---|
| `raycast(origin, direction, max_distance=1000, filter=None)` | the first body met, as `RayHit(uid, distance, fraction, point, normal, sub_shape)`, or None; `direction` need not be a unit vector and may not be zero |
| `raycast_batch(origins, directions, max_distance=1000, filter=None)` | `RayHits`: `hit` (n,) bool, `uid` (n,) uint32 (0 for a miss), `fraction` (n,) float32 of `max_distance`, `distance` (n,) float64 (inf for a miss), `point` (n, 3) float64, `normal` (n, 3) float32, `sub_shape` (n,) uint32; a zero direction is a miss |
| `sweep(shape, start, end, orientation=(0, 0, 0, 1), filter=None, all_hits=False)` | the first `ShapeHit(uid, fraction, distance, point, normal, depth, sub_shape)` along the way, or None; with `all_hits` a list of every body's first hit, in order |
| `sweep_sphere(radius, start, end, ...)`, `sweep_capsule(radius, height, start, end, ...)`, `sweep_box(size, start, end, ...)` | `sweep` with that primitive; a capsule's `height` is between its hemisphere centres |
| `overlap(shape, position, orientation=(0, 0, 0, 1), filter=None)` | (k,) uint32 ascending uids of the bodies the shape touches |
| `triangle_of(uid, sub_shape)` | the number of the mesh triangle a hit's `sub_shape` names on body `uid` (its row in the mesh's `triangles`, or the `ids` `Shape.mesh` was given), through compounds and decorations; -1 for any other part |

`shape` is a `Shape` (a primitive, or any kind `add_shape` builds) or a shape
id; a plane is neither swept nor overlapped. A ray's `point` and `normal` are
the surface's; a sweep's `point` is on the body hit and its `normal` points
from the body towards the swept shape; `depth` is positive where the shape
overlaps the body at its start. `sub_shape` is Jolt's sub-shape id of the part
hit: a mesh triangle, a compound child. Convex shapes are solid to rays, so a
ray starting inside a box meets it at fraction 0; triangles are met from their
front face only.

`QueryFilter(group, mask, exclude, triggers, characters)`: a body is met when
its group shares a bit with `mask` and its mask shares a bit with `group`, as
two bodies meet each other (both default to every bit); bodies whose uid is in
`exclude` are passed over; sensors are met only with `triggers=True`;
character proxies are met unless `characters=False`.

Measured (`benchmarks/characters.py`, gcc 13.3, CPython 3.14, a 32-core
machine with a load of 3.5 from other work): 100,000 scattered rays down on
the 20,000 triangle ground among 10,000 posts in one `raycast_batch`, 55 ms,
0.55 us a ray; one `raycast` from Python, 2.3 us. Single rays from the caller
while the world free-runs at 120 Hz with 5,000 active boxes, 8 job threads:
p50 3.2 us, p99 1.1 ms, max 2.8 ms; the wait for the lock is the rest of an
`Update` in progress (1 job thread: p99 2.5 ms, max 9.1 ms).

## Characters

```python
caps = pj.CharacterCapabilities(walkSpeed=3.0, jumpHeight=1.5, stepHeight=0.35, maxSlope=50.0)
hero = world.add_character((0.0, 1.0, 0.0), caps, gravity=42.0)   # capsule centre
hero.set_move((1.0, 0.0, 0.0), 'run')
hero.jump()
world.run_ticks(60)
hero.position, hero.velocity, hero.state, hero.grounded, hero.ground_uid
rows = world.latest().characters      # every character as read-only views
```

A character is omi_physics' controller over Jolt's `CharacterVirtual`: a
capsule moved by collision queries rather than by the solver. The world thread
moves every character once a tick, before the step and under the exclusive
world lock, with the inputs of the batches that tick applied. An input reaches
its character at the next tick, as every command does.

`add_character(position, capabilities=None, gravity=None, pass_group=0,
group=1, mask=ALL_GROUPS, ignore=(), proxy=True)` returns a `Character` whose
`id` is a uid from the bodies' sequence. `gravity` is the character's own,
m/s2 downward, by default the length of the world's. Characters of one
`pass_group` (other than 0) pass through each other; others block and push
each other. `group` and `mask` choose the bodies it meets, as for bodies;
`ignore` names bodies it passes through. With `proxy` a kinematic capsule
stands in the world under the character's id, moved over each step to where
the character is: rays, sweeps and overlaps meet it
(`QueryFilter(characters=False)` passes it over) and sensors, static ones
included, make trigger records with it as their `b`. Its collision layer meets
sensors and nothing else, so no body is solved against it; the character meets
bodies itself. `remove_characters(ids)` takes characters and their proxies
out.

`CharacterCapabilities` keeps omi_physics' field names and meanings:

| Field | Default | Meaning |
|---|---|---|
| `walkSpeed`, `runSpeed`, `sprintSpeed`, `crouchSpeed` | 3, 6, 9, 1.5 | m/s of each `MoveMode` |
| `canSprint`, `canCrouch`, `canJump`, `canFly` | True | what the character may do |
| `flySpeed`, `swimSpeed`, `swimDrag` | 8, 2, 4 | m/s, m/s, 1/s |
| `jumpHeight` | 1.2 | m; the launch speed is `sqrt(2 g jumpHeight)` |
| `jumpFollowsSlope` | False | a jump keeps the vertical speed of the run along the ground |
| `stepHeight` | 0.35 | m climbed as a stair, and dropped while staying on the floor |
| `maxSlope` | 50 | degrees; steeper ground is a wall, and slides the character down |
| `eyeHeight`, `standHeight`, `crouchHeight`, `radius` | 1.6, 1.8, 1.0, 0.3 | m; heights are feet to crown |
| `airControl` | 0.4 | the share of the input heard off the ground |
| `pushFriction` | 6 | 1/s, bleeds carried speed (knockback, slide, jump pad) on the ground |
| `terminalVelocity` | 55 | m/s, the fastest fall; 0 for none |
| `suppressOverVoid` | False | hover rather than fall where no ray down meets anything |
| `coyoteTime`, `jumpBuffer` | 0.12, 0.12 | s a jump is forgiven after walking off an edge, and remembered before landing |
| `mass` | 80 | kg; divides a knockback, presses on what it stands on |
| `pushMass` | 200 | kg, the heaviest dynamic body moved by walking into it; a heavier one blocks |
| `pushSpeed` | None | m/s, the fastest a body is pushed; None is `walkSpeed` |
| `pushForce` | 1000 | N, the most a character pushes with |
| `acceleration` | inf | m/s2 towards the input speed (times `airControl` in the air); inf reaches it at once |
| `stickToFloor` | True | a walker stays on the ground over drops up to `stepHeight` |
| `smoothEdges` | True | Jolt's enhanced internal edge removal for the capsule |

The vertical motion is the controller's: a constant gravity integrated
exactly, `v dt - g dt^2 / 2` a tick, so every sample of a jump lies on its
parabola and the apex is `jumpHeight` above the launch at any tick rate
(tested to 1e-6 m at 60, 120 and 240 Hz with gravity 42). The horizontal input
is turned along walkable ground at the same speed, so a slope is climbed at the
pace of the flat. A character on walkable ground that touches only static
bodies falls along the ground's normal rather than straight down, so one at
rest, standing or knocked down, stays where it is on a slope; touching a body
or a character that moves, it falls straight down and is slid aside by what
presses on it. A grounded character rides what it stands on: a kinematic
platform's velocity at its feet carries it, up and sideways.

`Character` methods queue inputs and return the batch's sequence number; each
takes `at_tick`:

| Method | Input |
|---|---|
| `set_move(direction, mode='walk', speed=None)` | move along `direction` flattened to the ground, at the `MoveMode` (`'walk'`, `'run'`, `'sprint'`, `'crouch'`) speed or `speed` m/s |
| `jump()` | jump if grounded or within `coyoteTime`; else held for `jumpBuffer` |
| `slide(duration, rise=0.3, steer=False)` | from the ground: crouch height, keeping its speed (or, with `steer`, moved by the input), rising to standing over the last `rise` s and waiting under a ceiling |
| `extend_slide(duration)`, `end_slide()` | lengthen a slide from now; end it |
| `knock_down()`, `get_up(start_speed=None)` | lie low, input ignored; stand (crouched under a low ceiling), carrying `start_speed` along the last heading |
| `knockback(impulse)` | change the velocity by `impulse / mass`; an upward part leaves the ground |
| `apply_impulse(velocity)` | launch at `velocity`, replacing the motion |
| `set_crouch(crouch)`, `set_fly(flying)`, `set_fly_move(direction)`, `set_swim(swimming, buoyancy=0.9)` | crouch or stand where there is room; fly (no collision, no gravity); swim (collides as walking, moves as flying, pulled by the gravity `buoyancy` leaves) |
| `teleport(position, search=4.0)` | put the centre at `position` at rest, seated on a floor within `search` m below |
| `set_gravity(g)`, `set_pass_group(group)`, `set_ignore(uids)`, `remove()` | |

`world.character_ops(ids, ops, codes, values)` queues inputs for many
characters as arrays (`CharacterOp` codes), as the methods do one at a time.

Its properties read the newest snapshot: `position` (the capsule's centre),
`feet`, `velocity` (the distance moved over the last tick, per second),
`state`, `grounded`, `ground_uid`, `ground_normal`, `slide_progress`,
`landing` (the landing speed in the newest snapshot's tick, else None) and
`height`. `CharacterState` is the first that holds of `FLYING`, `SWIMMING`,
`DOWN`, `AIRBORNE`, `SLIDING`, `CROUCHING`, `STANDING`.

`Snapshot.characters` is a `CharacterRows`, one row a character in ascending
id: `id` (n,) uint32, `position` (n, 3) float64, `velocity` (n, 3) float32,
`state` (n,) uint8, `grounded` (n,) bool, `ground_uid` (n,) uint32,
`ground_normal` (n, 3) float32, `slide_progress` (n,) float32, `landing_speed`
(n,) float32 and `landed_tick` (n,) uint32 of the last landing, `height`
(n,) float32, `previous` (n, 3) float64 (the centre at the tick before, for
drawing between ticks), `eye` (n, 3) float64 (`eyeHeight` above the feet,
scaled by the capsule's height over `standHeight`), `ground_material` (n,)
uint16 (the material index underfoot, a mesh triangle's own where it has one)
and `stuck` (n,) bool (the last teleport left the capsule more than 1 cm
inside a body), all read-only views of the snapshot's buffer.

Pushing. Walking into a dynamic body no heavier than `pushMass` pushes it with
Jolt's character impulse, at most `pushForce`; a heavier body blocks. A
character that walks into another shares the closing speed by their masses:
a 160 kg character walks a 40 kg one ahead of it.

Islands. Characters whose reach this tick (where their move, stairs, floor
snap and contacts can take them) shares no other character and no dynamic body
are moved by separate jobs on Jolt's job system; the rest are moved in id
order within their island. The results are the same with any number of job
threads, and a free-running run replays bit for bit in lockstep with 1 or 8
job threads (tests/test_character_lockstep.py).

Measured (`benchmarks/characters.py`, ms per 60 Hz frame, with proxies): 300
characters running on a 256 x 256 height field, 1.48 ms on 1 job thread and
0.60 ms on 8 at one tick a frame, 2.62 and 1.10 ms at 120 Hz (two ticks a
frame); 30 characters among 10,000 posts on the 20,000 triangle ground, 0.17
and 0.09 ms (0.30 and 0.19 at 120 Hz). The 300 proxies add 0.08 ms a tick to
`PhysicsSystem::Update` on 1 job thread.

Bodies moving into a character: a kinematic body (a door, a lift wall) pushes
it at the body's speed. A dynamic body is met by Jolt's character contact: the
capsule is moved out of it as it comes in, and the character's push slows a
body it could push; the character keeps none of the body's momentum, so a
thrown crate shoves it while they overlap rather than knocking it back at a
share of the crate's speed. The character's `BEGIN` record names the body and
its closing speed, and the body's momentum at that speed as `impulse`, for a
caller that applies `knockback`.

A character inside gravity zones falls by the vertical part of the gravity
the zones leave from its own, at its capsule's centre.

## Vehicles

```python
chassis = world.add_bodies(pj.BodyBatch.boxes([(0, 0.62, 0)], size=(1.85, 0.62, 4.2), mass=1180.0))[0][0]
car = world.add_vehicle(chassis, pj.VehicleSettings(wheels=(
    pj.WheelSettings((-0.85, 0.0, -1.3), steered=True, driven=False),
    pj.WheelSettings((0.85, 0.0, -1.3), steered=True, driven=False),
    pj.WheelSettings((-0.85, 0.0, 1.3), handbrake_torque=4000.0),
    pj.WheelSettings((0.85, 0.0, 1.3), handbrake_torque=4000.0))))
car.set_input(throttle=1.0, steer=0.2)       # held from the next tick until changed
world.vehicle_inputs(ids, rows)              # (n, 4) throttle, steer, brake, handbrake
rows = world.latest().vehicles               # VehicleRows; rows.wheels is WheelRows
```

A vehicle is Jolt's `VehicleConstraint` over a dynamic rigid chassis body,
with a `WheeledVehicleController` (`VehicleKind.WHEELED`),
`MotorcycleController` (`MOTORCYCLE`) or `TrackedVehicleController`
(`TRACKED`). Vehicle ids are assigned at submission, from 1, and never reused.
A vehicle whose chassis is not a dynamic rigid body when its batch is applied
(a soft body is dynamic and not rigid), whose chassis already carries a
vehicle, or a tracked vehicle with a track holding no wheel, is skipped and
counted in `stats()['rejected']`. Removing the chassis removes the vehicle,
whether `remove_bodies` or `remove_ragdolls` (for a bone) removes it;
`remove_vehicles` leaves the chassis. `stats()['vehicles']` counts the
vehicles in the world. `drag`, `downforce` and `rolling_resistance` never wake
a sleeping chassis, so a parked car falls asleep like any body.

Each tick, before `PhysicsSystem::Update`, the world thread turns each
vehicle's inputs into the controller's: the steering lock falls with speed and
the steer angle moves at `steer_rate`; an idle vehicle that comes down to
`holding_speed` brakes with `holding` until it is driven again; `drag`,
`downforce` and `rolling_resistance` act on the chassis for the step. Jolt
casts the wheels (one step listener a vehicle, on its job system) and solves
suspension, tyres and drivetrain with the other constraints, so the result
does not depend on the job count.

Inputs: `throttle` -1 to 1 (an automatic gearbox selects reverse below 0),
`steer` -1 to 1 (positive turns left, a positive yaw about +Y), `brake` and
`handbrake` 0 to 1. A tracked vehicle steers by slowing the inside track,
which runs backwards past half a turn.

`VehicleSettings` is plain data in SI units and the chassis's own frame:

| Field | Meaning |
|---|---|
| `wheels` | `WheelSettings`: `position` (suspension mount), `radius`, `width`, `suspension_min`, `suspension_max`, `preload` (m), `stiffness` (N/m), `damping` (N s/m), `steered`, `driven`, `brake_torque`, `handbrake_torque` (N m), `inertia` (kg m2), `angular_damping`, `grip` (multiplies the friction curves), `track` (0 left, 1 right, -1 none) |
| `kind` | `VehicleKind` |
| `engine` | `Engine`: `max_torque` (N m), `min_rpm`, `max_rpm`, `torque_curve` ((rpm / max_rpm, fraction) points), `inertia`, `damping` |
| `transmission` | `Transmission`: `automatic`, `gears`, `reverse_gears`, `switch_time`, `clutch_release_time`, `switch_latency` (s), `shift_up_rpm`, `shift_down_rpm`, `clutch_strength` |
| `differentials` | `Differential(left, right, ratio, split, limited_slip, torque_ratio)`; None builds one per driven axle with `differential_ratio` and `differential_limited_slip` (inf: open) |
| `anti_roll_bars` | `AntiRollBar(left, right, stiffness)` (N/m) |
| `tracks` | (left, right) `VehicleTrack(driven_wheel, inertia, damping, brake_torque, ratio)` |
| `lean` | `Lean(max_angle, spring, damping, integration, integration_decay, smoothing)`: a motorcycle's |
| `forward`, `up` | the chassis's axes; default -Z and +Y |
| `cast`, `cast_radius`, `max_slope` | `WheelCast.RAY`, `SPHERE` or `CYLINDER`; the sphere's radius (0: the smallest wheel's); the steepest ground a wheel stands on (rad) |
| `max_pitch_roll` | rad from upright the chassis is held within; pi leaves it free |
| `longitudinal_friction`, `lateral_friction` | (slip ratio, friction) and (slip angle in degrees, friction) points |
| `body_friction` | the ground body's friction scales the tyre's as `sqrt(curve * friction)` (Jolt's combination) |
| `max_steer`, `steer_falloff_speed`, `steer_rate` | the lock at rest (rad); the speed (m/s) where it has halved, as `max_steer / (1 + (v / falloff)^2)` (0: no falloff); rad/s the angle moves (0: at once) |
| `drag`, `downforce` | N at 1 m/s, rising with the square of speed; downforce only with a wheel on the ground |
| `rolling_resistance` | a fraction of each grounded wheel's load, against its rolling above 0.15 m/s |
| `holding`, `holding_speed` | the idle hold above |
| `impact_speed`, `crash_speed` | m/s closing speeds reported as `IMPACT` and `CRASH` records |

`VehicleSettings.as_dict()` and `from_dict()` convert to and from builtin
types for a scene file or a service. `VehicleSettings.from_raycast(tuning,
wheels, mass, gravity=9.81)` maps omi_physics' `VehicleTuning` and
`WheelSpec` list (any objects with those attributes); its docstring states
the mapping field by field: suspension in car weights per metre and fractions
of critical damping become N/m and N s/m; `engine_force` and `base_speed`
become a one-gear drivetrain whose torque is flat to `base_speed` and constant
power above; `brake_force` becomes brake torques; the tyres have full
grip at every slip ratio and slip angle, so a tyre holds without slipping up to its friction.

What a wheel stands on. The casts use the chassis's collision layer, so its
group, mask and filter decide what it meets; sensors and character proxies are
never stood on. A wheel's grip and extra rolling resistance come from its own
surface (`Vehicle.set_surface(grip, rolling, wheel=None)`, or
`World.set_vehicle_surfaces`), or else from `World.set_surface_table(grip,
rolling)` indexed by the material index of the body under it (grip 1 past
the table). `set_surface(None)` gives the wheels back to the table.

`Snapshot.vehicles` is a `VehicleRows`, a row a vehicle in ascending id: `id`,
`chassis`, `kind`, `first_wheel`, `wheel_count`, `rpm`, `gear` (-1 reverse, 0
neutral, 1 first), `forward_speed` (m/s), `steer` (rad), `impact_speed` (m/s),
`impact_impulse` (N s), `impact_uid` (0 for none) and `input` (n, 4). Its
`wheels` is a `WheelRows`, every wheel in vehicle order: `vehicle` (row),
`position` (float64 centre), `orientation` (a wheel model with its axle along
X and up along Y, to the world, steer and spin included), `length` and
`compression` (m), `steer`, `angle` (rad), `spin` (rad/s), `load` (N), `slip`
(longitudinal ratio, lateral angle in rad), `contact`, `ground` (uid),
`material`, `triangle` (a mesh triangle's number, -1 elsewhere), `grip`,
`contact_point` and `contact_normal`. `Vehicle` reads its own row:
`forward_speed`, `rpm`, `gear`, `steer_angle`, `impact()` and
`wheel_slice()`.

Blows. After each step the contact records naming a chassis give its vehicle
the heaviest blow of the tick: the largest closing speed among the `BEGIN`
and `PERSIST` records (persist records only with `BodyFlag.REPORT` on the
chassis or the other body), with that record's impulse and other body. One at
or above `impact_speed` is also a record of kind `VEHICLE`, phase `IMPACT`
(`CRASH` at or above `crash_speed`): `a` the vehicle, `b` the body struck,
`approach` the closing speed, `impulse`, `point`, and `normal` from the
chassis into `b`.

A world with a vehicle never counts quiet lockstep ticks: an asleep vehicle's
engine and wheels still change each tick. `save_state` holds Jolt's vehicle
state; the steer angle and holding latch are the world thread's and are not
saved.

## Ragdolls

```python
settings = pj.RagdollSettings.humanoid(root_force=4000.0, root_torque=1500.0)
doll = world.add_ragdoll(settings, position=(0, 0, 0), mode=pj.RagdollMode.KINEMATIC)
doll.play(times, rotations, translations, loop=True)   # (K,), (K, B, 4), (K, B, 3)
doll.set_mode(pj.RagdollMode.LIMP, blend=0.3)          # on death: the motors fade over 0.3 s
doll.hit(settings.bone_index('head'), (30.0, 0.0, 0.0))  # N s on one bone
rows = world.latest().ragdolls                         # RagdollRows
joints = rows.bone_position[rows.bones(0)]             # (B, 3) one slice a ragdoll
```

A ragdoll is Jolt's `Ragdoll`: one dynamic body a bone, each joined to its
parent by a `SwingTwistConstraint`. A bone's body frame is its joint's frame,
so the bone poses in the snapshot are the joint poses a renderer skins with.
Ragdoll ids and the bones' uids are assigned at submission (ids from 1, never
reused); each bone is a body row of every snapshot, so `add_impulses`,
`set_velocities`, events, queries and joints name bones by uid.
`remove_bodies` refuses a bone (counted in `stats()['rejected']`);
`remove_ragdolls` removes the ragdoll and its bodies, its pairs ending with
reason `REMOVED`. `stats()['ragdolls']` counts the ragdolls in the world.

`RagdollSettings` holds the bones parents first, as `Bone(name, parent,
translation, rotation, shape, shape_offset, shape_rotation, mass, twist_axis,
plane_axis, twist, swing, bend, strength, friction)`:

| Field | Meaning |
|---|---|
| `parent` | index of the parent bone, -1 for the first and only for it |
| `translation`, `rotation` | the rest pose in the parent joint's frame (the root's in the ragdoll's frame), as a glTF node's |
| `shape`, `shape_offset`, `shape_rotation` | a `Shape` that can be dynamic, placed in the joint's frame |
| `mass` | kg |
| `twist_axis`, `plane_axis` | joint frame; None: towards the first child joint (or along the bone's own translation), and a direction square to it |
| `twist`, `swing`, `bend` | (min, max) twist about the twist axis; the swing cone's half angles for turns about the plane axis (tipping the bone towards the normal axis, twist cross plane) and about the normal axis; the bend turns the cone's centre about the plane axis away from the rest pose, so a knee bends 1.3 rad one way and not the other; rad |
| `strength` | N m, the motors' torque limit when powered |
| `friction` | N m holding the joint in every mode |

The ragdoll's own fields: `self_collision` (bones collide with each other
except a bone and its parent and bones whose shapes overlap in the rest pose;
False: no bone pair collides), `motor_frequency` (Hz) and `motor_damping`,
`root_frequency`, `root_damping`, `root_force` (N) and `root_torque` (N m) for
the powered root (0: free), and `friction`, `restitution`, `linear_damping`,
`angular_damping` (1/s), `gravity_factor`, `group`, `mask`, `filter`,
`material` and `flags` for every bone's body. `RagdollSettings.from_skeleton(names,
parents, translations, rotations=None, radius=0.05, mass=70.0)` gives each
bone a capsule from its joint to its first child's (a leaf's 0.9 times its own
translation long) and shares the mass by volume; `RagdollSettings.humanoid(height=1.8,
mass=70.0)` is an 11-bone standing figure with limits for each joint.
`as_dict()` and `from_dict()` convert to builtin types (primitive shapes only).
`forward_kinematics(settings, rotations, position, rotation, translations=None)`
returns the joint poses a pose target makes, the arithmetic the world thread
uses.

Modes (`RagdollMode`):

- `KINEMATIC` - every bone is a kinematic body moved by `MoveKinematic` to the
  pose target at the end of each tick; it pushes dynamic bodies and is not
  pushed.
- `POWERED` - dynamic bones; each joint's swing and twist motors turn its bone
  towards the target rotation relative to its parent, with a spring of
  `motor_frequency` and a torque limit of the bone's `strength`. With
  `root_force` or `root_torque`, the root bone is pulled to its target pose by
  a spring of `root_frequency` carrying the whole ragdoll's mass and weight,
  limited to that force and torque, so a struck ragdoll stays up and returns
  to its pose.
- `LIMP` - dynamic bones, motors off; joint friction only.

`set_mode(mode, blend=0.0)` (`World.set_ragdoll_modes(ids, mode, blend)`) blends
over `blend` seconds: into powered or limp the motors' strength (`RagdollRows.strength`,
0 to 1, scaling every bone's torque and the root's limits) moves linearly to 1
or 0; leaving kinematic keeps the bones' velocities, at full strength; into
kinematic the targets move from the bones' current poses to the pose target.
`set_strength(strength)` sets each bone's torque limit.

The pose target is the held pose, `set_pose(rotations, translations=None,
position=None, rotation=None)` (local rotations (B, 4) and translations (B, 3)
in each parent joint's frame; `position` and `rotation` place the ragdoll's
frame; None keeps each), or a clip, `play(times, rotations, translations=None,
loop=False)`, sampled at `times` seconds from the tick that applies it and
evaluated natively at the end of every tick: rotations by slerp, translations
linearly, holding the last pose past the end or looping. `place(position,
rotation)` moves the frame alone; `stop()` clears the clip. A rig's pose
(`OpenGLContext.character`'s translation (N, 3) and rotation (N, 4) arrays,
parents first) is this layout.

`Snapshot.ragdolls` is a `RagdollRows`, a row a ragdoll in ascending id: `id`,
`mode`, `strength`, `self_collision`, `first_bone` and `bone_count`; and every
bone in ragdoll order: `bone_uid`, `bone_position` (float64) and
`bone_orientation`. `rows.bones(i)` is ragdoll row `i`'s slice. `Ragdoll.pose()`
reads its own rows.

`hit(bone, impulse, point=None)` is an impulse on one bone, at its centre of
mass or at a world point (`add_impulses` with `points`).

Limits: a powered ragdoll's
root spring stands in for balance, which nothing here computes from the feet;
`save_state` holds the bodies and constraints, and the mode, strength, pose
target and clip are the world thread's and are not saved.

A ragdoll drives its bones while its pose target moves: a clip plays, a blend
or a strength ramp runs, or the target changed in the last tick. At a still
target kinematic bones stop and powered bones hold the pose until they fall
asleep, after which they are left alone until a new target or a contact
wakes them; a world of still ragdolls idles and counts quiet lockstep ticks.

## Soft bodies

```python
cloth = pj.SoftBodySettings.cloth(2.0, 2.0, columns=32, rows=32, mass=1.0, vertex_radius=0.01)
a, b, _, _ = cloth.corner_indices()
sheet = world.add_soft_body(cloth, position=(0, 2, 0), pins=([a, b], cloth.positions[[a, b]] + (0, 2, 0)))
sheet.set_wind((2.0, 0.0, 0.0))                 # N, shared among the vertices every tick
rows = world.latest().soft_bodies               # SoftBodyRows
i = rows.rows()[sheet.uid]
vertex_buffer = rows.vertices(i)                # (V, 3) float32 view, relative to rows.origin[i]
```

A soft body is Jolt's soft-body simulation: particles held by edge, shear,
bend, volume and long-range attachment constraints, solved by XPBD with
`iterations` a step, with an optional pressure inside a closed surface. It is
a body: its uid is assigned at submission, it is a row of every snapshot (at
its centre of mass), it collides with rigid bodies by the world's groups,
masks and filter table, sleeps and wakes (`activate`), and `remove_bodies`
removes it. `stats()['soft_bodies']` counts them.

`SoftBodySettings(positions, faces=None, edges=None, edge_compliance=None,
tetrahedra=None, mass=1.0, masses=None, ...)`:

| Field | Meaning |
|---|---|
| `positions` | (V, 3) metres in the body's frame |
| `mass`, `masses` | kg shared equally, or (V,) each; a vertex of mass 0 is fixed |
| `faces` | (F, 3), counter-clockwise from the side the normal faces; edges, shear and bend constraints come from them unless `edges` is given |
| `edges`, `edge_compliance` | (E, 2) and (E,) explicit edges |
| `tetrahedra`, `volume_compliance` | (T, 4) volume constraints |
| `compliance`, `shear_compliance`, `bend_compliance` | m/N; 0 rigid, `inf` none; bend 0.01 by default |
| `bend` | `Bend.NONE`, `DISTANCE` or `DIHEDRAL` |
| `pressure` | n R T (Pa m3) inside a closed surface |
| `long_range`, `long_range_stretch` | `LongRange.NONE`, `EUCLIDEAN` or `GEODESIC` attachments to the nearest pinned vertex, and the slack allowed on the rest distance |
| `iterations`, `linear_damping` (1/s), `max_velocity` (m/s) | the solver; iterations are substeps, a whole number from 1 to 1000 (`softbodies.MAX_ITERATIONS`), None takes `softbodies.iterations_for(shortest edge)` |
| `friction`, `restitution`, `gravity_factor`, `vertex_radius` (m) | contact; the vertices keep `vertex_radius` from surfaces |
| `allow_sleep`, `double_sided` | sleeping; faces collide on both sides for queries |
| `group`, `mask`, `filter`, `material`, `flags` | as a body's |

Stability. A cloth with no bend constraints folds on what it meets, and with
edges of 5 cm or shorter Jolt's solver tears it apart within a second at
60 Hz (a 2 m cloth of 40 by 40 vertices on a sphere, reproduced with Jolt
alone); the default distance bend of compliance 0.01 holds it. Shorter edges
need more substeps: `iterations_for(edge)` is 5 from 5 cm, 11 at 3.4 cm and 20
at 2.5 cm, measured at 60 Hz; a world at 120 Hz needs about half.
With the defaults the edges still stretch: that 2 m cloth, draped over a
0.6 m sphere on a floor at 60 Hz, settles with its longest face edge 1.31 to
1.36 times its rest length at 32 to 80 vertices a side, and an edge reaches
1.54 times while it falls. `long_range` attachments limit the stretch of a
pinned cloth.

Builders: `cloth(width, depth, columns, rows, mass)` (y = 0, centred, faces
up; `corner_indices()`), `rope(points, segments=None, mass=0.1,
bend_compliance=inf)` (edges along a polyline, plus edges across every second
vertex with a finite bend compliance), `ball(radius, subdivisions, pressure,
mass)` (an icosphere held by pressure) and `from_mesh(positions, faces,
mass)`. `volume(positions=None)` is the enclosed volume. `as_dict()` and
`from_dict()` convert to builtin types.

Pins. `add_soft_body(settings, position, rotation, pins=(vertices, points),
pin_bodies=None)`, `SoftBody.pin(vertices, points, bodies=None)`
(`World.set_soft_pins`) and `unpin()`. A pinned vertex has no mass and is
moved each tick to reach its point by the end of the step: a world point, or a
point in the frame of the body at the same index of `bodies`, which it follows
as that body moves. A pin does not pull on its body. A vertex no longer pinned
takes back its mass. Pins to a removed body are dropped.

Wind. `SoftBody.set_wind(force)` (`World.set_wind(uids, forces)`) is a force
(N) applied to the body before every step, which Jolt shares among its free
vertices, and wakes it. It holds until changed.

`Snapshot.soft_bodies` is a `SoftBodyRows`, a row a soft body in ascending
uid: `uid`, `origin` (float64 centre of mass, the body row's position),
`first_vertex`, `vertex_count`, `version`, and every vertex in body order:
`positions` and `normals` (float32, relative to the origin; area-weighted
normals of the faces, zero for a vertex on no face). `vertices(i)` and
`normals_of(i)` are body row `i`'s views, ready for a vertex buffer.
`version` is the tick the vertices were taken at: a soft body copies its
vertices after a tick that moved it, and a snapshot buffer copies a body's
vertices only when its version differs from the one the buffer holds, so an
asleep cloth costs nothing to publish and a renderer re-uploads a body when
its version changes. `SoftBody.vertices()` returns world coordinates (a
float64 copy).

Limits: soft bodies do not collide with each other; contacts between a soft
body and a rigid body make no event records; `save_state` holds the vertices,
and pins and wind are the world thread's.

## Joints

```python
door = world.add_joint(pj.Joint.hinge(panel, frame, anchor=(0, 1, 0), axis=(0, 1, 0),
                                      limits=(0.0, 1.6)))
wheel = world.add_joint(pj.Joint.hinge(rotor, anchor=(6, 4, 0), axis=(0, 1, 0),
                                       motor=pj.Motor(pj.MotorMode.VELOCITY, 4.0, max=30.0)))
world.set_joint_motors([wheel], pj.MotorMode.VELOCITY, -4.0)
ids, seq = world.add_joints([pj.Joint.ball(link, prev, anchor=p, break_force=500.0) for ...])
rows = world.latest().joints          # id, kind, state, value, value2, force, torque
```

A `Joint` holds `body` to `to` (0 for the world). Anchors and axes are world
coordinates where the bodies stand when the batch is applied; each body
carries them from then on. Joint ids are assigned at submission, from 1, and
never reused. A joint naming a body that is not in the world when its batch is
applied, or a soft body (which takes pins), is skipped and counted in
`stats()['rejected']`.

| Builder | Jolt | Holds |
|---|---|---|
| `Joint.fixed(body, to)` | `FixedConstraint` | the pose the two have when it is made |
| `Joint.ball(body, to, anchor=)` | `PointConstraint` | a point of each together; they turn freely |
| `Joint.hinge(body, to, anchor=, axis=, limits=, motor=)` | `HingeConstraint` | turning about `axis`; `limits` within -pi to pi |
| `Joint.slider(body, to, axis=, anchor=, limits=, motor=)` | `SliderConstraint` | sliding along `axis` without turning; `limits` lower <= 0 <= upper; `anchor` defaults to the body's origin |
| `Joint.distance(body, to, anchor=, to_anchor=, length=)` | `DistanceConstraint` | the two points at `length`, a (shortest, longest) range, or as they stand |
| `Joint.six_dof(body, to, anchor=, axis=, axis2=, lower=, upper=, motors=)` | `SixDOFConstraint` | six axes, each locked `(0, 0)`, free `(-inf, inf)` or limited |
| `Joint.universal(body, to, anchor=, axis1=, axis2=)` | `SixDOFConstraint` | X3D's UniversalJoint: the two axes keep their angle, spin crosses |
| `Joint.two_axis_hinge(body, to, anchor=, axis1=, axis2=, limits1=, motor1=, motor2=, suspension=)` | `SixDOFConstraint` | X3D's DoubleAxisHingeJoint: steering about `axis1`, axle `axis2`, a sprung anchor along `axis1` |
| `Joint.motor(body, to, axes=, torques=)` | none | X3D's MotorJoint: torques (N m) about up to three axes `to` carries, each step |

Every builder also takes `limit_spring` (frequency Hz, damping ratio; `(0, 0)`
is rigid), `friction` (N or N m on the free axes), `break_force` (N),
`break_torque` (N m) and `collide`. A `Motor(mode, target, max, frequency,
damping)` drives towards a speed (`VELOCITY`, m/s or rad/s) or a position
(`POSITION`, m or rad, as a spring of `frequency` and `damping`) with at most
`max` N or N m. A six-axis joint's axes are translation x, y, z then rotation
x, y, z of its frame, x along `axis` and y along `axis2` made square to it.

| Call | Change |
|---|---|
| `add_joint(joint)`, `add_joints(joints)` | add; returns the id, or `(ids, seq)` |
| `remove_joints(ids)` | remove; each makes a `JOINT` `END` record |
| `set_joint_motors(ids, mode, target=nan, max=nan, axis=0)` | one motor axis of each; None or NaN keeps a value; a motor joint's axis 0 to 2 takes a torque as `target` |
| `set_joints_enabled(ids, enabled)` | switch off or on; a broken joint stays broken |

After each step the world reads each joint's total lambda and divides it by
the step: `force` (N) and `torque` (N m) in the snapshot. A joint whose force
or torque passes its limit is taken out of the world and reported once, a
`JOINT` `BREAK` record with `a` the joint, `b` its body, `point` its anchor,
`approach` the force and `impulse` the torque. Removing a body removes its
joints (`END`, reason `REMOVED`). The two bodies of a joint do not collide with
each other unless `collide`: they share a Jolt group filter that refuses the
pair in the broad phase.

`Snapshot.joints` is a `JointRows`, one row a joint in ascending id: `id`,
`kind` (`JointKind`), `state` (`JointState`: `ACTIVE`, `DISABLED`, `BROKEN`),
`value` (a hinge's angle, rad; a slider's position, m; a distance's length,
m; a six-axis joint's turn about its frame's y axis), `value2` (a six-axis
joint's turn about z), `force` and `torque`.

## Materials

```python
wood = world.materials.define(friction=0.5, restitution=0.2)
ice = world.materials.define(static_friction=0.05, dynamic_friction=0.02,
                             friction_combine=pj.Combine.MINIMUM)
world.materials.pair(wood, ice, friction=0.1)
world.add_bodies(pj.BodyBatch.boxes(positions, material=wood))
```

`World.materials` is the world's `MaterialTable`; `define` returns a material
index, `redefine(indices, ...)` gives one or more defined materials new values
(their bodies meet with them from the next tick, and pair entries stay), `pair` sets how
two materials meet and `pairing` reads it back (static and dynamic friction
and restitution, NaN for a value left to the materials). Each change is sent as a batch.
Index 0 is a body's own `friction` and `restitution`, combined as Jolt
combines them (the geometric mean of the frictions, the larger restitution).
Where either side of a contact has another material, the two rows combine by
the higher of their modes, `Combine.AVERAGE` < `MINIMUM` < `MULTIPLY` <
`MAXIMUM` (the OMI physics material's modes), and a pair entry replaces any
value it sets. The static friction applies while the bodies slide past each
other slower than `WorldSettings.static_friction_speed` (0.05 m/s) at the
contact, the dynamic friction above it; the contact listener sets Jolt's
combined friction and restitution for each contact as it is added and each
step it persists. A mesh triangle may carry its own material
(`Shape.mesh(points, triangles, materials=)`), which contacts and a character
standing on it report.

## Collision filters

```python
world.set_filter_table([[True, False], [False, True]])   # (F, F)
world.add_bodies(pj.BodyBatch.spheres(positions, filter=1))
```

A body's `filter` is an index into the world's (F, F) collides-with table,
made symmetric; a filter index outside the table meets every filter, so a
world with no table collides every pair the groups and masks allow. Each
distinct (group, mask, filter, moving or static, sensor) gets its own Jolt
object layer, and Jolt's object layer pair filter reads the table, so the
broad phase drops a refused pair. A layer is freed when the last body (or
character proxy, or vehicle on it) using it goes, so a world holds at most
65535 distinct keys at once. An add whose new keys would take the world past
that raises `ValueError`; batches queued together can pass that check and
overrun it, and the rows left with no layer, or a filter or motion type change
with none, are counted in `stats()['rejected']`.
`set_filter_table` takes effect at the next tick and wakes every dynamic body;
`set_body_filters` moves bodies to other filters.

## The contact filter

```python
def verdict(a, b, point, normal, approach):
    return approach < 5.0          # False: pass through until they part

world.set_contact_filter(verdict)
world.add_bodies(pj.BodyBatch.spheres(positions, flags=pj.BodyFlag.FILTER_CONTACTS))
```

The filter is asked about a pair of solid bodies that both carry
`BodyFlag.FILTER_CONTACTS` (either of them, with `set_contact_filter(fn, either=True)`); with `recheck=True` a refusal holds for the step that made it and the pair is asked again at each later step it meets when Jolt first finds them touching
(`ContactListener::OnContactValidate`), on the job thread that found them,
inside the step: `a < b` uids, the contact point, the unit normal from `a` to
`b` and the closing speed (m/s). False lets the two pass through each other;
True or None solves them. The answer holds until the pair parts: a solved
pair's when its contacts end, a refused pair's at the first step that does not
meet them again. Collision filters and groups are applied first, so only pairs
they allow are asked.

Each job thread keeps one Python thread state for the world's run, made the
first time it asks. The world thread holds the world lock exclusively during
the step and does not hold the GIL, and every call that can wait for the world
lock releases the GIL first, so a job thread waiting for the GIL waits for no
thread that waits for it. The filter may submit commands and call `stop()`,
which ends the thread after the tick, or `start()`. A query, `save_state`,
`flush`, `wait_tick` or `run_ticks` from it raises `RuntimeError` at once,
since each waits for the step it runs inside. An exception stops the world,
as from any callback.

## Gravity zones and drag

```python
world.set_gravity_zones([
    pj.GravityZone.box((0, 5, 0), (10, 10, 10), gravity=2.0, direction=(0, 1, 0), replace=True),
    pj.GravityZone.sphere((50, 0, 0), 30.0, field=pj.ZoneField.POINT, center=(50, 0, 0),
                          gravity=9.81, priority=1),
])
```

A zone (OMI_physics_gravity) is a region, `GravityZone.box(position, size)`,
`sphere(position, radius)` or `of_shape(shape_id, position, rotation)` (a
registered shape's inside), turned by `rotation`, and a field:
`DIRECTIONAL` along `direction` or `POINT` towards `center`, of `gravity`
m/s2. A body whose centre of mass is inside zones takes the world's gravity,
then each zone's in ascending `priority`: `replace` sets it to the zone's,
`stop` sets it to zero, any other adds the zone's. The body's gravity factor
scales the result. A Jolt step listener adds the difference from the world's
gravity as a force before every collision step, to awake dynamic bodies.
`set_gravity_zones` replaces the world's zones and wakes every dynamic body.

Quadratic drag (`BodyBatch.quadratic_drag`, `set_drag`) is a force of
`-k m |v| v` in the same listener: a falling body reaches `sqrt(g / k)`.

## Kinematic tracks

```python
world.set_track([lift], pj.Track([0.0, 4.0, 8.0], [(0, 0, 0), (0, 6, 0), (0, 0, 0)], loop=True))
world.drive(platform, lambda t: ((math.cos(t), 1.0, math.sin(t)), (0, 0, 0, 1)))
```

A `Track` is `times` (seconds from the tick that applies it, ascending),
`positions` (k, 3) and `rotations` (k, 4, or None for the identity), and
`loop`. Each tick the world moves the kinematic body to the track's pose at
the end of the tick with Jolt's `MoveKinematic`, so what rides on it is
carried. Positions are interpolated linearly and rotations by slerp; past the
last sample a track holds its last pose or starts again. `clear_tracks` takes
bodies off. `drive(uid, pose)` calls `pose(time)` on the world thread before
each step, `time` the simulation time at the end of the tick, and moves the
body there; it returns a position or `(position, quaternion)`.
`move_kinematic(uids, positions, orientations)` from a world callback moves
bodies in the current tick, and from any other thread in the next.

## Streaming ground

A landscape too large to hold whole is held as height field tiles
(`heightfield_tiles`) near the places that need ground: the camera, the
characters, awake bodies. `swap_bodies(remove, batch)` removes the tiles left
behind and adds the ones come near in one batch, applied in one tick, so no
tick has the old tiles gone and the new not yet there. Each new `Shape` is
built on the calling thread when the batch is submitted; the world thread only
creates and adds the bodies. In `benchmarks/characters.py` (10 characters
sprinting over a 1 km field cut into 32-cell tiles, tiles within 48 m kept) a
frame with a swap took 0.19 ms on 1 job thread against 0.13 ms without,
building the tile included.

## Callbacks

Callables run on the world thread with the GIL held (attached, on a
free-threaded build), through one Python thread state the world thread keeps
for its life. A world with none registered never takes the GIL. Each setter
returns its argument, so it can decorate the function; None removes it.

| Call | Called as |
|---|---|
| `set_before_step(fn)` | `fn(tick, time)` after the tick's batches are applied, `tick` the ticks completed |
| `set_after_step(fn)` | `fn(tick, time)` after the snapshot is published, `tick` including the tick just run |
| `set_on_events(fn)` | `fn(events, tick, time)` after a tick that made records |
| `at_tick(tick, fn)` | `fn(tick, time)` once, before `before_step`, at the first tick whose index is at least `tick` |

A callback may submit commands. Those submitted from an `at_tick` callable,
`before_step` or a `drive` pose function apply in the same tick, before its
step; those from `after_step` and `on_events` apply at the next tick. It may call
`stop()`, which ends the thread after the current tick, and `save_state()`,
which reads the tick as it stands. `run_ticks` and `wait_tick` from a
callback raise `RuntimeError`. A world dropped or collected from one of its
own callbacks ends its thread after the tick.

A callback that raises stops the world. The exception is kept as
`world.error`, logged once to the `pyjoltphysics` logger, and every waiting
or later `run_ticks`, `wait_tick` and `start` raises `WorldError` with the
exception as its `__cause__`.

On a GIL build a callback waits for the GIL. While another thread runs pure
Python, each callback waits up to the switch interval
(`sys.getswitchinterval()`, 5 ms by default). OpenGLContext's physics service
sets it to 1 ms while a physics-thread callback is registered.

A world's callbacks may refer to the world: the world type takes part in
Python's cyclic garbage collection, so a world and its callbacks are
collected together. When the collector finds such a cycle, the world's
finalizer runs first: no callback starts after it, and a callback running on
another thread returns before the collector clears anything.

The world thread has an 8 MiB stack on Linux and macOS, whatever the C
library's default (128 KiB on musl, 512 KiB for a macOS thread, RLIMIT_STACK
on glibc). Jolt's job threads, which run the contact filter, get 8 MiB on
Linux; on macOS they have 512 KiB. On Windows both have the executable's
default stack, 2 MB for python.exe, as Python's own threads do.

## Saving and restoring

`save_state()` returns the bodies' and joints' state as bytes: Jolt's
`PhysicsSystem::SaveState` (positions, velocities, sleep, contact cache and
constraint state) after the uids of the bodies it holds. It is read under the
world lock between ticks, or from a step callback as the tick stands. A
32-byte header leads: `PJST`, a format version (1), the Jolt version id, the
payload length and a 64-bit FNV-1a hash of the payload, integers in the
machine's byte order.

`restore_state(data)` raises `ValueError` when the header, length or hash do
not match, before Jolt reads anything: Jolt trusts the counts in the state, so
a truncated or damaged save would overrun its buffers. Only bytes
`save_state` of this build wrote are read; the hash finds damage and is no
defence against bytes made to match it. The restore is then a command: at the
start of the tick that applies it the world takes the state back, so a
lockstep world given the same batches afterwards repeats the run from the
save. A world holding other bodies than the state names, or a state Jolt
cannot read to its last byte, is rejected (`stats()['rejected']`) and the
world keeps the state it had. Characters, the tick count and queued batches
are not part of the state.

## Determinism and replay

Jolt's step is deterministic for the same bodies, batches and ticks, with any
number of job threads, and across x86-64 and arm64 with the
cross-platform-deterministic build every wheel has. The tick a free-running
command lands on is recorded by `drain_applied()`; a lockstep world given the
same batches with `at_tick` set to the recorded ticks repeats the run bit for
bit. Uids are assigned in submission order, so a replay that submits the same
batches in the same order gets the same uids, and joint ids likewise.
Joints, materials, filters, the contact filter (when its answers depend only
on its arguments), gravity zones, drag and tracks keep the replay bitwise on 1
and 8 job threads (tests/test_p3_lockstep.py), and so do vehicles
(tests/test_vehicle_lockstep.py) and ragdolls in every mode with soft bodies
under wind (tests/test_p6c_lockstep.py).

## Fork

A world belongs to the process that made it. In the child of a fork, every
call on a parent's world raises `RuntimeError`, and the world is never torn
down there (its threads exist only in the parent). The child makes its own
worlds. The hooks are `os.register_at_fork` handlers registered with the
first world.

Before the fork the hook waits for each world thread to finish its tick and
holds it between ticks until the fork returns, so the child inherits no lock
or half-made allocation from a world thread, and Jolt's job threads are idle.
A world forked from its own callback or contact filter is not waited for. A
tick that has not ended after 5 s, such as a callback waiting on the thread
that forks, is left running and the fork goes ahead.

## `pyjoltphysics.testing`

Scene helpers shared by the suites and the benchmarks: `floor(half,
**values)`, a static box floor with its top at y = 0, `values` further
`BodyBatch` fields; `latest(world)`, the newest snapshot, an `AssertionError`
before the first tick; `grid(count, spacing, y0)`, grid
positions; `pose_digest(snapshot, digest)`, a hashlib object updated with a
snapshot's uids and poses in uid order; `character_digest(snapshot, digest)`,
the same with every character's id, position, velocity, state, ground and
slide progress after the poses; `vehicle_digest(snapshot, digest)`, the same
with every vehicle's rpm, gear, steer, impact and wheel state;
`ragdoll_digest(snapshot, digest)`, the same with every ragdoll's mode,
strength and bone poses; `soft_digest(snapshot, digest)`, the same with every
soft body's origin, version, vertices and normals; `MakeWorld`, the type of the suites'
`make_world` fixture.

## Build information

### `jolt_version() -> str`

The version of Jolt Physics compiled into `pyjoltphysics._native`, as
`'major.minor.patch'`: `'5.6.0'`.

### `build_options() -> BuildOptions`

The options the module and its Jolt were built with. `BuildOptions` is a frozen
dataclass:

| Field | Type | Meaning |
|---|---|---|
| `double_precision` | `bool` | positions are 64-bit (`JPH_DOUBLE_PRECISION`) |
| `cross_platform_deterministic` | `bool` | `JPH_CROSS_PLATFORM_DETERMINISTIC` |
| `simd_level` | `str` | `'sse4.2'`, `'avx'`, `'avx2'` or `'avx512'` on x86-64, `'neon'` on arm64, `'none'` elsewhere |
| `free_threaded` | `bool` | the module was built for a free-threaded CPython |
| `debug` | `bool` | Jolt's assertions or the C++ debug checks are compiled in |
| `compiler` | `str` | the compiler's name and version, such as `'gcc 13.3.0'` or `'msvc 194334808'` |

A published wheel reports double precision, the deterministic build, SSE4.2 on
x86-64 or NEON on arm64, and `debug=False`.

### `selftest() -> bool`

Steps a Jolt world holding one 1 kg box, starting 10 m up, once by 1/60 s on
the calling thread with the GIL released. Returns `True` when the step reports
no error and the box has moved down. Raises `RuntimeError` when Jolt's version
ID (its version and feature bits, double precision among them) differs from
the one the binding was compiled against.

### `__version__`

The distribution's version, `'0.1.0a1'`.
