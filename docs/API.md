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
3. step the world by `1 / tick_hz` seconds (`PhysicsSystem::Update`, the exclusive world lock)
4. fold the step's contact reports into event records and append them to the event ring
5. publish the snapshot
6. call `after_step`, then `on_events` when the tick made records

The thread either free-runs or runs in lockstep:

| Call | Effect |
|---|---|
| `start()` | free-run at `tick_hz`, starting the thread if it is not running |
| `run_ticks(n)` | run exactly `n` ticks, unpaced, and return when they are done; the world stays in lockstep until the next `start()` |
| `stop()` | stop and join the thread; idempotent; from a callback, the thread ends after the current tick |
| `with World(...) as world:` | `stop()` at the end of the block |

A free-running world with no pending batch, no awake body and no step
callback (`before_step`, `after_step` or a pending `at_tick`) waits on a
condition variable: it runs no ticks and uses no CPU, and its tick count and
simulation time stand still. A command wakes it at once. A free-running world
that falls more than `max_catch_up` seconds behind its schedule drops the
ticks it is behind by (`stats()['dropped_ticks']`), so simulated time slows.

A world is also stopped when it is garbage collected and at interpreter exit
(an `atexit` hook registered with the first world). A world is never stepped
inline on the calling thread and never in another process.

### `WorldSettings`

A frozen dataclass; every field has a default.

| Field | Default | Meaning |
|---|---|---|
| `tick_hz` | 120.0 | ticks a second |
| `max_catch_up` | 0.1 | seconds behind schedule before a free-running world drops ticks |
| `gravity` | (0, -9.81, 0) | m/s2 |
| `threads` | 4 | Jolt job threads, the world thread included; 1 steps on the world thread alone |
| `max_bodies` | 65536 | bodies the world holds; an add batch past it raises `ValueError` |
| `max_body_pairs`, `max_contact_constraints` | 65536 | Jolt's pair and contact capacities |
| `event_capacity` | 65536 | event records held for `drain_events` |
| `allow_sleeping` | True | bodies at rest fall asleep |
| `time_before_sleep` | 0.5 | seconds below `sleep_velocity` before a body falls asleep |
| `sleep_velocity` | 0.03 | m/s at every point of the body |
| `min_velocity_for_restitution` | 1.0 | m/s closing speed below which a contact does not bounce |
| `collision_steps` | 1 | Jolt collision sub-steps a tick |
| `temp_allocator_mb` | 16 | step scratch memory allocated up front; more comes from the heap |
| `linear_damping`, `angular_damping` | 0.0 | 1/s, for bodies whose batch names none |

`settings.dt` is `1 / tick_hz`.

### Reading

| Call | Returns |
|---|---|
| `latest()` | the newest `Snapshot`, or None before the first tick; never waits |
| `wait_tick(tick, timeout=None)` | True once a snapshot of `tick` or later is published; False on timeout or when the world stops |
| `drain_events()` | every event record since the last drain, as an array of `EVENT_DTYPE` |
| `drain_applied()` | (k, 2) uint64 `(seq, tick)`: each batch applied since the last drain and the index of the tick that applied it |
| `tick` | ticks completed |
| `lockstep` | True unless `start()` is free-running the world |
| `error` | the exception a callback raised, or None |
| `submitted_seq` | the sequence number of the last batch submitted |
| `stats()` | counters (`ticks`, `dropped_ticks`, `events_dropped`, `rejected`, `pool`, `bodies`, `idle`, `running`) and mean milliseconds a tick since `reset_stats()` (`apply_ms`, `step_ms`, `events_ms`, `publish_ms`, `callbacks_ms`, `tick_ms`) |

## Commands

Every command takes arrays for many bodies, may be called from any thread
(a callback on the world thread included), and returns the batch's sequence
number. Batches are applied in sequence order at the start of a tick, before
the step: the first tick whose index (ticks completed before it) is at least
the batch's `at_tick`, 0 by default. A command naming a uid that is not a body
is skipped; one the body cannot take (a velocity for a static body) is
skipped and counted in `stats()['rejected']`.

| Call | Change |
|---|---|
| `add_bodies(batch)` | add a `BodyBatch`; returns `(uids, seq)`, the uids assigned at once |
| `remove_bodies(uids)` | remove bodies; their pairs end with `EndReason.REMOVED` |
| `set_transforms(uids, positions, orientations=(0, 0, 0, 1))` | teleport; waking moving bodies |
| `set_velocities(uids, linear=None, angular=None)` | m/s and rad/s; None leaves one as it is |
| `add_forces(uids, forces)` | N at the centre of mass, acting over the next step |
| `add_impulses(uids, impulses)` | N s, changing the velocity at once |
| `add_torques(uids, torques)` | N m, over the next step |
| `set_gravity(gravity)` | m/s2; wakes every dynamic body |
| `activate(uids)` | wake bodies |
| `set_motion_types(uids, motion)` | `MotionType` per body; a static body becomes moving only if added with `BodyFlag.MOVABLE` |
| `set_materials(uids, friction, restitution)` | new coefficients |

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
| `shape_kind`, `shape_params` | box, (1, 1, 1, 0) | a `ShapeKind` and its four parameters |
| `motion` | `DYNAMIC` | `MotionType` |
| `orientations` | identity | (n, 4) unit quaternions |
| `mass` | 1.0 | kg; 0 takes the mass from `density` |
| `density` | 0.0 | kg/m3, used where `mass` is 0 |
| `friction`, `restitution` | 0.5, 0.0 | Jolt combines a pair's as the geometric mean of the frictions and the larger restitution |
| `linear_damping`, `angular_damping` | None | 1/s; None takes the `WorldSettings` default |
| `gravity_factor` | 1.0 | multiplies gravity for the body |
| `linear_velocity`, `angular_velocity` | 0 | initial m/s and rad/s |
| `group`, `mask` | 1, 0xFFFFFFFF | 32-bit collision sets: two bodies collide when each one's group shares a bit with the other's mask |
| `flags` | `NONE` | `BodyFlag` bits |

Builders, each returning a batch, with every other field as a keyword:
`BodyBatch.boxes(positions, size=(x, y, z))`, `spheres(positions, radius)`,
`capsules(positions, radius, height)`, `cylinders(positions, radius, height)`
and `plane(position=(0, 0, 0), half_extent=0.0)`.

### Shapes

`ShapeKind`, with sizes as in the OMI physics shape model:

| Kind | Parameters | Notes |
|---|---|---|
| `BOX` | edge lengths x, y, z | |
| `SPHERE` | radius | |
| `CAPSULE` | radius, height of the cylinder between the hemisphere centres | along the body's Y axis |
| `CYLINDER` | radius, total height | along the body's Y axis |
| `PLANE` | half extent of its bounds, 0 for 1 km | through the body's origin, normal the body's +Y; static only |

Equal kinds and parameters share one Jolt shape. Boxes and cylinders are
rounded by a convex radius of 5 cm or half the smallest dimension, whichever
is smaller.

### `BodyFlag`

| Flag | Meaning |
|---|---|
| `SENSOR` | reports overlaps as trigger events; no contact response |
| `NO_SLEEP` | never falls asleep |
| `CONTINUOUS` | continuous collision detection (Jolt's linear cast), for small fast bodies |
| `REPORT` | persist events, once a tick, for every pair the body touches |
| `MOVABLE` | a static body a later `set_motion_types` may make kinematic or dynamic; needs a mass or density |

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

`EVENT_DTYPE` is a 48-byte record:

| Field | Type | Meaning |
|---|---|---|
| `kind` | uint8 | `EventKind`: `CONTACT`, `TRIGGER` or `STATE` |
| `phase` | uint8 | `EventPhase`: `BEGIN`, `PERSIST`, `END`, `SLEEP` or `WAKE` |
| `reason` | uint16 | `EndReason` of an end: `NONE`, or `REMOVED` when a body of the pair was removed |
| `tick` | uint32 | the tick the event happened in: the tick of the first snapshot that shows it |
| `a`, `b` | uint32 | uids: for a contact the lower uid first; for a trigger the sensor first; for sleep and wake the body, and 0 |
| `point` | 3 float32 | world contact point, m (begin and persist) |
| `normal` | 3 float32 | unit vector from `a` to `b` (contact begin and persist) |
| `approach` | float32 | closing speed along the normal at the point, m/s, positive when approaching |
| `impulse` | float32 | estimated normal impulse, N s, contact begin only |

A record's simulation time is `tick / tick_hz`; `event_times(events, tick_hz)`
computes it. The impulse is Jolt's `EstimateCollisionResponse` for the two
bodies alone, since the solver has not run when contacts are reported.

The world folds Jolt's per-sub-shape reports into one record per body pair:
one begin when the pair first touches and one end when it parts. Persist
records come once a tick for pairs where either body has `BodyFlag.REPORT`.
A sensor's begin and end are a body entering and leaving it.

Sleep. Jolt drops a sleeping body's contacts and adds them again when it
wakes. The world holds a pair whose contacts go while none of its bodies is
awake as touching: no end is recorded when it falls asleep and no begin when
it wakes still touching. A body that wakes and moves away ends its pairs in
the tick their contacts fail to return. A body asleep inside a sensor stays
inside it.

Each tick's records are sorted, so a run makes the same records in the same
order with any number of job threads.

The ring holds `event_capacity` records. When it is full, persist records go
first, oldest first, then sleep and wake records; begin and end records are
never dropped, so that each pair's state stays balanced, and a ring holding
only those grows past its capacity until it is drained. Dropped records are
counted in `stats()['events_dropped']`.

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

A callback may submit commands; they apply at the next tick. It may call
`stop()`, which ends the thread after the current tick. `run_ticks` and
`wait_tick` from a callback raise `RuntimeError`.

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
collected together.

## Determinism and replay

Jolt's step is deterministic for the same bodies, batches and ticks, with any
number of job threads, and across x86-64 and arm64 with the
cross-platform-deterministic build every wheel has. The tick a free-running
command lands on is recorded by `drain_applied()`; a lockstep world given the
same batches with `at_tick` set to the recorded ticks repeats the run bit for
bit. Uids are assigned in submission order, so a replay that submits the same
batches in the same order gets the same uids.

## Fork

A world belongs to the process that made it. In the child of a fork, every
call on a parent's world raises `RuntimeError`, and the world is never torn
down there (its threads exist only in the parent). The child makes its own
worlds. The hooks are `os.register_at_fork` handlers registered with the
first world.

## `pyjoltphysics.testing`

Scene helpers shared by the suites and the benchmarks: `floor(half)`, a
static box floor with its top at y = 0; `grid(count, spacing, y0)`, grid
positions; `pose_digest(snapshot, digest)`, a hashlib object updated with a
snapshot's uids and poses in uid order; `MakeWorld`, the type of the suites'
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
