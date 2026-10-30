# Changelog

## 0.1.0a1

- `World`: a Jolt `PhysicsSystem` stepped at `tick_hz` on a native world thread, free-running or in lockstep (`run_ticks`); idle with no pending work; `stop()`, a context manager, collection, `atexit` and fork hooks.
- Command batches as arrays from any thread: `add_bodies` (`BodyBatch`), `remove_bodies`, `set_transforms`, `set_velocities`, `add_forces`, `add_impulses`, `add_torques`, `set_gravity`, `activate`, `set_motion_types`, `set_materials`; sequence numbers, `applied_seq`, `at_tick` and `drain_applied`.
- Shapes: box, sphere, capsule, cylinder and static plane; collision groups and masks; sensors, sleep, CCD and report flags.
- `Snapshot`: read-only NumPy views of a pooled, reference-counted buffer; `latest()`, `wait_tick()`, `changed` rows, `rows`.
- Events as 48-byte `EVENT_DTYPE` records: contact and trigger begin, persist and end, aggregated per body pair, and sleep and wake; sleeping pairs held as touching.
- Callbacks on the world thread: `before_step`, `after_step`, `on_events`, `at_tick`; an exception stops the world and is raised as `WorldError`.
- `Shape` and shape ids: `World.add_shape`, `remove_shape`, `shape_info`; convex hulls, triangle meshes, height fields (holes, any rectangular grid), static and mutable compounds; one Jolt shape shared by every body naming an id.
- `BodyBatch.shape`, `scale`, `shape_offset`, `shape_rotation`, `center_of_mass`; builders `of_shape`, `hulls`, `meshes`, `heightfield`; `heightfield_tiles` and `heightfield_side`.
- `BodyFlag.SMOOTH_EDGES`, Jolt's enhanced internal edge removal.
- A sensor added or moved over sleeping bodies reports them; a kinematic sensor meets static bodies.
- The broad phase is optimised once after a large add of static bodies.
- The event record is 64 bytes, with a float64 contact point; `impulse` is in begin records only.
- A future `at_tick` leaves a free-running world idle.
- `stats()`: `bodies` counts bodies in the world, `queued_bodies` those submitted and not applied; `shapes`, `broadphase_optimizations`.
- Queries on the calling thread under the world lock's shared side, GIL released: `World.raycast`, `raycast_batch`, `sweep`, `sweep_sphere`, `sweep_capsule`, `sweep_box`, `overlap`; `QueryFilter`, `RayHit`, `RayHits`, `ShapeHit`.
- The world lock holds new queries behind the world thread when it waits, so queries from many threads cannot hold a tick off.
- Characters: omi_physics' controller over Jolt's `CharacterVirtual`, moved on the world thread each tick: `World.add_character`, `Character`, `CharacterCapabilities`, `CharacterState`, `MoveMode`, `CharacterOp`, `character_ops`, `remove_characters`; `Snapshot.characters` (`CharacterRows`); a kinematic sensor proxy per character.
- Event kind `CHARACTER` with phases `BEGIN`, `END`, `LAND`, `JUMP`, `SLIDE`, `DOWN`, `UP`.
- `World.swap_bodies`: removal and addition in one tick, for streamed ground.
- A static mesh or height field has enhanced internal edge removal by default; `BodyFlag.NO_SMOOTH_EDGES` turns it off.
- `stats()`: `characters`, `characters_ms`, `queries`, `query_wait_ms`, `query_wait_max_ms`.
- `pyjoltphysics.testing.character_digest`.

- `jolt_version()`, `build_options()` and `selftest()` over the compiled module `pyjoltphysics._native`.
- Jolt Physics v5.6.0 (commit e77f1755), double precision, cross-platform deterministic, SSE4.2 floor on x86-64.
- Wheels for CPython 3.10 and 3.11, an abi3 wheel for 3.12 and later, and free-threaded wheels for 3.13t and 3.14t.
