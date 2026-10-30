# Changelog

## 0.1.0a1

- `World`: a Jolt `PhysicsSystem` stepped at `tick_hz` on a native world thread, free-running or in lockstep (`run_ticks`); idle with no pending work; `stop()`, a context manager, collection, `atexit` and fork hooks.
- Command batches as arrays from any thread: `add_bodies` (`BodyBatch`), `remove_bodies`, `set_transforms`, `set_velocities`, `add_forces`, `add_impulses`, `add_torques`, `set_gravity`, `activate`, `set_motion_types`, `set_materials`; sequence numbers, `applied_seq`, `at_tick` and `drain_applied`.
- Shapes: box, sphere, capsule, cylinder and static plane; collision groups and masks; sensors, sleep, CCD and report flags.
- `Snapshot`: read-only NumPy views of a pooled, reference-counted buffer; `latest()`, `wait_tick()`, `changed` rows, `rows`.
- Events as 48-byte `EVENT_DTYPE` records: contact and trigger begin, persist and end, aggregated per body pair, and sleep and wake; sleeping pairs held as touching.
- Callbacks on the world thread: `before_step`, `after_step`, `on_events`, `at_tick`; an exception stops the world and is raised as `WorldError`.

- `jolt_version()`, `build_options()` and `selftest()` over the compiled module `pyjoltphysics._native`.
- Jolt Physics v5.6.0 (commit e77f1755), double precision, cross-platform deterministic, SSE4.2 floor on x86-64.
- Wheels for CPython 3.10 and 3.11, an abi3 wheel for 3.12 and later, and free-threaded wheels for 3.13t and 3.14t.
