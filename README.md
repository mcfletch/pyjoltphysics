# pyjoltphysics: Jolt Physics for Python

`pyjoltphysics` wraps [Jolt Physics](https://github.com/jrouwe/JoltPhysics) 5.6.0
for Python with [nanobind](https://github.com/wjakob/nanobind). The design steps
a Jolt `PhysicsSystem` on a native thread the module owns, takes commands as
arrays and publishes each tick's poses as read-only NumPy views.

```python
>>> import numpy as np
>>> import pyjoltphysics as pj
>>> with pj.World(pj.WorldSettings(threads=4)) as world:
...     floor = world.add_bodies(pj.BodyBatch.plane())
...     uids, seq = world.add_bodies(pj.BodyBatch.boxes(np.random.uniform(0, 4, (500, 3)) + (0, 2, 0),
...                                                     size=(0.25, 0.25, 0.25)))
...     world.run_ticks(240)                # or world.start() to free-run at tick_hz
...     snapshot = world.latest()           # read-only NumPy views, no copy
...     resting = snapshot.position[snapshot.rows_of(uids), 1]
...     events = world.drain_events()       # contact, trigger, sleep and wake records
...     hit = world.raycast((0, 10, 0), (0, -1, 0), 50.0)   # on this thread, under a shared lock
...     hero = world.add_character((0.0, 1.0, 0.0), pj.CharacterCapabilities(jumpHeight=1.5))
...     hero.set_move((1.0, 0.0, 0.0), 'run')
...     world.run_ticks(60)
...     where = hero.position, hero.state
```

A `World` steps on its own thread at `tick_hz` (120 by default) or in lockstep
with `run_ticks(n)`. Commands are batches of arrays applied at the start of
the next tick; each tick publishes a snapshot of every body's pose and
velocity into a pool of reference-counted buffers, and appends one 64-byte
record per contact begin, end and (on request) persist to an event ring.
Callbacks run on the world thread only when registered. Shapes are boxes,
spheres, capsules, cylinders, static planes, convex hulls, triangle meshes,
height fields (and landscapes of height field tiles) and compounds, each
built once and shared by every body naming its id, with scale, offset and
centre of mass as body parameters. Static meshes and height fields smooth
their inner edges (Jolt's enhanced internal edge removal) unless a body asks
otherwise.

Rays, ray batches, shape sweeps and overlaps run on the calling thread under
the world lock's shared side, with the GIL released: 100,000 scattered rays
take 55 ms. Characters are omi_physics' controller over Jolt's
`CharacterVirtual` (walk, run, sprint, crouch, an exact ballistic jump,
stairs, slopes, slides, knock-downs, landings, pushing, platforms, pass
groups), moved on the world thread every tick and read from the snapshot; 300
of them on a height field take 0.60 ms a 60 Hz frame on 8 job threads.
`swap_bodies` replaces tiles of a streamed landscape in one tick.

Joints (fixed, ball, hinge, slider, distance, six-axis with universal and
two-axis hinge builders, and X3D's motor joint) take limits, springs,
friction and motors, break past a force or torque, and report their angle,
position, force and torque in the snapshot. Materials combine by the OMI
modes, with an (M, M) pair table and static friction below a slip speed.
Collision filters are an (F, F) table changeable at run time. A Python
contact filter is asked on Jolt's job threads about pairs that opted in.
Gravity zones (box, sphere or shape regions; directional or point fields),
quadratic drag and kinematic tracks run natively each tick. Vehicles are
Jolt's vehicle constraint with a wheeled, motorcycle or tracked controller
(suspension, tyre curves, engine, gearbox, differentials, anti-roll bars),
driven by input batches and read from the snapshot with every wheel's pose,
load, slip and the surface under it; ten cars step in 0.05 ms a tick.
`VehicleSettings.from_raycast` maps omi_physics' raycast vehicle tuning.
Ragdolls are Jolt's ragdoll, a body a bone joined by swing-twist limits,
kinematic (following an animated pose exactly), powered (motors per bone
turning it towards the pose, the root held by a limited spring) or limp, with
blends between the modes; the pose target is a held pose or a clip the world
thread evaluates each tick, and every joint's pose is one slice of one
snapshot array. Soft bodies are Jolt's: cloth, ropes, pressure balls and any
triangle mesh, pinned to world points or to moving bodies, blown by wind,
colliding with rigid bodies, their vertices and normals float32 views in the
snapshot rewritten only when they moved. Twenty ragdolls and five cloths of
1,024 vertices take 0.89 ms a 60 Hz tick on 8 job threads.
`docs/API.md` describes every call.

The build information:

```python
>>> import pyjoltphysics
>>> pyjoltphysics.jolt_version()
'5.6.0'
>>> pyjoltphysics.build_options()
BuildOptions(double_precision=True, cross_platform_deterministic=True,
             simd_level='sse4.2', free_threaded=False, debug=False,
             compiler='gcc 13.3.0')
>>> pyjoltphysics.selftest()
True
```

| Function | Returns |
|---|---|
| `jolt_version()` | the Jolt version compiled in, `'5.6.0'` |
| `build_options()` | a frozen `BuildOptions`, fields below |
| `selftest()` | `True` after one 1/60 s step of a world holding one falling box, on the calling thread with the GIL released |

## Build options

Every wheel is built with the same Jolt options.

| `BuildOptions` field | Value in a wheel | Jolt setting |
|---|---|---|
| `double_precision` | `True` | `JPH_DOUBLE_PRECISION`: positions are 64-bit floats, where 32-bit ones resolve 0.98 mm between 8 and 16 km from the origin |
| `cross_platform_deterministic` | `True` | `JPH_CROSS_PLATFORM_DETERMINISTIC`: the same inputs step to the same bits on x86-64 and arm64 |
| `simd_level` | `'sse4.2'` on x86-64, `'neon'` on arm64 | x86-64 compiles SSE4.2 and POPCNT, with AVX, AVX2, F16C, FMA, LZCNT and TZCNT off |
| `free_threaded` | `True` in the cp313t and cp314t wheels | nanobind's `FREE_THREADED`: importing the module leaves the GIL off |
| `debug` | `False` | no `JPH_ENABLE_ASSERTS`, `NDEBUG` defined, no debug renderer or profiler |
| `compiler` | such as `'gcc 14.2.1'` | the compiler that built the module |

The module and Jolt are compiled with floating-point contraction off
(`-ffp-contract=off`, or `/fp:precise` on MSVC) and without fast-math, since a
fused multiply-add changes the result of a step between machines. Jolt is a
static library inside the module, its symbols hidden; the module needs no
system library beyond the C++ runtime.

## Install

```
pip install pyjoltphysics
```

Wheels cover CPython 3.10 and 3.11, one abi3 wheel for CPython 3.12 and later,
and free-threaded CPython 3.13t and 3.14t, on manylinux x86_64 and aarch64,
macOS arm64 and x86_64, and Windows AMD64 and ARM64. The package needs NumPy.

## Build from source

The sdist carries Jolt's library sources. A checkout carries them as a git
submodule:

```
git clone --recurse-submodules https://github.com/mcfletch/pyjoltphysics
pip install ./pyjoltphysics
```

A build needs a C++17 compiler; scikit-build-core fetches CMake and Ninja
where the machine has none. In an existing checkout,
`git submodule update --init` fetches Jolt.

The C++ is `src/_native.cpp` (the module and its build information) and
`src/native/`, a file per concern: `world` (the thread and the tick),
`commands` and `apply` (batches), `snapshot`, `events`, `callbacks` (Python on
the world thread), `layers` (collision groups), `shapes`, `queries` (rays,
sweeps, overlaps), `characters` (the controller over `CharacterVirtual`),
`vehicles` (the vehicle constraint and its inputs, forces and blows),
`ragdolls` (Jolt's ragdoll, its modes and pose targets), `softbodies` (soft
bodies, their pins, wind and vertex cache),
`seams` (edge removal across static bodies laid flush) and `bind_world` (the
nanobind classes).

An editable install keeps a build directory per wheel tag under `build/`, so
reinstalling after a change to a source file recompiles that file only:

```
pip install -e '.[dev]'
```

`src/pyjoltphysics/_native.pyi` is generated by nanobind from the built
module and committed. After a change to the module's signatures:

```
python -m nanobind.stubgen -m pyjoltphysics._native -o src/pyjoltphysics/_native.pyi
```

`tests/test_stub.py` and the CI `stubs` job fail when the committed stub and
the generated one differ.

## Tests

```
pytest
tox                 # lint, typecheck, stubs and the suite per interpreter
tox -e asan         # the suite against an AddressSanitizer and UBSan build (Linux, gcc)
python benchmarks/boxes5000.py --threads 1 8
python benchmarks/statics.py --threads 1 8
python benchmarks/characters.py --threads 1 8     # characters, tile streaming, rays, query waits
python benchmarks/vehicles.py --threads 1 8       # ten cars on a height field
python benchmarks/ragdolls_cloth.py --threads 1 8 # 20 ragdolls and 5 cloths of 1,024 vertices
```

`tests/differential/` runs each scene through Jolt and through `omi_physics`
and compares them within the tolerance each test states: shapes resting on
box, mesh and height field grounds, slides by friction, bounces, stacks, a
pyramid, stairs, triggers, continuous detection, a bin of 1,000 boxes, a
compound, and characters (a run across a height field, stairs, slopes, a jump
at gravity 42, a fall) against omi_physics' controller, and glisteel's car
(settling, acceleration, braking, cornering, a jump, a wall, a banked road,
bumps) against omi_physics' raycast vehicle. It is marked `differential` and skipped where `omi_physics` is not
installed. omi_physics has no ragdoll or soft body, so `tests/test_ragdolls.py`
and `tests/test_softbodies.py` hold those to analytic expectations: rest
within the joint limits, the animated pose to 1 mm, a hanging cloth's pins to
0.1 mm, no vertex inside a sphere by more than its radius, a rope's length.

## Licence

pyjoltphysics is MIT licensed; see `LICENSE`. The compiled module carries Jolt
Physics (MIT, Jorrit Rouwe) and the nanobind runtime (BSD 3-Clause, Wenzel
Jakob); `NOTICE` names both, and the wheel carries their licence files.
