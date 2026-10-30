# pyjoltphysics API

The package's public names, all in `pyjoltphysics`.

## `jolt_version() -> str`

The version of Jolt Physics compiled into `pyjoltphysics._native`, as
`'major.minor.patch'`: `'5.6.0'`.

## `build_options() -> BuildOptions`

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

## `selftest() -> bool`

Steps a Jolt world holding one 1 kg box, starting 10 m up, once by 1/60 s on
the calling thread with the GIL released. Returns `True` when the step reports
no error and the box has moved down. Raises `RuntimeError` when Jolt's version
ID (its version and feature bits, double precision among them) differs from
the one the binding was compiled against.

## `__version__`

The distribution's version, `'0.1.0a1'`.
