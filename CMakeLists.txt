cmake_minimum_required(VERSION 3.20)
project(pyjoltphysics LANGUAGES CXX)

# Jolt and the binding share one set of JPH_* defines: Jolt's target exports
# them to whatever links it, and a define that differs between the two changes
# the layout of RVec3 and Body.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Jolt is a static library linked into the module, so its symbols are hidden
# with the module's own and nothing but PyInit__native is exported.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(JOLT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics")
if(NOT EXISTS "${JOLT_ROOT}/Build/CMakeLists.txt")
  message(FATAL_ERROR
    "Jolt Physics is missing from ${JOLT_ROOT}. "
    "Run `git submodule update --init` in the checkout.")
endif()

# The physics options every wheel is built with (openglcontext/plans/NATIVE-PHYSICS.md,
# section 7): double precision positions, the cross-platform deterministic
# build (which also turns floating-point contraction off), and an SSE4.2 floor
# on x86-64. ARM64 takes NEON, which Jolt selects from the target.
set(DOUBLE_PRECISION ON CACHE BOOL "" FORCE)
set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
option(PYJOLTPHYSICS_SIMD_FLOOR_SSE42 "x86-64: compile Jolt for SSE4.2 and nothing wider" ON)
if(PYJOLTPHYSICS_SIMD_FLOOR_SSE42)
  foreach(isa USE_AVX USE_AVX2 USE_AVX512 USE_F16C USE_FMADD USE_LZCNT USE_TZCNT)
    set(${isa} OFF CACHE BOOL "" FORCE)
  endforeach()
  # Jolt turns on whichever instruction sets the compiler already targets, so a
  # toolchain whose default is wider than x86-64-v2 (a distribution compiler
  # defaulting to v3, or -march=native in CXXFLAGS) would raise the floor past
  # SSE4.2 whatever the options above say. x86-64-v2 is SSE3, SSSE3, SSE4.1,
  # SSE4.2, POPCNT, CMPXCHG16B and LAHF/SAHF: Intel Nehalem, Silvermont Atom,
  # AMD Bulldozer, Jaguar and every later generation have it; Intel Core 2,
  # Bonnell and Saltwell Atom and AMD K10 and Bobcat do not. It is set before
  # Jolt is added so both the library and the binding take it.
  if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=x86-64-v2 PYJOLTPHYSICS_HAS_X86_64_V2)
    if(PYJOLTPHYSICS_HAS_X86_64_V2)
      string(APPEND CMAKE_CXX_FLAGS " -march=x86-64-v2")
    endif()
  endif()
endif()

# A release wheel carries no assertions, debug renderer or profiler, and Jolt
# does not trap floating-point exceptions inside a host process.
set(USE_ASSERTS OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(FLOATING_POINT_EXCEPTIONS_ENABLED OFF CACHE BOOL "" FORCE)
# nanobind reports C++ exceptions as Python ones and needs RTTI for its types.
set(CPP_RTTI_ENABLED ON CACHE BOOL "" FORCE)
set(CPP_EXCEPTIONS_ENABLED ON CACHE BOOL "" FORCE)
# The module links the C runtime Python does: a static MSVC runtime in Jolt
# and a dynamic one in the module do not link together.
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
# Only the library: no samples, tests, viewer, GPU compute back ends, install
# rules, debug symbols, warnings-as-errors on a compiler Jolt has not seen, or
# link-time optimisation, whose objects a static archive passes on to every
# linker that reads it.
foreach(opt TARGET_UNIT_TESTS TARGET_HELLO_WORLD TARGET_PERFORMANCE_TEST TARGET_SAMPLES
            TARGET_VIEWER JPH_USE_VK JPH_USE_DX12 JPH_USE_MTL JPH_USE_CPU_COMPUTE
            ENABLE_OBJECT_STREAM ENABLE_INSTALL GENERATE_DEBUG_SYMBOLS ENABLE_ALL_WARNINGS
            INTERPROCEDURAL_OPTIMIZATION)
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${JOLT_ROOT}/Build" jolt EXCLUDE_FROM_ALL)

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI gives one abi3 wheel for CPython 3.12 and later; nanobind builds
# an ordinary extension where the interpreter has no stable ABI (3.10, 3.11
# and the free-threaded builds). FREE_THREADED declares the module safe to run
# with the GIL off. NOMINSIZE keeps -O3 for the binding.
# src/native holds the world, one file per concern: the thread and tick
# (world), command batches (commands, apply), snapshots, events, Python
# callbacks, collision layers, shapes and the nanobind classes (bind_world).
set(PYJOLTPHYSICS_SOURCES
  src/_native.cpp
  src/native/apply.cpp
  src/native/bind_world.cpp
  src/native/callbacks.cpp
  src/native/characters.cpp
  src/native/commands.cpp
  src/native/events.cpp
  src/native/fields.cpp
  src/native/joints.cpp
  src/native/jolt_setup.cpp
  src/native/layers.cpp
  src/native/materials.cpp
  src/native/queries.cpp
  src/native/ragdolls.cpp
  src/native/seams.cpp
  src/native/shapes.cpp
  src/native/snapshot.cpp
  src/native/softbodies.cpp
  src/native/threads.cpp
  src/native/vehicles.cpp
  src/native/world.cpp)
nanobind_add_module(_native NB_STATIC STABLE_ABI FREE_THREADED NOMINSIZE
  ${PYJOLTPHYSICS_SOURCES})
target_link_libraries(_native PRIVATE Jolt)

# Jolt's own directory sets these flags for its target only. The binding
# compiles Jolt's inline headers too, so it takes the same floating-point
# rules: no fast-math and no contraction into fused multiply-adds.
if(MSVC)
  target_compile_options(_native PRIVATE /fp:precise /Zc:__cplusplus)
else()
  target_compile_options(_native PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(_native PRIVATE -faligned-allocation)
  endif()
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(_native PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# A sanitizer build for the test suite: -DPYJOLTPHYSICS_SANITIZE=ON compiles
# the module and Jolt with AddressSanitizer and UndefinedBehaviorSanitizer,
# including the float-to-integer conversions that overflow their type, which
# -fsanitize=undefined leaves out. The interpreter then needs libasan
# preloaded (see tox -e asan).
option(PYJOLTPHYSICS_SANITIZE "Build with -fsanitize=address,undefined,float-cast-overflow" OFF)
if(PYJOLTPHYSICS_SANITIZE)
  set(PYJOLTPHYSICS_SANITIZERS "-fsanitize=address,undefined,float-cast-overflow")
  foreach(target _native Jolt)
    target_compile_options(${target} PRIVATE ${PYJOLTPHYSICS_SANITIZERS} -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE ${PYJOLTPHYSICS_SANITIZERS})
  endforeach()
endif()

install(TARGETS _native LIBRARY DESTINATION pyjoltphysics)
