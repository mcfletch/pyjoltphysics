// pyjoltphysics._native: the compiled module over Jolt Physics.
//
// This file reports how Jolt was built, steps a one-body world once on the
// calling thread, and registers the world classes from src/native.
#include "native/bindings.h"
#include "native/common.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

// One object layer and one broad-phase layer: the self-test world has a single
// body, so every pair collides.
constexpr JPH::ObjectLayer kLayer = 0;
constexpr JPH::uint kBroadPhaseLayers = 1;

class SingleBroadPhaseLayer final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return kBroadPhaseLayers; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer) const override {
        return JPH::BroadPhaseLayer(0);
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return "all"; }
#endif
};

class CollideAll final : public JPH::ObjectVsBroadPhaseLayerFilter,
                         public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer, JPH::BroadPhaseLayer) const override { return true; }
    bool ShouldCollide(JPH::ObjectLayer, JPH::ObjectLayer) const override { return true; }
};

std::string jolt_version() {
    return std::to_string(JPH_VERSION_MAJOR) + "." + std::to_string(JPH_VERSION_MINOR) + "." +
           std::to_string(JPH_VERSION_PATCH);
}

const char *simd_level() {
#if defined(JPH_USE_AVX512)
    return "avx512";
#elif defined(JPH_USE_AVX2)
    return "avx2";
#elif defined(JPH_USE_AVX)
    return "avx";
#elif defined(JPH_USE_SSE4_2)
    return "sse4.2";
#elif defined(JPH_USE_NEON)
    return "neon";
#else
    return "none";
#endif
}

std::string compiler() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

nb::dict build_options() {
    nb::dict options;
#if defined(JPH_DOUBLE_PRECISION)
    options["double_precision"] = true;
#else
    options["double_precision"] = false;
#endif
#if defined(JPH_CROSS_PLATFORM_DETERMINISTIC)
    options["cross_platform_deterministic"] = true;
#else
    options["cross_platform_deterministic"] = false;
#endif
    options["simd_level"] = simd_level();
#if defined(Py_GIL_DISABLED)
    options["free_threaded"] = true;
#else
    options["free_threaded"] = false;
#endif
#if defined(JPH_ENABLE_ASSERTS) || !defined(NDEBUG)
    options["debug"] = true;
#else
    options["debug"] = false;
#endif
    options["compiler"] = compiler();
    return options;
}

// Drop a 1 kg box from 10 m and step it once at 60 Hz. The step succeeds when
// Jolt reports no error and gravity has moved the box down.
bool selftest() {
    pj::register_jolt();
    if (!JPH::VerifyJoltVersionID())
        throw std::runtime_error("Jolt was built with options that differ from the binding's");

    constexpr JPH::uint kMaxBodies = 16;
    constexpr JPH::uint kMaxPairs = 16;
    constexpr JPH::uint kMaxContacts = 16;
    constexpr float kStep = 1.0f / 60.0f;
    constexpr JPH::Real kStartHeight = 10.0;

    JPH::TempAllocatorImpl temp(1024 * 1024);
    JPH::JobSystemSingleThreaded jobs(JPH::cMaxPhysicsJobs);
    SingleBroadPhaseLayer layers;
    CollideAll filter;
    JPH::PhysicsSystem system;
    system.Init(kMaxBodies, 0, kMaxPairs, kMaxContacts, layers, filter, filter);

    JPH::BodyInterface &bodies = system.GetBodyInterface();
    JPH::BodyCreationSettings box(new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f)),
                                  JPH::RVec3(0.0, kStartHeight, 0.0), JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Dynamic, kLayer);
    const JPH::BodyID id = bodies.CreateAndAddBody(box, JPH::EActivation::Activate);
    if (id.IsInvalid())
        throw std::runtime_error("Jolt could not create the self-test body");

    const JPH::EPhysicsUpdateError error = system.Update(kStep, 1, &temp, &jobs);
    const JPH::RVec3 position = bodies.GetCenterOfMassPosition(id);
    bodies.RemoveBody(id);
    bodies.DestroyBody(id);

    if (error != JPH::EPhysicsUpdateError::None)
        throw std::runtime_error("Jolt reported an error stepping the self-test world");
    return position.GetY() < kStartHeight;
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "The compiled pyjoltphysics module over Jolt Physics.";
    m.def("jolt_version", &jolt_version,
          "Return the version of the Jolt Physics compiled into the module, such as '5.6.0'.");
    m.def("build_options", &build_options, nb::sig("def build_options() -> dict[str, bool | str]"),
          "Return the options the module and its Jolt were built with, by name.");
    m.def("selftest", &selftest, nb::call_guard<nb::gil_scoped_release>(),
          "Return True after stepping a one-body Jolt world once on the calling thread.");
    pj::bind_world(m);
}
