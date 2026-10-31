// Soft bodies: Jolt's soft-body simulation of particles held by edge, shear,
// bend, volume and long-range attachment constraints, with an optional
// pressure inside a closed surface. Cloth, ropes and pressure balls are all
// soft bodies; they differ in their vertices, faces and constraints, which
// pyjoltphysics.softbodies builds.
//
// A soft body is a body: its uid is assigned when the batch is submitted and
// it is a row of every snapshot, at its centre of mass. Its vertices are
// relative to that point, in float32, so a renderer feeds them to a vertex
// buffer and places the buffer at the row's position. After each tick the
// soft bodies Jolt moved copy their vertices, and their normals where they
// have faces, into a cache stamped with the tick; a snapshot buffer copies a
// body's cache only when its stamp differs from the one the buffer holds.
//
// A pinned vertex has no inverse mass and follows its target: a world point,
// or a point in a body's frame, reached by the end of each tick. Wind is a
// force on the whole body, which Jolt shares among its vertices by their
// inverse mass, applied before each step.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

#include <functional>
#include <memory>
#include <vector>

namespace JPH {
class PhysicsSystem;
}

namespace pj {

struct SnapshotBuffer;

// The scalar settings of a soft body, in the order pyjoltphysics.softbodies sends them.
enum SoftParam : uint32_t {
    kSoftX, kSoftY, kSoftZ, kSoftQX, kSoftQY, kSoftQZ, kSoftQW,
    kSoftCompliance,         // edges, 1/N; 0 is rigid
    kSoftShearCompliance,
    kSoftBendCompliance,     // inf: no bend constraints
    kSoftBendType,           // 0 none, 1 distance, 2 dihedral
    kSoftVolumeCompliance,   // tetrahedra
    kSoftPressure,           // n R T, Pa m3; 0 for none
    kSoftLra,                // 0 none, 1 Euclidean, 2 geodesic
    kSoftLraStretch,         // multiplier on the rest distance
    kSoftIterations,
    kSoftLinearDamping,      // 1/s
    kSoftMaxVelocity,        // m/s
    kSoftFriction, kSoftRestitution, kSoftGravityFactor,
    kSoftVertexRadius,       // m
    kSoftAllowSleep, kSoftDoubleSided,
    kSoftGroup, kSoftMask, kSoftFilter, kSoftMaterial, kSoftFlags,
    kSoftParamCount,
};

struct SoftPin {
    uint32_t vertex = 0;
    Uid body = kNoBody;              // 0 pins to a world point
    double point[3] = {0, 0, 0};     // in the body's frame, or the world
};

struct SoftBodyDesc {
    Uid uid = kNoBody;                // assigned when the batch is submitted
    std::vector<double> params;       // kSoftParamCount
    std::vector<float> positions;     // (V, 3), in the body's frame
    std::vector<float> inv_mass;      // (V,) 1/kg
    std::vector<uint32_t> faces;      // (F, 3)
    std::vector<uint32_t> edges;      // (E, 2); empty: from the faces
    std::vector<float> edge_compliance;  // (E,)
    std::vector<uint32_t> tetrahedra; // (T, 4)
    std::vector<SoftPin> pins;
    // Built from the rest when the batch is submitted, on the submitting thread.
    JPH::Ref<JPH::SoftBodySharedSettings> shared;
    size_t vertex_count() const { return positions.size() / 3; }
};

struct AddSoftBody {
    SoftBodyDesc desc;
};

// Replaces a soft body's pins.
struct SetSoftPins {
    Uid uid = kNoBody;
    std::vector<SoftPin> pins;
};

struct SetWind {
    std::vector<Uid> uids;
    std::vector<double> force;  // (n, 3) N
};

// The shared settings a description makes: vertices, faces and constraints,
// optimised for Jolt's parallel solver. Throws std::invalid_argument.
JPH::Ref<JPH::SoftBodySharedSettings> soft_settings(const SoftBodyDesc &desc);

struct SoftBody {
    Uid uid = kNoBody;
    JPH::BodyID id;
    std::vector<float> inv_mass;     // as made, for unpinning
    std::vector<uint32_t> faces;
    std::vector<SoftPin> pins;
    JPH::Vec3 wind = JPH::Vec3::sZero();
    // The cache the snapshot copies.
    std::vector<float> positions, normals;  // (V, 3)
    double origin[3] = {0, 0, 0};
    uint64_t version = 0;
};

class SoftBodySystem {
public:
    using Resolve = std::function<JPH::BodyID(Uid)>;
    explicit SoftBodySystem(JPH::PhysicsSystem &system) : system_(system) {}

    // On the world thread with the world lock held exclusively.
    void add(const SoftBodyDesc &desc, const JPH::BodyID &id, uint64_t tick);
    // Forgets soft bodies among `uids`, and pins to bodies among them.
    void remove(const std::vector<Uid> &uids);
    bool set_pins(Uid uid, std::vector<SoftPin> pins);
    bool set_wind(Uid uid, JPH::Vec3Arg force);
    // Pins moved to their targets for the end of the step, and wind.
    void before_step(double dt, const Resolve &resolve);
    // Caches the vertices of the soft bodies that moved, stamped `tick`.
    void after_step(uint64_t tick);
    void write(SnapshotBuffer &buffer) const;
    bool empty() const { return bodies_.empty(); }
    size_t size() const { return bodies_.size(); }
    // Whether a pin or wind keeps a lockstep world stepping.
    bool driving() const;

private:
    SoftBody *find(Uid uid);
    void refresh(SoftBody &body, uint64_t tick, bool always);

    JPH::PhysicsSystem &system_;
    std::vector<std::unique_ptr<SoftBody>> bodies_;  // ascending uid
    uint64_t layout_ = 1;  // changes when a soft body is added or removed
};

}  // namespace pj
