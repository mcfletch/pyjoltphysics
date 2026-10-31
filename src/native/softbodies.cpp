#include "softbodies.h"

#include "snapshot.h"

#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace pj {

namespace {

using Shared = JPH::SoftBodySharedSettings;

float compliance(double value) {
    return std::isinf(value) ? std::numeric_limits<float>::max() : float(value);
}

}  // namespace

JPH::Ref<JPH::SoftBodySharedSettings> soft_settings(const SoftBodyDesc &desc) {
    const size_t n = desc.vertex_count();
    const double *p = desc.params.data();
    JPH::Ref<Shared> out = new Shared;
    out->mVertices.resize(n);
    for (size_t i = 0; i < n; ++i) {
        Shared::Vertex &v = out->mVertices[i];
        v.mPosition = JPH::Float3(desc.positions[3 * i], desc.positions[3 * i + 1], desc.positions[3 * i + 2]);
        v.mInvMass = desc.inv_mass[i];
    }
    for (const SoftPin &pin : desc.pins)
        out->mVertices[pin.vertex].mInvMass = 0.0f;
    for (size_t f = 0; f + 2 < desc.faces.size(); f += 3) {
        Shared::Face face(desc.faces[f], desc.faces[f + 1], desc.faces[f + 2]);
        if (face.IsDegenerate())
            throw std::invalid_argument("face " + std::to_string(f / 3) + " names one vertex twice");
        out->AddFace(face);
    }
    const auto lra = int(p[kSoftLra]);
    if (desc.edges.empty() && !desc.faces.empty()) {
        Shared::VertexAttributes attributes;
        attributes.mCompliance = compliance(p[kSoftCompliance]);
        attributes.mShearCompliance = compliance(p[kSoftShearCompliance]);
        attributes.mBendCompliance = compliance(p[kSoftBendCompliance]);
        attributes.mLRAType = lra == 2   ? Shared::ELRAType::GeodesicDistance
                              : lra == 1 ? Shared::ELRAType::EuclideanDistance
                                         : Shared::ELRAType::None;
        attributes.mLRAMaxDistanceMultiplier = float(p[kSoftLraStretch]);
        const auto bend = int(p[kSoftBendType]);
        out->CreateConstraints(&attributes, 1,
                               bend == 2   ? Shared::EBendType::Dihedral
                               : bend == 1 ? Shared::EBendType::Distance
                                           : Shared::EBendType::None);
    } else {
        for (size_t e = 0; e < desc.edge_compliance.size(); ++e)
            out->mEdgeConstraints.emplace_back(desc.edges[2 * e], desc.edges[2 * e + 1], desc.edge_compliance[e]);
        out->CalculateEdgeLengths();
        if (lra != 0) {
            // Each free vertex is held within reach of its nearest pinned vertex.
            std::vector<uint32_t> pinned;
            for (size_t i = 0; i < n; ++i)
                if (out->mVertices[i].mInvMass == 0.0f)
                    pinned.push_back(uint32_t(i));
            if (!pinned.empty()) {
                for (size_t i = 0; i < n; ++i) {
                    if (out->mVertices[i].mInvMass == 0.0f)
                        continue;
                    const JPH::Vec3 at(out->mVertices[i].mPosition);
                    uint32_t best = pinned[0];
                    float nearest = std::numeric_limits<float>::max();
                    for (uint32_t k : pinned) {
                        const float d = (JPH::Vec3(out->mVertices[k].mPosition) - at).LengthSq();
                        if (d < nearest) {
                            nearest = d;
                            best = k;
                        }
                    }
                    out->mLRAConstraints.emplace_back(best, uint32_t(i), 0.0f);
                }
                out->CalculateLRALengths(float(p[kSoftLraStretch]));
            }
        }
    }
    for (size_t t = 0; t + 3 < desc.tetrahedra.size(); t += 4)
        out->mVolumeConstraints.emplace_back(desc.tetrahedra[t], desc.tetrahedra[t + 1], desc.tetrahedra[t + 2],
                                             desc.tetrahedra[t + 3], compliance(p[kSoftVolumeCompliance]));
    if (!out->mVolumeConstraints.empty())
        out->CalculateVolumeConstraintVolumes();
    out->Optimize();
    return out;
}

SoftBody *SoftBodySystem::find(Uid uid) {
    auto at = std::lower_bound(bodies_.begin(), bodies_.end(), uid,
                               [](const std::unique_ptr<SoftBody> &b, Uid v) { return b->uid < v; });
    return at != bodies_.end() && (*at)->uid == uid ? at->get() : nullptr;
}

void SoftBodySystem::add(const SoftBodyDesc &desc, const JPH::BodyID &id, uint64_t tick) {
    auto body = std::make_unique<SoftBody>();
    body->uid = desc.uid;
    body->id = id;
    body->inv_mass = desc.inv_mass;
    body->faces = desc.faces;
    body->pins = desc.pins;
    refresh(*body, tick, true);
    auto at = std::lower_bound(bodies_.begin(), bodies_.end(), desc.uid,
                               [](const std::unique_ptr<SoftBody> &b, Uid v) { return b->uid < v; });
    bodies_.insert(at, std::move(body));
    ++layout_;
}

void SoftBodySystem::remove(const std::vector<Uid> &uids) {
    const size_t before = bodies_.size();
    bodies_.erase(std::remove_if(bodies_.begin(), bodies_.end(),
                                 [&](const std::unique_ptr<SoftBody> &b) {
                                     return std::find(uids.begin(), uids.end(), b->uid) != uids.end();
                                 }),
                  bodies_.end());
    if (bodies_.size() != before)
        ++layout_;
    for (auto &body : bodies_) {
        auto gone = [&](const SoftPin &pin) {
            return pin.body != kNoBody && std::find(uids.begin(), uids.end(), pin.body) != uids.end();
        };
        if (std::any_of(body->pins.begin(), body->pins.end(), gone)) {
            std::vector<SoftPin> kept;
            for (const SoftPin &pin : body->pins)
                if (!gone(pin))
                    kept.push_back(pin);
            set_pins(body->uid, std::move(kept));
        }
    }
}

bool SoftBodySystem::set_pins(Uid uid, std::vector<SoftPin> pins) {
    SoftBody *body = find(uid);
    if (body == nullptr)
        return false;
    for (const SoftPin &pin : pins)
        if (pin.vertex >= body->inv_mass.size())
            return false;
    JPH::BodyLockWrite lock(system_.GetBodyLockInterfaceNoLock(), body->id);
    if (!lock.Succeeded())
        return false;
    auto *motion = static_cast<JPH::SoftBodyMotionProperties *>(lock.GetBody().GetMotionProperties());
    JPH::Array<JPH::SoftBodyVertex> &vertices = motion->GetVertices();
    for (const SoftPin &pin : body->pins) {
        vertices[pin.vertex].mInvMass = body->inv_mass[pin.vertex];
        vertices[pin.vertex].mVelocity = JPH::Vec3::sZero();
    }
    for (const SoftPin &pin : pins)
        vertices[pin.vertex].mInvMass = 0.0f;
    body->pins = std::move(pins);
    system_.GetBodyInterfaceNoLock().ActivateBody(body->id);
    return true;
}

bool SoftBodySystem::set_wind(Uid uid, JPH::Vec3Arg force) {
    SoftBody *body = find(uid);
    if (body == nullptr)
        return false;
    body->wind = force;
    return true;
}

bool SoftBodySystem::driving() const {
    for (const auto &body : bodies_)
        if (body->wind != JPH::Vec3::sZero())
            return true;
    return false;
}

void SoftBodySystem::before_step(double dt, const Resolve &resolve) {
    JPH::BodyInterface &interface = system_.GetBodyInterfaceNoLock();
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const float inv_dt = float(1.0 / dt);
    for (auto &owned : bodies_) {
        SoftBody &body = *owned;
        if (body.wind != JPH::Vec3::sZero())
            interface.AddForce(body.id, body.wind, JPH::EActivation::Activate);
        if (body.pins.empty())
            continue;
        bool moved = false;
        {
            JPH::Body *soft = locks.TryGetBody(body.id);
            if (soft == nullptr)
                continue;
            auto *motion = static_cast<JPH::SoftBodyMotionProperties *>(soft->GetMotionProperties());
            JPH::Array<JPH::SoftBodyVertex> &vertices = motion->GetVertices();
            const JPH::RMat44 to_local = soft->GetInverseCenterOfMassTransform();
            for (const SoftPin &pin : body.pins) {
                JPH::RVec3 target(pin.point[0], pin.point[1], pin.point[2]);
                if (pin.body != kNoBody) {
                    const JPH::Body *carrier = locks.TryGetBody(resolve(pin.body));
                    if (carrier == nullptr)
                        continue;
                    // Where the carrier will be at the end of the step, as its velocity takes it.
                    const JPH::RVec3 at = carrier->GetWorldTransform() * target;
                    target = at + JPH::RVec3(carrier->GetPointVelocity(at) * float(dt));
                }
                JPH::SoftBodyVertex &v = vertices[pin.vertex];
                const JPH::Vec3 local(to_local * target);
                v.mVelocity = (local - v.mPosition) * inv_dt;
                moved = moved || v.mVelocity.LengthSq() > 0.0f;
            }
        }
        if (moved && !interface.IsActive(body.id))
            interface.ActivateBody(body.id);
    }
}

void SoftBodySystem::refresh(SoftBody &body, uint64_t tick, bool always) {
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    const JPH::Body *soft = locks.TryGetBody(body.id);
    if (soft == nullptr)
        return;
    const JPH::RVec3 origin = soft->GetCenterOfMassPosition();
    if (!always && !soft->IsActive() && origin.GetX() == body.origin[0] && origin.GetY() == body.origin[1] &&
        origin.GetZ() == body.origin[2])
        return;
    const auto *motion = static_cast<const JPH::SoftBodyMotionProperties *>(soft->GetMotionProperties());
    const JPH::Array<JPH::SoftBodyVertex> &vertices = motion->GetVertices();
    const JPH::Quat rotation = soft->GetRotation();
    const size_t n = vertices.size();
    body.positions.resize(3 * n);
    body.normals.assign(3 * n, 0.0f);
    std::vector<JPH::Vec3> world(n);
    for (size_t i = 0; i < n; ++i) {
        world[i] = rotation * vertices[i].mPosition;
        world[i].StoreFloat3(reinterpret_cast<JPH::Float3 *>(&body.positions[3 * i]));
    }
    if (!body.faces.empty()) {
        std::vector<JPH::Vec3> sum(n, JPH::Vec3::sZero());
        for (size_t f = 0; f + 2 < body.faces.size(); f += 3) {
            const uint32_t a = body.faces[f], b = body.faces[f + 1], c = body.faces[f + 2];
            // The cross product's length is twice the face's area, which weights the vertex normals.
            const JPH::Vec3 normal = (world[b] - world[a]).Cross(world[c] - world[a]);
            sum[a] += normal;
            sum[b] += normal;
            sum[c] += normal;
        }
        for (size_t i = 0; i < n; ++i)
            sum[i].NormalizedOr(JPH::Vec3::sZero()).StoreFloat3(reinterpret_cast<JPH::Float3 *>(&body.normals[3 * i]));
    }
    body.origin[0] = origin.GetX();
    body.origin[1] = origin.GetY();
    body.origin[2] = origin.GetZ();
    body.version = tick;
}

void SoftBodySystem::after_step(uint64_t tick) {
    for (auto &body : bodies_)
        refresh(*body, tick, false);
}

void SoftBodySystem::write(SnapshotBuffer &out) const {
    const bool whole = out.soft_layout != layout_;
    if (whole) {
        size_t vertices = 0;
        for (const auto &body : bodies_)
            vertices += body->positions.size() / 3;
        out.resize_soft_bodies(uint32_t(bodies_.size()), uint32_t(vertices));
        out.soft_layout = layout_;
    }
    uint32_t first = 0;
    for (size_t i = 0; i < bodies_.size(); ++i) {
        const SoftBody &body = *bodies_[i];
        const auto count = uint32_t(body.positions.size() / 3);
        if (whole) {
            out.soft_uid[i] = body.uid;
            out.soft_first_vertex[i] = first;
            out.soft_vertices[i] = count;
            out.soft_version[i] = std::numeric_limits<uint64_t>::max();
        }
        if (out.soft_version[i] != body.version) {
            out.soft_version[i] = body.version;
            std::memcpy(&out.soft_origin[3 * i], body.origin, sizeof(body.origin));
            std::memcpy(&out.soft_position[size_t(first) * 3], body.positions.data(), body.positions.size() * sizeof(float));
            std::memcpy(&out.soft_normal[size_t(first) * 3], body.normals.data(), body.normals.size() * sizeof(float));
        }
        first += count;
    }
}

}  // namespace pj
