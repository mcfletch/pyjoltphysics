// Materials: friction and restitution by material index, combined per contact.
//
// A body carries a material index in the top 16 bits of its user data; a mesh
// triangle may carry its own (an IndexedMaterial in the mesh's material list).
// Index 0 is the body's own friction and restitution, combined as Jolt does
// (the geometric mean of the frictions, the larger restitution). Any other
// index names a row of the world's table: a static and a dynamic friction, a
// restitution and a combine mode for each.
//
// The contact listener calls `combine` for every contact added or persisting
// where either side has a material other than 0. A pair override of the
// (M, M) table wins over the combined values. The static friction applies
// while the bodies slide past each other slower than the table's
// `static_speed` at the contact, the dynamic friction above it.
#pragma once

#include "common.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>

#include <vector>

namespace pj {

// Codes shared with pyjoltphysics.Combine. When the two sides name different
// modes the higher code is used.
enum class Combine : uint8_t { Default = 0, Average = 1, Minimum = 2, Multiply = 3, Maximum = 4 };

struct MaterialRow {
    float static_friction = 0.5f;
    float dynamic_friction = 0.5f;
    float restitution = 0.0f;
    uint8_t friction_combine = uint8_t(Combine::Average);
    uint8_t restitution_combine = uint8_t(Combine::Average);
};

// A mesh triangle's material: Jolt's PhysicsMaterial carrying our index.
class IndexedMaterial final : public JPH::PhysicsMaterial {
public:
    explicit IndexedMaterial(uint16_t index) : index(index) {}
    const uint16_t index;
};

// The process-wide IndexedMaterial for an index, made on first use.
const JPH::PhysicsMaterial *indexed_material(uint16_t index);

// The material index of one side of a contact: the sub-shape's, where it is a
// mesh triangle with an index other than 0, else the body's.
uint16_t material_of(const JPH::Body &body, const JPH::SubShapeID &sub);

// Written only by the world thread with the world lock held exclusively; read
// by Jolt's job threads during the step.
class MaterialTable {
public:
    std::vector<MaterialRow> rows;  // rows[0] is unused: index 0 is the body's own
    // (M, M, 3): static friction, dynamic friction, restitution of a pair; NaN
    // leaves that value to the two materials.
    std::vector<float> pairs;
    float static_speed = 0.05f;  // m/s of sliding below which static friction holds

    // Sets the contact's friction and restitution. `slip` is the sliding speed
    // at the contact, m/s.
    void combine(const JPH::Body &body1, uint16_t m1, const JPH::Body &body2, uint16_t m2, float slip,
                 JPH::ContactSettings &settings) const;
};

}  // namespace pj
