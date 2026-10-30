#include "materials.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <algorithm>
#include <cmath>
#include <mutex>

namespace pj {

namespace {

std::mutex registry_lock;
std::vector<JPH::Ref<IndexedMaterial>> registry;

float combine_values(float a, float b, uint8_t mode, bool friction) {
    switch (static_cast<Combine>(mode)) {
    case Combine::Average:
        return 0.5f * (a + b);
    case Combine::Minimum:
        return std::min(a, b);
    case Combine::Multiply:
        return a * b;
    case Combine::Maximum:
        return std::max(a, b);
    case Combine::Default:
        break;
    }
    return friction ? std::sqrt(a * b) : std::max(a, b);
}

}  // namespace

const JPH::PhysicsMaterial *indexed_material(uint16_t index) {
    std::lock_guard<std::mutex> guard(registry_lock);
    if (registry.size() <= index)
        registry.resize(size_t(index) + 1);
    if (registry[index] == nullptr)
        registry[index] = new IndexedMaterial(index);
    return registry[index].GetPtr();
}

uint16_t material_of(const JPH::Body &body, const JPH::SubShapeID &sub) {
    const JPH::PhysicsMaterial *material = body.GetShape()->GetMaterial(sub);
    if (material != JPH::PhysicsMaterial::sDefault.GetPtr()) {
        if (const auto *indexed = dynamic_cast<const IndexedMaterial *>(material); indexed && indexed->index)
            return indexed->index;
    }
    return material_bits_of(body.GetUserData());
}

void MaterialTable::combine(const JPH::Body &body1, uint16_t m1, const JPH::Body &body2, uint16_t m2, float slip,
                            JPH::ContactSettings &settings) const {
    auto side = [&](const JPH::Body &body, uint16_t m) {
        if (m != 0 && m < rows.size())
            return rows[m];
        MaterialRow own;
        own.static_friction = own.dynamic_friction = body.GetFriction();
        own.restitution = body.GetRestitution();
        own.friction_combine = own.restitution_combine = uint8_t(Combine::Default);
        return own;
    };
    const MaterialRow a = side(body1, m1);
    const MaterialRow b = side(body2, m2);
    const uint8_t fmode = std::max(a.friction_combine, b.friction_combine);
    const uint8_t rmode = std::max(a.restitution_combine, b.restitution_combine);
    float static_friction = combine_values(a.static_friction, b.static_friction, fmode, true);
    float dynamic_friction = combine_values(a.dynamic_friction, b.dynamic_friction, fmode, true);
    float restitution = combine_values(a.restitution, b.restitution, rmode, false);
    const size_t count = rows.size();
    if (m1 < count && m2 < count && !pairs.empty()) {
        const float *pair = &pairs[(size_t(m1) * count + m2) * 3];
        if (!std::isnan(pair[0]))
            static_friction = pair[0];
        if (!std::isnan(pair[1]))
            dynamic_friction = pair[1];
        if (!std::isnan(pair[2]))
            restitution = pair[2];
    }
    settings.mCombinedFriction = slip < static_speed ? static_friction : dynamic_friction;
    settings.mCombinedRestitution = restitution;
}

}  // namespace pj
