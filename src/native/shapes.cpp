#include "shapes.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include <algorithm>
#include <cmath>

namespace pj {

namespace {

bool positive(double value) { return std::isfinite(value) && value > 0.0; }

// Jolt rounds a box or cylinder's edges by a convex radius, which must not
// exceed half the smallest dimension.
float convex_radius(double smallest_half) {
    return float(std::min<double>(JPH::cDefaultConvexRadius, 0.5 * smallest_half));
}

}  // namespace

std::string shape_problem(uint8_t kind, const ShapeParams &p) {
    switch (static_cast<ShapeKind>(kind)) {
    case ShapeKind::Box:
        return positive(p[0]) && positive(p[1]) && positive(p[2])
                   ? ""
                   : "a box needs three positive edge lengths";
    case ShapeKind::Sphere:
        return positive(p[0]) ? "" : "a sphere needs a positive radius";
    case ShapeKind::Capsule:
        return positive(p[0]) && positive(p[1]) ? ""
                                                : "a capsule needs a positive radius and height";
    case ShapeKind::Cylinder:
        return positive(p[0]) && positive(p[1]) ? ""
                                                : "a cylinder needs a positive radius and height";
    case ShapeKind::Plane:
        return std::isfinite(p[0]) && p[0] >= 0.0 ? ""
                                                  : "a plane's half extent is zero or positive";
    }
    return "unknown shape kind " + std::to_string(kind);
}

JPH::RefConst<JPH::Shape> ShapeLibrary::get(ShapeKind kind, const ShapeParams &p) {
    const auto key = std::make_pair(kind, p);
    auto found = made_.find(key);
    if (found != made_.end())
        return found->second;
    JPH::RefConst<JPH::Shape> shape;
    switch (kind) {
    case ShapeKind::Box: {
        const JPH::Vec3 half(float(p[0] / 2), float(p[1] / 2), float(p[2] / 2));
        shape = new JPH::BoxShape(half, convex_radius(half.ReduceMin()));
        break;
    }
    case ShapeKind::Sphere:
        shape = new JPH::SphereShape(float(p[0]));
        break;
    case ShapeKind::Capsule:
        shape = new JPH::CapsuleShape(float(p[1] / 2), float(p[0]));
        break;
    case ShapeKind::Cylinder:
        shape = new JPH::CylinderShape(float(p[1] / 2), float(p[0]),
                                       convex_radius(std::min(p[0], p[1] / 2)));
        break;
    case ShapeKind::Plane: {
        const float half = p[0] > 0.0 ? float(p[0]) : JPH::PlaneShapeSettings::cDefaultHalfExtent;
        shape = new JPH::PlaneShape(JPH::Plane(JPH::Vec3::sAxisY(), 0.0f), nullptr, half);
        break;
    }
    }
    made_.emplace(key, shape);
    return shape;
}

}  // namespace pj
