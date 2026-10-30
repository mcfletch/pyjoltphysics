// Collision shapes for bodies, built from a kind code and four parameters.
//
// The kinds and their parameters follow the OMI physics shape model:
//   box       size x, size y, size z (full edge lengths, metres)
//   sphere    radius
//   capsule   radius, height of the cylinder between the two hemisphere centres
//   cylinder  radius, total height
//   plane     half extent of the plane's bounds (0 for Jolt's 1 km); the plane
//             passes through the body's origin with the body's +Y as normal,
//             and a plane is static
// Capsules and cylinders run along the body's Y axis. Equal kinds and
// parameters share one Jolt shape.
#pragma once

#include "common.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <array>
#include <map>
#include <string>

namespace pj {

enum class ShapeKind : uint8_t {
    Box = 1,
    Sphere = 2,
    Capsule = 3,
    Cylinder = 4,
    Plane = 5,
};

using ShapeParams = std::array<double, 4>;

// The reason the parameters do not make a shape of this kind, or an empty
// string when they do. Checked when a batch is submitted.
std::string shape_problem(uint8_t kind, const ShapeParams &params);

// Owned by the world thread.
class ShapeLibrary {
public:
    JPH::RefConst<JPH::Shape> get(ShapeKind kind, const ShapeParams &params);

private:
    std::map<std::pair<ShapeKind, ShapeParams>, JPH::RefConst<JPH::Shape>> made_;
};

}  // namespace pj
