// Collision shapes for bodies.
//
// Primitive kinds are named by a kind code and four parameters, following the
// OMI physics shape model:
//   box       size x, size y, size z (full edge lengths, metres)
//   sphere    radius
//   capsule   radius, height of the cylinder between the two hemisphere centres
//   cylinder  radius, total height
//   plane     half extent of the plane's bounds (0 for Jolt's 1 km); the plane
//             passes through the body's origin with the body's +Y as normal,
//             and a plane is static
// Capsules and cylinders run along the body's Y axis. Equal kinds and
// parameters share one Jolt shape.
//
// The other kinds (convex hull, triangle mesh, height field, compound) carry
// arrays, so they are built once by World.add_shape on the calling thread and
// registered under a shape id that bodies name. A body may also scale its
// shape, place it at an offset in the body's frame and move its centre of
// mass; equal decorations of one shape share one Jolt decorated shape.
#pragma once

#include "common.h"

#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace pj {

enum class ShapeKind : uint8_t {
    Box = 1,
    Sphere = 2,
    Capsule = 3,
    Cylinder = 4,
    Plane = 5,
    Convex = 6,
    Mesh = 7,
    HeightField = 8,
    Compound = 9,
};

using ShapeParams = std::array<double, 4>;
using ShapeId = uint32_t;
constexpr ShapeId kNoShape = 0;

inline bool primitive_kind(uint8_t kind) {
    return kind >= uint8_t(ShapeKind::Box) && kind <= uint8_t(ShapeKind::Plane);
}

// The reason the parameters do not make a primitive of this kind, or an empty
// string when they do. A registered kind's problem is that it needs a shape id.
std::string shape_problem(uint8_t kind, const ShapeParams &params);

// False for shapes Jolt cannot give a dynamic body: meshes, height fields,
// planes, and compounds or decorations holding one. Jolt computes no mass
// properties for them.
bool can_be_dynamic(const JPH::Shape &shape);

// Whether the shape holds a mesh or a height field, whose inner edges Jolt's
// enhanced internal edge removal smooths.
bool has_internal_edges(const JPH::Shape &shape);

// A body's decoration of its shape: scaled first, then placed at `offset`
// turned by `rotation` in the body's frame, then its centre of mass moved by
// `com`. The identity decoration leaves the shape as it is.
struct Decoration {
    std::array<float, 3> scale{1.0f, 1.0f, 1.0f};
    std::array<float, 3> offset{0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation{0.0f, 0.0f, 0.0f, 1.0f};
    std::array<float, 3> com{0.0f, 0.0f, 0.0f};
    bool identity() const;
    bool operator<(const Decoration &other) const {
        return std::tie(scale, offset, rotation, com) <
               std::tie(other.scale, other.offset, other.rotation, other.com);
    }
};

// -- builders: any thread; each throws std::invalid_argument with the reason --

// The convex hull of `count` points (x, y, z rows). The hull is rounded by a
// convex radius of at most `max_convex_radius`, which Jolt shrinks for small
// hulls; points may lie `hull_tolerance` outside the hull.
JPH::RefConst<JPH::Shape> make_convex(const double *points, size_t count, float max_convex_radius,
                                      float hull_tolerance);

// A triangle mesh from `count` vertices and `triangles` index triples.
// Degenerate and repeated triangles are dropped; active-edge detection is on.
JPH::RefConst<JPH::Shape> make_mesh(const double *points, size_t count, const uint32_t *indices,
                                    size_t triangles, const uint16_t *materials = nullptr);

struct HeightFieldSpec {
    const float *heights = nullptr;  // rows * columns, row major; NaN is a hole
    size_t rows = 0;                 // along z
    size_t columns = 0;              // along x
    std::array<float, 3> origin{0.0f, 0.0f, 0.0f};  // the position of heights[0, 0]
    std::array<float, 2> spacing{1.0f, 1.0f};       // metres between samples along x, z
    uint32_t block_size = 4;                        // Jolt's block, 2 to 8 samples
    uint32_t bits_per_sample = 8;                   // 1 to 16
    // Heights later edits may reach; NaN where the samples decide.
    float min_height = std::numeric_limits<float>::quiet_NaN();
    float max_height = std::numeric_limits<float>::quiet_NaN();
};

// The size of the square sample grid Jolt is given for `rows` by `columns`
// samples: at least both, a multiple of the block size, and two blocks.
uint32_t heightfield_side(size_t rows, size_t columns, uint32_t block_size);

JPH::RefConst<JPH::Shape> make_heightfield(const HeightFieldSpec &spec);

// Children placed in the body's frame: positions (x, y, z) and rotations
// (x, y, z, w) per child. A mutable compound is Jolt's MutableCompoundShape,
// quicker to build and slower to collide than the static one.
JPH::RefConst<JPH::Shape> make_compound(const std::vector<JPH::RefConst<JPH::Shape>> &children,
                                        const double *positions, const double *rotations,
                                        bool mutable_compound);

// Shapes of every kind. Any thread: primitives and decorations are made on
// first use and cached, registered shapes are added by World.add_shape.
class ShapeLibrary {
public:
    JPH::RefConst<JPH::Shape> primitive(ShapeKind kind, const ShapeParams &params);
    ShapeId add(JPH::RefConst<JPH::Shape> shape, ShapeKind kind);
    // The shape and its kind, or a null shape for an id that names none.
    std::pair<JPH::RefConst<JPH::Shape>, ShapeKind> find(ShapeId id) const;
    // Forgets the id; bodies made with the shape keep it.
    bool remove(ShapeId id);
    JPH::RefConst<JPH::Shape> decorate(const JPH::RefConst<JPH::Shape> &shape,
                                       const Decoration &decoration);
    size_t registered() const;

private:
    mutable std::mutex lock_;
    std::map<std::pair<ShapeKind, ShapeParams>, JPH::RefConst<JPH::Shape>> primitives_;
    std::unordered_map<ShapeId, std::pair<JPH::RefConst<JPH::Shape>, ShapeKind>> registered_;
    std::map<std::pair<const JPH::Shape *, Decoration>, JPH::RefConst<JPH::Shape>> decorated_;
    ShapeId next_ = 1;
};

}  // namespace pj
