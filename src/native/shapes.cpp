#include "shapes.h"

#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/DecoratedShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/MutableCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pj {

namespace {

bool positive(double value) { return std::isfinite(value) && value > 0.0; }

// Jolt rounds a box or cylinder's edges by a convex radius, which must not
// exceed half the smallest dimension.
float convex_radius(double smallest_half) {
    return float(std::min<double>(JPH::cDefaultConvexRadius, 0.5 * smallest_half));
}

JPH::RefConst<JPH::Shape> created(const JPH::ShapeSettings::ShapeResult &result, const char *what) {
    if (result.HasError())
        throw std::invalid_argument(std::string(what) + ": " + result.GetError().c_str());
    return result.Get();
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
    case ShapeKind::Convex:
    case ShapeKind::Mesh:
    case ShapeKind::HeightField:
    case ShapeKind::Compound:
        return "a convex hull, mesh, height field or compound is named by the shape id "
               "World.add_shape returns";
    }
    return "unknown shape kind " + std::to_string(kind);
}

bool can_be_dynamic(const JPH::Shape &shape) {
    switch (shape.GetSubType()) {
    case JPH::EShapeSubType::Mesh:
    case JPH::EShapeSubType::HeightField:
    case JPH::EShapeSubType::Plane:
        return false;
    default:
        break;
    }
    if (shape.GetType() == JPH::EShapeType::Decorated)
        return can_be_dynamic(*static_cast<const JPH::DecoratedShape &>(shape).GetInnerShape());
    if (shape.GetType() == JPH::EShapeType::Compound) {
        for (const auto &child : static_cast<const JPH::CompoundShape &>(shape).GetSubShapes())
            if (!can_be_dynamic(*child.mShape))
                return false;
    }
    return true;
}

bool has_internal_edges(const JPH::Shape &shape) {
    switch (shape.GetSubType()) {
    case JPH::EShapeSubType::Mesh:
    case JPH::EShapeSubType::HeightField:
        return true;
    default:
        break;
    }
    if (shape.GetType() == JPH::EShapeType::Decorated)
        return has_internal_edges(*static_cast<const JPH::DecoratedShape &>(shape).GetInnerShape());
    if (shape.GetType() == JPH::EShapeType::Compound)
        for (const auto &child : static_cast<const JPH::CompoundShape &>(shape).GetSubShapes())
            if (has_internal_edges(*child.mShape))
                return true;
    return false;
}

bool Decoration::identity() const {
    return scale == std::array<float, 3>{1.0f, 1.0f, 1.0f} &&
           offset == std::array<float, 3>{0.0f, 0.0f, 0.0f} &&
           rotation == std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f} &&
           com == std::array<float, 3>{0.0f, 0.0f, 0.0f};
}

// -- builders ----------------------------------------------------------------------

JPH::RefConst<JPH::Shape> make_convex(const double *points, size_t count, float max_convex_radius,
                                      float hull_tolerance) {
    if (count < 4)
        throw std::invalid_argument("a convex hull needs at least 4 points, got " +
                                    std::to_string(count));
    JPH::Array<JPH::Vec3> cloud;
    cloud.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const double *p = points + 3 * i;
        if (!(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])))
            throw std::invalid_argument("a convex hull's points are finite");
        cloud.push_back(JPH::Vec3(float(p[0]), float(p[1]), float(p[2])));
    }
    if (!(std::isfinite(max_convex_radius) && max_convex_radius >= 0.0f) ||
        !(std::isfinite(hull_tolerance) && hull_tolerance > 0.0f))
        throw std::invalid_argument(
            "a convex hull's radius is zero or positive and its tolerance positive");
    // Points on a plane or a line enclose no volume; Jolt's builder reports
    // that as a failure whose text names its internals, so it is checked here.
    JPH::Vec3 lo = cloud[0], hi = cloud[0];
    for (const JPH::Vec3 &p : cloud) {
        lo = JPH::Vec3::sMin(lo, p);
        hi = JPH::Vec3::sMax(hi, p);
    }
    const float span = (hi - lo).Length();
    float volume = 0.0f;
    if (span > 0.0f) {
        // The largest tetrahedron found from the extreme points along each axis.
        const JPH::Vec3 a = cloud[0];
        size_t far = 0;
        for (size_t i = 0; i < cloud.size(); ++i)
            if ((cloud[i] - a).LengthSq() > (cloud[far] - a).LengthSq())
                far = i;
        const JPH::Vec3 ab = cloud[far] - a;
        size_t wide = 0;
        for (size_t i = 0; i < cloud.size(); ++i)
            if (ab.Cross(cloud[i] - a).LengthSq() > ab.Cross(cloud[wide] - a).LengthSq())
                wide = i;
        const JPH::Vec3 normal = ab.Cross(cloud[wide] - a);
        for (const JPH::Vec3 &p : cloud)
            volume = std::max(volume, std::abs(normal.Dot(p - a)));
        volume /= 6.0f;
    }
    if (!(span > 0.0f) || volume <= 1e-9f * span * span * span)
        throw std::invalid_argument("a convex hull's points lie on a plane, a line or a point "
                                    "and enclose no volume");
    JPH::ConvexHullShapeSettings settings(cloud, max_convex_radius);
    settings.mHullTolerance = hull_tolerance;
    return created(settings.Create(), "convex hull");
}

JPH::RefConst<JPH::Shape> make_mesh(const double *points, size_t count, const uint32_t *indices,
                                    size_t triangles) {
    if (triangles == 0)
        throw std::invalid_argument("a mesh needs at least one triangle");
    JPH::VertexList vertices;
    vertices.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const double *p = points + 3 * i;
        if (!(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])))
            throw std::invalid_argument("a mesh's points are finite");
        vertices.push_back(JPH::Float3(float(p[0]), float(p[1]), float(p[2])));
    }
    JPH::IndexedTriangleList list;
    list.reserve(triangles);
    for (size_t t = 0; t < triangles; ++t) {
        const uint32_t *tri = indices + 3 * t;
        for (int k = 0; k < 3; ++k)
            if (tri[k] >= count)
                throw std::invalid_argument("triangle " + std::to_string(t) + " names vertex " +
                                            std::to_string(tri[k]) + " of " +
                                            std::to_string(count));
        list.push_back(JPH::IndexedTriangle(tri[0], tri[1], tri[2], 0));
    }
    // The constructor drops degenerate and repeated triangles.
    JPH::MeshShapeSettings settings(std::move(vertices), std::move(list));
    if (settings.mIndexedTriangles.empty())
        throw std::invalid_argument("every triangle of the mesh is degenerate");
    return created(settings.Create(), "mesh");
}

uint32_t heightfield_side(size_t rows, size_t columns, uint32_t block_size) {
    const size_t wanted = std::max<size_t>({rows, columns, size_t(2) * block_size});
    return uint32_t((wanted + block_size - 1) / block_size * block_size);
}

JPH::RefConst<JPH::Shape> make_heightfield(const HeightFieldSpec &spec) {
    if (spec.rows < 2 || spec.columns < 2)
        throw std::invalid_argument("a height field needs at least 2 rows and 2 columns of samples");
    if (spec.block_size < 2 || spec.block_size > 8)
        throw std::invalid_argument("a height field's block size is 2 to 8 samples");
    if (spec.bits_per_sample < 1 || spec.bits_per_sample > 16)
        throw std::invalid_argument("a height field's bits per sample are 1 to 16");
    if (!(positive(spec.spacing[0]) && positive(spec.spacing[1])))
        throw std::invalid_argument("a height field's spacing is positive");
    for (float o : spec.origin)
        if (!std::isfinite(o))
            throw std::invalid_argument("a height field's origin is finite");
    // Jolt takes a square grid whose side is a multiple of the block size. A
    // rectangular grid is padded on its +x and +z sides with samples that
    // collide with nothing, so the padding adds no surface.
    const uint32_t side = heightfield_side(spec.rows, spec.columns, spec.block_size);
    JPH::HeightFieldShapeSettings settings;
    settings.mOffset = JPH::Vec3(spec.origin[0], spec.origin[1], spec.origin[2]);
    settings.mScale = JPH::Vec3(spec.spacing[0], 1.0f, spec.spacing[1]);
    settings.mSampleCount = side;
    settings.mBlockSize = spec.block_size;
    settings.mBitsPerSample = spec.bits_per_sample;
    settings.mHeightSamples.resize(size_t(side) * side, JPH::HeightFieldShapeConstants::cNoCollisionValue);
    bool any = false;
    for (size_t j = 0; j < spec.rows; ++j)
        for (size_t i = 0; i < spec.columns; ++i) {
            const float h = spec.heights[j * spec.columns + i];
            if (std::isnan(h))
                continue;
            if (!std::isfinite(h))
                throw std::invalid_argument("a height field's heights are finite or NaN for a hole");
            settings.mHeightSamples[j * side + i] = h;
            any = true;
        }
    if (!any)
        throw std::invalid_argument("every sample of the height field is a hole");
    if (std::isfinite(spec.min_height))
        settings.mMinHeightValue = spec.min_height;
    if (std::isfinite(spec.max_height))
        settings.mMaxHeightValue = spec.max_height;
    return created(settings.Create(), "height field");
}

JPH::RefConst<JPH::Shape> make_compound(const std::vector<JPH::RefConst<JPH::Shape>> &children,
                                        const double *positions, const double *rotations,
                                        bool mutable_compound) {
    if (children.empty())
        throw std::invalid_argument("a compound needs at least one child");
    JPH::StaticCompoundShapeSettings fixed;
    JPH::MutableCompoundShapeSettings changing;
    JPH::CompoundShapeSettings &settings =
        mutable_compound ? static_cast<JPH::CompoundShapeSettings &>(changing) : fixed;
    for (size_t i = 0; i < children.size(); ++i) {
        const double *p = positions + 3 * i;
        const double *q = rotations + 4 * i;
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(p[k]))
                throw std::invalid_argument("a compound's child positions are finite");
        const double norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (!(std::isfinite(norm) && norm > 1e-12))
            throw std::invalid_argument("a compound's child rotations are non-zero quaternions");
        if (children[i]->GetSubType() == JPH::EShapeSubType::HeightField)
            throw std::invalid_argument("a compound does not hold a height field; a landscape "
                                        "of tiles is one static body a tile");
        settings.AddShape(JPH::Vec3(float(p[0]), float(p[1]), float(p[2])),
                          JPH::Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])).Normalized(),
                          children[i].GetPtr());
    }
    return created(settings.Create(), "compound");
}

// -- the library -------------------------------------------------------------------

JPH::RefConst<JPH::Shape> ShapeLibrary::primitive(ShapeKind kind, const ShapeParams &p) {
    std::lock_guard<std::mutex> guard(lock_);
    const auto key = std::make_pair(kind, p);
    auto found = primitives_.find(key);
    if (found != primitives_.end())
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
    default:
        throw std::invalid_argument(shape_problem(uint8_t(kind), p));
    }
    primitives_.emplace(key, shape);
    return shape;
}

ShapeId ShapeLibrary::add(JPH::RefConst<JPH::Shape> shape, ShapeKind kind) {
    std::lock_guard<std::mutex> guard(lock_);
    const ShapeId id = next_++;
    registered_.emplace(id, std::make_pair(std::move(shape), kind));
    return id;
}

std::pair<JPH::RefConst<JPH::Shape>, ShapeKind> ShapeLibrary::find(ShapeId id) const {
    std::lock_guard<std::mutex> guard(lock_);
    auto found = registered_.find(id);
    if (found == registered_.end())
        return {nullptr, ShapeKind::Box};
    return found->second;
}

bool ShapeLibrary::remove(ShapeId id) {
    JPH::RefConst<JPH::Shape> released;
    std::lock_guard<std::mutex> guard(lock_);
    auto found = registered_.find(id);
    if (found == registered_.end())
        return false;
    released = std::move(found->second.first);
    registered_.erase(found);
    // Decorations of the shape are dropped with it; bodies hold their own references.
    for (auto it = decorated_.begin(); it != decorated_.end();)
        it = it->first.first == released.GetPtr() ? decorated_.erase(it) : std::next(it);
    return true;
}

size_t ShapeLibrary::registered() const {
    std::lock_guard<std::mutex> guard(lock_);
    return registered_.size();
}

JPH::RefConst<JPH::Shape> ShapeLibrary::decorate(const JPH::RefConst<JPH::Shape> &shape,
                                                 const Decoration &d) {
    if (d.identity())
        return shape;
    std::lock_guard<std::mutex> guard(lock_);
    // The entry holds the decorated shape, which holds `shape`, so the
    // pointer in the key names a live shape for as long as the entry exists.
    const auto key = std::make_pair(shape.GetPtr(), d);
    auto found = decorated_.find(key);
    if (found != decorated_.end())
        return found->second;
    JPH::RefConst<JPH::Shape> out = shape;
    const JPH::Vec3 scale(d.scale[0], d.scale[1], d.scale[2]);
    if (scale != JPH::Vec3::sOne())
        out = new JPH::ScaledShape(out, scale);
    const JPH::Vec3 offset(d.offset[0], d.offset[1], d.offset[2]);
    const JPH::Quat rotation =
        JPH::Quat(d.rotation[0], d.rotation[1], d.rotation[2], d.rotation[3]).Normalized();
    if (offset != JPH::Vec3::sZero() || !rotation.IsClose(JPH::Quat::sIdentity(), 0.0f))
        out = new JPH::RotatedTranslatedShape(offset, rotation, out);
    const JPH::Vec3 com(d.com[0], d.com[1], d.com[2]);
    if (com != JPH::Vec3::sZero())
        out = new JPH::OffsetCenterOfMassShape(out, com);
    decorated_.emplace(key, out);
    return out;
}

}  // namespace pj
