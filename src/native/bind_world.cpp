// The nanobind classes World and Snapshot over WorldCore and its buffers.
//
// Arrays come in already converted by the Python layer (pyjoltphysics.world)
// to C-contiguous arrays of the dtypes below; these functions check shapes
// and values and copy them into a batch.
#include "bindings.h"
#include "world.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <cstring>
#include <optional>
#include <algorithm>

namespace pj {

namespace {

template <typename T, typename Shape>
using In = nb::ndarray<const T, Shape, nb::c_contig, nb::device::cpu>;
using Uids = In<uint32_t, nb::shape<-1>>;
using Vec3s = In<double, nb::shape<-1, 3>>;
using Quats = In<double, nb::shape<-1, 4>>;
using Params = In<double, nb::shape<-1, 4>>;
using Doubles = In<double, nb::shape<-1>>;
using Floats = In<float, nb::shape<-1>>;
using Bytes = In<uint8_t, nb::shape<-1>>;
using Words = In<uint32_t, nb::shape<-1>>;

template <typename T, typename Array> std::vector<T> copy(const Array &array, size_t rows, const char *name) {
    if (array.shape(0) != rows)
        throw std::invalid_argument(std::string(name) + " has " + std::to_string(array.shape(0)) +
                                    " rows where the batch has " + std::to_string(rows));
    const size_t size = array.size();
    std::vector<T> out(array.data(), array.data() + size);
    return out;
}

template <typename T> void check_finite(const std::vector<T> &values, const char *name) {
    for (T value : values)
        if (!std::isfinite(double(value)))
            throw std::invalid_argument(std::string(name) + " holds a value that is not finite");
}

void check_quaternions(const std::vector<double> &q, const char *name) {
    check_finite(q, name);
    for (size_t i = 0; i < q.size(); i += 4) {
        const double norm = q[i] * q[i] + q[i + 1] * q[i + 1] + q[i + 2] * q[i + 2] + q[i + 3] * q[i + 3];
        if (norm < 1e-12)
            throw std::invalid_argument(std::string(name) + " row " + std::to_string(i / 4) +
                                        " is a zero quaternion");
    }
}

std::vector<Uid> uids(const Uids &array) { return copy<Uid>(array, array.shape(0), "uids"); }

template <typename T> const T *data_or_empty(const std::vector<T> &values) {
    static const T empty[4] = {};
    return values.empty() ? empty : values.data();
}

// A read-only view of one buffer field, kept alive by the Snapshot.
template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> view(const std::vector<T> &values,
                                            std::initializer_list<size_t> shape) {
    return nb::ndarray<nb::numpy, const T, Shape>(data_or_empty(values), shape, nb::handle());
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, T, Shape> owned(std::vector<T> &&values, std::initializer_list<size_t> shape) {
    auto *heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
    return nb::ndarray<nb::numpy, T, Shape>(heap->empty() ? nullptr : heap->data(), shape, owner);
}

struct Snapshot {
    std::shared_ptr<SnapshotBuffer> buffer;
};

class World {
public:
    explicit World(const WorldSettings &settings) : core_(std::make_shared<WorldCore>(settings)) {
        register_world(core_);
    }
    ~World() {
        if (!core_)
            return;
        if (core_->forked()) {
            // In a fork's child the world's threads and locks belong to the
            // parent; it is kept, never destroyed.
            new std::shared_ptr<WorldCore>(std::move(core_));
            return;
        }
        core_->clear_python();
        nb::gil_scoped_release release;
        core_->shutdown();
        core_.reset();
    }
    World(const World &) = delete;
    World &operator=(const World &) = delete;

    WorldCore &core() const { return *core_; }
    std::shared_ptr<WorldCore> core_;
};

int world_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self))
        return 0;
    World *world = nb::inst_ptr<World>(self);
    return world->core_ ? world->core_->traverse(visit, arg) : 0;
}

int world_clear(PyObject *self) {
    World *world = nb::inst_ptr<World>(self);
    if (world->core_)
        world->core_->clear_python();
    return 0;
}

PyType_Slot world_slots[] = {{Py_tp_traverse, reinterpret_cast<void *>(world_traverse)},
                             {Py_tp_clear, reinterpret_cast<void *>(world_clear)},
                             {0, nullptr}};

// The shape of each row: a registered shape by id, or a primitive by kind and
// parameters, then the row's decoration. Consecutive rows naming the same
// shape and decoration share the previous row's reference.
class ShapeResolver {
public:
    explicit ShapeResolver(ShapeLibrary &library) : library_(library) {}

    JPH::RefConst<JPH::Shape> operator()(uint8_t kind, const ShapeParams &params, ShapeId id,
                                         const Decoration &decoration, const std::string &row) {
        if (made_ && kind == kind_ && params == params_ && id == id_ &&
            !(decoration < decoration_) && !(decoration_ < decoration))
            return made_;
        JPH::RefConst<JPH::Shape> base;
        if (id != kNoShape) {
            base = library_.find(id).first;
            if (base == nullptr)
                throw std::invalid_argument("shape id " + std::to_string(id) +
                                            " names no shape of this world" + row);
        } else {
            const std::string problem = shape_problem(kind, params);
            if (!problem.empty())
                throw std::invalid_argument(problem + row);
            base = library_.primitive(static_cast<ShapeKind>(kind), params);
        }
        const JPH::Vec3 scale(decoration.scale[0], decoration.scale[1], decoration.scale[2]);
        for (float v : decoration.scale)
            if (!(std::isfinite(v) && v != 0.0f))
                throw std::invalid_argument("a scale is finite and not zero" + row);
        if (!base->IsValidScale(scale))
            throw std::invalid_argument(
                "this shape cannot take the scale (a sphere and a capsule take a uniform scale, "
                "a cylinder the same scale in x and z)" + row);
        made_ = library_.decorate(base, decoration);
        kind_ = kind;
        params_ = params;
        id_ = id;
        decoration_ = decoration;
        dynamic_ok_ = can_be_dynamic(*made_);
        plane_ = made_->GetSubType() == JPH::EShapeSubType::Plane;
        return made_;
    }
    bool dynamic_ok() const { return dynamic_ok_; }
    bool plane() const { return plane_; }

private:
    ShapeLibrary &library_;
    JPH::RefConst<JPH::Shape> made_;
    uint8_t kind_ = 0;
    ShapeParams params_{};
    ShapeId id_ = kNoShape;
    Decoration decoration_;
    bool dynamic_ok_ = true;
    bool plane_ = false;
};

template <size_t N> std::array<float, N> row_of(const std::vector<double> &values, size_t i) {
    std::array<float, N> out{};
    for (size_t k = 0; k < N; ++k)
        out[k] = float(values[N * i + k]);
    return out;
}

#define PJ_ADD_ARGS                                                                               \
    const Vec3s &position, const Quats &orientation, const Bytes &shape_kind,                     \
        const Params &shape_params, const Words &shape_id, const Vec3s &scale,                    \
        const Vec3s &shape_offset, const Quats &shape_rotation, const Vec3s &center_of_mass,      \
        const Bytes &motion, const Doubles &mass, const Doubles &density, const Floats &friction, \
        const Floats &restitution, const Floats &linear_damping, const Floats &angular_damping,   \
        const Floats &gravity_factor, const Vec3s &linear_velocity,                               \
        const Vec3s &angular_velocity, const Words &group, const Words &mask, const Words &flags
#define PJ_ADD_NAMES                                                                              \
    position, orientation, shape_kind, shape_params, shape_id, scale, shape_offset,              \
        shape_rotation, center_of_mass, motion, mass, density, friction, restitution,             \
        linear_damping, angular_damping, gravity_factor, linear_velocity, angular_velocity,       \
        group, mask, flags

// The rows of an add batch, checked, with their shapes resolved.
AddBodies build_add(World &world, const Vec3s &position, const Quats &orientation, const Bytes &shape_kind,
           const Params &shape_params, const Words &shape_id, const Vec3s &scale,
           const Vec3s &shape_offset, const Quats &shape_rotation, const Vec3s &center_of_mass,
           const Bytes &motion, const Doubles &mass, const Doubles &density,
           const Floats &friction, const Floats &restitution, const Floats &linear_damping,
           const Floats &angular_damping, const Floats &gravity_factor,
           const Vec3s &linear_velocity, const Vec3s &angular_velocity, const Words &group,
           const Words &mask, const Words &flags) {
    const size_t n = position.shape(0);
    AddBodies add;
    add.position = copy<double>(position, n, "position");
    add.orientation = copy<double>(orientation, n, "orientation");
    const std::vector<uint8_t> kinds = copy<uint8_t>(shape_kind, n, "shape_kind");
    const std::vector<double> params = copy<double>(shape_params, n, "shape_params");
    const std::vector<uint32_t> ids = copy<uint32_t>(shape_id, n, "shape_id");
    const std::vector<double> scales = copy<double>(scale, n, "scale");
    const std::vector<double> offsets = copy<double>(shape_offset, n, "shape_offset");
    const std::vector<double> rotations = copy<double>(shape_rotation, n, "shape_rotation");
    const std::vector<double> coms = copy<double>(center_of_mass, n, "center_of_mass");
    add.motion = copy<uint8_t>(motion, n, "motion");
    add.mass = copy<double>(mass, n, "mass");
    add.density = copy<double>(density, n, "density");
    add.friction = copy<float>(friction, n, "friction");
    add.restitution = copy<float>(restitution, n, "restitution");
    add.linear_damping = copy<float>(linear_damping, n, "linear_damping");
    add.angular_damping = copy<float>(angular_damping, n, "angular_damping");
    add.gravity_factor = copy<float>(gravity_factor, n, "gravity_factor");
    add.linear_velocity = copy<double>(linear_velocity, n, "linear_velocity");
    add.angular_velocity = copy<double>(angular_velocity, n, "angular_velocity");
    add.group = copy<uint32_t>(group, n, "group");
    add.mask = copy<uint32_t>(mask, n, "mask");
    add.flags = copy<uint32_t>(flags, n, "flags");
    check_finite(add.position, "position");
    check_quaternions(add.orientation, "orientation");
    check_finite(offsets, "shape_offset");
    check_quaternions(rotations, "shape_rotation");
    check_finite(coms, "center_of_mass");
    check_finite(add.linear_velocity, "linear_velocity");
    check_finite(add.angular_velocity, "angular_velocity");
    ShapeResolver resolve(world.core().shapes());
    add.shapes.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const std::string row = " (row " + std::to_string(i) + ")";
        Decoration decoration;
        decoration.scale = row_of<3>(scales, i);
        decoration.offset = row_of<3>(offsets, i);
        decoration.rotation = row_of<4>(rotations, i);
        decoration.com = row_of<3>(coms, i);
        const ShapeParams p{params[4 * i], params[4 * i + 1], params[4 * i + 2],
                            params[4 * i + 3]};
        add.shapes.push_back(resolve(kinds[i], p, ids[i], decoration, row));
        if (!valid_motion(add.motion[i]))
            throw std::invalid_argument("unknown motion type " + std::to_string(add.motion[i]) + row);
        if (add.flags[i] & ~flag::kAll)
            throw std::invalid_argument("unknown body flags" + row);
        const bool moving = add.motion[i] != uint8_t(Motion::Static);
        if (resolve.plane() && (moving || (add.flags[i] & flag::kMovable)))
            throw std::invalid_argument("a plane is static" + row);
        if (add.motion[i] == uint8_t(Motion::Dynamic) && !resolve.dynamic_ok())
            throw std::invalid_argument(
                "a mesh or height field (or a compound holding one) makes a static or kinematic "
                "body only: Jolt computes no mass properties for it" + row);
        const bool massive = resolve.dynamic_ok() && (moving || (add.flags[i] & flag::kMovable));
        if (massive && !(std::isfinite(add.mass[i]) && add.mass[i] > 0.0) &&
            !(std::isfinite(add.density[i]) && add.density[i] > 0.0))
            throw std::invalid_argument("a moving body needs a positive mass or density" + row);
        for (float value : {add.friction[i], add.restitution[i], add.linear_damping[i],
                            add.angular_damping[i]})
            if (!(std::isfinite(value) && value >= 0.0f))
                throw std::invalid_argument(
                    "friction, restitution and damping are finite and not negative" + row);
        if (!std::isfinite(add.gravity_factor[i]))
            throw std::invalid_argument("gravity_factor is not finite" + row);
    }
    return add;
}

using MadeUids = std::pair<nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>, uint64_t>;

MadeUids submit_add(World &world, Command &&command, size_t n, uint64_t at_tick) {
    Uid first = kNoBody;
    const uint64_t seq = world.core().submit(std::move(command), at_tick, &first);
    std::vector<uint32_t> made(n);
    for (size_t i = 0; i < n; ++i)
        made[i] = first + uint32_t(i);
    return {owned<uint32_t, nb::shape<-1>>(std::move(made), {n}), seq};
}

MadeUids add_bodies(World &world, PJ_ADD_ARGS, uint64_t at_tick) {
    AddBodies add = build_add(world, PJ_ADD_NAMES);
    const size_t n = add.uids.size() ? add.uids.size() : add.shapes.size();
    return submit_add(world, Command(std::move(add)), n, at_tick);
}

MadeUids swap_bodies(World &world, const Uids &remove, PJ_ADD_ARGS, uint64_t at_tick) {
    SwapBodies swap{RemoveBodies{uids(remove)}, build_add(world, PJ_ADD_NAMES)};
    const size_t n = swap.add.shapes.size();
    return submit_add(world, Command(std::move(swap)), n, at_tick);
}

// -- add_shape: built on the calling thread with the GIL released -------------------

ShapeId add_convex(World &world, const In<double, nb::shape<-1, 3>> &points,
                   float max_convex_radius, float hull_tolerance) {
    std::vector<double> cloud(points.data(), points.data() + points.size());
    JPH::RefConst<JPH::Shape> shape;
    {
        nb::gil_scoped_release release;
        shape = make_convex(cloud.data(), points.shape(0), max_convex_radius, hull_tolerance);
    }
    return world.core().shapes().add(std::move(shape), ShapeKind::Convex);
}

ShapeId add_mesh(World &world, const In<double, nb::shape<-1, 3>> &points,
                 const In<uint32_t, nb::shape<-1, 3>> &triangles) {
    std::vector<double> vertices(points.data(), points.data() + points.size());
    std::vector<uint32_t> indices(triangles.data(), triangles.data() + triangles.size());
    JPH::RefConst<JPH::Shape> shape;
    {
        nb::gil_scoped_release release;
        shape = make_mesh(vertices.data(), points.shape(0), indices.data(), triangles.shape(0));
    }
    return world.core().shapes().add(std::move(shape), ShapeKind::Mesh);
}

ShapeId add_heightfield(World &world, const In<float, nb::shape<-1, -1>> &heights,
                        std::array<double, 3> origin, std::array<double, 2> spacing,
                        uint32_t block_size, uint32_t bits_per_sample, double min_height,
                        double max_height) {
    std::vector<float> samples(heights.data(), heights.data() + heights.size());
    HeightFieldSpec spec;
    spec.heights = samples.data();
    spec.rows = heights.shape(0);
    spec.columns = heights.shape(1);
    spec.origin = {float(origin[0]), float(origin[1]), float(origin[2])};
    spec.spacing = {float(spacing[0]), float(spacing[1])};
    spec.block_size = block_size;
    spec.bits_per_sample = bits_per_sample;
    spec.min_height = float(min_height);
    spec.max_height = float(max_height);
    JPH::RefConst<JPH::Shape> shape;
    {
        nb::gil_scoped_release release;
        shape = make_heightfield(spec);
    }
    return world.core().shapes().add(std::move(shape), ShapeKind::HeightField);
}

ShapeId add_compound(World &world, const Words &children, const Vec3s &positions,
                     const Quats &rotations, bool mutable_compound) {
    const size_t n = children.shape(0);
    const std::vector<double> p = copy<double>(positions, n, "positions");
    const std::vector<double> q = copy<double>(rotations, n, "orientations");
    std::vector<JPH::RefConst<JPH::Shape>> shapes;
    for (size_t i = 0; i < n; ++i) {
        const ShapeId id = children.data()[i];
        auto shape = world.core().shapes().find(id).first;
        if (shape == nullptr)
            throw std::invalid_argument("shape id " + std::to_string(id) +
                                        " names no shape of this world");
        shapes.push_back(std::move(shape));
    }
    JPH::RefConst<JPH::Shape> shape;
    {
        nb::gil_scoped_release release;
        shape = make_compound(shapes, p.data(), q.data(), mutable_compound);
    }
    return world.core().shapes().add(std::move(shape), ShapeKind::Compound);
}

ShapeId add_primitive(World &world, uint8_t kind, std::array<double, 4> params) {
    if (!primitive_kind(kind))
        throw std::invalid_argument(shape_problem(kind, params));
    const std::string problem = shape_problem(kind, params);
    if (!problem.empty())
        throw std::invalid_argument(problem);
    auto shape = world.core().shapes().primitive(static_cast<ShapeKind>(kind), params);
    return world.core().shapes().add(std::move(shape), static_cast<ShapeKind>(kind));
}

// Volume (m3), centre of mass and the local bounds of a registered shape.
nb::dict shape_info(World &world, ShapeId id) {
    auto [shape, kind] = world.core().shapes().find(id);
    if (shape == nullptr)
        throw std::invalid_argument("shape id " + std::to_string(id) + " names no shape of this world");
    nb::dict out;
    out["kind"] = uint8_t(kind);
    out["volume"] = double(shape->GetVolume());
    const JPH::Vec3 com = shape->GetCenterOfMass();
    out["center_of_mass"] = std::array<double, 3>{com.GetX(), com.GetY(), com.GetZ()};
    const JPH::AABox box = shape->GetLocalBounds();
    // Local bounds are about the centre of mass; these are about the shape's origin.
    out["bounds_min"] = std::array<double, 3>{box.mMin.GetX() + com.GetX(),
                                              box.mMin.GetY() + com.GetY(),
                                              box.mMin.GetZ() + com.GetZ()};
    out["bounds_max"] = std::array<double, 3>{box.mMax.GetX() + com.GetX(),
                                              box.mMax.GetY() + com.GetY(),
                                              box.mMax.GetZ() + com.GetZ()};
    out["dynamic"] = can_be_dynamic(*shape);
    return out;
}

uint64_t pushes(World &world, Push kind, const Uids &ids, const Vec3s &values, uint64_t at_tick) {
    AddPushes push{kind, uids(ids), copy<double>(values, ids.shape(0), "values")};
    check_finite(push.values, "values");
    return world.core().submit(Command(std::move(push)), at_tick);
}

nb::dict stats(const World &world) {
    const WorldStats s = world.core().stats();
    nb::dict out;
    out["ticks"] = s.ticks;
    out["dropped_ticks"] = s.dropped_ticks;
    out["events_dropped"] = s.events_dropped;
    out["rejected"] = s.rejected;
    out["pool"] = s.pool;
    out["bodies"] = s.bodies;
    out["queued_bodies"] = s.queued_bodies;
    out["shapes"] = s.shapes;
    out["characters"] = s.characters;
    out["characters_ms"] = s.characters_ms;
    out["queries"] = s.queries;
    out["query_wait_ms"] = s.query_wait_ms;
    out["query_wait_max_ms"] = s.query_wait_max_ms;
    out["broadphase_optimizations"] = s.broadphase_optimizations;
    out["idle"] = s.idle;
    out["running"] = s.running;
    out["apply_ms"] = s.apply_ms;
    out["step_ms"] = s.step_ms;
    out["events_ms"] = s.events_ms;
    out["publish_ms"] = s.publish_ms;
    out["callbacks_ms"] = s.callbacks_ms;
    out["tick_ms"] = s.tick_ms;
    out["timed_ticks"] = s.timed_ticks;
    return out;
}


// -- queries --------------------------------------------------------------------------

QueryFilter make_filter(uint32_t group, uint32_t mask, const Uids &exclude, bool triggers, bool characters) {
    QueryFilter filter;
    filter.group = group;
    filter.mask = mask;
    filter.exclude = uids(exclude);
    std::sort(filter.exclude.begin(), filter.exclude.end());
    filter.triggers = triggers;
    filter.characters = characters;
    return filter;
}

// A query's shape: a registered shape by id, or a primitive by kind and parameters.
JPH::RefConst<JPH::Shape> query_shape(World &world, uint8_t kind, std::array<double, 4> params, ShapeId id) {
    if (id != kNoShape) {
        auto shape = world.core().shapes().find(id).first;
        if (shape == nullptr)
            throw std::invalid_argument("shape id " + std::to_string(id) + " names no shape of this world");
        return shape;
    }
    const std::string problem = shape_problem(kind, params);
    if (!problem.empty())
        throw std::invalid_argument(problem);
    if (kind == uint8_t(ShapeKind::Plane))
        throw std::invalid_argument("a plane cannot be cast or overlapped");
    return world.core().shapes().primitive(static_cast<ShapeKind>(kind), params);
}

void check_vector(const std::array<double, 3> &v, const char *name) {
    for (double x : v)
        if (!std::isfinite(x))
            throw std::invalid_argument(std::string(name) + " is not finite");
}

void check_rotation(const std::array<double, 4> &q) {
    double norm = 0.0;
    for (double x : q) {
        if (!std::isfinite(x))
            throw std::invalid_argument("orientation is not finite");
        norm += x * x;
    }
    if (norm < 1e-12)
        throw std::invalid_argument("orientation is a zero quaternion");
}

using RayArrays = std::tuple<nb::ndarray<nb::numpy, uint8_t, nb::shape<-1>>, nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>,
                             nb::ndarray<nb::numpy, float, nb::shape<-1>>, nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>,
                             nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>, nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>>;
using CastArrays = std::tuple<nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>, nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>,
                              nb::ndarray<nb::numpy, float, nb::shape<-1>>, nb::ndarray<nb::numpy, float, nb::shape<-1>>,
                              nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>, nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>>;

RayArrays raycast_batch(World &world, const Vec3s &origins, const Vec3s &directions, double max_distance,
                        uint32_t group, uint32_t mask, const Uids &exclude, bool triggers, bool characters) {
    const size_t n = origins.shape(0);
    if (directions.shape(0) != n)
        throw std::invalid_argument("directions has " + std::to_string(directions.shape(0)) +
                                    " rows where origins has " + std::to_string(n));
    if (!(std::isfinite(max_distance) && max_distance > 0.0))
        throw std::invalid_argument("max_distance is finite and positive");
    const QueryFilter filter = make_filter(group, mask, exclude, triggers, characters);
    RayResults out;
    {
        nb::gil_scoped_release release;
        world.core().raycast(origins.data(), directions.data(), n, max_distance, filter, out);
    }
    return RayArrays(owned<uint8_t, nb::shape<-1>>(std::move(out.hit), {n}),
                          owned<uint32_t, nb::shape<-1>>(std::move(out.uid), {n}),
                          owned<float, nb::shape<-1>>(std::move(out.fraction), {n}),
                          owned<double, nb::shape<-1, 3>>(std::move(out.point), {n, 3}),
                          owned<float, nb::shape<-1, 3>>(std::move(out.normal), {n, 3}),
                          owned<uint32_t, nb::shape<-1>>(std::move(out.sub_shape), {n}));
}

using OneRay = std::optional<std::tuple<Uid, float, std::array<double, 3>, std::array<float, 3>, uint32_t>>;

// One ray, answered without arrays: (uid, fraction, point, normal, sub_shape) or None.
OneRay raycast_one(World &world, std::array<double, 3> origin, std::array<double, 3> direction, double max_distance,
                   uint32_t group, uint32_t mask, const Uids &exclude, bool triggers, bool characters) {
    check_vector(origin, "origin");
    check_vector(direction, "direction");
    if (direction[0] == 0.0 && direction[1] == 0.0 && direction[2] == 0.0)
        throw std::invalid_argument("a ray direction is not zero");
    if (!(std::isfinite(max_distance) && max_distance > 0.0))
        throw std::invalid_argument("max_distance is finite and positive");
    const QueryFilter filter = make_filter(group, mask, exclude, triggers, characters);
    RayResults out;
    {
        nb::gil_scoped_release release;
        world.core().raycast(origin.data(), direction.data(), 1, max_distance, filter, out);
    }
    if (!out.hit[0])
        return std::nullopt;
    return std::make_tuple(out.uid[0], out.fraction[0], std::array<double, 3>{out.point[0], out.point[1], out.point[2]},
                           std::array<float, 3>{out.normal[0], out.normal[1], out.normal[2]}, out.sub_shape[0]);
}

CastArrays cast_shape(World &world, uint8_t kind, std::array<double, 4> params, ShapeId id,
                     std::array<double, 3> start, std::array<double, 4> rotation, std::array<double, 3> end,
                     bool all_hits, uint32_t group, uint32_t mask, const Uids &exclude, bool triggers,
                     bool characters) {
    check_vector(start, "start");
    check_vector(end, "end");
    check_rotation(rotation);
    const JPH::RefConst<JPH::Shape> shape = query_shape(world, kind, params, id);
    const QueryFilter filter = make_filter(group, mask, exclude, triggers, characters);
    std::vector<ShapeHit> hits;
    {
        nb::gil_scoped_release release;
        hits = world.core().cast_shape(*shape, start.data(), rotation.data(), end.data(), all_hits, filter);
    }
    const size_t n = hits.size();
    std::vector<uint32_t> uid(n), sub(n);
    std::vector<float> fraction(n), depth(n), normal(3 * n);
    std::vector<double> point(3 * n);
    for (size_t i = 0; i < n; ++i) {
        uid[i] = hits[i].uid;
        sub[i] = hits[i].sub_shape;
        fraction[i] = hits[i].fraction;
        depth[i] = hits[i].depth;
        for (size_t k = 0; k < 3; ++k) {
            point[3 * i + k] = hits[i].point[k];
            normal[3 * i + k] = hits[i].normal[k];
        }
    }
    return CastArrays(owned<uint32_t, nb::shape<-1>>(std::move(uid), {n}),
                          owned<uint32_t, nb::shape<-1>>(std::move(sub), {n}),
                          owned<float, nb::shape<-1>>(std::move(fraction), {n}),
                          owned<float, nb::shape<-1>>(std::move(depth), {n}),
                          owned<double, nb::shape<-1, 3>>(std::move(point), {n, 3}),
                          owned<float, nb::shape<-1, 3>>(std::move(normal), {n, 3}));
}

nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>> overlap(World &world, uint8_t kind, std::array<double, 4> params,
                                                        ShapeId id, std::array<double, 3> position,
                                                        std::array<double, 4> rotation, uint32_t group,
                                                        uint32_t mask, const Uids &exclude, bool triggers,
                                                        bool characters) {
    check_vector(position, "position");
    check_rotation(rotation);
    const JPH::RefConst<JPH::Shape> shape = query_shape(world, kind, params, id);
    const QueryFilter filter = make_filter(group, mask, exclude, triggers, characters);
    std::vector<Uid> found;
    {
        nb::gil_scoped_release release;
        found = world.core().overlap(*shape, position.data(), rotation.data(), filter);
    }
    const size_t n = found.size();
    return owned<uint32_t, nb::shape<-1>>(std::move(found), {n});
}

// -- characters -----------------------------------------------------------------------

// The order of the float capabilities pyjoltphysics.characters sends.
constexpr size_t kCapsFloats = 24;
// Bits of the bool capabilities, in the same order.
enum CapsBit : uint32_t {
    kCanSprint = 1u << 0,
    kCanCrouch = 1u << 1,
    kCanJump = 1u << 2,
    kCanFly = 1u << 3,
    kJumpFollowsSlope = 1u << 4,
    kSuppressOverVoid = 1u << 5,
    kStickToFloor = 1u << 6,
    kSmoothEdges = 1u << 7,
};

CharacterCaps make_caps(const Doubles &values, uint32_t bits) {
    if (values.shape(0) != kCapsFloats)
        throw std::invalid_argument("capabilities has " + std::to_string(values.shape(0)) + " values where " +
                                    std::to_string(kCapsFloats) + " are needed");
    const double *v = values.data();
    CharacterCaps c;
    float *fields[kCapsFloats] = {&c.walk_speed,   &c.run_speed,        &c.sprint_speed,  &c.crouch_speed,
                                  &c.fly_speed,    &c.swim_speed,       &c.swim_drag,     &c.jump_height,
                                  &c.step_height,  &c.max_slope,        &c.eye_height,    &c.stand_height,
                                  &c.crouch_height, &c.radius,          &c.air_control,   &c.push_friction,
                                  &c.terminal_velocity, &c.coyote_time, &c.jump_buffer,   &c.mass,
                                  &c.push_mass,    &c.push_speed,       &c.push_force,    &c.acceleration};
    for (size_t i = 0; i < kCapsFloats; ++i)
        *fields[i] = float(v[i]);
    c.can_sprint = bits & kCanSprint;
    c.can_crouch = bits & kCanCrouch;
    c.can_jump = bits & kCanJump;
    c.can_fly = bits & kCanFly;
    c.jump_follows_slope = bits & kJumpFollowsSlope;
    c.suppress_over_void = bits & kSuppressOverVoid;
    c.stick_to_floor = bits & kStickToFloor;
    c.smooth_edges = bits & kSmoothEdges;
    if (!(c.radius > 0.0f && c.crouch_height >= 2.0f * c.radius && c.stand_height >= c.crouch_height))
        throw std::invalid_argument("a character's radius is positive, its crouch height at least twice the radius "
                                    "and its stand height at least the crouch height");
    if (!(c.mass > 0.0f))
        throw std::invalid_argument("a character's mass is positive");
    return c;
}

std::pair<Uid, uint64_t> add_character(World &world, std::array<double, 3> position, const Doubles &caps,
                                       uint32_t bits, double gravity, uint32_t group, uint32_t mask,
                                       uint32_t pass_group, bool proxy, const Uids &ignore, uint64_t at_tick) {
    check_vector(position, "position");
    if (!std::isfinite(gravity))
        throw std::invalid_argument("gravity is not finite");
    AddCharacter add;
    for (size_t i = 0; i < 3; ++i)
        add.position[i] = position[i];
    add.caps = make_caps(caps, bits);
    add.gravity = gravity;
    add.group = group;
    add.mask = mask;
    add.pass_group = pass_group;
    add.proxy = proxy;
    add.ignore = uids(ignore);
    std::sort(add.ignore.begin(), add.ignore.end());
    Uid first = kNoBody;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, &first);
    return {first, seq};
}

uint64_t character_ops(World &world, const Uids &ids, const Bytes &ops, const Bytes &codes, const Params &values,
                       uint64_t at_tick) {
    const size_t n = ids.shape(0);
    const std::vector<uint8_t> op = copy<uint8_t>(ops, n, "ops");
    const std::vector<uint8_t> code = copy<uint8_t>(codes, n, "codes");
    const std::vector<double> v = copy<double>(values, n, "values");
    CharacterOps batch;
    batch.ops.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (op[i] >= uint8_t(CharacterOpCode::Count))
            throw std::invalid_argument("unknown character op " + std::to_string(op[i]));
        CharacterOp &out = batch.ops[i];
        out.id = ids.data()[i];
        out.op = op[i];
        out.code = code[i];
        for (size_t k = 0; k < 4; ++k) {
            out.v[k] = v[4 * i + k];
            // NaN is the one value an op takes that is not finite: a Move's "the mode's speed".
            if (!std::isfinite(out.v[k]) && !(std::isnan(out.v[k]) && out.op == uint8_t(CharacterOpCode::Move)))
                throw std::invalid_argument("character op values are finite");
        }
    }
    return world.core().submit(Command(std::move(batch)), at_tick);
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> char_view(const Snapshot &s, const std::vector<T> &values, size_t width) {
    if (width == 1)
        return view<T, Shape>(values, {s.buffer->characters});
    return view<T, Shape>(values, {s.buffer->characters, width});
}
}  // namespace

void bind_world(nb::module_ &m) {
    nb::exception<CallbackFailed>(m, "CallbackFailed", PyExc_RuntimeError);
    nb::exception<WorldStopped>(m, "WorldStopped", PyExc_RuntimeError);

    m.attr("EVENT_SIZE") = sizeof(Event);

    nb::class_<Snapshot>(m, "Snapshot", "The state of every body after one tick.")
        .def_prop_ro("tick", [](const Snapshot &s) { return s.buffer->tick; })
        .def_prop_ro("time", [](const Snapshot &s) { return s.buffer->time; })
        .def_prop_ro("applied_seq", [](const Snapshot &s) { return s.buffer->applied_seq; })
        .def_prop_ro("count", [](const Snapshot &s) { return s.buffer->rows; })
        .def_prop_ro(
            "uid",
            [](const Snapshot &s) { return view<Uid, nb::shape<-1>>(s.buffer->uid, {s.buffer->rows}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "position",
            [](const Snapshot &s) {
                return view<double, nb::shape<-1, 3>>(s.buffer->position, {s.buffer->rows, 3});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "orientation",
            [](const Snapshot &s) {
                return view<float, nb::shape<-1, 4>>(s.buffer->orientation, {s.buffer->rows, 4});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "linear_velocity",
            [](const Snapshot &s) {
                return view<float, nb::shape<-1, 3>>(s.buffer->linear_velocity, {s.buffer->rows, 3});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "angular_velocity",
            [](const Snapshot &s) {
                return view<float, nb::shape<-1, 3>>(s.buffer->angular_velocity, {s.buffer->rows, 3});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "awake",
            [](const Snapshot &s) {
                return view<uint8_t, nb::shape<-1>>(s.buffer->awake, {s.buffer->rows});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "motion_type",
            [](const Snapshot &s) {
                return view<uint8_t, nb::shape<-1>>(s.buffer->motion, {s.buffer->rows});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("character_count", [](const Snapshot &s) { return s.buffer->characters; })
        .def_prop_ro(
            "character_id",
            [](const Snapshot &s) { return char_view<Uid, nb::shape<-1>>(s, s.buffer->char_id, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_position",
            [](const Snapshot &s) { return char_view<double, nb::shape<-1, 3>>(s, s.buffer->char_position, 3); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_velocity",
            [](const Snapshot &s) { return char_view<float, nb::shape<-1, 3>>(s, s.buffer->char_velocity, 3); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_state",
            [](const Snapshot &s) { return char_view<uint8_t, nb::shape<-1>>(s, s.buffer->char_state, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_grounded",
            [](const Snapshot &s) { return char_view<uint8_t, nb::shape<-1>>(s, s.buffer->char_grounded, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_ground",
            [](const Snapshot &s) { return char_view<Uid, nb::shape<-1>>(s, s.buffer->char_ground, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_ground_normal",
            [](const Snapshot &s) {
                return char_view<float, nb::shape<-1, 3>>(s, s.buffer->char_ground_normal, 3);
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_slide",
            [](const Snapshot &s) { return char_view<float, nb::shape<-1>>(s, s.buffer->char_slide, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_landing",
            [](const Snapshot &s) { return char_view<float, nb::shape<-1>>(s, s.buffer->char_landing, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_landed_tick",
            [](const Snapshot &s) { return char_view<uint32_t, nb::shape<-1>>(s, s.buffer->char_landed_tick, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_height",
            [](const Snapshot &s) { return char_view<float, nb::shape<-1>>(s, s.buffer->char_height, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "changed",
            [](const Snapshot &s) {
                return view<uint32_t, nb::shape<-1>>(s.buffer->changed, {s.buffer->changed.size()});
            },
            nb::rv_policy::reference_internal);

    nb::class_<World>(m, "World", nb::type_slots(world_slots),
                      "A Jolt physics world stepped on its own native thread.")
        .def(
            "__init__",
            [](World *self, double tick_hz, double max_catch_up, std::array<double, 3> gravity,
               int threads, uint32_t max_bodies, uint32_t max_body_pairs,
               uint32_t max_contact_constraints, size_t event_capacity, bool allow_sleeping,
               float time_before_sleep, float sleep_velocity, float min_velocity_for_restitution,
               int collision_steps, uint32_t temp_allocator_mb) {
                if (!(std::isfinite(tick_hz) && tick_hz > 0.0))
                    throw std::invalid_argument("tick_hz is positive");
                if (!(std::isfinite(max_catch_up) && max_catch_up >= 0.0))
                    throw std::invalid_argument("max_catch_up is zero or positive");
                if (threads < 1)
                    throw std::invalid_argument("threads is at least 1");
                if (max_bodies < 1 || collision_steps < 1 || event_capacity < 1)
                    throw std::invalid_argument(
                        "max_bodies, collision_steps and event_capacity are at least 1");
                WorldSettings settings;
                settings.tick_hz = tick_hz;
                settings.max_catch_up = max_catch_up;
                for (int i = 0; i < 3; ++i)
                    settings.gravity[i] = gravity[size_t(i)];
                settings.threads = threads;
                settings.max_bodies = max_bodies;
                settings.max_body_pairs = max_body_pairs;
                settings.max_contact_constraints = max_contact_constraints;
                settings.event_capacity = event_capacity;
                settings.allow_sleeping = allow_sleeping;
                settings.time_before_sleep = time_before_sleep;
                settings.sleep_velocity = sleep_velocity;
                settings.min_velocity_for_restitution = min_velocity_for_restitution;
                settings.collision_steps = collision_steps;
                settings.temp_allocator_mb = temp_allocator_mb;
                new (self) World(settings);
            },
            nb::kw_only(), nb::arg("tick_hz") = 120.0, nb::arg("max_catch_up") = 0.1,
            nb::arg("gravity").sig("(0.0, -9.81, 0.0)") = std::array<double, 3>{0.0, -9.81, 0.0}, nb::arg("threads") = 4,
            nb::arg("max_bodies") = 65536, nb::arg("max_body_pairs") = 65536,
            nb::arg("max_contact_constraints") = 65536, nb::arg("event_capacity") = 65536,
            nb::arg("allow_sleeping") = true, nb::arg("time_before_sleep") = 0.5f,
            nb::arg("sleep_velocity").sig("0.03") = 0.03f, nb::arg("min_velocity_for_restitution") = 1.0f,
            nb::arg("collision_steps") = 1, nb::arg("temp_allocator_mb") = 16)
        .def("add_bodies", &add_bodies, nb::arg("position"), nb::arg("orientation"),
             nb::arg("shape_kind"), nb::arg("shape_params"), nb::arg("shape_id"),
             nb::arg("scale"), nb::arg("shape_offset"), nb::arg("shape_rotation"),
             nb::arg("center_of_mass"), nb::arg("motion"), nb::arg("mass"),
             nb::arg("density"), nb::arg("friction"), nb::arg("restitution"),
             nb::arg("linear_damping"), nb::arg("angular_damping"), nb::arg("gravity_factor"),
             nb::arg("linear_velocity"), nb::arg("angular_velocity"), nb::arg("group"),
             nb::arg("mask"), nb::arg("flags"), nb::arg("at_tick") = 0,
             "Queue bodies to add; return their uids and the batch's sequence number.")
        .def("swap_bodies", &swap_bodies, nb::arg("remove"), nb::arg("position"), nb::arg("orientation"),
             nb::arg("shape_kind"), nb::arg("shape_params"), nb::arg("shape_id"),
             nb::arg("scale"), nb::arg("shape_offset"), nb::arg("shape_rotation"),
             nb::arg("center_of_mass"), nb::arg("motion"), nb::arg("mass"),
             nb::arg("density"), nb::arg("friction"), nb::arg("restitution"),
             nb::arg("linear_damping"), nb::arg("angular_damping"), nb::arg("gravity_factor"),
             nb::arg("linear_velocity"), nb::arg("angular_velocity"), nb::arg("group"),
             nb::arg("mask"), nb::arg("flags"), nb::arg("at_tick") = 0,
             "Queue removing bodies and adding others in one tick; return the new uids and the sequence number.")
        .def("raycast", &raycast_one, nb::arg("origin"), nb::arg("direction"), nb::arg("max_distance"),
             nb::arg("group"), nb::arg("mask"), nb::arg("exclude"), nb::arg("triggers"), nb::arg("characters"),
             "Cast one ray; return (uid, fraction, point, normal, sub_shape) of the first hit, or None.")
        .def("raycast_batch", &raycast_batch, nb::arg("origins"), nb::arg("directions"), nb::arg("max_distance"),
             nb::arg("group"), nb::arg("mask"), nb::arg("exclude"), nb::arg("triggers"), nb::arg("characters"),
             "Cast rays; return (hit, uid, fraction, point, normal, sub_shape) arrays.")
        .def("cast_shape", &cast_shape, nb::arg("kind"), nb::arg("params"), nb::arg("shape_id"), nb::arg("start"),
             nb::arg("rotation"), nb::arg("end"), nb::arg("all_hits"), nb::arg("group"), nb::arg("mask"),
             nb::arg("exclude"), nb::arg("triggers"), nb::arg("characters"),
             "Sweep a shape; return (uid, sub_shape, fraction, depth, point, normal) arrays, one row a body hit.")
        .def("overlap", &overlap, nb::arg("kind"), nb::arg("params"), nb::arg("shape_id"), nb::arg("position"),
             nb::arg("rotation"), nb::arg("group"), nb::arg("mask"), nb::arg("exclude"), nb::arg("triggers"),
             nb::arg("characters"), "Return the ascending uids of the bodies a shape at a pose touches.")
        .def("add_character", &add_character, nb::arg("position"), nb::arg("caps"), nb::arg("bits"),
             nb::arg("gravity"), nb::arg("group"), nb::arg("mask"), nb::arg("pass_group"), nb::arg("proxy"),
             nb::arg("ignore"), nb::arg("at_tick") = 0,
             "Queue a character; return its id and the batch's sequence number.")
        .def(
            "remove_characters",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(RemoveCharacters{uids(ids)}), at_tick);
            },
            nb::arg("ids"), nb::arg("at_tick") = 0)
        .def("character_ops", &character_ops, nb::arg("ids"), nb::arg("ops"), nb::arg("codes"), nb::arg("values"),
             nb::arg("at_tick") = 0, "Queue character inputs, applied in order at the next tick.")
        .def(
            "set_character_ignore",
            [](World &w, Uid id, const Uids &ignore, uint64_t at_tick) {
                SetCharacterIgnore set{id, uids(ignore)};
                std::sort(set.ignore.begin(), set.ignore.end());
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("id"), nb::arg("ignore"), nb::arg("at_tick") = 0)
        .def("add_convex", &add_convex, nb::arg("points"), nb::arg("max_convex_radius"),
             nb::arg("hull_tolerance"), "Build and register a convex hull; return its shape id.")
        .def("add_mesh", &add_mesh, nb::arg("points"), nb::arg("triangles"),
             "Build and register a triangle mesh; return its shape id.")
        .def("add_heightfield", &add_heightfield, nb::arg("heights"), nb::arg("origin"),
             nb::arg("spacing"), nb::arg("block_size"), nb::arg("bits_per_sample"),
             nb::arg("min_height"), nb::arg("max_height"),
             "Build and register a height field; return its shape id.")
        .def("add_compound", &add_compound, nb::arg("children"), nb::arg("positions"),
             nb::arg("orientations"), nb::arg("mutable"),
             "Build and register a compound of registered shapes; return its shape id.")
        .def("add_primitive", &add_primitive, nb::arg("kind"), nb::arg("params"),
             "Register a primitive shape; return its shape id.")
        .def(
            "remove_shape", [](World &w, ShapeId id) { return w.core().shapes().remove(id); },
            nb::arg("shape_id"), "Forget a shape id; bodies made with the shape keep it.")
        .def("shape_info", &shape_info, nb::arg("shape_id"),
             nb::sig("def shape_info(self, shape_id: int) -> dict[str, object]"))
        .def(
            "remove_bodies",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(RemoveBodies{uids(ids)}), at_tick);
            },
            nb::arg("uids"), nb::arg("at_tick") = 0)
        .def(
            "set_transforms",
            [](World &w, const Uids &ids, const Vec3s &position, const Quats &orientation,
               uint64_t at_tick) {
                const size_t n = ids.shape(0);
                SetTransforms set{uids(ids), copy<double>(position, n, "position"),
                                  copy<double>(orientation, n, "orientation")};
                check_finite(set.position, "position");
                check_quaternions(set.orientation, "orientation");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("position"), nb::arg("orientation"), nb::arg("at_tick") = 0)
        .def(
            "set_velocities",
            [](World &w, const Uids &ids, std::optional<Vec3s> linear,
               std::optional<Vec3s> angular, uint64_t at_tick) {
                const size_t n = ids.shape(0);
                SetVelocities set;
                set.uids = uids(ids);
                if (linear)
                    set.linear = copy<double>(*linear, n, "linear");
                if (angular)
                    set.angular = copy<double>(*angular, n, "angular");
                check_finite(set.linear, "linear");
                check_finite(set.angular, "angular");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("linear").none(), nb::arg("angular").none(),
            nb::arg("at_tick") = 0)
        .def(
            "add_forces",
            [](World &w, const Uids &ids, const Vec3s &values, uint64_t at_tick) {
                return pushes(w, Push::Force, ids, values, at_tick);
            },
            nb::arg("uids"), nb::arg("values"), nb::arg("at_tick") = 0)
        .def(
            "add_impulses",
            [](World &w, const Uids &ids, const Vec3s &values, uint64_t at_tick) {
                return pushes(w, Push::Impulse, ids, values, at_tick);
            },
            nb::arg("uids"), nb::arg("values"), nb::arg("at_tick") = 0)
        .def(
            "add_torques",
            [](World &w, const Uids &ids, const Vec3s &values, uint64_t at_tick) {
                return pushes(w, Push::Torque, ids, values, at_tick);
            },
            nb::arg("uids"), nb::arg("values"), nb::arg("at_tick") = 0)
        .def(
            "set_gravity",
            [](World &w, std::array<double, 3> gravity, uint64_t at_tick) {
                SetGravity set{{gravity[0], gravity[1], gravity[2]}};
                for (double g : set.gravity)
                    if (!std::isfinite(g))
                        throw std::invalid_argument("gravity is not finite");
                return w.core().submit(Command(set), at_tick);
            },
            nb::arg("gravity"), nb::arg("at_tick") = 0)
        .def(
            "activate",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(Activate{uids(ids)}), at_tick);
            },
            nb::arg("uids"), nb::arg("at_tick") = 0)
        .def(
            "set_motion_types",
            [](World &w, const Uids &ids, const Bytes &motion, uint64_t at_tick) {
                SetMotionTypes set{uids(ids), copy<uint8_t>(motion, ids.shape(0), "motion")};
                for (uint8_t code : set.motion)
                    if (!valid_motion(code))
                        throw std::invalid_argument("unknown motion type " + std::to_string(code));
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("motion"), nb::arg("at_tick") = 0)
        .def(
            "set_materials",
            [](World &w, const Uids &ids, const Floats &friction, const Floats &restitution,
               uint64_t at_tick) {
                const size_t n = ids.shape(0);
                SetMaterials set{uids(ids), copy<float>(friction, n, "friction"),
                                 copy<float>(restitution, n, "restitution")};
                for (const auto *values : {&set.friction, &set.restitution})
                    for (float value : *values)
                        if (!(std::isfinite(value) && value >= 0.0f))
                            throw std::invalid_argument(
                                "friction and restitution are finite and not negative");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("friction"), nb::arg("restitution"), nb::arg("at_tick") = 0)
        .def("start", [](World &w) { w.core().start(); }, nb::call_guard<nb::gil_scoped_release>())
        .def("stop", [](World &w) { w.core().stop(); }, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "run_ticks", [](World &w, uint64_t count) { w.core().run_ticks(count); },
            nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "wait_tick",
            [](World &w, uint64_t tick, double timeout) { return w.core().wait_tick(tick, timeout); },
            nb::arg("tick"), nb::arg("timeout") = -1.0, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "latest",
            [](const World &w) -> std::optional<Snapshot> {
                auto buffer = w.core().latest();
                if (!buffer)
                    return std::nullopt;
                return Snapshot{std::move(buffer)};
            })
        .def("drain_events",
             [](World &w) {
                 w.core().check();
                 std::vector<Event> events = w.core().drain_events();
                 std::vector<uint8_t> bytes(events.size() * sizeof(Event));
                 if (!bytes.empty())
                     std::memcpy(bytes.data(), events.data(), bytes.size());
                 const size_t size = bytes.size();
                 return owned<uint8_t, nb::shape<-1>>(std::move(bytes), {size});
             })
        .def("drain_applied",
             [](World &w) {
                 w.core().check();
                 std::vector<uint64_t> log = w.core().drain_applied();
                 const size_t rows = log.size() / 2;
                 return owned<uint64_t, nb::shape<-1, 2>>(std::move(log), {rows, 2});
             })
        .def(
            "set_before_step",
            [](World &w, nb::object fn) {
                CallbackSet &c = w.core().callbacks();
                c.set(c.before_step, c.has_before, std::move(fn));
                w.core().poke();
            },
            nb::arg("fn").none())
        .def(
            "set_after_step",
            [](World &w, nb::object fn) {
                CallbackSet &c = w.core().callbacks();
                c.set(c.after_step, c.has_after, std::move(fn));
                w.core().poke();
            },
            nb::arg("fn").none())
        .def(
            "set_on_events",
            [](World &w, nb::object fn) {
                CallbackSet &c = w.core().callbacks();
                c.set(c.on_events, c.has_events, std::move(fn));
            },
            nb::arg("fn").none())
        .def(
            "at_tick",
            [](World &w, uint64_t tick, nb::object fn) {
                w.core().check();
                w.core().callbacks().add_at_tick(tick, std::move(fn));
                w.core().poke();
            },
            nb::arg("tick"), nb::arg("fn"))
        .def_prop_ro("error", [](const World &w) { return w.core().error(); })
        .def_prop_ro("tick", [](const World &w) { return w.core().tick(); })
        .def_prop_ro("lockstep", [](const World &w) { return w.core().lockstep(); })
        .def_prop_ro("forked", [](const World &w) { return w.core().forked(); })
        .def_prop_ro("submitted_seq", [](const World &w) { return w.core().submitted_seq(); })
        .def_prop_ro("on_world_thread", [](const World &w) { return w.core().on_world_thread(); })
        .def("stats", &stats, nb::sig("def stats(self) -> dict[str, int | float | bool]"))
        .def("reset_stats", [](World &w) { w.core().reset_stats(); });

    m.def("stop_all_worlds", &stop_all_worlds, nb::call_guard<nb::gil_scoped_release>(),
          "Stop every world; from then on no world thread runs a callback. For atexit.");
    m.def("fork_prepare", &fork_prepare, nb::call_guard<nb::gil_scoped_release>(),
          "The os.register_at_fork hook run before a fork.");
    m.def("fork_parent", &fork_parent, "The os.register_at_fork hook run in the parent.");
    m.def("fork_child", &fork_child,
          "The os.register_at_fork hook run in the child: every world is marked forked.");
}

}  // namespace pj
