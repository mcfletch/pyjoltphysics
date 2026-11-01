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

#include <Jolt/Physics/Body/MassProperties.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

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
using Shorts = In<uint16_t, nb::shape<-1>>;

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

// Whether a double the world stores as a 32-bit float stays finite there.
bool fits_float(double value) { return std::abs(value) <= double(std::numeric_limits<float>::max()); }

// For values stored as floats: finite, and finite once narrowed.
void check_floats(const std::vector<double> &values, const char *name) {
    for (double value : values)
        if (!fits_float(value))
            throw std::invalid_argument(std::string(name) + " holds a value that is not finite as a 32-bit float");
}

// An integer setting carried in a double: a whole number in [0, limit], checked
// before any conversion, since converting an out-of-range double is undefined.
void check_whole(double value, double limit, const std::string &name) {
    if (!(value >= 0.0 && value <= limit && value == std::floor(value)))
        throw std::invalid_argument(name + " is a whole number from 0 to " + std::to_string(uint64_t(limit)) +
                                    ", not " + std::to_string(value));
}

constexpr double kMaxU32 = 4294967295.0;
constexpr double kMaxU16 = 65535.0;

// A body's mass as Jolt stores it: a positive float whose inverse is finite.
bool usable_mass(double mass) {
    if (!(mass > 0.0 && fits_float(mass)))
        return false;
    const auto narrowed = float(mass);
    return narrowed > 0.0f && std::isfinite(1.0f / narrowed);
}

// Whether a dynamic body of a shape with these mass properties (Jolt's, at its
// own density) and this mass gets an inertia finite and positive as floats.
bool usable_inertia(const JPH::MassProperties &properties, double mass) {
    if (!(properties.mMass > 0.0f))
        return true;  // Jolt fills in a mass of its own
    const double scale = mass / double(properties.mMass);
    for (int axis = 0; axis < 3; ++axis) {
        const double moment = double(properties.mInertia(axis, axis)) * scale;
        if (!(moment > 0.0 && fits_float(moment) && std::isfinite(1.0f / float(moment))))
            return false;
    }
    return true;
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
        // The core keeps no Python object past this point: clear_python also
        // stops a callback still running from storing its exception, and is
        // called again, attached, once the thread has stopped.
        core_->clear_python();
        {
            nb::gil_scoped_release release;
            core_->shutdown();
        }
        core_->clear_python();
        nb::gil_scoped_release release;
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

// Runs before the collector clears any object of a garbage cycle holding the World.
void world_finalize(PyObject *self) {
    if (!nb::inst_ready(self))
        return;
    World *world = nb::inst_ptr<World>(self);
    if (!world->core_ || world->core_->forked())
        return;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    world->core_->retire_python();
    PyErr_Restore(type, value, traceback);
}

PyType_Slot world_slots[] = {{Py_tp_traverse, reinterpret_cast<void *>(world_traverse)},
                             {Py_tp_clear, reinterpret_cast<void *>(world_clear)},
                             {Py_tp_finalize, reinterpret_cast<void *>(world_finalize)},
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
        const Vec3s &angular_velocity, const Words &group, const Words &mask, const Words &flags, \
        const Shorts &material, const Shorts &filter, const Floats &drag
#define PJ_ADD_NAMES                                                                              \
    position, orientation, shape_kind, shape_params, shape_id, scale, shape_offset,              \
        shape_rotation, center_of_mass, motion, mass, density, friction, restitution,             \
        linear_damping, angular_damping, gravity_factor, linear_velocity, angular_velocity,       \
        group, mask, flags, material, filter, drag

// The rows of an add batch, checked, with their shapes resolved.
AddBodies build_add(World &world, const Vec3s &position, const Quats &orientation, const Bytes &shape_kind,
           const Params &shape_params, const Words &shape_id, const Vec3s &scale,
           const Vec3s &shape_offset, const Quats &shape_rotation, const Vec3s &center_of_mass,
           const Bytes &motion, const Doubles &mass, const Doubles &density,
           const Floats &friction, const Floats &restitution, const Floats &linear_damping,
           const Floats &angular_damping, const Floats &gravity_factor,
           const Vec3s &linear_velocity, const Vec3s &angular_velocity, const Words &group,
           const Words &mask, const Words &flags, const Shorts &material, const Shorts &filter,
           const Floats &drag) {
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
    add.material = copy<uint16_t>(material, n, "material");
    add.filter = copy<uint16_t>(filter, n, "filter");
    add.drag = copy<float>(drag, n, "drag");
    for (float k : add.drag)
        if (!(std::isfinite(k) && k >= 0.0f))
            throw std::invalid_argument("drag is finite and not negative");
    check_finite(add.position, "position");
    check_quaternions(add.orientation, "orientation");
    check_finite(offsets, "shape_offset");
    check_quaternions(rotations, "shape_rotation");
    check_finite(coms, "center_of_mass");
    check_floats(add.linear_velocity, "linear_velocity");
    check_floats(add.angular_velocity, "angular_velocity");
    ShapeResolver resolve(world.core().shapes());
    add.shapes.reserve(n);
    // Consecutive rows usually share a shape; its mass properties are read once.
    const JPH::Shape *weighed = nullptr;
    JPH::MassProperties properties;
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
        if (massive) {
            const JPH::Shape &shape = *add.shapes.back();
            if (&shape != weighed) {
                properties = shape.GetMassProperties();
                weighed = &shape;
            }
            const double mass = add.mass[i] > 0.0 ? add.mass[i] : add.density[i] * double(shape.GetVolume());
            if (!usable_mass(mass) || !usable_inertia(properties, mass))
                throw std::invalid_argument("the body's mass " + std::to_string(mass) +
                                            " kg, or the inertia its shape gives it, does not fit a 32-bit float" +
                                            row);
        }
        for (float value : {add.friction[i], add.restitution[i], add.linear_damping[i],
                            add.angular_damping[i]})
            if (!(std::isfinite(value) && value >= 0.0f))
                throw std::invalid_argument(
                    "friction, restitution and damping are finite and not negative" + row);
        if (!std::isfinite(add.gravity_factor[i]))
            throw std::invalid_argument("gravity_factor is not finite" + row);
        if (!std::isfinite(add.friction[i] + add.restitution[i]))
            throw std::invalid_argument("friction and restitution are finite" + row);
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
                 const In<uint32_t, nb::shape<-1, 3>> &triangles, std::optional<Shorts> materials,
                 std::optional<Uids> ids) {
    std::vector<double> vertices(points.data(), points.data() + points.size());
    std::vector<uint32_t> indices(triangles.data(), triangles.data() + triangles.size());
    std::vector<uint16_t> palette;
    if (materials)
        palette = copy<uint16_t>(*materials, triangles.shape(0), "materials");
    std::vector<uint32_t> numbers;
    if (ids)
        numbers = copy<uint32_t>(*ids, triangles.shape(0), "ids");
    JPH::RefConst<JPH::Shape> shape;
    {
        nb::gil_scoped_release release;
        shape = make_mesh(vertices.data(), points.shape(0), indices.data(), triangles.shape(0),
                          palette.empty() ? nullptr : palette.data(), numbers.empty() ? nullptr : numbers.data());
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

uint64_t pushes(World &world, Push kind, const Uids &ids, const Vec3s &values, uint64_t at_tick,
               const std::optional<Vec3s> &points = std::nullopt) {
    AddPushes push{kind, uids(ids), copy<double>(values, ids.shape(0), "values"), {}};
    check_floats(push.values, "values");
    if (points) {
        push.points = copy<double>(*points, ids.shape(0), "points");
        check_finite(push.points, "points");
    }
    return world.core().submit(Command(std::move(push)), at_tick);
}

nb::dict stats(const World &world) {
    const WorldStats s = world.core().stats();
    nb::dict out;
    out["ticks"] = s.ticks;
    out["dropped_ticks"] = s.dropped_ticks;
    out["events_dropped"] = s.events_dropped;
    out["events_held"] = s.events_held;
    out["rejected"] = s.rejected;
    out["pool"] = s.pool;
    out["bodies"] = s.bodies;
    out["queued_bodies"] = s.queued_bodies;
    out["shapes"] = s.shapes;
    out["characters"] = s.characters;
    out["vehicles"] = s.vehicles;
    out["ragdolls"] = s.ragdolls;
    out["soft_bodies"] = s.soft_bodies;
    out["characters_ms"] = s.characters_ms;
    out["ragdolls_ms"] = s.ragdolls_ms;
    out["soft_bodies_ms"] = s.soft_bodies_ms;
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
    out["quiet_ticks"] = s.quiet_ticks;
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


// -- joints ---------------------------------------------------------------------------

using Sixes = In<double, nb::shape<-1, 6>>;
using SixBytes = In<uint8_t, nb::shape<-1, 6>>;
using FloatPairs = In<float, nb::shape<-1, 2>>;

std::pair<nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>, uint64_t> add_joints(
    World &world, const Bytes &kind, const Uids &a, const Uids &b, const Vec3s &anchor, const Vec3s &to_anchor,
    const Vec3s &axis, const Vec3s &axis2, const Vec3s &axis3, const Sixes &lower, const Sixes &upper,
    const FloatPairs &spring, const Floats &friction, const SixBytes &motor_mode, const Sixes &motor_target,
    const Sixes &motor_max, const FloatPairs &motor_spring, const Floats &break_force, const Floats &break_torque,
    const Bytes &collide, uint64_t at_tick) {
    const size_t n = kind.shape(0);
    const auto k = copy<uint8_t>(kind, n, "kind");
    const auto ua = copy<uint32_t>(a, n, "a");
    const auto ub = copy<uint32_t>(b, n, "b");
    const auto p0 = copy<double>(anchor, n, "anchor"), p1 = copy<double>(to_anchor, n, "to_anchor");
    const auto x0 = copy<double>(axis, n, "axis"), x1 = copy<double>(axis2, n, "axis2"),
               x2 = copy<double>(axis3, n, "axis3");
    const auto lo = copy<double>(lower, n, "lower"), hi = copy<double>(upper, n, "upper");
    const auto sp = copy<float>(spring, n, "spring"), fr = copy<float>(friction, n, "friction");
    const auto mm = copy<uint8_t>(motor_mode, n, "motor_mode");
    const auto mt = copy<double>(motor_target, n, "motor_target"), mx = copy<double>(motor_max, n, "motor_max");
    const auto ms = copy<float>(motor_spring, n, "motor_spring");
    const auto bf = copy<float>(break_force, n, "break_force"), bt = copy<float>(break_torque, n, "break_torque");
    const auto co = copy<uint8_t>(collide, n, "collide");
    for (const auto *values : {&p1, &x0, &x1, &x2, &mt})
        check_finite(*values, "joint vectors and motor targets");
    for (size_t i = 0; i < n; ++i) {
        // A slider's anchor may be NaN: the body's origin when the joint is made.
        const bool unset = k[i] == uint8_t(JointKind::Slider) && std::isnan(p0[3 * i]) &&
                           std::isnan(p0[3 * i + 1]) && std::isnan(p0[3 * i + 2]);
        for (size_t c = 0; c < 3 && !unset; ++c)
            if (!std::isfinite(p0[3 * i + c]))
                throw std::invalid_argument("joint anchor holds a value that is not finite");
    }
    AddJoints add;
    add.joints.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const std::string row = " (row " + std::to_string(i) + ")";
        JointDesc &d = add.joints[i];
        if (k[i] > uint8_t(JointKind::Motor))
            throw std::invalid_argument("unknown joint kind " + std::to_string(k[i]) + row);
        if (ub[i] == kNoBody || ua[i] == ub[i])
            throw std::invalid_argument("a joint holds a body to another body or to the world (0)" + row);
        d.kind = k[i];
        d.a = ua[i];
        d.b = ub[i];
        for (size_t c = 0; c < 3; ++c) {
            d.anchor[c] = p0[3 * i + c];
            d.to_anchor[c] = p1[3 * i + c];
            d.axis[c] = x0[3 * i + c];
            d.axis2[c] = x1[3 * i + c];
            d.axis3[c] = x2[3 * i + c];
        }
        for (size_t c = 0; c < 6; ++c) {
            d.lower[c] = lo[6 * i + c];
            d.upper[c] = hi[6 * i + c];
            if (std::isnan(d.lower[c]) || std::isnan(d.upper[c]) || d.lower[c] > d.upper[c])
                throw std::invalid_argument("joint limits are lower <= upper" + row);
            d.motor_mode[c] = mm[6 * i + c];
            if (d.motor_mode[c] > uint8_t(MotorMode::Position))
                throw std::invalid_argument("unknown motor mode" + row);
            d.motor_target[c] = mt[6 * i + c];
            d.motor_max[c] = mx[6 * i + c];
            if (!(d.motor_max[c] >= 0.0))
                throw std::invalid_argument("a motor's most force or torque is not negative" + row);
        }
        for (size_t c = 0; c < 2; ++c) {
            d.spring[c] = sp[2 * i + c];
            d.motor_spring[c] = ms[2 * i + c];
        }
        d.friction = fr[i];
        d.break_force = bf[i];
        d.break_torque = bt[i];
        d.collide = co[i] != 0;
        for (float v : {d.spring[0], d.spring[1], d.motor_spring[0], d.motor_spring[1], d.friction, d.break_force,
                        d.break_torque})
            if (!(std::isfinite(v) && v >= 0.0f))
                throw std::invalid_argument("joint springs, friction and break limits are finite and not negative" + row);
    }
    JointId first = 0;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, nullptr, &first);
    std::vector<uint32_t> ids(n);
    for (size_t i = 0; i < n; ++i)
        ids[i] = first + uint32_t(i);
    return {owned<uint32_t, nb::shape<-1>>(std::move(ids), {n}), seq};
}

uint64_t set_joint_motors(World &world, const Uids &ids, const Bytes &axis, const Bytes &mode, const Doubles &target,
                          const Doubles &max, uint64_t at_tick) {
    const size_t n = ids.shape(0);
    SetJointMotors set{uids(ids), copy<uint8_t>(axis, n, "axis"), copy<uint8_t>(mode, n, "mode"),
                       copy<double>(target, n, "target"), copy<double>(max, n, "max")};
    for (size_t i = 0; i < n; ++i) {
        if (set.axis[i] > 5)
            throw std::invalid_argument("a motor axis is 0 to 5");
        if (set.mode[i] > uint8_t(MotorMode::Position) && set.mode[i] != 0xff)
            throw std::invalid_argument("unknown motor mode");
        if (std::isinf(set.target[i]) || set.max[i] < 0.0)
            throw std::invalid_argument("a motor target is finite and its most force not negative");
    }
    return world.core().submit(Command(std::move(set)), at_tick);
}

// -- materials, filters, zones, drag, tracks -------------------------------------------

uint64_t set_material_table(World &world, const Floats &static_friction, const Floats &dynamic_friction,
                            const Floats &restitution, const Bytes &friction_combine, const Bytes &restitution_combine,
                            const In<float, nb::shape<-1, -1, 3>> &pairs, float static_speed, uint64_t at_tick) {
    const size_t m = static_friction.shape(0);
    SetMaterialTable set;
    set.rows.resize(m);
    const auto sf = copy<float>(static_friction, m, "static_friction");
    const auto df = copy<float>(dynamic_friction, m, "dynamic_friction");
    const auto re = copy<float>(restitution, m, "restitution");
    const auto fc = copy<uint8_t>(friction_combine, m, "friction_combine");
    const auto rc = copy<uint8_t>(restitution_combine, m, "restitution_combine");
    for (size_t i = 0; i < m; ++i) {
        for (float v : {sf[i], df[i], re[i]})
            if (!(std::isfinite(v) && v >= 0.0f))
                throw std::invalid_argument("friction and restitution are finite and not negative");
        if (fc[i] > uint8_t(Combine::Maximum) || rc[i] > uint8_t(Combine::Maximum))
            throw std::invalid_argument("unknown combine mode");
        set.rows[i] = MaterialRow{sf[i], df[i], re[i], fc[i], rc[i]};
    }
    if (pairs.size() != 0 && (pairs.shape(0) != m || pairs.shape(1) != m))
        throw std::invalid_argument("the pair table is (M, M, 3) for M materials");
    set.pairs.assign(pairs.data(), pairs.data() + pairs.size());
    if (!(std::isfinite(static_speed) && static_speed >= 0.0f))
        throw std::invalid_argument("static_speed is finite and not negative");
    set.static_speed = static_speed;
    return world.core().submit(Command(std::move(set)), at_tick);
}

uint64_t set_gravity_zones(World &world, const Bytes &shape, const Bytes &field, const Vec3s &position,
                           const Quats &rotation, const Vec3s &half, const Words &shape_id, const Doubles &gravity,
                           const Vec3s &direction, const Vec3s &center, const In<int32_t, nb::shape<-1>> &priority,
                           const Bytes &replace, const Bytes &stop, uint64_t at_tick) {
    const size_t n = shape.shape(0);
    const auto sh = copy<uint8_t>(shape, n, "shape"), fi = copy<uint8_t>(field, n, "field");
    const auto po = copy<double>(position, n, "position"), ro = copy<double>(rotation, n, "rotation");
    const auto ha = copy<double>(half, n, "half");
    const auto id = copy<uint32_t>(shape_id, n, "shape_id");
    const auto gr = copy<double>(gravity, n, "gravity");
    const auto di = copy<double>(direction, n, "direction"), ce = copy<double>(center, n, "center");
    const auto pr = copy<int32_t>(priority, n, "priority");
    const auto rp = copy<uint8_t>(replace, n, "replace"), st = copy<uint8_t>(stop, n, "stop");
    for (const auto *values : {&po, &ha, &gr, &di, &ce})
        check_finite(*values, "zone values");
    check_quaternions(ro, "rotation");
    SetGravityZones set;
    set.zones.resize(n);
    for (size_t i = 0; i < n; ++i) {
        Zone &z = set.zones[i];
        if (sh[i] > uint8_t(ZoneShape::Shape) || fi[i] > uint8_t(ZoneField::Point))
            throw std::invalid_argument("unknown zone shape or field");
        z.shape = sh[i];
        z.field = fi[i];
        for (size_t c = 0; c < 3; ++c) {
            z.position[c] = po[3 * i + c];
            z.half[c] = ha[3 * i + c];
            z.direction[c] = di[3 * i + c];
            z.center[c] = ce[3 * i + c];
        }
        for (size_t c = 0; c < 4; ++c)
            z.rotation[c] = ro[4 * i + c];
        if (z.shape == uint8_t(ZoneShape::Shape)) {
            z.region = world.core().shapes().find(id[i]).first;
            if (z.region == nullptr)
                throw std::invalid_argument("shape id " + std::to_string(id[i]) + " names no shape of this world");
        }
        z.gravity = gr[i];
        z.priority = pr[i];
        z.replace = rp[i] != 0;
        z.stop = st[i] != 0;
    }
    return world.core().submit(Command(std::move(set)), at_tick);
}

uint64_t set_tracks(World &world, const Uids &ids, const Doubles &times, const Vec3s &positions,
                    const Quats &rotations, bool loop, uint64_t at_tick) {
    SetTracks set;
    set.uids = uids(ids);
    const size_t k = times.shape(0);
    set.track.times = copy<double>(times, k, "times");
    set.track.positions = copy<double>(positions, k, "positions");
    set.track.rotations = copy<double>(rotations, k, "rotations");
    set.track.loop = loop;
    check_finite(set.track.times, "times");
    check_finite(set.track.positions, "positions");
    if (k)
        check_quaternions(set.track.rotations, "rotations");
    for (size_t i = 1; i < k; ++i)
        if (!(set.track.times[i] > set.track.times[i - 1]))
            throw std::invalid_argument("track times ascend");
    return world.core().submit(Command(std::move(set)), at_tick);
}

uint64_t move_kinematic(World &world, const Uids &ids, const Vec3s &positions, const Quats &rotations,
                        uint64_t at_tick) {
    const size_t n = ids.shape(0);
    MoveKinematic move{uids(ids), copy<double>(positions, n, "positions"), copy<double>(rotations, n, "rotations")};
    check_finite(move.positions, "positions");
    check_quaternions(move.rotations, "rotations");
    if (world.core().on_world_thread()) {
        world.core().move_kinematic_now(std::move(move));
        return 0;
    }
    return world.core().submit(Command(std::move(move)), at_tick);
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> char_view(const Snapshot &s, const std::vector<T> &values, size_t width) {
    if (width == 1)
        return view<T, Shape>(values, {s.buffer->characters});
    return view<T, Shape>(values, {s.buffer->characters, width});
}

// -- vehicles -------------------------------------------------------------------------

using Pairs = In<float, nb::shape<-1, 2>>;
using Ints = In<int32_t, nb::shape<-1>>;

std::vector<float> curve(const Pairs &pairs, const char *name) {
    std::vector<float> out(pairs.data(), pairs.data() + pairs.size());
    check_finite(out, name);
    for (size_t i = 2; i < out.size(); i += 2)
        if (!(out[i] > out[i - 2]))
            throw std::invalid_argument(std::string(name) + " x values ascend");
    return out;
}

std::pair<uint32_t, uint64_t> add_vehicle(World &world, Uid chassis, uint8_t kind, uint8_t cast, const Doubles &params,
                                          const In<double, nb::shape<-1, kWheelParamCount>> &wheels,
                                          const Pairs &torque_curve, const Pairs &longitudinal, const Pairs &lateral,
                                          const Floats &gears, const Floats &reverse_gears,
                                          const In<double, nb::shape<-1, 6>> &differentials,
                                          const In<double, nb::shape<-1, 3>> &anti_roll_bars,
                                          const In<double, nb::shape<2, 5>> &tracks, uint64_t at_tick) {
    if (kind > uint8_t(VehicleKind::Tracked))
        throw std::invalid_argument("unknown vehicle kind " + std::to_string(kind));
    if (cast > uint8_t(WheelCast::Cylinder))
        throw std::invalid_argument("unknown wheel cast " + std::to_string(cast));
    if (params.shape(0) != kParamCount)
        throw std::invalid_argument("params has " + std::to_string(params.shape(0)) + " values where " +
                                    std::to_string(kParamCount) + " are needed");
    AddVehicle add;
    VehicleDesc &d = add.desc;
    d.chassis = chassis;
    d.kind = kind;
    d.cast = cast;
    d.params.assign(params.data(), params.data() + kParamCount);
    for (size_t i = 0; i < kParamCount; ++i)
        if (!fits_float(d.params[i]) && !(i == kDiffLimitedSlip && d.params[i] > 0))
            throw std::invalid_argument("vehicle setting " + std::to_string(i) + " is not finite as a 32-bit float");
    const size_t n = wheels.shape(0);
    if (n == 0)
        throw std::invalid_argument("a vehicle has at least one wheel");
    d.wheels.assign(wheels.data(), wheels.data() + wheels.size());
    check_floats(d.wheels, "wheels");
    for (size_t i = 0; i < n; ++i) {
        const std::string row = " (wheel " + std::to_string(i) + ")";
        if (!(d.wheel(i, kRadius) > 0.0 && d.wheel(i, kWidth) >= 0.0))
            throw std::invalid_argument("a wheel's radius is positive and its width not negative" + row);
        if (!(d.wheel(i, kSuspensionMin) >= 0.0 && d.wheel(i, kSuspensionMax) >= d.wheel(i, kSuspensionMin)))
            throw std::invalid_argument("a wheel's suspension lengths are 0 <= min <= max" + row);
        if (!(d.wheel(i, kStiffness) > 0.0 && d.wheel(i, kDamping) >= 0.0))
            throw std::invalid_argument("a wheel's spring stiffness is positive and its damping not negative" + row);
        if (!(d.wheel(i, kWheelInertia) > 0.0))
            throw std::invalid_argument("a wheel's inertia is positive" + row);
        const double side = d.wheel(i, kTrackSide);
        if (!(side == -1.0 || side == 0.0 || side == 1.0))
            throw std::invalid_argument("a wheel's track is -1 (none), 0 (left) or 1 (right)" + row);
    }
    if (!(d.params[kMaxRpm] > d.params[kMinRpm] && d.params[kMinRpm] >= 0.0 && d.params[kMaxTorque] >= 0.0))
        throw std::invalid_argument("an engine's rpm range is 0 <= min < max and its torque not negative");
    if (!(d.params[kEngineInertia] > 0.0))
        throw std::invalid_argument("an engine's inertia is positive");
    d.torque_curve = curve(torque_curve, "torque_curve");
    d.longitudinal = curve(longitudinal, "longitudinal_friction");
    d.lateral = curve(lateral, "lateral_friction");
    d.gears.assign(gears.data(), gears.data() + gears.size());
    d.reverse_gears.assign(reverse_gears.data(), reverse_gears.data() + reverse_gears.size());
    check_finite(d.gears, "gears");
    check_finite(d.reverse_gears, "reverse_gears");
    if (d.gears.empty() || d.reverse_gears.empty())
        throw std::invalid_argument("a transmission has at least one forward and one reverse gear");
    auto wheel_index = [&](double value, bool optional) {
        // Compared as a double first: converting one outside int's range is undefined.
        if (!(value >= (optional ? -1.0 : 0.0) && value < double(n) && value == std::floor(value)))
            throw std::invalid_argument("wheel index " + std::to_string(value) + " names no wheel");
        return int(value);
    };
    auto floats = [](const double *row, size_t count, const char *name) {
        check_floats(std::vector<double>(row, row + count), name);
    };
    for (size_t i = 0; i < differentials.shape(0); ++i) {
        const double *r = differentials.data() + 6 * i;
        // A limited-slip ratio of inf is an open differential.
        floats(r + 2, 2, "differentials");
        floats(r + 5, 1, "differentials");
        if (!(fits_float(r[4]) || (std::isinf(r[4]) && r[4] > 0)))
            throw std::invalid_argument("a differential's limited-slip ratio is finite or inf");
        DifferentialDesc &out = d.differentials.emplace_back();
        out.left = wheel_index(r[0], true);
        out.right = wheel_index(r[1], true);
        out.ratio = float(r[2]);
        out.split = float(r[3]);
        out.limited_slip = float(r[4]);
        out.torque_ratio = float(r[5]);
    }
    for (size_t i = 0; i < anti_roll_bars.shape(0); ++i) {
        const double *r = anti_roll_bars.data() + 3 * i;
        floats(r + 2, 1, "anti_roll_bars");
        AntiRollBarDesc &out = d.anti_roll_bars.emplace_back();
        out.left = wheel_index(r[0], false);
        out.right = wheel_index(r[1], false);
        out.stiffness = float(r[2]);
        if (!(out.stiffness >= 0.0f))
            throw std::invalid_argument("an anti-roll bar's stiffness is not negative");
    }
    for (int side = 0; side < 2; ++side) {
        const double *r = tracks.data() + 5 * side;
        floats(r + 1, 4, "tracks");
        if (kind == uint8_t(VehicleKind::Tracked) && !(r[1] > 0.0))
            throw std::invalid_argument("a track's inertia is positive");
        TrackDesc &out = d.tracks[side];
        out.driven = kind == uint8_t(VehicleKind::Tracked) ? wheel_index(r[0], false) : -1;
        out.inertia = float(r[1]);
        out.damping = float(r[2]);
        out.brake_torque = float(r[3]);
        out.ratio = float(r[4]);
    }
    VehicleId id = 0;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, nullptr, nullptr, &id);
    return {id, seq};
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> veh_view(const Snapshot &s, const std::vector<T> &values, size_t width,
                                                bool wheels) {
    const size_t rows = wheels ? s.buffer->wheels : s.buffer->vehicles;
    if (width == 1)
        return view<T, Shape>(values, {rows});
    return view<T, Shape>(values, {rows, width});
}

// -- ragdolls -------------------------------------------------------------------------

std::tuple<uint32_t, uint32_t, uint64_t> add_ragdoll(World &world, const Doubles &params,
                                                     const In<double, nb::shape<-1, kBoneParamCount>> &bones,
                                                     const Words &shape_id, uint64_t at_tick) {
    if (params.shape(0) != kRagParamCount)
        throw std::invalid_argument("params has " + std::to_string(params.shape(0)) + " values where " +
                                    std::to_string(kRagParamCount) + " are needed");
    AddRagdoll add;
    RagdollDesc &d = add.desc;
    d.params.assign(params.data(), params.data() + kRagParamCount);
    check_finite(d.params, "params");
    check_floats(std::vector<double>(d.params.begin() + kRagQX, d.params.end()), "params");
    check_whole(d.params[kRagGroup], kMaxU32, "group");
    check_whole(d.params[kRagMask], kMaxU32, "mask");
    check_whole(d.params[kRagFilter], kMaxU16, "filter");
    check_whole(d.params[kRagMaterial], kMaxU16, "material");
    check_whole(d.params[kRagFlags], kMaxU32, "flags");
    if (d.params[kRagMode] < 0 || d.params[kRagMode] > double(uint8_t(RagdollMode::Limp)))
        throw std::invalid_argument("unknown ragdoll mode " + std::to_string(d.params[kRagMode]));
    check_quaternions(std::vector<double>(d.params.begin() + kRagQX, d.params.begin() + kRagQW + 1), "rotation");
    for (RagdollParam k : {kRagMotorFrequency, kRagMotorDamping, kRagRootFrequency, kRagRootDamping, kRagRootForce,
                           kRagRootTorque, kRagFriction, kRagRestitution, kRagLinearDamping, kRagAngularDamping})
        if (d.params[k] < 0.0)
            throw std::invalid_argument("ragdoll setting " + std::to_string(k) + " is negative");
    const size_t n = bones.shape(0);
    if (n == 0)
        throw std::invalid_argument("a ragdoll has at least one bone");
    if (shape_id.shape(0) != n)
        throw std::invalid_argument("shape_id has a row a bone");
    d.bones.assign(bones.data(), bones.data() + bones.size());
    check_floats(d.bones, "bones");
    ShapeResolver resolve(world.core().shapes());
    for (size_t i = 0; i < n; ++i) {
        const std::string row = " (bone " + std::to_string(i) + ")";
        const double parent = d.bone(i, kBoneParent);
        if (i == 0 ? parent != -1.0 : !(parent >= 0.0 && parent < double(i) && parent == std::floor(parent)))
            throw std::invalid_argument("a bone's parent comes before it, and only the first bone has none" + row);
        if (!usable_mass(d.bone(i, kBoneMass)))
            throw std::invalid_argument("a bone's mass is positive and fits a 32-bit float" + row);
        if (!(d.bone(i, kBoneTwistMin) <= d.bone(i, kBoneTwistMax)))
            throw std::invalid_argument("a bone's twist limits are min <= max" + row);
        if (std::abs(d.bone(i, kBoneBend)) > 3.0)
            throw std::invalid_argument("a bone's bend is within 3 rad" + row);
        for (BoneParam k : {kBoneNormalCone, kBonePlaneCone, kBoneStrength, kBoneFriction})
            if (d.bone(i, k) < 0.0)
                throw std::invalid_argument("a bone's cone angles, strength and friction are not negative" + row);
        const double *b = &d.bones[i * kBoneParamCount];
        check_quaternions(std::vector<double>(b + kBoneRX, b + kBoneRW + 1), "bone rotation");
        check_quaternions(std::vector<double>(b + kBoneQX, b + kBoneQW + 1), "bone shape rotation");
        if (std::abs(b[kBoneTwistX]) + std::abs(b[kBoneTwistY]) + std::abs(b[kBoneTwistZ]) == 0.0)
            throw std::invalid_argument("a bone's twist axis is not zero" + row);
        Decoration decoration;
        for (int c = 0; c < 3; ++c)
            decoration.offset[c] = float(b[kBoneOX + c]);
        for (int c = 0; c < 4; ++c)
            decoration.rotation[c] = float(b[kBoneQX + c]);
        d.shapes.push_back(resolve(0, ShapeParams{}, shape_id.data()[i], decoration, row));
        if (!resolve.dynamic_ok())
            throw std::invalid_argument("a bone's shape can be dynamic (no mesh, height field or plane)" + row);
        if (!usable_inertia(d.shapes.back()->GetMassProperties(), d.bone(i, kBoneMass)))
            throw std::invalid_argument("the inertia a bone's shape and mass give it does not fit a 32-bit float" + row);
    }
    RagdollId id = 0;
    Uid first = 0;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, &first, nullptr, nullptr, &id);
    return {id, first, seq};
}

uint64_t set_ragdoll_pose(World &world, uint32_t id, const Quats &rotations, const Vec3s &translations,
                          std::array<double, 3> position, std::array<double, 4> rotation, uint64_t at_tick) {
    SetRagdollPose pose;
    pose.id = id;
    pose.rotations = copy<double>(rotations, rotations.shape(0), "rotations");
    pose.translations = copy<double>(translations, translations.shape(0), "translations");
    check_quaternions(pose.rotations, "rotations");
    check_finite(pose.translations, "translations");
    for (int c = 0; c < 3; ++c)
        pose.position[c] = position[c];
    for (int c = 0; c < 4; ++c)
        pose.rotation[c] = rotation[c];
    if (!std::isnan(position[0])) {
        check_finite(std::vector<double>(position.begin(), position.end()), "position");
        check_quaternions(std::vector<double>(rotation.begin(), rotation.end()), "rotation");
    }
    return world.core().submit(Command(std::move(pose)), at_tick);
}

uint64_t set_ragdoll_clip(World &world, uint32_t id, const Doubles &times, const Quats &rotations,
                          const Vec3s &translations, bool loop, uint64_t at_tick) {
    SetRagdollClip clip;
    clip.id = id;
    clip.loop = loop;
    clip.times.assign(times.data(), times.data() + times.size());
    clip.rotations = copy<double>(rotations, rotations.shape(0), "rotations");
    clip.translations = copy<double>(translations, translations.shape(0), "translations");
    check_finite(clip.times, "times");
    check_quaternions(clip.rotations, "rotations");
    check_finite(clip.translations, "translations");
    if (!clip.times.empty() && clip.times[0] != 0.0)
        throw std::invalid_argument("clip times start at 0");
    for (size_t i = 1; i < clip.times.size(); ++i)
        if (!(clip.times[i] > clip.times[i - 1]))
            throw std::invalid_argument("clip times ascend");
    return world.core().submit(Command(std::move(clip)), at_tick);
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> rag_view(const Snapshot &s, const std::vector<T> &values, size_t width,
                                                bool bones) {
    const size_t rows = bones ? s.buffer->bones : s.buffer->ragdolls;
    if (width == 1)
        return view<T, Shape>(values, {rows});
    return view<T, Shape>(values, {rows, width});
}

// -- soft bodies ----------------------------------------------------------------------

std::vector<SoftPin> soft_pins(const Words &vertex, const Uids &body, const Vec3s &point, size_t vertices) {
    const size_t n = vertex.shape(0);
    const std::vector<uint32_t> v = copy<uint32_t>(vertex, n, "pin vertices");
    const std::vector<Uid> b = copy<Uid>(body, n, "pin bodies");
    const std::vector<double> p = copy<double>(point, n, "pin points");
    check_finite(p, "pin points");
    std::vector<SoftPin> out(n);
    for (size_t i = 0; i < n; ++i) {
        if (vertices && v[i] >= vertices)
            throw std::invalid_argument("pin vertex " + std::to_string(v[i]) + " names no vertex");
        out[i].vertex = v[i];
        out[i].body = b[i];
        for (int c = 0; c < 3; ++c)
            out[i].point[c] = p[3 * i + c];
    }
    return out;
}

std::pair<uint32_t, uint64_t> add_soft_body(World &world, const Doubles &params, const Vec3s &positions,
                                            const Floats &inv_mass, const In<uint32_t, nb::shape<-1, 3>> &faces,
                                            const In<uint32_t, nb::shape<-1, 2>> &edges, const Floats &edge_compliance,
                                            const In<uint32_t, nb::shape<-1, 4>> &tetrahedra, const Words &pin_vertex,
                                            const Uids &pin_body, const Vec3s &pin_point, uint64_t at_tick) {
    if (params.shape(0) != kSoftParamCount)
        throw std::invalid_argument("params has " + std::to_string(params.shape(0)) + " values where " +
                                    std::to_string(kSoftParamCount) + " are needed");
    AddSoftBody add;
    SoftBodyDesc &d = add.desc;
    d.params.assign(params.data(), params.data() + kSoftParamCount);
    for (size_t i = 0; i < kSoftParamCount; ++i) {
        const bool compliance = i == kSoftCompliance || i == kSoftShearCompliance || i == kSoftBendCompliance ||
                                i == kSoftVolumeCompliance;
        if (compliance && std::isinf(d.params[i]) && d.params[i] > 0)
            continue;  // inf compliance: no constraint of that kind
        if (!std::isfinite(d.params[i]) || (i >= kSoftQX && !fits_float(d.params[i])))
            throw std::invalid_argument("soft body setting " + std::to_string(i) + " is not finite as a 32-bit float");
    }
    if (!(d.params[kSoftIterations] >= 1.0 && d.params[kSoftIterations] <= double(kMaxSoftIterations) &&
          d.params[kSoftIterations] == std::floor(d.params[kSoftIterations])))
        throw std::invalid_argument("a soft body takes a whole number of solver iterations from 1 to " +
                                    std::to_string(kMaxSoftIterations));
    check_whole(d.params[kSoftLra], 2.0, "the long-range attachment kind");
    check_whole(d.params[kSoftBendType], 2.0, "the bend kind");
    check_whole(d.params[kSoftGroup], kMaxU32, "group");
    check_whole(d.params[kSoftMask], kMaxU32, "mask");
    check_whole(d.params[kSoftFilter], kMaxU16, "filter");
    check_whole(d.params[kSoftMaterial], kMaxU16, "material");
    check_whole(d.params[kSoftFlags], kMaxU32, "flags");
    check_quaternions(std::vector<double>(d.params.begin() + kSoftQX, d.params.begin() + kSoftQW + 1), "rotation");
    for (SoftParam k : {kSoftCompliance, kSoftShearCompliance, kSoftBendCompliance, kSoftVolumeCompliance,
                        kSoftPressure, kSoftLraStretch, kSoftLinearDamping, kSoftMaxVelocity, kSoftFriction,
                        kSoftRestitution, kSoftVertexRadius})
        if (d.params[k] < 0.0)
            throw std::invalid_argument("soft body setting " + std::to_string(k) + " is negative");
    const size_t n = positions.shape(0);
    if (n == 0)
        throw std::invalid_argument("a soft body has at least one vertex");
    const std::vector<double> points = copy<double>(positions, n, "positions");
    check_finite(points, "positions");
    d.positions.assign(points.begin(), points.end());
    d.inv_mass = copy<float>(inv_mass, n, "inv_mass");
    for (float m : d.inv_mass)
        if (!(std::isfinite(m) && m >= 0.0f))
            throw std::invalid_argument("inv_mass is finite and not negative");
    d.faces.assign(faces.data(), faces.data() + faces.size());
    d.edges.assign(edges.data(), edges.data() + edges.size());
    d.edge_compliance = copy<float>(edge_compliance, edges.shape(0), "edge_compliance");
    d.tetrahedra.assign(tetrahedra.data(), tetrahedra.data() + tetrahedra.size());
    for (const std::vector<uint32_t> *indices : {&d.faces, &d.edges, &d.tetrahedra})
        for (uint32_t v : *indices)
            if (v >= n)
                throw std::invalid_argument("a face, edge or tetrahedron names vertex " + std::to_string(v) +
                                            " of " + std::to_string(n));
    if (d.params[kSoftPressure] > 0.0 && d.faces.empty())
        throw std::invalid_argument("a soft body with pressure has faces");
    d.pins = soft_pins(pin_vertex, pin_body, pin_point, n);
    // The settings are built now, on the calling thread: a fault is raised here
    // and the world thread only creates the body.
    d.shared = soft_settings(d);
    Uid uid = 0;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, &uid);
    return {uid, seq};
}

template <typename T, typename Shape>
nb::ndarray<nb::numpy, const T, Shape> soft_view(const Snapshot &s, const std::vector<T> &values, size_t width,
                                                 bool vertices) {
    const size_t rows = vertices ? s.buffer->soft_vertex_total : s.buffer->soft_bodies;
    if (width == 1)
        return view<T, Shape>(values, {rows});
    return view<T, Shape>(values, {rows, width});
}

}  // namespace

void bind_world(nb::module_ &m) {
    keep_callback_logger();
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
            "character_previous",
            [](const Snapshot &s) { return char_view<double, nb::shape<-1, 3>>(s, s.buffer->char_previous, 3); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_eye",
            [](const Snapshot &s) { return char_view<double, nb::shape<-1, 3>>(s, s.buffer->char_eye, 3); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_ground_material",
            [](const Snapshot &s) {
                return char_view<uint16_t, nb::shape<-1>>(s, s.buffer->char_ground_material, 1);
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "character_flags",
            [](const Snapshot &s) { return char_view<uint8_t, nb::shape<-1>>(s, s.buffer->char_flags, 1); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("joint_count", [](const Snapshot &s) { return s.buffer->joints; })
        .def_prop_ro(
            "joint_id",
            [](const Snapshot &s) { return view<uint32_t, nb::shape<-1>>(s.buffer->joint_id, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_kind",
            [](const Snapshot &s) { return view<uint8_t, nb::shape<-1>>(s.buffer->joint_kind, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_state",
            [](const Snapshot &s) { return view<uint8_t, nb::shape<-1>>(s.buffer->joint_state, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_value",
            [](const Snapshot &s) { return view<float, nb::shape<-1>>(s.buffer->joint_value, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_value2",
            [](const Snapshot &s) { return view<float, nb::shape<-1>>(s.buffer->joint_value2, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_force",
            [](const Snapshot &s) { return view<float, nb::shape<-1>>(s.buffer->joint_force, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "joint_torque",
            [](const Snapshot &s) { return view<float, nb::shape<-1>>(s.buffer->joint_torque, {s.buffer->joints}); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("vehicle_count", [](const Snapshot &s) { return s.buffer->vehicles; })
        .def_prop_ro("wheel_count", [](const Snapshot &s) { return s.buffer->wheels; })
        .def_prop_ro(
            "vehicle_id",
            [](const Snapshot &s) { return veh_view<uint32_t, nb::shape<-1>>(s, s.buffer->veh_id, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_chassis",
            [](const Snapshot &s) { return veh_view<Uid, nb::shape<-1>>(s, s.buffer->veh_chassis, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_kind",
            [](const Snapshot &s) { return veh_view<uint8_t, nb::shape<-1>>(s, s.buffer->veh_kind, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_first_wheel",
            [](const Snapshot &s) { return veh_view<uint32_t, nb::shape<-1>>(s, s.buffer->veh_first_wheel, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_wheels",
            [](const Snapshot &s) { return veh_view<uint32_t, nb::shape<-1>>(s, s.buffer->veh_wheels, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_rpm",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->veh_rpm, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_gear",
            [](const Snapshot &s) { return veh_view<int8_t, nb::shape<-1>>(s, s.buffer->veh_gear, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_forward_speed",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->veh_forward_speed, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_steer",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->veh_steer, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_impact_speed",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->veh_impact_speed, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_impact_impulse",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->veh_impact_impulse, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_impact_uid",
            [](const Snapshot &s) { return veh_view<Uid, nb::shape<-1>>(s, s.buffer->veh_impact_uid, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "vehicle_input",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1, 4>>(s, s.buffer->veh_input, 4, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_vehicle",
            [](const Snapshot &s) { return veh_view<uint32_t, nb::shape<-1>>(s, s.buffer->wheel_vehicle, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_position",
            [](const Snapshot &s) { return veh_view<double, nb::shape<-1, 3>>(s, s.buffer->wheel_position, 3, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_orientation",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1, 4>>(s, s.buffer->wheel_orientation, 4, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_length",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_length, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_compression",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_compression, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_steer",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_steer, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_angle",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_angle, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_spin",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_spin, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_load",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_load, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_slip",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1, 2>>(s, s.buffer->wheel_slip, 2, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_contact",
            [](const Snapshot &s) { return veh_view<uint8_t, nb::shape<-1>>(s, s.buffer->wheel_contact, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_ground",
            [](const Snapshot &s) { return veh_view<Uid, nb::shape<-1>>(s, s.buffer->wheel_ground, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_material",
            [](const Snapshot &s) { return veh_view<uint16_t, nb::shape<-1>>(s, s.buffer->wheel_material, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_triangle",
            [](const Snapshot &s) { return veh_view<int32_t, nb::shape<-1>>(s, s.buffer->wheel_triangle, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_grip",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1>>(s, s.buffer->wheel_grip, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_contact_point",
            [](const Snapshot &s) { return veh_view<double, nb::shape<-1, 3>>(s, s.buffer->wheel_contact_point, 3, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "wheel_contact_normal",
            [](const Snapshot &s) { return veh_view<float, nb::shape<-1, 3>>(s, s.buffer->wheel_contact_normal, 3, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("ragdoll_count", [](const Snapshot &s) { return s.buffer->ragdolls; })
        .def_prop_ro("bone_count", [](const Snapshot &s) { return s.buffer->bones; })
        .def_prop_ro(
            "ragdoll_id", [](const Snapshot &s) { return rag_view<uint32_t, nb::shape<-1>>(s, s.buffer->rag_id, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "ragdoll_mode", [](const Snapshot &s) { return rag_view<uint8_t, nb::shape<-1>>(s, s.buffer->rag_mode, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "ragdoll_strength",
            [](const Snapshot &s) { return rag_view<float, nb::shape<-1>>(s, s.buffer->rag_strength, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "ragdoll_self_collision",
            [](const Snapshot &s) { return rag_view<uint8_t, nb::shape<-1>>(s, s.buffer->rag_self_collision, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "ragdoll_first_bone",
            [](const Snapshot &s) { return rag_view<uint32_t, nb::shape<-1>>(s, s.buffer->rag_first_bone, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "ragdoll_bones",
            [](const Snapshot &s) { return rag_view<uint32_t, nb::shape<-1>>(s, s.buffer->rag_bones, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "bone_uid", [](const Snapshot &s) { return rag_view<Uid, nb::shape<-1>>(s, s.buffer->bone_uid, 1, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "bone_position",
            [](const Snapshot &s) { return rag_view<double, nb::shape<-1, 3>>(s, s.buffer->bone_position, 3, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "bone_orientation",
            [](const Snapshot &s) { return rag_view<float, nb::shape<-1, 4>>(s, s.buffer->bone_orientation, 4, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro("soft_count", [](const Snapshot &s) { return s.buffer->soft_bodies; })
        .def_prop_ro("soft_vertex_total", [](const Snapshot &s) { return s.buffer->soft_vertex_total; })
        .def_prop_ro(
            "soft_uid", [](const Snapshot &s) { return soft_view<Uid, nb::shape<-1>>(s, s.buffer->soft_uid, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_first_vertex",
            [](const Snapshot &s) { return soft_view<uint32_t, nb::shape<-1>>(s, s.buffer->soft_first_vertex, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_vertices",
            [](const Snapshot &s) { return soft_view<uint32_t, nb::shape<-1>>(s, s.buffer->soft_vertices, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_version",
            [](const Snapshot &s) { return soft_view<uint64_t, nb::shape<-1>>(s, s.buffer->soft_version, 1, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_origin",
            [](const Snapshot &s) { return soft_view<double, nb::shape<-1, 3>>(s, s.buffer->soft_origin, 3, false); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_position",
            [](const Snapshot &s) { return soft_view<float, nb::shape<-1, 3>>(s, s.buffer->soft_position, 3, true); },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_normal",
            [](const Snapshot &s) { return soft_view<float, nb::shape<-1, 3>>(s, s.buffer->soft_normal, 3, true); },
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
               int collision_steps, uint32_t temp_allocator_mb, float static_friction_speed,
               float penetration_slop) {
                if (!(tick_hz >= 1.0 && tick_hz <= 10000.0))
                    throw std::invalid_argument("tick_hz is from 1 to 10000");
                if (temp_allocator_mb > 4095)
                    throw std::invalid_argument("temp_allocator_mb is from 0 to 4095");
                for (double g : gravity)
                    if (!fits_float(g))
                        throw std::invalid_argument("gravity is not finite as a 32-bit float");
                for (float value : {time_before_sleep, sleep_velocity, min_velocity_for_restitution})
                    if (!(std::isfinite(value) && value >= 0.0f))
                        throw std::invalid_argument(
                            "time_before_sleep, sleep_velocity and min_velocity_for_restitution are finite and "
                            "not negative");
                if (!(std::isfinite(max_catch_up) && max_catch_up >= 0.0))
                    throw std::invalid_argument("max_catch_up is zero or positive");
                if (threads < 1 || threads > 256)
                    throw std::invalid_argument("threads is from 1 to 256");
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
                if (!(std::isfinite(static_friction_speed) && static_friction_speed >= 0.0f))
                    throw std::invalid_argument("static_friction_speed is finite and not negative");
                settings.static_friction_speed = static_friction_speed;
                if (!(std::isfinite(penetration_slop) && penetration_slop >= 0.0f))
                    throw std::invalid_argument("penetration_slop is finite and not negative");
                settings.penetration_slop = penetration_slop;
                new (self) World(settings);
            },
            nb::kw_only(), nb::arg("tick_hz") = 120.0, nb::arg("max_catch_up") = 0.1,
            nb::arg("gravity").sig("(0.0, -9.81, 0.0)") = std::array<double, 3>{0.0, -9.81, 0.0}, nb::arg("threads") = 4,
            nb::arg("max_bodies") = 65536, nb::arg("max_body_pairs") = 65536,
            nb::arg("max_contact_constraints") = 65536, nb::arg("event_capacity") = 65536,
            nb::arg("allow_sleeping") = true, nb::arg("time_before_sleep") = 0.5f,
            nb::arg("sleep_velocity").sig("0.03") = 0.03f, nb::arg("min_velocity_for_restitution") = 1.0f,
            nb::arg("collision_steps") = 1, nb::arg("temp_allocator_mb") = 16,
            nb::arg("static_friction_speed").sig("0.05") = 0.05f,
            nb::arg("penetration_slop").sig("0.02") = 0.02f)
        .def("add_bodies", &add_bodies, nb::arg("position"), nb::arg("orientation"),
             nb::arg("shape_kind"), nb::arg("shape_params"), nb::arg("shape_id"),
             nb::arg("scale"), nb::arg("shape_offset"), nb::arg("shape_rotation"),
             nb::arg("center_of_mass"), nb::arg("motion"), nb::arg("mass"),
             nb::arg("density"), nb::arg("friction"), nb::arg("restitution"),
             nb::arg("linear_damping"), nb::arg("angular_damping"), nb::arg("gravity_factor"),
             nb::arg("linear_velocity"), nb::arg("angular_velocity"), nb::arg("group"),
             nb::arg("mask"), nb::arg("flags"), nb::arg("material"), nb::arg("filter"), nb::arg("drag"),
             nb::arg("at_tick") = 0, "Queue bodies to add; return their uids and the batch's sequence number.")
        .def("swap_bodies", &swap_bodies, nb::arg("remove"), nb::arg("position"), nb::arg("orientation"),
             nb::arg("shape_kind"), nb::arg("shape_params"), nb::arg("shape_id"),
             nb::arg("scale"), nb::arg("shape_offset"), nb::arg("shape_rotation"),
             nb::arg("center_of_mass"), nb::arg("motion"), nb::arg("mass"),
             nb::arg("density"), nb::arg("friction"), nb::arg("restitution"),
             nb::arg("linear_damping"), nb::arg("angular_damping"), nb::arg("gravity_factor"),
             nb::arg("linear_velocity"), nb::arg("angular_velocity"), nb::arg("group"),
             nb::arg("mask"), nb::arg("flags"), nb::arg("material"), nb::arg("filter"), nb::arg("drag"),
             nb::arg("at_tick") = 0,
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
        .def("add_mesh", &add_mesh, nb::arg("points"), nb::arg("triangles"), nb::arg("materials").none() = nb::none(),
             nb::arg("ids").none() = nb::none(),
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
                check_floats(set.linear, "linear");
                check_floats(set.angular, "angular");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("linear").none(), nb::arg("angular").none(),
            nb::arg("at_tick") = 0)
        .def(
            "add_forces",
            [](World &w, const Uids &ids, const Vec3s &values, uint64_t at_tick, const std::optional<Vec3s> &points) {
                return pushes(w, Push::Force, ids, values, at_tick, points);
            },
            nb::arg("uids"), nb::arg("values"), nb::arg("at_tick") = 0, nb::arg("points") = nb::none())
        .def(
            "add_impulses",
            [](World &w, const Uids &ids, const Vec3s &values, uint64_t at_tick, const std::optional<Vec3s> &points) {
                return pushes(w, Push::Impulse, ids, values, at_tick, points);
            },
            nb::arg("uids"), nb::arg("values"), nb::arg("at_tick") = 0, nb::arg("points") = nb::none())
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
                    if (!fits_float(g))
                        throw std::invalid_argument("gravity is not finite as a 32-bit float");
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
        .def("add_joints", &add_joints, nb::arg("kind"), nb::arg("a"), nb::arg("b"), nb::arg("anchor"),
             nb::arg("to_anchor"), nb::arg("axis"), nb::arg("axis2"), nb::arg("axis3"), nb::arg("lower"),
             nb::arg("upper"), nb::arg("spring"), nb::arg("friction"), nb::arg("motor_mode"),
             nb::arg("motor_target"), nb::arg("motor_max"), nb::arg("motor_spring"), nb::arg("break_force"),
             nb::arg("break_torque"), nb::arg("collide"), nb::arg("at_tick") = 0,
             "Queue joints; return their ids and the batch's sequence number.")
        .def(
            "remove_joints",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(RemoveJoints{uids(ids)}), at_tick);
            },
            nb::arg("ids"), nb::arg("at_tick") = 0)
        .def("set_joint_motors", &set_joint_motors, nb::arg("ids"), nb::arg("axis"), nb::arg("mode"),
             nb::arg("target"), nb::arg("max"), nb::arg("at_tick") = 0)
        .def(
            "set_joints_enabled",
            [](World &w, const Uids &ids, const Bytes &enabled, uint64_t at_tick) {
                SetJointsEnabled set{uids(ids), copy<uint8_t>(enabled, ids.shape(0), "enabled")};
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("ids"), nb::arg("enabled"), nb::arg("at_tick") = 0)
        .def("add_ragdoll", &add_ragdoll, nb::arg("params"), nb::arg("bones"), nb::arg("shape_id"),
             nb::arg("at_tick") = 0,
             "Queue a ragdoll; return its id, its first bone's uid and the batch's sequence number.")
        .def(
            "remove_ragdolls",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(RemoveRagdolls{uids(ids)}), at_tick);
            },
            nb::arg("ids"), nb::arg("at_tick") = 0)
        .def(
            "set_ragdoll_modes",
            [](World &w, const Uids &ids, uint8_t mode, double blend, uint64_t at_tick) {
                if (mode > uint8_t(RagdollMode::Limp))
                    throw std::invalid_argument("unknown ragdoll mode " + std::to_string(mode));
                if (!(std::isfinite(blend) && blend >= 0.0))
                    throw std::invalid_argument("blend is finite and not negative");
                return w.core().submit(Command(SetRagdollModes{uids(ids), mode, blend}), at_tick);
            },
            nb::arg("ids"), nb::arg("mode"), nb::arg("blend"), nb::arg("at_tick") = 0)
        .def("set_ragdoll_pose", &set_ragdoll_pose, nb::arg("id"), nb::arg("rotations"), nb::arg("translations"),
             nb::arg("position"), nb::arg("rotation"), nb::arg("at_tick") = 0)
        .def("set_ragdoll_clip", &set_ragdoll_clip, nb::arg("id"), nb::arg("times"), nb::arg("rotations"),
             nb::arg("translations"), nb::arg("loop"), nb::arg("at_tick") = 0)
        .def(
            "set_ragdoll_strength",
            [](World &w, uint32_t id, const Doubles &strength, uint64_t at_tick) {
                SetRagdollStrength set{id, std::vector<double>(strength.data(), strength.data() + strength.size())};
                for (double v : set.strength)
                    if (!(std::isfinite(v) && v >= 0.0))
                        throw std::invalid_argument("strength is finite and not negative");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("id"), nb::arg("strength"), nb::arg("at_tick") = 0)
        .def("add_soft_body", &add_soft_body, nb::arg("params"), nb::arg("positions"), nb::arg("inv_mass"),
             nb::arg("faces"), nb::arg("edges"), nb::arg("edge_compliance"), nb::arg("tetrahedra"),
             nb::arg("pin_vertex"), nb::arg("pin_body"), nb::arg("pin_point"), nb::arg("at_tick") = 0,
             "Queue a soft body; return its uid and the batch's sequence number.")
        .def(
            "set_soft_pins",
            [](World &w, Uid uid, const Words &vertex, const Uids &body, const Vec3s &point, uint64_t at_tick) {
                return w.core().submit(Command(SetSoftPins{uid, soft_pins(vertex, body, point, 0)}), at_tick);
            },
            nb::arg("uid"), nb::arg("vertex"), nb::arg("body"), nb::arg("point"), nb::arg("at_tick") = 0)
        .def(
            "set_wind",
            [](World &w, const Uids &ids, const Vec3s &force, uint64_t at_tick) {
                SetWind set{uids(ids), copy<double>(force, ids.shape(0), "force")};
                check_floats(set.force, "force");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("force"), nb::arg("at_tick") = 0)
        .def("add_vehicle", &add_vehicle, nb::arg("chassis"), nb::arg("kind"), nb::arg("cast"), nb::arg("params"),
             nb::arg("wheels"), nb::arg("torque_curve"), nb::arg("longitudinal"), nb::arg("lateral"), nb::arg("gears"),
             nb::arg("reverse_gears"), nb::arg("differentials"), nb::arg("anti_roll_bars"), nb::arg("tracks"),
             nb::arg("at_tick") = 0, "Queue a vehicle on a dynamic chassis; return its id and the batch's sequence number.")
        .def(
            "remove_vehicles",
            [](World &w, const Uids &ids, uint64_t at_tick) {
                return w.core().submit(Command(RemoveVehicles{uids(ids)}), at_tick);
            },
            nb::arg("ids"), nb::arg("at_tick") = 0)
        .def(
            "vehicle_inputs",
            [](World &w, const Uids &ids, const In<float, nb::shape<-1, 4>> &values, uint64_t at_tick) {
                VehicleInputs inputs{uids(ids), copy<float>(values, ids.shape(0), "values")};
                check_finite(inputs.values, "vehicle inputs");
                return w.core().submit(Command(std::move(inputs)), at_tick);
            },
            nb::arg("ids"), nb::arg("values"), nb::arg("at_tick") = 0,
            "Queue vehicle inputs (throttle, steer, brake, handbrake), held from the next tick.")
        .def(
            "set_vehicle_surfaces",
            [](World &w, const Uids &ids, const Ints &wheel, const Floats &grip, const Floats &rolling,
               uint64_t at_tick) {
                const size_t n = ids.shape(0);
                SetVehicleSurfaces set{uids(ids), copy<int32_t>(wheel, n, "wheel"), copy<float>(grip, n, "grip"),
                                       copy<float>(rolling, n, "rolling")};
                for (size_t i = 0; i < n; ++i)
                    if (!(std::isnan(set.grip[i]) || (std::isfinite(set.grip[i]) && set.grip[i] >= 0.0f)) ||
                        !std::isfinite(set.rolling[i]))
                        throw std::invalid_argument("a surface's grip is NaN or not negative, its rolling finite");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("ids"), nb::arg("wheel"), nb::arg("grip"), nb::arg("rolling"), nb::arg("at_tick") = 0)
        .def(
            "set_surface_table",
            [](World &w, const Floats &grip, const Floats &rolling, uint64_t at_tick) {
                const size_t n = grip.shape(0);
                SetSurfaceTable set{copy<float>(grip, n, "grip"), copy<float>(rolling, n, "rolling")};
                check_finite(set.grip, "grip");
                check_finite(set.rolling, "rolling");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("grip"), nb::arg("rolling"), nb::arg("at_tick") = 0)
        .def("set_material_table", &set_material_table, nb::arg("static_friction"), nb::arg("dynamic_friction"),
             nb::arg("restitution"), nb::arg("friction_combine"), nb::arg("restitution_combine"), nb::arg("pairs"),
             nb::arg("static_speed"), nb::arg("at_tick") = 0)
        .def(
            "set_body_materials",
            [](World &w, const Uids &ids, const Shorts &material, uint64_t at_tick) {
                SetBodyMaterials set{uids(ids), copy<uint16_t>(material, ids.shape(0), "material")};
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("material"), nb::arg("at_tick") = 0)
        .def(
            "set_filter_table",
            [](World &w, const In<uint8_t, nb::shape<-1, -1>> &table, uint64_t at_tick) {
                const size_t f = table.shape(0);
                if (table.shape(1) != f || f > 65535)
                    throw std::invalid_argument("the filter table is (F, F) with F under 65536");
                SetFilterTable set{uint32_t(f), std::vector<uint8_t>(table.data(), table.data() + table.size())};
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("table"), nb::arg("at_tick") = 0)
        .def(
            "set_body_filters",
            [](World &w, const Uids &ids, const Shorts &filter, uint64_t at_tick) {
                SetBodyFilters set{uids(ids), copy<uint16_t>(filter, ids.shape(0), "filter")};
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("filter"), nb::arg("at_tick") = 0)
        .def("set_gravity_zones", &set_gravity_zones, nb::arg("shape"), nb::arg("field"), nb::arg("position"),
             nb::arg("rotation"), nb::arg("half"), nb::arg("shape_id"), nb::arg("gravity"), nb::arg("direction"),
             nb::arg("center"), nb::arg("priority"), nb::arg("replace"), nb::arg("stop"), nb::arg("at_tick") = 0)
        .def(
            "set_motion",
            [](World &w, const Uids &ids, const Floats &mass, const Floats &linear, const Floats &angular,
               const Floats &gravity, const Floats &inertia, uint64_t at_tick) {
                const size_t n = ids.shape(0);
                SetMotion set{uids(ids),
                              copy<float>(mass, n, "mass"),
                              copy<float>(linear, n, "linear_damping"),
                              copy<float>(angular, n, "angular_damping"),
                              copy<float>(gravity, n, "gravity_factor"),
                              copy<float>(inertia, 9 * n, "inertia")};
                for (size_t i = 0; i < n; ++i) {
                    if (!std::isnan(set.mass[i]) && !(std::isfinite(set.mass[i]) && set.mass[i] > 0.0f))
                        throw std::invalid_argument("mass is a finite number above 0");
                    for (float k : {set.linear_damping[i], set.angular_damping[i]})
                        if (!std::isnan(k) && !(std::isfinite(k) && k >= 0.0f))
                            throw std::invalid_argument("damping is finite and not negative");
                    if (!std::isnan(set.gravity_factor[i]) && !std::isfinite(set.gravity_factor[i]))
                        throw std::invalid_argument("gravity_factor is finite");
                    const float *tensor = &set.inertia[9 * i];
                    if (!std::isnan(tensor[0])) {
                        for (int k = 0; k < 9; ++k)
                            if (!std::isfinite(tensor[k]))
                                throw std::invalid_argument("inertia is finite");
                        for (int k = 0; k < 3; ++k)
                            if (!(tensor[4 * k] > 0.0f))
                                throw std::invalid_argument("inertia has its diagonal above 0");
                    }
                }
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("mass"), nb::arg("linear_damping"), nb::arg("angular_damping"),
            nb::arg("gravity_factor"), nb::arg("inertia"), nb::arg("at_tick") = 0)
        .def(
            "set_flags",
            [](World &w, const Uids &ids, uint32_t bits, bool on, uint64_t at_tick) {
                if ((bits & ~(flag::kReport | flag::kNoSleep | flag::kFilterContacts)) != 0)
                    throw std::invalid_argument("only REPORT, NO_SLEEP and FILTER_CONTACTS change after a body is added");
                SetFlags set{uids(ids), bits, on};
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("bits"), nb::arg("on"), nb::arg("at_tick") = 0)
        .def(
            "bodies_state",
            [](World &w, const Uids &ids) {
                const size_t n = ids.shape(0);
                std::vector<double> position(n * 3, 0.0);
                std::vector<float> orientation(n * 4, 0.0f), linear(n * 3, 0.0f), angular(n * 3, 0.0f);
                std::vector<uint8_t> awake(n, 0), found(n, 0);
                {
                    nb::gil_scoped_release release;
                    w.core().bodies_state(ids.data(), n, position.data(), orientation.data(), linear.data(),
                                          angular.data(), awake.data(), found.data());
                }
                return std::make_tuple(owned<double, nb::shape<-1, 3>>(std::move(position), {n, 3}),
                                       owned<float, nb::shape<-1, 4>>(std::move(orientation), {n, 4}),
                                       owned<float, nb::shape<-1, 3>>(std::move(linear), {n, 3}),
                                       owned<float, nb::shape<-1, 3>>(std::move(angular), {n, 3}),
                                       owned<uint8_t, nb::shape<-1>>(std::move(awake), {n}),
                                       owned<uint8_t, nb::shape<-1>>(std::move(found), {n}));
            },
            nb::arg("uids"),
            "Each body's position, orientation, velocities, awake and found, as Jolt holds it now.")
        .def(
            "triangle_of",
            [](World &w, uint32_t uid, uint32_t sub_shape) {
                nb::gil_scoped_release release;
                return w.core().triangle_of(uid, sub_shape);
            },
            nb::arg("uid"), nb::arg("sub_shape"),
            "The number of the mesh triangle sub_shape names on body uid, or -1 for a part that is no mesh triangle.")
        .def(
            "flush",
            [](World &w) {
                nb::gil_scoped_release release;
                w.core().flush();
            })
        .def(
            "save_state",
            [](World &w) {
                std::string data;
                {
                    nb::gil_scoped_release release;
                    data = w.core().save_state();
                }
                return nb::bytes(data.data(), data.size());
            })
        .def(
            "restore_state",
            [](World &w, nb::bytes data, uint64_t at_tick) {
                auto copy = std::make_shared<const std::string>(data.c_str(), data.size());
                const std::string problem = state_problem(*copy);
                if (!problem.empty())
                    throw std::invalid_argument(problem);
                return w.core().submit(Command(RestoreState{std::move(copy)}), at_tick);
            },
            nb::arg("data"), nb::arg("at_tick") = 0)
        .def(
            "set_drag",
            [](World &w, const Uids &ids, const Floats &drag, uint64_t at_tick) {
                SetDrag set{uids(ids), copy<float>(drag, ids.shape(0), "drag")};
                for (float k : set.drag)
                    if (!(std::isfinite(k) && k >= 0.0f))
                        throw std::invalid_argument("drag is finite and not negative");
                return w.core().submit(Command(std::move(set)), at_tick);
            },
            nb::arg("uids"), nb::arg("drag"), nb::arg("at_tick") = 0)
        .def("set_tracks", &set_tracks, nb::arg("uids"), nb::arg("times"), nb::arg("positions"),
             nb::arg("rotations"), nb::arg("loop"), nb::arg("at_tick") = 0)
        .def("move_kinematic", &move_kinematic, nb::arg("uids"), nb::arg("positions"), nb::arg("rotations"),
             nb::arg("at_tick") = 0,
             "Move kinematic bodies to poses by the end of a tick: this one from a world callback, else the next.")
        .def(
            "set_drivers",
            [](World &w, nb::object fn) {
                CallbackSet &c = w.core().callbacks();
                c.set(c.drivers, c.has_drivers, std::move(fn));
                w.core().poke();
            },
            nb::arg("fn").none())
        .def(
            "set_contact_filter",
            [](World &w, nb::object fn, bool either, bool recheck) {
                w.core().set_filter_either(either);
                w.core().set_filter_recheck(recheck);
                CallbackSet &c = w.core().callbacks();
                c.set(c.contact_filter, c.has_contact_filter, std::move(fn));
            },
            nb::arg("fn").none(), nb::arg("either") = false, nb::arg("recheck") = false)
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
        .def_prop_ro(
            "last_rejection",
            [](const World &w) -> std::optional<std::string> {
                std::string message = w.core().last_rejection();
                if (message.empty())
                    return std::nullopt;
                return message;
            },
            "Why the world could not apply the last batch whose apply raised; None when none has.")
        .def_prop_ro("tick", [](const World &w) { return w.core().tick(); })
        .def_prop_ro("lockstep", [](const World &w) { return w.core().lockstep(); })
        .def_prop_ro("forked", [](const World &w) { return w.core().forked(); })
        .def_prop_ro("submitted_seq", [](const World &w) { return w.core().submitted_seq(); })
        .def_prop_ro("reserved_bodies", [](const World &w) { return w.core().reserved_bodies(); })
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
