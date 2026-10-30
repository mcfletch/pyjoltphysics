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

#include <cmath>
#include <cstring>
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

std::pair<nb::ndarray<nb::numpy, uint32_t, nb::shape<-1>>, uint64_t>
add_bodies(World &world, const Vec3s &position, const Quats &orientation, const Bytes &shape_kind,
           const Params &shape_params, const Bytes &motion, const Doubles &mass,
           const Doubles &density, const Floats &friction, const Floats &restitution,
           const Floats &linear_damping, const Floats &angular_damping,
           const Floats &gravity_factor, const Vec3s &linear_velocity,
           const Vec3s &angular_velocity, const Words &group, const Words &mask,
           const Words &flags, uint64_t at_tick) {
    const size_t n = position.shape(0);
    AddBodies add;
    add.position = copy<double>(position, n, "position");
    add.orientation = copy<double>(orientation, n, "orientation");
    add.shape_kind = copy<uint8_t>(shape_kind, n, "shape_kind");
    add.shape_params = copy<double>(shape_params, n, "shape_params");
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
    check_finite(add.linear_velocity, "linear_velocity");
    check_finite(add.angular_velocity, "angular_velocity");
    for (size_t i = 0; i < n; ++i) {
        const std::string row = " (row " + std::to_string(i) + ")";
        const ShapeParams params{add.shape_params[4 * i], add.shape_params[4 * i + 1],
                                 add.shape_params[4 * i + 2], add.shape_params[4 * i + 3]};
        const std::string problem = shape_problem(add.shape_kind[i], params);
        if (!problem.empty())
            throw std::invalid_argument(problem + row);
        if (!valid_motion(add.motion[i]))
            throw std::invalid_argument("unknown motion type " + std::to_string(add.motion[i]) + row);
        const auto kind = static_cast<ShapeKind>(add.shape_kind[i]);
        if (kind == ShapeKind::Plane && add.motion[i] != uint8_t(Motion::Static))
            throw std::invalid_argument("a plane is static" + row);
        if (add.flags[i] & ~flag::kAll)
            throw std::invalid_argument("unknown body flags" + row);
        const bool massive = add.motion[i] != uint8_t(Motion::Static) || (add.flags[i] & flag::kMovable);
        if (massive && !(std::isfinite(add.mass[i]) && add.mass[i] > 0.0) &&
            !(std::isfinite(add.density[i]) && add.density[i] > 0.0))
            throw std::invalid_argument("a moving body needs a positive mass or density" + row);
        if (kind == ShapeKind::Plane && (add.flags[i] & flag::kMovable))
            throw std::invalid_argument("a plane is static" + row);
        for (float value : {add.friction[i], add.restitution[i], add.linear_damping[i],
                            add.angular_damping[i]})
            if (!(std::isfinite(value) && value >= 0.0f))
                throw std::invalid_argument(
                    "friction, restitution and damping are finite and not negative" + row);
        if (!std::isfinite(add.gravity_factor[i]))
            throw std::invalid_argument("gravity_factor is not finite" + row);
    }
    Uid first = kNoBody;
    const uint64_t seq = world.core().submit(Command(std::move(add)), at_tick, &first);
    std::vector<uint32_t> made(n);
    for (size_t i = 0; i < n; ++i)
        made[i] = first + uint32_t(i);
    return {owned<uint32_t, nb::shape<-1>>(std::move(made), {n}), seq};
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
             nb::arg("shape_kind"), nb::arg("shape_params"), nb::arg("motion"), nb::arg("mass"),
             nb::arg("density"), nb::arg("friction"), nb::arg("restitution"),
             nb::arg("linear_damping"), nb::arg("angular_damping"), nb::arg("gravity_factor"),
             nb::arg("linear_velocity"), nb::arg("angular_velocity"), nb::arg("group"),
             nb::arg("mask"), nb::arg("flags"), nb::arg("at_tick") = 0,
             "Queue bodies to add; return their uids and the batch's sequence number.")
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
