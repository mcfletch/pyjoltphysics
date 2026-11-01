#include "world.h"

#include "seams.h"

#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/RegisterTypes.h>

#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace pj {

namespace {

double seconds(std::chrono::steady_clock::duration span) {
    return std::chrono::duration<double>(span).count();
}

}  // namespace

WorldCore::WorldCore(const WorldSettings &settings)
    : settings_(settings), dt_(1.0 / settings.tick_hz), ring_(settings.event_capacity) {
    register_jolt();
    if (!JPH::VerifyJoltVersionID())
        throw std::runtime_error("Jolt was built with options that differ from the binding's");
    temp_ = std::make_unique<JPH::TempAllocatorImplWithMallocFallback>(
        JPH::uint(size_t(settings.temp_allocator_mb) << 20));
    if (settings.threads <= 1)
        jobs_ = std::make_unique<JPH::JobSystemSingleThreaded>(JPH::cMaxPhysicsJobs);
    else
        // The world thread runs jobs while it waits for a step, so it is one
        // of the `threads`.
        jobs_ = std::make_unique<JPH::JobSystemThreadPool>(
            JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, settings.threads - 1);
    system_.Init(settings.max_bodies, 0, settings.max_body_pairs, settings.max_contact_constraints,
                 layers_, layers_, layers_);
    system_.SetGravity(JPH::Vec3(float(settings.gravity[0]), float(settings.gravity[1]),
                                 float(settings.gravity[2])));
    JPH::PhysicsSettings physics;
    physics.mAllowSleeping = settings.allow_sleeping;
    physics.mTimeBeforeSleep = settings.time_before_sleep;
    physics.mPointVelocitySleepThreshold = settings.sleep_velocity;
    physics.mMinVelocityForRestitution = settings.min_velocity_for_restitution;
    physics.mDeterministicSimulation = true;
    physics.mPenetrationSlop = settings.penetration_slop;
    system_.SetPhysicsSettings(physics);
    collector_.min_velocity_for_restitution = settings.min_velocity_for_restitution;
    materials_.static_speed = settings.static_friction_speed;
    collector_.materials = &materials_;
    collector_.filter = this;
    system_.AddStepListener(&fields_);
    system_.SetContactListener(&collector_);
    system_.SetSimCollideBodyVsBody([this](const JPH::Body &moving, const JPH::Body &other, JPH::Mat44Arg first,
                                           JPH::Mat44Arg second, JPH::CollideShapeSettings &settings,
                                           JPH::CollideShapeCollector &collector, const JPH::ShapeFilter &filter) {
        collide_across_seams(system_, moving, other, first, second, settings, collector, filter);
    });
    system_.SetBodyActivationListener(&collector_);
    row_of_index_.assign(settings.max_bodies, kNoRow);
    characters_ = std::make_unique<CharacterSystem>(system_, layers_, settings.threads);
    vehicles_ = std::make_unique<VehicleSystem>(system_, layers_, dt_);
    ragdolls_ = std::make_unique<RagdollSystem>(system_);
    soft_ = std::make_unique<SoftBodySystem>(system_);
}

WorldCore::~WorldCore() {
    if (thread_.joinable())
        thread_.detach();
    vehicles_->remove_all();
    // The ragdolls destroy their own bodies.
    std::unordered_set<Uid> bones;
    for (RagdollId id : ragdolls_->ids()) {
        std::vector<JPH::BodyID> ids;
        std::vector<Uid> uids;
        std::vector<std::pair<int, int>> pairs;
        ragdolls_->remove(id, ids, uids, pairs);
        bones.insert(uids.begin(), uids.end());
    }
    for (Joint &joint : joints_)
        if (joint.constraint != nullptr && joint.state != JointState::Broken)
            system_.RemoveConstraint(joint.constraint);
    joints_.clear();
    {
        JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
        std::vector<JPH::BodyID> ids = characters_->remove_all();
        ids.reserve(ids.size() + rows_.size());
        for (const Row &row : rows_)
            if (bones.count(row.uid) == 0)
                ids.push_back(row.id);
        if (!ids.empty()) {
            bodies.RemoveBodies(ids.data(), int(ids.size()));
            bodies.DestroyBodies(ids.data(), int(ids.size()));
        }
    }
}

// -- commands ------------------------------------------------------------------

namespace {

// The collision keys a command asks layers for; consecutive equal keys appear once.
std::vector<LayerKey> layer_keys(const Command &command) {
    std::vector<LayerKey> keys;
    auto put = [&](const LayerKey &key) {
        if (keys.empty() || keys.back().tie() != key.tie())
            keys.push_back(key);
    };
    auto bodies = [&](const AddBodies &add) {
        for (size_t i = 0; i < add.shapes.size(); ++i)
            put(LayerKey{add.group[i], add.mask[i], add.motion[i] != uint8_t(Motion::Static),
                         (add.flags[i] & flag::kSensor) ? LayerRole::Sensor : LayerRole::Body, add.filter[i]});
    };
    if (auto *add = std::get_if<AddBodies>(&command))
        bodies(*add);
    else if (auto *swap = std::get_if<SwapBodies>(&command))
        bodies(swap->add);
    else if (auto *doll = std::get_if<AddRagdoll>(&command)) {
        const double *p = doll->desc.params.data();
        put(LayerKey{uint32_t(p[kRagGroup]), uint32_t(p[kRagMask]), true, LayerRole::Body, uint16_t(p[kRagFilter])});
    } else if (auto *soft = std::get_if<AddSoftBody>(&command)) {
        const double *p = soft->desc.params.data();
        put(LayerKey{uint32_t(p[kSoftGroup]), uint32_t(p[kSoftMask]), true, LayerRole::Body,
                     uint16_t(p[kSoftFilter])});
    } else if (auto *character = std::get_if<AddCharacter>(&command)) {
        if (character->proxy)
            put(LayerKey{character->group, character->mask, true, LayerRole::Proxy, 0});
    }
    return keys;
}

}  // namespace

uint64_t WorldCore::submit(Command &&command, uint64_t at_tick, Uid *first_uid, JointId *first_joint,
                          VehicleId *vehicle, RagdollId *ragdoll) {
    check();
    const uint32_t uid_count = uids_wanted(command);
    if (uid_count)
        layers_.check_room(layer_keys(command));
    if (uid_count) {
        const int64_t after = reserved_bodies_.fetch_add(uid_count) + uid_count;
        if (after > int64_t(settings_.max_bodies)) {
            reserved_bodies_.fetch_sub(uid_count);
            throw std::length_error("the batch would take the world past max_bodies (" +
                                    std::to_string(settings_.max_bodies) + ")");
        }
    }
    Batch batch;
    batch.at_tick = at_tick;
    batch.command = std::move(command);
    const uint64_t seq = queue_.push(std::move(batch), uid_count, first_uid, first_joint, vehicle, ragdoll);
    std::lock_guard<std::mutex> guard(control_);
    wake_.notify_all();
    return seq;
}

void WorldCore::note_applied(uint64_t seq) {
    if (seq != applied_seq_ + 1) {
        applied_ahead_.insert(seq);
        return;
    }
    applied_seq_ = seq;
    while (!applied_ahead_.empty() && *applied_ahead_.begin() == applied_seq_ + 1) {
        applied_seq_ = *applied_ahead_.begin();
        applied_ahead_.erase(applied_ahead_.begin());
    }
}

std::vector<uint64_t> WorldCore::drain_applied() {
    std::vector<uint64_t> out;
    std::lock_guard<std::mutex> guard(applied_lock_);
    out.swap(applied_log_);
    return out;
}

// -- the thread ------------------------------------------------------------------

void WorldCore::start() {
    check();
    if (failed_.load())
        throw CallbackFailed(failure());
    if (in_callback()) {
        // From a callback the world free-runs from the next tick. A stop
        // already asked for stands; nothing here waits for the thread.
        std::lock_guard<std::mutex> guard(control_);
        if (mode_ == Mode::Lockstep)
            cut_lockstep();
        mode_ = Mode::Free;
        wake_.notify_all();
        return;
    }
    std::lock_guard<std::mutex> joining(join_lock_);
    {
        std::unique_lock<std::mutex> guard(control_);
        if (running_ && !stopping_) {
            if (mode_ == Mode::Lockstep)
                cut_lockstep();
            mode_ = Mode::Free;
            wake_.notify_all();
            return;
        }
    }
    // A thread told to stop from a callback may still be finishing its tick.
    if (thread_.joinable())
        thread_.join();
    std::lock_guard<std::mutex> guard(control_);
    mode_ = Mode::Free;
    stopping_ = false;
    running_ = true;
    thread_ = std::thread([self = shared_from_this()] { self->loop(); });
}

void WorldCore::cut_lockstep() {
    if (wanted_ == 0)
        return;
    wanted_ = 0;
    ++cuts_;
    done_.notify_all();
}

void WorldCore::request_stop() {
    {
        std::lock_guard<std::mutex> guard(control_);
        stopping_ = true;
        wake_.notify_all();
        done_.notify_all();
    }
    std::lock_guard<std::mutex> guard(publish_lock_);
    publish_cv_.notify_all();
}

// The flag is set and the thread joined under the join lock, so no run_ticks
// or start can clear the flag and start a thread in between that this call
// would then join for ever.
void WorldCore::stop() {
    if (forked_.load())
        return;
    if (in_callback()) {
        request_stop();
        return;
    }
    std::lock_guard<std::mutex> joining(join_lock_);
    request_stop();
    if (thread_.joinable())
        thread_.join();
}

// From inside a callback, the thread that runs it cannot be joined, and a
// thread joining the world thread (stop_all_worlds at exit) may hold the join
// lock: the stop is asked for and the thread, which holds its own reference
// to the world, ends after the tick and detaches itself in ~WorldCore.
void WorldCore::shutdown() {
    if (in_callback()) {
        request_stop();
        return;
    }
    stop();
}

void WorldCore::run_ticks(uint64_t count) {
    check();
    if (in_callback_job())
        throw std::runtime_error("run_ticks cannot be called from the contact filter, which runs inside the step");
    if (on_world_thread())
        throw std::runtime_error("run_ticks cannot be called from the world thread");
    if (failed_.load())
        throw CallbackFailed(failure());
    if (count == 0 || run_quiet(count))
        return;
    uint64_t cuts = 0;
    {
        std::lock_guard<std::mutex> joining(join_lock_);
        std::unique_lock<std::mutex> guard(control_);
        if (!running_ || stopping_) {
            guard.unlock();
            if (thread_.joinable())
                thread_.join();
            guard.lock();
            stopping_ = false;
            running_ = true;
            thread_ = std::thread([self = shared_from_this()] { self->loop(); });
        }
        // Set after the join: a callback of the thread being joined may still
        // call start() and switch the mode.
        mode_ = Mode::Lockstep;
        wanted_ += count;
        cuts = cuts_;
        wake_.notify_all();
    }
    std::unique_lock<std::mutex> guard(control_);
    done_.wait(guard, [&] { return wanted_ == 0 || stopping_; });
    if (failed_.load())
        throw CallbackFailed(failure());
    if (cuts_ != cuts)
        throw WorldStopped("start() switched the world to free-running before the ticks ran");
    if (wanted_ != 0) {
        wanted_ = 0;
        throw WorldStopped();
    }
}

bool WorldCore::settled() const {
    return !callbacks_.any_step() && characters_->empty() && vehicles_->empty() && tracks_.empty() && driven_.empty() &&
           fields_.torques.empty() && !ragdolls_->driving() && !soft_->driving() &&
           system_.GetNumActiveBodies(JPH::EBodyType::RigidBody) == 0 &&
           system_.GetNumActiveBodies(JPH::EBodyType::SoftBody) == 0;
}

// A settled tick that applies no batch, runs no callable and makes no contact
// report leaves Jolt's contact caches empty; after kQuietAfter of them in a row
// a step changes nothing, so the ticks are counted rather than stepped. The
// join lock keeps out another run_ticks, and lockstep with nothing wanted
// keeps the world thread out of a tick; the world lock keeps out flush and a
// broad phase optimisation.
bool WorldCore::run_quiet(uint64_t count) {
    std::lock_guard<std::mutex> joining(join_lock_);
    {
        std::lock_guard<std::mutex> guard(control_);
        if (mode_ != Mode::Lockstep || wanted_ != 0)
            return false;
    }
    std::unique_lock<WorldLock> exclusive(world_lock_);
    const uint64_t index = tick_.load();
    if (quiet_ < kQuietAfter || !queue_.empty() || callbacks_.earliest_at_tick.load() < index + count ||
        !settled())
        return false;
    const uint64_t tick = index + count;
    time_ = double(tick) * dt_;
    publish(tick);
    std::lock_guard<std::mutex> guard(stats_lock_);
    sums_.quiet_ticks += count;
    return true;
}

bool WorldCore::wait_tick(uint64_t tick, double timeout) {
    check();
    if (in_callback_job())
        throw std::runtime_error("wait_tick cannot be called from the contact filter, which runs inside the step");
    if (on_world_thread())
        throw std::runtime_error("wait_tick cannot be called from the world thread");
    std::unique_lock<std::mutex> guard(publish_lock_);
    auto reached = [&] {
        return (published_ && published_->tick >= tick) || failed_.load() || [&] {
            std::lock_guard<std::mutex> control(control_);
            return stopping_ || !running_;
        }();
    };
    if (timeout < 0)
        publish_cv_.wait(guard, reached);
    else
        publish_cv_.wait_for(guard, std::chrono::duration<double>(timeout), reached);
    if (failed_.load())
        throw CallbackFailed(failure());
    return published_ && published_->tick >= tick;
}

bool WorldCore::lockstep() const {
    std::lock_guard<std::mutex> guard(control_);
    return mode_ == Mode::Lockstep;
}

// A future at_tick does not keep a free-running world ticking: an idle world's
// tick count stands still, so the callable runs when other work or run_ticks
// brings the world to its tick.
bool WorldCore::busy() const {
    return !queue_.empty() || callbacks_.any_step() || !characters_->empty() ||
           callbacks_.earliest_at_tick.load() <= tick_.load() ||
           ragdolls_->driving() || soft_->driving() || system_.GetNumActiveBodies(JPH::EBodyType::RigidBody) > 0 ||
           system_.GetNumActiveBodies(JPH::EBodyType::SoftBody) > 0;
}

void WorldCore::optimize_if_due() {
    if (statics_since_optimize_ < kOptimizeAfterStatics)
        return;
    statics_since_optimize_ = 0;
    std::unique_lock<WorldLock> exclusive(world_lock_);
    system_.OptimizeBroadPhase();
    ++optimizations_;
}

void WorldCore::loop() {
    thread_id_ = std::this_thread::get_id();
    const auto step = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(dt_));
    const auto catch_up = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(settings_.max_catch_up));
    auto next = Clock::now();
    Mode last = Mode::Lockstep;
    for (;;) {
        Mode mode;
        {
            std::unique_lock<std::mutex> guard(control_);
            bool go = false;
            while (!go) {
                if (stopping_)
                    break;
                if (statics_since_optimize_ >= kOptimizeAfterStatics &&
                    (mode_ == Mode::Lockstep ? wanted_ == 0 : Clock::now() < next || !busy())) {
                    guard.unlock();
                    optimize_if_due();
                    guard.lock();
                    continue;
                }
                if (mode_ == Mode::Lockstep) {
                    if (wanted_ > 0) {
                        go = true;
                        continue;
                    }
                    idle_ = true;
                    wake_.wait(guard);
                    idle_ = false;
                    continue;
                }
                if (last != Mode::Free) {
                    next = Clock::now();
                    last = Mode::Free;
                }
                if (!busy()) {
                    idle_ = true;
                    wake_.wait(guard);
                    idle_ = false;
                    next = Clock::now();
                    continue;
                }
                if (Clock::now() >= next) {
                    go = true;
                    continue;
                }
                wake_.wait_until(guard, next);
            }
            if (!go)
                break;
            mode = mode_;
            last = mode;
        }
        // Nothing thrown may leave the thread, where std::terminate would end
        // the process: the world stops with the message instead.
        bool ok = false;
        try {
            ok = tick_once();
        } catch (const std::exception &error) {
            fail_native(std::string("the world thread failed: ") + error.what());
        } catch (...) {
            fail_native("the world thread failed with an exception that is not a std::exception");
        }
        std::unique_lock<std::mutex> guard(control_);
        if (!ok) {
            stopping_ = true;
            done_.notify_all();
            break;
        }
        if (mode == Mode::Lockstep) {
            if (wanted_ > 0 && --wanted_ == 0)
                done_.notify_all();
        } else {
            next += step;
            const auto now = Clock::now();
            if (now - next > catch_up) {
                dropped_ticks_ += uint64_t((now - next) / step);
                next = now;
            }
        }
    }
    release_job_python();
    python_.release();
    {
        std::lock_guard<std::mutex> guard(control_);
        running_ = false;
        done_.notify_all();
    }
    {
        std::lock_guard<std::mutex> guard(publish_lock_);
        publish_cv_.notify_all();
    }
    thread_id_ = std::thread::id();
}

// -- one tick ----------------------------------------------------------------------

// The thread detaches whatever the body throws: a C++ exception (a failed
// cast, an allocation) stops the world as a Python one does.
template <typename Body> bool WorldCore::call_python(Body &&body) {
    PythonCall call(*this);
    if (!call.admitted() || !python_.attach())
        return true;
    bool ok = true;
    try {
        body();
    } catch (nb::python_error &error) {
        fail(error);
        ok = false;
    } catch (const std::exception &error) {
        fail_native(std::string("the world thread failed in a callback: ") + error.what());
        ok = false;
    }
    python_.detach();
    return ok;
}

void WorldCore::fail(nb::python_error &error) {
    nb::object exception = nb::borrow(error.value());
    {
        // Once the World is gone (clear_python) nothing may be kept: the last
        // reference would be dropped by ~WorldCore with no thread state.
        std::lock_guard<std::mutex> guard(error_lock_);
        if (!python_cleared_)
            error_ = exception;
    }
    failed_ = true;
    try {
        callback_logger().attr("error")("a world callback raised; the world has stopped",
                                        nb::arg("exc_info") = exception);
    } catch (nb::python_error &logging_error) {
        logging_error.discard_as_unraisable("logging a pyjoltphysics callback error");
    }
}

void WorldCore::fail_native(const std::string &message) {
    {
        std::lock_guard<std::mutex> guard(error_lock_);
        if (native_error_.empty())
            native_error_ = message;
    }
    failed_ = true;
}

std::string WorldCore::failure() const {
    std::lock_guard<std::mutex> guard(error_lock_);
    return native_error_.empty() ? CallbackFailed().what() : native_error_ + "; the world has stopped";
}

std::string WorldCore::last_rejection() const {
    std::lock_guard<std::mutex> guard(error_lock_);
    return last_rejection_;
}

nb::object WorldCore::error() const {
    std::lock_guard<std::mutex> guard(error_lock_);
    if (error_.is_valid())
        return error_;
    if (!native_error_.empty())
        return nb::handle(PyExc_RuntimeError)(nb::str(native_error_.c_str()));
    return nb::none();
}

void WorldCore::poke() {
    std::lock_guard<std::mutex> guard(control_);
    wake_.notify_all();
}

int WorldCore::traverse(visitproc visit, void *arg) {
    if (int result = callbacks_.traverse(visit, arg))
        return result;
    if (error_.is_valid())
        Py_VISIT(error_.ptr());
    return 0;
}

WorldCore::PythonCall::PythonCall(WorldCore &world) : world_(world) {
    std::lock_guard<std::mutex> guard(world_.calls_lock_);
    admitted_ = !world_.retired_;
    if (admitted_)
        ++world_.calls_;
}

WorldCore::PythonCall::~PythonCall() {
    if (!admitted_)
        return;
    std::lock_guard<std::mutex> guard(world_.calls_lock_);
    if (--world_.calls_ == 0)
        world_.calls_done_.notify_all();
}

// The cyclic collector of a free-threaded build finds a World and its
// callables unreachable with every thread stopped, then lets the threads run
// before it finalizes and clears them. The world thread reads the callables
// through the core, which the collector cannot see, so it would call a
// function the collector had cleared. The World's finalizer runs before any
// of them is cleared: from then on no call starts, and the calls already
// running finish first.
void WorldCore::retire_python() {
    // A call on this thread (the collection runs inside it) is not waited for.
    const int own = in_callback() ? 1 : 0;
    {
        std::lock_guard<std::mutex> guard(calls_lock_);
        retired_ = true;
    }
    {
        nb::gil_scoped_release release;
        std::unique_lock<std::mutex> guard(calls_lock_);
        calls_done_.wait(guard, [&] { return calls_ <= own; });
    }
    clear_python();
}

void WorldCore::clear_python() {
    callbacks_.clear();
    nb::object old;
    std::lock_guard<std::mutex> guard(error_lock_);
    python_cleared_ = true;
    old = std::move(error_);
}

bool WorldCore::run_python_before(uint64_t index) {
    const bool at = callbacks_.has_at_tick.load();
    const bool before = callbacks_.has_before.load();
    const bool drive = callbacks_.has_drivers.load();
    if (!at && !before && !drive)
        return true;
    const double time = time_;
    return call_python([&] {
        if (at) {
            std::vector<nb::object> due = callbacks_.take_due(index);
            for (nb::object &fn : due)
                fn(index, time);
        }
        if (before) {
            nb::object fn;
            {
                std::lock_guard<std::mutex> guard(callbacks_.lock);
                fn = callbacks_.before_step;
            }
            if (fn.is_valid())
                fn(index, time);
        }
        if (drive) {
            nb::object fn;
            {
                std::lock_guard<std::mutex> guard(callbacks_.lock);
                fn = callbacks_.drivers;
            }
            if (fn.is_valid())
                fn(index, time);
        }
    });
}

bool WorldCore::run_python_after(uint64_t tick) {
    const bool after = callbacks_.has_after.load();
    const bool events = callbacks_.has_events.load() && !tick_events_.empty();
    if (!after && !events)
        return true;
    const double time = time_;
    return call_python([&] {
        if (after) {
            nb::object fn;
            {
                std::lock_guard<std::mutex> guard(callbacks_.lock);
                fn = callbacks_.after_step;
            }
            if (fn.is_valid())
                fn(tick, time);
        }
        if (events) {
            nb::object fn;
            {
                std::lock_guard<std::mutex> guard(callbacks_.lock);
                fn = callbacks_.on_events;
            }
            if (fn.is_valid()) {
                const size_t bytes = tick_events_.size() * sizeof(Event);
                auto *copy = new std::vector<uint8_t>(bytes);
                std::memcpy(copy->data(), tick_events_.data(), bytes);
                nb::capsule owner(copy, [](void *p) noexcept {
                    delete static_cast<std::vector<uint8_t> *>(p);
                });
                fn(nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>>(copy->data(), {bytes}, owner),
                   tick, time);
            }
        }
    });
}

bool WorldCore::apply_due(uint64_t index, std::vector<Batch> &due) {
    due.clear();
    queue_.take_due(index, due);
    for (Batch &batch : due) {
        // A batch that throws (std::bad_alloc) is rejected and the world goes on.
        try {
            apply(batch, index);
        } catch (const std::exception &error) {
            ++rejected_;
            std::lock_guard<std::mutex> guard(error_lock_);
            last_rejection_ = error.what();
        }
        note_applied(batch.seq);
    }
    characters_->flush_events(tick_events_);
    if (due.empty())
        return false;
    std::lock_guard<std::mutex> guard(applied_lock_);
    for (const Batch &batch : due) {
        applied_log_.push_back(batch.seq);
        applied_log_.push_back(index);
    }
    return true;
}

void WorldCore::flush() {
    check();
    if (in_callback_job())
        throw std::runtime_error("flush cannot be called from the contact filter, which runs inside the step");
    if (on_world_thread())
        return;
    std::unique_lock<WorldLock> exclusive(world_lock_);
    const uint64_t index = tick_.load();
    std::vector<Batch> due;
    // The snapshot of the tick it applied at, with what the batches changed.
    if (apply_due(index, due)) {
        quiet_ = 0;
        publish(index);
    }
}

bool WorldCore::tick_once() {
    const auto t0 = Clock::now();
    const uint64_t index = tick_.load();
    std::unique_lock<WorldLock> exclusive(world_lock_);
    bool applied = apply_due(index, due_);
    exclusive.unlock();
    const auto t1 = Clock::now();
    double python_s = 0.0;
    bool before = false;
    {
        std::shared_lock<WorldLock> shared(world_lock_);
        const auto c0 = Clock::now();
        before = callbacks_.has_at_tick.load() || callbacks_.has_before.load() ||
                 callbacks_.has_drivers.load();
        const bool ok = run_python_before(index);
        python_s += seconds(Clock::now() - c0);
        if (!ok)
            return false;
    }
    exclusive.lock();
    // What the before-step code submitted applies in this tick, before the step.
    if (before)
        applied = apply_due(index, due_) || applied;
    const bool was_settled = !before && !applied && settled();
    const auto t2 = Clock::now();
    move_kinematics(index);
    activate_motor_joints();
    characters_->step(dt_, uint32_t(index + 1), *jobs_, tick_events_);
    vehicles_->before_step(dt_);
    const auto r0 = Clock::now();
    ragdolls_->before_step(time_, dt_);
    const auto r1 = Clock::now();
    soft_->before_step(dt_, [this](Uid uid) {
        const Row *row = find(uid);
        return row ? row->id : JPH::BodyID();
    });
    const auto t2b = Clock::now();
    verdict_tick_ = uint32_t(index + 1);
    system_.Update(float(dt_), settings_.collision_steps, temp_.get(), jobs_.get());
    joints_after_step(index + 1);
    const auto t3 = Clock::now();
    exclusive.unlock();
    if (failed_.load())
        return false;

    std::shared_lock<WorldLock> shared(world_lock_);
    const uint64_t tick = index + 1;
    time_ = double(tick) * dt_;
    collector_.take(raw_contacts_, raw_activations_);
    const bool reported = !raw_contacts_.empty() || !raw_activations_.empty();
    for (const RawActivation &change : raw_activations_) {
        const uint32_t row = change.uid < row_of_uid_.size() ? row_of_uid_[change.uid] : kNoRow;
        if (row != kNoRow)
            publisher_.mark(row);
    }
    pairs_.fold(uint32_t(tick), raw_contacts_, raw_activations_, system_.GetBodyInterfaceNoLock(),
                tick_events_);
    expire_verdicts(uint32_t(tick));
    vehicles_->after_step(uint32_t(tick), tick_events_);
    const auto s0 = Clock::now();
    soft_->after_step(tick);
    const auto s1 = Clock::now();
    ring_.append(tick_events_);
    quiet_ = was_settled && !reported && tick_events_.empty() && settled() ? quiet_ + 1 : 0;
    const auto t4 = Clock::now();
    publish(tick);
    const auto t5 = Clock::now();
    // A body whose pose or velocity overflowed a float spreads NaN to every
    // body it meets: the world stops, naming it.
    if (nonfinite_ != kNoBody) {
        fail_native("body " + std::to_string(nonfinite_) + " has a position or velocity that is not finite after tick " +
                    std::to_string(tick));
        return false;
    }
    const auto c0 = Clock::now();
    const bool ok = run_python_after(tick);
    python_s += seconds(Clock::now() - c0);
    tick_events_.clear();
    const auto t6 = Clock::now();
    {
        std::lock_guard<std::mutex> guard(stats_lock_);
        sums_.apply_ms += seconds(t1 - t0);
        sums_.characters_ms += seconds(t2b - t2);
        sums_.ragdolls_ms += seconds(r1 - r0);
        sums_.soft_bodies_ms += seconds(t2b - r1) + seconds(s1 - s0);
        sums_.step_ms += seconds(t3 - t2b);
        sums_.events_ms += seconds(t4 - t3);
        sums_.publish_ms += seconds(t5 - t4);
        sums_.callbacks_ms += python_s;
        sums_.tick_ms += seconds(t6 - t0);
        ++sums_.timed_ticks;
    }
    return ok;
}

void WorldCore::publish(uint64_t tick) {
    for (JPH::EBodyType type : {JPH::EBodyType::RigidBody, JPH::EBodyType::SoftBody}) {
        system_.GetActiveBodies(type, active_);
        for (const JPH::BodyID &id : active_) {
            const uint32_t row = row_of_index_[id.GetIndex()];
            if (row != kNoRow)
                publisher_.mark(row);
        }
    }
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    auto buffer = publisher_.publish(
        tick, time_, applied_seq_, uint32_t(rows_.size()),
        [&](uint32_t row, SnapshotBuffer &out) {
            const Row &entry = rows_[row];
            const JPH::Body *body = locks.TryGetBody(entry.id);
            out.uid[row] = entry.uid;
            if (body == nullptr)
                return;
            const JPH::RVec3 p = body->GetPosition();
            const JPH::Quat q = body->GetRotation();
            const JPH::Vec3 v = body->GetLinearVelocity();
            const JPH::Vec3 w = body->GetAngularVelocity();
            double *position = &out.position[size_t(row) * 3];
            float *orientation = &out.orientation[size_t(row) * 4];
            float *linear = &out.linear_velocity[size_t(row) * 3];
            float *angular = &out.angular_velocity[size_t(row) * 3];
            position[0] = p.GetX();
            position[1] = p.GetY();
            position[2] = p.GetZ();
            orientation[0] = q.GetX();
            orientation[1] = q.GetY();
            orientation[2] = q.GetZ();
            orientation[3] = q.GetW();
            linear[0] = v.GetX();
            linear[1] = v.GetY();
            linear[2] = v.GetZ();
            angular[0] = w.GetX();
            angular[1] = w.GetY();
            angular[2] = w.GetZ();
            out.awake[row] = body->IsActive() ? 1 : 0;
            out.motion[row] = uint8_t(body->GetMotionType());
            if (nonfinite_ == kNoBody &&
                !(std::isfinite(position[0] + position[1] + position[2]) &&
                  std::isfinite(linear[0] + linear[1] + linear[2] + angular[0] + angular[1] + angular[2])))
                nonfinite_ = entry.uid;
        });
    characters_->write(*buffer);
    write_joints(*buffer);
    vehicles_->write(*buffer);
    ragdolls_->write(*buffer);
    soft_->write(*buffer);
    pool_size_ = publisher_.pool_size();
    std::lock_guard<std::mutex> guard(publish_lock_);
    published_ = std::move(buffer);
    tick_.store(tick);
    publish_cv_.notify_all();
}

// -- the contact filter ----------------------------------------------------------------

namespace {

// Marks the calling thread as inside a world's contact filter while it calls Python.
struct InContactFilter {
    explicit InContactFilter(const WorldCore *world) : outer(filtering_world) { filtering_world = world; }
    ~InContactFilter() { filtering_world = outer; }
    const void *outer;
};

}  // namespace

// On a Jolt job thread, or the world thread running jobs, during Update. The
// world thread holds the world lock exclusively and is not attached, so this
// thread may wait for the GIL: no thread that holds the GIL waits for the
// world lock (every such entry point releases the GIL first).
bool WorldCore::solve_pair(const JPH::Body &body1, const JPH::Body &body2, JPH::RVec3Arg base,
                           const JPH::CollideShapeResult &result) {
    if (!callbacks_.has_contact_filter.load() || failed_.load())
        return true;
    const uint64_t key = pair_key(body1.GetID(), body2.GetID());
    {
        std::lock_guard<std::mutex> guard(verdict_lock_);
        auto found = verdicts_.find(key);
        // With recheck a refusal holds for the step that made it, and the pair is asked again.
        if (found != verdicts_.end() &&
            (found->second.solve || !filter_recheck_.load() || found->second.tick == verdict_tick_)) {
            found->second.tick = verdict_tick_;
            return found->second.solve;
        }
    }
    const JPH::RVec3 point = base + result.mContactPointOn1;
    JPH::Vec3 normal = result.mPenetrationAxis;
    normal = normal.LengthSq() > 1e-18f ? normal.Normalized() : JPH::Vec3::sAxisY();
    const float approach = (body1.GetPointVelocity(point) - body2.GetPointVelocity(point)).Dot(normal);
    Uid a = uid_of(body1.GetUserData()), b = uid_of(body2.GetUserData());
    const bool flip = b < a;
    if (flip) {
        std::swap(a, b);
        normal = -normal;
    }
    bool solve = true;
    PythonCall call(*this);
    if (!call.admitted() || !python_available())
        return true;
    PyThreadState *state = nullptr;
    const bool world_thread = on_world_thread();
    if (world_thread) {
        if (!python_.attach())
            return true;
    } else {
        {
            std::lock_guard<std::mutex> guard(job_python_lock_);
            PyThreadState *&kept = job_python_[std::this_thread::get_id()];
            if (kept == nullptr)
                kept = PyThreadState_New(python_.interpreter());
            state = kept;
        }
        PyEval_RestoreThread(state);
    }
    {
        InContactFilter inside(this);
        try {
            nb::object fn;
            {
                std::lock_guard<std::mutex> guard(callbacks_.lock);
                fn = callbacks_.contact_filter;
            }
            if (fn.is_valid()) {
                nb::object answer = fn(a, b, nb::make_tuple(point.GetX(), point.GetY(), point.GetZ()),
                                       nb::make_tuple(normal.GetX(), normal.GetY(), normal.GetZ()), approach);
                solve = !(answer.is(nb::handle(Py_False)));
            }
        } catch (nb::python_error &error) {
            fail(error);
        } catch (const std::exception &error) {
            fail_native(std::string("the world thread failed in the contact filter: ") + error.what());
        }
    }
    if (world_thread)
        python_.detach();
    else
        PyEval_SaveThread();
    std::lock_guard<std::mutex> guard(verdict_lock_);
    auto [found, made] = verdicts_.try_emplace(key, Verdict{solve, verdict_tick_});
    if (!made && filter_recheck_.load())
        found->second = Verdict{solve, verdict_tick_};
    return found->second.solve;
}

// A refused pair's verdict goes in the first tick it is not met again; a
// solved pair's goes when its contacts end.
void WorldCore::expire_verdicts(uint32_t tick) {
    std::lock_guard<std::mutex> guard(verdict_lock_);
    for (auto it = verdicts_.begin(); it != verdicts_.end();) {
        const bool met = it->second.tick == tick;
        if (!met && !(it->second.solve && pairs_.touching(it->first)))
            it = verdicts_.erase(it);
        else
            ++it;
    }
}

// At the end of the world thread, attached: every job thread's kept state is
// deleted, so a later run of the thread makes new ones.
void WorldCore::release_job_python() {
    std::lock_guard<std::mutex> guard(job_python_lock_);
    if (job_python_.empty())
        return;
    if (python_available() && python_.attach()) {
        for (auto &[id, state] : job_python_) {
            PyThreadState_Clear(state);
            PyThreadState_Delete(state);
        }
        python_.detach();
    }
    job_python_.clear();
}

// -- reading ---------------------------------------------------------------------

std::shared_ptr<SnapshotBuffer> WorldCore::latest() const {
    check();
    std::lock_guard<std::mutex> guard(publish_lock_);
    return published_;
}

WorldStats WorldCore::stats() const {
    WorldStats out;
    {
        std::lock_guard<std::mutex> guard(stats_lock_);
        out = sums_;
    }
    if (out.timed_ticks) {
        const double per = 1e3 / double(out.timed_ticks);
        out.apply_ms *= per;
        out.characters_ms *= per;
        out.ragdolls_ms *= per;
        out.soft_bodies_ms *= per;
        out.step_ms *= per;
        out.events_ms *= per;
        out.publish_ms *= per;
        out.callbacks_ms *= per;
        out.tick_ms *= per;
    }
    out.ticks = tick_.load();
    out.dropped_ticks = dropped_ticks_.load();
    out.events_dropped = ring_.dropped();
    out.events_held = ring_.size();
    out.rejected = rejected_.load();
    out.pool = pool_size_.load();
    out.bodies = applied_bodies_.load();
    out.characters = applied_characters_.load();
    out.vehicles = applied_vehicles_.load();
    out.ragdolls = applied_ragdolls_.load();
    out.soft_bodies = applied_soft_.load();
    out.queued_bodies =
        size_t(std::max<int64_t>(0, reserved_bodies_.load())) - out.bodies - out.characters;
    out.queries = queries_.load();
    out.query_wait_ms = out.queries ? 1e-6 * double(query_wait_ns_.load()) / double(out.queries) : 0.0;
    out.query_wait_max_ms = 1e-6 * double(query_wait_max_ns_.load());
    out.shapes = shapes_.registered();
    out.broadphase_optimizations = optimizations_.load();
    {
        std::lock_guard<std::mutex> guard(control_);
        out.idle = idle_;
        out.running = running_;
    }
    return out;
}

void WorldCore::reset_stats() {
    std::lock_guard<std::mutex> guard(stats_lock_);
    sums_ = WorldStats();
    queries_ = 0;
    query_wait_ns_ = 0;
    query_wait_max_ns_ = 0;
}

// -- the registry, atexit and fork ----------------------------------------------------

namespace {

std::mutex registry_lock;
std::vector<std::weak_ptr<WorldCore>> registry;
std::vector<std::shared_ptr<WorldCore>> forking;

std::vector<std::shared_ptr<WorldCore>> live_worlds() {
    std::vector<std::shared_ptr<WorldCore>> out;
    for (auto &weak : registry)
        if (auto world = weak.lock())
            out.push_back(std::move(world));
    return out;
}

}  // namespace

void register_world(const std::shared_ptr<WorldCore> &world) {
    std::lock_guard<std::mutex> guard(registry_lock);
    registry.erase(std::remove_if(registry.begin(), registry.end(),
                                  [](const std::weak_ptr<WorldCore> &w) { return w.expired(); }),
                   registry.end());
    registry.push_back(world);
}

void stop_all_worlds() {
    mark_interpreter_exiting();
    std::vector<std::shared_ptr<WorldCore>> worlds;
    {
        std::lock_guard<std::mutex> guard(registry_lock);
        worlds = live_worlds();
    }
    for (auto &world : worlds)
        world->stop();
}

// The registry lock is held from before the fork to after it, so the child
// starts with no world half made and can make its own.
void fork_prepare() {
    registry_lock.lock();
    forking = live_worlds();
}

void fork_parent() {
    forking.clear();
    registry_lock.unlock();
}

void fork_child() {
    for (auto &world : forking)
        world->mark_forked();
    registry.clear();
    // The child keeps these references: destroying a world here would touch
    // threads and locks that belong to the parent.
    new std::vector<std::shared_ptr<WorldCore>>(std::move(forking));
    forking.clear();
    registry_lock.unlock();
}

void WorldCore::mark_forked() {
    forked_ = true;
    // The interpreter deleted every thread state but the forking thread's.
    python_.forget();
    job_python_.clear();
}

}  // namespace pj
