// The world: one Jolt PhysicsSystem stepped at a fixed rate on a native
// thread the world owns.
//
// A tick, on the world thread:
//   1. apply the command batches due, under the exclusive world lock
//   2. at_tick callables, then before_step, under the shared lock (if any)
//   3. move the characters, then PhysicsSystem::Update, under the exclusive
//      world lock
//   4. fold the step's contact reports into events, append them to the ring
//   5. publish the snapshot
//   6. after_step, then on_events (if any)
// The world lock is held exclusively only for steps 1 and 3, so queries, which
// take its shared side, run beside the rest.
//
// The thread free-runs at tick_hz, or in lockstep runs only the ticks
// run_ticks asks for. A free-running world with no pending batch, no awake
// body and no step callback waits on a condition variable and runs no ticks.
// In lockstep, once such a world has stepped kQuietAfter ticks that applied
// and reported nothing, Jolt's state no longer changes from tick to tick, and
// run_ticks counts its ticks and publishes on the calling thread without
// stepping.
#pragma once

#include "callbacks.h"
#include "characters.h"
#include "commands.h"
#include "common.h"
#include "events.h"
#include "layers.h"
#include "queries.h"
#include "ragdolls.h"
#include "shapes.h"
#include "snapshot.h"
#include "softbodies.h"
#include "vehicles.h"

#include <Jolt/Core/JobSystem.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pj {

struct WorldSettings {
    double tick_hz = 120.0;
    double max_catch_up = 0.1;  // seconds behind schedule before ticks are dropped
    double gravity[3] = {0.0, -9.81, 0.0};
    int threads = 4;            // job threads, the world thread included
    uint32_t max_bodies = 65536;
    uint32_t max_body_pairs = 65536;
    uint32_t max_contact_constraints = 65536;
    size_t event_capacity = 65536;
    bool allow_sleeping = true;
    float time_before_sleep = 0.5f;    // seconds below the sleep speed
    float sleep_velocity = 0.03f;      // m/s at every point of the body
    float min_velocity_for_restitution = 1.0f;  // m/s
    int collision_steps = 1;
    uint32_t temp_allocator_mb = 16;
    float static_friction_speed = 0.05f;  // m/s of sliding below which a material's static friction holds
    float penetration_slop = 0.02f;       // m bodies may sink into each other before the solver separates them
};

// A callback raised; the world has stopped and keeps the exception.
struct CallbackFailed : std::runtime_error {
    CallbackFailed() : std::runtime_error("a world callback raised; the world has stopped") {}
};

// The world was stopped before what the caller waited for happened.
struct WorldStopped : std::runtime_error {
    WorldStopped() : std::runtime_error("the world was stopped") {}
};

struct WorldStats {
    uint64_t ticks = 0;
    uint64_t dropped_ticks = 0;
    uint64_t events_dropped = 0;
    uint64_t rejected = 0;  // commands naming a body that cannot take them
    size_t pool = 0;
    size_t bodies = 0;          // in the world: applied adds less applied removals
    size_t queued_bodies = 0;   // submitted in add batches not yet applied
    size_t shapes = 0;          // registered shape ids
    size_t characters = 0;
    size_t vehicles = 0;
    size_t ragdolls = 0;
    size_t soft_bodies = 0;
    uint64_t broadphase_optimizations = 0;
    size_t pairs = 0;
    size_t events_held = 0;
    uint32_t awake = 0;
    bool idle = false;
    bool running = false;
    // Mean milliseconds a tick, since the last reset.
    double apply_ms = 0, characters_ms = 0, ragdolls_ms = 0, soft_bodies_ms = 0, step_ms = 0, events_ms = 0, publish_ms = 0, callbacks_ms = 0, tick_ms = 0;
    uint64_t timed_ticks = 0;
    // Lockstep ticks a quiet world counted without stepping, since the last reset.
    uint64_t quiet_ticks = 0;
    // Queries that waited for the world lock: how many, the mean and the longest wait.
    uint64_t queries = 0;
    double query_wait_ms = 0, query_wait_max_ms = 0;
};

// The world lock: a shared mutex that admits no new reader while the world
// thread waits for it. std::shared_mutex prefers readers on glibc, so queries
// from several threads in a row would hold the world thread off its tick
// indefinitely; here a query arriving while the world thread waits queues
// behind it.
class WorldLock {
public:
    void lock() {
        writer_waiting_.store(true);
        mutex_.lock();
        {
            std::lock_guard<std::mutex> guard(gate_);
            writer_waiting_.store(false);
        }
        open_.notify_all();
    }
    void unlock() { mutex_.unlock(); }
    void lock_shared() {
        if (writer_waiting_.load()) {
            std::unique_lock<std::mutex> guard(gate_);
            open_.wait(guard, [this] { return !writer_waiting_.load(); });
        }
        mutex_.lock_shared();
    }
    void unlock_shared() { mutex_.unlock_shared(); }

private:
    std::shared_mutex mutex_;
    std::atomic<bool> writer_waiting_{false};
    std::mutex gate_;
    std::condition_variable open_;
};

class WorldCore : public std::enable_shared_from_this<WorldCore>, public ContactFilterHost {
public:
    explicit WorldCore(const WorldSettings &settings);
    ~WorldCore();
    WorldCore(const WorldCore &) = delete;
    WorldCore &operator=(const WorldCore &) = delete;

    const WorldSettings &settings() const { return settings_; }

    // -- commands: any thread, never waits for the world ---------------------
    // Returns the batch's sequence number; an add also returns its first uid.
    uint64_t submit(Command &&command, uint64_t at_tick, Uid *first_uid = nullptr, JointId *first_joint = nullptr,
                    VehicleId *vehicle = nullptr, RagdollId *ragdoll = nullptr);
    uint64_t submitted_seq() const { return queue_.last_seq(); }

    // -- the thread: callers release the GIL ---------------------------------
    void start();
    void stop();
    void run_ticks(uint64_t count);
    // Applies the batches due at the current tick on the calling thread, under
    // the world lock, without stepping; nothing on the world thread itself.
    void flush();
    bool wait_tick(uint64_t tick, double timeout);
    bool lockstep() const;
    bool on_world_thread() const { return std::this_thread::get_id() == thread_id_.load(); }
    // stop() for a world dropped by Python: joins, or on the world thread
    // itself detaches the thread, which ends after the current tick.
    void shutdown();

    // -- reading: any thread -------------------------------------------------
    std::shared_ptr<SnapshotBuffer> latest() const;
    std::vector<Event> drain_events() { return ring_.drain(); }
    std::vector<uint64_t> drain_applied();
    uint64_t tick() const { return tick_.load(); }
    WorldStats stats() const;
    void reset_stats();

    ShapeLibrary &shapes() { return shapes_; }
    // Whether the contact filter is asked where either body carries FILTER_CONTACTS.
    void set_filter_either(bool either) { collector_.filter_either.store(either); }
    // Whether a pair the contact filter refused is asked again at each later step.
    void set_filter_recheck(bool recheck) { filter_recheck_.store(recheck); }

    // The bodies' and constraints' state as Jolt's StateRecorder writes it,
    // read between ticks under the world lock; callers release the GIL.
    std::string save_state();

    // -- the world thread, inside a callback ---------------------------------
    // Kinematic targets for the end of the current tick, from Body.drive's
    // callables; applied after before_step.
    void move_kinematic_now(MoveKinematic &&move);

    // -- the contact filter: Jolt's job threads ------------------------------
    // Asks the Python contact filter about a pair; true solves it.
    bool solve_pair(const JPH::Body &body1, const JPH::Body &body2, JPH::RVec3Arg base,
                    const JPH::CollideShapeResult &result) override;

    // -- queries: any thread; callers release the GIL ------------------------
    void raycast(const double *origins, const double *directions, size_t count, double max_distance,
                 const QueryFilter &filter, RayResults &out);
    std::vector<ShapeHit> cast_shape(const JPH::Shape &shape, const double start[3], const double rotation[4],
                                     const double end[3], bool all_hits, const QueryFilter &filter);
    // Each body's pose, velocities and wakefulness as Jolt holds them now; found is 0 for a uid
    // that names no body.
    void bodies_state(const Uid *uids, size_t count, double *position, float *orientation, float *linear,
                      float *angular, uint8_t *awake, uint8_t *found);
    // The number a mesh triangle was given, for a query's sub-shape on body uid; -1 for other parts.
    int64_t triangle_of(Uid uid, uint32_t sub_shape);
    std::vector<Uid> overlap(const JPH::Shape &shape, const double position[3], const double rotation[4],
                             const QueryFilter &filter);

    // -- Python state: callers are attached ----------------------------------
    CallbackSet &callbacks() { return callbacks_; }
    // Wakes an idle world thread, as after a step callback is registered.
    void poke();
    // The cyclic garbage collector's view: the callables and the stored error.
    int traverse(visitproc visit, void *arg);
    nb::object error() const;
    bool failed() const { return failed_.load(); }
    void clear_python();

    // -- fork ----------------------------------------------------------------
    void mark_forked();
    bool forked() const { return forked_.load(); }
    void check() const {
        if (forked_.load())
            throw ForkedError();
    }

private:
    enum class Mode { Free, Lockstep };
    using Clock = std::chrono::steady_clock;
    static constexpr uint32_t kNoRow = 0xffffffffu;
    static constexpr size_t kOptimizeAfterStatics = 256;

    struct Row {
        Uid uid;
        JPH::BodyID id;
    };

    // Ticks of a settled world after which a tick changes nothing but the tick count.
    static constexpr uint64_t kQuietAfter = 2;

    void loop();
    bool busy() const;
    bool tick_once();
    // Whether nothing in the world moves or acts at a tick: no awake body, character,
    // kinematic track or target, motor torque or step callback.
    bool settled() const;
    // run_ticks for a quiet world: the tick count and time advance by `count` and one
    // snapshot is published, with no step; false where any of the ticks could change
    // the world.
    bool run_quiet(uint64_t count);
    bool run_python_before(uint64_t index);
    bool run_python_after(uint64_t tick);
    template <typename Body> bool call_python(Body &&body);
    void fail(nb::python_error &error);
    void note_applied(uint64_t seq);
    // Applies the batches due at tick `index`, logging them; whether there were any.
    bool apply_due(uint64_t index, std::vector<Batch> &due);
    void publish(uint64_t tick);

    // apply.cpp: one function per command, on the world thread.
    void apply(Batch &batch, uint64_t index);
    void apply(AddBodies &add, uint64_t index);
    void apply(RemoveBodies &remove, uint64_t index);
    void apply(SetTransforms &set, uint64_t index);
    void apply(SetVelocities &set, uint64_t index);
    void apply(AddPushes &push, uint64_t index);
    void apply(SetGravity &set, uint64_t index);
    void apply(Activate &activate, uint64_t index);
    void apply(SetMotionTypes &set, uint64_t index);
    void apply(SetMaterials &set, uint64_t index);
    void apply(SwapBodies &swap, uint64_t index);
    void apply(AddCharacter &add, uint64_t index);
    void apply(RemoveCharacters &remove, uint64_t index);
    void apply(CharacterOps &ops, uint64_t index);
    void apply(SetCharacterIgnore &set, uint64_t index);
    void apply(AddJoints &add, uint64_t index);
    void apply(RemoveJoints &remove, uint64_t index);
    void apply(SetJointMotors &set, uint64_t index);
    void apply(SetJointsEnabled &set, uint64_t index);
    void apply(SetBodyMaterials &set, uint64_t index);
    void apply(SetBodyFilters &set, uint64_t index);
    void apply(SetMaterialTable &set, uint64_t index);
    void apply(SetFilterTable &set, uint64_t index);
    void apply(SetGravityZones &set, uint64_t index);
    void apply(SetDrag &set, uint64_t index);
    void apply(SetTracks &set, uint64_t index);
    void apply(MoveKinematic &move, uint64_t index);
    void apply(SetMotion &set, uint64_t index);
    void apply(SetFlags &set, uint64_t index);
    void apply(RestoreState &restore, uint64_t index);
    void apply(AddVehicle &add, uint64_t index);
    void apply(RemoveVehicles &remove, uint64_t index);
    void apply(VehicleInputs &inputs, uint64_t index);
    void apply(SetVehicleSurfaces &set, uint64_t index);
    void apply(SetSurfaceTable &set, uint64_t index);
    void apply(AddRagdoll &add, uint64_t index);
    void apply(RemoveRagdolls &remove, uint64_t index);
    void apply(SetRagdollModes &set, uint64_t index);
    void apply(SetRagdollPose &set, uint64_t index);
    void apply(SetRagdollClip &set, uint64_t index);
    void apply(SetRagdollStrength &set, uint64_t index);
    void apply(AddSoftBody &add, uint64_t index);
    void apply(SetSoftPins &set, uint64_t index);
    void apply(SetWind &set, uint64_t index);
    // Takes a ragdoll out: its pairs, rows, constraints and bodies.
    void remove_ragdoll(RagdollId id, uint64_t index);
    // joints.cpp
    void remove_joint(Joint &joint, uint64_t index, uint16_t reason);
    void remove_joints_of(const std::vector<Uid> &uids, uint64_t index);
    void joints_after_step(uint64_t tick);
    void write_joints(SnapshotBuffer &out) const;
    void activate_motor_joints();
    void rebuild_torques();
    // fields: tracks and Body.drive's targets, before the step
    void move_kinematics(uint64_t index);
    void wake_dynamic();
    // The contact filter's verdicts, dropped once a pair parts.
    void expire_verdicts(uint32_t tick);
    void remove_proxies(std::vector<JPH::BodyID> ids, uint64_t index);
    void wake_under_sensor(const JPH::BodyID &sensor);
    // Rebuilds the broad phase's trees after a large add of static bodies, once
    // the world thread has a moment with no tick due.
    void optimize_if_due();
    const Row *find(Uid uid) const;
    void bind_row(Uid uid, const JPH::BodyID &id);
    void unbind_row(Uid uid);

    WorldSettings settings_;
    double dt_;

    // Jolt. The system holds pointers to the layers and listeners, so it is declared
    // after them and destroyed first.
    CollisionLayers layers_;
    Collector collector_;
    std::unique_ptr<JPH::TempAllocator> temp_;
    std::unique_ptr<JPH::JobSystem> jobs_;
    JPH::PhysicsSystem system_;
    // Destroyed before the system, whose bodies its characters reference.
    std::unique_ptr<CharacterSystem> characters_;
    // Destroyed before the system, which holds its constraints and step listeners.
    std::unique_ptr<VehicleSystem> vehicles_;
    std::atomic<size_t> applied_vehicles_{0};
    // Destroyed before the system, which holds their bodies and constraints.
    std::unique_ptr<RagdollSystem> ragdolls_;
    std::atomic<size_t> applied_ragdolls_{0};
    std::unique_ptr<SoftBodySystem> soft_;
    std::atomic<size_t> applied_soft_{0};
    ShapeLibrary shapes_;

    // Rows are dense: the snapshot's arrays have one per live body. Removing a
    // body moves the last row into its place.
    std::vector<Row> rows_;
    std::vector<uint32_t> row_of_uid_;
    std::vector<uint32_t> row_of_index_;  // by BodyID index
    std::atomic<int64_t> reserved_bodies_{0};
    std::atomic<size_t> applied_bodies_{0};
    size_t statics_since_optimize_ = 0;
    std::atomic<uint64_t> optimizations_{0};

    CommandQueue queue_;
    std::vector<Batch> due_;
    uint64_t applied_seq_ = 0;
    std::set<uint64_t> applied_ahead_;  // applied, above a gap left by an at_tick batch
    std::mutex applied_lock_;
    std::vector<uint64_t> applied_log_;  // (seq, tick index) pairs

    SnapshotPublisher publisher_;
    mutable std::mutex publish_lock_;
    std::condition_variable publish_cv_;
    std::shared_ptr<SnapshotBuffer> published_;
    JPH::BodyIDVector active_;

    MaterialTable materials_;
    FieldListener fields_;
    std::vector<Joint> joints_;  // ascending id
    JPH::Ref<JointPairFilter> joint_filter_;
    struct Tracked {
        Track track;
        double start;  // simulation time the track starts at
    };
    std::map<Uid, Tracked> tracks_;
    std::vector<MoveKinematic> driven_;  // targets from Body.drive, this tick

    // The contact filter: verdicts by body pair, and a Python thread state
    // kept for each job thread that has asked.
    struct Verdict {
        bool solve;
        uint32_t tick;
    };
    std::mutex verdict_lock_;
    std::unordered_map<uint64_t, Verdict> verdicts_;
    std::atomic<bool> filter_recheck_{false};
    uint32_t verdict_tick_ = 0;
    std::mutex job_python_lock_;
    std::unordered_map<std::thread::id, PyThreadState *> job_python_;
    void release_job_python();

    PairTracker pairs_;
    EventRing ring_;
    std::vector<Event> tick_events_;
    std::vector<RawContact> raw_contacts_;
    std::vector<RawActivation> raw_activations_;

    PythonThread python_;
    CallbackSet callbacks_;
    mutable std::mutex error_lock_;
    nb::object error_;
    std::atomic<bool> failed_{false};

    WorldLock world_lock_;
    // The shared side of the world lock for a query; nothing on the world thread,
    // whose tick already holds it.
    class QueryLock {
    public:
        explicit QueryLock(WorldCore &world);

    private:
        std::shared_lock<WorldLock> lock_;
    };
    std::atomic<uint64_t> queries_{0};
    std::atomic<uint64_t> query_wait_ns_{0};
    std::atomic<uint64_t> query_wait_max_ns_{0};
    std::atomic<size_t> applied_characters_{0};
    mutable std::mutex control_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::mutex join_lock_;
    std::thread thread_;
    std::atomic<std::thread::id> thread_id_{};
    Mode mode_ = Mode::Lockstep;
    bool stopping_ = false;
    bool running_ = false;
    bool idle_ = false;
    uint64_t wanted_ = 0;
    std::atomic<bool> forked_{false};

    std::atomic<uint64_t> tick_{0};
    double time_ = 0.0;
    // Consecutive ticks that stepped a settled world, applied nothing and reported nothing.
    uint64_t quiet_ = 0;
    mutable std::mutex stats_lock_;
    WorldStats sums_;
    std::atomic<uint64_t> dropped_ticks_{0};
    std::atomic<uint64_t> rejected_{0};
    std::atomic<size_t> pool_size_{0};
};

// Every live world, for the atexit and fork hooks.
void register_world(const std::shared_ptr<WorldCore> &world);
void stop_all_worlds();
void fork_prepare();
void fork_parent();
void fork_child();

}  // namespace pj
