// Python callables run on the world thread.
//
// The world thread keeps one Python thread state for its life and attaches it
// only to run a callback: a world with no callback registered never takes the
// GIL. Attaching a kept state costs about 0.1 us; creating one per call costs
// 2 us with the GIL and over 100 us free-threaded.
//
// Lock order: the world thread may attach while it holds the shared side of
// the world lock, so no thread may wait for the world lock, or for anything
// the world thread holds, while attached. Every entry point that can wait
// releases the GIL first.
#pragma once

#include "common.h"

#include <nanobind/nanobind.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace pj {

namespace nb = nanobind;

// Called by the atexit hook: from then on no world thread attaches.
void mark_interpreter_exiting();
// False while the interpreter is finalizing or exiting, when no callback runs.
bool python_available();
// The "pyjoltphysics" logger, looked up once when the module loads: a world
// thread logging a callback's error imports nothing, which during interpreter
// exit could wait on the import lock held by the thread joining it.
void keep_callback_logger();
nb::handle callback_logger();

// The world whose contact filter the calling thread runs, inside the step;
// null on a thread outside every contact filter.
extern thread_local const void *filtering_world;

class PythonThread {
public:
    PythonThread();
    // False while the interpreter is finalizing, when no callback runs.
    bool attach();
    void detach();
    // At the end of the world thread: deletes the kept thread state.
    void release();
    // In the child of a fork, where the interpreter has deleted every thread
    // state but the forking thread's.
    void forget() { state_ = nullptr; }
    PyInterpreterState *interpreter() const { return interpreter_; }

private:
    PyInterpreterState *interpreter_;
    PyThreadState *state_ = nullptr;
};

// The registered callables. Written by Python threads and read by the world
// thread, each attached, under `lock`. The flags let the world thread skip
// attaching when nothing is registered.
struct CallbackSet {
    std::mutex lock;
    nb::object before_step;
    nb::object after_step;
    nb::object on_events;
    nb::object drivers;         // fn(tick, time) after before_step: Body.drive's pose callables
    nb::object contact_filter;  // fn(a, b, point, normal, approach) on Jolt's job threads
    std::multimap<uint64_t, nb::object> at_tick;  // equal ticks fire in registration order
    std::atomic<bool> has_before{false};
    std::atomic<bool> has_after{false};
    std::atomic<bool> has_events{false};
    std::atomic<bool> has_at_tick{false};
    std::atomic<bool> has_drivers{false};
    std::atomic<bool> has_contact_filter{false};
    // The lowest tick an at_tick callable waits for; UINT64_MAX for none.
    std::atomic<uint64_t> earliest_at_tick{UINT64_MAX};

    void set(nb::object &slot, std::atomic<bool> &flag, nb::object fn);
    void add_at_tick(uint64_t tick, nb::object fn);
    // Removes and returns the at_tick callables due at `tick`.
    std::vector<nb::object> take_due(uint64_t tick);
    bool any_step() const { return has_before.load() || has_after.load() || has_drivers.load(); }
    // The cyclic garbage collector's view: every callable held.
    int traverse(visitproc visit, void *arg);
    // Drops every callable; the caller is attached.
    void clear();
};

}  // namespace pj
