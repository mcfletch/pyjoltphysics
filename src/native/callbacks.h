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
#include <map>
#include <mutex>
#include <vector>

namespace pj {

namespace nb = nanobind;

// Called by the atexit hook: from then on no world thread attaches.
void mark_interpreter_exiting();

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
    std::multimap<uint64_t, nb::object> at_tick;  // equal ticks fire in registration order
    std::atomic<bool> has_before{false};
    std::atomic<bool> has_after{false};
    std::atomic<bool> has_events{false};
    std::atomic<bool> has_at_tick{false};

    void set(nb::object &slot, std::atomic<bool> &flag, nb::object fn);
    void add_at_tick(uint64_t tick, nb::object fn);
    // Removes and returns the at_tick callables due at `tick`.
    std::vector<nb::object> take_due(uint64_t tick);
    bool any_step() const { return has_before.load() || has_after.load(); }
    // The cyclic garbage collector's view: every callable held.
    int traverse(visitproc visit, void *arg);
    // Drops every callable; the caller is attached.
    void clear();
};

}  // namespace pj
