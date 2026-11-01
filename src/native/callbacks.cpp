#include "callbacks.h"

namespace pj {

namespace {

// Set by the atexit hook, which stops every world before the interpreter
// starts finalizing. The stable ABI of 3.12 has no Py_IsFinalizing, and
// CPython before 3.13 names it _Py_IsFinalizing.
std::atomic<bool> exiting{false};

bool finalizing() {
#if PY_VERSION_HEX >= 0x030D0000 && (!defined(Py_LIMITED_API) || Py_LIMITED_API >= 0x030D0000)
    if (Py_IsFinalizing())
        return true;
#elif !defined(Py_LIMITED_API)
    if (_Py_IsFinalizing())
        return true;
#endif
    return exiting.load();
}

}  // namespace

void mark_interpreter_exiting() { exiting = true; }

bool python_available() { return !finalizing(); }

thread_local const void *filtering_world = nullptr;

namespace {
PyObject *logger = nullptr;  // a reference kept for the life of the process
}

void keep_callback_logger() {
    if (logger == nullptr)
        logger = nb::module_::import_("logging").attr("getLogger")("pyjoltphysics").release().ptr();
}

nb::handle callback_logger() { return logger; }

PythonThread::PythonThread() : interpreter_(PyInterpreterState_Get()) {}

bool PythonThread::attach() {
    if (finalizing())
        return false;
    if (state_ == nullptr)
        state_ = PyThreadState_New(interpreter_);
    PyEval_RestoreThread(state_);
    return true;
}

void PythonThread::detach() { PyEval_SaveThread(); }

void PythonThread::release() {
    if (state_ == nullptr || finalizing())
        return;
    PyEval_RestoreThread(state_);
    PyThreadState_Clear(state_);
    PyEval_SaveThread();
    PyThreadState_Delete(state_);
    state_ = nullptr;
}

void CallbackSet::set(nb::object &slot, std::atomic<bool> &flag, nb::object fn) {
    nb::object old;
    {
        std::lock_guard<std::mutex> guard(lock);
        old = std::move(slot);
        slot = fn.is_none() ? nb::object() : std::move(fn);
        flag = slot.is_valid();
    }
    // `old` is released here, outside the lock: its finalizer may run Python.
}

void CallbackSet::add_at_tick(uint64_t tick, nb::object fn) {
    std::lock_guard<std::mutex> guard(lock);
    at_tick.emplace(tick, std::move(fn));
    has_at_tick = true;
    earliest_at_tick = at_tick.begin()->first;
}

std::vector<nb::object> CallbackSet::take_due(uint64_t tick) {
    std::vector<nb::object> due;
    std::lock_guard<std::mutex> guard(lock);
    auto end = at_tick.upper_bound(tick);
    for (auto it = at_tick.begin(); it != end; ++it)
        due.push_back(std::move(it->second));
    at_tick.erase(at_tick.begin(), end);
    has_at_tick = !at_tick.empty();
    earliest_at_tick = at_tick.empty() ? UINT64_MAX : at_tick.begin()->first;
    return due;
}

int CallbackSet::traverse(visitproc visit, void *arg) {
    for (nb::object *slot : {&before_step, &after_step, &on_events, &drivers, &contact_filter})
        if (slot->is_valid())
            Py_VISIT(slot->ptr());
    for (auto &entry : at_tick)
        if (entry.second.is_valid())
            Py_VISIT(entry.second.ptr());
    return 0;
}

void CallbackSet::clear() {
    nb::object before, after, events, drive, filter;
    std::multimap<uint64_t, nb::object> pending;
    {
        std::lock_guard<std::mutex> guard(lock);
        before = std::move(before_step);
        after = std::move(after_step);
        events = std::move(on_events);
        drive = std::move(drivers);
        filter = std::move(contact_filter);
        pending.swap(at_tick);
        has_before = has_after = has_events = has_at_tick = has_drivers = has_contact_filter = false;
        earliest_at_tick = UINT64_MAX;
    }
}

}  // namespace pj
