// The threads that run Python callbacks: the world thread, and Jolt's job
// threads, which run the contact filter.
#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace pj {

// The stack of the world thread and of Jolt's job threads: 8 MiB, glibc's
// usual default. musl gives a new thread 128 KiB and macOS 512 KiB, which a
// Python callback recursing through C code can overflow.
constexpr size_t kThreadStack = size_t(8) << 20;

// A joinable thread, like std::thread, with a stack of `stack` bytes on POSIX
// systems. On Windows it has the executable's default stack (2 MB for
// python.exe), as Python's own threads do.
class Thread {
public:
    Thread() = default;
    Thread(size_t stack, std::function<void()> body);
    Thread(Thread &&other) noexcept;
    Thread &operator=(Thread &&other) noexcept;
    Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;
    // As std::thread's: a thread still joinable ends the process.
    ~Thread();

    bool joinable() const;
    void join();
    void detach();

private:
#ifdef _WIN32
    std::thread thread_;
#else
    pthread_t handle_{};
    bool joinable_ = false;
#endif
};

// While one is alive, threads made in the process without a stack size of
// their own get at least `stack` bytes. Jolt's thread pool makes std::threads,
// which take no stack size, so the world makes its pool under one. It acts on
// Linux, through the C library's default thread attributes (glibc and musl);
// a thread another library makes meanwhile also gets the larger stack. On
// macOS Jolt's job threads keep 512 KiB.
class DefaultThreadStack {
public:
    explicit DefaultThreadStack(size_t stack);
    ~DefaultThreadStack();
    DefaultThreadStack(const DefaultThreadStack &) = delete;
    DefaultThreadStack &operator=(const DefaultThreadStack &) = delete;

private:
    std::unique_lock<std::mutex> lock_;
    size_t previous_ = 0;  // the default to put back; 0 when unchanged
};

}  // namespace pj
