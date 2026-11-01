#include "threads.h"

#include <exception>
#include <memory>
#include <system_error>
#include <utility>

namespace pj {

#ifdef _WIN32

Thread::Thread(size_t, std::function<void()> body) : thread_(std::move(body)) {}
Thread::Thread(Thread &&other) noexcept = default;
Thread &Thread::operator=(Thread &&other) noexcept = default;
Thread::~Thread() = default;
bool Thread::joinable() const { return thread_.joinable(); }
void Thread::join() { thread_.join(); }
void Thread::detach() { thread_.detach(); }

#else

namespace {

void *run(void *arg) {
    std::unique_ptr<std::function<void()>> body(static_cast<std::function<void()> *>(arg));
    try {
        (*body)();
    } catch (...) {
        std::terminate();
    }
    return nullptr;
}

}  // namespace

Thread::Thread(size_t stack, std::function<void()> body) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stack);
    auto *heap = new std::function<void()>(std::move(body));
    const int error = pthread_create(&handle_, &attr, run, heap);
    pthread_attr_destroy(&attr);
    if (error != 0) {
        delete heap;
        throw std::system_error(error, std::generic_category(), "could not start the world thread");
    }
    joinable_ = true;
}

Thread::Thread(Thread &&other) noexcept
    : handle_(other.handle_), joinable_(std::exchange(other.joinable_, false)) {}

Thread &Thread::operator=(Thread &&other) noexcept {
    if (joinable_)
        std::terminate();
    handle_ = other.handle_;
    joinable_ = std::exchange(other.joinable_, false);
    return *this;
}

Thread::~Thread() {
    if (joinable_)
        std::terminate();
}

bool Thread::joinable() const { return joinable_; }

void Thread::join() {
    if (!joinable_)
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "the thread is not joinable");
    const int error = pthread_join(handle_, nullptr);
    if (error != 0)
        throw std::system_error(error, std::generic_category(), "could not join the world thread");
    joinable_ = false;
}

void Thread::detach() {
    if (!joinable_)
        throw std::system_error(std::make_error_code(std::errc::invalid_argument), "the thread is not joinable");
    pthread_detach(handle_);
    joinable_ = false;
}

#endif

namespace {
std::mutex default_stack_lock;
}  // namespace

DefaultThreadStack::DefaultThreadStack(size_t stack) : lock_(default_stack_lock) {
#if defined(__linux__)
    pthread_attr_t attr;
    if (pthread_getattr_default_np(&attr) != 0)
        return;
    size_t current = 0;
    pthread_attr_getstacksize(&attr, &current);
    if (current < stack && pthread_attr_setstacksize(&attr, stack) == 0 && pthread_setattr_default_np(&attr) == 0)
        previous_ = current;
    pthread_attr_destroy(&attr);
#else
    (void)stack;
#endif
}

DefaultThreadStack::~DefaultThreadStack() {
#if defined(__linux__)
    if (previous_ == 0)
        return;
    pthread_attr_t attr;
    if (pthread_getattr_default_np(&attr) != 0)
        return;
    if (pthread_attr_setstacksize(&attr, previous_) == 0)
        pthread_setattr_default_np(&attr);
    pthread_attr_destroy(&attr);
#endif
}

}  // namespace pj
