#include "commands.h"

#include <limits>

namespace pj {

uint64_t CommandQueue::push(Batch &&batch, uint32_t uid_count, Uid *first_uid) {
    std::lock_guard<std::mutex> guard(lock_);
    if (uid_count) {
        if (uint64_t(next_uid_) + uid_count > std::numeric_limits<Uid>::max())
            throw std::length_error("the world has assigned every 32-bit body uid");
        auto &uids = std::get<AddBodies>(batch.command).uids;
        uids.resize(uid_count);
        for (uint32_t i = 0; i < uid_count; ++i)
            uids[i] = next_uid_ + i;
        if (first_uid)
            *first_uid = next_uid_;
        next_uid_ += uid_count;
    }
    batch.seq = ++seq_;
    const uint64_t seq = batch.seq;
    queue_.push_back(std::move(batch));
    return seq;
}

void CommandQueue::take_due(uint64_t tick, std::vector<Batch> &out) {
    std::lock_guard<std::mutex> guard(lock_);
    // Batches for a later tick stay, in order, at the front of what is left.
    std::deque<Batch> later;
    for (Batch &batch : queue_) {
        if (batch.at_tick <= tick)
            out.push_back(std::move(batch));
        else
            later.push_back(std::move(batch));
    }
    queue_.swap(later);
}

bool CommandQueue::empty() const {
    std::lock_guard<std::mutex> guard(lock_);
    return queue_.empty();
}

uint64_t CommandQueue::last_seq() const {
    std::lock_guard<std::mutex> guard(lock_);
    return seq_;
}

}  // namespace pj
