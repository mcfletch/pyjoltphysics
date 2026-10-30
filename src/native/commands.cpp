#include "commands.h"

#include <limits>

namespace pj {

uint32_t uids_wanted(const Command &command) {
    if (auto *add = std::get_if<AddBodies>(&command))
        return uint32_t(add->shapes.size());
    if (auto *swap = std::get_if<SwapBodies>(&command))
        return uint32_t(swap->add.shapes.size());
    if (std::holds_alternative<AddCharacter>(command))
        return 1;
    return 0;
}

uint64_t CommandQueue::push(Batch &&batch, uint32_t uid_count, Uid *first_uid) {
    std::lock_guard<std::mutex> guard(lock_);
    if (uid_count) {
        if (uint64_t(next_uid_) + uid_count > std::numeric_limits<Uid>::max())
            throw std::length_error("the world has assigned every 32-bit body uid");
        std::vector<Uid> *uids = nullptr;
        if (auto *add = std::get_if<AddBodies>(&batch.command))
            uids = &add->uids;
        else if (auto *swap = std::get_if<SwapBodies>(&batch.command))
            uids = &swap->add.uids;
        if (uids != nullptr) {
            uids->resize(uid_count);
            for (uint32_t i = 0; i < uid_count; ++i)
                (*uids)[i] = next_uid_ + i;
        } else if (auto *character = std::get_if<AddCharacter>(&batch.command)) {
            character->uid = next_uid_;
        }
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
