#include "snapshot.h"

#include <algorithm>
#include <atomic>

namespace pj {

void SnapshotBuffer::resize(uint32_t count) {
    rows = count;
    uid.resize(count);
    position.resize(size_t(count) * 3);
    orientation.resize(size_t(count) * 4);
    linear_velocity.resize(size_t(count) * 3);
    angular_velocity.resize(size_t(count) * 3);
    awake.resize(count);
    motion.resize(count);
}

void SnapshotBuffer::resize_characters(uint32_t count) {
    characters = count;
    char_id.resize(count);
    char_position.resize(size_t(count) * 3);
    char_velocity.resize(size_t(count) * 3);
    char_state.resize(count);
    char_grounded.resize(count);
    char_ground.resize(count);
    char_ground_normal.resize(size_t(count) * 3);
    char_slide.resize(count);
    char_landing.resize(count);
    char_landed_tick.resize(count);
    char_height.resize(count);
    char_previous.resize(size_t(count) * 3);
    char_eye.resize(size_t(count) * 3);
    char_ground_material.resize(count);
    char_flags.resize(count);
}

void SnapshotBuffer::resize_joints(uint32_t count) {
    joints = count;
    joint_id.resize(count);
    joint_kind.resize(count);
    joint_state.resize(count);
    joint_value.resize(count);
    joint_value2.resize(count);
    joint_force.resize(count);
    joint_torque.resize(count);
}

void SnapshotPublisher::mark(uint32_t row) {
    if (mark_stamp_.size() <= row)
        mark_stamp_.resize(size_t(row) + 1024, 0);
    if (mark_stamp_[row] == mark_epoch_)
        return;
    mark_stamp_[row] = mark_epoch_;
    tick_changed_.push_back(row);
}

std::shared_ptr<SnapshotBuffer> SnapshotPublisher::publish(
    uint64_t tick, double time, uint64_t applied_seq, uint32_t rows,
    const std::function<void(uint32_t, SnapshotBuffer &)> &write) {
    ++mark_epoch_;
    // History and buffers count publishes, not ticks: a flush publishes again within a tick.
    const uint64_t stamp = ++stamps_;
    if (history_.size() >= kHistory) {
        spare_ = std::move(history_.front().second);
        history_.pop_front();
    }
    std::sort(tick_changed_.begin(), tick_changed_.end());
    history_.emplace_back(stamp, std::move(tick_changed_));
    tick_changed_ = std::move(spare_);
    tick_changed_.clear();

    std::shared_ptr<SnapshotBuffer> buffer;
    for (auto &candidate : pool_) {
        if (candidate.use_count() == 1) {
            buffer = candidate;
            break;
        }
    }
    if (buffer) {
        // The last reader dropped its reference on another thread; its reads
        // of this buffer happen before the writes below.
        std::atomic_thread_fence(std::memory_order_acquire);
    } else {
        buffer = std::make_shared<SnapshotBuffer>();
        pool_.push_back(buffer);
    }

    // The oldest tick the history covers: a buffer written before it misses
    // changes and is rewritten whole.
    const uint64_t covered_from = history_.front().first;
    const bool whole = !buffer->written || buffer->written_stamp + 1 < covered_from;
    buffer->resize(rows);
    if (whole) {
        for (uint32_t row = 0; row < rows; ++row)
            write(row, *buffer);
    } else {
        ++write_epoch_;
        if (write_stamp_.size() < rows)
            write_stamp_.resize(rows, 0);
        for (const auto &[changed_stamp, changed_rows] : history_) {
            if (changed_stamp <= buffer->written_stamp)
                continue;
            for (uint32_t row : changed_rows) {
                if (row < rows && write_stamp_[row] != write_epoch_) {
                    write_stamp_[row] = write_epoch_;
                    write(row, *buffer);
                }
            }
        }
    }
    const auto &latest = history_.back().second;
    buffer->changed.clear();
    for (uint32_t row : latest)
        if (row < rows)
            buffer->changed.push_back(row);
    buffer->written = true;
    buffer->written_stamp = stamp;
    buffer->tick = tick;
    buffer->time = time;
    buffer->applied_seq = applied_seq;
    return buffer;
}

}  // namespace pj
