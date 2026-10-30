// Snapshots: the state of every body after a tick, published into a pool of
// reference-counted buffers that Python reads without copying.
//
// Lifetime rule: the pool holds one reference to each buffer and the world
// holds one more to the latest. A reader's Snapshot holds another, and every
// NumPy view of it holds the Snapshot. The world thread writes into a buffer
// only when the pool's reference is the only one left, so a buffer a reader
// can see is never written; holding snapshots grows the pool and never blocks
// the world thread.
//
// Every row of a buffer is current. A buffer that is reused is brought up to
// date from a history of the rows each tick changed, or rewritten whole when
// it is older than the history.
#pragma once

#include "common.h"

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace pj {

struct SnapshotBuffer {
    uint64_t tick = 0;
    double time = 0.0;
    uint64_t applied_seq = 0;
    uint32_t rows = 0;
    std::vector<Uid> uid;               // (rows,)
    std::vector<double> position;       // (rows, 3) metres
    std::vector<float> orientation;     // (rows, 4) x, y, z, w
    std::vector<float> linear_velocity;   // (rows, 3) m/s
    std::vector<float> angular_velocity;  // (rows, 3) rad/s
    std::vector<uint8_t> awake;         // (rows,) 0 or 1
    std::vector<uint8_t> motion;        // (rows,) Motion
    std::vector<uint32_t> changed;      // ascending rows written by this tick

    // Characters, every row written each tick, in ascending id.
    uint32_t characters = 0;
    std::vector<Uid> char_id;
    std::vector<double> char_position;      // (n, 3) capsule centre, metres
    std::vector<float> char_velocity;       // (n, 3) m/s over the last tick
    std::vector<uint8_t> char_state;        // CharacterState
    std::vector<uint8_t> char_grounded;
    std::vector<Uid> char_ground;           // uid of the body underfoot; 0 for none
    std::vector<float> char_ground_normal;  // (n, 3)
    std::vector<float> char_slide;          // slide progress, 0 to 1
    std::vector<float> char_landing;        // speed of the last landing, m/s
    std::vector<uint32_t> char_landed_tick; // tick of the last landing; 0 for none
    std::vector<float> char_height;         // capsule height, metres

    bool written = false;
    uint64_t written_tick = 0;

    void resize(uint32_t count);
    void resize_characters(uint32_t count);
};

// Owned by the world thread.
class SnapshotPublisher {
public:
    static constexpr size_t kHistory = 16;

    // Records that `row` changed during the current tick.
    void mark(uint32_t row);
    // Fills a free buffer with the rows as of `tick`, calling `write(row,
    // buffer)` for each row that differs from what the buffer holds.
    std::shared_ptr<SnapshotBuffer> publish(
        uint64_t tick, double time, uint64_t applied_seq, uint32_t rows,
        const std::function<void(uint32_t, SnapshotBuffer &)> &write);
    size_t pool_size() const { return pool_.size(); }

private:
    std::vector<std::shared_ptr<SnapshotBuffer>> pool_;
    std::deque<std::pair<uint64_t, std::vector<uint32_t>>> history_;
    std::vector<uint32_t> tick_changed_;
    std::vector<uint32_t> spare_;
    std::vector<uint64_t> mark_stamp_;
    std::vector<uint64_t> write_stamp_;
    uint64_t mark_epoch_ = 1;
    uint64_t write_epoch_ = 0;
};

}  // namespace pj
