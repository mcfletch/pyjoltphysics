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
    std::vector<double> char_previous;      // (n, 3) capsule centre at the tick before
    std::vector<double> char_eye;           // (n, 3) eye position, metres
    std::vector<uint16_t> char_ground_material;  // material index underfoot
    std::vector<uint8_t> char_flags;        // CharacterFlag bits

    // Joints, every row written each tick, in ascending id.
    uint32_t joints = 0;
    std::vector<uint32_t> joint_id;
    std::vector<uint8_t> joint_kind;
    std::vector<uint8_t> joint_state;
    std::vector<float> joint_value;   // hinge angle (rad), slider position (m), distance length (m)
    std::vector<float> joint_value2;  // SIXDOF: rotation about its y and z axes (rad) in value, value2
    std::vector<float> joint_force;   // N carried in the last step
    std::vector<float> joint_torque;  // N m carried in the last step

    // Vehicles, every row written each tick, in ascending id; their wheels in
    // vehicle order, each vehicle's from veh_first_wheel.
    uint32_t vehicles = 0;
    uint32_t wheels = 0;
    std::vector<uint32_t> veh_id;
    std::vector<Uid> veh_chassis;
    std::vector<uint8_t> veh_kind;
    std::vector<uint32_t> veh_first_wheel;
    std::vector<uint32_t> veh_wheels;
    std::vector<float> veh_rpm;
    std::vector<int8_t> veh_gear;            // -1 reverse, 0 neutral, 1 first
    std::vector<float> veh_forward_speed;    // m/s along the chassis's forward axis
    std::vector<float> veh_steer;            // steer angle handed to the wheels, rad, positive left
    std::vector<float> veh_impact_speed;     // closing speed of the tick's heaviest blow, m/s
    std::vector<float> veh_impact_impulse;   // its estimated impulse, N s
    std::vector<Uid> veh_impact_uid;         // the body struck; 0 for none
    std::vector<float> veh_input;            // (n, 4) throttle, steer, brake, handbrake
    std::vector<uint32_t> wheel_vehicle;     // row of the wheel's vehicle
    std::vector<double> wheel_position;      // (w, 3) wheel centre, metres
    std::vector<float> wheel_orientation;    // (w, 4) wheel model (axle x, up y) to world
    std::vector<float> wheel_length;         // suspension length, m
    std::vector<float> wheel_compression;    // suspension max length less its length, m
    std::vector<float> wheel_steer;          // rad, positive left
    std::vector<float> wheel_angle;          // rotation about the axle, rad
    std::vector<float> wheel_spin;           // rad/s, positive rolling forward
    std::vector<float> wheel_load;           // suspension force, N
    std::vector<float> wheel_slip;           // (w, 2) longitudinal slip ratio, lateral slip angle (rad)
    std::vector<uint8_t> wheel_contact;
    std::vector<Uid> wheel_ground;           // uid under the wheel; 0 for none
    std::vector<uint16_t> wheel_material;    // material index of the body under the wheel
    std::vector<int32_t> wheel_triangle;     // mesh triangle number under the wheel; -1 for none
    std::vector<float> wheel_grip;           // grip the surface gives the wheel
    std::vector<double> wheel_contact_point; // (w, 3)
    std::vector<float> wheel_contact_normal; // (w, 3)

    bool written = false;
    uint64_t written_stamp = 0;  // the publish that last wrote it

    void resize(uint32_t count);
    void resize_characters(uint32_t count);
    void resize_joints(uint32_t count);
    void resize_vehicles(uint32_t count, uint32_t wheel_count);
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
    std::deque<std::pair<uint64_t, std::vector<uint32_t>>> history_;  // (publish stamp, rows changed)
    uint64_t stamps_ = 0;
    std::vector<uint32_t> tick_changed_;
    std::vector<uint32_t> spare_;
    std::vector<uint64_t> mark_stamp_;
    std::vector<uint64_t> write_stamp_;
    uint64_t mark_epoch_ = 1;
    uint64_t write_epoch_ = 0;
};

}  // namespace pj
