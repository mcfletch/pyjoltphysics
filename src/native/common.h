// Names every part of the native world shares: body uids, motion types and
// Jolt's one-time setup.
#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/MotionType.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace pj {

// A body's uid is assigned when its add batch is submitted and is never
// reused within one world. 0 names no body.
using Uid = uint32_t;
constexpr Uid kNoBody = 0;

// Codes shared with the Python enums; MotionType follows JPH::EMotionType.
enum class Motion : uint8_t { Static = 0, Kinematic = 1, Dynamic = 2 };

inline JPH::EMotionType to_jolt(Motion motion) {
    return static_cast<JPH::EMotionType>(static_cast<uint8_t>(motion));
}

inline bool valid_motion(uint8_t code) { return code <= static_cast<uint8_t>(Motion::Dynamic); }

// Per-body flags, a bit field shared with pyjoltphysics.BodyFlag.
namespace flag {
constexpr uint32_t kSensor = 1u << 0;       // reports overlaps and makes no contact response
constexpr uint32_t kNoSleep = 1u << 1;      // never falls asleep
constexpr uint32_t kContinuous = 1u << 2;   // linear-cast motion quality (CCD)
constexpr uint32_t kReport = 1u << 3;       // persist events for this body's pairs
constexpr uint32_t kMovable = 1u << 4;      // a static body that may become kinematic or dynamic
constexpr uint32_t kSmoothEdges = 1u << 5;  // Jolt's enhanced internal edge removal
// A static mesh or height field has enhanced internal edge removal unless it
// carries this flag.
constexpr uint32_t kNoSmoothEdges = 1u << 6;
constexpr uint32_t kAll = (1u << 7) - 1;
// A character's proxy body; set by the world, never by a batch.
constexpr uint32_t kCharacter = 1u << 8;
}  // namespace flag

// Jolt's body user data: the uid in the low 32 bits and the flags above it,
// so a contact listener reads both without a table lookup.
inline uint64_t user_data(Uid uid, uint32_t flags) {
    return uint64_t(uid) | (uint64_t(flags) << 32);
}
inline Uid uid_of(uint64_t user) { return Uid(user & 0xffffffffu); }
inline uint32_t flags_of(uint64_t user) { return uint32_t(user >> 32); }

// Jolt's allocator hooks, factory and type registry are process-wide and are
// set up once, by whichever thread first needs them.
void register_jolt();

// A world used after a fork, in the child process.
struct ForkedError : std::runtime_error {
    ForkedError() : std::runtime_error("this world belongs to the parent process of a fork") {}
};

}  // namespace pj
