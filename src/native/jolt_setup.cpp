#include "common.h"

#include <Jolt/Core/Factory.h>
#include <Jolt/RegisterTypes.h>

#include <mutex>

namespace pj {

void register_jolt() {
    static std::once_flag once;
    std::call_once(once, [] {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    });
}

}  // namespace pj
