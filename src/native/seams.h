// Enhanced internal edge removal across the seams between static bodies.
//
// Jolt removes a contact with an inner edge only among the faces of one body
// pair: a ball rolling from one static box onto another laid flush beside it
// still meets the second box's edge and is kicked up. For a moving body with
// enhanced internal edge removal, the pair's collision is made against the
// static body and every static body beside it that the moving body collides
// with, through one InternalEdgeRemovingCollector, and only the pair's own
// contacts are kept: an edge of the pair's body that lies on a face the moving
// body rests on, of whichever body, is then no contact.
#pragma once

#include "common.h"
#include "layers.h"

#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/PhysicsSystem.h>

namespace pj {

// A PhysicsSystem::SimCollideBodyVsBody for `system`: Jolt's own collision,
// with the seams between static bodies removed for a moving body that has
// enhanced internal edge removal.
void collide_across_seams(const JPH::PhysicsSystem &system, const JPH::Body &moving, const JPH::Body &other,
                          JPH::Mat44Arg moving_transform, JPH::Mat44Arg other_transform,
                          JPH::CollideShapeSettings &settings, JPH::CollideShapeCollector &collector,
                          const JPH::ShapeFilter &filter);

}  // namespace pj
