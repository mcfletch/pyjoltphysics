#include "characters.h"

#include "layers.h"
#include "snapshot.h"

#include <Jolt/Core/Color.h>
#include <Jolt/Geometry/Plane.h>
#include <Jolt/Geometry/RayAABox.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace pj {

namespace {

const JPH::Vec3 kUp = JPH::Vec3::sAxisY();

// How far past the capsule a body still counts as touched, metres.
constexpr float kTouchSkin = 0.02f;
// A teleport that leaves the capsule this far inside a body leaves it stuck.
constexpr float kStuckDepth = 0.01f;
constexpr uint8_t kFlagStuck = 1;
// The raise of the capsule tested for standing up, so the floor it stands on
// is not in the way.
constexpr float kStandClearance = 0.02f;

JPH::RefConst<JPH::Shape> capsule(float height, float radius) {
    const float half = std::max(0.5f * height - radius, 1e-3f);
    JPH::RotatedTranslatedShapeSettings settings(JPH::Vec3(0.0f, half + radius, 0.0f),
                                                 JPH::Quat::sIdentity(),
                                                 new JPH::CapsuleShape(half, radius));
    return settings.Create().Get();
}

JPH::Vec3 flat(JPH::Vec3Arg v) { return JPH::Vec3(v.GetX(), 0.0f, v.GetZ()); }

double flat_length(JPH::Vec3Arg v) { return std::hypot(double(v.GetX()), double(v.GetZ())); }

}  // namespace

struct Slide {
    bool active = false;
    bool steer = false;
    double duration = 0, rise = 0, elapsed = 0, nominal = 0;
};

class CharacterLayerFilter final : public JPH::ObjectLayerFilter {
public:
    CharacterLayerFilter(const CollisionLayers &layers, uint32_t group, uint32_t mask)
        : layers_(layers), group_(group), mask_(mask) {}
    bool ShouldCollide(JPH::ObjectLayer layer) const override { return layers_.meets(layer, group_, mask_); }

private:
    const CollisionLayers &layers_;
    uint32_t group_, mask_;
};

// A character meets solid bodies only: not sensors, not character proxies
// (characters meet each other through their islands), not what it ignores.
class CharacterBodyFilter final : public JPH::BodyFilter {
public:
    explicit CharacterBodyFilter(const std::vector<Uid> &ignore) : ignore_(ignore) {}
    bool ShouldCollideLocked(const JPH::Body &body) const override {
        const uint64_t user = body.GetUserData();
        if (body.IsSensor() || (flags_of(user) & flag::kCharacter))
            return false;
        return ignore_.empty() || !std::binary_search(ignore_.begin(), ignore_.end(), uid_of(user));
    }

private:
    const std::vector<Uid> &ignore_;
};

struct Character {
    Uid id = kNoBody;
    CharacterCaps caps;
    double gravity = 9.81;       // this tick's: the character's own, as the zones leave it
    double own_gravity = 9.81;   // m/s2 downward, as added or set
    uint32_t group = 1, mask = 0xffffffffu, pass_group = 0;
    std::vector<Uid> ignore;
    JPH::Ref<JPH::CharacterVirtual> cv;
    JPH::BodyID proxy;
    float proxy_height = 0.0f;
    CharacterLayerFilter layer_filter;
    CharacterBodyFilter body_filter;

    // The controller's state, as omi_physics names it.
    double vy = 0.0;
    JPH::Vec3 push = JPH::Vec3::sZero();       // carried horizontal velocity
    JPH::Vec3 walk = JPH::Vec3::sZero();       // the input's velocity, after acceleration
    JPH::Vec3 move_dir = JPH::Vec3::sZero();
    JPH::Vec3 fly_dir = JPH::Vec3::sZero();
    JPH::Vec3 heading = JPH::Vec3::sZero();
    JPH::Vec3 wanted = JPH::Vec3::sZero();     // this tick's horizontal velocity asked for
    MoveMode mode = MoveMode::Walk;
    double move_speed = std::numeric_limits<double>::quiet_NaN();
    bool grounded = false, crouching = false, flying = false, swimming = false, down = false;
    double buoyancy = 0.0;
    JPH::Vec3 swim_velocity = JPH::Vec3::sZero();
    double swim_contact = 0.0;
    double airborne_for = -1.0;     // seconds since falling off the ground; negative: none
    double jump_wanted_for = -1.0;  // seconds since a refused jump; negative: none
    Slide slide;
    double pose_height = std::numeric_limits<double>::quiet_NaN();
    double shape_height = 0.0;
    float landing_speed = 0.0f;
    uint32_t landed_tick = 0;
    JPH::Vec3 velocity = JPH::Vec3::sZero();
    JPH::Vec3 ground_normal = JPH::Vec3::sZero();
    Uid ground_uid = kNoBody;
    bool pushing = false;  // walked into a body it can push, last tick
    bool teleported = true;  // the proxy is put, not moved, where the character is
    bool stuck = false;      // the last teleport left the capsule inside geometry
    JPH::RVec3 previous = JPH::RVec3::sZero();  // capsule centre at the tick before
    std::vector<Uid> touching;  // ascending
    std::vector<Event> events;

    Character(const CollisionLayers &layers, uint32_t g, uint32_t m)
        : layer_filter(layers, g, m), body_filter(ignore) {}

    double height() const {
        if (!std::isnan(pose_height))
            return pose_height;
        return crouching ? caps.crouch_height : caps.stand_height;
    }
    JPH::RVec3 feet() const { return cv->GetPosition(); }

    void event(EventPhase phase, uint32_t tick, Uid other = kNoBody, float approach = 0.0f,
               JPH::RVec3Arg point = JPH::RVec3::sZero(), JPH::Vec3Arg normal = JPH::Vec3::sZero()) {
        Event e{};
        e.kind = uint8_t(EventKind::Character);
        e.phase = uint8_t(phase);
        e.tick = tick;
        e.a = id;
        e.b = other;
        e.point[0] = point.GetX();
        e.point[1] = point.GetY();
        e.point[2] = point.GetZ();
        e.normal[0] = normal.GetX();
        e.normal[1] = normal.GetY();
        e.normal[2] = normal.GetZ();
        e.approach = approach;
        events.push_back(e);
    }

    // -- omi_physics' controller, method by method ------------------------------

    double speed() const {
        const bool steered = slide.active && slide.steer;
        if (crouching && !steered)
            return std::isnan(move_speed) ? caps.crouch_speed : std::min(move_speed, double(caps.crouch_speed));
        if (!std::isnan(move_speed))
            return move_speed;
        switch (mode) {
        case MoveMode::Sprint:
            return caps.can_sprint ? caps.sprint_speed : caps.walk_speed;
        case MoveMode::Run:
            return caps.run_speed;
        case MoveMode::Crouch:
            return caps.crouch_speed;
        case MoveMode::Walk:
            break;
        }
        return caps.walk_speed;
    }

    JPH::Vec3 intent() const {
        if (down || (slide.active && !slide.steer))
            return JPH::Vec3::sZero();
        return move_dir * float(speed());
    }

    double slope_cos() const { return std::cos(double(caps.max_slope) * JPH::JPH_PI / 180.0); }
    bool walkable(JPH::Vec3Arg normal) const { return double(normal.GetY()) >= slope_cos(); }

    // A horizontal velocity turned to run along the ground, at the same speed.
    JPH::Vec3 along_ground(JPH::Vec3Arg velocity) const {
        const JPH::Vec3 n = ground_normal;
        if (!grounded || !walkable(n))
            return velocity;
        const double speed = velocity.Length();
        if (speed <= 1e-9 || std::abs(n.GetY()) <= 1e-6f)
            return velocity;
        const double climb = -(double(n.GetX()) * velocity.GetX() + double(n.GetZ()) * velocity.GetZ()) / n.GetY();
        const JPH::Vec3 along(velocity.GetX(), float(climb), velocity.GetZ());
        const double reach = along.Length();
        return reach <= 1e-9 ? JPH::Vec3(velocity) : along * float(speed / reach);
    }

    bool may_jump() const { return caps.can_jump && !crouching && !flying && !down; }
    bool can_launch() const {
        if (grounded)
            return true;
        return caps.coyote_time > 0.0f && airborne_for >= 0.0 && airborne_for <= caps.coyote_time;
    }
    void launch(uint32_t tick) {
        vy = std::sqrt(2.0 * gravity * caps.jump_height);
        if (caps.jump_follows_slope && grounded)
            vy += double(along_ground(flat(intent() + push)).GetY());
        grounded = false;
        airborne_for = -1.0;
        jump_wanted_for = -1.0;
        event(EventPhase::Jump, tick, kNoBody, float(vy), feet());
    }
    void jump(uint32_t tick) {
        if (!may_jump())
            return;
        if (can_launch()) {
            launch(tick);
            return;
        }
        if (caps.jump_buffer > 0.0f)
            jump_wanted_for = 0.0;
    }
    void tick_jump_windows(double dt, bool was_grounded, uint32_t tick) {
        if (grounded)
            airborne_for = -1.0;
        else if (was_grounded)
            airborne_for = 0.0;
        else if (airborne_for >= 0.0)
            airborne_for += dt;
        if (jump_wanted_for < 0.0)
            return;
        if (grounded && may_jump()) {
            launch(tick);
            return;
        }
        jump_wanted_for += dt;
        if (jump_wanted_for > caps.jump_buffer)
            jump_wanted_for = -1.0;
    }

    // Advances vy by one step of gravity; returns the height gained, exactly.
    double fall(double dt) {
        const double g = gravity;
        const double v0 = vy;
        const double limit = caps.terminal_velocity > 0.0f ? double(caps.terminal_velocity)
                                                           : std::numeric_limits<double>::infinity();
        const double v1 = v0 - g * dt;
        if (v1 >= -limit) {
            vy = v1;
            return v0 * dt - 0.5 * g * dt * dt;
        }
        const double reach = g > 0.0 ? std::clamp((v0 + limit) / g, 0.0, dt) : 0.0;
        vy = -limit;
        return v0 * reach - 0.5 * g * reach * reach - limit * (dt - reach);
    }

    float slide_progress() const {
        if (!slide.active || slide.nominal <= 0.0)
            return 0.0f;
        const double remaining = slide.duration - slide.elapsed;
        const double held = remaining > slide.rise ? std::min(slide.elapsed, slide.nominal - slide.rise)
                                                   : slide.nominal - remaining;
        return float(std::clamp(held / slide.nominal, 0.0, 1.0));
    }

    CharacterState state() const {
        if (flying)
            return CharacterState::Flying;
        if (swimming)
            return CharacterState::Swimming;
        if (down)
            return CharacterState::Down;
        if (!grounded)
            return CharacterState::Airborne;
        if (slide.active)
            return CharacterState::Sliding;
        if (crouching)
            return CharacterState::Crouching;
        return CharacterState::Standing;
    }
};

// Characters of one island meet each other; characters of one pass group do not.
class IslandCollision final : public JPH::CharacterVsCharacterCollision {
public:
    const std::vector<Character *> *members = nullptr;

    static const Character *of(const JPH::CharacterVirtual *cv) {
        return reinterpret_cast<const Character *>(cv->GetUserData());
    }
    bool meets(const Character &a, const Character &b) const {
        return &a != &b && !b.flying && (a.pass_group == 0 || a.pass_group != b.pass_group);
    }

    void CollideCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform,
                          const JPH::CollideShapeSettings &inCollideShapeSettings, JPH::RVec3Arg inBaseOffset,
                          JPH::CollideShapeCollector &ioCollector) const override {
        const JPH::Mat44 transform1 = inCenterOfMassTransform.PostTranslated(-inBaseOffset).ToMat44();
        const JPH::Shape *shape1 = inCharacter->GetShape();
        JPH::CollideShapeSettings settings = inCollideShapeSettings;
        const JPH::AABox bounds1 = shape1->GetWorldSpaceBounds(transform1, JPH::Vec3::sOne());
        const Character &self = *of(inCharacter);
        for (const Character *other : *members) {
            if (!meets(self, *other) || ioCollector.ShouldEarlyOut())
                continue;
            const JPH::CharacterVirtual *c = other->cv.GetPtr();
            const JPH::Mat44 transform2 = c->GetCenterOfMassTransform().PostTranslated(-inBaseOffset).ToMat44();
            settings.mMaxSeparationDistance = inCollideShapeSettings.mMaxSeparationDistance + c->GetCharacterPadding();
            JPH::AABox bounds2 = c->GetShape()->GetWorldSpaceBounds(transform2, JPH::Vec3::sOne());
            bounds2.ExpandBy(JPH::Vec3::sReplicate(settings.mMaxSeparationDistance));
            if (!bounds1.Overlaps(bounds2))
                continue;
            ioCollector.SetUserData(reinterpret_cast<JPH::uint64>(c));
            JPH::CollisionDispatch::sCollideShapeVsShape(shape1, c->GetShape(), JPH::Vec3::sOne(), JPH::Vec3::sOne(),
                                                         transform1, transform2, JPH::SubShapeIDCreator(),
                                                         JPH::SubShapeIDCreator(), settings, ioCollector);
        }
        ioCollector.SetUserData(0);
    }

    void CastCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform,
                       JPH::Vec3Arg inDirection, const JPH::ShapeCastSettings &inShapeCastSettings,
                       JPH::RVec3Arg inBaseOffset, JPH::CastShapeCollector &ioCollector) const override {
        const JPH::Mat44 transform1 = inCenterOfMassTransform.PostTranslated(-inBaseOffset).ToMat44();
        const JPH::ShapeCast shape_cast(inCharacter->GetShape(), JPH::Vec3::sOne(), transform1, inDirection);
        const JPH::Vec3 origin = shape_cast.mShapeWorldBounds.GetCenter();
        const JPH::Vec3 extents =
            shape_cast.mShapeWorldBounds.GetExtent() + JPH::Vec3::sReplicate(inShapeCastSettings.mExtraConvexRadius);
        JPH::ShapeCastSettings cast_settings = inShapeCastSettings;
        const Character &self = *of(inCharacter);
        for (const Character *other : *members) {
            if (!meets(self, *other) || ioCollector.ShouldEarlyOut())
                continue;
            const JPH::CharacterVirtual *c = other->cv.GetPtr();
            const JPH::Mat44 transform2 = c->GetCenterOfMassTransform().PostTranslated(-inBaseOffset).ToMat44();
            cast_settings.mExtraConvexRadius = inShapeCastSettings.mExtraConvexRadius + c->GetCharacterPadding();
            JPH::AABox bounds2 = c->GetShape()->GetWorldSpaceBounds(transform2, JPH::Vec3::sOne());
            bounds2.ExpandBy(extents + JPH::Vec3::sReplicate(c->GetCharacterPadding()));
            if (!JPH::RayAABoxHits(origin, inDirection, bounds2.mMin, bounds2.mMax))
                continue;
            ioCollector.SetUserData(reinterpret_cast<JPH::uint64>(c));
            JPH::CollisionDispatch::sCastShapeVsShapeWorldSpace(shape_cast, cast_settings, c->GetShape(),
                                                                JPH::Vec3::sOne(), {}, transform2,
                                                                JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(),
                                                                ioCollector);
        }
        ioCollector.SetUserData(0);
    }
};

// Decides which dynamic bodies a character pushes: those no heavier than its push mass.
struct CharacterSystem::Listener final : public JPH::CharacterContactListener {
    JPH::PhysicsSystem *system = nullptr;

    void decide(const JPH::CharacterVirtual *cv, const JPH::CharacterContact &contact,
                 JPH::CharacterContactSettings &settings) const {
        if (contact.mMotionTypeB != JPH::EMotionType::Dynamic)
            return;
        auto *character = reinterpret_cast<Character *>(cv->GetUserData());
        // Islands give each dynamic body to one job, so reading it without a lock is safe.
        const JPH::Body *body = system->GetBodyLockInterfaceNoLock().TryGetBody(contact.mBodyB);
        if (body == nullptr)
            return;
        const float inverse = body->GetMotionProperties()->GetInverseMass();
        const bool light = inverse > 0.0f && 1.0f / inverse <= character->caps.push_mass;
        settings.mCanReceiveImpulses = light;
        if (light && character->wanted.Dot(contact.mContactNormal) < 0.0f)
            character->pushing = true;
    }
    void OnContactAdded(const JPH::CharacterVirtual *cv, const JPH::CharacterContact &contact,
                        JPH::CharacterContactSettings &settings) override {
        decide(cv, contact, settings);
    }
    void OnContactPersisted(const JPH::CharacterVirtual *cv, const JPH::CharacterContact &contact,
                            JPH::CharacterContactSettings &settings) override {
        decide(cv, contact, settings);
    }
};

CharacterSystem::CharacterSystem(JPH::PhysicsSystem &system, const CollisionLayers &layers, int threads)
    : system_(system), layers_(layers), listener_(std::make_unique<Listener>()) {
    listener_->system = &system;
    for (int i = 0; i < std::max(threads, 1); ++i)
        temps_.push_back(std::make_unique<JPH::TempAllocatorImplWithMallocFallback>(1024u * 1024u));
}

CharacterSystem::~CharacterSystem() = default;

std::vector<JPH::BodyID> CharacterSystem::remove_all() {
    std::vector<JPH::BodyID> proxies;
    for (const auto &c : characters_)
        if (!c->proxy.IsInvalid())
            proxies.push_back(c->proxy);
    characters_.clear();
    return proxies;
}

const Character *CharacterSystem::find(Uid id) const {
    auto at = std::lower_bound(characters_.begin(), characters_.end(), id,
                               [](const std::unique_ptr<Character> &c, Uid v) { return c->id < v; });
    return at != characters_.end() && (*at)->id == id ? at->get() : nullptr;
}

Character *CharacterSystem::find(Uid id) {
    auto at = std::lower_bound(characters_.begin(), characters_.end(), id,
                               [](const std::unique_ptr<Character> &c, Uid v) { return c->id < v; });
    return at != characters_.end() && (*at)->id == id ? at->get() : nullptr;
}

namespace {

// The broad phase filter for a character: every tree.
const JPH::BroadPhaseLayerFilter kEveryTree;
const JPH::ShapeFilter kEveryShape;

bool set_shape(Character &c, double height, JPH::TempAllocator &temp, bool check) {
    if (height == c.shape_height)
        return true;
    const JPH::RefConst<JPH::Shape> shape = capsule(float(height), c.caps.radius);
    const float tolerance = check ? 0.02f : FLT_MAX;
    if (!c.cv->SetShape(shape, tolerance, kEveryTree, c.layer_filter, c.body_filter, kEveryShape, temp))
        return false;
    c.shape_height = height;
    return true;
}

// Whether a capsule of `height` on the character's feet, raised a little, overlaps a solid body.
bool blocked_at(const JPH::PhysicsSystem &system, const Character &c, double height) {
    const JPH::RefConst<JPH::Shape> shape = capsule(float(height), c.caps.radius);
    const JPH::RVec3 at = c.feet() + JPH::RVec3(0.0, kStandClearance + c.cv->GetCharacterPadding(), 0.0);
    JPH::CollideShapeSettings settings;
    settings.mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
    JPH::AnyHitCollisionCollector<JPH::CollideShapeCollector> hit;
    // CollideShape places the shape by its centre of mass.
    const JPH::RMat44 transform = JPH::RMat44::sTranslation(at).PreTranslated(shape->GetCenterOfMass());
    system.GetNarrowPhaseQueryNoLock().CollideShape(shape, JPH::Vec3::sOne(), transform, settings, at, hit, kEveryTree,
                                                    c.layer_filter, c.body_filter);
    return hit.HadHit();
}

// Sets the capsule to the pose's height: a slide's or a knock-down's, else crouching or standing.
void fit(const JPH::PhysicsSystem &system, Character &c, JPH::TempAllocator &temp) {
    (void)system;
    set_shape(c, c.height(), temp, false);
}

void end_slide(const JPH::PhysicsSystem &system, Character &c, JPH::TempAllocator &temp) {
    c.slide.active = false;
    c.crouching = blocked_at(system, c, c.caps.stand_height);
    c.pose_height = std::numeric_limits<double>::quiet_NaN();
    fit(system, c, temp);
}

// The slide's pose: low, then rising over its last `rise` seconds, waiting under a ceiling.
void advance_pose(const JPH::PhysicsSystem &system, Character &c, double dt, JPH::TempAllocator &temp) {
    Slide &s = c.slide;
    if (!s.active)
        return;
    s.elapsed += dt;
    const bool done = s.elapsed >= s.duration - 1e-9;
    const double into = s.elapsed - (s.duration - s.rise);
    const double fraction = done ? 1.0 : s.rise > 0.0 ? std::clamp(into / s.rise, 0.0, 1.0) : 0.0;
    const double low = c.caps.crouch_height, high = c.caps.stand_height;
    double height = low + (high - low) * fraction;
    if (height > c.height() && blocked_at(system, c, height))
        height = c.height();
    c.pose_height = height;
    set_shape(c, height, temp, false);
    if (!done)
        return;
    s.active = false;
    c.crouching = height < high - 1e-9;
    c.pose_height = std::numeric_limits<double>::quiet_NaN();
    fit(system, c, temp);
}

bool over_void(const JPH::PhysicsSystem &system, const Character &c) {
    const double r = c.caps.radius;
    const JPH::RVec3 base = c.feet();
    const double offsets[5][2] = {{0, 0}, {r, 0}, {-r, 0}, {0, r}, {0, -r}};
    for (const auto &o : offsets) {
        const JPH::RRayCast ray(base + JPH::RVec3(o[0], 0.0, o[1]), JPH::Vec3(0.0f, -1.0e4f, 0.0f));
        JPH::RayCastResult hit;
        if (system.GetNarrowPhaseQueryNoLock().CastRay(ray, hit, kEveryTree, c.layer_filter, c.body_filter))
            return false;
    }
    return true;
}

// The uid of what a character stands on: a body's uid, or another character's id.
Uid ground_uid_of(const Character &c) {
    const JPH::CharacterVirtual &cv = *c.cv;
    if (!cv.GetGroundBodyID().IsInvalid())
        return uid_of(cv.GetGroundUserData());
    // Another character's ground user data is that character's own: its record.
    if (cv.GetGroundState() != JPH::CharacterVirtual::EGroundState::InAir && cv.GetGroundUserData() != 0)
        return reinterpret_cast<const Character *>(cv.GetGroundUserData())->id;
    return kNoBody;
}

// Touches of bodies and of other characters. A begin's approach is the
// closing speed along the normal; its impulse is the momentum a dynamic body
// brings at that speed, for a caller deciding a knockback.
void track_touches(const JPH::PhysicsSystem &system, Character &c, uint32_t tick) {
    std::vector<std::pair<Uid, const JPH::CharacterContact *>> now;
    for (const JPH::CharacterContact &contact : c.cv->GetActiveContacts()) {
        if (contact.mWasDiscarded || contact.mDistance >= kTouchSkin)
            continue;
        if (!contact.mBodyB.IsInvalid())
            now.emplace_back(uid_of(contact.mUserData), &contact);
        else if (!contact.mCharacterIDB.IsInvalid())
            now.emplace_back(Uid(contact.mCharacterIDB.GetValue()), &contact);
    }
    std::sort(now.begin(), now.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    now.erase(std::unique(now.begin(), now.end(), [](const auto &a, const auto &b) { return a.first == b.first; }),
              now.end());
    std::vector<Uid> uids;
    uids.reserve(now.size());
    for (const auto &[uid, contact] : now) {
        uids.push_back(uid);
        if (!std::binary_search(c.touching.begin(), c.touching.end(), uid)) {
            const float approach =
                std::max(0.0f, (contact->mLinearVelocity - c.velocity).Dot(contact->mContactNormal));
            c.event(EventPhase::Begin, tick, uid, approach, contact->mPosition, -contact->mContactNormal);
            if (contact->mMotionTypeB == JPH::EMotionType::Dynamic && !contact->mBodyB.IsInvalid()) {
                // Islands give each dynamic body to one job, so reading it without a lock is safe.
                if (const JPH::Body *body = system.GetBodyLockInterfaceNoLock().TryGetBody(contact->mBodyB)) {
                    const float inverse = body->GetMotionProperties()->GetInverseMass();
                    if (inverse > 0.0f)
                        c.events.back().impulse = approach / inverse;
                }
            }
        }
    }
    for (Uid uid : c.touching)
        if (!std::binary_search(uids.begin(), uids.end(), uid))
            c.event(EventPhase::End, tick, uid);
    c.touching.swap(uids);
}

void step_one(const JPH::PhysicsSystem &system, Character &c, double dt, uint32_t tick, JPH::TempAllocator &temp) {
    const auto dt_f = float(dt);
    // Velocities handed to Jolt cover a distance in Jolt's float time step.
    const double per = 1.0 / double(dt_f);
    advance_pose(system, c, dt, temp);
    const JPH::RVec3 start = c.feet();
    const bool was_pushing = c.pushing;
    c.pushing = false;
    if (c.flying) {
        const JPH::Vec3 move = c.down ? JPH::Vec3::sZero() : (c.move_dir + c.fly_dir) * c.caps.fly_speed;
        c.cv->SetPosition(start + JPH::RVec3(move) * dt);
        c.vy = 0.0;
        c.push = JPH::Vec3::sZero();
        c.velocity = move;
        return;
    }
    const JPH::Vec3 gravity(0.0f, float(-c.gravity), 0.0f);
    JPH::CharacterVirtual::ExtendedUpdateSettings update;
    update.mWalkStairsStepUp = JPH::Vec3(0.0f, c.caps.step_height, 0.0f);
    if (c.swimming) {
        c.grounded = false;
        c.ground_normal = JPH::Vec3::sZero();
        const JPH::Vec3 desired = c.down ? JPH::Vec3::sZero() : (c.move_dir + c.fly_dir) * c.caps.swim_speed;
        c.swim_velocity += (desired - c.swim_velocity) * std::min(1.0f, c.caps.swim_drag * dt_f);
        c.vy += (c.buoyancy - 1.0) * c.gravity * dt;
        if (c.swim_contact * c.vy > 0.0)
            c.vy = 0.0;
        c.vy *= std::max(0.0, 1.0 - c.caps.swim_drag * dt);
        c.push *= std::max(0.0f, 1.0f - c.caps.swim_drag * dt_f);
        const JPH::Vec3 velocity = c.swim_velocity + JPH::Vec3(0.0f, float(c.vy), 0.0f) + c.push;
        c.cv->SetLinearVelocity(velocity * float(dt * per));
        update.mStickToFloorStepDown = JPH::Vec3::sZero();
        update.mWalkStairsStepUp = JPH::Vec3::sZero();
        c.cv->ExtendedUpdate(dt_f, gravity, update, kEveryTree, c.layer_filter, c.body_filter, kEveryShape, temp);
        const double rose = c.feet().GetY() - start.GetY();
        if (std::abs(rose - velocity.GetY() * dt) > 1e-6) {
            c.swim_contact = velocity.GetY() > 0.0f ? 1.0 : -1.0;
            c.vy = 0.0;
        } else if (c.swim_contact * velocity.GetY() < 0.0) {
            c.swim_contact = 0.0;
        }
        c.velocity = JPH::Vec3(c.feet() - start) / dt_f;
        return;
    }
    const bool was_grounded = c.grounded;
    if (c.grounded)
        c.push *= std::max(0.0f, 1.0f - c.caps.push_friction * dt_f);
    const float control = c.grounded ? 1.0f : c.caps.air_control;
    const JPH::Vec3 target = c.intent() * control;
    if (std::isinf(c.caps.acceleration)) {
        c.walk = target;
    } else {
        const JPH::Vec3 change = target - c.walk;
        const float most = c.caps.acceleration * control * dt_f;
        const float size = change.Length();
        c.walk = size <= most ? target : c.walk + change * (most / size);
    }
    JPH::Vec3 horizontal = flat(c.walk + c.push);
    const float push_limit = std::isnan(c.caps.push_speed) ? c.caps.walk_speed : c.caps.push_speed;
    c.wanted = horizontal;
    if (horizontal.Length() > push_limit && was_pushing)
        horizontal *= push_limit / horizontal.Length();
    if (c.grounded)
        horizontal = c.along_ground(horizontal);
    JPH::Vec3 carried = JPH::Vec3::sZero();
    if (c.grounded) {
        c.cv->UpdateGroundVelocity();
        carried = c.cv->GetGroundVelocity();
    }
    if (c.caps.suppress_over_void && c.vy <= 0.0 && !c.grounded && over_void(system, c)) {
        c.vy = 0.0;
        c.cv->SetLinearVelocity(flat(horizontal) * float(dt * per));
        update.mStickToFloorStepDown = JPH::Vec3::sZero();
        c.cv->ExtendedUpdate(dt_f, gravity, update, kEveryTree, c.layer_filter, c.body_filter, kEveryShape, temp);
        c.velocity = JPH::Vec3(c.feet() - start) / dt_f;
        return;
    }
    const double v0 = c.vy;
    const double rise = c.fall(dt);
    const double impact = std::max(0.0, -c.vy);
    const JPH::Vec3 velocity =
        (horizontal + carried) * float(dt * per) + JPH::Vec3(0.0f, float(rise * per), 0.0f);
    c.cv->SetLinearVelocity(velocity);
    update.mStickToFloorStepDown = JPH::Vec3::sZero();
    c.cv->ExtendedUpdate(dt_f, gravity, update, kEveryTree, c.layer_filter, c.body_filter, kEveryShape, temp);
    // A walker that was on the ground and did not jump is held to it over a
    // drop of up to its step height. Jolt's own stick (in ExtendedUpdate)
    // passes over a walker climbing a slope whose crest it crosses, since
    // that walker is moving up.
    if (c.caps.stick_to_floor && was_grounded && v0 <= 1e-6 && !c.cv->IsSupported())
        c.cv->StickToFloor(JPH::Vec3(0.0f, -c.caps.step_height, 0.0f), kEveryTree, c.layer_filter, c.body_filter,
                           kEveryShape, temp);
    const JPH::RVec3 end = c.feet();
    const double moved_y = end.GetY() - start.GetY() - double(carried.GetY()) * dt;
    const auto ground = c.cv->GetGroundState();
    if (c.vy > 0.0) {
        c.grounded = false;
        c.ground_normal = JPH::Vec3::sZero();
        c.ground_uid = kNoBody;
        // A ceiling stops the rise.
        if (rise > 0.0 && moved_y < 0.5 * rise)
            c.vy = 0.0;
    } else if (ground == JPH::CharacterVirtual::EGroundState::OnGround) {
        c.grounded = true;
        c.vy = 0.0;
        c.ground_normal = c.cv->GetGroundNormal();
        c.ground_uid = ground_uid_of(c);
    } else {
        c.grounded = false;
        c.ground_uid = kNoBody;
        if (ground == JPH::CharacterVirtual::EGroundState::OnSteepGround) {
            // Sliding down a slope too steep to stand on: gravity goes on building
            // vy, and the slope turns the fall along it, at g sin(angle) along the slope.
            c.ground_normal = c.cv->GetGroundNormal();
        } else {
            c.ground_normal = JPH::Vec3::sZero();
        }
    }
    if (c.grounded && !was_grounded && impact > 0.0) {
        c.landing_speed = float(impact);
        c.landed_tick = tick;
        c.event(EventPhase::Land, tick, c.ground_uid, float(impact), c.cv->GetGroundPosition(), c.ground_normal);
    }
    c.tick_jump_windows(dt, was_grounded, tick);
    c.velocity = JPH::Vec3(end - start) / dt_f;
}

// Characters in an island that walked into another push it, sharing the
// closing speed by their masses.
void push_each_other(const std::vector<Character *> &members) {
    for (Character *a : members) {
        for (const JPH::CharacterContact &contact : a->cv->GetActiveContacts()) {
            if (contact.mCharacterIDB.IsInvalid() || contact.mDistance > kTouchSkin)
                continue;
            Character *b = nullptr;
            for (Character *m : members)
                if (m->cv->GetID() == contact.mCharacterIDB)
                    b = m;
            if (b == nullptr)
                continue;
            const JPH::Vec3 normal = flat(contact.mContactNormal).NormalizedOr(JPH::Vec3::sZero());
            const float closing = -(a->wanted - flat(b->velocity)).Dot(normal);
            if (closing <= 0.0f)
                continue;
            const float share = a->caps.mass / (a->caps.mass + b->caps.mass);
            b->push -= normal * (closing * share);
        }
    }
}

}  // namespace

void CharacterSystem::add(const AddCharacter &add, JPH::ObjectLayer proxy_layer, JPH::TempAllocator &temp) {
    auto made = std::make_unique<Character>(layers_, add.group, add.mask);
    Character &c = *made;
    c.id = add.uid;
    c.caps = add.caps;
    c.gravity = c.own_gravity = add.gravity;
    c.group = add.group;
    c.mask = add.mask;
    c.pass_group = add.pass_group;
    c.ignore = add.ignore;
    JPH::Ref<JPH::CharacterVirtualSettings> settings = new JPH::CharacterVirtualSettings();
    settings->mShape = capsule(c.caps.stand_height, c.caps.radius);
    settings->mSupportingVolume = JPH::Plane(kUp, -c.caps.radius);
    settings->mMaxSlopeAngle = JPH::DegreesToRadians(c.caps.max_slope);
    settings->mMass = c.caps.mass;
    settings->mMaxStrength = c.caps.push_force;
    settings->mEnhancedInternalEdgeRemoval = c.caps.smooth_edges;
    settings->mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
    settings->mID = JPH::CharacterID(add.uid);
    const JPH::RVec3 feet(add.position[0], add.position[1] - 0.5 * c.caps.stand_height, add.position[2]);
    c.cv = new JPH::CharacterVirtual(settings, feet, JPH::Quat::sIdentity(), reinterpret_cast<JPH::uint64>(&c),
                                     &system_);
    c.cv->SetListener(listener_.get());
    c.shape_height = c.caps.stand_height;
    c.previous = JPH::RVec3(add.position[0], add.position[1], add.position[2]);
    (void)temp;
    if (add.proxy) {
        JPH::BodyCreationSettings body(capsule(c.caps.stand_height, c.caps.radius), feet, JPH::Quat::sIdentity(),
                                       JPH::EMotionType::Kinematic, proxy_layer);
        // Not a sensor, so that sensors (static ones included) meet it; its
        // layer meets sensors and nothing else. Rays and sweeps meet it.
        body.mCollideKinematicVsNonDynamic = true;
        body.mAllowSleeping = false;
        body.mUserData = user_data(add.uid, flag::kCharacter);
        JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
        if (JPH::Body *proxy = bodies.CreateBody(body)) {
            c.proxy = proxy->GetID();
            c.proxy_height = c.caps.stand_height;
            bodies.AddBody(c.proxy, JPH::EActivation::Activate);
        }
    }
    auto at = std::lower_bound(characters_.begin(), characters_.end(), c.id,
                               [](const std::unique_ptr<Character> &x, Uid v) { return x->id < v; });
    characters_.insert(at, std::move(made));
}

std::vector<JPH::BodyID> CharacterSystem::remove(const std::vector<Uid> &uids) {
    std::vector<JPH::BodyID> proxies;
    for (Uid id : uids) {
        auto at = std::lower_bound(characters_.begin(), characters_.end(), id,
                                   [](const std::unique_ptr<Character> &x, Uid v) { return x->id < v; });
        if (at == characters_.end() || (*at)->id != id)
            continue;
        if (!(*at)->proxy.IsInvalid())
            proxies.push_back((*at)->proxy);
        characters_.erase(at);
    }
    return proxies;
}

void CharacterSystem::set_ignore(Uid id, std::vector<Uid> ignore) {
    if (Character *c = find(id))
        c->ignore = std::move(ignore);
}

bool CharacterSystem::apply(const CharacterOp &op, uint32_t tick, JPH::TempAllocator &temp) {
    Character *found = find(op.id);
    if (found == nullptr)
        return false;
    Character &c = *found;
    const JPH::Vec3 v3(float(op.v[0]), float(op.v[1]), float(op.v[2]));
    switch (static_cast<CharacterOpCode>(op.op)) {
    case CharacterOpCode::Move: {
        const JPH::Vec3 d = flat(v3);
        const double n = flat_length(d);
        c.move_dir = n > 1e-9 ? d / float(n) : JPH::Vec3::sZero();
        if (n > 1e-9)
            c.heading = c.move_dir;
        c.mode = static_cast<MoveMode>(op.code);
        c.move_speed = op.v[3];
        break;
    }
    case CharacterOpCode::Jump:
        c.jump(tick);
        break;
    case CharacterOpCode::Slide: {
        const double duration = op.v[0];
        if (duration <= 0.0 || !c.grounded || c.flying || c.swimming || c.down || c.slide.active)
            break;
        JPH::Vec3 momentum = flat(c.velocity);
        if (flat_length(momentum) < 1e-6)
            momentum = c.intent();
        c.push = op.code ? JPH::Vec3::sZero() : momentum;
        c.slide = Slide{true, op.code != 0, duration, std::clamp(op.v[1], 0.0, duration), 0.0, duration};
        c.crouching = true;
        c.pose_height = c.caps.crouch_height;
        set_shape(c, c.caps.crouch_height, temp, false);
        c.event(EventPhase::Slide, tick, kNoBody, 0.0f, c.feet());
        break;
    }
    case CharacterOpCode::ExtendSlide:
        if (c.slide.active) {
            c.slide.duration = c.slide.elapsed + std::max(op.v[0], 0.0);
            c.slide.rise = std::min(c.slide.rise, std::max(op.v[0], 0.0));
        }
        break;
    case CharacterOpCode::EndSlide:
        if (c.slide.active)
            end_slide(system_, c, temp);
        break;
    case CharacterOpCode::KnockDown:
        c.slide.active = false;
        c.down = true;
        c.push = JPH::Vec3::sZero();
        c.walk = JPH::Vec3::sZero();
        c.jump_wanted_for = -1.0;
        c.pose_height = c.caps.crouch_height;
        set_shape(c, c.caps.crouch_height, temp, false);
        c.event(EventPhase::Down, tick, kNoBody, 0.0f, c.feet());
        break;
    case CharacterOpCode::GetUp: {
        if (!c.down)
            break;
        c.down = false;
        double height = c.crouching ? c.caps.crouch_height : c.caps.stand_height;
        if (height > c.height() && blocked_at(system_, c, height)) {
            c.crouching = true;
            height = c.caps.crouch_height;
        }
        c.pose_height = std::numeric_limits<double>::quiet_NaN();
        set_shape(c, height, temp, false);
        if (op.v[0] > 0.0)
            c.push = c.heading * float(op.v[0]);
        c.event(EventPhase::Up, tick, kNoBody, 0.0f, c.feet());
        break;
    }
    case CharacterOpCode::Knockback: {
        const JPH::Vec3 change = v3 / c.caps.mass;
        c.push += flat(change);
        c.vy += double(change.GetY());
        if (c.vy > 0.0) {
            c.grounded = false;
            c.airborne_for = -1.0;
            c.jump_wanted_for = -1.0;
        }
        break;
    }
    case CharacterOpCode::Impulse:
        c.push = flat(v3);
        c.vy = op.v[1];
        c.grounded = false;
        c.airborne_for = -1.0;
        c.jump_wanted_for = -1.0;
        break;
    case CharacterOpCode::Crouch:
        if (c.slide.active)
            break;
        if (op.code && c.caps.can_crouch) {
            c.crouching = true;
            fit(system_, c, temp);
        } else if (!op.code && c.crouching && !blocked_at(system_, c, c.caps.stand_height)) {
            c.crouching = false;
            fit(system_, c, temp);
        }
        break;
    case CharacterOpCode::Fly:
        if (op.code && !c.caps.can_fly)
            break;
        c.flying = op.code != 0;
        if (c.flying) {
            c.grounded = false;
            c.airborne_for = -1.0;
            c.jump_wanted_for = -1.0;
        }
        break;
    case CharacterOpCode::FlyMove: {
        const double n = v3.Length();
        c.fly_dir = n > 1e-9 ? v3 / float(n) : JPH::Vec3::sZero();
        break;
    }
    case CharacterOpCode::Swim: {
        const bool was = c.swimming;
        c.swimming = op.code != 0;
        c.buoyancy = op.v[0];
        if (c.swimming != was) {
            c.vy = 0.0;
            c.swim_velocity = JPH::Vec3::sZero();
            c.swim_contact = 0.0;
        }
        if (c.swimming) {
            c.grounded = false;
            c.ground_normal = JPH::Vec3::sZero();
            c.airborne_for = -1.0;
            c.jump_wanted_for = -1.0;
        }
        break;
    }
    case CharacterOpCode::Teleport: {
        c.teleported = true;
        const JPH::RVec3 feet(op.v[0], op.v[1] - 0.5 * c.height(), op.v[2]);
        c.cv->SetPosition(feet);
        c.cv->SetLinearVelocity(JPH::Vec3::sZero());
        c.flying = false;
        c.push = JPH::Vec3::sZero();
        c.walk = JPH::Vec3::sZero();
        c.vy = 0.0;
        c.velocity = JPH::Vec3::sZero();
        c.grounded = false;
        c.ground_normal = JPH::Vec3::sZero();
        c.ground_uid = kNoBody;
        const double search = op.v[3];
        if (search > 0.0)
            c.cv->StickToFloor(JPH::Vec3(0.0f, float(-search), 0.0f), kEveryTree, c.layer_filter, c.body_filter,
                               kEveryShape, temp);
        c.cv->RefreshContacts(kEveryTree, c.layer_filter, c.body_filter, kEveryShape, temp);
        c.stuck = false;
        for (const JPH::CharacterContact &contact : c.cv->GetActiveContacts())
            if (!contact.mBodyB.IsInvalid() && contact.mDistance < -kStuckDepth)
                c.stuck = true;
        if (c.cv->GetGroundState() == JPH::CharacterVirtual::EGroundState::OnGround) {
            c.grounded = true;
            c.ground_normal = c.cv->GetGroundNormal();
            c.ground_uid = ground_uid_of(c);
        }
        break;
    }
    case CharacterOpCode::Gravity:
        c.gravity = c.own_gravity = op.v[0];
        break;
    case CharacterOpCode::PassGroup:
        c.pass_group = uint32_t(op.v[0]);
        break;
    case CharacterOpCode::Count:
        return false;
    }
    return true;
}

void CharacterSystem::islands(double dt) {
    const size_t n = characters_.size();
    std::vector<uint32_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0u);
    auto root = [&](uint32_t i) {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];
        return i;
    };
    auto join = [&](uint32_t a, uint32_t b) {
        a = root(a);
        b = root(b);
        if (a != b)
            parent[std::max(a, b)] = std::min(a, b);
    };
    // Each character's reach this tick: where its move, its stairs, its floor
    // snap and its contacts can take it.
    std::vector<JPH::AABox> reach(n);
    for (size_t i = 0; i < n; ++i) {
        const Character &c = *characters_[i];
        const double travel = (c.speed() + c.push.Length() + std::abs(c.vy) + c.gravity * dt +
                               (c.flying ? c.caps.fly_speed : 0.0f)) * dt;
        const float margin = float(0.3 + c.caps.step_height + travel);
        const JPH::Vec3 feet(c.feet());
        const float r = c.caps.radius;
        reach[i] = JPH::AABox(feet + JPH::Vec3(-r, 0.0f, -r) - JPH::Vec3::sReplicate(margin),
                              feet + JPH::Vec3(r, float(c.height()), r) + JPH::Vec3::sReplicate(margin));
    }
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return reach[a].mMin.GetX() < reach[b].mMin.GetX() || (reach[a].mMin.GetX() == reach[b].mMin.GetX() && a < b);
    });
    for (size_t k = 0; k < n; ++k)
        for (size_t j = k + 1; j < n && reach[order[j]].mMin.GetX() <= reach[order[k]].mMax.GetX(); ++j)
            if (reach[order[k]].Overlaps(reach[order[j]]))
                join(order[k], order[j]);
    // A dynamic body in the reach of two characters puts them in one island:
    // both may push it.
    std::unordered_map<uint32_t, uint32_t> owner;
    const JPH::BodyLockInterfaceNoLock &locks = system_.GetBodyLockInterfaceNoLock();
    for (uint32_t i = 0; i < n; ++i) {
        JPH::AllHitCollisionCollector<JPH::CollideShapeBodyCollector> hits;
        system_.GetBroadPhaseQuery().CollideAABox(reach[i], hits);
        for (const JPH::BodyID &id : hits.mHits) {
            const JPH::Body *body = locks.TryGetBody(id);
            if (body == nullptr || !body->IsDynamic())
                continue;
            auto [at, fresh] = owner.emplace(id.GetIndex(), i);
            if (!fresh)
                join(at->second, i);
        }
    }
    islands_.clear();
    island_of_.assign(n, 0xffffffffu);
    std::vector<uint32_t> index_of_root(n, 0xffffffffu);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t r = root(i);
        if (index_of_root[r] == 0xffffffffu) {
            index_of_root[r] = uint32_t(islands_.size());
            islands_.emplace_back();
        }
        islands_[index_of_root[r]].push_back(i);
    }
}

void CharacterSystem::step(double dt, uint32_t tick, JPH::JobSystem &jobs, std::vector<Event> &events) {
    if (characters_.empty())
        return;
    for (const auto &held : characters_) {
        Character &c = *held;
        c.previous = c.feet() + JPH::Vec3(0.0f, float(0.5 * c.height()), 0.0f);
        if (zones_ != nullptr && !zones_->empty())
            c.gravity = -double(zone_gravity(*zones_, c.previous, JPH::Vec3(0.0f, float(-c.own_gravity), 0.0f))
                                    .GetY());
        else
            c.gravity = c.own_gravity;
    }
    islands(dt);
    const size_t count = islands_.size();
    const size_t chunks = std::min(temps_.size(), count);
    // Contiguous runs of islands, balanced by the characters in them.
    std::vector<size_t> bounds{0};
    {
        const size_t total = characters_.size();
        size_t acc = 0;
        for (size_t k = 0; k < count && bounds.size() < chunks; ++k) {
            acc += islands_[k].size();
            if (acc * chunks >= total * bounds.size())
                bounds.push_back(k + 1);
        }
        bounds.push_back(count);
    }
    auto run = [&, dt, tick](size_t chunk) {
        JPH::TempAllocator &temp = *temps_[chunk];
        IslandCollision collision;
        std::vector<Character *> members;
        for (size_t k = bounds[chunk]; k < bounds[chunk + 1]; ++k) {
            members.clear();
            for (uint32_t i : islands_[k])
                members.push_back(characters_[i].get());
            collision.members = &members;
            for (Character *c : members) {
                c->cv->SetCharacterVsCharacterCollision(&collision);
                step_one(system_, *c, dt, tick, temp);
            }
            if (members.size() > 1)
                push_each_other(members);
            for (Character *c : members) {
                track_touches(system_, *c, tick);
                c->cv->SetCharacterVsCharacterCollision(nullptr);
            }
        }
    };
    if (bounds.size() <= 2) {
        run(0);
    } else {
        JPH::JobSystem::Barrier *barrier = jobs.CreateBarrier();
        for (size_t chunk = 0; chunk + 1 < bounds.size(); ++chunk)
            barrier->AddJob(jobs.CreateJob("characters", JPH::Color::sGreen, [&run, chunk] { run(chunk); }));
        jobs.WaitForJobs(barrier);
        jobs.DestroyBarrier(barrier);
    }
    // The proxies follow, in id order, once every character has moved.
    JPH::BodyInterface &bodies = system_.GetBodyInterfaceNoLock();
    for (const auto &held : characters_) {
        Character &c = *held;
        if (!c.proxy.IsInvalid()) {
            if (float(c.shape_height) != c.proxy_height) {
                bodies.SetShape(c.proxy, capsule(float(c.shape_height), c.caps.radius), false,
                                JPH::EActivation::Activate);
                c.proxy_height = float(c.shape_height);
            }
            // Moved over the step to where the character now is, so a sensor
            // sees it move; put there after a teleport.
            if (c.teleported)
                bodies.SetPositionAndRotation(c.proxy, c.feet(), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
            else
                bodies.MoveKinematic(c.proxy, c.feet(), JPH::Quat::sIdentity(), float(dt));
            c.teleported = false;
        }
        events.insert(events.end(), c.events.begin(), c.events.end());
        c.events.clear();
    }
}

void CharacterSystem::flush_events(std::vector<Event> &events) {
    for (const auto &held : characters_) {
        events.insert(events.end(), held->events.begin(), held->events.end());
        held->events.clear();
    }
}

void CharacterSystem::write(SnapshotBuffer &out) const {
    const size_t n = characters_.size();
    out.resize_characters(uint32_t(n));
    for (size_t i = 0; i < n; ++i) {
        const Character &c = *characters_[i];
        const JPH::RVec3 feet = c.feet();
        const double h = c.height();
        out.char_id[i] = c.id;
        out.char_position[3 * i] = feet.GetX();
        out.char_position[3 * i + 1] = feet.GetY() + 0.5 * h;
        out.char_position[3 * i + 2] = feet.GetZ();
        for (int k = 0; k < 3; ++k) {
            out.char_velocity[3 * i + k] = c.velocity[k];
            out.char_ground_normal[3 * i + k] = c.ground_normal[k];
        }
        out.char_state[i] = uint8_t(c.state());
        out.char_grounded[i] = c.grounded ? 1 : 0;
        out.char_ground[i] = c.ground_uid;
        out.char_slide[i] = c.slide_progress();
        out.char_landing[i] = c.landing_speed;
        out.char_landed_tick[i] = c.landed_tick;
        out.char_height[i] = float(h);
        for (int k = 0; k < 3; ++k)
            out.char_previous[3 * i + k] = c.previous[k];
        const double eye = double(c.caps.eye_height) * h / double(c.caps.stand_height);
        out.char_eye[3 * i] = feet.GetX();
        out.char_eye[3 * i + 1] = feet.GetY() + eye;
        out.char_eye[3 * i + 2] = feet.GetZ();
        uint16_t material = 0;
        if (c.grounded) {
            const JPH::PhysicsMaterial *ground = c.cv->GetGroundMaterial();
            const auto *indexed = dynamic_cast<const IndexedMaterial *>(ground);
            if (indexed != nullptr && indexed->index != 0)
                material = indexed->index;
            else if (!c.cv->GetGroundBodyID().IsInvalid())
                material = material_bits_of(c.cv->GetGroundUserData());
        }
        out.char_ground_material[i] = material;
        out.char_flags[i] = uint8_t((c.stuck ? kFlagStuck : 0));
    }
}

}  // namespace pj
