#pragma once

namespace jvk {

class Renderer;

// =============================================================================
// FrameRetained — mixin for objects whose pointer/handles get captured into
// the deferred command stream
//
// jvk records draw commands on the message thread and replays them on a
// worker thread that submits to the GPU. Most recorded payloads are POD
// (hashes, transforms, indices into per-frame caches), but some carry raw
// pointers to user-owned objects (e.g. jvk::Shader in DrawShaderParams).
// Without coordination, a user destroying such an object on the message
// thread between record and execute leaves the worker dereferencing freed
// memory; even after execute returns, the GPU may still be consuming the
// object's VkPipeline / VkDescriptorSet from a submitted command buffer.
//
// Inheriting from FrameRetained gives the object two guarantees:
//
//   1. Renderer::retain() bumps an in-flight counter when the object is
//      pinned during record. The natural unpin runs only after the GPU
//      has finished the frame that pinned it (driven by the per-slot fence
//      already waited on inside Renderer::execute()).
//
//   2. The destructor blocks until the counter reaches zero. If the
//      counter is already zero (steady state), it returns immediately. If
//      not (the destructor was called between render ticks while pins
//      from the last frame are still held), it triggers a synchronous
//      Renderer::forceDrainAll — every live Renderer's worker is idled,
//      every unique Device is vkDeviceWaitIdle'd, and every pin is
//      dropped. This avoids the deadlock that would otherwise occur when
//      destruction itself runs on the message thread (no future render
//      tick is going to fire and drain the pins via the normal path).
//
// To use: inherit publicly from FrameRetained, and call waitUntilUnretained()
// at the TOP of your derived destructor — before tearing down any GPU
// handles. The base destructor calls it again as a safety net, but by then
// the derived class has already destroyed its handles, so the explicit
// up-front call is what actually keeps you safe.
// =============================================================================

class FrameRetained {
public:
    virtual ~FrameRetained();

protected:
    FrameRetained() = default;
    FrameRetained(const FrameRetained&) = delete;
    FrameRetained& operator=(const FrameRetained&) = delete;

    // Call as the FIRST line of the derived destructor, before destroying
    // any GPU resources. If the in-flight counter is already zero, this
    // is a single atomic load. If not, it forces every live Renderer to
    // synchronously drain its retain ring — guaranteeing that no command
    // buffer the GPU might still be consuming references this object's
    // handles. Bounded wait: at most one vkDeviceWaitIdle per unique
    // Device (typically a few ms), paid only on the first non-zero-pin
    // destruction in a batch (subsequent destructors see count==0).
    void waitUntilUnretained() const noexcept;

public:
    // True iff any Renderer currently holds a pin on this object. Safe to
    // call concurrently; observed value may race with a worker-thread
    // unpin but never with a pin (pins only happen on the single JUCE
    // message thread via Renderer::retain). Used by cache eviction to
    // skip entries still referenced by any editor's in-flight frame
    // instead of synchronously draining the GPU.
    bool isPinned() const noexcept
    {
        return inFlight_.load(std::memory_order_acquire) != 0;
    }

    // Durable pin/unpin — increments/decrements the in-flight counter
    // directly, outside the per-frame retain ring. Use when the owner
    // needs to hold a reference for longer than a single frame rotation.
    // Example: Shader stashing a cached Image's VkImageView+VkSampler
    // into its descriptor set must keep the CachedImage alive past the
    // next eviction tick, which Renderer::retain's slot-based rotation
    // would not do. Must be balanced with an equal number of unpin()
    // calls before the object is destroyed (otherwise ~FrameRetained
    // blocks in waitUntilUnretained forever — forceDrainAll only drops
    // frame-scoped pins, not durable ones).
    void pin()   noexcept { inFlight_.fetch_add(1, std::memory_order_acq_rel); }
    void unpin() noexcept { inFlight_.fetch_sub(1, std::memory_order_acq_rel); }

private:
    friend class Renderer;

    mutable std::atomic<uint32_t> inFlight_ { 0 };
};

} // namespace jvk
