namespace jvk {

// =============================================================================
// Construction / destruction — starts and stops the worker thread.
// =============================================================================

// Process-wide registry of live Renderer instances. Maintained on the
// message thread (Renderer ctor/dtor are message-thread-only) but read
// from FrameRetained destructors that may run on the same thread; the
// CriticalSection is there for hygiene rather than contention.
juce::CriticalSection& Renderer::registryLock()
{
    static juce::CriticalSection lk;
    return lk;
}

std::vector<Renderer*>& Renderer::registry()
{
    static std::vector<Renderer*> r;
    return r;
}

int Renderer::liveCount()
{
    const juce::ScopedLock lk(registryLock());
    return static_cast<int>(registry().size());
}

Renderer::Renderer(Device& device, RenderTarget& target)
    : device_(device), target_(target),
      vertices_(device.physicalDevice(), device.device()),
      staging_(device.physicalDevice(), device.device()),
      gradientAtlas_(std::make_unique<GradientAtlas>()),
      worker_(std::make_unique<Worker>(*this))
{
    // Init the per-Renderer gradient atlas now that Device is ready (it
    // allocates a backing Image from device.pool() and a descriptor set
    // from device.bindings()).
    gradientAtlas_->init(device);

    {
        const juce::ScopedLock lk(registryLock());
        registry().push_back(this);
    }
    worker_->startThread();
}

// ---- Gradient-atlas accessors (out-of-line; header forward-declares type) --

float Renderer::registerGradient(const juce::ColourGradient& g)
{
    return gradientAtlas_->getRow(g, ResourceCaches::hashGradient(g));
}

VkDescriptorSet Renderer::gradientDescriptor() const
{
    return gradientAtlas_->descriptorSet();
}

void Renderer::reset()
{
    commands_.clear();
    arena_.reset();
    fonts_.clear();
    fills_.clear();
    // Per-frame gradient atlas reset. Runs on the message thread (called from
    // the editor's render timer) — same thread that does registerGradient
    // during paint, so no sync needed.
    gradientAtlas_->beginFrame();
}

// Worker is the last-declared member, so under default destruction it would
// be the first destroyed (stopThread + notify + join), letting its final
// execute() run against still-valid Renderer members. We need an explicit
// body so we can also drop ourselves from the global registry FIRST (so
// no FrameRetained dtor running concurrently can see us mid-teardown),
// then join the worker, then drain the FrameRetained pin counts. The
// caller is contractually required to have vkDeviceWaitIdle'd before
// destroying us (see teardownVulkan), so the GPU is also done with
// everything we're about to release the pin on.
Renderer::~Renderer()
{
    {
        const juce::ScopedLock lk(registryLock());
        auto& r = registry();
        r.erase(std::remove(r.begin(), r.end(), this), r.end());
    }
    if (worker_) worker_.reset();
    flushRetains();
    // Return slot-parked staging blocks to the belt so ~L2 frees them —
    // Block is a plain handle struct, so destroying the vectors alone would
    // leak the VkBuffer/VkDeviceMemory. GPU is idle per the dtor contract.
    // (retiredImages vectors destroy via ~Image below, same contract.)
    for (auto& slot : frameSlots_)
        staging_.recycle(slot.staging);
}

// =============================================================================
// Threaded-execute control
// =============================================================================

void Renderer::submit()
{
    // Release here publishes every CPU-side write made during record (command
    // list, arena, path SSBO, mapped Vulkan buffers) to the worker's acquire
    // load at the top of execute().
    workerBusy_.store(true, std::memory_order_release);
    if (worker_) worker_->notify();
}

void Renderer::waitForIdle()
{
    // Used only for resize and teardown — both rare. Yield rather than burn
    // a CPU core; the wait is bounded by one execute duration (~16 ms on
    // VSync, longer only under DWM drag stalls).
    while (workerBusy_.load(std::memory_order_acquire))
        juce::Thread::yield();
}

void Renderer::flushRetains()
{
    // Gather under the lock, unpin outside it — unpin is an atomic decrement
    // but keeping the critical section minimal costs nothing.
    std::vector<FrameRetained*> pins;
    {
        const juce::ScopedLock lk(uploadLock_);
        for (auto& slot : frameSlots_)
        {
            pins.insert(pins.end(), slot.pins.begin(), slot.pins.end());
            slot.pins.clear();
        }
        pins.insert(pins.end(), recordingRetains_.begin(), recordingRetains_.end());
        recordingRetains_.clear();
    }
    for (auto* obj : pins) obj->unpin();
}

void Renderer::forceDrainAll()
{
    // Snapshot the registry under lock so we don't iterate while a
    // sibling Renderer ctor/dtor mutates it. After this point we operate
    // on the snapshot — any Renderer destroyed concurrently has already
    // removed itself, so the snapshot can only contain still-live ones
    // (the assumption is registry mutation runs on the message thread,
    // same as our caller — destructors can't interleave on a single
    // thread).
    std::vector<Renderer*> snapshot;
    {
        const juce::ScopedLock lk(registryLock());
        snapshot = registry();
    }

    // Phase 1: idle every worker. Once each workerBusy_ goes false, no
    // further pin/unpin will happen via execute() — we have a stable
    // view of every FrameSlot pin bucket.
    for (auto* r : snapshot) r->waitForIdle();

    // Phase 2: vkDeviceWaitIdle once per unique Device. After this every
    // submission referencing pinned objects has retired on the GPU, so
    // dropping the pins below can't strand the GPU mid-sample of a
    // freed VkPipeline / VkDescriptorSet.
    std::set<VkDevice> seen;
    for (auto* r : snapshot)
    {
        VkDevice d = r->device().device();
        if (d != VK_NULL_HANDLE && seen.insert(d).second)
        {
            // Device-idle is queue-use on every queue — same external-sync
            // rule as submits. The workers are idle (phase 1) but ANOTHER
            // message-thread path (ShaderImage, submitImmediate) could be
            // mid-submit.
            const juce::ScopedLock queueSync(queueLock());
            vkDeviceWaitIdle(d);
        }
    }

    // Phase 3: drop every pin. After this the in-flight counter on every
    // FrameRetained is zero and the destructor that called us can fall
    // straight through.
    for (auto* r : snapshot) r->flushRetains();
}

// =============================================================================
// FrameRetained — out-of-line so waitUntilUnretained can call into Renderer
// without forcing FrameRetained.h to know Renderer's full definition.
// =============================================================================

FrameRetained::~FrameRetained()
{
    waitUntilUnretained();
}

void FrameRetained::waitUntilUnretained() const noexcept
{
    if (inFlight_.load(std::memory_order_acquire) == 0) return;

    // Pins are still held — typically because the destructor is running
    // on the message thread between render ticks, and the per-slot
    // drain inside Renderer::execute() can't fire because we're the
    // very thread that would have queued the next frame. Force every
    // live Renderer to drain synchronously (worker idle + GPU idle +
    // flushRetains) so the count drops to zero.
    Renderer::forceDrainAll();

    // Safety net: forceDrainAll should have brought us to zero. Spin
    // briefly just in case a Renderer joined the registry after our
    // snapshot and pinned us in a frame already in flight (extremely
    // unlikely on the single-message-thread plugin model, but cheap to
    // cover).
    while (inFlight_.load(std::memory_order_acquire) > 0)
        juce::Thread::yield();
}

// =============================================================================
// Deferred uploads (L2 staging → image / buffer, flushed into the frame's
// command buffer just before the scene render pass).
// =============================================================================

void Renderer::upload(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height)
{
    const juce::ScopedLock lk(uploadLock_);
    staging_.commit(src);   // allocation now visible to the flush that records it
    pendingUploads_.push_back({ dst, width, height, src.buffer, src.offset });
}

void Renderer::uploadDynamic(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height)
{
    const juce::ScopedLock lk(uploadLock_);
    staging_.commit(src);
    pendingUploads_.push_back({ dst, width, height, src.buffer, src.offset, true });
}

void Renderer::uploadRect(Memory::L2::Allocation src, VkImage dst,
                          int32_t x, int32_t y, uint32_t width, uint32_t height)
{
    const juce::ScopedLock lk(uploadLock_);
    staging_.commit(src);
    pendingUploads_.push_back({ dst, width, height, src.buffer, src.offset,
                                /*dynamicContent*/ false, /*partial*/ true, x, y });
}

void Renderer::cancelUploads(VkImage dst)
{
    const juce::ScopedLock lk(uploadLock_);
    std::erase_if(pendingUploads_,
                  [dst](const PendingUpload& u) { return u.dstImage == dst; });
}

// Drop queued uploads for `dst` on EVERY live Renderer. Used by ~Shader,
// which has no Renderer back-pointer: a queued dynamic-feed upload must not
// survive the destruction of its target image, or the next flushUploads
// records vkCmdCopyBufferToImage into a freed VkImage.
void Renderer::cancelUploadsAllRenderers(VkImage dst)
{
    std::vector<Renderer*> snapshot;
    {
        const juce::ScopedLock lk(registryLock());
        snapshot = registry();
    }
    for (auto* r : snapshot) r->cancelUploads(dst);
}

void Renderer::recordImageUpload(VkCommandBuffer cmd, Memory::L2::Allocation src,
                                  VkImage dst, uint32_t width, uint32_t height,
                                  bool dynamicContent, bool partial,
                                  int32_t dstX, int32_t dstY)
{
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // Partial uploads must PRESERVE the untouched texels, so the transition
    // must name the image's actual layout (SHADER_READ_ONLY) — UNDEFINED
    // permits the driver to discard contents. Full uploads rewrite every
    // texel, so UNDEFINED is fine (and cheaper) there.
    barrier.oldLayout = partial ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                : VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dst;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // Fresh images (cache inserts) have never been read — TOP_OF_PIPE orders
    // nothing and lets the copy overlap prior GPU work. Dynamic re-uploads
    // and partial updates of live images must wait for every previously
    // submitted frame's fragment reads of this image (pipeline barriers
    // order against ALL earlier commands on the queue, across command
    // buffers). Write-after-read only needs the execution dependency, so
    // srcAccessMask stays 0.
    vkCmdPipelineBarrier(cmd,
        (dynamicContent || partial) ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region {};
    region.bufferOffset = src.offset;      // bufferRowLength 0 = tightly packed
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset = { dstX, dstY, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(cmd, src.buffer, dst,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void Renderer::flushUploads(VkCommandBuffer cmd)
{
    // Swap the queue out under the lock, record from the local copy — the
    // message thread can keep queueing next frame's uploads while we record.
    // uploadScratch_ is a member so its capacity survives across frames.
    {
        const juce::ScopedLock lk(uploadLock_);
        uploadScratch_.swap(pendingUploads_);

        // Park staging blocks against THIS slot inside the SAME lock hold as
        // the queue swap. A block is parked only when (a) it is not the
        // current write target and (b) it has no uncommitted allocations —
        // an alloc whose matching upload() push hasn't landed yet keeps its
        // block active, so a message-thread alloc+memcpy straddling this
        // swap can never have its block filed one frame early and recycled
        // under a copy that still reads it (the old moveActiveTo-in-execute
        // had exactly that window). upload()/uploadDynamic() commit the
        // allocation, clearing the hold.
        staging_.parkAllButCurrent(frameSlots_[activeRetireSlot_].staging);
    }

    for (auto& u : uploadScratch_) {
        // The pending-upload struct carries the L2 allocation broken out into
        // (srcBuffer, srcOffset); repack so we can share recordImageUpload.
        Memory::L2::Allocation src { nullptr, u.srcBuffer, u.srcOffset };
        recordImageUpload(cmd, src, u.dstImage, u.width, u.height,
                          u.dynamicContent, u.partial, u.dstX, u.dstY);
    }
    uploadScratch_.clear();
}

// =============================================================================
// Pipeline registration
// =============================================================================

void Renderer::registerPipeline(Pipeline& pipeline)
{
    auto ops = pipeline.supportedOps();
    for (auto op : ops)
        pipelineForOp_[static_cast<size_t>(op)] = &pipeline;
}

// =============================================================================
// Execute — sort + beginFrame + flush + render pass + replay + endFrame
// =============================================================================

void Renderer::execute()
{
    auto frame = target_.beginFrame();
    if (frame.cmd == VK_NULL_HANDLE) {
        // Swapchain acquire returned VK_NULL_HANDLE — typically
        // VK_ERROR_OUT_OF_DATE_KHR during a resize / DPI-change / window
        // minimise. Pins and STATIC uploads must survive the skip:
        // `getTexture` inserts a CachedImage into the process-wide texture
        // cache, pins it, and queues its upload — dropping the pin lets a
        // sibling editor's evict destroy the entry while its VkImage sits
        // in our queue (use-after-free at the next flushUploads), and
        // dropping the upload leaves the cache entry permanently
        // UNDEFINED-layout (UB when a later hit samples it).
        //
        // What must NOT survive unboundedly is the per-tick refresh work: a
        // minimised window on Windows sits in this state at 60 Hz, and each
        // tick a dynamic feed (Shader::update) stages a fresh full-image
        // block. Unbounded, that grew L2 by ~8 MB per tick (~500 MB/s) —
        // the same append-only-belt failure the slot rotation fixed on the
        // happy path. Bound it: only the LATEST dynamic upload per image
        // matters (each rewrites every texel), so drop superseded entries
        // and hand their never-recorded staging blocks straight back to the
        // free list. Static uploads (cache inserts, glyph rects — bounded
        // by content, possibly partial-image) all survive untouched.
        const juce::ScopedLock lk(uploadLock_);

        if (!pendingUploads_.empty()) {
            auto* base = pendingUploads_.data();
            auto  n    = pendingUploads_.size();
            std::vector<PendingUpload> kept;
            kept.reserve(n);
            for (size_t i = 0; i < n; i++) {
                auto& u = base[i];
                if (u.dynamicContent) {
                    bool superseded = false;
                    for (size_t j = i + 1; j < n; j++)
                        if (base[j].dynamicContent && base[j].dstImage == u.dstImage) {
                            superseded = true;
                            break;
                        }
                    if (superseded) continue;
                }
                kept.push_back(u);
            }
            pendingUploads_.swap(kept);

            // Blocks created since the last flush whose every allocation is
            // committed and unreferenced by the surviving queue were never
            // recorded into any command buffer — recycle them immediately.
            std::vector<VkBuffer> referenced;
            referenced.reserve(pendingUploads_.size());
            for (auto& u : pendingUploads_) referenced.push_back(u.srcBuffer);
            staging_.recycleUnrecorded(referenced);
        }
        return;
    }

    // target_.beginFrame() above waited on this slot's fence, so the GPU
    // has now finished every command buffer it submitted the LAST time this
    // slot was used. Drain the slot's ENTIRE lifetime bucket in one place:
    // unpin FrameRetained objects, destroy retired images, recycle staging
    // blocks. Swap out under the lock, release outside it. Subsequent
    // retire()/retain()/staging traffic during this execute goes into THIS
    // slot's bucket and isn't touched until the slot next comes around.
    {
        const juce::ScopedLock lk(uploadLock_);
        activeRetireSlot_ = frame.frameSlot % kFrameSlots;
        auto& slot = frameSlots_[activeRetireSlot_];
        flushScratch_.pins.swap(slot.pins);
        flushScratch_.retiredImages.swap(slot.retiredImages);
        flushScratch_.staging.swap(slot.staging);
    }
    for (auto* obj : flushScratch_.pins) obj->unpin();
    flushScratch_.pins.clear();
    flushScratch_.retiredImages.clear();       // ~Image frees the GPU handles
    staging_.recycle(flushScratch_.staging);   // returns blocks to the free list

    // Snapshot the device clock once per frame so every shader dispatched
    // during this execute() sees an identical `time` value — guarantees
    // deterministic per-frame motion and prevents drift between e.g. two
    // DrawShader ops in the same frame reading slightly different clocks.
    const float frameTime = device_.time();

    vertices_.beginFrame(frame.frameSlot);

    // Pipeline prepare (atlas dirty pages, etc.) and gradient/texture uploads.
    // Each pipeline pushes its pending uploads onto *this* Renderer's queue,
    // not Device's — so two editors' workers never share the upload vector.
    {
        Pipeline* seen[static_cast<size_t>(DrawOp::COUNT)] = {};
        int n = 0;
        for (auto* p : pipelineForOp_) {
            if (!p) continue;
            bool dup = false;
            for (int i = 0; i < n; i++) if (seen[i] == p) { dup = true; break; }
            if (!dup) { seen[n++] = p; p->prepare(*this); }
        }
    }
    gradientAtlas_->stageUploads(*this);
    // Flush record-phase path/clip segments into THIS frame slot's SSBO.
    // Safe now because target_.beginFrame() above waited on this slot's
    // fence, so the GPU is done reading this slot's buffer from 2 frames
    // ago. The OTHER slot's buffer may still be in flight (frame N-1)
    // and is not touched.
    if (pathPipeline_) pathPipeline_->flushToGPU(frame.frameSlot);
    // flushUploads records the queued copies AND parks the consumed staging
    // blocks against this slot under one lock hold (see body).
    flushUploads(frame.cmd);

    auto const& sb = target_.sceneBuffers(frame.frameSlot);

    // =========================================================================
    // Ping-pong model (synchronous, one active buffer at a time).
    //
    //   pp[0] is half A, pp[1] is half B. `cur` is the index of the half
    //   that is currently the active scene target. Normal draws write pp[cur].
    //
    //   When a post-process effect runs:
    //     1. End the active scene RP (finalLayout transitions pp[cur] color
    //        to SHADER_READ_ONLY; stencil transitions to DS_ATT_OPTIMAL).
    //     2. Run the effect pass in its own RP: sample pp[cur] color, write
    //        pp[1-cur] color.
    //     3. Swap: cur ^= 1. The active half is now the one we just wrote.
    //     4. Resume scene RP LOAD on pp[cur].sceneFB. Stencil image is the
    //        SAME physical image (shared between framebufferA and
    //        framebufferB) so LOAD_OP_LOAD restores the clip state we left.
    //
    //   Separable effects (Gaussian blur) do two effect passes and therefore
    //   two swaps — they naturally land back on the half they started on.
    //   Non-separable effects (HSV, 2D-kernel blurs): FULL-window regions do
    //   one pass + one swap (cheapest — dst is fully covered; pre-copy seeds
    //   outside-clip when clipped); PARTIAL regions do a region pass plus a
    //   region-bounded identity copy-back so pixels outside the ROI keep the
    //   original scene without any full-screen seeding pass.
    //
    //   At end of frame we blit pp[cur] → swapchain. There is never a race
    //   between the effect writing and the scene RP LOAD because they target
    //   DIFFERENT halves; the scene RP LOAD reads the half the effect just
    //   wrote, whose writes are made visible by the effect RP's outgoing
    //   subpass dependency.
    // =========================================================================
    struct Half {
        VkFramebuffer   sceneFB;
        VkFramebuffer   effectFB;
        VkDescriptorSet sampler;
        VkImage         image;
    };
    Half pp[2] = {
        { sb.framebufferA, sb.effectFBtoA, sb.samplerA, sb.colorA.image() },
        { sb.framebufferB, sb.effectFBtoB, sb.samplerB, sb.colorB.image() },
    };
    int cur = 0;

    VkRenderPass scenePassLoad = target_.sceneRenderPassLoad();
    VkRenderPass sceneBuildRP  = target_.sceneRenderPassClear(); // compat for build

    auto beginSceneRP = [&](VkRenderPass rp, bool withClears) {
        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = rp;
        rpbi.framebuffer = pp[cur].sceneFB;
        rpbi.renderArea.extent = frame.extent;

        VkClearValue clears[2] {};
        clears[0].color = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
        clears[1].depthStencil = { 1.0f, 0 };
        if (withClears) {
            rpbi.clearValueCount = 2;
            rpbi.pClearValues = clears;
        }
        vkCmdBeginRenderPass(frame.cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp {};
        vp.width    = static_cast<float>(frame.extent.width);
        vp.height   = static_cast<float>(frame.extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &vp);
        VkRect2D sc {}; sc.extent = frame.extent;
        vkCmdSetScissor(frame.cmd, 0, 1, &sc);

        state_.invalidate();
    };

    // Runs a single post-process pass (caller provides the dispatch via
    // `applyPass`) and swaps the active half. Must be called between a
    // vkCmdEndRenderPass and the matching resume of the scene RP.
    auto effectPassAndSwap = [&](auto applyPass) {
        int dst = cur ^ 1;
        applyPass(pp[cur].sampler, pp[dst].effectFB);
        cur = dst;
    };

    // ---- Effect ROI (region-of-interest back-propagation) -----------------
    //
    // The standardized per-pass read-margin walk: each pass's scissor is the
    // NEXT pass's input requirement — the final output region expanded by
    // each intermediate pass's kernel apron — clamped to the framebuffer.
    // Every chain is arranged to end on the half it STARTED on (even number
    // of swaps; single-pass effects append a region-bounded identity
    // copy-back), so pixels outside the region / outside the clip simply
    // keep the original scene in place. That deletes the old FULL-SCREEN
    // pre-copy seeding passes: a small blurred knob now costs region-sized
    // passes instead of 2+ full-screen ones.

    // Final output region: recorded effect region ∩ recorded clip. Empty
    // region param = whole clip (legacy / inverted blurs).
    auto roiFinal = [](const juce::Rectangle<float>& region,
                       const juce::Rectangle<int>& clip) {
        auto c = clip.toFloat();
        return region.isEmpty() ? c : region.getIntersection(c);
    };

    // Rect (+ per-axis apron) → framebuffer-clamped scissor.
    auto roiScissor = [&](juce::Rectangle<float> r, float padX, float padY) -> VkRect2D {
        r = r.expanded(padX, padY);
        int x0 = std::max(0, (int) std::floor(r.getX()));
        int y0 = std::max(0, (int) std::floor(r.getY()));
        int x1 = std::min((int) frame.extent.width,  (int) std::ceil(r.getRight()));
        int y1 = std::min((int) frame.extent.height, (int) std::ceil(r.getBottom()));
        if (x1 <= x0 || y1 <= y0) return { { 0, 0 }, { 0, 0 } };
        return { { x0, y0 },
                 { static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0) } };
    };

    // Separable chains still need outside-clip pixels of the INTERMEDIATE
    // half seeded with source (pass 2's taps cross the clip edge and read
    // pass 1's stencil-discarded pixels otherwise) — but only within the
    // intermediate pass's scissor, not full-screen.
    auto preCopyIfClipped = [&](uint8_t stencilDepth, const VkRect2D* scissor) {
        if (stencilDepth == 0 || !copyEffect_) return;
        int dst = cur ^ 1;
        copyEffect_->applyPass(frame.cmd,
            pp[cur].sampler, pp[dst].effectFB, target_.effectRenderPass(),
            frame.extent,
            /*dir unused*/ 0.0f, 0.0f,
            /*radius unused*/ 0.0f,
            /*stencilRef*/ stencilDepth,
            scissor);
    };

    // Region-bounded identity pass (HSV with identity constants): copies the
    // effect's output back onto the original half so the chain's swap count
    // is even. Stencil-inside like the effect itself, so outside-clip pixels
    // of the ORIGINAL half are never touched.
    //
    // ONLY for PARTIAL regions. A full-framebuffer single-pass effect fully
    // covers the destination half (pre-copy seeds outside-clip when
    // clipped), so the old single-swap behavior is correct there — and the
    // copy-back would DOUBLE the cost of every full-window saturate/hue
    // (measured as a general slowdown in the benchmark). `cur` ending on
    // either half is fine: the blit reads pp[cur].
    auto copyBackAndSwap = [&](const VkRect2D& sc, uint8_t stencilDepth) {
        effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
            hsvPipeline_->applyPass(frame.cmd, src, dst,
                target_.effectRenderPass(), frame.extent,
                HSVPipeline::identity(),
                static_cast<uint32_t>(stencilDepth), &sc);
        });
    };
    auto coversFrame = [&](const VkRect2D& sc) {
        return sc.offset.x == 0 && sc.offset.y == 0
            && sc.extent.width == frame.extent.width
            && sc.extent.height == frame.extent.height;
    };

    beginSceneRP(target_.sceneRenderPassClear(), /*withClears=*/true);
    state_.begin(frame.cmd, vertices_,
        static_cast<float>(frame.extent.width),
        static_cast<float>(frame.extent.height));

    for (auto& cmd : commands_) {
        if (cmd.op == DrawOp::EffectKernel) {
            // Separable Gaussian blur, ROI-walked: V (last) outputs the
            // final region R, so H must output R padded by V's VERTICAL
            // apron; H's own horizontal apron reads the true scene (always
            // valid). Two swaps land `cur` back on the starting half, so
            // pixels outside R keep the original scene untouched.
            if (postProcess_) {
                auto& bp = arena_.read<BlurParams>(cmd.dataOffset);
                const auto R = roiFinal(bp.region, cmd.clipBounds);
                if (R.isEmpty()) continue;
                const float m = std::ceil(bp.radius * std::max(1.0f, bp.scale)) + 2.0f;
                const VkRect2D scH = roiScissor(R, 0.0f, m);
                const VkRect2D scV = roiScissor(R, 0.0f, 0.0f);

                vkCmdEndRenderPass(frame.cmd);
                // Seed the intermediate half's outside-clip pixels within
                // H's scissor (V's taps cross the clip edge and read them).
                preCopyIfClipped(cmd.stencilDepth, &scH);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    postProcess_->applyPass(frame.cmd, src, dst,
                        target_.effectRenderPass(), frame.extent,
                        1.0f, 0.0f, bp.radius,
                        static_cast<uint32_t>(cmd.stencilDepth), &scH);
                });
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    postProcess_->applyPass(frame.cmd, src, dst,
                        target_.effectRenderPass(), frame.extent,
                        0.0f, 1.0f, bp.radius,
                        static_cast<uint32_t>(cmd.stencilDepth), &scV);
                });
                beginSceneRP(scenePassLoad, /*withClears=*/false);
            }
            continue;
        }
        if (cmd.op == DrawOp::PushClipRect) {
            // cmd.clipBounds is already the pixel-space intersection of the
            // new rect and the existing clip stack (computed at record
            // time in Graphics::clipToRectangle). State tracks this as the
            // current scissor bounds.
            state_.pushClipRect(cmd.clipBounds);
            continue;
        }
        if (cmd.op == DrawOp::PopClipRect) {
            state_.popClipRect();
            continue;
        }
        if (cmd.op == DrawOp::PushClipPath) {
            // Analytical-SDF clip push — fragment shader discards outside the
            // clip shape, stencil INCR_WRAP where inside at stencil == parentDepth.
            if (clipPipeline_ && pathPipeline_) {
                auto& p = arena_.read<ClipShapeParams>(cmd.dataOffset);
                ClipPipeline::PushConstants pc {};
                pc.shapeType    = p.shapeType;
                pc.centerX      = p.centerX;
                pc.centerY      = p.centerY;
                pc.halfW        = p.halfW;
                pc.halfH        = p.halfH;
                pc.cornerRadius = p.cornerRadius;
                pc.segmentStart = p.segmentStart;
                pc.segmentCount = p.segmentCount;
                pc.fillRule     = p.fillRule;
                pc.stripCount   = p.stripCount;
                pc.stripMinY    = p.stripMinY;
                pc.invStripH    = p.invStripH;
                clipPipeline_->pushClip(state_, frame.cmd, *this, cmd,
                    pc, p.coverRect,
                    pathPipeline_->ssboDescriptorSet(),
                    static_cast<uint32_t>(cmd.stencilDepth),
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height));
            }
            state_.pushStencilDepth();
            continue;
        }
        if (cmd.op == DrawOp::PopClipPath) {
            // Analytical-SDF clip pop — DECR_WRAP at stencil == currentDepth
            // (before the CPU decrement). cmd.stencilDepth here is the depth
            // one level DEEPER than parent (i.e. the push's target depth),
            // so stencil == that value is exactly what the INCR produced.
            if (clipPipeline_ && pathPipeline_) {
                auto& p = arena_.read<ClipShapeParams>(cmd.dataOffset);
                ClipPipeline::PushConstants pc {};
                pc.shapeType    = p.shapeType;
                pc.centerX      = p.centerX;
                pc.centerY      = p.centerY;
                pc.halfW        = p.halfW;
                pc.halfH        = p.halfH;
                pc.cornerRadius = p.cornerRadius;
                pc.segmentStart = p.segmentStart;
                pc.segmentCount = p.segmentCount;
                pc.fillRule     = p.fillRule;
                pc.stripCount   = p.stripCount;
                pc.stripMinY    = p.stripMinY;
                pc.invStripH    = p.invStripH;
                clipPipeline_->popClip(state_, frame.cmd, *this, cmd,
                    pc, p.coverRect,
                    pathPipeline_->ssboDescriptorSet(),
                    static_cast<uint32_t>(cmd.stencilDepth),
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height));
            }
            state_.popStencilDepth();
            continue;
        }
        if (cmd.op == DrawOp::FillPath) {
            // Analytical SDF path fill — dispatched via PathPipeline which
            // owns the segment storage buffer + the SDF fragment shader.
            // Stays inside the scene render pass (no RP transitions).
            if (pathPipeline_) {
                auto& p    = arena_.read<FillPathParams>(cmd.dataOffset);
                auto& fill = getFill(p.fillIndex);
                // The per-tile UIVertex span sits right after the params in
                // the arena, 4-byte aligned (Graphics::fillPath pushed it).
                uint32_t vertsOff =
                    (cmd.dataOffset + static_cast<uint32_t>(sizeof(FillPathParams)) + 3u) & ~3u;
                auto verts = arena_.readSpan<UIVertex>(vertsOff, p.vertexCount);
                // Colour source descriptor — matches the ColorPipeline pattern
                // (gradient atlas row for gradient fills, 1x1 default for
                // solids). path_sdf.frag samples it iff gradientInfo.z > 0.
                VkDescriptorSet colorDesc =
                    (fill.isGradient() && fill.gradient)
                        ? gradientDescriptor()
                        : caches().defaultDescriptor();
                pathPipeline_->dispatch(state_, frame.cmd, *this, cmd,
                    verts.data(), p.vertexCount,
                    p.segmentStart, p.fillRule,
                    colorDesc,
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height),
                    p.tileSize);
            }
            continue;
        }
        if (cmd.op == DrawOp::DrawShader) {
            // DrawShader is a regular scene draw — no render-pass transition.
            // Dispatch updates State's bound pipeline/layout and issues one
            // fullscreen triangle against the current scene framebuffer.
            if (shaderPipeline_) {
                auto& sp = arena_.read<DrawShaderParams>(cmd.dataOffset);
                auto* shader = static_cast<Shader*>(sp.shader);
                if (shader) {
                    shaderPipeline_->dispatch(state_, frame.cmd, *shader,
                        sp.region,
                        static_cast<float>(frame.extent.width),
                        static_cast<float>(frame.extent.height),
                        cmd.clipBounds,
                        cmd.stencilDepth,
                        frameTime);
                }
            }
            continue;
        }
        if (cmd.op == DrawOp::EffectHSV) {
            // Non-separable HSV, ROI-walked: one region-bounded transform
            // pass + one region-bounded identity copy-back so the chain ends
            // on the starting half. No pre-copy at all — outside-clip and
            // outside-region pixels of the original half are simply never
            // written (the old path paid a FULL-SCREEN pre-copy + full-
            // screen transform for a saturate on one strip).
            if (hsvPipeline_) {
                auto& hp = arena_.read<HSVParams>(cmd.dataOffset);
                const auto R = roiFinal(hp.region, cmd.clipBounds);
                if (R.isEmpty()) continue;
                const VkRect2D sc = roiScissor(R, 0.0f, 0.0f);

                HSVPipeline::PushConstants pc {};
                pc.scaleH = hp.scaleH; pc.scaleS = hp.scaleS; pc.scaleV = hp.scaleV;
                pc.deltaH = hp.deltaH; pc.deltaS = hp.deltaS; pc.deltaV = hp.deltaV;

                vkCmdEndRenderPass(frame.cmd);
                if (coversFrame(sc)) {
                    // Full-window transform: ONE pass, single swap (the old
                    // cost). Pre-copy seeds outside-clip pixels when clipped.
                    preCopyIfClipped(cmd.stencilDepth, &sc);
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        hsvPipeline_->applyPass(frame.cmd, src, dst,
                            target_.effectRenderPass(), frame.extent, pc,
                            static_cast<uint32_t>(cmd.stencilDepth), &sc);
                    });
                } else {
                    // Partial region: region pass + region copy-back — still
                    // far cheaper than the old full-screen passes.
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        hsvPipeline_->applyPass(frame.cmd, src, dst,
                            target_.effectRenderPass(), frame.extent, pc,
                            static_cast<uint32_t>(cmd.stencilDepth), &sc);
                    });
                    copyBackAndSwap(sc, cmd.stencilDepth);
                }
                beginSceneRP(scenePassLoad, /*withClears=*/false);
            }
            continue;
        }
        if (cmd.op == DrawOp::BlurShape) {
            if (shapeBlur_ && hsvPipeline_) {
                auto& sp = arena_.read<BlurShapeParams>(cmd.dataOffset);

                ShapeBlurPipeline::PushConstants pc {};
                pc.invCol0X = sp.invXform[0]; pc.invCol0Y = sp.invXform[1];
                pc.invCol1X = sp.invXform[2]; pc.invCol1Y = sp.invXform[3];
                pc.invCol2X = sp.invXform[4]; pc.invCol2Y = sp.invXform[5];
                pc.shapeHalfX = sp.shapeHalf[0];
                pc.shapeHalfY = sp.shapeHalf[1];
                pc.lineBX     = sp.lineB[0];
                pc.lineBY     = sp.lineB[1];
                pc.maxRadius     = sp.maxRadius;
                pc.falloff       = sp.falloff;
                pc.blurStep      = sp.blurStep;
                pc.cornerRadius  = sp.cornerRadius;
                pc.lineThickness = sp.lineThickness;
                pc.shapeType     = static_cast<int>(sp.shapeType);
                pc.edgePlacement = static_cast<int>(sp.edgePlacement);
                pc.inverted      = static_cast<int>(sp.inverted);

                VkRenderPass rp = target_.effectRenderPass();
                // mode: 0 (Low)    → 2 separable passes (H, V). The V pass
                //                    gates each tap by the TAP's own radius
                //                    (see shape_blur.frag) so low-radius
                //                    falloff rows can't streak sharp content
                //                    into the interior — the classic
                //                    variable-radius separability error.
                //       1 (Medium) → 1 pass, 32-tap Poisson-disc blue-noise
                //                    kernel. Constant cost regardless of
                //                    radius. No streaks — anisotropic noise
                //                    reads as stochastic texture, not a
                //                    directional smear.
                //       2 (High)   → 1 pass, true 2D Gaussian using the 2D
                //                    Linear Sampling Trick (¼ the taps of a
                //                    naïve N² loop, mathematically exact).
                //                    No streaks.
                //
                // Medium + High both read the scene source directly; only
                // one tile-resolve. For small-N blurs this can beat Low.
                const int32_t kt = (sp.mode == 0) ? 0
                                 : (sp.mode == 1) ? 1 : 2;
                pc.kernelType = kt;

                // ROI: record side computed the shape's blur AABB (empty for
                // inverted = whole clip). Kernel apron in physical px.
                const auto R = roiFinal(sp.region, cmd.clipBounds);
                if (R.isEmpty()) continue;
                const float m = std::ceil(sp.maxRadius * sp.blurStep) + 2.0f;
                const VkRect2D scR = roiScissor(R, 0.0f, 0.0f);

                vkCmdEndRenderPass(frame.cmd);
                if (kt == 0) {
                    // Separable H then V — V's vertical apron dictates H's
                    // output rect; two swaps land back on the start half.
                    const VkRect2D scH = roiScissor(R, 0.0f, m);
                    preCopyIfClipped(cmd.stencilDepth, &scH);
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                            frame.extent, scH, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                            frame.extent, scR, 0.0f, 1.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                } else if (coversFrame(scR)) {
                    // Full-window 2D kernel (inverted blurs land here): one
                    // pass, single swap — the old cost.
                    preCopyIfClipped(cmd.stencilDepth, &scR);
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                            frame.extent, scR, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                } else {
                    // 2D kernel, partial region: one region pass reading the
                    // true scene (taps beyond R are always valid) + region
                    // copy-back.
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                            frame.extent, scR, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                    copyBackAndSwap(scR, cmd.stencilDepth);
                }
                beginSceneRP(scenePassLoad, /*withClears=*/false);
            }
            continue;
        }
        if (cmd.op == DrawOp::BlurPath) {
            if (pathBlur_ && pathPipeline_ && hsvPipeline_) {
                auto& bp = arena_.read<BlurPathParams>(cmd.dataOffset);

                PathBlurPipeline::PushConstants pc {};
                pc.maxRadius       = bp.maxRadius;
                pc.falloff         = bp.falloff;
                pc.strokeHalfWidth = bp.strokeHalfWidth;
                pc.segmentStart    = bp.segmentStart;
                pc.segmentCount    = bp.segmentCount;
                pc.fillRule        = bp.fillRule;
                pc.edgePlacement   = static_cast<int32_t>(bp.edgePlacement);
                pc.inverted        = static_cast<int32_t>(bp.inverted);
                pc.stripCount      = bp.stripCount;
                pc.stripMinY       = bp.stripMinY;
                pc.invStripH       = bp.invStripH;

                // PathPipeline::ssboDescriptorSet() returns the descriptor
                // for the CURRENT frame slot (set by the most recent
                // flushToGPU in execute's prologue). That's the slot whose
                // SSBO holds this frame's segments — including those we
                // uploaded in Graphics::{draw,fill}BlurredPath.
                VkDescriptorSet pathDesc = pathPipeline_->ssboDescriptorSet();

                VkRenderPass rp = target_.effectRenderPass();
                // Same mode layout + ROI walk as BlurShape above. Distances
                // here are already physical px.
                const int32_t kt = (bp.mode == 0) ? 0
                                 : (bp.mode == 1) ? 1 : 2;
                pc.kernelType = kt;

                const auto R = roiFinal(bp.region, cmd.clipBounds);
                if (R.isEmpty()) continue;
                const float m = std::ceil(bp.maxRadius) + 2.0f;
                const VkRect2D scR = roiScissor(R, 0.0f, 0.0f);

                vkCmdEndRenderPass(frame.cmd);
                if (kt == 0) {
                    const VkRect2D scH = roiScissor(R, 0.0f, m);
                    preCopyIfClipped(cmd.stencilDepth, &scH);
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        pathBlur_->applyPass(frame.cmd, src, pathDesc, dst, rp,
                            frame.extent, scH, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        pathBlur_->applyPass(frame.cmd, src, pathDesc, dst, rp,
                            frame.extent, scR, 0.0f, 1.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                } else if (coversFrame(scR)) {
                    preCopyIfClipped(cmd.stencilDepth, &scR);
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        pathBlur_->applyPass(frame.cmd, src, pathDesc, dst, rp,
                            frame.extent, scR, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                } else {
                    effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                        pathBlur_->applyPass(frame.cmd, src, pathDesc, dst, rp,
                            frame.extent, scR, 1.0f, 0.0f, pc,
                            static_cast<uint32_t>(cmd.stencilDepth));
                    });
                    copyBackAndSwap(scR, cmd.stencilDepth);
                }
                beginSceneRP(scenePassLoad, /*withClears=*/false);
            }
            continue;
        }
        auto* pipeline = pipelineForOp_[static_cast<size_t>(cmd.op)];
        if (!pipeline) continue;
        if (!pipeline->isBuilt())
            pipeline->build(sceneBuildRP);
        if (!pipeline->isBuilt()) continue;   // build failed — never bind null
        state_.setPipeline(pipeline);
        pipeline->execute(*this, arena_, cmd);
    }

    vkCmdEndRenderPass(frame.cmd);

    // After the command stream, `cur` is the index of whichever ping-pong
    // half holds the final composited frame (full-window single-pass effects
    // leave it flipped; everything else returns to the starting half). The
    // blit reads pp[cur] either way.
    VkImage currentImage = pp[cur].image;

    if (frame.swapImage == VK_NULL_HANDLE) {
        // Offscreen target — no swap image to blit to. Scene content remains
        // in currentImage (SHADER_READ_ONLY_OPTIMAL) for the caller to sample.
        // Serialize queue submit against other editors sharing the Device's
        // VkQueue (Vulkan external sync requirement).
        {
            const juce::ScopedLock queueSync(queueLock());
            target_.endFrame(frame);
        }
        {
            const juce::ScopedLock lk(uploadLock_);
            frameSlots_[activeRetireSlot_].pins.swap(recordingRetains_);
        }
        return;
    }

    // Blit currentImage → swap image. currentImage is in SHADER_READ_ONLY
    // (finalLayout of the last scene RP). Transition both images and copy.
    VkImageMemoryBarrier barriers[2] {};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // Cover both possible last-use types of colorA: it was either last
    // written as a color attachment (final scene RP's writes → finalLayout
    // transition) or last sampled in a shader (if an effect pass ran and
    // then we re-entered scene RP LOAD). Including both bits eliminates
    // a subtle write-after-read hazard on tile renderers at slow frame
    // rates that caused post-effect draws to flicker every other frame.
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT
                              | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = currentImage;
    barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = frame.swapImage;
    barriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(frame.cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, barriers);

    VkImageBlit region {};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[1] = { (int32_t)frame.extent.width, (int32_t)frame.extent.height, 1 };
    region.dstOffsets[1] = { (int32_t)frame.extent.width, (int32_t)frame.extent.height, 1 };
    vkCmdBlitImage(frame.cmd,
        currentImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        frame.swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region, VK_FILTER_NEAREST);

    VkImageMemoryBarrier present {};
    present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    present.dstAccessMask = 0;
    present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    present.image = frame.swapImage;
    present.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(frame.cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &present);

    // Serialize queue submit+present against other editors sharing the Device's
    // VkQueue (Vulkan external sync requirement).
    {
        const juce::ScopedLock queueSync(queueLock());
        target_.endFrame(frame);
    }
    // File this record's pins against the slot just submitted. Swap (not
    // move) so recordingRetains_ keeps its capacity for the next record;
    // the slot's pins vector was emptied by the drain at the top.
    {
        const juce::ScopedLock lk(uploadLock_);
        frameSlots_[activeRetireSlot_].pins.swap(recordingRetains_);
    }
}

// =============================================================================
// State implementation
// =============================================================================

void State::begin(VkCommandBuffer cmd, Memory::V& vertices, float vpWidth, float vpHeight)
{
    cmd_ = cmd;
    vertices_ = &vertices;
    vpWidth_ = vpWidth;
    vpHeight_ = vpHeight;
    invalidate();
    // Reset the CPU-side clip state every frame. The stencil buffer and
    // scissor start each frame clean (CLEAR load op / full-extent scissor),
    // so a stale stack from an unbalanced push last frame (exception unwind
    // mid-paint, a component painting outside ScopedSaveState) must not
    // survive: a leaked stencilDepth_ > 0 binds clip-variant pipelines whose
    // reference can never match the cleared stencil — a permanently black
    // window until the editor is recreated. With the reset it heals in one
    // frame.
    clipRectStack_.clear();
    currentClipBounds_ = {};
    stencilDepth_ = 0;
}

void State::invalidate()
{
    currentPipeline_ = nullptr;
    boundPipeline_ = VK_NULL_HANDLE;
    boundLayout_ = VK_NULL_HANDLE;
    boundColorSet_ = VK_NULL_HANDLE;
    boundShapeSet_ = VK_NULL_HANDLE;
    boundStencilRef_ = 0;
    boundScissor_ = { -1, -1, 0, 0 };
    boundVertexBuffer_ = VK_NULL_HANDLE;
}

void State::setPipeline(Pipeline* pipeline)
{
    if (!pipeline) return;

    // Clip variant of the pipeline has stencilTest on (cmp = EQUAL, ops =
    // KEEP), reference pushed dynamically below. Non-clip variant has no
    // stencil test at all — used whenever stencilDepth_ == 0.
    VkPipeline handle = (stencilDepth_ > 0) ? pipeline->clipHandle()
                                            : pipeline->handle();

    if (handle != boundPipeline_) {
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, handle);
        boundPipeline_ = handle;

        // Layout change invalidates descriptor set bindings in Vulkan
        if (boundLayout_ != pipeline->layout()) {
            boundColorSet_ = VK_NULL_HANDLE;
            boundShapeSet_ = VK_NULL_HANDLE;
            boundLayout_ = pipeline->layout();
        }

        // Push viewport size for pixel→NDC conversion in vertex shader
        float vpSize[2] = { vpWidth_, vpHeight_ };
        vkCmdPushConstants(cmd_, boundLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vpSize), vpSize);
    }

    // Stencil reference = clip depth. Every draw op passes only where
    // stencil == depth (i.e. inside all active clips).
    if (stencilDepth_ > 0 && stencilDepth_ != boundStencilRef_) {
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth_);
        boundStencilRef_ = stencilDepth_;
    }

    currentPipeline_ = pipeline;
}

void State::setCustomPipeline(VkPipeline pipeline, VkPipelineLayout layout)
{
    if (pipeline != boundPipeline_) {
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        boundPipeline_ = pipeline;
        boundLayout_ = layout;
    }
    if (stencilDepth_ > 0)
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth_);

    currentPipeline_ = nullptr; // force rebind on next setPipeline
}

void State::setResources(VkDescriptorSet colorSet, VkDescriptorSet shapeSet)
{
    setColorResource(colorSet);
    setShapeResource(shapeSet);
}

void State::setColorResource(VkDescriptorSet set)
{
    if (set != boundColorSet_) {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
            boundLayout_, 0, 1, &set, 0, nullptr);
        boundColorSet_ = set;
    }
}

void State::setShapeResource(VkDescriptorSet set)
{
    if (set != boundShapeSet_) {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
            boundLayout_, 1, 1, &set, 0, nullptr);
        boundShapeSet_ = set;
    }
}

void State::draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count)
{
    if (count == 0) return;

    // Set scissor from command clip bounds
    juce::Rectangle<int> clip = cmd.clipBounds;
    if (!currentClipBounds_.isEmpty())
        clip = clip.getIntersection(currentClipBounds_);

    if (clip != boundScissor_) {
        // Clamp each edge to the framebuffer before computing the extent,
        // so a clip starting off-screen (x<0) doesn't produce an offset of
        // 0 with the un-clipped width and leak past the right edge.
        int x0 = std::max(0, clip.getX());
        int y0 = std::max(0, clip.getY());
        int x1 = std::max(x0, clip.getRight());
        int y1 = std::max(y0, clip.getBottom());
        VkRect2D sc;
        sc.offset = { x0, y0 };
        sc.extent = { static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0) };
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        boundScissor_ = clip;
    }

    // Ring buffer — bind it once at offset 0 and use firstVertex for the
    // subrange. Each write() returns a vertex-aligned byte offset; we convert
    // to a vertex index so the bind only changes when the slot grows.
    VkDeviceSize byteCount  = count * sizeof(UIVertex);
    VkDeviceSize byteOffset = vertices_->write(verts, byteCount);
    VkBuffer     buf        = vertices_->getBuffer();
    uint32_t     firstVert  = static_cast<uint32_t>(byteOffset / sizeof(UIVertex));

    if (buf != boundVertexBuffer_) {
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd_, 0, 1, &buf, &zero);
        boundVertexBuffer_ = buf;
    }
    vkCmdDraw(cmd_, count, 1, firstVert, 0);
}

void State::pushClipRect(const juce::Rectangle<int>& rect)
{
    auto clipped = currentClipBounds_.isEmpty()
        ? rect
        : currentClipBounds_.getIntersection(rect);
    clipRectStack_.push_back(clipped);
    currentClipBounds_ = clipped;
}

void State::popClipRect()
{
    if (clipRectStack_.empty()) return;
    clipRectStack_.pop_back();
    currentClipBounds_ = clipRectStack_.empty()
        ? juce::Rectangle<int>{}
        : clipRectStack_.back();
}

void State::pushStencilDepth()
{
    stencilDepth_++;
}

void State::popStencilDepth()
{
    if (stencilDepth_ > 0) stencilDepth_--;
}

} // namespace jvk
