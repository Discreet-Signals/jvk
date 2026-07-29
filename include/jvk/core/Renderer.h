#pragma once

namespace jvk {

class Pipeline;
class RenderTarget;
namespace pipelines { class ColorPipeline; }
class EffectPipeline;
class HSVPipeline;
class ShapeBlurPipeline;
class PathBlurPipeline;
class ShaderPipeline;
class PathPipeline;
class ClipPipeline;
class GradientAtlas;   // defined in Cache.h (included after this header)

// =============================================================================
// UIVertex — shared vertex format for 2D pipelines
// =============================================================================

struct UIVertex {
    glm::vec2 position;     // screen pixel coords
    glm::vec4 color;        // RGBA — used directly when gradientInfo.z==0 (solid)
    glm::vec2 uv;           // 0-1 within quad (or glyph/image UV)
    glm::vec4 shapeInfo;    // x=type(0=flat,1=rounded,2=ellipse,3=image,4=MSDF), y=halfW/pxRange, z=halfH, w=param
    glm::vec4 gradientInfo; // x,y = gradient t (linear: x; radial: length(x,y)); z = mode (0=solid,1=linear,2=radial); w=unused
};

// =============================================================================
// DrawOp — what to draw or what state change to make
// =============================================================================

enum class DrawOp : uint8_t {
    FillRect, FillRectList, FillRoundedRect, FillEllipse,
    StrokeRoundedRect, StrokeEllipse,
    DrawImage, DrawGlyphs, DrawLine,
    FillPath,           // analytical SDF path fill (vger-style)
    DrawShader,
    EffectBlend,
    EffectNoise,        // procedural white-noise overlay (Graphics::drawNoise)
    EffectKernel,
    EffectHSV,          // full-screen HSV scale/delta (saturate, shiftHue, etc.)
    BlurShape,
    BlurPath,          // analytical path SDF blur (fill or stroke ring)
    PushClipRect, PopClipRect,      // scissor-only clips (State-side stack)
    PushClipPath, PopClipPath,      // stencil INCR/DECR via ClipPipeline
    FillTiledImage,                 // tiled image fill (setTiledImageFill)
    ExcludeClipRect,                // stencil-park the rect out of the clip
    RestoreClipExclude,             // un-park it on restoreState
    COUNT
};

// =============================================================================
// DrawCommand — one entry in the command vector
// =============================================================================

// Slimmed to what replay actually reads. The old zOrder/dataSize/scopeDepth
// fields were written on every push and never read — the z-sort they were
// designed for was never implemented, and commands replay in recording order
// (JUCE's painter's algorithm order, which is already correct).
struct DrawCommand {
    DrawOp               op;
    uint8_t              stencilDepth;
    uint32_t             dataOffset;
    juce::Rectangle<int> clipBounds;
};

// =============================================================================
// Arena — flat byte allocator for command parameter data
// =============================================================================

class Arena {
public:
    template <typename T>
    uint32_t push(const T& data)
    {
        static_assert(std::is_trivially_copyable_v<T>, "Arena is POD-only. Use Renderer side vectors for non-POD types.");
        uint32_t offset = head_;
        uint32_t needed = static_cast<uint32_t>(sizeof(T));
        ensureCapacity(offset + needed);
        memcpy(buffer_.data() + offset, &data, sizeof(T));
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    uint32_t pushSpan(std::span<const T> data)
    {
        uint32_t offset = head_;
        uint32_t needed = static_cast<uint32_t>(data.size_bytes());
        ensureCapacity(offset + needed);
        memcpy(buffer_.data() + offset, data.data(), data.size_bytes());
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    const T& read(uint32_t offset) const
    {
        return *reinterpret_cast<const T*>(buffer_.data() + offset);
    }

    template <typename T>
    std::span<const T> readSpan(uint32_t offset, uint32_t count) const
    {
        return { reinterpret_cast<const T*>(buffer_.data() + offset), count };
    }

    // Advance head to the next alignment boundary (e.g., align(4) for float data after uint16 data)
    void align(uint32_t alignment)
    {
        uint32_t mask = alignment - 1;
        head_ = (head_ + mask) & ~mask;
    }

    void reset() { head_ = 0; }
    uint32_t size() const { return head_; }

private:
    std::vector<std::byte> buffer_;
    uint32_t head_ = 0;

    void ensureCapacity(uint32_t needed)
    {
        if (needed > buffer_.size())
            buffer_.resize(std::max(needed, static_cast<uint32_t>(buffer_.size() * 2 + 1024)));
    }
};

// =============================================================================
// State — dirty-tracked GPU state + clip stack (internal to Renderer)
// =============================================================================

class State {
public:
    State() = default;

    void setPipeline(Pipeline* pipeline);
    void setCustomPipeline(VkPipeline pipeline, VkPipelineLayout layout);
    // set 0 = color source (solid default or gradient LUT), set 1 = shape source
    // (1x1 default, MSDF atlas page, or image texture). Each dirty-tracked.
    void setResources(VkDescriptorSet colorSet, VkDescriptorSet shapeSet);
    void setShapeResource(VkDescriptorSet shapeSet);
    void setColorResource(VkDescriptorSet colorSet);
    void draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count);

    void pushClipRect(const juce::Rectangle<int>& rect);
    void pushStencilDepth(); // CPU-only: increment the clip counter used as
                              // stencil reference for INCR/compare. The GPU
                              // INCR pass is a separate DrawOp::PushClipPath
                              // dispatched via ClipPipeline.
    void popStencilDepth();
    void popClipRect();

    juce::Rectangle<int> clipBounds() const { return currentClipBounds_; }
    uint8_t              stencilDepth() const { return stencilDepth_; }

    void invalidate();
    void begin(VkCommandBuffer cmd, Memory::V& vertices, float vpWidth, float vpHeight);

private:
    VkCommandBuffer  cmd_       = VK_NULL_HANDLE;
    Memory::V*       vertices_  = nullptr;
    float            vpWidth_   = 0;
    float            vpHeight_  = 0;

    Pipeline*        currentPipeline_ = nullptr;
    VkPipeline       boundPipeline_   = VK_NULL_HANDLE;
    VkPipelineLayout boundLayout_     = VK_NULL_HANDLE;
    VkDescriptorSet  boundColorSet_   = VK_NULL_HANDLE;
    VkDescriptorSet  boundShapeSet_   = VK_NULL_HANDLE;
    uint32_t         boundStencilRef_ = 0;
    juce::Rectangle<int> boundScissor_ { -1, -1, 0, 0 };
    VkBuffer         boundVertexBuffer_ = VK_NULL_HANDLE;

    // Clip rectangle stack — scissor-based (axis-aligned rect clips). Path
    // / rrect clips don't use this stack; they're tracked purely via the
    // stencil buffer + stencilDepth_ counter.
    std::vector<juce::Rectangle<int>> clipRectStack_;
    juce::Rectangle<int>              currentClipBounds_;
    uint8_t                           stencilDepth_ = 0;
};

// =============================================================================
// Renderer — the core execution engine
// =============================================================================

class Renderer {
public:
    Renderer(Device& device, RenderTarget& target);
    ~Renderer();

    template <typename Params>
    void push(DrawOp op, const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, const Params& params)
    {
        uint32_t offset = arena_.push(params);
        commands_.push_back({ op, stencilDepth, offset, clip });
    }

    // Pin a FrameRetained so its destructor will block until the GPU is
    // done with the frame this record is being assembled into. Called by
    // record-time hooks in jvk::Graphics for any payload that captures a
    // raw pointer to a user-owned GPU object (currently jvk::Shader). The
    // matching unpin runs from execute() once the slot's fence has been
    // waited on — so user code can free the object from the message
    // thread at any time without coordinating with the worker.
    //
    // Defined as a template so callers don't need the full definition of
    // FrameRetained's subclass at the point that records the command —
    // Graphics.h only forward-declares jvk::Shader, and the base-class
    // conversion has to be checked where the caller has already pulled in
    // Shader.h via the umbrella.
    template <typename T>
    void retain(T* obj)
    {
        static_assert(std::is_base_of_v<FrameRetained, T>,
            "Renderer::retain(T*) requires T to inherit from jvk::FrameRetained");
        if (!obj) return;
        FrameRetained* base = obj;
        base->pin();
        // uploadLock_ guards every record↔worker shared container. retain()
        // is reachable from the message thread OUTSIDE the isBusy() gate
        // (Shader::update, Cache::getTexture via Shader::set), so an
        // unlocked push_back can race the worker's swap — a lost entry means
        // the pin is never dropped (object leaks pinned forever) or, torn,
        // the vector corrupts. Same bug class as the fixed upload-queue race.
        {
            const juce::ScopedLock lk(uploadLock_);
            recordingRetains_.push_back(base);
        }
    }

    void registerPipeline(Pipeline& pipeline);

    // Drop every registered pipeline pointer (dispatch table + module
    // attachments). Teardown calls this BEFORE destroying the pipeline
    // objects so the Renderer never holds dangling pipeline pointers.
    void clearPipelines()
    {
        for (auto& p : pipelineForOp_) p = nullptr;
        postProcess_ = nullptr;  copyEffect_   = nullptr;
        hsvPipeline_ = nullptr;  shapeBlur_    = nullptr;
        pathBlur_    = nullptr;  shaderPipeline_ = nullptr;
        pathPipeline_ = nullptr; clipPipeline_ = nullptr;
        colorPipeline_ = nullptr;
    }

    // ---- Threaded execution -------------------------------------------------
    //
    // Frames are executed on a dedicated worker thread so the caller (typically
    // the JUCE message thread) is never stuck inside vkQueuePresentKHR. NVIDIA's
    // Windows driver busy-loops the calling thread inside present under FIFO
    // mode; running that on the message thread pegs the CPU and starves the OS
    // input/DWM dispatch, freezing window drag and mouse response.
    //
    // Contract (strict, 1 frame in flight):
    //   1. Caller fills the Renderer (reset, beginFrame on caches, paint).
    //   2. Caller calls submit(). Control returns immediately; the worker runs
    //      execute() asynchronously. submit() MUST NOT be called while
    //      isBusy() returns true — caller must guard with an isBusy() check.
    //   3. Worker clears the busy flag after execute() fully returns.
    //   4. Caller's next frame must wait until isBusy() == false before
    //      touching Renderer state again.
    //
    // The busy flag's release/acquire pair publishes every CPU-side write
    // (command list, arena, path SSBO, CPU-mapped Vulkan buffers) so the
    // worker sees consistent record-phase state.

    // Post the current frame to the worker. Non-blocking. Preconditions: the
    // caller has just finished the record phase, and isBusy() == false.
    void submit();

    // True while the worker is mid-execute. Check before recording the next
    // frame — the caller must not touch Renderer state while this is true.
    bool isBusy() const { return workerBusy_.load(std::memory_order_acquire); }

    // Block the caller until the worker is idle. Use before operations that
    // need exclusive access outside the normal record→submit cycle
    // (SwapchainTarget::resize, teardown). Bounded wait ≈ one execute
    // duration (~16 ms on VSync).
    void waitForIdle();

    // Drop every FrameRetained pin held by this Renderer (recording bucket
    // + every per-slot bucket). Used by teardown / mode-switch paths that
    // stop submitting new frames — without this the matching unpins would
    // never run and any subsequent ~Shader (or other FrameRetained) would
    // deadlock spinning on its in-flight counter.
    //
    // CALLER CONTRACT: GPU must already be idle for any submission that
    // referenced these pins. waitForIdle() guarantees the worker is done,
    // but the worker's final submit may still have GPU work in flight —
    // the caller is responsible for vkDeviceWaitIdle (or equivalent fence
    // wait) before invoking this.
    void flushRetains();

    // Force every live Renderer to a fully-quiescent state and drop all
    // FrameRetained pins. Walks the process-wide registry, idles each
    // worker, calls vkDeviceWaitIdle once per unique Device, then runs
    // flushRetains on each. After this returns, every FrameRetained's
    // in-flight counter is zero and every GPU submission referencing
    // pinned objects has retired.
    //
    // Called by FrameRetained::waitUntilUnretained when the natural
    // per-slot drain can't be relied on — most commonly when the
    // destructor is itself running on the message thread (e.g. inside a
    // chain refresh) and would otherwise prevent any future render tick
    // from firing.
    static void forceDrainAll();

    // Number of live Renderer instances in the process. A cheap probe for
    // "is anything rendering through jvk right now" — finec::Images uses it
    // to auto-select image backing (GPU-resident vs software) with zero
    // per-app wiring. Message-thread callers only need a moment-in-time
    // answer; the registry lock makes the read safe from anywhere.
    static int liveCount();

    // The synchronous body of a frame's GPU work. Called internally by the
    // worker thread. Kept public because some non-windowed consumers (e.g.
    // benchmark harnesses, offscreen probes) may want a direct synchronous
    // path — but the normal windowed path must go through submit().
    void execute();

    // Attach a post-process helper that handles EffectKernel ops. When set,
    // the next execute() that encounters an EffectKernel routes the main
    // render pass through the target's sampleable-intermediate variant and
    // applies the 2-pass separable effect (e.g. Gaussian blur) before present.
    void setPostProcess(EffectPipeline* ep) { postProcess_ = ep; }

    // Attach the pre-copy pipeline used for clipped effects. It shares
    // EffectPipeline's plumbing but uses StencilMode::Outside, so the pass
    // writes only the pixels OUTSIDE the active clip (source passthrough),
    // leaving the subsequent effect pass to fill the inside. Without this,
    // clipped effects would leave outside-clip pixels holding garbage
    // (whatever was in the destination ping-pong half before).
    void setCopyEffect(EffectPipeline* ep) { copyEffect_ = ep; }

    // Attach the HSV transform pipeline — required for EffectHSV ops
    // (g.saturate / g.shiftHue / g.hsv). Ping-pongs through the scene
    // intermediate and returns to `current`.
    void setHSVPipeline(HSVPipeline* hp) { hsvPipeline_ = hp; }

    // Attach the shape-aware blur pipeline. Required for BlurShape draw ops
    // (Graphics::{draw,fill}Blurred{Rectangle,RoundedRectangle,Ellipse} +
    // drawBlurredLine).
    void setShapeBlur(ShapeBlurPipeline* sb) { shapeBlur_ = sb; }

    // Attach the path-blur pipeline. Required for BlurPath draw ops
    // (Graphics::{draw,fill}BlurredPath). Shares PathPipeline's per-frame
    // segment SSBO — must be set AFTER setPathPipeline() so the dispatch
    // can grab that descriptor.
    void setPathBlur(PathBlurPipeline* pb) { pathBlur_ = pb; }

    // Attach the DrawShader dispatcher. Required for DrawShader draw ops
    // (g.drawShader). User shaders own their own VkPipeline; this module
    // handles the per-command bind + push-constants + draw.
    void setShaderPipeline(ShaderPipeline* sp) { shaderPipeline_ = sp; }

    // Attach the analytical-SDF path renderer. Used by Graphics::fillPath
    // to upload flattened path segments and emit DrawOp::FillPath ops.
    // Owns a storage buffer ring that holds the per-frame segment data —
    // ALSO shared with the clip-stencil pipelines for DrawOp::PushClipPath
    // / PopClip of arbitrary paths (they read the same SSBO).
    void setPathPipeline(PathPipeline* pp) { pathPipeline_ = pp; }
    PathPipeline* pathPipeline() const { return pathPipeline_; }

    // Attach the clip-stencil pipeline. Owns two VkPipelines sharing one
    // shader + layout: push variant uses stencilPassOp=INCR_WRAP, pop
    // variant uses DECR_WRAP. Called for rrect + arbitrary-path clips —
    // axis-aligned rect clips bypass this via plain scissor.
    void setClipPipeline(ClipPipeline* p) { clipPipeline_ = p; }
    ClipPipeline* clipPipeline() const { return clipPipeline_; }

    // Attach the color pipeline (glyph atlas owner). Graphics::drawGlyphs
    // rasterizes missing glyphs through this at RECORD time (message thread),
    // so the atlas page is staged + uploaded before the frame's render pass —
    // the old replay-time rasterization created pages AFTER the upload flush,
    // sampling them in UNDEFINED layout the frame a glyph first appeared.
    void setColorPipeline(pipelines::ColorPipeline* cp) { colorPipeline_ = cp; }
    pipelines::ColorPipeline* colorPipeline() const { return colorPipeline_; }

    // ---- Deferred uploads ---------------------------------------------------
    //
    // Every texture GPU upload triggered during record (image cache
    // inserts, glyph-atlas dirty pages, gradient-atlas row uploads) queues
    // into THIS Renderer's pending list. flushUploads records the copy +
    // barrier commands into the frame's command buffer just before the
    // scene render pass. Keeping the queue on the Renderer — not Device —
    // means each editor's worker drains its own queue and two editors never
    // race on a shared vector.
    void upload(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height);

    // Partial-image upload: copies a w×h rect (tightly packed in staging)
    // into dst at (x, y), PRESERVING the rest of the image — the pre-copy
    // barrier transitions from SHADER_READ_ONLY instead of UNDEFINED. Used
    // by the glyph atlas to upload only the dirty rect of a page instead of
    // re-staging all 16 MB for one new glyph.
    void uploadRect(Memory::L2::Allocation src, VkImage dst,
                    int32_t x, int32_t y, uint32_t width, uint32_t height);

    // upload() variant for images whose CONTENTS are refreshed every frame
    // (jvk::Shader::update's per-binding video/procedural feeds). Identical
    // queue/flush path, but the recorded pre-copy barrier waits for all
    // previously submitted fragment-shader work: with two frames in flight,
    // the copy would otherwise overwrite texels the prior frame is still
    // sampling (plain upload() assumes a freshly created, never-read image
    // and uses a TOP_OF_PIPE barrier that orders nothing).
    void uploadDynamic(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height);

    // Drop any queued uploads targeting `dst`. Call before destroying /
    // retiring an image that may still have a pending entry (e.g. a dynamic
    // shader input resized twice between executes) so flushUploads never
    // records a copy into a freed VkImage.
    void cancelUploads(VkImage dst);

    // Same, across every live Renderer in the process. For destructors that
    // have no Renderer back-pointer (~Shader): a queued upload must never
    // outlive its destination image.
    static void cancelUploadsAllRenderers(VkImage dst);

    void flushUploads(VkCommandBuffer cmd);

    // Record a single image-copy upload (TRANSFER_DST barrier → copy →
    // SHADER_READ_ONLY barrier) into `cmd` directly. Use for one-shot
    // setup uploads that have to run before any Renderer exists
    // (Device::submitImmediate bootstrap paths). Per-frame work should go
    // through upload()/flushUploads() so it batches with the scene.
    // `dynamicContent` selects the fragment-ordered pre-copy barrier
    // described at uploadDynamic().
    static void recordImageUpload(VkCommandBuffer cmd, Memory::L2::Allocation src,
                                   VkImage dst, uint32_t width, uint32_t height,
                                   bool dynamicContent = false,
                                   bool partial = false,
                                   int32_t dstX = 0, int32_t dstY = 0);

    // ---- Deferred destruction ----------------------------------------------
    //
    // Resources replaced mid-frame (e.g. a dynamic shader input resized)
    // move into the active FrameSlot's retired bucket. The bucket is drained
    // the NEXT time the slot rolls around — after target_.beginFrame() has
    // waited on the slot's fence — so the GPU is provably done with whatever
    // we destroy. Locked: retire() is message-thread-reachable outside the
    // isBusy() gate (Shader::update) while the worker rotates the slot.
    void retire(Image&& img)
    {
        const juce::ScopedLock lk(uploadLock_);
        frameSlots_[activeRetireSlot_].retiredImages.push_back(std::move(img));
    }

    // Capture non-POD types into side vectors. Returns index. Consecutive
    // duplicates dedupe: JUCE paints whole components with one fill/font, so
    // most captures repeat the previous entry — and a gradient FillType copy
    // heap-allocates a fresh ColourGradient every time. One equality check
    // saves an allocation per draw in the common case.
    uint32_t captureFont(const juce::Font& f)
    {
        if (!fonts_.empty() && fonts_.back() == f)
            return static_cast<uint32_t>(fonts_.size() - 1);
        fonts_.push_back(f);
        return static_cast<uint32_t>(fonts_.size() - 1);
    }
    uint32_t captureFill(const juce::FillType& f)
    {
        if (!fills_.empty() && fills_.back() == f)
            return static_cast<uint32_t>(fills_.size() - 1);
        fills_.push_back(f);
        return static_cast<uint32_t>(fills_.size() - 1);
    }

    const juce::Font&     getFont(uint32_t i) const { return fonts_[i]; }
    const juce::FillType& getFill(uint32_t i) const { return fills_[i]; }

    // Forward to arena for appending POD data after params
    void arena_align(uint32_t alignment) { arena_.align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { arena_.pushSpan(data); }

    // Per-frame reset: drop the command list, arena, captured non-POD types,
    // and the gradient atlas's per-frame state. Out-of-line because
    // gradientAtlas_ is a unique_ptr<GradientAtlas> and this header only
    // forward-declares the type.
    void reset();

    Device&         device()   { return device_; }
    RenderTarget&   target()   { return target_; }
    ResourceCaches& caches()   { return device_.caches(); }
    State&          state()    { return state_; }
    Memory::V&      vertices() { return vertices_; }
    const Arena&    arena() const { return arena_; }

    // Per-Renderer staging allocator for per-frame CPU→GPU uploads (glyph
    // atlas pages, gradient atlas rows, texture cache inserts, etc.).
    // Device's shared staging is retained only for the one-shot black-pixel
    // bootstrap inside Device::initCaches. Keeping per-frame staging
    // per-Renderer means the bump allocator is only ever touched by one
    // editor's threads, not shared across instances.
    Memory::L2&     staging() { return staging_; }

    // ---- Gradient atlas (per-Renderer) -------------------------------------
    //
    // One gradient atlas per Renderer — mutated on the message thread during
    // record (registerGradient) and uploaded to the GPU from the worker
    // thread (stageUploads inside execute). Keeping it per-Renderer keeps
    // those two threads inside one editor only, so two editors never race
    // on cursor_/hashToRow_/cpuBuffer_. Accessors are out-of-line so this
    // header only needs a forward declaration of GradientAtlas.
    float           registerGradient(const juce::ColourGradient& g);
    VkDescriptorSet gradientDescriptor() const;

private:
    Device&       device_;
    RenderTarget& target_;
    State         state_;
    Memory::V     vertices_;
    Memory::L2    staging_;

    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    // Non-POD captures (proper RAII, cleared each frame)
    std::vector<juce::Font>     fonts_;
    std::vector<juce::FillType> fills_;

    // Pending uploads queued during record; drained by flushUploads at the
    // top of execute(). Per-Renderer so two editors never share this vector.
    struct PendingUpload {
        VkImage      dstImage;
        uint32_t     width, height;
        VkBuffer     srcBuffer;
        VkDeviceSize srcOffset;
        bool         dynamicContent = false;
        // Partial-rect upload (uploadRect): copy lands at (dstX, dstY) and
        // the pre-copy barrier preserves existing contents.
        bool         partial = false;
        int32_t      dstX = 0, dstY = 0;
    };
    // The upload queue is pushed from the message thread (record:
    // Shader::update dynamic feeds, cache inserts, atlas pages) while the
    // render worker drains it in flushUploads — every access goes through
    // uploadLock_. Without it a push_back racing the worker's drain is
    // silently lost, and a per-frame dynamic texture (CRT screen) can miss
    // EVERY upload when the two clocks phase-lock: the image never leaves
    // UNDEFINED and samples black.
    //
    // uploadLock_ is THE record↔worker shared-state lock: it also guards
    // recordingRetains_, activeRetireSlot_, and every FrameSlot vector.
    juce::CriticalSection      uploadLock_;
    std::vector<PendingUpload> pendingUploads_;
    std::vector<PendingUpload> uploadScratch_;   // worker-only swap target

    // ---- Per-frame-slot lifetime rotation ----------------------------------
    //
    // ONE structure per frame-in-flight slot holding everything whose release
    // is keyed to that slot's fence:
    //   - retiredImages: resources replaced mid-frame, destroyed post-fence
    //   - staging:       L2 blocks consumed by the slot's submitted frame,
    //                    recycled post-fence
    //   - pins:          FrameRetained objects referenced by the slot's
    //                    command buffer, unpinned post-fence
    // execute() drains the slot in ONE place at the top (after
    // target_.beginFrame() waited the slot's fence) and refills it at the
    // bottom. A single rotation makes the historical failure modes
    // structurally impossible: the v2-refactor L2 leak (one leg of the
    // rotation dropped) and the retain/retire races (vectors mutated from
    // the message thread while the worker moved them).
    struct FrameSlot {
        std::vector<Image>             retiredImages;
        std::vector<Memory::L2::Block> staging;
        std::vector<FrameRetained*>    pins;
    };
    static constexpr int kFrameSlots = 2; // matches MAX_FRAMES_IN_FLIGHT
    std::array<FrameSlot, kFrameSlots> frameSlots_;
    int       activeRetireSlot_ = 0;      // guarded by uploadLock_
    FrameSlot flushScratch_;              // worker-only swap target (keeps capacity)

    // Per-Renderer gradient atlas (see public accessors above for rationale).
    // unique_ptr so this header only needs a forward declaration.
    std::unique_ptr<GradientAtlas> gradientAtlas_;

    // FrameRetained pins gathered during record (msg thread, under
    // uploadLock_). At the end of execute they swap into the active
    // FrameSlot's pins; unpinned the next time that slot rolls around.
    std::vector<FrameRetained*> recordingRetains_;

    // Process-wide registry of live Renderer instances. Walked by
    // forceDrainAll (invoked from FrameRetained destructors) so a
    // destruction running on the message thread can synchronously drain
    // every renderer's pin counts without depending on the next render
    // tick — which would never come, because the message thread is the
    // very thread blocked in the destructor.
    static juce::CriticalSection& registryLock();
    static std::vector<Renderer*>& registry();

    Pipeline* pipelineForOp_[static_cast<size_t>(DrawOp::COUNT)] = {};
    pipelines::ColorPipeline* colorPipeline_ = nullptr;
    EffectPipeline*    postProcess_      = nullptr;
    EffectPipeline*    copyEffect_       = nullptr;
    HSVPipeline*       hsvPipeline_      = nullptr;
    ShapeBlurPipeline* shapeBlur_        = nullptr;
    PathBlurPipeline*  pathBlur_         = nullptr;
    ShaderPipeline*    shaderPipeline_   = nullptr;
    PathPipeline*      pathPipeline_     = nullptr;
    ClipPipeline*      clipPipeline_     = nullptr;

public:
    // ---- Threading ---------------------------------------------------------
    // Vulkan's VkQueue requires external synchronization (spec §4.2.1).
    // Every code path that submits to or waits on the shared graphics queue
    // MUST take this lock for the full submit+present+wait window:
    //   - Renderer::execute → target_.endFrame (vkQueueSubmit + vkQueuePresentKHR)
    //   - Device::submitImmediate (vkQueueSubmit + vkQueueWaitIdle, one-shot
    //     staging uploads during setup)
    // Contention is only across editors; within a single editor the worker
    // is the sole submitter and the lock is uncontended.
    static juce::CriticalSection& queueLock()
    {
        static juce::CriticalSection lock;
        return lock;
    }
private:

    class Worker : public juce::Thread
    {
    public:
        explicit Worker(Renderer& r) : juce::Thread("jvk-render-worker"), owner(r) {}
        ~Worker() override { stopThread(2000); }

        void run() override
        {
            while (! threadShouldExit())
            {
                wait(-1);
                if (threadShouldExit()) break;
                owner.execute();
                owner.workerBusy_.store(false, std::memory_order_release);
            }
        }

    private:
        Renderer& owner;
    };

    // workerBusy_ is the record↔execute gate. Msg thread reads false →
    // records → store(true, release) before notifying the worker. Worker
    // store(false, release) only after execute() has fully returned.
    std::atomic<bool> workerBusy_ { false };

    // Declared last so destruction order stops + joins the worker BEFORE
    // any other Renderer member is torn down — everything the worker touches
    // in execute() is still valid until worker_ is destroyed.
    std::unique_ptr<Worker> worker_;
};

} // namespace jvk
