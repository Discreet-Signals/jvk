#pragma once

namespace jvk {

class Pipeline;
class RenderTarget;
class EffectPipeline;
class HSVPipeline;
class BlurPipeline;
class FillPipeline;
class ShaderPipeline;
class PathPipeline;
class ClipPipeline;
class CommandScheduler;

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
    EffectResolve,
    EffectKernel,
    EffectHSV,          // full-screen HSV scale/delta (saturate, shiftHue, etc.)
    BlurShape,
    BlurPath,          // analytical path SDF blur (fill or stroke ring)
    PushClipRect, PopClipRect,      // scissor-only clips (State-side stack)
    PushClipPath, PopClipPath,      // stencil INCR/DECR via ClipPipeline
    COUNT
};

// =============================================================================
// Scope — one entry per clip push (rect or path). Indexed by `DrawCommand::scopeId`.
//
// Scopes form a tree rooted at id 0 (the full viewport). A command inside scope
// S writes only to pixels in scopes[S].bounds (and transitively, only inside
// scopes[S].subtreeBounds for any command deeper in S's subtree). This is the
// hierarchical bbox the CommandScheduler uses to prune overlap queries.
//
//   bounds         set at push time — clip's physical-pixel rect at that scope.
//   subtreeBounds  computed once per frame by Renderer::finalizeScopes() after
//                  recording but before the scheduler runs. Equal to the union
//                  of writesPx over every command in this scope and every
//                  descendant scope.
// =============================================================================

struct Scope {
    uint32_t             parentId = 0;   // 0 == root is its own parent
    juce::Rectangle<int> bounds   {};    // clip bounds at this scope
};

// =============================================================================
// StateKey — 64-bit fusion-compatibility hash. Layout:
//   [ 8b op | 24b pipelineHint | 16b resourceKey | 16b blendKey ]
// pipelineHint differentiates within an op class where a pipeline variant
// differs (e.g. BlurShape with different `mode` uses a different shader).
// resourceKey is a per-op cheap hash of the bound resources. blendKey bundles
// blend + edge placement + inversion for effects. Not cryptographic: collisions
// inside an op just mean the scheduler will attempt fusion that the DAG
// independence check at emit time will then correctly reject.
// =============================================================================

namespace state_key {

inline uint64_t make(DrawOp    op,
                     uint32_t  pipelineHint = 0,   // 24b used
                     uint16_t  resourceKey  = 0,
                     uint16_t  blendKey     = 0) noexcept
{
    uint64_t k = 0;
    k |=  static_cast<uint64_t>(op)                        & 0xFFULL;
    k |= (static_cast<uint64_t>(pipelineHint) & 0xFFFFFFULL) <<  8;
    k |= (static_cast<uint64_t>(resourceKey)  & 0xFFFFULL)   << 32;
    k |= (static_cast<uint64_t>(blendKey)     & 0xFFFFULL)   << 48;
    return k;
}

// Truncate a 64-bit hash (like ResourceCaches::hashImage) into a 16-bit
// resourceKey slot. Collisions cause over-splitting in rare cases —
// correctness-preserving.
inline uint16_t truncHash(uint64_t h) noexcept
{
    return static_cast<uint16_t>((h ^ (h >> 16) ^ (h >> 32) ^ (h >> 48)) & 0xFFFFULL);
}

} // namespace state_key

// =============================================================================
// DrawCommand — one entry in the command vector
//
// Scheduling metadata (writesPx, readsPx, stateKey, scopeId, isOpaque,
// recordOrder) populates at push time and is consumed by CommandScheduler.
// It's all conservative — callers that don't know a tight value can pass
// sentinels (empty rect, 0) and push() fills in safe defaults derived from
// `clipBounds`. When the scheduler runs in Identity mode the metadata is
// recorded but unused, preserving exact current behaviour.
//
//   writesPx   tight physical-pixel bbox of the pixels this command writes.
//              Subset of clipBounds; equal to clipBounds when the caller
//              hasn't computed a shape AABB.
//
//   readsPx    tight physical-pixel bbox of pixels this command reads from
//              the scene buffer. Equal to writesPx for non-effect ops.
//              For blurs: writesPx dilated by the kernel reach.
//
//   stateKey   see Scope.h / state_key::make — the fusion-compatibility hash.
//
//   scopeId    index into Renderer::scopes_. Commands inherit the scope
//              currently active at push time; clip push/pop transitions
//              this via Renderer::pushScope / popScope.
//
//   isOpaque   true iff this command writes alpha=1 everywhere in writesPx.
//              Used by the SortCull pass to delete fully-covered earlier ops.
//              Conservative: default false.
//
//   recordOrder  original insertion index. Stable tiebreaker for scheduler
//                heuristics and reproducible debug dumps.
// =============================================================================

struct DrawCommand {
    float                zOrder;
    DrawOp               op;
    uint32_t             dataOffset;
    uint32_t             dataSize;
    juce::Rectangle<int> clipBounds;     // active scissor at record time
    uint8_t              stencilDepth;
    uint32_t             scopeDepth;

    // Scheduler metadata (Step 1 — see CommandScheduler.md).
    juce::Rectangle<int> writesPx;
    juce::Rectangle<int> readsPx;
    uint64_t             stateKey     = 0;
    uint32_t             scopeId      = 0;
    uint32_t             recordOrder  = 0;
    // Peer-tree depth — computed as a CommandScheduler pre-pass from scopeId
    // + writesPx. Left at 0 by the renderer. Siblings at the same component-
    // hierarchy layer that are spatially disjoint share a peerDepth, which
    // the scheduler uses to cluster them for batching.
    uint32_t             peerDepth    = 0;
    bool                 isOpaque     = false;
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
    T& readMut(uint32_t offset)
    {
        return *reinterpret_cast<T*>(buffer_.data() + offset);
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

    // Bind the stock 2D pipeline. stencilDepth comes from the command so
    // the clip variant is picked per-draw — this is what lets the scheduler
    // reorder commands across PushClipPath boundaries. State's tracked
    // stencilDepth_ is kept as bookkeeping but no longer consulted for
    // pipeline choice or stencilRef.
    void setPipeline(Pipeline* pipeline, uint8_t stencilDepth);

    // User-shader / custom pipeline bind, also parameterised by per-command
    // stencil depth for the same reason.
    void setCustomPipeline(VkPipeline pipeline, VkPipelineLayout layout,
                           uint8_t stencilDepth);
    // set 0 = color source (solid default or gradient LUT), set 1 = shape source
    // (1x1 default, MSDF atlas page, or image texture). Each dirty-tracked.
    void setResources(VkDescriptorSet colorSet, VkDescriptorSet shapeSet);
    void setShapeResource(VkDescriptorSet shapeSet);
    void setColorResource(VkDescriptorSet colorSet);
    void draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count);
    // Push a payload to the vertex-stage push-constant range at the given
    // byte offset (after the viewport already written by setPipeline()).
    void pushConstants(uint32_t offset, uint32_t size, const void* data);

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

    // Record one draw command. Captures the current scopeId from the renderer
    // and a recordOrder tiebreaker; everything else is caller-supplied. Empty
    // writesPx/readsPx fall back to `clip`; stateKey==0 falls back to
    // state_key::make(op). Most call sites have a tight shape AABB and a
    // proper resource key and should pass them explicitly — the defaults
    // exist for ops where no tighter value is cheaply computable.
    template <typename Params>
    void push(DrawOp op, float zOrder,
              const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, uint32_t scopeDepth,
              const Params& params,
              const juce::Rectangle<int>& writesPx = {},
              const juce::Rectangle<int>& readsPx  = {},
              uint64_t stateKey = 0,
              bool isOpaque = false)
    {
        uint32_t offset = arena_.push(params);
        DrawCommand cmd {};
        cmd.zOrder       = zOrder;
        cmd.op           = op;
        cmd.dataOffset   = offset;
        cmd.dataSize     = static_cast<uint32_t>(sizeof(Params));
        cmd.clipBounds   = clip;
        cmd.stencilDepth = stencilDepth;
        cmd.scopeDepth   = scopeDepth;
        cmd.writesPx     = writesPx.isEmpty() ? clip         : writesPx;
        cmd.readsPx      = readsPx.isEmpty()  ? cmd.writesPx : readsPx;
        cmd.stateKey     = stateKey ? stateKey : state_key::make(op);
        cmd.scopeId      = currentScopeId_;
        cmd.recordOrder  = static_cast<uint32_t>(commands_.size());
        cmd.isOpaque     = isOpaque;
        commands_.push_back(cmd);
    }

    // ---- Scope management (clip-push/pop on Graphics calls into these) -----
    // Push a new scope with the supplied clip bounds (physical pixels). Returns
    // the new scope id, which becomes the currentScopeId_ until popScope().
    uint32_t pushScope(const juce::Rectangle<int>& bounds)
    {
        uint32_t id = static_cast<uint32_t>(scopes_.size());
        Scope s {};
        s.parentId = currentScopeId_;
        s.bounds   = bounds;
        scopes_.push_back(s);
        currentScopeId_ = id;
        return id;
    }

    // Pop back to the parent scope. Safe to call redundantly at scope 0 (root
    // is its own parent, so currentScopeId_ stays at 0).
    void popScope()
    {
        currentScopeId_ = scopes_[currentScopeId_].parentId;
    }

    uint32_t currentScopeId() const { return currentScopeId_; }
    const std::vector<Scope>& scopes() const { return scopes_; }

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
        recordingRetains_.push_back(base);
    }

    void registerPipeline(Pipeline& pipeline);

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

    // Attach the geometry-abstracted blur pipeline. Handles both BlurShape
    // (Graphics::{draw,fill}Blurred{Rectangle,RoundedRectangle,Ellipse} +
    // drawBlurredLine) and BlurPath (Graphics::{draw,fill}BlurredPath).
    // Shares PathPipeline's per-frame segment SSBO for the path-geometry
    // branch (tag=5), so it must be set AFTER setPathPipeline().
    void setBlurPipeline(BlurPipeline* bp) { blurPipeline_ = bp; }
    BlurPipeline* blurPipeline() const { return blurPipeline_; }

    // Attach the geometry-abstracted fill pipeline. Handles every non-effect
    // 2D op: rect / rrect / ellipse / strokes / line / image / glyphs / path.
    // Must be set AFTER setPathPipeline() so it can share the segment-SSBO
    // descriptor-set layout for the path-geometry branch.
    void setFillPipeline(FillPipeline* fp) { fillPipeline_ = fp; }
    FillPipeline* fillPipeline() const { return fillPipeline_; }

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

    // Capture non-POD types into side vectors. Returns index.
    uint32_t captureFont(const juce::Font& f)     { fonts_.push_back(f); return static_cast<uint32_t>(fonts_.size() - 1); }
    uint32_t captureFill(const juce::FillType& f)  { fills_.push_back(f); return static_cast<uint32_t>(fills_.size() - 1); }

    const juce::Font&     getFont(uint32_t i) const { return fonts_[i]; }
    const juce::FillType& getFill(uint32_t i) const { return fills_[i]; }

    // Forward to arena for appending POD data after params
    void arena_align(uint32_t alignment) { arena_.align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { arena_.pushSpan(data); }
    template <typename T>
    uint32_t arena_push(const T& data) { return arena_.push(data); }
    template <typename T>
    uint32_t arena_pushSpanReturn(std::span<const T> data) { return arena_.pushSpan(data); }
    // Byte offset where the next push will land. Capture before an
    // arena_pushSpan call to remember that data's location for later
    // cross-command references (e.g. PopClip reusing PushClipPath's verts).
    uint32_t arena_offset() const { return arena_.size(); }
    // Mutable access to arena contents — used by the post-scheduler upload
    // pass to write back firstInstance onto each dispatch ref once the
    // primitive has been uploaded to its pipeline's SSBO. Safe at the
    // single-writer worker-thread call site; do not use during record.
    Arena& arenaMut() { return arena_; }

    // Defined out-of-line in Renderer.cpp — the body touches RenderTarget,
    // which is only forward-declared at this point in the header.
    void reset();

    // Defined in Renderer.cpp. Called from execute() after the scheduler
    // has reordered / fused commands. Walks the final command list and
    // uploads each command's primitive arena span into the appropriate
    // pipeline's per-frame SSBO, writing the resulting `firstInstance`
    // back onto the command's dispatch ref via DispatchRefPrefix.
    // Ensures that same-stateKey runs (including Fuse mode's merged
    // primitive spans) land contiguously in the SSBO so the dispatch
    // loop can issue batched vkCmdDraw(6, primCount, 0, firstInstance)
    // calls without touching upload order.
    void uploadScheduledPrimitives();

    // (Removed finalizeScopes — subtreeBounds is now computed inside
    // CommandScheduler's peer-depth pre-pass, from the same underlying data.)

    Device&         device()   { return device_; }
    RenderTarget&   target()   { return target_; }
    ResourceCaches& caches()   { return device_.caches(); }
    State&          state()    { return state_; }
    Memory::V&      vertices() { return vertices_; }
    const Arena&    arena() const { return arena_; }

    // Post-record pass that reorders / culls / fuses draw commands before
    // dispatch. Defaults to Identity (passthrough). Message thread calls
    // setMode to switch algorithms live; worker thread runs it each frame
    // inside execute().
    CommandScheduler& commandScheduler() { return *scheduler_; }

private:
    Device&       device_;
    RenderTarget& target_;
    State         state_;
    Memory::V     vertices_;

    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    // Scope tree — hierarchical bbox structure used by the CommandScheduler
    // to prune overlap queries. Root (id 0) is the full viewport and is
    // repopulated by reset() each frame. Clip push/pop in Graphics calls
    // into pushScope/popScope above; every recorded command captures the
    // currentScopeId_ at push time.
    std::vector<Scope> scopes_;
    uint32_t           currentScopeId_ = 0;

    // Scheduler is constructed in the Renderer ctor (out of line, where
    // CommandScheduler is complete). unique_ptr so the forward declaration
    // at the top of this header is enough here.
    std::unique_ptr<CommandScheduler> scheduler_;

    // Non-POD captures (proper RAII, cleared each frame)
    std::vector<juce::Font>     fonts_;
    std::vector<juce::FillType> fills_;

    // FrameRetained pins gathered during record (msg thread). At execute
    // time these are moved into retainsBySlot_[slot]; they get unpinned the
    // next time that slot rolls around — i.e. after target_.beginFrame()
    // has waited on the slot's fence and the GPU has provably finished
    // consuming any command buffer that referenced them.
    std::vector<FrameRetained*> recordingRetains_;
    static constexpr int kRetainSlots = 2; // matches MAX_FRAMES_IN_FLIGHT
    std::array<std::vector<FrameRetained*>, kRetainSlots> retainsBySlot_;

    // Process-wide registry of live Renderer instances. Walked by
    // forceDrainAll (invoked from FrameRetained destructors) so a
    // destruction running on the message thread can synchronously drain
    // every renderer's pin counts without depending on the next render
    // tick — which would never come, because the message thread is the
    // very thread blocked in the destructor.
    static juce::CriticalSection& registryLock();
    static std::vector<Renderer*>& registry();

    Pipeline* pipelineForOp_[static_cast<size_t>(DrawOp::COUNT)] = {};
    EffectPipeline*    postProcess_      = nullptr;
    EffectPipeline*    copyEffect_       = nullptr;
    HSVPipeline*       hsvPipeline_      = nullptr;
    BlurPipeline*      blurPipeline_     = nullptr;
    FillPipeline*      fillPipeline_     = nullptr;
    ShaderPipeline*    shaderPipeline_   = nullptr;
    PathPipeline*      pathPipeline_     = nullptr;
    ClipPipeline*      clipPipeline_     = nullptr;
    uint64_t frameCounter_ = 0;

    // ---- Threading ---------------------------------------------------------
    // Vulkan's VkQueue requires external synchronization. Multiple editors
    // share Device's single graphics/present queue, so every queue-submitting
    // Renderer serializes through this lock. Contention is only across
    // editors (within one editor the worker is the sole submitter), and only
    // during the brief submit/present window — not all of execute().
    static juce::CriticalSection& queueLock()
    {
        static juce::CriticalSection lock;
        return lock;
    }

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
