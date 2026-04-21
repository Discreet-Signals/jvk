#pragma once

namespace jvk {

class Pipeline;
class RenderTarget;
class EffectPipeline;
class HSVPipeline;
class ShapeBlurPipeline;
class PathBlurPipeline;
class ShaderPipeline;
class PathPipeline;
class ClipPipeline;

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
// DrawCommand — one entry in the command vector
// =============================================================================

struct DrawCommand {
    float                zOrder;
    DrawOp               op;
    uint32_t             dataOffset;
    uint32_t             dataSize;
    juce::Rectangle<int> clipBounds;
    uint8_t              stencilDepth;
    uint32_t             scopeDepth;
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

    template <typename Params>
    void push(DrawOp op, float zOrder, const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, uint32_t scopeDepth, const Params& params)
    {
        uint32_t offset = arena_.push(params);
        commands_.push_back({
            zOrder, op, offset, static_cast<uint32_t>(sizeof(Params)),
            clip, stencilDepth, scopeDepth
        });
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

    // Capture non-POD types into side vectors. Returns index.
    uint32_t captureFont(const juce::Font& f)     { fonts_.push_back(f); return static_cast<uint32_t>(fonts_.size() - 1); }
    uint32_t captureFill(const juce::FillType& f)  { fills_.push_back(f); return static_cast<uint32_t>(fills_.size() - 1); }

    const juce::Font&     getFont(uint32_t i) const { return fonts_[i]; }
    const juce::FillType& getFill(uint32_t i) const { return fills_[i]; }

    // Forward to arena for appending POD data after params
    void arena_align(uint32_t alignment) { arena_.align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { arena_.pushSpan(data); }
    // Byte offset where the next push will land. Capture before an
    // arena_pushSpan call to remember that data's location for later
    // cross-command references (e.g. PopClip reusing PushClipPath's verts).
    uint32_t arena_offset() const { return arena_.size(); }

    void reset()
    {
        commands_.clear();
        arena_.reset();
        fonts_.clear();
        fills_.clear();
    }

    Device&         device()   { return device_; }
    RenderTarget&   target()   { return target_; }
    ResourceCaches& caches()   { return device_.caches(); }
    State&          state()    { return state_; }
    Memory::V&      vertices() { return vertices_; }
    const Arena&    arena() const { return arena_; }

private:
    Device&       device_;
    RenderTarget& target_;
    State         state_;
    Memory::V     vertices_;

    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    // Non-POD captures (proper RAII, cleared each frame)
    std::vector<juce::Font>     fonts_;
    std::vector<juce::FillType> fills_;

    Pipeline* pipelineForOp_[static_cast<size_t>(DrawOp::COUNT)] = {};
    EffectPipeline*    postProcess_      = nullptr;
    EffectPipeline*    copyEffect_       = nullptr;
    HSVPipeline*       hsvPipeline_      = nullptr;
    ShapeBlurPipeline* shapeBlur_        = nullptr;
    PathBlurPipeline*  pathBlur_         = nullptr;
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
