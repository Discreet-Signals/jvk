#pragma once

namespace jvk {

// =============================================================================
// Scene-buffer model
//
// All scene rendering goes to a per-frame-slot pair of sampleable color
// images (`A`, `B`) plus a shared depth-stencil. Draws accumulate on the
// "current" buffer. When an effect (blur, color-correct, etc.) fires, it
// samples the current buffer and writes to the other — the current pointer
// swaps, scene rendering resumes on the new buffer with `loadOp=LOAD` so
// subsequent draws land on top of the effect's output. At end of frame the
// current buffer is blitted to the acquired swapchain image and presented.
//
// Everything is single-sample. Per-pixel AA comes from shader (SDF/MSDF).
// Polygonal path-fill AA can be added as a final subpixel-offset supersample
// effect pass if desired — same mechanism as any other effect.
// =============================================================================

class RenderTarget {
public:
    virtual ~RenderTarget();

    // Per-frame scene-buffer set. Colors A/B ping-pong as effect source/dest;
    // the depth-stencil image is shared across all segments within the frame
    // (stencil state survives effect boundaries via `loadOp=LOAD`).
    struct SceneBuffers {
        Image           colorA;
        Image           colorB;
        Image           depthStencil;
        VkFramebuffer   framebufferA = VK_NULL_HANDLE; // scene RP targeting colorA
        VkFramebuffer   framebufferB = VK_NULL_HANDLE; // scene RP targeting colorB
        VkFramebuffer   effectFBtoA  = VK_NULL_HANDLE; // effect RP writing colorA (sampling B)
        VkFramebuffer   effectFBtoB  = VK_NULL_HANDLE; // effect RP writing colorB (sampling A)
        VkDescriptorSet samplerA     = VK_NULL_HANDLE; // samples colorA
        VkDescriptorSet samplerB     = VK_NULL_HANDLE; // samples colorB
    };

    struct Frame {
        VkCommandBuffer cmd         = VK_NULL_HANDLE;
        VkExtent2D      extent      = {};
        uint32_t        imageIndex  = 0;
        int             frameSlot   = 0;
        VkImage         swapImage   = VK_NULL_HANDLE; // blit target for present
    };

    virtual Frame beginFrame() = 0;
    virtual void  endFrame(const Frame& frame) = 0;
    virtual void  resize(uint32_t w, uint32_t h) = 0;

    // Whether beginFrame can produce a frame right now: the device isn't lost
    // and there is something to present to. The editor asks on the message
    // thread (the render worker idle) before recording a frame: one that
    // can't begin would only queue uploads and pins that no submission ever
    // releases. `revive` tries to make the target presentable again (a
    // swapchain the window lost while minimised, a create that failed); the
    // worker must be idle.
    virtual bool canBegin() const { return !device_.isLost(); }
    virtual void revive() {}

    virtual uint32_t width()  const = 0;
    virtual uint32_t height() const = 0;
    virtual VkFormat format() const = 0;

    // Render passes used by the scene pipeline. Clear-variant for the first
    // segment of a frame, load-variant for continuation segments after an
    // effect. Pipeline-compatible (same attachments/formats/samples), so one
    // pipeline build works with both.
    virtual VkRenderPass sceneRenderPassClear() const = 0;
    virtual VkRenderPass sceneRenderPassLoad()  const = 0;
    // The scene pass a frame actually begins (render worker): clear or load
    // variant, where only the targets in `liveTargets` (bit i = target i: a
    // later command reads it, Renderer::liveTargets) are cleared or loaded
    // and stored; the rest are neither (DONT_CARE). Load and store ops don't
    // affect render-pass compatibility, so every variant runs the same
    // pipelines and framebuffers.
    virtual VkRenderPass sceneRenderPass(bool clear, TargetMask liveTargets)
    {
        juce::ignoreUnused(liveTargets);
        return clear ? sceneRenderPassClear() : sceneRenderPassLoad();
    }
    // Render pass used by effect passes (1 sampleable color attachment).
    virtual VkRenderPass effectRenderPass() const = 0;

    virtual const SceneBuffers& sceneBuffers(int frameSlot) const = 0;

    // ---- Render targets (PipelineConfig.h) ---------------------------------
    // The extra colour targets of the scene render passes, in attachment
    // order after the main colour (target i = output location i + 1; a scaled
    // one is no attachment of theirs). One image each, shared by every frame slot: a frame clears its
    // targets when its first scene pass begins, and the passes' external
    // dependencies order that after the previous frame's reads. Between scene
    // passes (effects, target-reading shaders, the final blit) they sit in
    // SHADER_READ_ONLY_OPTIMAL and can be sampled.
    const std::vector<Target>& targets() const { return targets_; }
    const Image* targetImage(size_t i) const { return i < targetImages_.size() ? &targetImages_[i] : nullptr; }
    // Changes every time the target images are (re)created (a resize, a new
    // set), unique across every RenderTarget in the process. A shader that
    // samples the targets rebinds when it moves: a recreated image can come
    // back with the very same VkImageView handle value, so handles can't say.
    uint64_t targetsGeneration() const { return targetsGeneration_; }
    // Index of the target called `name`, or -1.
    int targetIndex(const juce::Identifier& name) const
    {
        for (size_t i = 0; i < targets_.size(); ++i)
            if (targets_[i].name == name) return static_cast<int>(i);
        return -1;
    }
    // The names of the targets declared but DROPPED (validTargets: over the
    // limit, a format the device can't render). A shader whose output is
    // named after one draws nothing (Shader::resolveTargets) rather than
    // falling back to the target at its location.
    const std::vector<juce::Identifier>& droppedTargets() const { return droppedTargets_; }
    // The scaled targets (Target::scale, Target::height), bit i = target i: not scene-pass attachments.
    TargetMask scaledTargets() const
    {
        TargetMask mask = 0;
        for (size_t i = 0; i < targets_.size(); ++i)
            if (targets_[i].isScaled()) mask |= static_cast<TargetMask>(1u << i);
        return mask;
    }
    // Target i's size relative to the frame's, per side: its scale, or its fixed height over the frame's.
    float targetScale(size_t i) const
    {
        if (i >= targets_.size()) return 1.0f;
        const auto& t = targets_[i];
        return t.height > 0 ? static_cast<float>(t.height) / static_cast<float>(std::max(1u, height())) : t.scale;
    }
    // Target i's size in pixels: the frame's, times its scale, rounded up (a fixed height exactly).
    VkExtent2D targetExtent(size_t i) const
    {
        if (i >= targets_.size() || !targets_[i].isScaled()) return { width(), height() };
        const float s = targetScale(i);
        const auto h = targets_[i].height > 0 ? static_cast<uint32_t>(targets_[i].height)
                                              : static_cast<uint32_t>(std::ceil(static_cast<float>(height()) * s));
        return { std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(width()) * s))), std::max(1u, h) };
    }
    // ---- Passes of their own (Renderer::execute: the render graph) ----------
    //
    // A draw runs inside the scene pass when everything it writes is attached
    // there and nothing it reads is; otherwise as a pass of its own, whose
    // attachments are exactly what it writes. These are those passes. Null
    // where this target can't run one (it is skipped). Render passes live as
    // long as the target; framebuffers, scratches and samplers until its
    // images are made again (a resize).
    //
    // A TARGET PASS: a jvk::Shader drawing into the targets of `slots` (one
    // size) and not the main colour. Its render pass, those targets as
    // attachments at the output locations their shader writes them at (the
    // rest unused), clearing the ones in `clears` first and loading the rest.
    virtual VkRenderPass targetPass(TargetSlots slots, TargetMask clears) { juce::ignoreUnused(slots, clears); return VK_NULL_HANDLE; }
    // The same, or with `main` also the main colour: its current half at
    // location 0, beside targets at scale 1, with the shared stencil so the
    // clip applies.
    virtual VkRenderPass shaderPass(TargetSlots slots, TargetMask clears, bool main)
    {
        return main ? VK_NULL_HANDLE : targetPass(slots, clears);
    }
    // Its framebuffer: `half` (0 = A, 1 = B; -1 = none) of frame slot
    // `frameSlot`'s main colour, then the targets of `slots`, each one in
    // `scratch` drawn into through its SCRATCH instead (an image of its size
    // and format: how a pass writes a target it also reads, Renderer).
    // Render worker, outside any render pass: `cmd` takes the first-use
    // transition of a scratch made by this call.
    virtual VkFramebuffer shaderFramebuffer(int frameSlot, int half, TargetSlots slots, TargetMask scratch, VkCommandBuffer cmd)
    {
        juce::ignoreUnused(frameSlot, half, slots, scratch, cmd);
        return VK_NULL_HANDLE;
    }
    // EFFECTS ON A TARGET (Graphics::setTarget, then a blur or an HSV
    // transform): target i's single-colour render pass (by its format:
    // loading, storing, no stencil), and a framebuffer and a sampler for it
    // or for its scratch. Render worker, outside any render pass.
    virtual VkRenderPass    targetEffectPass(size_t i) { juce::ignoreUnused(i); return VK_NULL_HANDLE; }
    virtual VkFramebuffer   targetEffectFramebuffer(size_t i, bool scratch, VkCommandBuffer cmd) { juce::ignoreUnused(i, scratch, cmd); return VK_NULL_HANDLE; }
    virtual VkDescriptorSet targetSampler(size_t i, bool scratch, VkCommandBuffer cmd) { juce::ignoreUnused(i, scratch, cmd); return VK_NULL_HANDLE; }

    // Each RenderTarget owns a dedicated VkCommandPool. Vulkan command pools
    // are externally synchronized — every vkCmd* recording call on any
    // buffer from the pool counts as use. Sharing a single pool across
    // plugin instances means each editor's jvk-render-worker thread can
    // race on MoltenVK's internal pool state; a torn pointer there caused
    // a byte-write into __DATA_CONST (SIGBUS) in multi-instance hosts.
    // One pool per target = one pool per worker thread, no contention.
    VkCommandPool commandPool() const { return commandPool_; }

protected:
    Device& device_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    RenderTarget(Device& device);

    std::vector<Target> targets_;
    std::vector<Image>   targetImages_;
    uint64_t             targetsGeneration_ = 0;
    std::vector<juce::Identifier> droppedTargets_;   // validTargets: the names it dropped

    // The targets this device can carry: independentBlend is required (a
    // draw writes one attachment and leaves the rest), and the main colour
    // plus the targets must fit the device's colour-attachment limit and the
    // built-in shaders' 8 outputs. The rest are dropped, with a log line,
    // their names kept (droppedTargets).
    std::vector<Target> validTargets(std::vector<Target> requested);
};


// =============================================================================
// SwapchainTarget — renders to a window via VkSwapchainKHR
// =============================================================================

class SwapchainTarget : public RenderTarget {
public:
    // `nativeWindow` is the platform window the surface was created from
    // (Win32: the child HWND; unused elsewhere). On Windows every driver
    // call whose result depends on the window's client rect — capability
    // queries, acquire, present — runs with the calling thread pinned to
    // that window's DPI-awareness context, so the size the driver reports
    // is the same on every thread (see the SurfaceDpiScope note in
    // RenderTarget.cpp). Pass null only for targets that never need
    // consistent surface metrics (tests, offscreen experiments).
    SwapchainTarget(Device& device, VkSurfaceKHR surface,
                    uint32_t w, uint32_t h,
                    void* nativeWindow = nullptr,
                    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR,
                    std::vector<Target> targets = {});
    ~SwapchainTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    // Presentable: a swapchain exists (its creation fails while the window is
    // minimised: a zero extent) on a device that isn't lost. Message thread
    // with the worker idle, or the worker: both are the only writers of the
    // swapchain handle and never run at once.
    bool     canBegin() const override { return swapchain_ != VK_NULL_HANDLE && !device_.isLost(); }
    void     revive() override;
    uint32_t width()  const override { return width_; }
    uint32_t height() const override { return height_; }
    VkFormat format() const override { return format_; }

    // The window/layer's current physical pixel size, straight from the driver
    // (Win32: child HWND client area; Metal: CAMetalLayer drawableSize). This
    // is the ground truth a caller should resize the swapchain to. Returns the
    // 0xFFFFFFFF sentinel in .width when the surface has no preferred size
    // (some Wayland compositors); callers then fall back to a logical*scale
    // estimate. Cheap — a single vkGetPhysicalDeviceSurfaceCapabilitiesKHR.
    VkExtent2D surfaceExtent() const;

    VkRenderPass sceneRenderPassClear() const override { return sceneRPClear_; }
    VkRenderPass sceneRenderPassLoad()  const override { return sceneRPLoad_;  }
    VkRenderPass sceneRenderPass(bool clear, TargetMask liveTargets) override;
    VkRenderPass effectRenderPass()     const override { return effectRP_;     }
    VkRenderPass    targetPass(TargetSlots slots, TargetMask clears) override;
    VkRenderPass    shaderPass(TargetSlots slots, TargetMask clears, bool main) override;
    VkFramebuffer   shaderFramebuffer(int frameSlot, int half, TargetSlots slots, TargetMask scratch, VkCommandBuffer cmd) override;
    VkRenderPass    targetEffectPass(size_t i) override;
    VkFramebuffer   targetEffectFramebuffer(size_t i, bool scratch, VkCommandBuffer cmd) override;
    VkDescriptorSet targetSampler(size_t i, bool scratch, VkCommandBuffer cmd) override;

    const SceneBuffers& sceneBuffers(int frameSlot) const override
    {
        return sceneBuffers_[frameSlot];
    }

private:
    void createSwapchain();
    void createRenderPasses();
    VkRenderPass createSceneRenderPass(bool clear, TargetMask liveTargets) const;
    void createSceneBuffers();
    void createSyncObjects();
    void destroySwapchain();
    void destroySceneBuffers();
    // Target i's scratch, made on first use (its first-use transition recorded into `cmd`).
    const Image* scratch(size_t i, VkCommandBuffer cmd);

    VkSurfaceKHR   surface_;
    void*          nativeWindow_ = nullptr; // Win32 HWND behind surface_ (DPI-context pinning)
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat       format_    = VK_FORMAT_B8G8R8A8_UNORM;
    VkPresentModeKHR presentMode_;
    uint32_t       width_  = 0;
    uint32_t       height_ = 0;
    // Consecutive fence waits that timed out (beginFrame): one stall is a
    // skipped frame, kMaxFenceTimeouts in a row a lost device.
    int            fenceTimeouts_ = 0;
    static constexpr int kMaxFenceTimeouts = 5;

    std::vector<VkImage> swapImages_;

    // Three render passes total:
    //   sceneRPClear_  : scene attachments, loadOp=CLEAR  (frame start)
    //   sceneRPLoad_   : scene attachments, loadOp=LOAD   (after effect)
    //   effectRP_      : 1 color attachment, loadOp=DONT_CARE (ping-pong write)
    VkRenderPass sceneRPClear_ = VK_NULL_HANDLE;
    VkRenderPass sceneRPLoad_  = VK_NULL_HANDLE;
    VkRenderPass effectRP_     = VK_NULL_HANDLE;
    // sceneRenderPass variants with some targets dead, made on first use by
    // the render worker (at most two per distinct live set a frame passes
    // through). Keyed clear << 16 | live.
    std::vector<std::pair<uint32_t, VkRenderPass>> sceneRPVariants_;
    // Passes of their own, made on first use by the render worker: render passes keyed by
    // their slots and clears (with the main colour or not), and the effect pass of each
    // target format (they outlive resizes); framebuffers, scratches and samplers (remade
    // with the target images).
    std::vector<std::pair<std::pair<TargetSlots, TargetMask>, VkRenderPass>> targetPasses_, mainPasses_;
    std::vector<std::pair<PixelFormat, VkRenderPass>>                        effectPasses_;
    struct FramebufferKey
    {
        int frameSlot, half; TargetSlots slots; TargetMask scratch;
        bool operator== (const FramebufferKey& o) const
        {
            return frameSlot == o.frameSlot && half == o.half && slots == o.slots && scratch == o.scratch;
        }
    };
    std::vector<std::pair<FramebufferKey, VkFramebuffer>>                    shaderFramebuffers_;
    std::vector<std::pair<std::pair<size_t, bool>, VkFramebuffer>>          effectFramebuffers_;
    std::vector<Image>                                                       scratches_;     // by target, made on first use
    std::vector<VkDescriptorSet>                                             samplers_[2];   // by target: [0] its own, [1] its scratch's

    static constexpr int MAX_FRAMES = 2;
    SceneBuffers sceneBuffers_[MAX_FRAMES];

    // imageAvailable per-frame-slot (acquire target, consumed by submit wait);
    // renderFinished per-swap-image (submit signal, consumed by present wait).
    VkSemaphore imageAvailable_[MAX_FRAMES] {};
    VkFence     inFlightFence_[MAX_FRAMES]  {};
    VkCommandBuffer commandBuffers_[MAX_FRAMES] {};
    std::vector<VkSemaphore> renderFinished_;
    int currentFrame_ = 0;
};


// =============================================================================
// OffscreenTarget — renders to an Image (no post-process support)
// =============================================================================

class OffscreenTarget : public RenderTarget {
public:
    OffscreenTarget(Device& device, uint32_t w, uint32_t h,
                    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
    ~OffscreenTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()  const override { return width_; }
    uint32_t height() const override { return height_; }
    VkFormat format() const override { return format_; }

    VkRenderPass sceneRenderPassClear() const override { return renderPass_; }
    VkRenderPass sceneRenderPassLoad()  const override { return renderPass_; }
    VkRenderPass effectRenderPass()     const override { return renderPass_; }

    const SceneBuffers& sceneBuffers(int) const override { return sceneBuffers_; }

    Image& getImage() { return renderImage_; }

private:
    void create();
    void destroy();

    VkRenderPass    renderPass_   = VK_NULL_HANDLE;
    VkFramebuffer   framebuffer_  = VK_NULL_HANDLE;
    VkCommandBuffer cmd_          = VK_NULL_HANDLE;
    VkFence         fence_        = VK_NULL_HANDLE;
    Image           renderImage_;
    Image           depthStencil_;
    VkFormat        format_;
    uint32_t        width_  = 0;
    uint32_t        height_ = 0;
    SceneBuffers    sceneBuffers_ {};
};

} // namespace jvk
