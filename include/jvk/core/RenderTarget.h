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

    virtual uint32_t width()  const = 0;
    virtual uint32_t height() const = 0;
    virtual VkFormat format() const = 0;

    // Render passes used by the scene pipeline. Clear-variant for the first
    // segment of a frame, load-variant for continuation segments after an
    // effect. Pipeline-compatible (same attachments/formats/samples), so one
    // pipeline build works with both.
    virtual VkRenderPass sceneRenderPassClear() const = 0;
    virtual VkRenderPass sceneRenderPassLoad()  const = 0;
    // Render pass used by effect passes (1 sampleable color attachment).
    virtual VkRenderPass effectRenderPass() const = 0;

    virtual const SceneBuffers& sceneBuffers(int frameSlot) const = 0;

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
};


// =============================================================================
// SwapchainTarget — renders to a window via VkSwapchainKHR
// =============================================================================

class SwapchainTarget : public RenderTarget {
public:
    SwapchainTarget(Device& device, VkSurfaceKHR surface,
                    uint32_t w, uint32_t h,
                    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR);
    ~SwapchainTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()  const override { return width_; }
    uint32_t height() const override { return height_; }
    VkFormat format() const override { return format_; }

    VkRenderPass sceneRenderPassClear() const override { return sceneRPClear_; }
    VkRenderPass sceneRenderPassLoad()  const override { return sceneRPLoad_;  }
    VkRenderPass effectRenderPass()     const override { return effectRP_;     }

    const SceneBuffers& sceneBuffers(int frameSlot) const override
    {
        return sceneBuffers_[frameSlot];
    }

private:
    void createSwapchain();
    void createRenderPasses();
    void createSceneBuffers();
    void createSyncObjects();
    void destroySwapchain();
    void destroySceneBuffers();

    VkSurfaceKHR   surface_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat       format_    = VK_FORMAT_B8G8R8A8_UNORM;
    VkPresentModeKHR presentMode_;
    uint32_t       width_  = 0;
    uint32_t       height_ = 0;

    std::vector<VkImage>     swapImages_;
    std::vector<VkImageView> swapImageViews_;

    // Three render passes total:
    //   sceneRPClear_  : scene attachments, loadOp=CLEAR  (frame start)
    //   sceneRPLoad_   : scene attachments, loadOp=LOAD   (after effect)
    //   effectRP_      : 1 color attachment, loadOp=DONT_CARE (ping-pong write)
    VkRenderPass sceneRPClear_ = VK_NULL_HANDLE;
    VkRenderPass sceneRPLoad_  = VK_NULL_HANDLE;
    VkRenderPass effectRP_     = VK_NULL_HANDLE;

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
