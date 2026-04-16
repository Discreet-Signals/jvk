#pragma once

namespace jvk {

class RenderTarget {
public:
    virtual ~RenderTarget() = default;

    struct Frame {
        VkCommandBuffer cmd          = VK_NULL_HANDLE;
        VkFramebuffer   framebuffer  = VK_NULL_HANDLE;
        VkExtent2D      extent       = {};
        uint32_t        imageIndex   = 0;
        int             frameSlot    = 0;
        VkImage         resolveImage = VK_NULL_HANDLE;
    };

    virtual Frame beginFrame() = 0;
    virtual void  endFrame(const Frame& frame) = 0;
    virtual void  resize(uint32_t w, uint32_t h) = 0;

    virtual uint32_t              width()       const = 0;
    virtual uint32_t              height()      const = 0;
    virtual VkFormat              format()      const = 0;
    virtual VkSampleCountFlagBits msaaSamples() const = 0;
    virtual VkRenderPass          renderPass()  const = 0;

protected:
    Device& device_;
    RenderTarget(Device& device) : device_(device) {}
};


// =============================================================================
// SwapchainTarget — renders to a window via VkSwapchainKHR
// =============================================================================

class SwapchainTarget : public RenderTarget {
public:
    SwapchainTarget(Device& device, VkSurfaceKHR surface,
                    uint32_t w, uint32_t h,
                    VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_2_BIT,
                    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR);
    ~SwapchainTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()       const override { return width_; }
    uint32_t height()      const override { return height_; }
    VkFormat format()      const override { return format_; }
    VkSampleCountFlagBits msaaSamples() const override { return msaa_; }
    VkRenderPass renderPass() const override { return renderPass_; }

private:
    void createSwapchain();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();
    void destroySwapchain();

    VkSurfaceKHR   surface_;
    VkSwapchainKHR swapchain_  = VK_NULL_HANDLE;
    VkRenderPass   renderPass_ = VK_NULL_HANDLE;
    VkFormat       format_     = VK_FORMAT_B8G8R8A8_UNORM;
    VkPresentModeKHR presentMode_;
    VkSampleCountFlagBits msaa_;
    uint32_t       width_  = 0;
    uint32_t       height_ = 0;

    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapImageViews_;
    std::vector<VkFramebuffer> framebuffers_;

    // MSAA color image (shared across all swapchain images)
    Image msaaColorImage_;
    // Depth/stencil image (shared)
    Image depthStencilImage_;

    static constexpr int MAX_FRAMES = 2;
    VkSemaphore imageAvailable_[MAX_FRAMES] {};
    VkSemaphore renderFinished_[MAX_FRAMES] {};
    VkFence     inFlightFence_[MAX_FRAMES] {};
    VkCommandBuffer commandBuffers_[MAX_FRAMES] {};
    int currentFrame_ = 0;
};


// =============================================================================
// OffscreenTarget — renders to an Image
// =============================================================================

class OffscreenTarget : public RenderTarget {
public:
    OffscreenTarget(Device& device, uint32_t w, uint32_t h,
                    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
    ~OffscreenTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()       const override { return width_; }
    uint32_t height()      const override { return height_; }
    VkFormat format()      const override { return format_; }
    VkSampleCountFlagBits msaaSamples() const override { return VK_SAMPLE_COUNT_1_BIT; }
    VkRenderPass renderPass() const override { return renderPass_; }

    Image& getImage() { return renderImage_; }

private:
    void create();
    void destroy();

    VkRenderPass    renderPass_  = VK_NULL_HANDLE;
    VkFramebuffer   framebuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_         = VK_NULL_HANDLE;
    VkFence         fence_       = VK_NULL_HANDLE;
    Image           renderImage_;
    VkFormat        format_;
    uint32_t        width_  = 0;
    uint32_t        height_ = 0;
};

} // namespace jvk
