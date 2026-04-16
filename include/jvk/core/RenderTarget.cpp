namespace jvk {

// =============================================================================
// SwapchainTarget
// =============================================================================

SwapchainTarget::SwapchainTarget(Device& device, VkSurfaceKHR surface,
                                 uint32_t w, uint32_t h,
                                 VkSampleCountFlagBits msaa,
                                 VkPresentModeKHR presentMode)
    : RenderTarget(device), surface_(surface), presentMode_(presentMode), msaa_(msaa),
      width_(w), height_(h)
{
    createSwapchain();
    createRenderPass();
    createFramebuffers();
    createSyncObjects();
}

SwapchainTarget::~SwapchainTarget()
{
    VkDevice d = device_.device();
    vkDeviceWaitIdle(d);

    for (int i = 0; i < MAX_FRAMES; i++) {
        vkDestroySemaphore(d, imageAvailable_[i], nullptr);
        vkDestroySemaphore(d, renderFinished_[i], nullptr);
        vkDestroyFence(d, inFlightFence_[i], nullptr);
    }
    vkFreeCommandBuffers(d, device_.commandPool(), MAX_FRAMES, commandBuffers_);
    destroySwapchain();
    if (renderPass_ != VK_NULL_HANDLE) vkDestroyRenderPass(d, renderPass_, nullptr);
    vkDestroySurfaceKHR(device_.instance(), surface_, nullptr);
}

void SwapchainTarget::createSwapchain()
{
    VkDevice d = device_.device();
    VkPhysicalDevice pd = device_.physicalDevice();

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface_, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, fmts.data());

    format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { colorSpace = f.colorSpace; break; }
    }

    VkExtent2D ext;
    ext.width = std::clamp(width_, caps.minImageExtent.width, caps.maxImageExtent.width);
    ext.height = std::clamp(height_, caps.minImageExtent.height, caps.maxImageExtent.height);
    width_ = ext.width;
    height_ = ext.height;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci {};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = format_;
    ci.imageColorSpace = colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    VkCompositeAlphaFlagBitsKHR prefs[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
    };
    for (auto p : prefs) { if (caps.supportedCompositeAlpha & p) { ci.compositeAlpha = p; break; } }

    ci.presentMode = presentMode_;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain_;

    vkCreateSwapchainKHR(d, &ci, nullptr, &swapchain_);

    if (ci.oldSwapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(d, ci.oldSwapchain, nullptr);

    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, nullptr);
    swapImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, swapImages_.data());

    swapImageViews_.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = swapImages_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format_;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(d, &vi, nullptr, &swapImageViews_[i]);
    }

    // MSAA color image (shared)
    msaaColorImage_ = Image(device_.pool(), d, width_, height_, format_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        msaa_);

    // Depth/stencil image (shared)
    depthStencilImage_ = Image(device_.pool(), d, width_, height_,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        msaa_,
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
}

void SwapchainTarget::createRenderPass()
{
    if (renderPass_ != VK_NULL_HANDLE) return;
    VkDevice d = device_.device();

    VkAttachmentDescription attachments[3] {};
    // 0: MSAA color
    attachments[0].format = format_;
    attachments[0].samples = msaa_;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // 1: Resolve (swapchain image)
    attachments[1].format = format_;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // 2: Depth/stencil
    attachments[2].format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    attachments[2].samples = msaa_;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference resolveRef = { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef = { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pResolveAttachments = &resolveRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo rpci {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 3;
    rpci.pAttachments = attachments;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;

    vkCreateRenderPass(d, &rpci, nullptr, &renderPass_);
}

void SwapchainTarget::createFramebuffers()
{
    VkDevice d = device_.device();
    framebuffers_.resize(swapImages_.size());

    for (size_t i = 0; i < swapImages_.size(); i++) {
        VkImageView views[] = {
            msaaColorImage_.view(),
            swapImageViews_[i],
            depthStencilImage_.view()
        };

        VkFramebufferCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = renderPass_;
        fci.attachmentCount = 3;
        fci.pAttachments = views;
        fci.width = width_;
        fci.height = height_;
        fci.layers = 1;
        vkCreateFramebuffer(d, &fci, nullptr, &framebuffers_[i]);
    }
}

void SwapchainTarget::createSyncObjects()
{
    VkDevice d = device_.device();

    VkSemaphoreCreateInfo sci {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES; i++) {
        vkCreateSemaphore(d, &sci, nullptr, &imageAvailable_[i]);
        vkCreateSemaphore(d, &sci, nullptr, &renderFinished_[i]);
        vkCreateFence(d, &fci, nullptr, &inFlightFence_[i]);
    }

    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = device_.commandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES;
    vkAllocateCommandBuffers(d, &ai, commandBuffers_);
}

void SwapchainTarget::destroySwapchain()
{
    VkDevice d = device_.device();
    for (auto fb : framebuffers_) vkDestroyFramebuffer(d, fb, nullptr);
    framebuffers_.clear();
    for (auto v : swapImageViews_) vkDestroyImageView(d, v, nullptr);
    swapImageViews_.clear();
    swapImages_.clear();
    msaaColorImage_ = {};
    depthStencilImage_ = {};
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

RenderTarget::Frame SwapchainTarget::beginFrame()
{
    VkDevice d = device_.device();
    vkWaitForFences(d, 1, &inFlightFence_[currentFrame_], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(d, swapchain_, UINT64_MAX,
        imageAvailable_[currentFrame_], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize(width_, height_);
        return {}; // caller checks cmd == VK_NULL_HANDLE
    }

    vkResetFences(d, 1, &inFlightFence_[currentFrame_]);
    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffers_[currentFrame_], &bi);

    return {
        commandBuffers_[currentFrame_],
        framebuffers_[imageIndex],
        { width_, height_ },
        imageIndex,
        currentFrame_,
        swapImages_[imageIndex]
    };
}

void SwapchainTarget::endFrame(const Frame& frame)
{
    VkDevice d = device_.device();
    vkEndCommandBuffer(frame.cmd);

    VkSemaphore waitSems[] = { imageAvailable_[currentFrame_] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore sigSems[] = { renderFinished_[currentFrame_] };

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = waitSems;
    si.pWaitDstStageMask = waitStages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frame.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = sigSems;

    vkQueueSubmit(device_.graphicsQueue(), 1, &si, inFlightFence_[currentFrame_]);

    VkSwapchainKHR swapchains[] = { swapchain_ };
    VkPresentInfoKHR pi {};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = sigSems;
    pi.swapchainCount = 1;
    pi.pSwapchains = swapchains;
    pi.pImageIndices = &frame.imageIndex;

    vkQueuePresentKHR(device_.presentQueue(), &pi);
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
}

void SwapchainTarget::resize(uint32_t w, uint32_t h)
{
    VkDevice d = device_.device();
    vkDeviceWaitIdle(d);
    width_ = w;
    height_ = h;
    destroySwapchain();
    createSwapchain();
    createFramebuffers();
}


// =============================================================================
// OffscreenTarget
// =============================================================================

OffscreenTarget::OffscreenTarget(Device& device, uint32_t w, uint32_t h, VkFormat format)
    : RenderTarget(device), format_(format), width_(w), height_(h)
{
    create();
}

OffscreenTarget::~OffscreenTarget()
{
    destroy();
}

void OffscreenTarget::create()
{
    VkDevice d = device_.device();

    renderImage_ = Image(device_.pool(), d, width_, height_, format_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    // Render pass: single color attachment, no MSAA, no depth
    VkAttachmentDescription att {};
    att.format = format_;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpci {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    vkCreateRenderPass(d, &rpci, nullptr, &renderPass_);

    VkImageView views[] = { renderImage_.view() };
    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = renderPass_;
    fci.attachmentCount = 1;
    fci.pAttachments = views;
    fci.width = width_;
    fci.height = height_;
    fci.layers = 1;
    vkCreateFramebuffer(d, &fci, nullptr, &framebuffer_);

    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = device_.commandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(d, &ai, &cmd_);

    VkFenceCreateInfo fenceCI {};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(d, &fenceCI, nullptr, &fence_);
}

void OffscreenTarget::destroy()
{
    VkDevice d = device_.device();
    if (d == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(d);

    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(d, fence_, nullptr);
    if (cmd_ != VK_NULL_HANDLE) vkFreeCommandBuffers(d, device_.commandPool(), 1, &cmd_);
    if (framebuffer_ != VK_NULL_HANDLE) vkDestroyFramebuffer(d, framebuffer_, nullptr);
    if (renderPass_ != VK_NULL_HANDLE) vkDestroyRenderPass(d, renderPass_, nullptr);
    renderImage_ = {};
    fence_ = VK_NULL_HANDLE;
    cmd_ = VK_NULL_HANDLE;
    framebuffer_ = VK_NULL_HANDLE;
    renderPass_ = VK_NULL_HANDLE;
}

RenderTarget::Frame OffscreenTarget::beginFrame()
{
    VkDevice d = device_.device();
    vkWaitForFences(d, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(d, 1, &fence_);
    vkResetCommandBuffer(cmd_, 0);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_, &bi);

    return { cmd_, framebuffer_, { width_, height_ }, 0, 0, VK_NULL_HANDLE };
}

void OffscreenTarget::endFrame(const Frame& frame)
{
    vkEndCommandBuffer(frame.cmd);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frame.cmd;
    vkQueueSubmit(device_.graphicsQueue(), 1, &si, fence_);
}

void OffscreenTarget::resize(uint32_t w, uint32_t h)
{
    destroy();
    width_ = w;
    height_ = h;
    create();
}

} // namespace jvk
