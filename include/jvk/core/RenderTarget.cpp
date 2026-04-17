namespace jvk {

// =============================================================================
// SwapchainTarget
// =============================================================================

SwapchainTarget::SwapchainTarget(Device& device, VkSurfaceKHR surface,
                                 uint32_t w, uint32_t h,
                                 VkPresentModeKHR presentMode)
    : RenderTarget(device), surface_(surface), presentMode_(presentMode),
      width_(w), height_(h)
{
    createSwapchain();
    createRenderPasses();
    createSceneBuffers();
    createSyncObjects();
}

SwapchainTarget::~SwapchainTarget()
{
    VkDevice d = device_.device();
    vkDeviceWaitIdle(d);

    for (int i = 0; i < MAX_FRAMES; i++) {
        vkDestroySemaphore(d, imageAvailable_[i], nullptr);
        vkDestroyFence(d, inFlightFence_[i], nullptr);
    }
    for (auto sem : renderFinished_) vkDestroySemaphore(d, sem, nullptr);
    vkFreeCommandBuffers(d, device_.commandPool(), MAX_FRAMES, commandBuffers_);

    destroySceneBuffers();
    destroySwapchain();
    if (sceneRPClear_ != VK_NULL_HANDLE) vkDestroyRenderPass(d, sceneRPClear_, nullptr);
    if (sceneRPLoad_  != VK_NULL_HANDLE) vkDestroyRenderPass(d, sceneRPLoad_,  nullptr);
    if (effectRP_     != VK_NULL_HANDLE) vkDestroyRenderPass(d, effectRP_,     nullptr);
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

    // Triple-buffering per MoltenVK guidance — at least 3 images to avoid
    // Direct-to-Display drawable hold causing acquire stalls.
    uint32_t imageCount = std::max(caps.minImageCount + 1, 3u);
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
    // Scene renders to offscreen, we blit into swap images → need TRANSFER_DST.
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
}

void SwapchainTarget::createRenderPasses()
{
    VkDevice d = device_.device();

    // ---- Scene render passes (clear + load variants) ----
    // 0: Color (sampleable) — finalLayout=SHADER_READ_ONLY so next effect/blit samples it
    // 1: Depth/stencil — preserved across segments via SHADER_READ_ONLY-ish layout? No —
    //    depth is written, not sampled. Keep it in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    VkAttachmentDescription sceneAtts[2] {};

    // Color
    sceneAtts[0].format         = format_;
    sceneAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    sceneAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    sceneAtts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    sceneAtts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Depth/stencil
    sceneAtts[1].format         = VK_FORMAT_D32_SFLOAT_S8_UINT;
    sceneAtts[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    sceneAtts[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub {};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    // Two-sided dep: incoming (prior effect's sampler read completes before
    // we write color) + outgoing (our color write visible to the next effect's
    // sampler read).
    VkSubpassDependency deps[2] {};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT
                          | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpci {};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments    = sceneAtts;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;

    // Clear variant (frame start)
    sceneAtts[0].loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
    sceneAtts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    sceneAtts[1].loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
    sceneAtts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    sceneAtts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateRenderPass(d, &rpci, nullptr, &sceneRPClear_);

    // Load variant (continuation after effect)
    sceneAtts[0].loadOp        = VK_ATTACHMENT_LOAD_OP_LOAD;
    sceneAtts[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sceneAtts[1].loadOp        = VK_ATTACHMENT_LOAD_OP_LOAD;
    sceneAtts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    sceneAtts[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    vkCreateRenderPass(d, &rpci, nullptr, &sceneRPLoad_);

    // ---- Effect render pass (1 attachment, sample src → write dst) ----
    VkAttachmentDescription effectAtt {};
    effectAtt.format         = format_;
    effectAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    effectAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    effectAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    effectAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    effectAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    effectAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    effectAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference effectColorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription effectSub {};
    effectSub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    effectSub.colorAttachmentCount = 1;
    effectSub.pColorAttachments    = &effectColorRef;

    VkSubpassDependency effectDeps[2] {};
    effectDeps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    effectDeps[0].dstSubpass    = 0;
    effectDeps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    effectDeps[0].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    effectDeps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    effectDeps[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    effectDeps[1].srcSubpass    = 0;
    effectDeps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    effectDeps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    effectDeps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                | VK_PIPELINE_STAGE_TRANSFER_BIT;
    effectDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    effectDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                                | VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo effectRPCI {};
    effectRPCI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    effectRPCI.attachmentCount = 1;
    effectRPCI.pAttachments    = &effectAtt;
    effectRPCI.subpassCount    = 1;
    effectRPCI.pSubpasses      = &effectSub;
    effectRPCI.dependencyCount = 2;
    effectRPCI.pDependencies   = effectDeps;
    vkCreateRenderPass(d, &effectRPCI, nullptr, &effectRP_);
}

void SwapchainTarget::createSceneBuffers()
{
    VkDevice d = device_.device();
    for (int i = 0; i < MAX_FRAMES; i++) {
        auto& sb = sceneBuffers_[i];
        sb.colorA = Image(device_.pool(), d, width_, height_, format_,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
          | VK_IMAGE_USAGE_SAMPLED_BIT
          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        sb.colorB = Image(device_.pool(), d, width_, height_, format_,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
          | VK_IMAGE_USAGE_SAMPLED_BIT
          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        sb.depthStencil = Image(device_.pool(), d, width_, height_,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

        // Scene framebuffers (color + depth). Clear and load variants share
        // the same layout, so one framebuffer targets both.
        VkImageView viewsA[2] = { sb.colorA.view(), sb.depthStencil.view() };
        VkImageView viewsB[2] = { sb.colorB.view(), sb.depthStencil.view() };
        VkFramebufferCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = sceneRPClear_; // compatible with sceneRPLoad_ too
        fci.attachmentCount = 2;
        fci.width = width_;
        fci.height = height_;
        fci.layers = 1;
        fci.pAttachments = viewsA;
        vkCreateFramebuffer(d, &fci, nullptr, &sb.framebufferA);
        fci.pAttachments = viewsB;
        vkCreateFramebuffer(d, &fci, nullptr, &sb.framebufferB);

        // Effect framebuffers (1 color attachment). When writing to A we
        // sample from B, and vice versa.
        VkImageView effA[1] = { sb.colorA.view() };
        VkImageView effB[1] = { sb.colorB.view() };
        fci.renderPass = effectRP_;
        fci.attachmentCount = 1;
        fci.pAttachments = effA;
        vkCreateFramebuffer(d, &fci, nullptr, &sb.effectFBtoA);
        fci.pAttachments = effB;
        vkCreateFramebuffer(d, &fci, nullptr, &sb.effectFBtoB);

        // Sampler descriptors for reading A and B.
        sb.samplerA = device_.bindings().alloc();
        sb.samplerB = device_.bindings().alloc();
        Memory::M::writeImage(d, sb.samplerA, 0, sb.colorA.view(), sb.colorA.sampler());
        Memory::M::writeImage(d, sb.samplerB, 0, sb.colorB.view(), sb.colorB.sampler());
    }
}

void SwapchainTarget::destroySceneBuffers()
{
    VkDevice d = device_.device();
    for (int i = 0; i < MAX_FRAMES; i++) {
        auto& sb = sceneBuffers_[i];
        if (sb.framebufferA != VK_NULL_HANDLE) vkDestroyFramebuffer(d, sb.framebufferA, nullptr);
        if (sb.framebufferB != VK_NULL_HANDLE) vkDestroyFramebuffer(d, sb.framebufferB, nullptr);
        if (sb.effectFBtoA  != VK_NULL_HANDLE) vkDestroyFramebuffer(d, sb.effectFBtoA,  nullptr);
        if (sb.effectFBtoB  != VK_NULL_HANDLE) vkDestroyFramebuffer(d, sb.effectFBtoB,  nullptr);
        if (sb.samplerA     != VK_NULL_HANDLE) device_.bindings().free(sb.samplerA);
        if (sb.samplerB     != VK_NULL_HANDLE) device_.bindings().free(sb.samplerB);
        sb = {};
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
        vkCreateFence(d, &fci, nullptr, &inFlightFence_[i]);
    }
    renderFinished_.resize(swapImages_.size(), VK_NULL_HANDLE);
    for (auto& sem : renderFinished_)
        vkCreateSemaphore(d, &sci, nullptr, &sem);

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
    for (auto v : swapImageViews_) vkDestroyImageView(d, v, nullptr);
    swapImageViews_.clear();
    swapImages_.clear();
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

    Frame f {};
    f.cmd        = commandBuffers_[currentFrame_];
    f.extent     = { width_, height_ };
    f.imageIndex = imageIndex;
    f.frameSlot  = currentFrame_;
    f.swapImage  = swapImages_[imageIndex];
    return f;
}

void SwapchainTarget::endFrame(const Frame& frame)
{
    VkDevice d = device_.device();
    vkEndCommandBuffer(frame.cmd);

    VkSemaphore waitSems[] = { imageAvailable_[currentFrame_] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
    VkSemaphore sigSems[]  = { renderFinished_[frame.imageIndex] };

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
    destroySceneBuffers();
    destroySwapchain();
    for (auto sem : renderFinished_) vkDestroySemaphore(d, sem, nullptr);
    renderFinished_.clear();

    createSwapchain();
    createSceneBuffers();

    VkSemaphoreCreateInfo sci {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    renderFinished_.resize(swapImages_.size(), VK_NULL_HANDLE);
    for (auto& sem : renderFinished_)
        vkCreateSemaphore(d, &sci, nullptr, &sem);
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

    // Single render target, exposed as sceneBuffers.colorA so the Renderer's
    // ping-pong execute path works uniformly with SwapchainTarget. Only
    // framebufferA is populated; effects are not supported on offscreen yet.
    renderImage_ = Image(device_.pool(), d, width_, height_, format_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    depthStencil_ = Image(device_.pool(), d, width_, height_,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    // Render pass matches the SwapchainTarget scene-clear RP shape (2 atts)
    // so the unified Renderer::execute path handles both targets uniformly.
    VkAttachmentDescription atts[2] {};
    atts[0].format = format_;
    atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    atts[1].format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo rpci {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    vkCreateRenderPass(d, &rpci, nullptr, &renderPass_);

    VkImageView views[] = { renderImage_.view(), depthStencil_.view() };
    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = renderPass_;
    fci.attachmentCount = 2;
    fci.pAttachments = views;
    fci.width = width_;
    fci.height = height_;
    fci.layers = 1;
    vkCreateFramebuffer(d, &fci, nullptr, &framebuffer_);

    // Expose as sceneBuffers so Renderer's unified execute path works.
    sceneBuffers_.framebufferA = framebuffer_;

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

    Frame f {};
    f.cmd = cmd_;
    f.extent = { width_, height_ };
    return f;
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
    VkDevice d = device_.device();
    vkDeviceWaitIdle(d);
    destroy();
    width_ = w;
    height_ = h;
    create();
}

} // namespace jvk
