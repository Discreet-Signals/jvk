namespace jvk {

// =============================================================================
// RenderTarget — owns a per-target VkCommandPool so each worker thread has
// exclusive access to its own pool (Vulkan pools are externally synchronized).
// =============================================================================

RenderTarget::RenderTarget(Device& device)
    : device_(device)
{
    VkCommandPoolCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = device.graphicsFamily();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.device(), &ci, nullptr, &commandPool_);
}

RenderTarget::~RenderTarget()
{
    // Runs AFTER the derived destructor body, which vkDeviceWaitIdle's and
    // tears down sync objects. Destroying the pool auto-frees every command
    // buffer allocated from it — no explicit vkFreeCommandBuffers needed.
    if (commandPool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_.device(), commandPool_, nullptr);
}

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
    // Command buffers are freed automatically when ~RenderTarget destroys
    // our per-target VkCommandPool.

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

    // -----------------------------------------------------------------
    // Scene RP subpass dependencies — CANONICAL form.
    //
    // The scene RP writes BOTH color (fragment draws) and depth/stencil
    // (path-clip INCR/DECR). Every access type either attachment can see
    // is listed here so the dependency holds for every combination of
    // prior/next work: scene→effect→scene, scene→blit, scene→scene (ping-
    // pong with no effect in between). Missing any one stage or access
    // bit on a tile renderer (MoltenVK on Apple Silicon) causes
    // rectangular-tile corruption because the tile writeback from the
    // prior RP races with this RP's LOAD_OP_LOAD of the same image.
    // -----------------------------------------------------------------

    // deps[0] — EXTERNAL → this subpass. What must complete before I run.
    //
    //   Prior color writes (scene draws, effect passes)
    //   Prior depth/stencil writes (path clips in an earlier scene RP)
    //   Prior sampler reads (an effect that sampled my attachments)
    //   Prior transfer reads (a blit that read my attachments)
    //
    //   I access color + depth/stencil at COLOR_OUTPUT and EARLY/LATE
    //   fragment tests. LOAD_OP_LOAD reads each attachment at RP begin,
    //   so ATTACHMENT_READ is required on top of ATTACHMENT_WRITE.
    VkSubpassDependency deps[2] {};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                          | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                          | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_SHADER_READ_BIT
                          | VK_ACCESS_TRANSFER_READ_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                          | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                          | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // deps[1] — this subpass → EXTERNAL. What my work makes available.
    //
    //   My color writes → next scene LOAD, next effect sampling, final blit
    //   My depth/stencil writes → next scene LOAD (stencil test)
    VkSubpassDependency& out = deps[1];
    out.srcSubpass    = 0;
    out.dstSubpass    = VK_SUBPASS_EXTERNAL;
    out.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                      | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    out.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    out.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                      | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                      | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                      | VK_PIPELINE_STAGE_TRANSFER_BIT;
    out.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                      | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                      | VK_ACCESS_SHADER_READ_BIT
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

    // ---- Effect render pass (color + depth/stencil) ------------------
    //
    // The effect pass carries the shared depth/stencil image so effect
    // pipelines can stencil-test against the current clip state. Stencil
    // is READ (LOAD) but never written (writeMask=0 in pipelines).
    //
    // Color's loadOp is LOAD so pixels outside the clip retain whatever
    // Renderer::execute pre-copied into the destination. Without the
    // pre-copy, a clipped effect pass would leave outside-clip pixels
    // holding whatever was in the destination before (prior frame's
    // content), which is garbage.
    //
    // Stencil's storeOp is STORE — the effect doesn't modify stencil but
    // its contents must remain valid for the NEXT scene RP's LOAD.
    VkAttachmentDescription effectAtts[2] {};

    // Color
    effectAtts[0].format         = format_;
    effectAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    effectAtts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    effectAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    effectAtts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    effectAtts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    effectAtts[0].initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    effectAtts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Depth/stencil (shared image — same one the scene RPs use).
    effectAtts[1].format         = VK_FORMAT_D32_SFLOAT_S8_UINT;
    effectAtts[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    effectAtts[1].loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // depth unused
    effectAtts[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    effectAtts[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
    effectAtts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    effectAtts[1].initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    effectAtts[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference effectColorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference effectDSRef    = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription effectSub {};
    effectSub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    effectSub.colorAttachmentCount    = 1;
    effectSub.pColorAttachments       = &effectColorRef;
    effectSub.pDepthStencilAttachment = &effectDSRef;

    // -----------------------------------------------------------------
    // Effect RP subpass dependencies — CANONICAL form.
    //
    // The effect pass samples a color input (set=0) and writes the color
    // attachment. It stencil-TESTS against the shared DS attachment but
    // never writes to it. Still, the DS attachment's LOAD happens at
    // EARLY_FRAGMENT_TESTS and must wait for the prior scene RP's
    // stencil writes.
    // -----------------------------------------------------------------
    VkSubpassDependency effectDeps[2] {};
    effectDeps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    effectDeps[0].dstSubpass    = 0;
    effectDeps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    effectDeps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                | VK_ACCESS_SHADER_READ_BIT
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    effectDeps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    effectDeps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                | VK_ACCESS_SHADER_READ_BIT
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    effectDeps[1].srcSubpass    = 0;
    effectDeps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    effectDeps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    effectDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    effectDeps[1].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                                | VK_PIPELINE_STAGE_TRANSFER_BIT;
    effectDeps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                | VK_ACCESS_SHADER_READ_BIT
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                | VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo effectRPCI {};
    effectRPCI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    effectRPCI.attachmentCount = 2;
    effectRPCI.pAttachments    = effectAtts;
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

        // Effect framebuffers now also carry the shared depth/stencil
        // image — effect pipelines stencil-test against the active clip
        // state so effects inside a clipToPath/Rectangle are clipped to
        // that region (the shader writes only where stencil == depth;
        // outside fragments discard and preserve the destination's LOAD
        // contents, which Renderer::execute pre-copies from the source
        // before clipped effects).
        VkImageView effA[2] = { sb.colorA.view(), sb.depthStencil.view() };
        VkImageView effB[2] = { sb.colorB.view(), sb.depthStencil.view() };
        fci.renderPass = effectRP_;
        fci.attachmentCount = 2;
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
    ai.commandPool = commandPool_;
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
    ai.commandPool = commandPool_;
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
    // cmd_ is freed by ~RenderTarget destroying our per-target VkCommandPool.
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
