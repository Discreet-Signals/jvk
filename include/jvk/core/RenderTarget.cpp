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

std::vector<Target> RenderTarget::validTargets(std::vector<Target> requested) const
{
    if (requested.empty())
        return requested;

    if (!device_.supportsIndependentBlend()) {
        diag::log("render targets unavailable: the device lacks independentBlend");
        return {};
    }

    // The scene passes carry the targets at scale 1, target i at location
    // i + 1, and the built-in fragment shaders write locations 0..7: such a
    // target is among the first 7 (fewer where the device allows fewer colour
    // attachments: the spec minimum is 4). Scaled ones are no attachment of
    // theirs: up to kMaxTargets in all.
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(device_.physicalDevice(), &props);
    const size_t sceneLimit = std::min<size_t>(kMaxSceneTargets, props.limits.maxColorAttachments - 1);

    std::vector<Target> kept;
    for (auto& c : requested) {
        // Every built-in draw into a target blends (as into the main colour),
        // and so does paint clearing it (clearedByMain); the paint shaders
        // write floats. So a target's format must render, sample AND blend:
        // never an integer or block-compressed one, whatever its Blend says.
        const auto info = formatInfo(c.format);
        VkFormatProperties fp {};
        vkGetPhysicalDeviceFormatProperties(device_.physicalDevice(), toVkFormat(c.format), &fp);
        const auto needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                          | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        if (info.integer || info.compressed || (fp.optimalTilingFeatures & needed) != needed) {
            diag::log("render target '" + c.name.toString()
                      + "' dropped: its format cannot be rendered, sampled and blended");
            continue;
        }
        if (kept.size() == kMaxTargets) {
            diag::log("render target '" + c.name.toString() + "' dropped: more than "
                      + juce::String((int) kMaxTargets) + " targets");
            continue;
        }
        if (!c.isScaled() && kept.size() >= sceneLimit) {
            diag::log("render target '" + c.name.toString() + "' dropped: a target at scale 1 must be among the first "
                      + juce::String((int) sceneLimit));
            continue;
        }
        if (!(c.scale > 0.0f) || c.height < 0) {
            diag::log("render target '" + c.name.toString() + "' dropped: its size is not positive");
            continue;
        }
        if (c.isScaled() && c.clearedByMain) {
            diag::log("render target '" + c.name.toString() + "': a scaled target isn't cleared by main");
            c.clearedByMain = false;
        }
        kept.push_back(std::move(c));
    }
    return kept;
}

// =============================================================================
// SwapchainTarget
// =============================================================================

// Pins the calling thread to the surface window's DPI-awareness context for
// the duration of any driver call whose result depends on the window's client
// rect: capability queries (currentExtent), acquire and present. On Windows
// the ICD resolves GetClientRect on the CALLING thread, and Win32 coordinate
// APIs DPI-virtualize their results whenever the thread's context differs
// from the window's — so without pinning, the "authoritative" currentExtent
// changes value depending on who asks (message thread running in the host's
// context vs. our render worker in the process-default context), the two
// callers resize the swapchain to different sizes, every acquire reports
// SUBOPTIMAL/OUT_OF_DATE, and the swapchain never converges (black screen in
// hosts that run plugin UIs DPI-virtualized, e.g. Ableton's auto-scale).
// No-op off Windows, without a native window, or pre-Win10-1607.
namespace {
struct SurfaceDpiScope
{
#ifdef _WIN32
    core::windows::dpi::ScopedWindowContext scope;
    explicit SurfaceDpiScope(void* nativeWindow) : scope(nativeWindow) {}
#else
    explicit SurfaceDpiScope(void*) {}
#endif
};
} // namespace

SwapchainTarget::SwapchainTarget(Device& device, VkSurfaceKHR surface,
                                 uint32_t w, uint32_t h,
                                 void* nativeWindow,
                                 VkPresentModeKHR presentMode,
                                 std::vector<Target> targets)
    : RenderTarget(device), surface_(surface), nativeWindow_(nativeWindow),
      presentMode_(presentMode), width_(w), height_(h)
{
    targets_ = validTargets(std::move(targets));
    createSwapchain();
    createRenderPasses();
    createSceneBuffers();
    createSyncObjects();
}

SwapchainTarget::~SwapchainTarget()
{
    VkDevice d = device_.device();
    {
        const juce::ScopedLock queueSync(Renderer::queueLock());
        vkDeviceWaitIdle(d);
    }

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
    for (auto& [key, rp] : sceneRPVariants_)
        if (rp != VK_NULL_HANDLE) vkDestroyRenderPass(d, rp, nullptr);
    for (auto& [key, rp] : targetPasses_)
        if (rp != VK_NULL_HANDLE) vkDestroyRenderPass(d, rp, nullptr);
    if (effectRP_     != VK_NULL_HANDLE) vkDestroyRenderPass(d, effectRP_,     nullptr);
    vkDestroySurfaceKHR(device_.instance(), surface_, nullptr);
}

VkExtent2D SwapchainTarget::surfaceExtent() const
{
    SurfaceDpiScope dpiScope(nativeWindow_);
    VkSurfaceCapabilitiesKHR caps {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_.physicalDevice(), surface_, &caps);
    return caps.currentExtent;
}

void SwapchainTarget::createSwapchain()
{
    SurfaceDpiScope dpiScope(nativeWindow_);
    VkDevice d = device_.device();
    VkPhysicalDevice pd = device_.physicalDevice();

    // Zero-init + check: on query failure the fallback path below would
    // otherwise clamp against garbage min/maxImageExtent.
    VkSurfaceCapabilitiesKHR caps {};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface_, &caps) != VK_SUCCESS)
        return;

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, fmts.data());

    format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { colorSpace = f.colorSpace; break; }
    }

    // The surface's currentExtent is the AUTHORITATIVE physical pixel size of
    // the window/layer we render into. On Win32 it is exactly the child HWND's
    // client area in device pixels; on Metal/MoltenVK it is the CAMetalLayer's
    // drawableSize (layer bounds x contentsScale). It already reflects the host
    // content-scale (which JUCE applies as an AffineTransform on the editor, so
    // it is folded into the HWND size) AND the per-monitor DPI — neither of
    // which getPlatformScaleFactor() reports reliably (it returns 1.0 for an
    // embedded plugin window, and on macOS always). Trust currentExtent over
    // the caller's hint, which can be derived from a host-misreported scale
    // factor (e.g. Cubase on Windows reporting the wrong DPI). The 0xFFFFFFFF
    // sentinel means "surface has no preferred size" (some Wayland) — only then
    // fall back to clamping the requested size into the allowed range.
    VkExtent2D ext;
    if (caps.currentExtent.width  != 0xFFFFFFFFu
     && caps.currentExtent.width  != 0
     && caps.currentExtent.height != 0) {
        ext = caps.currentExtent;
    } else {
        // currentExtent is the 0xFFFFFFFF "no preferred size" sentinel (some
        // Wayland) or a transient 0 (surface/layer not sized yet). Fall back to
        // the exact pre-existing behaviour: clamp the caller's requested size
        // into the allowed range. This guarantees we are never worse than the
        // hint-based path in any case where currentExtent isn't usable.
        ext.width  = std::clamp(width_,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        ext.height = std::clamp(height_, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
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

    if (vkCreateSwapchainKHR(d, &ci, nullptr, &swapchain_) != VK_SUCCESS) {
        // Leave swapchain_ null — beginFrame guards and returns a null frame,
        // and the next resize retries. ci.oldSwapchain (if any) is retired by
        // the create call per spec regardless of success; destroy our handle.
        swapchain_ = VK_NULL_HANDLE;
        if (ci.oldSwapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(d, ci.oldSwapchain, nullptr);
        return;
    }

    if (ci.oldSwapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(d, ci.oldSwapchain, nullptr);

    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, nullptr);
    swapImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, swapImages_.data());
    // No per-image views: the scene renders offscreen and is BLITTED into the
    // raw swap images — views were created and destroyed on every resize
    // without ever being read.
}

VkRenderPass SwapchainTarget::sceneRenderPass(bool clear, TargetMask liveTargets)
{
    // Scaled targets aren't attachments of the scene pass: their liveness doesn't vary it.
    const TargetMask all = static_cast<TargetMask>((1u << targets_.size()) - 1u) & static_cast<TargetMask>(~scaledTargets());
    liveTargets &= all;
    if (liveTargets == all)
        return clear ? sceneRPClear_ : sceneRPLoad_;
    const uint32_t key = (clear ? 0x10000u : 0u) | liveTargets;
    for (auto& [k, rp] : sceneRPVariants_)
        if (k == key) return rp;
    VkRenderPass rp = createSceneRenderPass(clear, liveTargets);
    sceneRPVariants_.push_back({ key, rp });
    return rp;
}

VkRenderPass SwapchainTarget::targetPass(TargetSlots slots, TargetMask clears)
{
    const std::pair key { slots, clears };
    for (auto& [k, rp] : targetPasses_)
        if (k == key) return rp;

    // The written targets as attachments, in location order, each at the output location
    // its shader writes it at (the rest unused). Between passes a target sits in
    // SHADER_READ_ONLY_OPTIMAL like every target; a clear enters from UNDEFINED.
    std::vector<VkAttachmentDescription> atts;
    std::vector<VkAttachmentReference> refs(slotCount(slots), { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED });
    for (uint32_t l = 1; l < refs.size(); ++l) {
        const int i = slotTarget(slots, l);
        if (i < 0 || static_cast<size_t>(i) >= targets_.size()) continue;
        const bool clear = (clears & (1u << i)) != 0;
        VkAttachmentDescription a {};
        a.format         = toVkFormat(targets_[i].format);
        a.samples        = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp         = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout  = clear ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        a.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        refs[l] = { static_cast<uint32_t>(atts.size()), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        atts.push_back(a);
    }

    VkSubpassDescription sub {};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = static_cast<uint32_t>(refs.size());
    sub.pColorAttachments    = refs.data();

    // As the effect pass: after earlier writes and reads of these images (the scene pass,
    // other passes, the frame before), and before the shaders that sample them.
    VkSubpassDependency deps[2] {};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                          | VK_ACCESS_TRANSFER_WRITE_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_SHADER_READ_BIT;
    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci {};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = static_cast<uint32_t>(atts.size());
    rpci.pAttachments    = atts.data();
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;
    VkRenderPass rp = VK_NULL_HANDLE;
    vkCreateRenderPass(device_.device(), &rpci, nullptr, &rp);
    targetPasses_.push_back({ key, rp });
    return rp;
}

VkFramebuffer SwapchainTarget::targetFramebuffer(TargetSlots slots)
{
    for (auto& [k, fb] : targetFramebuffers_)
        if (k == slots) return fb;
    std::vector<VkImageView> views;
    size_t first = targets_.size();
    for (uint32_t l = 1; l < 8; ++l)
        if (const int i = slotTarget(slots, l); i >= 0 && static_cast<size_t>(i) < targetImages_.size()) {
            views.push_back(targetImages_[static_cast<size_t>(i)].view());
            if (first == targets_.size()) first = static_cast<size_t>(i);
        }
    VkRenderPass rp = targetPass(slots, 0);   // compatible with every clear variant
    if (views.empty() || rp == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    const auto e = targetExtent(first);
    VkFramebufferCreateInfo fci {};
    fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass      = rp;
    fci.attachmentCount = static_cast<uint32_t>(views.size());
    fci.pAttachments    = views.data();
    fci.width           = e.width;
    fci.height          = e.height;
    fci.layers          = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    vkCreateFramebuffer(device_.device(), &fci, nullptr, &fb);
    targetFramebuffers_.push_back({ slots, fb });
    return fb;
}

VkRenderPass SwapchainTarget::createSceneRenderPass(bool clear, TargetMask liveTargets) const
{
    VkDevice d = device_.device();

    // ---- Scene render pass: clear (frame start) or load (after an effect) ----
    // 0: Color (sampleable) — finalLayout=SHADER_READ_ONLY so next effect/blit samples it
    // 1: Depth/stencil — preserved across segments via SHADER_READ_ONLY-ish layout? No —
    //    depth is written, not sampled. Keep it in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    // Attachments: [0] main colour, then the render targets at scale 1, then
    // depth/stencil. Colour reference i + 1 is target i (its output location),
    // up to the last at scale 1 (sceneSlots); a scaled target's is unused (it
    // takes its own passes: targetPass).
    const uint32_t targetCount = static_cast<uint32_t>(sceneSlots(targets_) - 1);
    std::vector<VkAttachmentReference> colorRefs(1 + targetCount, { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED });
    colorRefs[0] = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    uint32_t attached = 0;
    for (uint32_t i = 0; i < targetCount; ++i)
        if (!targets_[i].isScaled())
            colorRefs[1 + i] = { 1 + attached++, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    const uint32_t depthIndex   = 1 + attached;
    std::vector<VkAttachmentDescription> sceneAtts(depthIndex + 1);

    // Color
    sceneAtts[0].format         = format_;
    sceneAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    sceneAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    sceneAtts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    sceneAtts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Render targets: like the main colour, sampleable between scene passes.
    // A DEAD one (not in liveTargets: nothing reads it again this frame) is
    // neither cleared/loaded nor stored; its contents are undefined from here.
    for (uint32_t i = 0; i < targetCount; ++i) {
        if (targets_[i].isScaled()) continue;
        const bool live = (liveTargets & (1u << i)) != 0;
        auto& a = sceneAtts[colorRefs[1 + i].attachment];
        a.format         = toVkFormat(targets_[i].format);
        a.samples        = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp         = !live ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                         : clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a.initialLayout  = live && !clear ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_UNDEFINED;
        a.storeOp        = live ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Depth/stencil
    sceneAtts[depthIndex].format         = VK_FORMAT_D32_SFLOAT_S8_UINT;
    sceneAtts[depthIndex].samples        = VK_SAMPLE_COUNT_1_BIT;
    sceneAtts[depthIndex].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[depthIndex].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    sceneAtts[depthIndex].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef = { depthIndex, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub {};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = static_cast<uint32_t>(colorRefs.size());
    sub.pColorAttachments       = colorRefs.data();
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
    rpci.attachmentCount = static_cast<uint32_t>(sceneAtts.size());
    rpci.pAttachments    = sceneAtts.data();
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;
    rpci.dependencyCount = 2;
    rpci.pDependencies   = deps;

    // Clear variant (frame start): the main colour, depth/stencil and every
    // live target clear (each target to its own clear value, see
    // Renderer::execute). Load variant (continuation after an effect): they
    // load what the last scene pass stored.
    sceneAtts[0].loadOp        = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    sceneAtts[0].initialLayout = clear ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sceneAtts[depthIndex].loadOp        = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    sceneAtts[depthIndex].stencilLoadOp = sceneAtts[depthIndex].loadOp;
    sceneAtts[depthIndex].initialLayout = clear ? VK_IMAGE_LAYOUT_UNDEFINED
                                                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkRenderPass rp = VK_NULL_HANDLE;
    vkCreateRenderPass(d, &rpci, nullptr, &rp);
    return rp;
}

void SwapchainTarget::createRenderPasses()
{
    VkDevice d = device_.device();

    // The scene passes with every target live; the variants with some dead
    // are made on first use (sceneRenderPass).
    const TargetMask all = static_cast<TargetMask>((1u << targets_.size()) - 1u);
    sceneRPClear_ = createSceneRenderPass(true,  all);
    sceneRPLoad_  = createSceneRenderPass(false, all);

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

    // Render targets: one image each, shared by both frame slots (see
    // RenderTarget::targets). Created before the framebuffers that use them.
    targetImages_.clear();
    for (size_t i = 0; i < targets_.size(); ++i) {
        // A scaled target is its own size, and is cleared by a transfer when it's read
        // before its first write in a frame (Renderer).
        const auto e = targetExtent(i);
        targetImages_.emplace_back(device_.pool(), d, e.width, e.height, toVkFormat(targets_[i].format),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
          | (targets_[i].isScaled() ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u));
    }
    static std::atomic<uint64_t> generations { 0 };
    targetsGeneration_ = ++generations;

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

        // Scene framebuffers (color + targets + depth). Clear and load
        // variants share the same layout, so one framebuffer targets both.
        std::vector<VkImageView> viewsA { sb.colorA.view() };
        std::vector<VkImageView> viewsB { sb.colorB.view() };
        for (size_t t = 0; t < targetImages_.size(); ++t)
            if (!targets_[t].isScaled()) { viewsA.push_back(targetImages_[t].view()); viewsB.push_back(targetImages_[t].view()); }
        viewsA.push_back(sb.depthStencil.view());
        viewsB.push_back(sb.depthStencil.view());
        VkFramebufferCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = sceneRPClear_; // compatible with sceneRPLoad_ too
        fci.attachmentCount = static_cast<uint32_t>(viewsA.size());
        fci.width = width_;
        fci.height = height_;
        fci.layers = 1;
        fci.pAttachments = viewsA.data();
        vkCreateFramebuffer(d, &fci, nullptr, &sb.framebufferA);
        fci.pAttachments = viewsB.data();
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

    // colorA always enters the frame through sceneRPClear_ (initialLayout
    // UNDEFINED), but colorB is only ever entered through effectFBtoB /
    // framebufferB, whose render passes declare initialLayout
    // SHADER_READ_ONLY — a declared layout must match the image's ACTUAL
    // layout (VUID-vkCmdBeginRenderPass-initialLayout-00900). Freshly
    // created images are UNDEFINED, so the first effect pass after startup
    // and after every resize violated it (the familiar "expects
    // SHADER_READ_ONLY_OPTIMAL — current layout is UNDEFINED" signature).
    // One-shot transition here makes the declared contract true.
    device_.submitImmediate([this](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barriers[MAX_FRAMES] {};
        for (int i = 0; i < MAX_FRAMES; i++) {
            auto& b = barriers[i];
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = sceneBuffers_[i].colorB.image();
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, MAX_FRAMES, barriers);
    });
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
    for (auto& [k, fb] : targetFramebuffers_)
        if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(d, fb, nullptr);
    targetFramebuffers_.clear();
    targetImages_.clear();   // after the framebuffers that referenced them
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
    swapImages_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

RenderTarget::Frame SwapchainTarget::beginFrame()
{
    SurfaceDpiScope dpiScope(nativeWindow_);
    VkDevice d = device_.device();

    // A lost device or a failed swapchain create leaves nothing to render
    // into — return a null frame instead of touching dead handles.
    if (device_.isLost() || swapchain_ == VK_NULL_HANDLE)
        return {};

    // Finite timeout instead of UINT64_MAX: if a submit failed (fence never
    // signals) or the driver hangs, an infinite wait wedges the worker
    // forever — the message thread then spins in waitForIdle and teardown
    // force-kills the thread mid-Vulkan-call. 2 s is far beyond any real
    // frame; on timeout treat the device as unusable and bail gracefully.
    VkResult waitResult = vkWaitForFences(d, 1, &inFlightFence_[currentFrame_],
                                          VK_TRUE, 2'000'000'000ull);
    if (waitResult != VK_SUCCESS) {
        if (waitResult == VK_ERROR_DEVICE_LOST || waitResult == VK_TIMEOUT)
            device_.markLost();
        return {};
    }

    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(d, swapchain_, UINT64_MAX,
        imageAvailable_[currentFrame_], VK_NULL_HANDLE, &imageIndex);

    // OUT_OF_DATE: swapchain no longer matches the surface (window resized /
    // DPI changed) and cannot present — recreate before using. No image was
    // acquired and imageAvailable_[currentFrame_] was NOT signaled, so it is
    // safe to drop this frame and retry the same slot next tick. resize()
    // re-queries currentExtent (pinned to the window's own DPI context), so
    // passing the old size is fine.
    //
    // SUBOPTIMAL is deliberately NOT handled here: an image WAS acquired and
    // the semaphore WILL be signaled, so the frame must proceed and present
    // normally — dropping it would leave a pending signal on
    // imageAvailable_[currentFrame_], and the next acquire against that
    // semaphore is a spec violation (VUID 01286) and a black screen on strict
    // drivers. endFrame() sees the present-side SUBOPTIMAL report and
    // recreates AFTER the image is consumed — that is the self-healing path
    // for live DPI changes a host applies without a logical-size change
    // (e.g. dragging a plugin window between monitors of different scale,
    // where componentMovedOrResized may not fire).
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize(width_, height_);
        return {}; // caller checks cmd == VK_NULL_HANDLE
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        // SURFACE_LOST (host destroyed the parent window under us — routine
        // for embedded plugin views), DEVICE_LOST, OOM. No image acquired,
        // no semaphore signaled: dropping the frame is safe. The old code
        // fell through here with an UNINITIALIZED imageIndex and indexed
        // swapImages_ with garbage.
        if (result == VK_ERROR_DEVICE_LOST)
            device_.markLost();
        return {};
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
    SurfaceDpiScope dpiScope(nativeWindow_);
    vkEndCommandBuffer(frame.cmd);

    // Index sync objects by the frame's OWN slot, not the member counter —
    // identical while exactly one frame records at a time, but the member
    // breaks silently the moment frames-in-flight changes.
    VkSemaphore waitSems[] = { imageAvailable_[frame.frameSlot] };
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

    // A failed submit means the fence NEVER signals — unchecked, the next
    // beginFrame waited on it forever, the message thread spun in
    // waitForIdle, and plugin close force-killed the worker mid-call.
    // Mark the device unusable; beginFrame's guard + finite wait handle the
    // rest gracefully.
    if (vkQueueSubmit(device_.graphicsQueue(), 1, &si,
                      inFlightFence_[frame.frameSlot]) != VK_SUCCESS) {
        device_.markLost();
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
        return;
    }

    VkSwapchainKHR swapchains[] = { swapchain_ };
    VkPresentInfoKHR pi {};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = sigSems;
    pi.swapchainCount = 1;
    pi.pSwapchains = swapchains;
    pi.pImageIndices = &frame.imageIndex;

    VkResult result = vkQueuePresentKHR(device_.presentQueue(), &pi);
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;

    // Post-present is the safe place to recreate: the acquired image and its
    // semaphores have been consumed, so no sync object holds a pending
    // operation. OUT_OF_DATE must recreate. For SUBOPTIMAL — the frame DID
    // present — rebuild only when the surface's size actually disagrees with
    // ours: drivers can report SUBOPTIMAL for reasons a recreate won't cure
    // (transform hints), and rebuilding every frame would turn a warning into
    // a vkDeviceWaitIdle-per-frame stall.
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize(width_, height_);
    } else if (result == VK_SUBOPTIMAL_KHR) {
        VkExtent2D cur = surfaceExtent();
        if (cur.width != 0xFFFFFFFFu && cur.width != 0 && cur.height != 0
            && (cur.width != width_ || cur.height != height_))
            resize(width_, height_);
    } else if (result == VK_ERROR_DEVICE_LOST) {
        device_.markLost();
    }
}

void SwapchainTarget::resize(uint32_t w, uint32_t h)
{
    VkDevice d = device_.device();
    // vkDeviceWaitIdle is spec-equivalent to vkQueueWaitIdle on EVERY queue
    // and carries the same external-sync requirement — unguarded, a resize
    // on this editor races a sibling editor's worker inside vkQueueSubmit.
    // Renderer::queueLock() is reentrant (juce::CriticalSection), so taking
    // it here is safe on the endFrame path that already holds it.
    {
        const juce::ScopedLock queueSync(Renderer::queueLock());
        vkDeviceWaitIdle(d);
    }
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

    // External dependencies, mirroring SwapchainTarget's scene RP: incoming
    // (prior sampling/transfer of the render image must complete before we
    // write it again) and outgoing (our color writes must be visible to the
    // sampling/copy that consumes the result). The old pass declared NONE —
    // the implicit BOTTOM_OF_PIPE dependency makes writes available but not
    // visible, so sampling or copying the image after endFrame was racy.
    VkSubpassDependency offDeps[2] {};
    offDeps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    offDeps[0].dstSubpass    = 0;
    offDeps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                             | VK_PIPELINE_STAGE_TRANSFER_BIT;
    offDeps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT
                             | VK_ACCESS_TRANSFER_READ_BIT;
    offDeps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                             | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                             | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    offDeps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                             | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    offDeps[1].srcSubpass    = 0;
    offDeps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    offDeps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    offDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    offDeps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                             | VK_PIPELINE_STAGE_TRANSFER_BIT;
    offDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                             | VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpci {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = 2;
    rpci.pDependencies = offDeps;
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
    {
        const juce::ScopedLock queueSync(Renderer::queueLock());
        vkDeviceWaitIdle(d);
    }

    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(d, fence_, nullptr);
    // cmd_ is freed by ~RenderTarget destroying our per-target VkCommandPool.
    if (framebuffer_ != VK_NULL_HANDLE) vkDestroyFramebuffer(d, framebuffer_, nullptr);
    if (renderPass_ != VK_NULL_HANDLE) vkDestroyRenderPass(d, renderPass_, nullptr);
    renderImage_ = {};
    depthStencil_ = {};
    sceneBuffers_ = {};   // framebufferA aliased the just-destroyed handle
    fence_ = VK_NULL_HANDLE;
    cmd_ = VK_NULL_HANDLE;
    framebuffer_ = VK_NULL_HANDLE;
    renderPass_ = VK_NULL_HANDLE;
}

RenderTarget::Frame OffscreenTarget::beginFrame()
{
    VkDevice d = device_.device();
    if (device_.isLost() || fence_ == VK_NULL_HANDLE)
        return {};
    if (vkWaitForFences(d, 1, &fence_, VK_TRUE, 2'000'000'000ull) != VK_SUCCESS) {
        device_.markLost();
        return {};
    }
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

    if (vkQueueSubmit(device_.graphicsQueue(), 1, &si, fence_) != VK_SUCCESS)
        device_.markLost();
}

void OffscreenTarget::resize(uint32_t w, uint32_t h)
{
    destroy();   // includes the queue-locked vkDeviceWaitIdle
    width_ = w;
    height_ = h;
    create();
}

} // namespace jvk
