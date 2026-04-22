#pragma once

namespace jvk {

class AudioProcessorEditor : public juce::AudioProcessorEditor,
                              private juce::ComponentListener
{
public:
    using juce::Component::addAndMakeVisible;
    using juce::Component::setCachedComponentImage;
    using juce::Component::setOpaque;
    using juce::Component::getLocalBounds;
    using juce::Component::getWidth;
    using juce::Component::getHeight;
    using juce::Component::paintEntireComponent;
    using juce::Component::getPeer;
    using juce::Component::repaint;
    using juce::Component::setSize;
    using juce::AudioProcessorEditor::setResizable;
    using juce::AudioProcessorEditor::setResizeLimits;
    using juce::AudioProcessorEditor::getConstrainer;

    explicit AudioProcessorEditor(juce::AudioProcessor& p)
        : juce::AudioProcessorEditor(p)
    {
        init();
    }

    explicit AudioProcessorEditor(juce::AudioProcessor* p)
        : juce::AudioProcessorEditor(p)
    {
        init();
    }

    ~AudioProcessorEditor() override
    {
        removeComponentListener(this);
        teardownVulkan();
    }

    // Active Vulkan Renderer, or nullptr in Simple/software mode. Exposed so
    // host-side tooling (debug overlays, scheduler mode switchers) can poke
    // live Renderer state without needing to own the device/target plumbing.
    Renderer* renderer() { return renderer_.get(); }

    // Toggle the Vulkan render path on/off at runtime. When disabled, every
    // per-editor Vulkan resource — pipelines, renderer, swapchain, surface,
    // platform subview, and our reference to the shared Device — is released
    // so this editor keeps no Vulkan footprint. Re-enabling rebuilds them.
    // (The Device itself is refcounted process-wide via weak_ptr; dropping
    // our reference only destroys the instance if nothing else holds it.)
    void setVulkanEnabled(bool enabled)
    {
        if (enabled == vulkanEnabled_) return;
        vulkanEnabled_ = enabled;
        if (enabled) acquireVulkan();
        else         teardownVulkan();
        juce::AudioProcessorEditor::repaint();
    }

    bool isVulkanEnabled() const { return vulkanEnabled_; }
    // Reports whether a usable Vulkan device was acquired at construction.
    // `false` means the runtime has no MoltenVK / ICD or the GPU can't
    // expose a compatible device — callers should fall back to JUCE's
    // native renderer and not attempt to enable Vulkan. Result is cached so
    // it stays accurate even after setVulkanEnabled(false) has released our
    // Device reference.
    bool isVulkanAvailable() const { return vulkanAvailable_; }
    Device& getDevice() { return *device_; }

    // Block until the renderer is fully quiescent: worker thread idle, GPU
    // idle, and every FrameRetained pin released. After this returns, any
    // Vulkan-touching resource captured into the recent command stream
    // (jvk::Shader and other FrameRetained subclasses) can be destroyed
    // immediately on this thread — Shader's destructor would otherwise
    // spin until those pins drop the natural way (next time their slot is
    // used), which never happens once the timer is stopped.
    //
    // The Shader destructor's own waitUntilUnretained call is enough for
    // the steady-state case where frames keep flowing; this method exists
    // for teardown / mode-switch paths that stop submitting new frames
    // before destroying GPU-backed objects.
    //
    // No-op if Vulkan isn't acquired. The timer is unaffected; rendering
    // resumes automatically on the next tick unless the caller also stops
    // the timer (e.g. via setVulkanEnabled(false)).
    void waitForRenderIdle()
    {
        if (renderer_) renderer_->waitForIdle();
        if (device_ && device_->device() != VK_NULL_HANDLE)
            vkDeviceWaitIdle(device_->device());
        if (renderer_) renderer_->flushRetains();
    }

protected:
    void paint(juce::Graphics& g) override
    {
        if (!vulkanEnabled_ || !renderer_)
            g.fillAll(juce::Colours::black);
    }

    float getDisplayScale() const
    {
#if JUCE_MAC
        return 2.0f;
#else
        // Peer's platformScaleFactor is the per-monitor DPI scale — what we
        // need to size the swapchain in physical pixels so it matches the
        // HWND (JUCE's HWNDComponent positions the child HWND using this
        // same value). NOT getDesktopScaleFactor(), which returns
        // Desktop::getGlobalScaleFactor() — a user-settable multiplier that
        // defaults to 1.0 and is unrelated to system DPI. Using the wrong
        // one produces a logical-sized swapchain that the driver stretches
        // into the physical HWND: blurry output and a slow stretch-blit
        // present path that starves the message thread.
        if (auto* peer = getPeer())
            return static_cast<float>(peer->getPlatformScaleFactor());
        return 1.0f;
#endif
    }

private:
    void init()
    {
        addAndMakeVisible(metalView_);
        metalView_.setInterceptsMouseClicks(false, false);

        setOpaque(true);
        addComponentListener(this);
        renderTimer_.callback = [this] { if (vulkanEnabled_ && renderer_) render(); };

        // vulkanEnabled_ defaults to true — wire up the Vulkan resources so
        // construction matches the historical behavior. Subclasses that want
        // to start disabled can call setVulkanEnabled(false) immediately.
        acquireVulkan();
        vulkanAvailable_ = (device_ && device_->device() != VK_NULL_HANDLE);
    }

    void acquireVulkan()
    {
        if (!device_) {
            device_ = Device::acquire();
            if (device_) device_->initCaches();
        }
        metalView_.setVisible(true);
        setCachedComponentImage(new NullCachedImage());

        // If we already have a size, stand up the swapchain + pipelines
        // immediately. Otherwise componentMovedOrResized will do it on the
        // first layout pass. Scale may be 1.0 at this point if the peer
        // isn't attached yet — that's fine; parentHierarchyChanged /
        // visibilityChanged will call target_->resize() with the real
        // per-monitor scale once the peer is live.
        const float scale = getDisplayScale();
        const auto w = static_cast<uint32_t>(getWidth()  * scale);
        const auto h = static_cast<uint32_t>(getHeight() * scale);
        if (w > 0 && h > 0 && !target_) {
            initVulkan(w, h);
            if (target_) renderTimer_.startTimerHz(60);
        } else if (target_) {
            renderTimer_.startTimerHz(60);
        }
    }

    void teardownVulkan()
    {
        renderTimer_.stopTimer();
        // Ensure the worker finished any in-flight execute before we
        // vkDeviceWaitIdle / free resources. Renderer's dtor also stops
        // the worker, but doing it explicitly here guarantees the queue
        // is quiescent before any pipeline/target reset runs below.
        if (renderer_) renderer_->waitForIdle();
        if (device_ && device_->device() != VK_NULL_HANDLE)
            vkDeviceWaitIdle(device_->device());

        // Pipelines hold VkPipeline handles against device_ and render
        // passes from target_ — must go before both. Reverse registration
        // order, mirrored from registerPipelines.
        clipPipeline_.reset();
        blurPipeline_.reset();
        fillPipeline_.reset();
        pathPipeline_.reset();
        shaderPipeline_.reset();
        hsvPipeline_.reset();
        copyEffect_.reset();
        blurEffect_.reset();
        blendPipeline_.reset();

        renderer_.reset();
        target_.reset();

    #if JUCE_MAC
        metalView_.setView(nullptr);
        nsViewGen_.reset();
    #elif JUCE_WINDOWS
        metalView_.setHWND(nullptr);
        hwndGen_.reset();
    #endif
        metalView_.setVisible(false);

        setCachedComponentImage(nullptr);
        // Drop our per-editor reference to the shared Device singleton. The
        // weak_ptr in Device::acquire expires only if no other editor or
        // ShaderImage is still holding it — so other instances keep running.
        device_.reset();
    }

    // ComponentListener — fires even when subclass overrides resized()
    void componentMovedOrResized(juce::Component&, bool, bool wasResized) override
    {
        if (!wasResized) return;
        metalView_.setBounds(getLocalBounds());
        updateVulkanTarget();
    }

    // Peer attaches after construction in many flows. Standalone's
    // setContentOwned() attaches the editor to the window BEFORE the window
    // is shown, so parentHierarchyChanged fires while the peer is still
    // null. The peer appears when the window is shown, which fires
    // visibilityChanged rather than parentHierarchyChanged. Hooking both
    // events covers every peer-attachment path without requiring a
    // ComponentMovementWatcher. updateVulkanTarget is idempotent: no-op
    // when target_ already exists, retry otherwise.
    void parentHierarchyChanged() override
    {
        if (target_ == nullptr)
            updateVulkanTarget();
    }

    void visibilityChanged() override
    {
        if (target_ == nullptr && isShowing())
            updateVulkanTarget();
    }

    void updateVulkanTarget()
    {
        if (!vulkanEnabled_) return;

        // Scale may be 1.0 if the peer isn't attached yet (e.g. first
        // componentMovedOrResized fired by setSize before the window is
        // shown). We still init at whatever size we can — otherwise the
        // first render never happens and the window stays blank until the
        // user manually resizes. When the peer arrives later,
        // parentHierarchyChanged / visibilityChanged re-enter here and
        // target_->resize() picks up the corrected physical size.
        float scale = getDisplayScale();
        auto w = static_cast<uint32_t>(getWidth() * scale);
        auto h = static_cast<uint32_t>(getHeight() * scale);
        if (w == 0 || h == 0) return;

        if (!target_) {
            initVulkan(w, h);
            if (target_) renderTimer_.startTimerHz(60);
        } else {
            // resize destroys and recreates the swapchain. The worker must
            // be idle — otherwise it could be mid-execute holding references
            // to the swapchain images we're about to free. Brief wait
            // (bounded by one execute duration) on this infrequent path.
            renderer_->waitForIdle();
            target_->resize(w, h);
        }
    }

    void render()
    {
        // Renderer::execute() runs on a dedicated worker thread. If it's
        // still in flight (NVIDIA busy-loop on present can soak a full VSync
        // interval, longer under DWM drag stalls), skip this tick — we
        // never block the message thread. Resulting render rate adapts to
        // what the worker can sustain; input responsiveness stays at 60 Hz.
        if (renderer_->isBusy()) return;

        renderer_->reset();
        device_->caches().beginFrame(frameCounter_++);
        // Wipe the per-frame segment ring for the analytical-SDF path
        // renderer before JUCE paint fills it up again.
        if (pathPipeline_) pathPipeline_->beginFrame();
        // Same lifecycle for the geometry-abstracted blur / fill / clip
        // pipelines' per-frame GeometryPrimitive SSBOs. Cleared on paint,
        // flushed to GPU inside Renderer::execute after the frame fence wait.
        if (blurPipeline_) blurPipeline_->beginFrame();
        if (fillPipeline_) fillPipeline_->beginFrame();
        if (clipPipeline_) clipPipeline_->beginFrame();

        // Phase 1: Record — JUCE paints, Graphics pushes commands into Renderer
        float scale = getDisplayScale();
        jvk::Graphics graphics(*renderer_, scale);
        juce::Graphics g(graphics);
        paintEntireComponent(g, true);

        // Phase 2: Submit to worker — non-blocking. The worker's execute()
        // handles prepare pipelines, flush uploads, render pass, replay,
        // submit, present.
        renderer_->submit();
    }

    void initVulkan(uint32_t w, uint32_t h)
    {
#if JUCE_MAC
        nsViewGen_ = std::make_unique<jvk::core::macos::NSViewGenerator>();
        void* nsView = nsViewGen_->create();
        if (!nsView) return;
        metalView_.setView(nsView);

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkMacOSSurfaceCreateInfoMVK ci {};
        ci.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
        ci.pView = nsView;
        if (vkCreateMacOSSurfaceMVK(device_->instance(), &ci, nullptr, &surface) != VK_SUCCESS)
            return;
#elif JUCE_WINDOWS
        hwndGen_ = std::make_unique<jvk::core::windows::HWNDGenerator>();
        HWND childHwnd = hwndGen_->create();
        if (!childHwnd) return;
        metalView_.setHWND((void*)childHwnd);

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkWin32SurfaceCreateInfoKHR ci {};
        ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        ci.hwnd = childHwnd;
        // Must match the HINSTANCE that registered the window class and
        // created the HWND — i.e. the plugin DLL, not the host process
        // (GetModuleHandle(nullptr) would return the host .exe).
        ci.hinstance = jvk::core::windows::HWNDGenerator::moduleHInstance();
        if (vkCreateWin32SurfaceKHR(device_->instance(), &ci, nullptr, &surface) != VK_SUCCESS)
            return;
#else
        VkSurfaceKHR surface = VK_NULL_HANDLE;
#endif
        if (surface == VK_NULL_HANDLE) return;

        target_ = std::make_unique<SwapchainTarget>(*device_, surface, w, h);
        renderer_ = std::make_unique<Renderer>(*device_, *target_);
        registerPipelines();
    }

    static std::span<const uint32_t> spv(const char* data, int byteSize)
    {
        return { reinterpret_cast<const uint32_t*>(data), static_cast<size_t>(byteSize / 4) };
    }

    void registerPipelines()
    {
        using namespace shaders::ui2d;

        blendPipeline_ = std::make_unique<pipelines::BlendPipeline>(
            *device_, spv(vert_spv, vert_spvSize), spv(frag_spv, frag_spvSize));
        renderer_->registerPipeline(*blendPipeline_);

        // Generic post-process pipeline. Runs any single-input fullscreen
        // fragment shader as an intra-frame effect. Configured here for
        // separable Gaussian blur (StencilMode::Inside — writes only
        // inside the active clip).
        blurEffect_ = std::make_unique<EffectPipeline>();
        blurEffect_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(blur_frag_spv, blur_frag_spvSize),
            EffectPipeline::StencilMode::Inside);
        renderer_->setPostProcess(blurEffect_.get());

        // Ping-pong seed-copy pipeline. Full-viewport copy from source to
        // destination, no stencil test — runs BEFORE every effect dispatch
        // that swaps the ping-pong so the destination half starts with the
        // current scene content. Lets every effect pipeline use tight
        // dispatch regions (bbox / stencil-clipped) without leaving stale
        // ping-pong data where the effect doesn't touch. Shares blur.vert
        // (fullscreen triangle) + copy.frag (passthrough sample).
        copyEffect_ = std::make_unique<EffectPipeline>();
        copyEffect_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(copy_frag_spv, copy_frag_spvSize),
            EffectPipeline::StencilMode::Always);
        renderer_->setCopyEffect(copyEffect_.get());

        // HSV transform pipeline — one shader, covers every HSV-space
        // operation (saturate, shift hue, lift value, tint-like effects)
        // via scale-then-delta per channel. Reuses blur's fullscreen-
        // triangle vertex shader.
        hsvPipeline_ = std::make_unique<HSVPipeline>();
        hsvPipeline_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(hsv_frag_spv,  hsv_frag_spvSize));
        renderer_->setHSVPipeline(hsvPipeline_.get());

        // DrawShader dispatcher — runs user shaders (jvk::Shader) inside the
        // scene render pass. Each user shader lazily builds its own pipeline
        // variants (normal + clip) on first draw, against the scene RP
        // captured here.
        shaderPipeline_ = std::make_unique<ShaderPipeline>();
        shaderPipeline_->init(*device_, target_->sceneRenderPassClear());
        renderer_->setShaderPipeline(shaderPipeline_.get());

        // Path-segment storage utility. No longer dispatches FillPath itself
        // (that moved into FillPipeline); PathPipeline is now just the per-
        // frame segment SSBO + its descriptor set layout, shared by the fill,
        // blur, and clip pipelines.
        pathPipeline_ = std::make_unique<PathPipeline>();
        pathPipeline_->init(*device_,
            target_->sceneRenderPassClear(),
            spv(path_sdf_vert_spv, path_sdf_vert_spvSize),
            spv(path_sdf_frag_spv, path_sdf_frag_spvSize));
        renderer_->setPathPipeline(pathPipeline_.get());

        // Geometry-abstracted fill pipeline. Absorbs ColorPipeline + path-
        // fill dispatch. Owns a per-frame GeometryPrimitive SSBO + MSDF
        // glyph atlas. Must be built AFTER pathPipeline_ so it can share
        // that pipeline's segment-SSBO descriptor-set layout for tag=5.
        fillPipeline_ = std::make_unique<FillPipeline>();
        fillPipeline_->init(*device_,
            target_->sceneRenderPassClear(),
            pathPipeline_->ssboSetLayout(),
            spv(geometry_vert_spv, geometry_vert_spvSize),
            spv(fill_frag_spv, fill_frag_spvSize));
        renderer_->setFillPipeline(fillPipeline_.get());

        // Geometry-abstracted variable-radius Gaussian blur. Handles both
        // shape blurs (rect / rrect / ellipse / capsule — tags 0..3) and
        // path blurs (tag 5). Owns its own per-frame GeometryPrimitive
        // SSBO; reuses PathPipeline's segment-SSBO layout for the path
        // geometry branch. Must be initialised AFTER pathPipeline_.
        blurPipeline_ = std::make_unique<BlurPipeline>();
        blurPipeline_->init(*device_,
            target_->effectRenderPass(),
            pathPipeline_->ssboSetLayout(),
            // Tight-bbox dispatch via the shared geometry.vert. The
            // ping-pong seed-copy prologue in Renderer::effectPassAndSwap
            // already seeds the destination half with source content, so
            // the blur only has to write its shape band — outside-bbox
            // pixels retain the copied source.
            spv(geometry_vert_spv, geometry_vert_spvSize),
            spv(geom_blur_frag_spv, geom_blur_frag_spvSize));
        renderer_->setBlurPipeline(blurPipeline_.get());

        // Geometry-abstracted clip-stencil pipeline. Owns its own per-frame
        // GeometryPrimitive SSBO; shares PathPipeline's segment-SSBO layout
        // for the tag=5 path branch. Two VkPipeline variants share one
        // layout + one pair of shader modules (geometry.vert + clip.frag).
        clipPipeline_ = std::make_unique<ClipPipeline>();
        clipPipeline_->init(*device_,
            target_->sceneRenderPassClear(),
            pathPipeline_->ssboSetLayout(),
            spv(geometry_vert_spv, geometry_vert_spvSize),
            spv(clip_frag_spv, clip_frag_spvSize));
        renderer_->setClipPipeline(clipPipeline_.get());
    }

    struct RenderTimer : public juce::Timer {
        std::function<void()> callback;
        void timerCallback() override { if (callback) callback(); }
    };

    struct NullCachedImage : public juce::CachedComponentImage {
        void paint(juce::Graphics&) override {}
        bool invalidateAll() override { return false; }
        bool invalidate(const juce::Rectangle<int>&) override { return false; }
        void releaseResources() override {}
    };

    std::shared_ptr<Device> device_;
    std::unique_ptr<SwapchainTarget> target_;
    std::unique_ptr<Renderer> renderer_;

#if JUCE_MAC
    std::unique_ptr<jvk::core::macos::NSViewGenerator> nsViewGen_;
    juce::NSViewComponent metalView_;
#elif JUCE_WINDOWS
    std::unique_ptr<jvk::core::windows::HWNDGenerator> hwndGen_;
    juce::HWNDComponent metalView_;
#else
    juce::Component metalView_;
#endif

    std::unique_ptr<pipelines::BlendPipeline>         blendPipeline_;
    std::unique_ptr<EffectPipeline>                   blurEffect_;
    std::unique_ptr<EffectPipeline>                   copyEffect_;
    std::unique_ptr<HSVPipeline>                      hsvPipeline_;
    std::unique_ptr<ShaderPipeline>                   shaderPipeline_;
    std::unique_ptr<PathPipeline>                     pathPipeline_;
    std::unique_ptr<FillPipeline>                     fillPipeline_;
    std::unique_ptr<BlurPipeline>                     blurPipeline_;
    std::unique_ptr<ClipPipeline>                     clipPipeline_;

    RenderTimer renderTimer_;
    bool vulkanEnabled_ = true;
    bool vulkanAvailable_ = false;
    uint64_t frameCounter_ = 0;
};

} // namespace jvk
