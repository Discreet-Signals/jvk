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
        renderTimer_.stopTimer();
        renderer_.reset();
        target_.reset();
    }

    void setVulkanEnabled(bool enabled)
    {
        vulkanEnabled_ = enabled;
        metalView_.setVisible(enabled);
        if (enabled) {
            setCachedComponentImage(new NullCachedImage());
            renderTimer_.startTimerHz(60);
        } else {
            renderTimer_.stopTimer();
            setCachedComponentImage(nullptr);
        }
        juce::AudioProcessorEditor::repaint();
    }

    bool isVulkanEnabled() const { return vulkanEnabled_; }
    Device& getDevice() { return *device_; }

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
        return static_cast<float>(getDesktopScaleFactor());
#endif
    }

private:
    void init()
    {
        device_ = Device::acquire();
        device_->initCaches();

        addAndMakeVisible(metalView_);
        metalView_.setInterceptsMouseClicks(false, false);

        setCachedComponentImage(new NullCachedImage());
        setOpaque(true);

        addComponentListener(this);
        renderTimer_.callback = [this] { if (vulkanEnabled_ && renderer_) render(); };
    }

    // ComponentListener — fires even when subclass overrides resized()
    void componentMovedOrResized(juce::Component&, bool, bool wasResized) override
    {
        if (!wasResized) return;
        metalView_.setBounds(getLocalBounds());

        float scale = getDisplayScale();
        auto w = static_cast<uint32_t>(getWidth() * scale);
        auto h = static_cast<uint32_t>(getHeight() * scale);
        if (w == 0 || h == 0) return;

        if (!target_) {
            initVulkan(w, h);
            if (target_) renderTimer_.startTimerHz(60);
        } else {
            target_->resize(w, h);
        }
    }

    void render()
    {
        renderer_->reset();
        device_->caches().beginFrame(frameCounter_++);
        // Wipe the per-frame segment ring for the analytical-SDF path
        // renderer before JUCE paint fills it up again.
        if (pathPipeline_) pathPipeline_->beginFrame();

        // Phase 1: Record — JUCE paints, Graphics pushes commands into Renderer
        float scale = getDisplayScale();
        jvk::Graphics graphics(*renderer_, scale);
        juce::Graphics g(graphics);
        paintEntireComponent(g, true);

        // Phase 2: Execute — prepare pipelines, flush uploads, render pass, replay, present
        renderer_->execute();
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
        ci.hinstance = GetModuleHandle(nullptr);
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

        colorPipeline_ = std::make_unique<pipelines::ColorPipeline>(
            *device_, spv(vert_spv, vert_spvSize), spv(frag_spv, frag_spvSize));
        renderer_->registerPipeline(*colorPipeline_);

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

        // Pre-copy pipeline for clipped effects. Same vertex shader as blur;
        // the frag just samples the source texture verbatim. StencilMode::
        // Outside means the pass writes ONLY the pixels outside the active
        // clip — the immediately-following effect pass then fills the inside.
        // Together the two passes cover every destination pixel exactly once.
        copyEffect_ = std::make_unique<EffectPipeline>();
        copyEffect_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(copy_frag_spv, copy_frag_spvSize),
            EffectPipeline::StencilMode::Outside);
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

        // Shape-aware blur — variable per-pixel radius driven by the shape's
        // SDF. Used by Graphics::blurRect / blurRoundedRectangle / blurEllipse
        // / blurLine. Reuses the fullscreen-triangle vertex shader from the
        // standard blur.
        shapeBlur_ = std::make_unique<ShapeBlurPipeline>();
        shapeBlur_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(shape_blur_frag_spv, shape_blur_frag_spvSize));
        renderer_->setShapeBlur(shapeBlur_.get());

        // DrawShader dispatcher — runs user shaders (jvk::Shader) inside the
        // scene render pass. Each user shader lazily builds its own pipeline
        // variants (normal + clip) on first draw, against the scene RP
        // captured here.
        shaderPipeline_ = std::make_unique<ShaderPipeline>();
        shaderPipeline_->init(*device_, target_->sceneRenderPassClear());
        renderer_->setShaderPipeline(shaderPipeline_.get());

        // Analytical-SDF path renderer. Owns a per-frame storage-buffer
        // ring for segment uploads and a fragment shader that walks the
        // segments per pixel to compute the SDF + winding analytically.
        pathPipeline_ = std::make_unique<PathPipeline>();
        pathPipeline_->init(*device_,
            target_->sceneRenderPassClear(),
            spv(path_sdf_vert_spv, path_sdf_vert_spvSize),
            spv(path_sdf_frag_spv, path_sdf_frag_spvSize));
        renderer_->setPathPipeline(pathPipeline_.get());

        // Clip-stencil pipeline. Shares PathPipeline's segment SSBO for
        // arbitrary-path clips; rrect clips are purely analytical (no SSBO
        // read). Owns two VkPipelines — push (INCR_WRAP) and pop (DECR_WRAP)
        // — sharing one layout and one pair of shader modules.
        clipPipeline_ = std::make_unique<ClipPipeline>();
        clipPipeline_->init(*device_,
            target_->sceneRenderPassClear(),
            spv(clip_vert_spv, clip_vert_spvSize),
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

    std::unique_ptr<pipelines::ColorPipeline>         colorPipeline_;
    std::unique_ptr<pipelines::BlendPipeline>         blendPipeline_;
    std::unique_ptr<EffectPipeline>                   blurEffect_;
    std::unique_ptr<EffectPipeline>                   copyEffect_;
    std::unique_ptr<HSVPipeline>                      hsvPipeline_;
    std::unique_ptr<ShapeBlurPipeline>                shapeBlur_;
    std::unique_ptr<ShaderPipeline>                   shaderPipeline_;
    std::unique_ptr<PathPipeline>                     pathPipeline_;
    std::unique_ptr<ClipPipeline>                     clipPipeline_;

    RenderTimer renderTimer_;
    bool vulkanEnabled_ = true;
    uint64_t frameCounter_ = 0;
};

} // namespace jvk
