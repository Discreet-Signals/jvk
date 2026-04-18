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

        stencilWritePipeline_ = std::make_unique<pipelines::StencilWritePipeline>(
            *device_, spv(stencil_vert_spv, stencil_vert_spvSize), spv(stencil_frag_spv, stencil_frag_spvSize));
        renderer_->registerPipeline(*stencilWritePipeline_);

        stencilCoverPipeline_ = std::make_unique<pipelines::StencilCoverPipeline>(
            *device_, spv(stencil_vert_spv, stencil_vert_spvSize), spv(stencil_frag_spv, stencil_frag_spvSize));
        renderer_->registerPipeline(*stencilCoverPipeline_);

        blendPipeline_ = std::make_unique<pipelines::BlendPipeline>(
            *device_, spv(vert_spv, vert_spvSize), spv(frag_spv, frag_spvSize));
        renderer_->registerPipeline(*blendPipeline_);

        // Generic post-process pipeline. Runs any single-input fullscreen
        // fragment shader as an intra-frame effect. Configured here for
        // separable Gaussian blur.
        blurEffect_ = std::make_unique<EffectPipeline>();
        blurEffect_->init(*device_,
            target_->effectRenderPass(),
            spv(blur_vert_spv, blur_vert_spvSize),
            spv(blur_frag_spv, blur_frag_spvSize));
        renderer_->setPostProcess(blurEffect_.get());

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

    std::unique_ptr<pipelines::ColorPipeline>        colorPipeline_;
    std::unique_ptr<pipelines::StencilWritePipeline>  stencilWritePipeline_;
    std::unique_ptr<pipelines::StencilCoverPipeline>  stencilCoverPipeline_;
    std::unique_ptr<pipelines::BlendPipeline>         blendPipeline_;
    std::unique_ptr<EffectPipeline>                   blurEffect_;
    std::unique_ptr<ShapeBlurPipeline>                shapeBlur_;
    std::unique_ptr<ShaderPipeline>                   shaderPipeline_;

    RenderTimer renderTimer_;
    bool vulkanEnabled_ = true;
    uint64_t frameCounter_ = 0;
};

} // namespace jvk
