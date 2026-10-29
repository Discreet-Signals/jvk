#pragma once

namespace jvk {

// =============================================================================
// AudioProcessorEditor — a juce::AudioProcessorEditor that paints on the GPU
//
// The JUCE host of the translation layer (TRANSLATION_LAYER.md, "Hosts"): its
// paint runs through jvk::Graphics into a core Renderer, which lowers each
// frame on its worker and submits it to a window surface. Which backend draws
// is the WindowHost's business (createWindowHost: Vulkan for now); the frame
// loop, paint, resize policy and the fall back to JUCE's renderer are here.
// =============================================================================

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
        teardownGpu();
    }

    // Toggle the GPU render path on/off at runtime. When disabled, every
    // per-editor GPU resource — renderer, surface, platform subview, and our
    // reference to the shared backend — is released so this editor keeps no
    // GPU footprint. Re-enabling rebuilds them. (The backend itself is shared
    // process-wide; dropping our reference only destroys it if nothing else
    // holds it.)
    void setVulkanEnabled(bool enabled)
    {
        // REENTRANCY GUARD: this is legitimately called from INSIDE paint()
        // (the benchmark switches renderer between timed phases from its
        // paint code). Applying the toggle immediately would tear down
        // mid-paintEntireComponent — destroying the paintContext_/renderer
        // the live paint stack is standing on, so the unwinding
        // ScopedSaveState destructors call restoreState() on a freed object.
        // Defer to the end of the render tick instead; render() applies it
        // after paintEntireComponent returns and drops the half-valid frame.
        if (inRenderTick_) {
            pendingGpuToggle_ = enabled;
            return;
        }
        if (enabled == gpuEnabled_) return;
        gpuEnabled_ = enabled;
        if (enabled) acquireGpu();
        else         teardownGpu();
        juce::AudioProcessorEditor::repaint();
    }

    bool isVulkanEnabled() const { return gpuEnabled_; }

    // ===== Render targets ==================================================
    //
    // Extra per-pixel targets the frame draws into next to the main colour
    // (see Target in core/Targets.h). None by default. Paint code selects
    // one with jvk::Graphics::setTarget and draws into it with ordinary
    // graphics calls; a jvk::Shader drawn later in paint reads them by
    // declaring samplers with their names, and turns them into colour.
    //
    // FIXED WHILE THE GPU RUNS: every pipeline is built for one frame layout,
    // so the set is declared before the GPU starts (in the constructor: it
    // starts at the first layout with a size) or while it is off
    // (setVulkanEnabled(false)). While it runs this changes nothing and
    // returns false. Targets the backend can't carry are dropped when it
    // starts: the frame's own set is Renderer::targets().
    bool setTargets(std::vector<Target> targets)
    {
        if (renderer_ != nullptr) {
            jassertfalse;   // the GPU is running: setVulkanEnabled(false) first
            return false;
        }
        targets_ = std::move(targets);
        return true;
    }
    // The targets declared with setTargets.
    const std::vector<Target>& getTargets() const { return targets_; }

    // False once the GPU has failed this editor, at any stage: no driver, no
    // GPU, no device, no surface or swapchain, the device lost. jvk has then
    // already fallen back to JUCE's renderer by itself (GPU off,
    // jvk::Graphics::create returns nullptr), and stays there; the reason says
    // why, in words for the user. A failure during construction happens before
    // onVulkanUnavailable can be set, so check this once constructed; a later
    // one calls onVulkanUnavailable, on the message thread.
    bool isVulkanAvailable() const { return gpuAvailable_; }
    const juce::String& getVulkanUnavailableReason() const { return unavailableReason_; }
    std::function<void()> onVulkanUnavailable;

    // The backend drawing this editor (null while the GPU is off).
    gpu::Backend* getBackend() const { return backend_.get(); }

    // Block until the renderer is quiescent: the worker idle and the GPU done
    // with everything submitted. Rarely needed: everything a frame names is
    // kept alive by the Context until the GPU is done with it.
    void waitForRenderIdle()
    {
        if (renderer_) renderer_->waitForIdle();
        if (backend_) backend_->waitIdle();
    }

protected:
    void paint(juce::Graphics& g) override
    {
        if (!gpuEnabled_ || !renderer_)
            g.fillAll(juce::Colours::black);
    }

    float getDisplayScale() const
    {
#if JUCE_MAC
        return 2.0f;
#else
        // Peer's platformScaleFactor is the per-monitor DPI scale — what we
        // need to size the swapchain in physical pixels so it matches the
        // native child window (JUCE positions it using this same value). NOT
        // getDesktopScaleFactor(), which is a user-settable multiplier
        // unrelated to system DPI.
        if (auto* peer = getPeer())
            return static_cast<float>(peer->getPlatformScaleFactor());
        return 1.0f;
#endif
    }

    // The scale that maps our LOGICAL coordinates (getWidth()/getHeight()) to
    // the surface's PHYSICAL pixels — i.e. the factor jvk::Graphics must use
    // so paint output fills the surface exactly. Derived from the live
    // surface size, NOT from getPlatformScaleFactor(): on Windows VST3 hosts
    // apply DPI as an AffineTransform on the editor (the peer stays at 1.0),
    // and on macOS the peer's factor is always 1.0 while the backing scale
    // varies. The surface's true physical width over getWidth() captures peer
    // DPI AND the host content-scale transform on every platform.
    float getPaintScale() const
    {
        if (auto* s = surface(); s != nullptr && getWidth() > 0 && getHeight() > 0)
            return static_cast<float>(s->extent().width) / static_cast<float>(getWidth());
        return getDisplayScale();
    }

private:
    gpu::Surface* surface() const { return host_ != nullptr ? host_->surface() : nullptr; }

    void init()
    {
        setOpaque(true);
        addComponentListener(this);
        renderTimer_.callback = [this] { if (gpuEnabled_ && renderer_) render(); };
        catchUp_.callback = [this] { if (missedTick_ && gpuEnabled_ && renderer_) render(); };

        // GPU rendering defaults to on — wire up the resources so construction
        // matches the historical behaviour (or falls back, with no device).
        // Subclasses that want to start disabled can call
        // setVulkanEnabled(false) immediately.
        acquireGpu();
        if (gpuAvailable_) diag::log("editor: GPU available");
    }

    // The one way out of the GPU, whatever failed: GPU off, JUCE's renderer
    // paints from here on, the reason kept and logged, the app told. Sticky.
    void fallBack(const juce::String& reason)
    {
        gpuAvailable_ = false;
        unavailableReason_ = reason;
        diag::log("editor: " + reason + ": JUCE renderer");
        setVulkanEnabled(false);
        if (onVulkanUnavailable) onVulkanUnavailable();
    }

    void acquireGpu()
    {
        if (host_ == nullptr)
            host_ = createWindowHost();
        if (backend_ == nullptr)
            backend_ = host_ != nullptr ? host_->backend() : nullptr;
        if (backend_ == nullptr) {
            fallBack(host_ != nullptr ? host_->reason() : juce::String("no GPU backend"));
            return;
        }
        setCachedComponentImage(new NullCachedImage());

        // If we already have a size, stand up the surface now. Otherwise
        // componentMovedOrResized will do it on the first layout pass. On
        // Windows attaching additionally defers until the peer exists (the
        // child window must be created in the host tree's DPI-awareness
        // context), and parentHierarchyChanged / visibilityChanged re-enter
        // once it does.
        const float scale = getDisplayScale();
        const auto w = static_cast<uint32_t>(getWidth()  * scale);
        const auto h = static_cast<uint32_t>(getHeight() * scale);
        if (w > 0 && h > 0 && !renderer_) {
            initGpu(w, h);
            if (renderer_) renderTimer_.startTimerHz(60);
        } else if (renderer_) {
            renderTimer_.startTimerHz(60);
        }
    }

    void teardownGpu()
    {
        // Must never run while a jvk-driven paint is on the stack — it destroys
        // paintContext_/renderer_ the paint is executing through.
        // setVulkanEnabled defers to end-of-tick; direct callers (destructor,
        // mode-switch helpers) run outside the tick.
        jassert(!inRenderTick_);
        renderTimer_.stopTimer();
        if (renderer_) renderer_->waitForIdle();

        paintContext_.reset();   // references renderer_ — must go first
        renderer_.reset();       // joins the worker; retires its textures through the Context
        catchUp_.cancelPendingUpdate();   // the worker is gone: nothing posts after this
        missedTick_ = false;
        if (host_) host_->detach();   // the surface and the native view
        context_.reset();
        setCachedComponentImage(nullptr);
        // Drop our references to the shared backend: it goes only if no other
        // editor or offscreen renderer still holds it.
        backend_.reset();
        host_.reset();
    }

    // ComponentListener — fires even when a subclass overrides resized().
    // React to BOTH move and resize: a host DPI change arrives as a
    // setContentScaleFactor → setTransform on the editor, which can change our
    // on-screen (physical) size without changing our logical size — that
    // surfaces as a move, not a resize. updateSurface compares against the
    // real surface size, so a spurious call is a cheap no-op.
    void componentMovedOrResized(juce::Component&, bool wasMoved, bool wasResized) override
    {
        if (!wasMoved && !wasResized) return;
        if (host_) host_->setBounds(getLocalBounds());
        updateSurface();
    }

    // Peer attaches after construction in many flows. Standalone's
    // setContentOwned() attaches the editor to the window BEFORE the window is
    // shown, so parentHierarchyChanged fires while the peer is still null. The
    // peer appears when the window is shown, which fires visibilityChanged.
    // Hooking both covers every peer-attachment path. updateSurface is
    // idempotent: no-op when the surface already exists, retry otherwise.
    //
    // PROTECTED, and subclasses that override MUST chain: on Windows the
    // surface is only created once the peer exists, and these hooks are the
    // creation trigger.
protected:
    void parentHierarchyChanged() override
    {
        if (renderer_ == nullptr)
            updateSurface();
    }

    void visibilityChanged() override
    {
        if (renderer_ == nullptr && isShowing())
            updateSurface();
    }

private:
    void updateSurface()
    {
        if (!gpuEnabled_) return;

        if (!renderer_) {
            // No surface yet — create one. The size is only a hint: the
            // surface takes the platform's real physical size when it has
            // one. If we have no size at all, bail — the hooks above re-enter
            // once the peer/bounds arrive.
            const float scale = getDisplayScale();
            const auto w = static_cast<uint32_t>(getWidth()  * scale);
            const auto h = static_cast<uint32_t>(getHeight() * scale);
            if (w == 0 || h == 0) return;
            initGpu(w, h);
            if (renderer_) renderTimer_.startTimerHz(60);
            return;
        }

        // A native view made in a context the window no longer has (Windows:
        // a host rebuilt the window in the other DPI awareness): rebuild the
        // whole stack in the new one.
        if (host_->needsReattach(*this))
        {
            teardownGpu();
            acquireGpu();
            return;
        }

        // Resize to the window's TRUE physical size, asked of the platform.
        // Only rebuild when it actually changed: a resize is expensive and
        // must not run every layout pass.
        auto* s = surface();
        if (s == nullptr) return;
        gpu::Extent want = s->preferredExtent();
        if (want.width == 0xFFFFFFFFu) {
            // No preferred size (some Wayland) — a logical*scale estimate.
            const float scale = getDisplayScale();
            want.width  = static_cast<uint32_t>(getWidth()  * scale);
            want.height = static_cast<uint32_t>(getHeight() * scale);
        }
        if (want.width == 0 || want.height == 0) return;   // minimised / hidden
        if (want == s->extent()) return;                    // already correct

        // The worker must be idle: it could be mid-frame on the swapchain the
        // resize replaces. Brief wait (one frame) on this infrequent path.
        renderer_->waitForIdle();
        s->resize(want);
    }

    void render()
    {
        // The frame is lowered and submitted on the Renderer's worker. If it's
        // still busy (a driver busy-looping in present can soak a full vsync
        // interval), skip this tick — the message thread never blocks. A
        // skipped tick isn't lost: the worker's onIdle renders it as soon as
        // it frees up (catchUp_).
        if (renderer_->isBusy()) { missedTick_ = true; return; }
        missedTick_ = false;

        // A lost device never renders again: fall back (the next editor gets a
        // fresh backend). Without a swapchain, a minimised window (no extent)
        // just waits; otherwise try to make one now, and fall back once that
        // has failed for a second at a real size. Until there is one, record
        // NOTHING.
        if (backend_->isLost()) { fallBack("the graphics device was lost"); return; }
        auto* s = surface();
        if (s == nullptr) return;
        if (!s->canBegin()) {
            const auto extent = s->preferredExtent();
            if (extent.width == 0 || extent.height == 0) return;
            s->revive();   // the worker is idle: the swapchain is ours to rebuild
            if (!s->canBegin()) {
                if (++swapchainFailures_ >= 60) fallBack("the swapchain could not be created");
                return;
            }
        }
        swapchainFailures_ = 0;

        if (!renderer_->beginRecord()) return;

        // Record — JUCE paints, Graphics records into the Renderer. Paint scale
        // comes from the live surface size (getPaintScale), so content always
        // fills the surface however the host reports DPI.
        //
        // ONE persistent Graphics, reset per tick: a stack-local per frame
        // threw away every scratch vector's capacity 60×/s, and reset
        // getFrameId() to zero each frame — breaking JUCE's per-frame caches,
        // which key on a monotonic id.
        const float scale = getPaintScale();
        if (!paintContext_)
            paintContext_ = std::make_unique<jvk::Graphics>(*renderer_, scale);
        else
            paintContext_->beginFrame(scale);
        {
            juce::Graphics g(*paintContext_);
            inRenderTick_ = true;
            paintEntireComponent(g, true);
            inRenderTick_ = false;
        }

        // A setVulkanEnabled() call landed DURING paint (benchmark phase
        // switches do this): apply it now that the paint stack has fully
        // unwound, and drop this frame's recording.
        if (pendingGpuToggle_.has_value()) {
            const bool enable = *pendingGpuToggle_;
            pendingGpuToggle_.reset();
            if (renderer_) renderer_->abandonRecord();
            setVulkanEnabled(enable);
            return;
        }

        renderer_->submit();

        // Bring-up milestones for the GPU log: the first frame exercises the
        // whole path; the 120th says it held.
        if (++framesSubmitted_ == 1)        diag::log("first frame submitted");
        else if (framesSubmitted_ == 120)   diag::log("120 frames submitted: GPU path stable");
    }

    void initGpu(uint32_t w, uint32_t h)
    {
        juce::String failure;
        if (!host_->attach(*this, { w, h }, failure)) {
            if (failure.isNotEmpty()) fallBack(failure);
            return;   // empty: not yet (no peer); the hooks re-enter
        }
        host_->setBounds(getLocalBounds());
        context_ = Context::forBackend(backend_);
        renderer_ = std::make_unique<Renderer>(context_, *host_->surface(), targets_);
        renderer_->onIdle = [this] { catchUp_.triggerAsyncUpdate(); };
        const auto e = host_->surface()->extent();
        diag::log("surface " + juce::String((int) e.width) + "x" + juce::String((int) e.height) + " ready");
    }

    struct RenderTimer : public juce::Timer {
        std::function<void()> callback;
        void timerCallback() override { if (callback) callback(); }
    };
    // The worker's onIdle posts here; on the message thread, a tick skipped while
    // the worker was busy (missedTick_) renders now. One skipped tick, one frame.
    struct RenderCatchUp : public juce::AsyncUpdater {
        std::function<void()> callback;
        void handleAsyncUpdate() override { if (callback) callback(); }
    };

    struct NullCachedImage : public juce::CachedComponentImage {
        void paint(juce::Graphics&) override {}
        bool invalidateAll() override { return false; }
        bool invalidate(const juce::Rectangle<int>&) override { return false; }
        void releaseResources() override {}
    };

    RenderCatchUp catchUp_;   // before renderer_: outlives the worker that posts to it
    bool          missedTick_ = false;   // message thread only

    std::unique_ptr<WindowHost>   host_;
    std::shared_ptr<gpu::Backend> backend_;
    std::shared_ptr<Context>      context_;
    std::unique_ptr<Renderer>     renderer_;
    // Persistent record context, reset via beginFrame() each tick (scratch
    // capacity + monotonic getFrameId survive across frames). References
    // renderer_, so teardown resets it first.
    std::unique_ptr<jvk::Graphics> paintContext_;
    // True while paintEntireComponent is on the stack in render(). A
    // setVulkanEnabled() arriving then is DEFERRED (pendingGpuToggle_) and
    // applied after the paint unwinds.
    bool                inRenderTick_ = false;
    std::optional<bool> pendingGpuToggle_;
    uint64_t            framesSubmitted_ = 0;

    RenderTimer renderTimer_;

    // Render targets (setTargets): fixed while the GPU runs.
    std::vector<Target> targets_;
    bool gpuEnabled_ = true;
    bool gpuAvailable_ = true;   // until fallBack
    juce::String unavailableReason_;
    int swapchainFailures_ = 0;   // render ticks in a row with no swapchain at a real size
};

} // namespace jvk
