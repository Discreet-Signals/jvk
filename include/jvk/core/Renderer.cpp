/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Renderer.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk {

namespace
{
    std::atomic<int> g_liveRenderers { 0 };

    // The main colour's format: the two halves the scene ping-pongs between.
    constexpr PixelFormat kSceneFormat = PixelFormat::BGRA8;

    // A jvk::Shader draw's constants: jvk's 7 floats (and one reserved), then
    // the draw's own (Shader::setDrawConstants). 128 bytes, the push-constant
    // size every Vulkan device has.
    struct ShaderConstants
    {
        float resolutionX = 0, resolutionY = 0;   // the region's size (pixels)
        float time = 0;
        float viewportW = 0, viewportH = 0;
        float regionX = 0, regionY = 0;
        float reserved = 0;
        float draw[kShaderDrawConstants] {};
    };
    static_assert(sizeof(ShaderConstants) == 128);
}

int Renderer::liveCount() { return g_liveRenderers.load(std::memory_order_relaxed); }

// =============================================================================
// Construction / destruction — starts and stops the worker thread
// =============================================================================

Renderer::Renderer(std::shared_ptr<Context> context, gpu::Surface& surface, std::vector<Target> targets)
    : context_(std::move(context)), surface_(surface),
      worker_(std::make_unique<Worker>(*this))
{
    gradientAtlas_.init(context_->backend());
    glyphAtlas_.init(context_->backend());
    validateTargets(std::move(targets));
    const auto& caps = context_->caps();
    builder_.setAlignments(caps.constantsAlignment, caps.storageAlignment);
    context_->addRenderer(pending_);
    g_liveRenderers.fetch_add(1, std::memory_order_relaxed);
    worker_->startThread();
}

// The worker is joined first, so nothing it touches goes before it. Then
// every texture this Renderer made is retired through the Context: frames
// already submitted may still read them on the GPU, which the backend waits
// out before freeing them.
Renderer::~Renderer()
{
    worker_.reset();
    Context::endFrame(pending_);
    shaders_.clear();
    releaseFrameTextures();
    context_->retire(gradientAtlas_.release(), serial_);
    glyphAtlas_.clear([this](gpu::TextureId t) { context_->retire(t, serial_); });
    context_->removeRenderer(pending_);
    g_liveRenderers.fetch_sub(1, std::memory_order_relaxed);
}

// =============================================================================
// The frame
// =============================================================================

bool Renderer::beginRecord()
{
    jassert(!isBusy());
    const auto e = surface_.extent();
    if (e.width == 0 || e.height == 0) return false;
    if (!ensureFrameTextures(e)) return false;
    extent_ = e;
    serial_ = context_->beginRecord(pending_);
    recorder_.reset(scaledTargets());
    shaderCaptures_.clear();
    shaderData_.clear();
    return true;
}

void Renderer::abandonRecord()
{
    Context::endFrame(pending_);
}

void Renderer::submit()
{
    // Release here publishes every CPU-side write made during record to the
    // worker's acquire load at the top of runFrame().
    workerBusy_.store(true, std::memory_order_release);
    if (worker_) worker_->notify();
}

void Renderer::waitForIdle()
{
    // Used only for resize and teardown — both rare. Yield rather than burn a
    // CPU core; the wait is bounded by one frame.
    while (workerBusy_.load(std::memory_order_acquire))
        juce::Thread::yield();
}

void Renderer::runFrame()
{
    // A frame that can't begin (minimised, mid-resize, swapchain out of date)
    // is dropped: what it recorded is gone, but nothing it needs is lost — the
    // texture writes made while recording land before the next frame
    // submitted anywhere, and the atlases' dirty pages stay dirty until a
    // frame writes them.
    // A begun frame is always submitted: its output must be presented. One
    // whose output changed size between record and acquire (a live resize)
    // still draws the recorded frame, stretched over the output by the final
    // copy for that one frame, rather than presenting an unwritten image.
    gpu::FrameInfo info;
    if (surface_.beginFrame(info))
    {
        if (info.output)
            lower(info);
        else
            builder_.reset(++frameNumber_);
        surface_.submit(builder_.packet());
    }
    Context::endFrame(pending_);
}

// =============================================================================
// Recording helpers
// =============================================================================

float Renderer::registerGradient(const juce::ColourGradient& g)
{
    return gradientAtlas_.getRow(g, GradientAtlas::hash(g));
}

gpu::TextureId Renderer::texture(const juce::Image& img)
{
    if (!img.isValid()) return {};
    // A jvk image keeps its own texture (ImageType): no cache entry.
    if (auto* jvkImage = dynamic_cast<PixelData*>(img.getPixelData().get()))
        if (auto t = jvkImage->prepareForDraw(*this)) return t;
    return context_->images().get(img, serial_);
}

gpu::TextureId Renderer::texture(const ImageSource& src)
{
    return context_->images().get(src, serial_);
}

Shader& Renderer::shader(std::span<const uint32_t> spirv, std::string_view msl)
{
    auto& s = shaders_[spirv.data()];
    if (s == nullptr) {
        s = std::make_unique<Shader>();
        s->load(spirv, msl);
    }
    return *s;
}

// =============================================================================
// Targets and frame textures
// =============================================================================

void Renderer::validateTargets(std::vector<Target> requested)
{
    droppedTargets_.clear();
    targets_.clear();
    if (requested.empty())
        return;

    const auto& caps = context_->caps();
    if (!caps.independentBlend) {
        diag::log("render targets unavailable: the device lacks independent blending");
        for (auto& c : requested) droppedTargets_.push_back(c.name);
        return;
    }

    // The scene passes carry the targets at scale 1, target i at location
    // i + 1, and the built-in fragment shaders write locations 0..7: such a
    // target is among the first 7 (fewer where the device allows fewer colour
    // attachments: the Vulkan minimum is 4). Scaled ones are no attachment of
    // theirs: up to kMaxTargets in all.
    const size_t sceneLimit = std::min<size_t>(kMaxSceneTargets, caps.maxColorAttachments - 1);

    for (auto& c : requested) {
        // Every built-in draw into a target blends (as into the main colour),
        // and so does paint clearing it (clearedByMain); the paint shaders
        // write floats. So a target's format must render, sample AND blend:
        // never an integer or block-compressed one, whatever its Blend says.
        const auto info = formatInfo(c.format);
        if (info.integer || info.compressed || !context_->backend().canRenderTo(c.format)) {
            diag::log("render target '" + c.name.toString()
                      + "' dropped: its format cannot be rendered, sampled and blended");
            droppedTargets_.push_back(c.name);
            continue;
        }
        if (targets_.size() == kMaxTargets) {
            diag::log("render target '" + c.name.toString() + "' dropped: more than "
                      + juce::String((int) kMaxTargets) + " targets");
            jassertfalse;   // the application declares more than jvk carries: a programming error
            droppedTargets_.push_back(c.name);
            continue;
        }
        if (!c.isScaled() && targets_.size() >= sceneLimit) {
            diag::log("render target '" + c.name.toString() + "' dropped: a target at scale 1 must be among the first "
                      + juce::String((int) sceneLimit));
            droppedTargets_.push_back(c.name);
            continue;
        }
        if (!(c.scale > 0.0f) || c.height < 0) {
            diag::log("render target '" + c.name.toString() + "' dropped: its size is not positive");
            droppedTargets_.push_back(c.name);
            continue;
        }
        if (c.isScaled() && c.clearedByMain) {
            diag::log("render target '" + c.name.toString() + "': a scaled target isn't cleared by main");
            c.clearedByMain = false;
        }
        targets_.push_back(std::move(c));
    }
}

TargetMask Renderer::scaledTargets() const
{
    TargetMask mask = 0;
    for (size_t i = 0; i < targets_.size(); ++i)
        if (targets_[i].isScaled()) mask |= static_cast<TargetMask>(1u << i);
    return mask;
}

float Renderer::targetScale(size_t i) const
{
    if (i >= targets_.size()) return 1.0f;
    const auto& t = targets_[i];
    return t.height > 0 ? static_cast<float>(t.height) / static_cast<float>(std::max(1u, textureExtent_.height)) : t.scale;
}

gpu::Extent Renderer::targetExtent(size_t i) const
{
    const auto e = textureExtent_;
    if (i >= targets_.size() || !targets_[i].isScaled()) return e;
    const float s = targetScale(i);
    const auto h = targets_[i].height > 0 ? static_cast<uint32_t>(targets_[i].height)
                                          : static_cast<uint32_t>(std::ceil(static_cast<float>(e.height) * s));
    return { std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(e.width) * s))), std::max(1u, h) };
}

bool Renderer::ensureFrameTextures(gpu::Extent extent)
{
    if (extent == textureExtent_ && halves_[0] && halves_[1] && stencil_)
        return true;

    releaseFrameTextures();
    textureExtent_ = extent;
    auto& backend = context_->backend();

    gpu::TextureDesc d;
    d.format = kSceneFormat;
    d.usage  = gpu::TextureUsage::Sampled | gpu::TextureUsage::RenderTarget;
    d.width  = extent.width;
    d.height = extent.height;
    d.label  = "jvk scene A";
    halves_[0] = backend.createTexture(d);
    d.label  = "jvk scene B";
    halves_[1] = backend.createTexture(d);

    gpu::TextureDesc ds;
    ds.kind   = gpu::TextureKind::DepthStencil;
    ds.usage  = gpu::TextureUsage::RenderTarget;
    ds.width  = extent.width;
    ds.height = extent.height;
    ds.label  = "jvk clip stencil";
    stencil_ = backend.createTexture(ds);

    targetTextures_.assign(targets_.size(), {});
    targetScratches_.assign(targets_.size(), {});
    for (size_t i = 0; i < targets_.size(); ++i)
    {
        const auto te = targetExtent(i);
        gpu::TextureDesc td;
        td.format = targets_[i].format;
        td.usage  = gpu::TextureUsage::Sampled | gpu::TextureUsage::RenderTarget;
        td.width  = te.width;
        td.height = te.height;
        td.label  = "jvk render target";
        targetTextures_[i] = backend.createTexture(td);
    }

    if (!halves_[0] || !halves_[1] || !stencil_)
    {
        releaseFrameTextures();
        return false;
    }
    return true;
}

void Renderer::releaseFrameTextures()
{
    for (auto& t : halves_) { context_->retire(t, serial_); t = {}; }
    context_->retire(stencil_, serial_);
    stencil_ = {};
    for (auto t : targetTextures_)  context_->retire(t, serial_);
    for (auto t : targetScratches_) context_->retire(t, serial_);
    targetTextures_.clear();
    targetScratches_.clear();
    textureExtent_ = {};
}

// A target's scratch: a texture of its size and format, versioning it when a
// pass reads and writes it. Made on first use (worker).
gpu::TextureId Renderer::targetScratch(size_t i)
{
    if (i >= targetScratches_.size()) return {};
    if (!targetScratches_[i])
    {
        const auto te = targetExtent(i);
        gpu::TextureDesc td;
        td.format = targets_[i].format;
        td.usage  = gpu::TextureUsage::Sampled | gpu::TextureUsage::RenderTarget;
        td.width  = te.width;
        td.height = te.height;
        td.label  = "jvk render target scratch";
        targetScratches_[i] = context_->backend().createTexture(td);
    }
    return targetScratches_[i];
}

// =============================================================================
// Pipelines
// =============================================================================

gpu::AttachmentFormats Renderer::sceneFormats() const
{
    gpu::AttachmentFormats f;
    const size_t slots = sceneSlots(targets_);
    f.colorCount = static_cast<uint8_t>(slots);
    f.colorFormats[0] = kSceneFormat;
    for (size_t i = 0; i + 1 < slots; ++i)
    {
        if (targets_[i].isScaled()) f.unusedMask |= static_cast<uint8_t>(1u << (1 + i));
        else                        f.colorFormats[1 + i] = targets_[i].format;
    }
    f.depthStencil = true;
    return f;
}

const Renderer::PaintVariant* Renderer::paintPipeline(Paint kind, uint8_t attachment, bool clip, TargetMask live)
{
    const bool paint = kind == Paint::Color || kind == Paint::Path;
    const TargetMask clears = paintClears(targets_, attachment, paint, live);
    for (const auto& v : paintVariants_)
        if (v.kind == kind && v.attachment == attachment && v.clip == clip && v.clears == clears)
            return v.id ? &v : nullptr;
    if (attachment >= sceneSlots(targets_)) return nullptr;   // a target the pass doesn't have

    const auto& programs = context_->programs();
    gpu::PipelineDesc d;
    d.vertexLayout = ops::uiVertexLayout();
    d.attachments = sceneFormats();
    gpu::BlendState blend;
    switch (kind)
    {
        case Paint::Color:
            d.program = programs.ui2d;
            blend = gpu::BlendState::premultiplied();
            break;
        case Paint::Blend:
            // Multiply (colour grading): result = src * dst.
            d.program = programs.ui2d;
            blend = { true, gpu::BlendFactor::DstColor, gpu::BlendFactor::Zero, gpu::BlendOp::Add,
                            gpu::BlendFactor::Zero,     gpu::BlendFactor::One,  gpu::BlendOp::Add, gpu::WriteAll };
            break;
        case Paint::Noise:
            d.program = programs.noise;
            blend = gpu::BlendState::over();
            break;
        case Paint::Path:
            d.program = programs.path;
            blend = gpu::BlendState::premultiplied();   // path_sdf.frag writes premultiplied, as ui2d.frag
            break;
    }
    const bool paintsMain = drawTargets(targets_, attachment, blend, clears, d);
    if (clip)
        d.stencil = { true, gpu::CompareOp::Equal, gpu::StencilOp::Keep, gpu::StencilOp::Keep, 0xFF, 0xFF };

    paintVariants_.push_back({ kind, attachment, clip, clears, context_->pipeline(d), paintsMain });
    return paintVariants_.back().id ? &paintVariants_.back() : nullptr;
}

// =============================================================================
// Lowering — the render graph
//
// The recorded frame becomes passes of draws, in recording order (a painter's
// order, never reordered). Every command names what it reads and writes
// (DrawShaderParams, the recording target of an effect), and the lowering
// places it: inside the scene pass when everything it writes is attached
// there and nothing it reads is (the cheapest: no pass boundary, no load or
// store), else as a pass of its own attached to exactly what it writes, at
// that point in paint order. So the graph is one linear walk: pass boundaries
// only where a read needs one, every dead write and every load or store of a
// dead target dropped (Recorder::liveTargets), scaled targets cleared lazily,
// each pass's render area the region it touches (a tile GPU skips the tiles
// outside). A resource both read and written in one pass is VERSIONED, never
// aliased: the main colour through its two halves (effectPassAndSwap), a
// target through its scratch (drawn into the scratch while the pass samples
// the target, then copied back over the region it touched).
//
// Ping-pong model: halves_[0] is half A, halves_[1] half B; `cur` is the half
// that currently holds the scene. A post-process effect ends the scene pass,
// draws from the current half into the other, and swaps; the next scene pass
// loads the half the effect wrote. The stencil is one texture attached to
// every pass of either half, LOADED, so clips persist across effects.
// Separable effects (Gaussian blur) do two passes and so land back on the
// half they started on; non-separable ones over a PARTIAL region add a
// region-bounded identity copy-back so pixels outside the region keep the
// original scene; full-window ones swap once (the destination is fully
// covered). The frame ends with a copy of the current half into its output.
// =============================================================================

void Renderer::lower(const gpu::FrameInfo& info)
{
    using namespace ops;
    using gpu::LoadOp;
    using gpu::StoreOp;

    auto& ctx = *context_;
    const auto& programs = ctx.programs();
    const auto& commands = recorder_.commands();
    const auto& arena = recorder_.arena();
    const gpu::Extent fe = extent_;
    const float fw = static_cast<float>(fe.width), fh = static_cast<float>(fe.height);
    // One clock reading per frame: every shader this frame sees the same
    // `time` (two DrawShader ops never drift apart).
    const float frameTime = ctx.time();

    builder_.reset(++frameNumber_);
    gradientAtlas_.stageUploads();
    glyphAtlas_.stageDirtyPages();

    // The frame's storage: the path segments every path fill, clip and path
    // blur indexes into, and the targets' clear colours the paint reads.
    const auto segmentBytes = recorder_.segments();
    const gpu::Range segments = builder_.pushStorage(segmentBytes.data(), static_cast<uint32_t>(segmentBytes.size()));
    FrameData frameDataValues;
    for (size_t i = 0; i < targets_.size() && i < kMaxSceneTargets; ++i)
    {
        frameDataValues.targetClear[i][0] = targets_[i].clear.r;
        frameDataValues.targetClear[i][1] = targets_[i].clear.g;
        frameDataValues.targetClear[i][2] = targets_[i].clear.b;
        frameDataValues.targetClear[i][3] = targets_[i].clear.a;
    }
    const gpu::Range frameData = builder_.pushStorage(&frameDataValues, sizeof(frameDataValues));

    emitter_.beginFrame(builder_, fe);

    const gpu::TextureBinding defaultBinding { ctx.defaultTexture(), kLinearClamp };
    auto bind = [](gpu::TextureId t) { return gpu::TextureBinding { t, kLinearClamp }; };

    int cur = 0;
    const size_t slots = sceneSlots(targets_);
    const TargetMask scaled = scaledTargets();
    TargetMask scaledWritten = 0;
    TargetMask live = TargetMask(~0u);   // liveTargets of the command being lowered

    // ---- Passes -------------------------------------------------------------

    // A scene pass, beginning at command `firstCommand`: clearing at the frame
    // start, loading after an effect. A target with no reader from here on is
    // neither loaded nor stored: it can only die at a reader, and a reader
    // always ends the scene pass, so whatever is live where a pass begins
    // stays live to its end. The scene pass is open or not: a pass of its own
    // ends it (endScene), and it reopens only for the next command that draws
    // into it, so a run of passes pays no empty scene segments in between.
    bool sceneOpen = false;
    auto beginScene = [&](bool withClears, size_t firstCommand) {
        auto& p = builder_.beginPass(withClears ? "scene (clear)" : "scene");
        const TargetMask liveHere = recorder_.liveTargets(firstCommand);
        p.colorCount = static_cast<uint8_t>(slots);
        p.color[0] = { halves_[cur], withClears ? LoadOp::Clear : LoadOp::Load, StoreOp::Store, {} };
        for (size_t i = 0; i + 1 < slots; ++i)
        {
            const auto& t = targets_[i];
            if (t.isScaled()) { p.color[1 + i] = {}; continue; }   // not a scene attachment
            const bool isLive = (liveHere & (1u << i)) != 0;
            p.color[1 + i] = { targetTextures_[i],
                               !isLive ? LoadOp::DontCare : withClears ? LoadOp::Clear : LoadOp::Load,
                               isLive ? StoreOp::Store : StoreOp::DontCare, t.clear };
        }
        p.depthStencil = { stencil_, withClears ? LoadOp::Clear : LoadOp::Load, StoreOp::Store, 0 };
        p.renderArea = { 0, 0, static_cast<int32_t>(fe.width), static_cast<int32_t>(fe.height) };
        sceneOpen = true;
        emitter_.invalidate();
    };
    auto endScene = [&] {
        if (sceneOpen) builder_.endPass();
        sceneOpen = false;
    };

    enum class StencilTest { None, Inside, Outside };

    // A pass of one fullscreen-triangle draw of `program`, writing `dst` (of
    // `format`) within `sc`, which is also its render area: pixels outside it
    // are untouched. With a stencil test (the main colour's passes) only the
    // pixels inside the clip (Inside: stencil == ref) or outside it (Outside:
    // !=, the pre-copy) are written; the rest keep what `dst` held (Load).
    auto fullscreenPass = [&](const char* label, gpu::ProgramId program, gpu::TextureId dst, PixelFormat format,
                              StencilTest test, std::initializer_list<gpu::TextureBinding> textures,
                              std::initializer_list<gpu::Range> storage, gpu::Range constants,
                              gpu::IRect sc, uint8_t stencilRef, LoadOp load = LoadOp::Load)
    {
        if (sc.isEmpty() || !dst) return;
        const bool ds = test != StencilTest::None;
        gpu::PipelineDesc d;
        d.program = program;
        d.attachments.colorCount = 1;
        d.attachments.colorFormats[0] = format;
        d.attachments.depthStencil = ds;
        d.blend[0] = gpu::BlendState::replace();
        if (ds)
            d.stencil = { true, test == StencilTest::Inside ? gpu::CompareOp::Equal : gpu::CompareOp::NotEqual,
                          gpu::StencilOp::Keep, gpu::StencilOp::Keep, 0xFF, 0x00 };
        const auto pipeline = ctx.pipeline(d);
        if (!pipeline) return;

        auto& p = builder_.beginPass(label);
        p.colorCount = 1;
        p.color[0] = { dst, load, StoreOp::Store, {} };
        if (ds) p.depthStencil = { stencil_, LoadOp::Load, StoreOp::Store, 0 };
        p.renderArea = sc;

        gpu::Draw dr;
        dr.pipeline = pipeline;
        dr.scissor = sc;
        dr.stencilRef = stencilRef;
        for (const auto& t : textures) dr.textures[dr.textureCount++] = t;
        for (const auto& s : storage)  dr.storage[dr.storageCount++] = s;
        dr.constants = constants;
        dr.vertexCount = 3;
        builder_.draw(dr);
        builder_.endPass();
    };

    auto effectConstants = [&](gpu::Extent e, float dirX, float dirY, float radius) {
        EffectConstants c;
        c.viewportW = static_cast<float>(e.width);
        c.viewportH = static_cast<float>(e.height);
        c.dirX = dirX; c.dirY = dirY; c.radius = radius;
        return builder_.pushConstants(c);
    };

    // Runs a single post-process pass (`applyPass(src, dst)`) and swaps the
    // active half. Only between scene passes (endScene first).
    auto effectPassAndSwap = [&](auto applyPass) {
        const int dst = cur ^ 1;
        applyPass(halves_[cur], halves_[dst]);
        cur = dst;
    };

    // ---- Effect ROI (region-of-interest back-propagation) -------------------
    //
    // Each pass's scissor is the NEXT pass's input requirement — the final
    // output region expanded by each intermediate pass's kernel apron —
    // clamped to the frame. Every chain is arranged to end on the half it
    // STARTED on (even number of swaps; single-pass effects append a
    // region-bounded identity copy-back), so pixels outside the region /
    // outside the clip simply keep the original scene in place: a small
    // blurred knob costs region-sized passes, not full-screen ones.

    // Final output region: recorded effect region ∩ recorded clip. Empty
    // region param = whole clip (inverted blurs).
    auto roiFinal = [](const juce::Rectangle<float>& region, const juce::Rectangle<int>& clip) {
        auto c = clip.toFloat();
        return region.isEmpty() ? c : region.getIntersection(c);
    };

    // Rect (+ per-axis apron) → frame-clamped scissor.
    auto roiScissor = [&](juce::Rectangle<float> r, float padX, float padY) -> gpu::IRect {
        r = r.expanded(padX, padY);
        const int x0 = std::max(0, (int) std::floor(r.getX()));
        const int y0 = std::max(0, (int) std::floor(r.getY()));
        const int x1 = std::min((int) fe.width,  (int) std::ceil(r.getRight()));
        const int y1 = std::min((int) fe.height, (int) std::ceil(r.getBottom()));
        if (x1 <= x0 || y1 <= y0) return {};
        return { x0, y0, x1 - x0, y1 - y0 };
    };
    auto coversFrame = [&](const gpu::IRect& sc) {
        return sc.x == 0 && sc.y == 0
            && sc.width == (int32_t) fe.width && sc.height == (int32_t) fe.height;
    };

    // Separable chains still need outside-clip pixels of the INTERMEDIATE half
    // seeded with source (pass 2's taps cross the clip edge and read pass 1's
    // stencil-discarded pixels otherwise) — but only within the intermediate
    // pass's scissor, not full-screen.
    auto preCopyIfClipped = [&](uint8_t stencilDepth, const gpu::IRect& sc) {
        if (stencilDepth == 0 && !emitter_.stencilTested()) return;
        fullscreenPass("pre-copy", programs.copy, halves_[cur ^ 1], kSceneFormat, StencilTest::Outside,
                       { bind(halves_[cur]) }, {}, effectConstants(fe, 0, 0, 0), sc, stencilDepth);
    };

    // Region-bounded identity pass (HSV with identity constants): copies the
    // effect's output back onto the original half so the chain's swap count
    // is even. Stencil-inside like the effect itself, so outside-clip pixels
    // of the ORIGINAL half are never touched. ONLY for PARTIAL regions: a
    // full-frame single-pass effect fully covers the destination half
    // (pre-copy seeds outside-clip when clipped), and a copy-back there would
    // double the cost of every full-window saturate/hue. `cur` ending on
    // either half is fine: the final copy reads halves_[cur].
    auto copyBackAndSwap = [&](const gpu::IRect& sc, uint8_t stencilDepth) {
        effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
            fullscreenPass("copy back", programs.hsv, dst, kSceneFormat, StencilTest::Inside,
                           { bind(src) }, {}, builder_.pushConstants(HSVConstants::identity(fw, fh)),
                           sc, stencilDepth);
        });
    };

    // SCALED targets (Target::scale) aren't scene attachments: they hold their
    // clear value until their first write in a frame. That write's pass clears
    // them first; a pass that reads one before then clears it here, as a pass
    // of its own with no draws.
    auto clearScaled = [&](TargetMask reads) {
        const TargetMask pendingClear = reads & scaled & static_cast<TargetMask>(~scaledWritten);
        for (size_t i = 0; pendingClear != 0 && i < targets_.size(); ++i) {
            if ((pendingClear & (1u << i)) == 0) continue;
            const auto te = targetExtent(i);
            auto& p = builder_.beginPass("clear target");
            p.colorCount = 1;
            p.color[0] = { targetTextures_[i], LoadOp::Clear, StoreOp::Store, targets_[i].clear };
            p.renderArea = { 0, 0, static_cast<int32_t>(te.width), static_cast<int32_t>(te.height) };
            builder_.endPass();
            scaledWritten |= static_cast<TargetMask>(1u << i);
        }
    };

    // Target t into its scratch (`toScratch`) or back, over `sc` in its pixels:
    // a plain copy draw (a target pass has no stencil, so it writes the whole
    // scissor).
    auto copyTarget = [&](size_t t, bool toScratch, const gpu::IRect& sc) {
        if (sc.isEmpty()) return;
        const auto scratch = targetScratch(t);
        if (!scratch) return;
        const auto src = toScratch ? targetTextures_[t] : scratch;
        const auto dst = toScratch ? scratch : targetTextures_[t];
        fullscreenPass("copy target", programs.copy, dst, targets_[t].format, StencilTest::None,
                       { bind(src) }, {}, effectConstants(targetExtent(t), 0, 0, 0), sc, 0);
    };
    // The targets of `mask` copied in (before a pass: a Replace target too,
    // since a stencil-clipped or discarding shader leaves pixels of the rect
    // unwritten, and the copy back takes the whole rect) or back (after it).
    auto copyTargets = [&](TargetMask mask, bool in, const gpu::IRect& sc) {
        for (size_t t = 0; mask != 0 && t < targets_.size(); ++t)
            if ((mask & (1u << t)) != 0)
                copyTarget(t, in, sc);
    };
    // `r` (frame px) in a target's pixels at scale `s`, padded, clamped to its extent `e`.
    auto inPixels = [](juce::Rectangle<float> r, float s, float padX, float padY, gpu::Extent e) -> gpu::IRect {
        r = (r * s).expanded(padX, padY);
        const int x0 = std::max(0, (int) std::floor(r.getX())), y0 = std::max(0, (int) std::floor(r.getY()));
        const int x1 = std::min((int) e.width, (int) std::ceil(r.getRight())), y1 = std::min((int) e.height, (int) std::ceil(r.getBottom()));
        if (x1 <= x0 || y1 <= y0) return {};
        return { x0, y0, x1 - x0, y1 - y0 };
    };

    // ---- jvk::Shader draws ----------------------------------------------------

    // The pipeline of a Shader draw in a pass whose colour attachments are
    // `formats`: location L blends as its attachment takes it (location 0:
    // Replace for a scene reader, which REPLACES its region; else Over; a
    // target location: that target's Blend), nothing where it doesn't write.
    // `locationTarget(L)`: the target at location L (-1: none).
    auto shaderPipeline = [&](const ShaderCapture& sc, const gpu::AttachmentFormats& formats,
                              auto&& locationTarget, bool clip) -> gpu::PipelineId
    {
        gpu::PipelineDesc d;
        d.program = sc.program;
        d.attachments = formats;
        for (uint32_t l = 0; l < formats.colorCount; ++l)
        {
            const bool writes = (sc.outputs & (1u << l)) != 0 && (formats.unusedMask & (1u << l)) == 0;
            if (!writes)  { d.blend[l] = gpu::BlendState::none(); continue; }
            if (l == 0)   { d.blend[0] = sc.readsScene ? gpu::BlendState::replace() : gpu::BlendState::over(); continue; }
            const int t = locationTarget(l);
            d.blend[l] = t >= 0 ? targetBlend(targets_[static_cast<size_t>(t)].blend) : gpu::BlendState::none();
        }
        if (clip && formats.depthStencil)
            d.stencil = { true, gpu::CompareOp::Equal, gpu::StencilOp::Keep, gpu::StencilOp::Keep, 0xFF, 0xFF };
        return ctx.pipeline(d);
    };

    // One Shader draw into the open pass: the region quad (one triangle
    // covering twice the region; the scissor — the clip within the region —
    // keeps it there), in a pass of `viewport` pixels. `scene`: the half a
    // scene reader reads (none: a reader draws nothing). Targets it samples
    // are bound by slot (`targetTexture(t)`: the texture it reads target t
    // through).
    auto shaderDraw = [&](const ShaderCapture& sc, const DrawShaderParams& sp, gpu::PipelineId pipeline,
                          juce::Rectangle<float> region, gpu::Extent viewport,
                          const juce::Rectangle<int>& clip, uint8_t stencilRef, gpu::TextureId scene)
    {
        if (!pipeline || (sc.readsScene && !scene)) return;
        gpu::Draw d;
        d.pipeline = pipeline;
        const auto c = clip.getIntersection(region.getSmallestIntegerContainer());
        d.scissor = Emitter::clamp(c, viewport);
        if (d.scissor.isEmpty()) return;
        d.stencilRef = stencilRef;
        d.textureCount = sc.textureCount;
        for (uint32_t i = 0; i < sc.textureCount; ++i)
        {
            auto b = sc.textures[i];
            if (static_cast<int>(i) == sc.sceneSlot) b = bind(scene ? scene : ctx.defaultTexture());
            else if (sc.slotTarget[i] >= 0 && static_cast<size_t>(sc.slotTarget[i]) < targetTextures_.size())
                b = bind(targetTextures_[static_cast<size_t>(sc.slotTarget[i])]);
            if (!b.texture) b = defaultBinding;
            d.textures[i] = b;
        }
        d.storageCount = sc.storageCount;
        for (uint32_t i = 0; i < sc.storageCount; ++i)
            d.storage[i] = builder_.pushStorage(shaderData_.data() + sc.storage[i].offset, sc.storage[i].size);
        ShaderConstants k;
        k.resolutionX = region.getWidth();
        k.resolutionY = region.getHeight();
        k.time        = frameTime;
        k.viewportW   = static_cast<float>(viewport.width);
        k.viewportH   = static_cast<float>(viewport.height);
        k.regionX     = region.getX();
        k.regionY     = region.getY();
        for (uint32_t i = 0; i < sp.constantCount && i < static_cast<uint32_t>(kShaderDrawConstants); ++i)
            k.draw[i] = sp.constants[i];
        d.constants = builder_.pushConstants(k);
        d.vertexCount = 3;
        builder_.draw(d);
    };

    // The colour formats of a TARGET PASS / a MAIN PASS: location L = the
    // target its shader writes there (TargetSlots), location 0 the main colour
    // (a main pass) or nothing (a target pass); the stencil with the main colour.
    auto slotFormats = [&](TargetSlots tslots, bool withMain) {
        gpu::AttachmentFormats f;
        f.colorCount = static_cast<uint8_t>(std::max<uint32_t>(slotCount(tslots), 1u));
        if (withMain) f.colorFormats[0] = kSceneFormat;
        else          f.unusedMask |= 1u;
        for (uint32_t l = 1; l < f.colorCount; ++l)
        {
            const int t = slotTarget(tslots, l);
            if (t < 0) f.unusedMask |= static_cast<uint8_t>(1u << l);
            else       f.colorFormats[l] = targets_[static_cast<size_t>(t)].format;
        }
        f.depthStencil = withMain;
        return f;
    };

    // A TARGET PASS: a shader writing targets (one size) and not the main
    // colour, in their pixels (its region and clip scale with them). A scene
    // reader reads the current half.
    auto targetPass = [&](const DrawCommand& c, const DrawShaderParams& sp, const ShaderCapture& sc) {
        int first = -1;
        for (uint32_t l = 1; l < 8 && first < 0; ++l)
            first = slotTarget(sp.targetSlots, l);
        if (first < 0) return;
        const auto e = targetExtent(static_cast<size_t>(first));
        const float s = targetScale(static_cast<size_t>(first));
        // The ONE rect of this pass, in the targets' pixels: the shader's
        // scissor, the copies' rect and the render area, so no pixel the
        // shader writes escapes the copy back.
        const auto regionT = sp.region * s;
        const auto clipT   = (c.clipBounds.toFloat() * s).getSmallestIntegerContainer();
        const auto cT      = clipT.getIntersection(regionT.getSmallestIntegerContainer());
        const gpu::IRect sc0 = Emitter::clamp(cT, e);
        if (sc0.isEmpty()) return;
        endScene();
        clearScaled(sp.targetsRead);
        const TargetMask written = sp.targetsWritten, versioned = sp.targetsRead & written;
        const TargetMask clears = written & scaled & static_cast<TargetMask>(~scaledWritten);
        copyTargets(versioned, true, sc0);

        const auto formats = slotFormats(sp.targetSlots, false);
        const auto pipeline = shaderPipeline(sc, formats, [&](uint32_t l) { return slotTarget(sp.targetSlots, l); }, false);
        auto& p = builder_.beginPass("shader → targets");
        p.colorCount = formats.colorCount;
        for (uint32_t l = 1; l < formats.colorCount; ++l)
            if (const int t = slotTarget(sp.targetSlots, l); t >= 0)
            {
                const auto ti = static_cast<size_t>(t);
                const bool clear = (clears & (1u << ti)) != 0;
                p.color[l] = { (versioned & (1u << ti)) != 0 ? targetScratch(ti) : targetTextures_[ti],
                               clear ? LoadOp::Clear : LoadOp::Load, StoreOp::Store, targets_[ti].clear };
            }
        // Whole while a target clears (a clear covers the attachment), else the
        // rect the pass touches: the tiles outside it are neither loaded nor stored.
        p.renderArea = clears != 0 ? gpu::IRect { 0, 0, (int32_t) e.width, (int32_t) e.height } : sc0;
        shaderDraw(sc, sp, pipeline, regionT, e, clipT, 0, sc.readsScene ? halves_[cur] : gpu::TextureId {});
        builder_.endPass();
        scaledWritten |= written & scaled;
        copyTargets(versioned, false, sc0);
    };

    // A pass writing the main colour AND targets at scale 1: the main colour's
    // half and the targets attached together, with the stencil, so the clip
    // applies to both. A scene reader samples the current half and REPLACES
    // its region in the other, as every scene reader does (pre-copy,
    // copy-back); any other blends over the current half in place.
    auto mainPass = [&](const DrawCommand& c, const DrawShaderParams& sp, const ShaderCapture& sc) {
        const auto R = roiFinal(sp.region, c.clipBounds);
        if (R.isEmpty()) return;
        const gpu::IRect sc0 = roiScissor(R, 0.0f, 0.0f);
        if (sc0.isEmpty()) return;
        endScene();
        clearScaled(sp.targetsRead);
        const TargetMask versioned = sp.targetsRead & sp.targetsWritten;
        copyTargets(versioned, true, sc0);
        const auto formats = slotFormats(sp.targetSlots, true);
        const bool clip = emitter_.stencilTested();
        const auto pipeline = shaderPipeline(sc, formats, [&](uint32_t l) { return slotTarget(sp.targetSlots, l); }, clip);
        auto pass = [&](gpu::TextureId scene, int half) {
            auto& p = builder_.beginPass("shader → main + targets");
            p.colorCount = formats.colorCount;
            p.color[0] = { halves_[half], LoadOp::Load, StoreOp::Store, {} };
            for (uint32_t l = 1; l < formats.colorCount; ++l)
                if (const int t = slotTarget(sp.targetSlots, l); t >= 0)
                {
                    const auto ti = static_cast<size_t>(t);
                    p.color[l] = { (versioned & (1u << ti)) != 0 ? targetScratch(ti) : targetTextures_[ti],
                                   LoadOp::Load, StoreOp::Store, targets_[ti].clear };
                }
            p.depthStencil = { stencil_, LoadOp::Load, StoreOp::Store, 0 };
            p.renderArea = sc0;
            shaderDraw(sc, sp, pipeline, sp.region, fe, c.clipBounds, c.stencilDepth, scene);
            builder_.endPass();
        };
        if (sc.readsScene) {
            const bool whole = coversFrame(sc0);
            if (whole) preCopyIfClipped(c.stencilDepth, sc0);
            pass(halves_[cur], cur ^ 1);
            cur ^= 1;
            if (!whole) copyBackAndSwap(sc0, c.stencilDepth);
        } else {
            pass({}, cur);
        }
        copyTargets(versioned, false, sc0);
    };

    // A shader writing the main colour alone, as a pass of its own: a scene
    // reader (ping-ponged) or a target reader (blended Over the current half).
    auto shaderMainOnly = [&](const DrawCommand& c, const DrawShaderParams& sp, const ShaderCapture& sc) {
        gpu::AttachmentFormats formats;
        formats.colorCount = 1;
        formats.colorFormats[0] = kSceneFormat;
        formats.depthStencil = true;
        const auto pipeline = shaderPipeline(sc, formats, [](uint32_t) { return -1; }, emitter_.stencilTested());
        auto pass = [&](gpu::TextureId scene, gpu::TextureId dst, const gpu::IRect& area) {
            auto& p = builder_.beginPass("shader → main");
            p.colorCount = 1;
            p.color[0] = { dst, LoadOp::Load, StoreOp::Store, {} };
            p.depthStencil = { stencil_, LoadOp::Load, StoreOp::Store, 0 };
            p.renderArea = area;
            shaderDraw(sc, sp, pipeline, sp.region, fe, c.clipBounds, c.stencilDepth, scene);
            builder_.endPass();
        };
        const gpu::IRect whole { 0, 0, (int32_t) fe.width, (int32_t) fe.height };
        if (sc.readsScene) {
            // A SCENE READER (it may read targets too): a non-separable effect,
            // ROI-walked like EffectHSV. One region pass samples the current half
            // as its `scene` and REPLACES the region in the other; a partial
            // region is copied back so the chain ends on the starting half.
            const auto R = roiFinal(sp.region, c.clipBounds);
            if (R.isEmpty()) return;
            const gpu::IRect sc0 = roiScissor(R, 0.0f, 0.0f);
            if (sc0.isEmpty()) return;
            endScene();
            clearScaled(sp.targetsRead);
            if (coversFrame(sc0)) {
                preCopyIfClipped(c.stencilDepth, sc0);
                effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) { pass(src, dst, whole); });
            } else {
                effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) { pass(src, dst, whole); });
                copyBackAndSwap(sc0, c.stencilDepth);
            }
        } else {
            // A TARGET READER writing the main colour alone, blended Over the
            // current half in place.
            endScene();
            clearScaled(sp.targetsRead);
            pass({}, halves_[cur], whole);
        }
    };

    // ---- Effects on a target ------------------------------------------------
    //
    // An EFFECT ON A TARGET (Graphics::setTarget, then a blur, a blurred shape
    // or path, an hsv transform): read and written, so it runs through the
    // target's scratch — a separable blur across into the scratch and down
    // back into the target; anything else copies what it reads into the
    // scratch, then draws from there back over its region. In the target's
    // pixels: the recorded region, clip and radii scale with it, and the clip
    // is its bounds (a target has no stencil). A blurred path, whose segments
    // are in frame pixels, runs on a target at scale 1 only. `k`: its index.
    auto targetEffect = [&](const DrawCommand& c, size_t k) {
        const auto t = static_cast<size_t>(c.target - 1);
        if (t >= targets_.size() || (recorder_.liveTargets(k + 1) & (1u << t)) == 0)
            return;   // what it would make, nothing reads
        const float s = targetScale(t);
        const gpu::Extent e = targetExtent(t);
        const auto format = targets_[t].format;
        endScene();
        clearScaled(static_cast<TargetMask>(1u << t));   // not drawn this frame: its clear value
        const auto target = targetTextures_[t], scratch = targetScratch(t);
        if (!target || !scratch) return;

        if (c.op == DrawOp::EffectKernel) {
            const auto& bp = arena.read<BlurParams>(c.dataOffset);
            const auto R = roiFinal(bp.region, c.clipBounds);
            const float m = std::ceil(bp.radius * std::max(1.0f, bp.scale) * s) + 2.0f;
            const gpu::IRect scH = inPixels(R, s, 0.0f, m, e), scV = inPixels(R, s, 0.0f, 0.0f, e);
            if (scV.isEmpty()) return;
            fullscreenPass("blur H", programs.blur, scratch, format, StencilTest::None,
                           { bind(target) }, {}, effectConstants(e, 1.0f, 0.0f, bp.radius * s), scH, 0);
            fullscreenPass("blur V", programs.blur, target, format, StencilTest::None,
                           { bind(scratch) }, {}, effectConstants(e, 0.0f, 1.0f, bp.radius * s), scV, 0);
        }
        else if (c.op == DrawOp::EffectHSV) {
            const auto& hp = arena.read<HSVParams>(c.dataOffset);
            const gpu::IRect sc = inPixels(roiFinal(hp.region, c.clipBounds), s, 0.0f, 0.0f, e);
            if (sc.isEmpty()) return;
            HSVConstants k2;
            k2.viewportW = (float) e.width; k2.viewportH = (float) e.height;
            k2.scaleH = hp.scaleH; k2.scaleS = hp.scaleS; k2.scaleV = hp.scaleV;
            k2.deltaH = hp.deltaH; k2.deltaS = hp.deltaS; k2.deltaV = hp.deltaV;
            copyTarget(t, true, sc);
            fullscreenPass("hsv", programs.hsv, target, format, StencilTest::None,
                           { bind(scratch) }, {}, builder_.pushConstants(k2), sc, 0);
        }
        else if (c.op == DrawOp::BlurShape) {
            const auto& sp = arena.read<BlurShapeParams>(c.dataOffset);
            ShapeBlurConstants pc;
            pc.viewportW = (float) e.width; pc.viewportH = (float) e.height;
            // Frame px = target px / s: the inverse transform takes the target's pixels.
            pc.invCol0X = sp.invXform[0] / s; pc.invCol0Y = sp.invXform[1] / s;
            pc.invCol1X = sp.invXform[2] / s; pc.invCol1Y = sp.invXform[3] / s;
            pc.invCol2X = sp.invXform[4];     pc.invCol2Y = sp.invXform[5];
            pc.shapeHalfX = sp.shapeHalf[0];  pc.shapeHalfY = sp.shapeHalf[1];
            pc.lineBX = sp.lineB[0];          pc.lineBY = sp.lineB[1];
            pc.maxRadius = sp.maxRadius;      pc.falloff = sp.falloff;
            pc.blurStep = sp.blurStep * s;    pc.cornerRadius = sp.cornerRadius;
            pc.lineThickness = sp.lineThickness;
            pc.shapeType = static_cast<int>(sp.shapeType);
            pc.edgePlacement = static_cast<int>(sp.edgePlacement);
            pc.inverted = static_cast<int>(sp.inverted);
            pc.kernelType = sp.mode == 0 ? 0 : sp.mode == 1 ? 1 : 2;
            const auto R = roiFinal(sp.region, c.clipBounds);
            const float m = std::ceil(sp.maxRadius * pc.blurStep) + 2.0f;
            const gpu::IRect scR = inPixels(R, s, 0.0f, 0.0f, e);
            if (scR.isEmpty()) return;
            if (pc.kernelType == 0) {
                pc.dirX = 1.0f; pc.dirY = 0.0f;
                fullscreenPass("shape blur H", programs.shapeBlur, scratch, format, StencilTest::None,
                               { bind(target) }, {}, builder_.pushConstants(pc), inPixels(R, s, 0.0f, m, e), 0);
                pc.dirX = 0.0f; pc.dirY = 1.0f;
                fullscreenPass("shape blur V", programs.shapeBlur, target, format, StencilTest::None,
                               { bind(scratch) }, {}, builder_.pushConstants(pc), scR, 0);
            } else {
                copyTarget(t, true, inPixels(R, s, m, m, e));
                pc.dirX = 1.0f; pc.dirY = 0.0f;
                fullscreenPass("shape blur", programs.shapeBlur, target, format, StencilTest::None,
                               { bind(scratch) }, {}, builder_.pushConstants(pc), scR, 0);
            }
        }
        else if (c.op == DrawOp::BlurPath && s == 1.0f) {
            const auto& bp = arena.read<BlurPathParams>(c.dataOffset);
            PathBlurConstants pc;
            pc.viewportW = (float) e.width; pc.viewportH = (float) e.height;
            pc.maxRadius = bp.maxRadius;          pc.falloff = bp.falloff;
            pc.strokeHalfWidth = bp.strokeHalfWidth;
            pc.segmentStart = bp.segmentStart;    pc.segmentCount = bp.segmentCount;
            pc.fillRule = bp.fillRule;
            pc.edgePlacement = static_cast<int32_t>(bp.edgePlacement);
            pc.inverted = static_cast<int32_t>(bp.inverted);
            pc.stripCount = bp.stripCount;        pc.stripMinY = bp.stripMinY;
            pc.invStripH = bp.invStripH;
            pc.kernelType = bp.mode == 0 ? 0 : bp.mode == 1 ? 1 : 2;
            const auto R = roiFinal(bp.region, c.clipBounds);
            const float m = std::ceil(bp.maxRadius) + 2.0f;
            const gpu::IRect scR = inPixels(R, 1.0f, 0.0f, 0.0f, e);
            if (scR.isEmpty()) return;
            if (pc.kernelType == 0) {
                pc.dirX = 1.0f; pc.dirY = 0.0f;
                fullscreenPass("path blur H", programs.pathBlur, scratch, format, StencilTest::None,
                               { bind(target) }, { segments }, builder_.pushConstants(pc), inPixels(R, 1.0f, 0.0f, m, e), 0);
                pc.dirX = 0.0f; pc.dirY = 1.0f;
                fullscreenPass("path blur V", programs.pathBlur, target, format, StencilTest::None,
                               { bind(scratch) }, { segments }, builder_.pushConstants(pc), scR, 0);
            } else {
                copyTarget(t, true, inPixels(R, 1.0f, m, m, e));
                pc.dirX = 1.0f; pc.dirY = 0.0f;
                fullscreenPass("path blur", programs.pathBlur, target, format, StencilTest::None,
                               { bind(scratch) }, { segments }, builder_.pushConstants(pc), scR, 0);
            }
        }
    };

    // ---- Scene draws ----------------------------------------------------------

    // Binds the pipeline, constants and storage of a built-in scene draw.
    // False (draw nothing) when its pipeline couldn't be made.
    auto setPaint = [&](Paint kind, const DrawCommand& cmd) -> const PaintVariant* {
        const auto* v = paintPipeline(kind, cmd.target, emitter_.stencilTested(), live);
        if (v == nullptr) return nullptr;
        emitter_.invalidate();
        emitter_.setPipeline(v->id);
        if (kind != Paint::Path)
        {
            PaintConstants k;
            k.viewportW = fw; k.viewportH = fh;
            k.paintsMain = v->paintsMain ? 1 : 0;
            emitter_.setConstants(builder_.pushConstants(k));
            if (kind == Paint::Noise) emitter_.setStorage({});
            else                      emitter_.setStorage({ frameData });
        }
        return v;
    };

    // The pipeline of a clip draw: colour writes off on every scene attachment;
    // the stencil compared EQUAL against the reference under `readMask` and
    // `passOp` where it passes.
    auto clipPipeline = [&](gpu::ProgramId program, gpu::StencilOp passOp, uint8_t readMask) {
        gpu::PipelineDesc d;
        d.program = program;
        d.vertexLayout = uiVertexLayout();
        d.attachments = sceneFormats();
        for (uint32_t i = 0; i < d.attachments.colorCount; ++i)
            d.blend[i] = gpu::BlendState::none();
        d.stencil = { true, gpu::CompareOp::Equal, passOp, gpu::StencilOp::Keep, readMask, 0xFF };
        return ctx.pipeline(d);
    };

    // Rasterizes a clip shape's cover quad (physical px) with `pipeline`:
    // fragments the clip shader keeps apply the stencil op at `reference`.
    auto coverQuad = [&](const DrawCommand& cmd, const juce::Rectangle<float>& coverRect, uint8_t reference) {
        const float x = coverRect.getX(), y = coverRect.getY();
        const float w = coverRect.getWidth(), h = coverRect.getHeight();
        auto mkv = [](float px, float py) {
            return UIVertex { glm::vec2 { px, py }, glm::vec4 { 0.0f }, glm::vec2 { 0.0f },
                              glm::vec4 { 0.0f }, glm::vec4 { 0.0f } };
        };
        const UIVertex verts[6] = { mkv(x, y), mkv(x + w, y), mkv(x + w, y + h),
                                    mkv(x, y), mkv(x + w, y + h), mkv(x, y + h) };
        emitter_.draw(cmd, verts, 6, reference);
    };

    auto clipShape = [&](const DrawCommand& cmd, bool push) {
        const auto& p = arena.read<ClipShapeParams>(cmd.dataOffset);
        emitter_.invalidate();
        if (p.shapeType == 3u)
        {
            // An image-alpha clip: the mask's alpha, at least one half, is inside.
            if (!p.mask) return;
            emitter_.setPipeline(clipPipeline(programs.clipMask,
                push ? gpu::StencilOp::IncrementWrap : gpu::StencilOp::DecrementWrap, 0xFF));
            emitter_.setResources(bind(p.mask), defaultBinding);
            emitter_.setTextureCount(1);
            ClipMaskConstants k;
            k.viewportW = fw; k.viewportH = fh;
            std::copy(std::begin(p.toMask), std::end(p.toMask), k.toMask);
            k.maskW = p.maskW; k.maskH = p.maskH;
            k.nearest = p.maskNearest;
            emitter_.setConstants(builder_.pushConstants(k));
            coverQuad(cmd, p.coverRect, cmd.stencilDepth);
        }
        else
        {
            emitter_.setPipeline(clipPipeline(programs.clip,
                push ? gpu::StencilOp::IncrementWrap : gpu::StencilOp::DecrementWrap, 0xFF));
            emitter_.setTextureCount(0);
            emitter_.setStorage({ segments });
            ClipConstants k;
            k.viewportW = fw; k.viewportH = fh;
            k.shapeType    = p.shapeType;
            k.centerX      = p.centerX;      k.centerY = p.centerY;
            k.halfW        = p.halfW;        k.halfH   = p.halfH;
            k.cornerRadius = p.cornerRadius;
            k.segmentStart = p.segmentStart; k.segmentCount = p.segmentCount;
            k.fillRule     = p.fillRule;
            k.stripCount   = p.stripCount;   k.stripMinY = p.stripMinY; k.invStripH = p.invStripH;
            emitter_.setConstants(builder_.pushConstants(k));
            coverQuad(cmd, p.coverRect, cmd.stencilDepth);
        }
        emitter_.invalidate();
    };

    // ---- The walk -------------------------------------------------------------

    // Commands that run as passes of their own (each ends the scene pass if it's open).
    auto runsOwnPass = [&](const DrawCommand& c) {
        switch (c.op) {
            case DrawOp::EffectKernel: case DrawOp::EffectHSV: case DrawOp::BlurShape: case DrawOp::BlurPath:
                return true;
            case DrawOp::DrawShader: {
                const auto& sp = arena.read<DrawShaderParams>(c.dataOffset);
                return sp.shaderIndex == ~0u || !sp.inScene;
            }
            default:
                return false;
        }
    };

    // A scaled target an in-scene shader reads before any pass writes it this
    // frame holds its clear value: cleared before the first scene pass, rather
    // than by breaking the scene pass at the read (a pass that reads one
    // clears it at its own boundary).
    {
        TargetMask writtenAhead = 0, preclear = 0;
        for (const auto& c : commands) {
            if (c.op != DrawOp::DrawShader) continue;
            const auto& sp = arena.read<DrawShaderParams>(c.dataOffset);
            if (sp.shaderIndex == ~0u) continue;
            if (sp.inScene) preclear     |= sp.targetsRead & scaled & static_cast<TargetMask>(~writtenAhead);
            else            writtenAhead |= sp.targetsWritten & scaled;
        }
        clearScaled(preclear);
    }

    beginScene(/*withClears=*/true, 0);

    for (size_t k = 0; k < commands.size(); ++k) {
        const auto& cmd = commands[k];
        live = recorder_.liveTargets(k);
        // A draw INTO a target nothing reads from here on is a dead store.
        // (Clip ops keep the stencil balanced whatever the target — a missing
        // one included, kNoTarget: tested before the shift; a shader draw
        // writes the locations its shader declares, judged below.)
        if (cmd.target > 0 && !Recorder::isClipOp(cmd.op) && cmd.op != DrawOp::DrawShader
            && cmd.target != kNoTarget && (live & (1u << (cmd.target - 1))) == 0)
            continue;
        // A shader inside the scene pass that reads a scaled target before its
        // first write this frame: that target is cleared first.
        if (cmd.op == DrawOp::DrawShader) {
            const auto& sp = arena.read<DrawShaderParams>(cmd.dataOffset);
            if (sp.inScene && (sp.targetsRead & scaled & static_cast<TargetMask>(~scaledWritten)) != 0) {
                endScene();
                clearScaled(sp.targetsRead);
            }
        }
        if (!sceneOpen && !runsOwnPass(cmd))
            beginScene(/*withClears=*/false, k);
        if (cmd.target > 0 && cmd.target != kNoTarget && Recorder::isPassEffect(cmd.op)) {
            targetEffect(cmd, k);
            continue;
        }

        switch (cmd.op)
        {
            case DrawOp::EffectKernel: {
                // Separable Gaussian blur, ROI-walked: V (last) outputs the final
                // region R, so H must output R padded by V's VERTICAL apron; H's
                // own horizontal apron reads the true scene (always valid). Two
                // swaps land `cur` back on the starting half, so pixels outside
                // R keep the original scene untouched.
                const auto& bp = arena.read<BlurParams>(cmd.dataOffset);
                const auto R = roiFinal(bp.region, cmd.clipBounds);
                if (R.isEmpty()) break;
                const float m = std::ceil(bp.radius * std::max(1.0f, bp.scale)) + 2.0f;
                const gpu::IRect scH = roiScissor(R, 0.0f, m);
                const gpu::IRect scV = roiScissor(R, 0.0f, 0.0f);
                endScene();
                // Seed the intermediate half's outside-clip pixels within H's
                // scissor (V's taps cross the clip edge and read them).
                preCopyIfClipped(cmd.stencilDepth, scH);
                effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                    fullscreenPass("blur H", programs.blur, dst, kSceneFormat, StencilTest::Inside,
                                   { bind(src) }, {}, effectConstants(fe, 1.0f, 0.0f, bp.radius), scH, cmd.stencilDepth);
                });
                effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                    fullscreenPass("blur V", programs.blur, dst, kSceneFormat, StencilTest::Inside,
                                   { bind(src) }, {}, effectConstants(fe, 0.0f, 1.0f, bp.radius), scV, cmd.stencilDepth);
                });
                break;
            }

            case DrawOp::EffectHSV: {
                // Non-separable HSV, ROI-walked: one region-bounded transform pass
                // + one region-bounded identity copy-back so the chain ends on the
                // starting half. No pre-copy: outside-clip and outside-region
                // pixels of the original half are simply never written.
                const auto& hp = arena.read<HSVParams>(cmd.dataOffset);
                const auto R = roiFinal(hp.region, cmd.clipBounds);
                if (R.isEmpty()) break;
                const gpu::IRect sc = roiScissor(R, 0.0f, 0.0f);
                HSVConstants pc;
                pc.viewportW = fw; pc.viewportH = fh;
                pc.scaleH = hp.scaleH; pc.scaleS = hp.scaleS; pc.scaleV = hp.scaleV;
                pc.deltaH = hp.deltaH; pc.deltaS = hp.deltaS; pc.deltaV = hp.deltaV;
                const auto constants = builder_.pushConstants(pc);
                endScene();
                if (coversFrame(sc)) {
                    // Full-window transform: ONE pass, single swap. Pre-copy
                    // seeds outside-clip pixels when clipped.
                    preCopyIfClipped(cmd.stencilDepth, sc);
                    effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                        fullscreenPass("hsv", programs.hsv, dst, kSceneFormat, StencilTest::Inside,
                                       { bind(src) }, {}, constants, sc, cmd.stencilDepth);
                    });
                } else {
                    // Partial region: region pass + region copy-back.
                    effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                        fullscreenPass("hsv", programs.hsv, dst, kSceneFormat, StencilTest::Inside,
                                       { bind(src) }, {}, constants, sc, cmd.stencilDepth);
                    });
                    copyBackAndSwap(sc, cmd.stencilDepth);
                }
                break;
            }

            case DrawOp::BlurShape: {
                const auto& sp = arena.read<BlurShapeParams>(cmd.dataOffset);
                ShapeBlurConstants pc;
                pc.viewportW = fw; pc.viewportH = fh;
                pc.invCol0X = sp.invXform[0]; pc.invCol0Y = sp.invXform[1];
                pc.invCol1X = sp.invXform[2]; pc.invCol1Y = sp.invXform[3];
                pc.invCol2X = sp.invXform[4]; pc.invCol2Y = sp.invXform[5];
                pc.shapeHalfX = sp.shapeHalf[0];
                pc.shapeHalfY = sp.shapeHalf[1];
                pc.lineBX     = sp.lineB[0];
                pc.lineBY     = sp.lineB[1];
                pc.maxRadius     = sp.maxRadius;
                pc.falloff       = sp.falloff;
                pc.blurStep      = sp.blurStep;
                pc.cornerRadius  = sp.cornerRadius;
                pc.lineThickness = sp.lineThickness;
                pc.shapeType     = static_cast<int>(sp.shapeType);
                pc.edgePlacement = static_cast<int>(sp.edgePlacement);
                pc.inverted      = static_cast<int>(sp.inverted);
                // mode: 0 (Low)    → 2 separable passes (H, V). The V pass gates
                //                    each tap by the TAP's own radius (see
                //                    shape_blur.frag) so low-radius falloff rows
                //                    can't streak sharp content into the interior.
                //       1 (Medium) → 1 pass, 32-tap Poisson-disc kernel.
                //       2 (High)   → 1 pass, true 2D Gaussian (Linear Sampling
                //                    Trick: ¼ the taps of a naïve N² loop).
                // Medium + High both read the scene source directly.
                const int32_t kt = (sp.mode == 0) ? 0 : (sp.mode == 1) ? 1 : 2;
                pc.kernelType = kt;

                // ROI: record side computed the shape's blur AABB (empty for
                // inverted = whole clip). Kernel apron in physical px.
                const auto R = roiFinal(sp.region, cmd.clipBounds);
                if (R.isEmpty()) break;
                const float m = std::ceil(sp.maxRadius * sp.blurStep) + 2.0f;
                const gpu::IRect scR = roiScissor(R, 0.0f, 0.0f);

                endScene();
                auto blurPass = [&](const char* label, float dirX, float dirY, const gpu::IRect& sc) {
                    pc.dirX = dirX; pc.dirY = dirY;
                    const auto constants = builder_.pushConstants(pc);
                    effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                        fullscreenPass(label, programs.shapeBlur, dst, kSceneFormat, StencilTest::Inside,
                                       { bind(src) }, {}, constants, sc, cmd.stencilDepth);
                    });
                };
                if (kt == 0) {
                    // Separable H then V — V's vertical apron dictates H's output
                    // rect; two swaps land back on the start half.
                    const gpu::IRect scH = roiScissor(R, 0.0f, m);
                    preCopyIfClipped(cmd.stencilDepth, scH);
                    blurPass("shape blur H", 1.0f, 0.0f, scH);
                    blurPass("shape blur V", 0.0f, 1.0f, scR);
                } else if (coversFrame(scR)) {
                    // Full-window 2D kernel (inverted blurs land here): one pass,
                    // single swap.
                    preCopyIfClipped(cmd.stencilDepth, scR);
                    blurPass("shape blur", 1.0f, 0.0f, scR);
                } else {
                    // 2D kernel, partial region: one region pass reading the true
                    // scene (taps beyond R are always valid) + region copy-back.
                    blurPass("shape blur", 1.0f, 0.0f, scR);
                    copyBackAndSwap(scR, cmd.stencilDepth);
                }
                break;
            }

            case DrawOp::BlurPath: {
                const auto& bp = arena.read<BlurPathParams>(cmd.dataOffset);
                PathBlurConstants pc;
                pc.viewportW = fw; pc.viewportH = fh;
                pc.maxRadius       = bp.maxRadius;
                pc.falloff         = bp.falloff;
                pc.strokeHalfWidth = bp.strokeHalfWidth;
                pc.segmentStart    = bp.segmentStart;
                pc.segmentCount    = bp.segmentCount;
                pc.fillRule        = bp.fillRule;
                pc.edgePlacement   = static_cast<int32_t>(bp.edgePlacement);
                pc.inverted        = static_cast<int32_t>(bp.inverted);
                pc.stripCount      = bp.stripCount;
                pc.stripMinY       = bp.stripMinY;
                pc.invStripH       = bp.invStripH;
                // Same mode layout + ROI walk as BlurShape above. Distances here
                // are already physical px.
                const int32_t kt = (bp.mode == 0) ? 0 : (bp.mode == 1) ? 1 : 2;
                pc.kernelType = kt;

                const auto R = roiFinal(bp.region, cmd.clipBounds);
                if (R.isEmpty()) break;
                const float m = std::ceil(bp.maxRadius) + 2.0f;
                const gpu::IRect scR = roiScissor(R, 0.0f, 0.0f);

                endScene();
                auto blurPass = [&](const char* label, float dirX, float dirY, const gpu::IRect& sc) {
                    pc.dirX = dirX; pc.dirY = dirY;
                    const auto constants = builder_.pushConstants(pc);
                    effectPassAndSwap([&](gpu::TextureId src, gpu::TextureId dst) {
                        fullscreenPass(label, programs.pathBlur, dst, kSceneFormat, StencilTest::Inside,
                                       { bind(src) }, { segments }, constants, sc, cmd.stencilDepth);
                    });
                };
                if (kt == 0) {
                    const gpu::IRect scH = roiScissor(R, 0.0f, m);
                    preCopyIfClipped(cmd.stencilDepth, scH);
                    blurPass("path blur H", 1.0f, 0.0f, scH);
                    blurPass("path blur V", 0.0f, 1.0f, scR);
                } else if (coversFrame(scR)) {
                    preCopyIfClipped(cmd.stencilDepth, scR);
                    blurPass("path blur", 1.0f, 0.0f, scR);
                } else {
                    blurPass("path blur", 1.0f, 0.0f, scR);
                    copyBackAndSwap(scR, cmd.stencilDepth);
                }
                break;
            }

            case DrawOp::PushClipRect:
                // cmd.clipBounds is already the pixel-space intersection of the
                // new rect and the existing clip stack (computed at record time
                // in Graphics::clipToRectangle): the emitter's scissor bounds.
                emitter_.pushClipRect(cmd.clipBounds);
                break;

            case DrawOp::PopClipRect:
                emitter_.popClipRect();
                break;

            case DrawOp::PushClipPath:
                // Analytical-SDF clip push — the fragment shader discards outside
                // the clip shape, stencil INCR_WRAP where inside at stencil ==
                // parentDepth. An image-alpha clip (shape 3) runs the mask program.
                clipShape(cmd, true);
                emitter_.pushStencilDepth();
                break;

            case DrawOp::PopClipPath:
                // Analytical-SDF clip pop — DECR_WRAP at stencil == currentDepth
                // (before the decrement). cmd.stencilDepth here is the depth one
                // level DEEPER than parent (i.e. the push's target depth), so
                // stencil == that value is exactly what the INCR produced.
                clipShape(cmd, false);
                emitter_.popStencilDepth();
                break;

            case DrawOp::ExcludeClipRect:
            case DrawOp::RestoreClipExclude: {
                // excludeClipRectangle: stencil-park the rect's drawable pixels
                // out of the current depth. REPLACE with the low six bits masked
                // out of the COMPARE. Park: reference = 192 + depth matches pixels
                // whose (value & 0x3F) == depth — exactly the pixels drawable at
                // that depth — and REPLACE writes the full 192 + depth, making
                // them fail the draws' EQUAL(depth) test. Un-park: reference =
                // depth matches the same low bits, REPLACE writes depth back.
                // Nested clips can't touch parked pixels (their EQUAL refs never
                // match 192 + d), so exclusions survive any push/pop ordering.
                // Requires clip depth < 64 — depths that deep are pathological.
                const auto& p = arena.read<ExcludeRectParams>(cmd.dataOffset);
                const auto depth = static_cast<uint8_t>(cmd.stencilDepth);
                const auto ref = static_cast<uint8_t>(cmd.op == DrawOp::ExcludeClipRect ? 192u + depth : depth);
                emitter_.invalidate();
                emitter_.setPipeline(clipPipeline(programs.clip, gpu::StencilOp::Replace, 0x3F));
                emitter_.setTextureCount(0);
                emitter_.setStorage({ segments });
                ClipConstants k;
                k.viewportW = fw; k.viewportH = fh;
                k.shapeType    = 1;   // rect = rrect with radius 0
                k.centerX      = p.rect.getCentreX();
                k.centerY      = p.rect.getCentreY();
                k.halfW        = p.rect.getWidth()  * 0.5f;
                k.halfH        = p.rect.getHeight() * 0.5f;
                emitter_.setConstants(builder_.pushConstants(k));
                coverQuad(cmd, p.rect, ref);
                emitter_.invalidate();
                if (cmd.op == DrawOp::ExcludeClipRect) emitter_.pushExclusion();
                else                                   emitter_.popExclusion();
                break;
            }

            case DrawOp::FillPath: {
                // Analytical SDF path fill — tile quads into the scene pass.
                const auto* v = paintPipeline(Paint::Path, cmd.target, emitter_.stencilTested(), live);
                if (v == nullptr) break;
                const auto& p = arena.read<FillPathParams>(cmd.dataOffset);
                const auto& fill = recorder_.getFill(p.fillIndex);
                // The per-tile UIVertex span sits right after the params in the
                // arena, 4-byte aligned (Graphics::fillPath pushed it).
                const uint32_t vertsOff =
                    (cmd.dataOffset + static_cast<uint32_t>(sizeof(FillPathParams)) + 3u) & ~3u;
                const auto verts = arena.readSpan<UIVertex>(vertsOff, p.vertexCount);
                // Colour source — a tiled image fill's texture, the gradient
                // atlas for gradient fills, the default for solids.
                // path_sdf.frag samples it iff gradientInfo.z > 0.
                const auto& fillColour = recorder_.getFillColour(p.fillIndex);
                const gpu::TextureId colour =
                    fillColour.texture ? fillColour.texture
                    : (fill.isGradient() && fill.gradient) ? gradientAtlas_.texture()
                    : ctx.defaultTexture();
                emitter_.invalidate();
                emitter_.setPipeline(v->id);
                emitter_.setResources(bind(colour), defaultBinding);
                emitter_.setTextureCount(1);
                emitter_.setStorage({ segments, frameData });
                PathConstants k;
                k.viewportW    = fw;
                k.viewportH    = fh;
                k.segmentStart = p.segmentStart;
                k.fillRule     = p.fillRule;
                k.tileW        = p.tileSize;
                k.paintsMain   = v->paintsMain ? 1 : 0;
                emitter_.setConstants(builder_.pushConstants(k));
                emitter_.draw(cmd, verts.data(), p.vertexCount);
                emitter_.invalidate();
                break;
            }

            case DrawOp::DrawShader: {
                // Inside the scene pass when it can be (DrawShaderParams::inScene):
                // one region triangle against the current scene attachments, no
                // pass boundary. Otherwise a pass of its own attached to what it
                // writes: targets alone, the main colour with targets, or the
                // main colour alone — a scene reader (it REPLACES its region,
                // ping-ponged like EffectHSV) or a target reader (blended Over
                // the current half in place). One whose writes nothing reads is
                // skipped.
                const auto& sp = arena.read<DrawShaderParams>(cmd.dataOffset);
                if (sp.shaderIndex == ~0u || sp.shaderIndex >= shaderCaptures_.size()) break;
                const auto& sc = shaderCaptures_[sp.shaderIndex];
                if (sp.inScene) {
                    if (sp.writesMain || (sp.targetsWritten & live) != 0)
                    {
                        const auto pipeline = shaderPipeline(sc, sceneFormats(),
                            [&](uint32_t l) { return l >= 1 && l - 1 < targets_.size() ? static_cast<int>(l - 1) : -1; },
                            emitter_.stencilTested());
                        shaderDraw(sc, sp, pipeline, sp.region, fe, cmd.clipBounds,
                                   emitter_.stencilDepth(), {});
                        emitter_.invalidate();
                    }
                }
                else if (!sp.writesMain) {
                    if ((sp.targetsWritten & recorder_.liveTargets(k + 1)) != 0)
                        targetPass(cmd, sp, sc);
                }
                else if (sp.targetsWritten != 0) {
                    mainPass(cmd, sp, sc);
                }
                else {
                    shaderMainOnly(cmd, sp, sc);
                }
                break;
            }

            case DrawOp::EffectBlend:
                if (setPaint(Paint::Blend, cmd)) lowerBlend(*this, emitter_, cmd);
                break;

            case DrawOp::EffectNoise:
                if (setPaint(Paint::Noise, cmd)) lowerNoise(*this, emitter_, cmd);
                break;

            case DrawOp::FillRect: case DrawOp::FillRectList: case DrawOp::FillRoundedRect:
            case DrawOp::FillEllipse: case DrawOp::StrokeRoundedRect: case DrawOp::StrokeEllipse:
            case DrawOp::DrawImage: case DrawOp::DrawGlyphs: case DrawOp::DrawLine:
            case DrawOp::FillTiledImage:
                if (setPaint(Paint::Color, cmd)) lowerColor(*this, emitter_, cmd);
                break;

            case DrawOp::COUNT:
                break;
        }
    }

    endScene();

    // The frame's output: a copy of whichever half holds the composited scene
    // (full-window single-pass effects leave it flipped; everything else
    // returns to the starting half). A draw, not a blit: the output's format
    // needn't match the scene's, and some APIs can't blit at all. It covers
    // every pixel, so what the output held before is never loaded.
    fullscreenPass("present copy", programs.copy, info.output, info.outputFormat, StencilTest::None,
                   { bind(halves_[cur]) }, {}, effectConstants(info.extent, 0, 0, 0),
                   { 0, 0, static_cast<int32_t>(info.extent.width), static_cast<int32_t>(info.extent.height) }, 0,
                   LoadOp::DontCare);
}

} // namespace jvk
