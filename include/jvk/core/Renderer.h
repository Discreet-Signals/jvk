/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Renderer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

class Shader;

// =============================================================================
// ShaderCapture — what a DrawShader draw needs of its jvk::Shader
//
// Captured at record (Shader::capture), so lowering never touches the Shader,
// which its owner may destroy at any time; its program and textures outlive
// the frame through the Context's retire list.
// =============================================================================

struct ShaderCapture
{
    gpu::ProgramId      program;
    uint8_t             outputs = 1;                 // bit L: it writes output location L
    bool                readsScene = false;
    int8_t              sceneSlot = -1;              // the texture slot the scene goes in
    uint8_t             textureCount = 0;
    gpu::TextureBinding textures[gpu::kMaxDrawTextures] {};
    int8_t              slotTarget[gpu::kMaxDrawTextures] {};   // slot → the target it reads (-1: none)
    uint8_t             storageCount = 0;
    gpu::Range          storage[gpu::kMaxDrawStorage] {};      // in Renderer::shaderData()
};

// =============================================================================
// Renderer — one surface's frames
//
// The core's per-window (or per-offscreen-target) engine. Owns the frame's
// textures (the main colour's two halves, the stencil, the render targets and
// their scratches), the record half (Recorder), and a worker thread that
// lowers each recorded frame into a gpu::FramePacket and submits it:
//
//   message thread                worker
//   beginRecord → paint (Graphics) → submit() ─► surface.beginFrame
//                                                lower (the render graph)
//                                                surface.submit(packet)
//
// Strict one frame in flight: record only while !isBusy(). The busy flag's
// release/acquire pair publishes every record-time write to the worker.
// =============================================================================

class Renderer {
public:
    Renderer(std::shared_ptr<Context> context, gpu::Surface& surface, std::vector<Target> targets = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // ---- The frame (message thread) ----------------------------------------

    // Starts recording a frame at the surface's current size: resets the
    // recorder, takes the frame's record serial, (re)makes the frame's
    // textures if the size changed. False when there is nothing to draw into
    // (zero size, textures that can't be made): record nothing.
    bool beginRecord();
    // The recorded frame will never be submitted (a mode switch mid-paint).
    void abandonRecord();
    // Posts the recorded frame to the worker. Non-blocking. Preconditions: the
    // caller just recorded it, and isBusy() == false.
    void submit();

    // True while the worker is mid-frame. The caller must not record while
    // this is true.
    bool isBusy() const { return workerBusy_.load(std::memory_order_acquire); }
    // Block until the worker is idle (resize, teardown).
    void waitForIdle();
    // Called on the worker each time it finishes a frame (isBusy() is false by
    // then): set once before the first submit.
    std::function<void()> onIdle;

    // Number of live Renderers in the process: a cheap probe for "is anything
    // rendering through jvk right now".
    static int liveCount();

    // ---- Record API (jvk::Graphics, message thread) -------------------------

    Recorder&       recorder()       { return recorder_; }
    const Recorder& recorder() const { return recorder_; }

    static constexpr uint8_t kNoTarget = Recorder::kNoTarget;

    template <typename Params>
    bool push(DrawOp op, const juce::Rectangle<int>& clip, uint8_t stencilDepth, const Params& params)
    {
        return recorder_.push(op, clip, stencilDepth, params);
    }
    void setRecordTarget(uint8_t attachment) { recorder_.setRecordTarget(attachment); }
    void noteTargetsRead(TargetMask targets) { recorder_.noteTargetsRead(targets); }
    uint32_t captureFont(const juce::Font& f) { return recorder_.captureFont(f); }
    uint32_t captureFill(const juce::FillType& f, const Recorder::FillColour& c) { return recorder_.captureFill(f, c); }
    void arena_align(uint32_t alignment) { recorder_.arena_align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { recorder_.arena_pushSpan(data); }
    uint32_t pushSegments(const void* data, uint32_t count) { return recorder_.pushSegments(data, count); }

    // The gradient's row in this renderer's atlas (GradientAtlas::getRow).
    float registerGradient(const juce::ColourGradient& g);

    // The texture for a CPU image: a jvk image's own, else the shared image
    // cache's (uploaded on a miss). Null when it can't be made.
    gpu::TextureId texture(const juce::Image& img);
    gpu::TextureId texture(const ImageSource& src);

    GlyphAtlas& glyphAtlas() { return glyphAtlas_; }

    // The context's shader cache: the Shader for `spirv` and its `msl` (keyed
    // by the SPIR-V's address: a program compiled into the binary), built on
    // first ask and kept for this Renderer's life. Shared by every caller, so
    // what differs per caller goes per draw (Shader::setDrawConstants, setDrawImage).
    Shader& shader(std::span<const uint32_t> spirv, std::string_view msl = {});

    // A DrawShader draw's capture (DrawShaderParams::shaderIndex), and the
    // uniform bytes its blocks point at.
    uint32_t captureShader(const ShaderCapture& c)
    {
        shaderCaptures_.push_back(c);
        return static_cast<uint32_t>(shaderCaptures_.size() - 1);
    }
    gpu::Range pushShaderData(std::span<const std::byte> bytes)
    {
        const auto offset = static_cast<uint32_t>(shaderData_.size());
        shaderData_.insert(shaderData_.end(), bytes.begin(), bytes.end());
        return { offset, static_cast<uint32_t>(bytes.size()) };
    }

    // ---- Frame and targets -------------------------------------------------

    gpu::Extent extent() const { return extent_; }   // the frame being recorded
    uint64_t    serial() const { return serial_; }   // its record serial (Context)

    // The targets this frame carries (the ones declared that the backend can
    // render: validTargets), and the names it dropped.
    const std::vector<Target>& targets() const { return targets_; }
    const std::vector<juce::Identifier>& droppedTargets() const { return droppedTargets_; }
    int targetIndex(const juce::Identifier& name) const
    {
        for (size_t i = 0; i < targets_.size(); ++i)
            if (targets_[i].name == name) return static_cast<int>(i);
        return -1;
    }
    // The scaled targets (Target::scale, Target::height), bit i = target i: not scene-pass attachments.
    TargetMask scaledTargets() const;
    // Target i's size relative to the frame's, per side: its scale, or its fixed height over the frame's.
    float targetScale(size_t i) const;
    // Target i's size in pixels: the frame's, times its scale, rounded up (a fixed height exactly).
    gpu::Extent targetExtent(size_t i) const;

    // ---- Lowering helpers (core/ops, worker) --------------------------------

    const Arena&          arena() const { return recorder_.arena(); }
    const juce::Font&     getFont(uint32_t i) const { return recorder_.getFont(i); }
    const juce::FillType& getFill(uint32_t i) const { return recorder_.getFill(i); }
    const Recorder::FillColour& getFillColour(uint32_t i) const { return recorder_.getFillColour(i); }
    gpu::TextureId gradientTexture() const { return gradientAtlas_.texture(); }
    gpu::TextureId defaultTexture() const  { return context_->defaultTexture(); }
    const ShaderCapture& shaderCapture(uint32_t i) const { return shaderCaptures_[i]; }
    std::span<const std::byte> shaderData() const { return shaderData_; }

    Context&       context()       { return *context_; }
    gpu::Backend&  backend()       { return context_->backend(); }
    gpu::Surface&  surface()       { return surface_; }

private:
    // ---- Worker ----
    void runFrame();
    void lower(const gpu::FrameInfo& info);

    // ---- Frame textures ----
    bool ensureFrameTextures(gpu::Extent extent);
    void releaseFrameTextures();
    gpu::TextureId targetScratch(size_t i);
    void validateTargets(std::vector<Target> requested);

    // ---- Pipelines ----
    enum class Paint : uint8_t { Color, Blend, Noise, Path };
    struct PaintVariant { Paint kind; uint8_t attachment; bool clip; TargetMask clears; gpu::PipelineId id; bool paintsMain; };
    // The pipeline of a built-in scene draw into `attachment`, the stencil
    // variant when `clip`, clearing the live targets its paint clears.
    const PaintVariant* paintPipeline(Paint kind, uint8_t attachment, bool clip, TargetMask live);
    gpu::AttachmentFormats sceneFormats() const;

    std::shared_ptr<Context> context_;
    gpu::Surface&            surface_;

    Recorder      recorder_;
    FrameBuilder  builder_;
    Emitter       emitter_;
    GradientAtlas gradientAtlas_;
    GlyphAtlas    glyphAtlas_;

    std::vector<Target>           targets_;
    std::vector<juce::Identifier> droppedTargets_;

    // The frame's textures, at extent_.
    gpu::Extent    textureExtent_;
    gpu::TextureId halves_[2], stencil_;
    std::vector<gpu::TextureId> targetTextures_, targetScratches_;

    gpu::Extent extent_;
    uint64_t    serial_ = 0;
    uint64_t    frameNumber_ = 0;
    std::atomic<uint64_t> pending_ { 0 };   // Context: the unsubmitted frame's serial

    std::vector<PaintVariant> paintVariants_;   // worker only

    std::map<const uint32_t*, std::unique_ptr<Shader>> shaders_;   // shader()
    std::vector<ShaderCapture> shaderCaptures_;
    std::vector<std::byte>     shaderData_;

    std::atomic<bool> workerBusy_ { false };

    class Worker : public juce::Thread
    {
    public:
        explicit Worker(Renderer& r) : juce::Thread("jvk-render-worker"), owner(r) {}
        ~Worker() override { stopThread(2000); }

        void run() override
        {
            while (! threadShouldExit())
            {
                wait(-1);
                if (threadShouldExit()) break;
                owner.runFrame();
                owner.workerBusy_.store(false, std::memory_order_release);
                if (owner.onIdle) owner.onIdle();
            }
        }

    private:
        Renderer& owner;
    };

    // Declared last so destruction stops + joins the worker BEFORE any other
    // member is torn down.
    std::unique_ptr<Worker> worker_;
};

} // namespace jvk
