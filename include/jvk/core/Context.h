/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Context.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// Context — what every Renderer on one backend shares
//
// One per gpu::Backend, made on first ask (forBackend) and kept while any
// window or offscreen renderer on that backend holds it:
//
//   - the built-in programs, created from the shader bundle in the backend's
//     shader format, and the interned pipelines (PipelineDesc → PipelineId);
//   - the default 1x1 textures every unbound slot samples;
//   - the image cache (every window's uploads of CPU images);
//   - LIFETIMES. Each frame a Renderer records takes the next RECORD SERIAL; a
//     resource used by a frame is stamped with it. retire() destroys a
//     resource once every frame that could name it has been submitted (no
//     Renderer has an unsubmitted frame with a serial at or below its stamp);
//     the backend then defers the native object until the GPU is done. No
//     pins and no waits, on any thread.
// =============================================================================

struct BuiltinPrograms
{
    gpu::ProgramId ui2d, noise, path, clip, clipMask, blur, copy, hsv, shapeBlur, pathBlur;
};

class Context : public std::enable_shared_from_this<Context>
{
public:
    static std::shared_ptr<Context> forBackend(const std::shared_ptr<gpu::Backend>& backend);

    explicit Context(std::shared_ptr<gpu::Backend> backend);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    gpu::Backend&       backend()       { return *backend_; }
    const gpu::Caps&    caps() const    { return backend_->caps(); }
    std::shared_ptr<gpu::Backend> sharedBackend() const { return backend_; }

    const BuiltinPrograms& programs() const { return programs_; }

    // A program from bundle-shaped stages (Bundle.h), in the backend's format.
    gpu::ProgramId createProgram(const shaders::Program& program);

    // The pipeline for `desc`, created on first ask. Any thread.
    gpu::PipelineId pipeline(const gpu::PipelineDesc& desc);

    // 1x1 opaque black, what an unbound texture slot samples; its 2-layer
    // twin, for a slot a program declares as an array.
    gpu::TextureId defaultTexture() const      { return defaultTexture_; }
    gpu::TextureId defaultArrayTexture() const { return defaultArrayTexture_; }

    ImageCache& images() { return *images_; }

    // Seconds since the Context was made: the `time` every jvk::Shader sees,
    // snapshotted once per frame.
    float time() const
    {
        return static_cast<float>((juce::Time::getMillisecondCounterHiRes() - startMs_) * 0.001);
    }

    // ---- Lifetimes ---------------------------------------------------------

    // A Renderer's slot in the serial bookkeeping: `pending` is the serial of
    // its recorded-but-unsubmitted frame, 0 when it has none.
    void addRenderer(std::atomic<uint64_t>& pending);
    void removeRenderer(std::atomic<uint64_t>& pending);

    // Starts a frame's recording (message thread): returns its serial, marks
    // it pending, and destroys whatever has become safe.
    uint64_t beginRecord(std::atomic<uint64_t>& pending);
    // The frame was submitted, or will never be: it names nothing any more.
    static void endFrame(std::atomic<uint64_t>& pending) { pending.store(0, std::memory_order_release); }

    uint64_t serial() const { return serial_.load(std::memory_order_acquire); }

    // Destroys `texture` / `program` once no unsubmitted frame stamped at or
    // after `lastUsed` exists. Any thread.
    void retire(gpu::TextureId texture, uint64_t lastUsed);
    void retire(gpu::ProgramId program, uint64_t lastUsed);
    // The same for compute's buffers and kernels (jvk::compute), destroyed at
    // once when no such frame exists: a Context nothing renders with (a
    // compute-only user) never collects.
    void retire(gpu::BufferId buffer, uint64_t lastUsed);
    void retire(gpu::KernelId kernel, uint64_t lastUsed);

    // Destroys everything retired that is now safe. Any thread.
    void collect();

private:
    uint64_t safeSerial() const;   // under lock_

    std::shared_ptr<gpu::Backend> backend_;
    BuiltinPrograms               programs_;
    gpu::TextureId                defaultTexture_, defaultArrayTexture_;
    std::unique_ptr<ImageCache>   images_;
    double                        startMs_ = juce::Time::getMillisecondCounterHiRes();

    std::mutex pipelineLock_;
    std::unordered_map<gpu::PipelineDesc, gpu::PipelineId, gpu::PipelineDescHash> pipelines_;

    enum class Kind : uint8_t { Texture, Program, Buffer, Kernel };
    struct Retired { uint32_t handle; uint64_t lastUsed; Kind kind; };
    void destroy(const Retired& r);
    void retireOrDestroy(const Retired& r);

    mutable std::mutex lock_;   // the serial bookkeeping and the retire lists
    std::atomic<uint64_t> serial_ { 0 };
    std::vector<std::atomic<uint64_t>*> renderers_;
    std::vector<Retired> retired_;
    std::mutex collectLock_;    // one collect at a time: collectScratch_ (taken before lock_)
    std::vector<Retired> collectScratch_;
};

} // namespace jvk
