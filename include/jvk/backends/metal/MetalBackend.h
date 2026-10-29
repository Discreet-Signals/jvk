/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MetalBackend.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

// Objective-C++ only (jvk.mm includes the backend on macOS); manual reference
// counting, as the rest of the module.

namespace jvk::metal
{

// =============================================================================
// metal::Backend — gpu::Backend on Metal (macOS)
//
// One per process (acquire()), shared by every editor and offscreen surface.
// It draws the core's frame packets straight into Metal command buffers: no
// recorded command list to translate at submit, no descriptor sets, no
// pipeline-cache file to load at start-up.
//
// Lifetimes: a command buffer retains every resource it encodes, so freeing a
// texture, program or pipeline the moment the core does is always safe; the
// object lives on until the GPU has finished every frame that used it.
//
// Memory: on unified memory (Apple silicon) what the CPU fills is a SHARED
// texture: writeTexture writes into it in place when no frame on the GPU can be
// reading it, and readTexture reads it in place, with no staging buffer and no
// copy pass. Otherwise the write is staged and copied at the start of the next
// frame submitted, which is the ordering FramePacket promises. Render targets
// stay private (the GPU compresses them). Without unified memory every write is
// staged.
//
// Locks, in this order: submitMutex_ (a frame's encode and commit, and a direct
// write's check-and-write), tableMutex_ (handle tables, the pipeline intern
// map), writeMutex_ (staged writes). Resolving a handle takes none.
// =============================================================================

// Where slots live (TRANSLATION_LAYER.md, "Binding convention"): the bundle's
// texture i at [[texture(i)]] with its sampler at [[sampler(8 + i)]], storage j
// at [[buffer(16 + j)]] (kBlockBase), the constants at [[buffer(24)]] on both
// stages (kConstantsIndex), vertices at [[buffer(30)]]. A jvk::Shader's where
// jvk_msl put them (MslBindings.h).
inline constexpr uint8_t kSamplerBase = 8, kVertexBufferIndex = 30;

// A strong reference to an Objective-C object under manual reference counting.
template <typename T>
class Strong
{
public:
    Strong() = default;
    static Strong adopt(T object)  { Strong s; s.object_ = object; return s; }   // a +1 already (new..., alloc, Create)
    static Strong retain(T object) { Strong s; s.object_ = object != nil ? (T) [object retain] : nil; return s; }

    Strong(const Strong& o) : object_(o.object_ != nil ? (T) [o.object_ retain] : nil) {}
    Strong(Strong&& o) noexcept : object_(o.object_) { o.object_ = nil; }
    Strong& operator=(Strong o) noexcept { std::swap(object_, o.object_); return *this; }
    ~Strong() { if (object_ != nil) [object_ release]; }

    T get() const { return object_; }
    explicit operator bool() const { return object_ != nil; }
    bool operator==(const Strong& o) const { return object_ == o.object_; }

private:
    T object_ = nil;
};

struct TextureSlot
{
    Strong<id<MTLTexture>> texture;
    PixelFormat      pixelFormat = PixelFormat::RGBA8;
    gpu::TextureKind kind = gpu::TextureKind::Color;
    uint32_t         width = 0, height = 0, layers = 1, mips = 1;
    bool             shared = false;         // CPU-visible: written and read in place
    bool             renderTarget = false;
    uint64_t         lastUse = 0;            // the serial of the last frame encoded with it (under submitMutex_)
    uint32_t         pendingWrites = 0;      // staged writes not yet encoded (under writeMutex_)
};

struct ProgramSlot
{
    Strong<id<MTLFunction>> vertex, fragment;
    std::string name;
    uint32_t    constantsBytes = 0;
    // Where slot i lives in Metal: the bundle's portable layout or a jvk::Shader's own.
    uint8_t textureIndex[gpu::kMaxDrawTextures] {};
    uint8_t samplerIndex[gpu::kMaxDrawTextures] {};
    uint8_t storageIndex[gpu::kMaxDrawStorage] {};
    std::vector<uint32_t> pipelines;   // built from it (under tableMutex_)
};

struct PipelineSlot
{
    gpu::PipelineDesc desc;
    Strong<id<MTLRenderPipelineState>> state;          // null: the build failed; its draws are skipped
    Strong<id<MTLDepthStencilState>>   depthStencil;   // null: the pipeline has no depth/stencil attachment
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    uint32_t         program = 0;
};

class Backend final : public gpu::Backend, public std::enable_shared_from_this<Backend>
{
public:
    // The process's backend, made on first ask; a lost one is replaced on the next.
    static std::shared_ptr<Backend> acquire();
    ~Backend() override;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    bool                usable() const noexcept            { return device_ != nil && queue_ != nil; }
    const juce::String& unavailableReason() const noexcept { return reason_; }
    id<MTLDevice>       device() const noexcept            { return device_; }
    id<MTLCommandQueue> queue() const noexcept             { return queue_; }

    // ---- gpu::Backend ---------------------------------------------------------
    const gpu::Caps& caps() const override { return caps_; }
    bool isLost() const override { return lost_.load(std::memory_order_acquire); }
    void waitIdle() override;

    gpu::TextureId createTexture(const gpu::TextureDesc& desc) override;
    void           destroyTexture(gpu::TextureId texture) override;
    bool           canRenderTo(PixelFormat format) const override;
    bool writeTexture(gpu::TextureId texture, uint32_t layer, uint32_t mip,
                      gpu::IRect region, std::span<const std::byte> data) override;
    void clearTexture(gpu::TextureId texture, gpu::ClearColor colour) override;
    bool readTexture(gpu::TextureId texture, gpu::IRect region, std::span<std::byte> dst) override;

    gpu::ProgramId  createProgram(const gpu::ProgramDesc& desc) override;
    void            destroyProgram(gpu::ProgramId program) override;
    gpu::PipelineId createPipeline(const gpu::PipelineDesc& desc) override;

    std::unique_ptr<gpu::Surface> createOffscreenSurface(gpu::TextureId target) override;

    // ---- For the surfaces -----------------------------------------------------
    TextureSlot*  texture(gpu::TextureId t)   const { return textures_.resolve(t.value); }
    ProgramSlot*  program(uint32_t handle)    const { return programs_.resolve(handle); }
    PipelineSlot* pipeline(gpu::PipelineId p) const { return pipelines_.resolve(p.value); }
    TextureSlot*  dummyTexture() const              { return texture(dummy_); }
    id<MTLSamplerState> sampler(gpu::SamplerDesc d) const
    {
        return samplers_[static_cast<size_t>(d.filter) * 3 + static_cast<size_t>(d.address)].get();
    }

    // A surface's output texture, re-pointed at each frame's drawable (worker only).
    gpu::TextureId createSwapTexture();
    void           setSwapTexture(gpu::TextureId swap, id<MTLTexture> texture);

    // ---- Submission (under submitMutex()) -------------------------------------
    std::mutex& submitMutex() noexcept { return submitMutex_; }
    // The serial of the frame being encoded now: every texture it touches is
    // stamped with it.
    uint64_t beginSubmission() noexcept { return ++submitted_; }
    // The staged writes and clears, encoded at the start of `cb` (submission `serial`).
    void     encodePendingWrites(id<MTLCommandBuffer> cb, uint64_t serial);
    // Commits `cb` as submission `serial`, then runs `done` (any thread) once
    // the GPU has finished it.
    void     commit(id<MTLCommandBuffer> cb, uint64_t serial, std::function<void()> done = {});

private:
    Backend();
    void initCaps();
    uint64_t completed() const noexcept { return completed_.load(std::memory_order_acquire); }
    void     flushAndWait();   // commits the staged writes, then waits for every submission
    id<MTLDepthStencilState> depthStencilState(const gpu::StencilState& stencil);   // under tableMutex_
    bool     stage(TextureSlot& t, uint32_t handle, uint32_t layer, uint32_t mip, gpu::IRect region,
                   const std::byte* data, size_t rowBytes, size_t imageBytes);      // under writeMutex_
    Strong<id<MTLFunction>> function(const gpu::ProgramDesc& desc, bool vertex, juce::String& error);

    id<MTLDevice>       device_ = nil;   // owned (+1)
    id<MTLCommandQueue> queue_  = nil;   // owned (+1)
    juce::String        reason_;
    gpu::Caps           caps_;
    bool                unified_ = false;
    std::atomic<bool>   lost_ { false };

    backends::SlotTable<TextureSlot>  textures_;
    backends::SlotTable<ProgramSlot>  programs_;
    backends::SlotTable<PipelineSlot> pipelines_;
    std::mutex tableMutex_;
    std::unordered_map<gpu::PipelineDesc, uint32_t, gpu::PipelineDescHash> pipelineIds_;
    std::vector<std::pair<gpu::StencilState, Strong<id<MTLDepthStencilState>>>> depthStencilStates_;   // under tableMutex_
    // Compiled functions by source (the bundle's stages are shared between programs).
    struct Function { Strong<id<MTLLibrary>> library; Strong<id<MTLFunction>> function; };
    std::unordered_map<std::string, Function> functions_;                                           // under tableMutex_

    Strong<id<MTLSamplerState>> samplers_[6];   // Filter x AddressMode
    gpu::TextureId dummy_;

    // ---- Submissions ------------------------------------------------------------
    std::mutex              submitMutex_;
    std::atomic<uint64_t>   submitted_ { 0 }, completed_ { 0 };
    std::mutex              idleMutex_;
    std::condition_variable idle_;

    // ---- Staged writes ------------------------------------------------------------
    struct PendingWrite
    {
        uint32_t                texture = 0;
        uint32_t                layer = 0, mip = 0;
        gpu::IRect              region;
        Strong<id<MTLBuffer>>   buffer;
        size_t                  offset = 0, rowBytes = 0, imageBytes = 0;
        bool                    clear = false;
        gpu::ClearColor         colour;
    };
    // Staging memory: the chunk being written, the ones written (freeAfter: the
    // submission that read them, 0 until one is encoded), the free ones.
    struct StagingChunk { Strong<id<MTLBuffer>> buffer; size_t head = 0; uint64_t freeAfter = 0; };
    static constexpr size_t kStagingChunk = 4u << 20;
    std::mutex                  writeMutex_;
    std::vector<PendingWrite>   pending_;
    StagingChunk                filling_;
    std::vector<StagingChunk>   waiting_, free_;

    // Counters for JVK_METAL_STATS.
    std::atomic<uint64_t> directBytes_ { 0 }, stagedBytes_ { 0 };
    bool stats_ = false;
};

// =============================================================================
// Surfaces and the packet executor
// =============================================================================

// A frame slot's copies of the packet's vertex and storage streams.
struct FrameSlot
{
    Strong<id<MTLBuffer>> vertices, storage;
};

class PacketExecutor
{
public:
    explicit PacketExecutor(Backend& backend) : backend_(backend) {}
    // Encodes the packet's passes into `cb` (under the submit lock).
    void encode(id<MTLCommandBuffer> cb, FrameSlot& slot, const gpu::FramePacket& packet, uint64_t serial);

private:
    struct Bound
    {
        PipelineSlot*            pipeline = nullptr;
        id<MTLDepthStencilState> depthStencil = nil;
        int                      stencilRef = -1;
        gpu::IRect               scissor { -1, -1, -1, -1 };
        id<MTLTexture>           textures[32] {};
        id<MTLSamplerState>      samplers[16] {};
        id<MTLBuffer>            buffers[32] {};
        size_t                   offsets[32] {};
        gpu::Range               constants { ~0u, 0 };
        uint32_t                 constantsLength = 0;
        bool                     vertexBuffer = false;
    };

    bool ensure(Strong<id<MTLBuffer>>& buffer, size_t size);
    void encodePass(id<MTLCommandBuffer> cb, FrameSlot& slot, const gpu::FramePacket& packet,
                    const gpu::Pass& pass, uint64_t serial);
    void encodeDraw(id<MTLRenderCommandEncoder> enc, FrameSlot& slot, const gpu::FramePacket& packet,
                    const gpu::Draw& draw, const gpu::IRect& area, uint64_t serial);

    Backend& backend_;
    Bound    bound_;
};

// ---- A window's CAMetalLayer ----------------------------------------------------
class LayerSurface final : public gpu::Surface
{
public:
    LayerSurface(std::shared_ptr<Backend> backend, CAMetalLayer* layer, gpu::Extent sizeHint);
    ~LayerSurface() override;

    bool        beginFrame(gpu::FrameInfo& info) override;
    void        submit(const gpu::FramePacket& packet) override;
    gpu::Extent extent() const override { return { width_, height_ }; }
    gpu::Extent preferredExtent() const override;
    bool        canBegin() const override { return layer_ != nil && !backend_->isLost(); }
    void        resize(gpu::Extent extent) override;

private:
    static constexpr uint32_t kFrames = 2;   // in flight, as the Vulkan backend

    std::shared_ptr<Backend>    backend_;
    CAMetalLayer*               layer_ = nil;    // retained
    dispatch_semaphore_t        inFlight_ = nullptr;
    Strong<id<CAMetalDrawable>> drawable_;
    gpu::TextureId              swap_;
    FrameSlot                   slots_[kFrames];
    PacketExecutor              exec_;
    uint32_t width_ = 0, height_ = 0;
    uint32_t current_ = 0;
    bool     frameBegun_ = false;
};

// ---- An offscreen texture ---------------------------------------------------------
class OffscreenSurface final : public gpu::Surface
{
public:
    OffscreenSurface(std::shared_ptr<Backend> backend, gpu::TextureId target);
    ~OffscreenSurface() override;

    bool        beginFrame(gpu::FrameInfo& info) override;
    void        submit(const gpu::FramePacket& packet) override;
    gpu::Extent extent() const override;

private:
    std::shared_ptr<Backend> backend_;
    gpu::TextureId           target_;
    dispatch_semaphore_t     inFlight_ = nullptr;
    FrameSlot                slot_;
    PacketExecutor           exec_;
    bool                     frameBegun_ = false;
};

// ---- The window host ----------------------------------------------------------------
class WindowHost final : public jvk::WindowHost
{
public:
    WindowHost() = default;
    ~WindowHost() override { detach(); }

    std::shared_ptr<gpu::Backend> backend() override;
    juce::String reason() const override { return reason_; }

    bool attach(juce::Component& editor, gpu::Extent sizeHint, juce::String& failure) override;
    void detach() override;

    gpu::Surface* surface() override { return surface_.get(); }
    void setBounds(juce::Rectangle<int> localBounds) override { view_.setBounds(localBounds); }

private:
    std::shared_ptr<Backend>      backend_;
    std::unique_ptr<LayerSurface> surface_;
    juce::String                  reason_;
    std::unique_ptr<jvk::core::macos::NSViewGenerator> nsView_;
    juce::NSViewComponent         view_;
};

std::unique_ptr<jvk::WindowHost> createWindowHost();

} // namespace jvk::metal
