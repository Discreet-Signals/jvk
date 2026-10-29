/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Backend.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::gpu
{

// =============================================================================
// Backend — the GPU behind the core
//
// Deliberately narrow: a handful of calls per frame, shaped around what jvk
// draws rather than a general GPU abstraction, so a new backend (Metal,
// WebGPU, eacp) is a few thousand lines, not a port of an engine.
//
// Two objects, mirroring the two lifetimes jvk has:
//
//   Backend   one per process per GPU, shared by every editor and offscreen
//             renderer. Owns all GPU memory, every texture, program and
//             pipeline, and their deferred destruction.
//   Surface   one per thing a frame renders into: a window's swapchain or an
//             offscreen texture. Begins, executes and presents frames.
//
// Window surfaces are made by backend-specific host code (it needs the
// platform view and the backend's own types), so Backend only creates
// offscreen ones.
//
// Threading: every Backend call is safe from any thread, and may overlap a
// Surface's beginFrame/submit on another thread. beginFrame/submit for one
// Surface come from one thread at a time.
// =============================================================================

// Which blob of the shader bundle a backend consumes (TRANSLATION_LAYER.md,
// "Shaders"). SPIR-V is the bundle's canonical form; the others are
// translated from it ahead of time.
enum class ShaderFormat : uint8_t { SPIRV, MSL, HLSL, WGSL };

struct Caps
{
    ShaderFormat shaderFormat = ShaderFormat::SPIRV;

    uint32_t maxTextureSize          = 8192;
    uint32_t maxColorAttachments     = kMaxColorAttachments;
    // Total bytes per pixel across a pass's colour attachments. WebGPU
    // guarantees only 32 (RGBA8 main colour + 3 RGBA16F targets = 28); render
    // targets beyond it must be refused or split by the core.
    uint32_t maxColorBytesPerSample  = 32;
    uint32_t maxDrawConstantsBytes   = 128;   // Vulkan's guaranteed push-constant size
    uint32_t constantsAlignment      = 4;     // WebGPU: 256 (dynamic uniform offsets)
    uint32_t storageAlignment        = 256;   // a storage range's offset is a multiple of this

    bool independentBlend     = true;         // per-attachment blend state (render targets need it)
    bool textureCompressionBC = false;

    // Whether createProgram takes SPIR-V with native bindings (a jvk::Shader's
    // program), natively or through a run-time translator.
    bool spirvPrograms = true;
};

// =============================================================================
// Programs
//
// What a program binds, by slot: textures 0..textures-1 (each with its own
// sampler), storage buffers 0..storageBuffers-1, one block of per-draw
// constants shared by both stages. The bundle's programs bind the portable
// way (TRANSLATION_LAYER.md, "Binding convention"); a program from elsewhere (a
// user's SPIR-V) can name its native bindings instead, slot by slot.
// =============================================================================

enum class BindingKind : uint8_t
{
    Texture,                 // a sampled texture; its sampler is a separate Sampler binding
    Sampler,
    CombinedTextureSampler,  // both in one binding (Vulkan's sampler2D)
    StorageBuffer,           // read-only
    UniformBuffer,
};

struct BindingDesc
{
    BindingKind kind = BindingKind::Texture;
    uint8_t     slot = 0;                    // Draw::textures[slot] or Draw::storage[slot]
    uint8_t     set = 0, binding = 0;        // the native location
};

struct ProgramLayout
{
    uint8_t  textures       = 0;
    uint8_t  storageBuffers = 0;
    uint16_t constantsBytes = 0;
    uint8_t  colorOutputs   = 1;

    // Empty: the portable convention. Otherwise every binding the program
    // declares, where it is: a backend that can't honour native bindings
    // (anything but the format they were written in) refuses the program.
    std::span<const BindingDesc> bindings;
};

struct ProgramDesc
{
    const char* name = nullptr;          // debug name; may be null

    // Both stages in `format`: the backend's Caps::shaderFormat (the bundle's
    // programs), or SPIR-V with native bindings where Caps::spirvPrograms says
    // the backend takes it. SPIR-V words, or source text (MSL, HLSL, WGSL). The
    // bytes need only outlive createProgram().
    ShaderFormat format = ShaderFormat::SPIRV;
    std::span<const std::byte> vertex;
    std::span<const std::byte> fragment;
    const char* vertexEntry   = "main";
    const char* fragmentEntry = "main";

    ProgramLayout layout;
};

// =============================================================================
// Frames
// =============================================================================

struct FrameInfo
{
    // What the frame's last pass renders into: this frame's swap image for a
    // window, the target texture for an offscreen surface. Only valid until
    // submit(); the core never keeps it.
    TextureId   output;
    PixelFormat outputFormat = PixelFormat::BGRA8;
    Extent      extent;
};

class Surface
{
public:
    virtual ~Surface() = default;

    // Acquire the next frame, on the thread that will submit it. False when
    // there is nothing to draw into this time (minimised, mid-resize,
    // swapchain out of date): skip the frame.
    virtual bool beginFrame(FrameInfo& info) = 0;

    // Execute the packet and present FrameInfo::output (window surfaces).
    // Returns once it no longer reads the packet's memory; the GPU may still
    // be running it.
    virtual void submit(const FramePacket& packet) = 0;

    // The size frames are rendered at now.
    virtual Extent extent() const = 0;

    // ---- Window surfaces (offscreen ones keep the defaults) ----------------
    // The window's true size in physical pixels, asked of the platform: what
    // extent() should be. {0, 0} while hidden or minimised; {~0u, ~0u} when
    // the platform has no preference.
    virtual Extent preferredExtent() const { return extent(); }
    // Whether beginFrame can succeed at all (a swapchain exists, the device
    // isn't lost), and an attempt to make it so. Call neither during a frame.
    virtual bool canBegin() const { return true; }
    virtual void revive() {}
    // Rebuild at a new size. Waits for this surface's work on the GPU.
    virtual void resize(Extent extent) { (void) extent; }
};

class Backend
{
public:
    virtual ~Backend() = default;

    virtual const Caps& caps() const = 0;

    // Sticky: once the GPU is lost nothing more renders, and the host falls
    // back (a new Backend is made for the next window).
    virtual bool isLost() const { return false; }

    // Blocks until the GPU has finished everything submitted on every surface.
    virtual void waitIdle() = 0;

    // ---- Textures -----------------------------------------------------------
    // Null on failure (out of memory, unsupported format). destroyTexture is
    // always safe for a texture a SUBMITTED frame still reads: the backend
    // frees it once the GPU is done. (The core never destroys one a frame it
    // has not yet submitted names.)
    virtual TextureId createTexture(const TextureDesc& desc) = 0;
    virtual void      destroyTexture(TextureId texture) = 0;
    // Whether `format` can be a render target that blends (Target's rule).
    virtual bool      canRenderTo(PixelFormat format) const = 0;

    // Copies `data` now (it need not outlive the call): `region` of (layer,
    // mip), rows tightly packed (formatInfo(format).bytesFor). The rest of the
    // texture is kept. The write lands before the next frame submitted on any
    // surface, and after every frame submitted before the call.
    virtual bool writeTexture(TextureId texture, uint32_t layer, uint32_t mip,
                              IRect region, std::span<const std::byte> data) = 0;
    // Every layer and mip to `colour`, ordered like writeTexture.
    virtual void clearTexture(TextureId texture, ClearColor colour) = 0;

    // Blocking: waits for every submitted frame, then copies `region` into
    // `dst` (formatInfo(format).bytesFor(region) bytes, rows tightly packed).
    virtual bool readTexture(TextureId texture, IRect region, std::span<std::byte> dst) = 0;

    // ---- Programs and pipelines ---------------------------------------------
    // A pipeline lives until its program is destroyed (deferred, like a
    // texture). createPipeline may compile lazily; preparePipelines lets the
    // core warm the ones it knows it will need so no frame stalls on one.
    virtual ProgramId  createProgram(const ProgramDesc& desc) = 0;
    virtual void       destroyProgram(ProgramId program) = 0;
    virtual PipelineId createPipeline(const PipelineDesc& desc) = 0;
    virtual void       preparePipelines(std::span<const PipelineId> pipelines) { (void) pipelines; }

    // ---- Surfaces -----------------------------------------------------------
    // An offscreen surface renders into `target` (TextureUsage::RenderTarget |
    // Sampled): FrameInfo::output is `target`. Null if the backend can't.
    virtual std::unique_ptr<Surface> createOffscreenSurface(TextureId target) = 0;
};

} // namespace jvk::gpu
