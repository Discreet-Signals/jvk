/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MetalBackend.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::metal
{

namespace
{

void logOnce(std::atomic<bool>& flag, const juce::String& line)
{
    if (!flag.exchange(true)) diag::log(line);
}

juce::String describe(NSError* error)
{
    return error != nil ? juce::String::fromUTF8([[error localizedDescription] UTF8String]) : juce::String("unknown error");
}

MTLPixelFormat toMtl(PixelFormat f)
{
    switch (f)
    {
        case PixelFormat::R8:          return MTLPixelFormatR8Unorm;
        case PixelFormat::RG8:         return MTLPixelFormatRG8Unorm;
        case PixelFormat::RGBA8:       return MTLPixelFormatRGBA8Unorm;
        case PixelFormat::BGRA8:       return MTLPixelFormatBGRA8Unorm;
        case PixelFormat::RGBA8sRGB:   return MTLPixelFormatRGBA8Unorm_sRGB;
        case PixelFormat::BGRA8sRGB:   return MTLPixelFormatBGRA8Unorm_sRGB;
        case PixelFormat::R16:         return MTLPixelFormatR16Unorm;
        case PixelFormat::RG16:        return MTLPixelFormatRG16Unorm;
        case PixelFormat::RGBA16:      return MTLPixelFormatRGBA16Unorm;
        case PixelFormat::R16F:        return MTLPixelFormatR16Float;
        case PixelFormat::RG16F:       return MTLPixelFormatRG16Float;
        case PixelFormat::RGBA16F:     return MTLPixelFormatRGBA16Float;
        case PixelFormat::R32F:        return MTLPixelFormatR32Float;
        case PixelFormat::RG32F:       return MTLPixelFormatRG32Float;
        case PixelFormat::RGBA32F:     return MTLPixelFormatRGBA32Float;
        case PixelFormat::RGB10A2:     return MTLPixelFormatRGB10A2Unorm;   // R in the low bits
        case PixelFormat::R11G11B10F:  return MTLPixelFormatRG11B10Float;   // R in the low bits
        case PixelFormat::R8UI:        return MTLPixelFormatR8Uint;
        case PixelFormat::R16UI:       return MTLPixelFormatR16Uint;
        case PixelFormat::R32UI:       return MTLPixelFormatR32Uint;
        case PixelFormat::RG32UI:      return MTLPixelFormatRG32Uint;
        case PixelFormat::BC1:         return MTLPixelFormatBC1_RGBA;
        case PixelFormat::BC3:         return MTLPixelFormatBC3_RGBA;
        case PixelFormat::BC4:         return MTLPixelFormatBC4_RUnorm;
        case PixelFormat::BC5:         return MTLPixelFormatBC5_RGUnorm;
        case PixelFormat::BC6H:        return MTLPixelFormatBC6H_RGBUfloat;
        case PixelFormat::BC7:         return MTLPixelFormatBC7_RGBAUnorm;
        case PixelFormat::BC7sRGB:     return MTLPixelFormatBC7_RGBAUnorm_sRGB;
    }
    return MTLPixelFormatInvalid;
}

MTLBlendFactor toMtl(gpu::BlendFactor f)
{
    using F = gpu::BlendFactor;
    switch (f)
    {
        case F::Zero:             return MTLBlendFactorZero;
        case F::One:              return MTLBlendFactorOne;
        case F::SrcColor:         return MTLBlendFactorSourceColor;
        case F::OneMinusSrcColor: return MTLBlendFactorOneMinusSourceColor;
        case F::SrcAlpha:         return MTLBlendFactorSourceAlpha;
        case F::OneMinusSrcAlpha: return MTLBlendFactorOneMinusSourceAlpha;
        case F::DstColor:         return MTLBlendFactorDestinationColor;
        case F::OneMinusDstColor: return MTLBlendFactorOneMinusDestinationColor;
        case F::DstAlpha:         return MTLBlendFactorDestinationAlpha;
        case F::OneMinusDstAlpha: return MTLBlendFactorOneMinusDestinationAlpha;
    }
    return MTLBlendFactorZero;
}

MTLBlendOperation toMtl(gpu::BlendOp o)
{
    switch (o)
    {
        case gpu::BlendOp::Add:             return MTLBlendOperationAdd;
        case gpu::BlendOp::Subtract:        return MTLBlendOperationSubtract;
        case gpu::BlendOp::ReverseSubtract: return MTLBlendOperationReverseSubtract;
        case gpu::BlendOp::Min:             return MTLBlendOperationMin;
        case gpu::BlendOp::Max:             return MTLBlendOperationMax;
    }
    return MTLBlendOperationAdd;
}

MTLCompareFunction toMtl(gpu::CompareOp c)
{
    using C = gpu::CompareOp;
    switch (c)
    {
        case C::Never:          return MTLCompareFunctionNever;
        case C::Less:           return MTLCompareFunctionLess;
        case C::Equal:          return MTLCompareFunctionEqual;
        case C::LessOrEqual:    return MTLCompareFunctionLessEqual;
        case C::Greater:        return MTLCompareFunctionGreater;
        case C::NotEqual:       return MTLCompareFunctionNotEqual;
        case C::GreaterOrEqual: return MTLCompareFunctionGreaterEqual;
        case C::Always:         return MTLCompareFunctionAlways;
    }
    return MTLCompareFunctionAlways;
}

MTLStencilOperation toMtl(gpu::StencilOp o)
{
    using S = gpu::StencilOp;
    switch (o)
    {
        case S::Keep:           return MTLStencilOperationKeep;
        case S::Zero:           return MTLStencilOperationZero;
        case S::Replace:        return MTLStencilOperationReplace;
        case S::IncrementClamp: return MTLStencilOperationIncrementClamp;
        case S::DecrementClamp: return MTLStencilOperationDecrementClamp;
        case S::Invert:         return MTLStencilOperationInvert;
        case S::IncrementWrap:  return MTLStencilOperationIncrementWrap;
        case S::DecrementWrap:  return MTLStencilOperationDecrementWrap;
    }
    return MTLStencilOperationKeep;
}

MTLVertexFormat toMtl(gpu::VertexFormat f)
{
    using V = gpu::VertexFormat;
    switch (f)
    {
        case V::Float:      return MTLVertexFormatFloat;
        case V::Float2:     return MTLVertexFormatFloat2;
        case V::Float3:     return MTLVertexFormatFloat3;
        case V::Float4:     return MTLVertexFormatFloat4;
        case V::UInt:       return MTLVertexFormatUInt;
        case V::UByte4Norm: return MTLVertexFormatUChar4Normalized;
    }
    return MTLVertexFormatFloat4;
}

juce::CriticalSection g_backendLock;
std::weak_ptr<Backend> g_backend;

} // namespace

// =============================================================================
// The backend
// =============================================================================

std::shared_ptr<Backend> Backend::acquire()
{
    const juce::ScopedLock lk(g_backendLock);
    auto b = g_backend.lock();
    if (b && !b->isLost()) return b;
    b = std::shared_ptr<Backend>(new Backend());
    g_backend = b;
    return b;
}

Backend::Backend()
{
    @autoreleasepool
    {
        // The GPU driving the main display (an eGPU or the discrete GPU when it
        // is the one showing the window), else the system's default.
        device_ = CGDirectDisplayCopyCurrentMetalDevice(CGMainDisplayID());
        if (device_ == nil) device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) { reason_ = "no Metal device"; diag::log("metal: " + reason_); return; }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) { reason_ = "no Metal command queue"; diag::log("metal: " + reason_); return; }
        // Compute streams' own: their kernels run beside the frames, not in
        // their queue (COMPUTE.md, "The queues"). Nil leaves compute off.
        computeQueue_ = [device_ newCommandQueue];
        if (computeQueue_ != nil) computeQueue_.label = @"jvk compute";

        unified_ = [device_ hasUnifiedMemory];
        stats_   = std::getenv("JVK_METAL_STATS") != nullptr;
        initCaps();

        for (uint32_t f = 0; f < 2; ++f)
            for (uint32_t a = 0; a < 3; ++a)
            {
                MTLSamplerDescriptor* d = [[MTLSamplerDescriptor alloc] init];
                d.minFilter = d.magFilter = f == 1 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
                d.mipFilter = f == 1 ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
                const MTLSamplerAddressMode mode = a == 1 ? MTLSamplerAddressModeRepeat
                                                 : a == 2 ? MTLSamplerAddressModeMirrorRepeat
                                                          : MTLSamplerAddressModeClampToEdge;
                d.sAddressMode = d.tAddressMode = d.rAddressMode = mode;
                samplers_[f * 3 + a] = Strong<id<MTLSamplerState>>::adopt([device_ newSamplerStateWithDescriptor:d]);
                [d release];
            }

        // What a draw samples where it names a dead texture: transparent black.
        gpu::TextureDesc dd;
        dd.width = dd.height = 1;
        dd.label = "jvk dummy";
        dummy_ = createTexture(dd);
        const std::byte zero[4] {};
        writeTexture(dummy_, 0, 0, { 0, 0, 1, 1 }, zero);

        diag::log("metal: " + juce::String::fromUTF8([[device_ name] UTF8String])
                  + (unified_ ? ", unified memory" : ", discrete memory"));
    }
}

// Releases its references only: a command buffer still on the GPU holds its own
// (and this may run on Metal's completion thread, holding the last reference).
Backend::~Backend()
{
    textures_.forEachLive([](uint32_t, TextureSlot& t) { t = {}; });
    pipelines_.forEachLive([](uint32_t, PipelineSlot& p) { p = {}; });
    programs_.forEachLive([](uint32_t, ProgramSlot& p) { p = {}; });
    buffers_.forEachLive([](uint32_t, BufferSlot& b) { b = {}; });
    kernels_.forEachLive([](uint32_t, KernelSlot& k) { k = {}; });
    kernelLibraries_.clear();
    fillState_ = {};
    copyState_ = {};
    functions_.clear();
    depthStencilStates_.clear();
    for (auto& s : samplers_) s = {};
    pending_.clear();
    filling_ = {};
    waiting_.clear();
    free_.clear();
    if (computeQueue_ != nil) [computeQueue_ release];
    if (queue_ != nil)  [queue_ release];
    if (device_ != nil) [device_ release];
}

void Backend::initCaps()
{
    caps_.shaderFormat  = gpu::ShaderFormat::MSL;
    const bool family16k = [device_ supportsFamily:MTLGPUFamilyMac2] || [device_ supportsFamily:MTLGPUFamilyApple3];
    caps_.maxTextureSize      = family16k ? 16384 : 8192;
    caps_.maxColorAttachments = gpu::kMaxColorAttachments;
    // Tile memory per sample on Apple GPUs; Mac-family GPUs have no such limit.
    caps_.maxColorBytesPerSample = [device_ supportsFamily:MTLGPUFamilyMac2] ? 128 : 64;
    caps_.maxDrawConstantsBytes  = 128;
    caps_.constantsAlignment     = 4;
    caps_.storageAlignment       = 256;   // constant-address-space buffer offsets on Mac-family GPUs
    caps_.independentBlend       = true;
    caps_.textureCompressionBC   = [device_ respondsToSelector:@selector(supportsBCTextureCompression)]
                                       ? (bool) [device_ supportsBCTextureCompression]
                                       : !unified_;
    initComputeCaps();
}

// ---- Submissions ------------------------------------------------------------------

void Backend::commit(id<MTLCommandBuffer> cb, uint64_t serial, std::function<void()> done)
{
    std::shared_ptr<Backend> self = shared_from_this();
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if ([c status] == MTLCommandBufferStatusError)
        {
            static std::atomic<bool> logged { false };
            logOnce(logged, "metal: a command buffer failed: " + describe([c error]));
            // Only a GPU that is gone is lost; a frame the watchdog stopped (a
            // timeout) costs that frame, not the GPU path.
            if ([[c error] code] == MTLCommandBufferErrorDeviceRemoved)
                self->lost_.store(true, std::memory_order_release);
        }
        uint64_t seen = self->completed_.load(std::memory_order_acquire);
        while (seen < serial && !self->completed_.compare_exchange_weak(seen, serial, std::memory_order_acq_rel)) {}
        {
            std::lock_guard<std::mutex> lk(self->idleMutex_);
        }
        self->idle_.notify_all();
        if (done) done();
    }];
    [cb commit];
    if (stats_ && serial % 600 == 0)   // JVK_METAL_STATS: every ~10 s at 60 fps
        diag::log("metal: texture writes since start: " + juce::String(directBytes_.load() / 1048576.0, 1)
                  + " MB in place, " + juce::String(stagedBytes_.load() / 1048576.0, 1) + " MB staged");
}

void Backend::flushAndWait()
{
    uint64_t serial = 0;
    {
        std::lock_guard<std::mutex> lk(submitMutex_);
        @autoreleasepool
        {
            serial = beginSubmission();
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            encodePendingWrites(cb, serial);
            commit(cb, serial);
        }
    }
    std::unique_lock<std::mutex> lk(idleMutex_);
    idle_.wait_for(lk, std::chrono::seconds(5), [this, serial] { return completed() >= serial; });
}

void Backend::waitIdle()
{
    if (usable()) flushAndWait();
}

// ---- Textures ---------------------------------------------------------------------

gpu::TextureId Backend::createTexture(const gpu::TextureDesc& desc)
{
    if (!usable() || desc.width == 0 || desc.height == 0) return {};
    const bool depthStencil = desc.kind == gpu::TextureKind::DepthStencil;
    const bool renderTarget = gpu::hasUsage(desc.usage, gpu::TextureUsage::RenderTarget);
    const bool readback     = gpu::hasUsage(desc.usage, gpu::TextureUsage::Readback);
    // In place on unified memory: what the CPU fills (and a target read back);
    // other render targets private, where the GPU can compress them.
    const bool shared = unified_ && !depthStencil && (!renderTarget || readback);

    id<MTLTexture> texture = nil;
    @autoreleasepool
    {
        MTLTextureDescriptor* d = [[MTLTextureDescriptor alloc] init];
        d.textureType      = desc.layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
        d.pixelFormat      = depthStencil ? MTLPixelFormatStencil8 : toMtl(desc.format);
        d.width            = desc.width;
        d.height           = desc.height;
        d.arrayLength      = std::max(1u, desc.layers);
        d.mipmapLevelCount = std::max(1u, desc.mips);
        MTLTextureUsage usage = 0;
        if (gpu::hasUsage(desc.usage, gpu::TextureUsage::Sampled)) usage |= MTLTextureUsageShaderRead;
        if (renderTarget || depthStencil)                         usage |= MTLTextureUsageRenderTarget;
        d.usage       = usage != 0 ? usage : MTLTextureUsageShaderRead;
        d.storageMode = shared ? MTLStorageModeShared : MTLStorageModePrivate;
        if (shared && !readback) d.cpuCacheMode = MTLCPUCacheModeWriteCombined;   // written, never read, by the CPU
        if (d.pixelFormat == MTLPixelFormatInvalid) { [d release]; return {}; }
        texture = [device_ newTextureWithDescriptor:d];
        [d release];
        if (texture != nil && desc.label != nullptr)
            texture.label = [NSString stringWithUTF8String:desc.label];
    }
    if (texture == nil)
    {
        static std::atomic<bool> logged { false };
        logOnce(logged, "metal: a texture could not be created (" + juce::String((int) desc.width) + "x"
                        + juce::String((int) desc.height) + ")");
        return {};
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    TextureSlot* slot = nullptr;
    const uint32_t handle = textures_.allocate(slot);
    if (handle == 0) { [texture release]; return {}; }
    slot->texture      = Strong<id<MTLTexture>>::adopt(texture);
    slot->pixelFormat  = desc.format;
    slot->kind         = desc.kind;
    slot->width        = desc.width;
    slot->height       = desc.height;
    slot->layers       = std::max(1u, desc.layers);
    slot->mips         = std::max(1u, desc.mips);
    slot->shared       = shared;
    slot->renderTarget = renderTarget;
    return { handle };
}

void Backend::destroyTexture(gpu::TextureId texture)
{
    // Every command buffer that encoded it holds its own reference.
    std::lock_guard<std::mutex> lk(tableMutex_);
    textures_.release(texture.value);
}

gpu::TextureId Backend::createSwapTexture()
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    TextureSlot* slot = nullptr;
    const uint32_t handle = textures_.allocate(slot);
    if (handle == 0) return {};
    slot->pixelFormat  = PixelFormat::BGRA8;
    slot->renderTarget = true;
    return { handle };
}

void Backend::setSwapTexture(gpu::TextureId swap, id<MTLTexture> texture)
{
    if (auto* t = this->texture(swap))
    {
        t->texture = Strong<id<MTLTexture>>::retain(texture);
        t->width   = texture != nil ? static_cast<uint32_t>([texture width]) : 0;
        t->height  = texture != nil ? static_cast<uint32_t>([texture height]) : 0;
    }
}

bool Backend::canRenderTo(PixelFormat format) const
{
    switch (format)
    {
        case PixelFormat::R8: case PixelFormat::RG8: case PixelFormat::RGBA8: case PixelFormat::BGRA8:
        case PixelFormat::RGBA8sRGB: case PixelFormat::BGRA8sRGB:
        case PixelFormat::R16: case PixelFormat::RG16: case PixelFormat::RGBA16:
        case PixelFormat::R16F: case PixelFormat::RG16F: case PixelFormat::RGBA16F:
        case PixelFormat::RGB10A2: case PixelFormat::R11G11B10F:
            return true;
        case PixelFormat::R32F: case PixelFormat::RG32F: case PixelFormat::RGBA32F:
            return [device_ supportsFamily:MTLGPUFamilyMac2];   // blendable there only
        default:
            return false;
    }
}

bool Backend::stage(TextureSlot& t, uint32_t handle, uint32_t layer, uint32_t mip, gpu::IRect region,
                    const std::byte* data, size_t rowBytes, size_t imageBytes)
{
    // Free again: every chunk whose last reader has finished.
    const uint64_t done = completed();
    for (size_t i = 0; i < waiting_.size();)
    {
        if (waiting_[i].freeAfter != 0 && waiting_[i].freeAfter <= done)
        {
            waiting_[i].head = 0;
            waiting_[i].freeAfter = 0;
            free_.push_back(std::move(waiting_[i]));
            waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        else ++i;
    }

    const size_t size = (imageBytes + 15) & ~size_t(15);   // offsets stay 16-byte aligned
    if (!filling_.buffer || filling_.head + size > [filling_.buffer.get() length])
    {
        if (filling_.buffer)
        {
            if (filling_.head > 0) waiting_.push_back(std::move(filling_));
            else                   free_.push_back(std::move(filling_));
        }
        filling_ = {};
        for (size_t i = 0; i < free_.size(); ++i)
            if ([free_[i].buffer.get() length] >= size)
            {
                filling_ = std::move(free_[i]);
                free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        if (!filling_.buffer)
        {
            id<MTLBuffer> b = [device_ newBufferWithLength:std::max(size, kStagingChunk)
                                                   options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
            if (b == nil) return false;
            filling_.buffer = Strong<id<MTLBuffer>>::adopt(b);
        }
    }

    auto& chunk = filling_;
    std::memcpy(static_cast<std::byte*>([chunk.buffer.get() contents]) + chunk.head, data, imageBytes);
    PendingWrite w;
    w.texture    = handle;
    w.layer      = layer;
    w.mip        = mip;
    w.region     = region;
    w.buffer     = chunk.buffer;
    w.offset     = chunk.head;
    w.rowBytes   = rowBytes;
    w.imageBytes = imageBytes;
    pending_.push_back(std::move(w));
    chunk.head += size;
    ++t.pendingWrites;
    if (stats_) stagedBytes_ += imageBytes;
    return true;
}

bool Backend::writeTexture(gpu::TextureId texture, uint32_t layer, uint32_t mip,
                           gpu::IRect region, std::span<const std::byte> data)
{
    TextureSlot* t = this->texture(texture);
    if (t == nullptr || !t->texture || t->kind != gpu::TextureKind::Color) return false;
    if (region.isEmpty()) return true;
    if (layer >= t->layers || mip >= t->mips) return false;
    const auto info = formatInfo(t->pixelFormat);
    const size_t rowBytes   = static_cast<size_t>(info.bytesFor(static_cast<uint32_t>(region.width), 1));
    const size_t imageBytes = static_cast<size_t>(info.bytesFor(static_cast<uint32_t>(region.width),
                                                                static_cast<uint32_t>(region.height)));
    if (data.size() < imageBytes) return false;

    // In place when nothing can be reading it: no frame being encoded (the
    // submit lock is free), none on the GPU still using it, and no earlier
    // staged write to it waiting to land.
    if (t->shared)
    {
        std::unique_lock<std::mutex> submit(submitMutex_, std::try_to_lock);
        if (submit.owns_lock())
        {
            bool direct = false;
            {
                std::lock_guard<std::mutex> lk(writeMutex_);
                direct = t->pendingWrites == 0 && t->lastUse <= completed();
            }
            if (direct)
            {
                [t->texture.get() replaceRegion:MTLRegionMake2D(region.x, region.y, region.width, region.height)
                                    mipmapLevel:mip
                                          slice:layer
                                      withBytes:data.data()
                                    bytesPerRow:rowBytes
                                  bytesPerImage:imageBytes];
                if (stats_) directBytes_ += imageBytes;
                return true;
            }
        }
    }

    std::lock_guard<std::mutex> lk(writeMutex_);
    return stage(*t, texture.value, layer, mip, region, data.data(), rowBytes, imageBytes);
}

void Backend::clearTexture(gpu::TextureId texture, gpu::ClearColor colour)
{
    TextureSlot* t = this->texture(texture);
    if (t == nullptr || !t->texture) return;
    std::lock_guard<std::mutex> lk(writeMutex_);
    PendingWrite w;
    w.texture = texture.value;
    w.clear   = true;
    w.colour  = colour;
    pending_.push_back(std::move(w));
    ++t->pendingWrites;
}

void Backend::encodePendingWrites(id<MTLCommandBuffer> cb, uint64_t serial)
{
    std::vector<PendingWrite> writes;
    {
        std::lock_guard<std::mutex> lk(writeMutex_);
        if (pending_.empty()) return;
        writes.swap(pending_);
        for (auto& w : writes)
            if (auto* t = textures_.resolve(w.texture); t != nullptr && t->pendingWrites > 0)
                --t->pendingWrites;
        // Every chunk written so far is read by this submission: reusable once it is done.
        if (filling_.head > 0)
        {
            waiting_.push_back(std::move(filling_));
            filling_ = {};
        }
        for (auto& c : waiting_)
            if (c.freeAfter == 0) c.freeAfter = serial;
    }

    id<MTLBlitCommandEncoder> blit = nil;
    for (const auto& w : writes)
    {
        TextureSlot* t = textures_.resolve(w.texture);
        if (t == nullptr || !t->texture) continue;   // destroyed meanwhile
        t->lastUse = serial;
        if (w.clear)
        {
            if (blit != nil) { [blit endEncoding]; blit = nil; }
            if (!t->renderTarget) continue;   // only a target clears (the core never clears another)
            for (uint32_t layer = 0; layer < t->layers; ++layer)
                for (uint32_t mip = 0; mip < t->mips; ++mip)
                {
                    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
                    rp.colorAttachments[0].texture     = t->texture.get();
                    rp.colorAttachments[0].slice       = layer;
                    rp.colorAttachments[0].level       = mip;
                    rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
                    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                    rp.colorAttachments[0].clearColor  = MTLClearColorMake(w.colour.r, w.colour.g, w.colour.b, w.colour.a);
                    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
                }
            continue;
        }
        if (blit == nil) blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:w.buffer.get()
                sourceOffset:w.offset
           sourceBytesPerRow:w.rowBytes
         sourceBytesPerImage:w.imageBytes
                  sourceSize:MTLSizeMake(static_cast<NSUInteger>(w.region.width), static_cast<NSUInteger>(w.region.height), 1)
                   toTexture:t->texture.get()
            destinationSlice:w.layer
            destinationLevel:w.mip
           destinationOrigin:MTLOriginMake(static_cast<NSUInteger>(w.region.x), static_cast<NSUInteger>(w.region.y), 0)];
    }
    if (blit != nil) [blit endEncoding];
}

bool Backend::readTexture(gpu::TextureId texture, gpu::IRect region, std::span<std::byte> dst)
{
    if (!usable()) return false;
    flushAndWait();
    TextureSlot* t = this->texture(texture);
    if (t == nullptr || !t->texture || region.isEmpty()) return false;
    const auto info = formatInfo(t->pixelFormat);
    const size_t rowBytes   = static_cast<size_t>(info.bytesFor(static_cast<uint32_t>(region.width), 1));
    const size_t imageBytes = static_cast<size_t>(info.bytesFor(static_cast<uint32_t>(region.width),
                                                                static_cast<uint32_t>(region.height)));
    if (dst.size() < imageBytes) return false;
    const MTLRegion r = MTLRegionMake2D(region.x, region.y, region.width, region.height);

    if (t->shared)   // in place: the GPU is done with it
    {
        [t->texture.get() getBytes:dst.data() bytesPerRow:rowBytes fromRegion:r mipmapLevel:0];
        return true;
    }

    bool ok = false;
    @autoreleasepool
    {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:imageBytes options:MTLResourceStorageModeShared];
        if (buffer == nil) return false;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:t->texture.get() sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(static_cast<NSUInteger>(region.x), static_cast<NSUInteger>(region.y), 0)
                   sourceSize:MTLSizeMake(static_cast<NSUInteger>(region.width), static_cast<NSUInteger>(region.height), 1)
                     toBuffer:buffer destinationOffset:0 destinationBytesPerRow:rowBytes destinationBytesPerImage:imageBytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusCompleted)
        {
            std::memcpy(dst.data(), [buffer contents], imageBytes);
            ok = true;
        }
        [buffer release];
    }
    return ok;
}

// ---- Programs and pipelines ---------------------------------------------------------

Strong<id<MTLFunction>> Backend::function(const gpu::ProgramDesc& desc, bool vertex, const std::string*& held, juce::String& error)
{
    held = nullptr;
    if (desc.format != gpu::ShaderFormat::MSL) { error = "a shader format Metal can't take"; return {}; }
    const auto bytes = vertex ? desc.vertex : desc.fragment;
    const std::string source(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const std::string entry = vertex ? desc.vertexEntry : desc.fragmentEntry;

    const std::string key = entry + '\n' + source;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (auto it = functions_.find(key); it != functions_.end())
        {
            ++it->second.users;
            held = &it->first;
            return it->second.function;
        }
    }

    Function compiled;
    @autoreleasepool
    {
        NSString* text = [[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_1;
        NSError* err = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:text options:options error:&err];
        [options release];
        [text release];
        if (library == nil) { error = describe(err); return {}; }
        compiled.library = Strong<id<MTLLibrary>>::adopt(library);
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:entry.c_str()]];
        if (fn == nil) { error = juce::String("no entry point ") + entry.c_str(); return {}; }
        compiled.function = Strong<id<MTLFunction>>::adopt(fn);
    }

    // Another program may have made the same stage meanwhile: the first one in is shared.
    std::lock_guard<std::mutex> lk(tableMutex_);
    auto [it, inserted] = functions_.emplace(key, std::move(compiled));
    ++it->second.users;
    held = &it->first;
    return it->second.function;
}

void Backend::releaseFunction(const std::string* key)
{
    if (key == nullptr) return;
    const auto it = functions_.find(*key);
    jassert(it != functions_.end() && it->second.users > 0);
    if (it != functions_.end() && --it->second.users == 0) functions_.erase(it);
}

gpu::ProgramId Backend::createProgram(const gpu::ProgramDesc& desc)
{
    if (!usable() || desc.vertex.empty() || desc.fragment.empty()) return {};

    ProgramSlot p;
    p.name           = desc.name != nullptr ? desc.name : "?";
    p.constantsBytes = std::min<uint32_t>(desc.layout.constantsBytes, 4096u);
    const auto bindings = desc.layout.bindings;
    if (bindings.empty())
    {
        for (uint8_t i = 0; i < gpu::kMaxDrawTextures; ++i)
        {
            p.textureIndex[i] = i;
            p.samplerIndex[i] = static_cast<uint8_t>(i < gpu::kPortableDrawTextures ? kSamplerBase + i : 0);
        }
        for (uint8_t j = 0; j < gpu::kMaxDrawStorage; ++j) p.storageIndex[j] = static_cast<uint8_t>(kBlockBase + j);
    }
    else
    {
        // A jvk::Shader's own bindings, where jvk_msl translated them (MslBindings.h).
        std::vector<MslBinding> all;
        for (const auto& b : bindings)
            all.push_back({ b.set, b.binding, b.kind == gpu::BindingKind::StorageBuffer || b.kind == gpu::BindingKind::UniformBuffer });
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto slot  = bindings[i].slot;
            const auto index = static_cast<uint8_t>(mslIndex(all.data(), all.size(), all[i]));
            if (all[i].block) { if (slot < gpu::kMaxDrawStorage) p.storageIndex[slot] = index; }
            else if (slot < gpu::kMaxDrawTextures) p.textureIndex[slot] = p.samplerIndex[slot] = index;
        }
    }

    juce::String error;
    p.vertex = function(desc, true, p.vertexKey, error);
    if (p.vertex) p.fragment = function(desc, false, p.fragmentKey, error);
    if (!p.vertex || !p.fragment)
    {
        diag::log("metal: program " + juce::String(p.name) + " could not be created: " + error);
        std::lock_guard<std::mutex> lk(tableMutex_);
        releaseFunction(p.vertexKey);
        releaseFunction(p.fragmentKey);
        return {};
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    ProgramSlot* slot = nullptr;
    const uint32_t handle = programs_.allocate(slot);
    if (handle == 0)
    {
        releaseFunction(p.vertexKey);
        releaseFunction(p.fragmentKey);
        return {};
    }
    *slot = std::move(p);
    return { handle };
}

void Backend::destroyProgram(gpu::ProgramId program)
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    ProgramSlot* p = programs_.resolve(program.value);
    if (p == nullptr) return;
    for (uint32_t ph : p->pipelines)
        if (PipelineSlot* ps = pipelines_.resolve(ph))
        {
            pipelineIds_.erase(ps->desc);
            pipelines_.release(ph);
        }
    // Its stages' functions go with the last program that uses them (a pipeline
    // a command buffer still draws with needs neither).
    releaseFunction(p->vertexKey);
    releaseFunction(p->fragmentKey);
    programs_.release(program.value);
}

id<MTLDepthStencilState> Backend::depthStencilState(const gpu::StencilState& stencil)
{
    for (const auto& [s, state] : depthStencilStates_)
        if (s == stencil) return state.get();
    id<MTLDepthStencilState> state = nil;
    @autoreleasepool
    {
        MTLDepthStencilDescriptor* d = [[MTLDepthStencilDescriptor alloc] init];
        d.depthCompareFunction = MTLCompareFunctionAlways;
        d.depthWriteEnabled    = NO;
        if (stencil.enabled)
        {
            MTLStencilDescriptor* s = [[MTLStencilDescriptor alloc] init];
            s.stencilCompareFunction    = toMtl(stencil.compare);
            s.stencilFailureOperation   = toMtl(stencil.failOp);
            s.depthFailureOperation     = MTLStencilOperationKeep;
            s.depthStencilPassOperation = toMtl(stencil.passOp);
            s.readMask                  = stencil.readMask;
            s.writeMask                 = stencil.writeMask;
            d.frontFaceStencil = s;
            d.backFaceStencil  = s;
            [s release];
        }
        state = [device_ newDepthStencilStateWithDescriptor:d];
        [d release];
    }
    depthStencilStates_.emplace_back(stencil, Strong<id<MTLDepthStencilState>>::adopt(state));
    return state;
}

gpu::PipelineId Backend::createPipeline(const gpu::PipelineDesc& desc)
{
    if (!usable()) return {};
    // The program's functions held here, so building outside the lock never
    // races its destruction (they go with the last program that uses them).
    Strong<id<MTLFunction>> vertex, fragment;
    std::string name;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (auto it = pipelineIds_.find(desc); it != pipelineIds_.end()) return { it->second };
        const ProgramSlot* prog = programs_.resolve(desc.program.value);
        if (prog == nullptr) return {};
        vertex   = prog->vertex;
        fragment = prog->fragment;
        name     = prog->name;
    }

    PipelineSlot built;
    built.desc      = desc;
    built.program   = desc.program.value;
    built.primitive = desc.topology == gpu::Topology::TriangleStrip ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
    @autoreleasepool
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = vertex.get();
        d.fragmentFunction = fragment.get();
        if (name.size() > 0) d.label = [NSString stringWithUTF8String:name.c_str()];
        const auto& att = desc.attachments;
        for (uint32_t i = 0; i < att.colorCount && i < gpu::kMaxColorAttachments; ++i)
        {
            MTLRenderPipelineColorAttachmentDescriptor* ca = d.colorAttachments[i];
            if ((att.unusedMask & (1u << i)) != 0) { ca.pixelFormat = MTLPixelFormatInvalid; continue; }
            ca.pixelFormat = toMtl(att.colorFormats[i]);
            const auto& b = desc.blend[i];
            ca.blendingEnabled             = b.enabled ? YES : NO;
            ca.sourceRGBBlendFactor        = toMtl(b.srcColor);
            ca.destinationRGBBlendFactor   = toMtl(b.dstColor);
            ca.rgbBlendOperation           = toMtl(b.colorOp);
            ca.sourceAlphaBlendFactor      = toMtl(b.srcAlpha);
            ca.destinationAlphaBlendFactor = toMtl(b.dstAlpha);
            ca.alphaBlendOperation         = toMtl(b.alphaOp);
            ca.writeMask = ((b.writeMask & gpu::WriteR) ? MTLColorWriteMaskRed : 0)
                         | ((b.writeMask & gpu::WriteG) ? MTLColorWriteMaskGreen : 0)
                         | ((b.writeMask & gpu::WriteB) ? MTLColorWriteMaskBlue : 0)
                         | ((b.writeMask & gpu::WriteA) ? MTLColorWriteMaskAlpha : 0);
        }
        if (att.depthStencil) d.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;

        const auto& vl = desc.vertexLayout;
        if (vl.stride > 0)
        {
            MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
            for (uint32_t i = 0; i < vl.attributeCount && i < gpu::kMaxVertexAttributes; ++i)
            {
                const auto& a = vl.attributes[i];
                vd.attributes[a.location].format      = toMtl(a.format);
                vd.attributes[a.location].offset      = a.offset;
                vd.attributes[a.location].bufferIndex = kVertexBufferIndex;
            }
            vd.layouts[kVertexBufferIndex].stride       = vl.stride;
            vd.layouts[kVertexBufferIndex].stepFunction = MTLVertexStepFunctionPerVertex;
            vd.layouts[kVertexBufferIndex].stepRate     = 1;
            d.vertexDescriptor = vd;
        }

        NSError* err = nil;
        id<MTLRenderPipelineState> state = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        [d release];
        if (state != nil) built.state = Strong<id<MTLRenderPipelineState>>::adopt(state);
        else
        {
            static std::atomic<bool> logged { false };
            logOnce(logged, "metal: pipeline " + juce::String(name) + " could not be built: " + describe(err));
        }
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    if (auto it = pipelineIds_.find(desc); it != pipelineIds_.end()) return { it->second };
    ProgramSlot* prog = programs_.resolve(desc.program.value);   // destroyed meanwhile?
    if (prog == nullptr) return {};
    if (desc.attachments.depthStencil) built.depthStencil = Strong<id<MTLDepthStencilState>>::retain(depthStencilState(desc.stencil));
    PipelineSlot* slot = nullptr;
    const uint32_t handle = pipelines_.allocate(slot);
    if (handle == 0) return {};
    *slot = std::move(built);
    prog->pipelines.push_back(handle);
    pipelineIds_.emplace(desc, handle);
    return { handle };
}

std::unique_ptr<gpu::Surface> Backend::createOffscreenSurface(gpu::TextureId target)
{
    if (!usable() || texture(target) == nullptr) return nullptr;
    return std::make_unique<OffscreenSurface>(shared_from_this(), target);
}

} // namespace jvk::metal
