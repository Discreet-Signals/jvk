/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MetalSurface.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::metal
{

namespace
{

gpu::IRect intersect(const gpu::IRect& a, const gpu::IRect& b)
{
    const int32_t x0 = std::max(a.x, b.x), y0 = std::max(a.y, b.y);
    const int32_t x1 = std::min(a.x + a.width, b.x + b.width), y1 = std::min(a.y + a.height, b.y + b.height);
    return { x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0) };
}

MTLLoadAction toMtl(gpu::LoadOp op)
{
    switch (op)
    {
        case gpu::LoadOp::Load:     return MTLLoadActionLoad;
        case gpu::LoadOp::Clear:    return MTLLoadActionClear;
        case gpu::LoadOp::DontCare: return MTLLoadActionDontCare;
    }
    return MTLLoadActionLoad;
}

MTLStoreAction toMtl(gpu::StoreOp op)
{
    return op == gpu::StoreOp::Store ? MTLStoreActionStore : MTLStoreActionDontCare;
}

// Waits for every frame a surface has in flight (each one holds one count).
void drain(dispatch_semaphore_t semaphore, uint32_t frames)
{
    for (uint32_t i = 0; i < frames; ++i)
        if (dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, 2 * (int64_t) NSEC_PER_SEC)) != 0)
            return;   // a GPU that never answers: give up rather than hang
    for (uint32_t i = 0; i < frames; ++i) dispatch_semaphore_signal(semaphore);
}

// What a command buffer's completion signals: the semaphore it holds a reference to.
std::function<void()> signalOnDone(dispatch_semaphore_t semaphore)
{
    dispatch_retain(semaphore);
    return [semaphore] { dispatch_semaphore_signal(semaphore); dispatch_release(semaphore); };
}

} // namespace

// =============================================================================
// The packet executor
// =============================================================================

bool PacketExecutor::ensure(Strong<id<MTLBuffer>>& buffer, size_t size)
{
    // Grown, never shrunk: the steady state allocates nothing. The old buffer
    // goes now; a command buffer still reading it holds its own reference.
    if (buffer && [buffer.get() length] >= size) return true;
    const size_t grown = std::max<size_t>({ size, buffer ? [buffer.get() length] * 2 : 0, size_t(1u << 20) });
    id<MTLBuffer> b = [backend_.device() newBufferWithLength:grown
                                                     options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
    buffer = Strong<id<MTLBuffer>>::adopt(b);
    return b != nil;
}

void PacketExecutor::encode(id<MTLCommandBuffer> cb, FrameSlot& slot, const gpu::FramePacket& packet, uint64_t serial)
{
    // One copy of each stream into this slot's buffers (constants go inline
    // with each draw). The slot's last frame has finished: beginFrame waited.
    if (!ensure(slot.vertices, std::max<size_t>(packet.vertexData.size(), 256))
        || !ensure(slot.storage, std::max<size_t>(packet.storageData.size(), 256)))
        return;
    if (!packet.vertexData.empty())
        std::memcpy([slot.vertices.get() contents], packet.vertexData.data(), packet.vertexData.size());
    if (!packet.storageData.empty())
        std::memcpy([slot.storage.get() contents], packet.storageData.data(), packet.storageData.size());

    for (const auto& pass : packet.passes)
        encodePass(cb, slot, packet, pass, serial);
}

void PacketExecutor::encodePass(id<MTLCommandBuffer> cb, FrameSlot& slot, const gpu::FramePacket& packet,
                                const gpu::Pass& pass, uint64_t serial)
{
    if (pass.firstDraw > packet.draws.size() || pass.drawCount > packet.draws.size() - pass.firstDraw)
        return;

    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    uint32_t width = 0, height = 0;
    const uint32_t colorCount = std::min<uint32_t>(pass.colorCount, gpu::kMaxColorAttachments);
    for (uint32_t i = 0; i < colorCount; ++i)
    {
        const auto& a = pass.color[i];
        if (!a.texture) continue;   // an unused slot (AttachmentFormats::unusedMask)
        TextureSlot* t = backend_.texture(a.texture);
        if (t == nullptr || !t->texture) return;   // a dead attachment: nothing valid to draw into
        MTLRenderPassColorAttachmentDescriptor* ca = rp.colorAttachments[i];
        ca.texture     = t->texture.get();
        ca.loadAction  = toMtl(a.load);
        ca.storeAction = toMtl(a.store);
        ca.clearColor  = MTLClearColorMake(a.clear.r, a.clear.g, a.clear.b, a.clear.a);
        t->lastUse = serial;
        if (width == 0) { width = t->width; height = t->height; }
    }
    if (pass.depthStencil.texture)
    {
        TextureSlot* ds = backend_.texture(pass.depthStencil.texture);
        if (ds == nullptr || !ds->texture) return;
        rp.stencilAttachment.texture      = ds->texture.get();
        rp.stencilAttachment.loadAction   = toMtl(pass.depthStencil.load);
        rp.stencilAttachment.storeAction  = toMtl(pass.depthStencil.store);
        rp.stencilAttachment.clearStencil = pass.depthStencil.clearStencil;
        ds->lastUse = serial;
        if (width == 0) { width = ds->width; height = ds->height; }
    }
    if (width == 0 || height == 0) return;

    const gpu::IRect whole { 0, 0, int32_t(width), int32_t(height) };
    const gpu::IRect area = intersect(pass.renderArea, whole);
    if (area.isEmpty()) return;
    // Metal has no render area: a clear or discard would reach pixels outside
    // it, which a pass keeps (FramePacket). Load them instead (the core only
    // clears or discards whole attachments, so this never runs today).
    if (!(area == whole))
    {
        for (uint32_t i = 0; i < colorCount; ++i)
            if (rp.colorAttachments[i].texture != nil && rp.colorAttachments[i].loadAction != MTLLoadActionLoad)
                rp.colorAttachments[i].loadAction = MTLLoadActionLoad;
        if (rp.stencilAttachment.texture != nil && rp.stencilAttachment.loadAction != MTLLoadActionLoad)
            rp.stencilAttachment.loadAction = MTLLoadActionLoad;
    }

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    if (enc == nil) return;
    // The whole attachment: gl_FragCoord is its pixels.
    [enc setViewport:(MTLViewport) { 0.0, 0.0, double(width), double(height), 0.0, 1.0 }];

    bound_ = {};
    for (uint32_t k = 0; k < pass.drawCount; ++k)
        encodeDraw(enc, slot, packet, packet.draws[pass.firstDraw + k], area, serial);
    [enc endEncoding];
}

void PacketExecutor::encodeDraw(id<MTLRenderCommandEncoder> enc, FrameSlot& slot, const gpu::FramePacket& packet,
                                const gpu::Draw& draw, const gpu::IRect& area, uint64_t serial)
{
    if (draw.vertexCount == 0) return;
    PipelineSlot* ps = backend_.pipeline(draw.pipeline);
    if (ps == nullptr || !ps->state) return;   // a failed build: skipped
    ProgramSlot* prog = backend_.program(ps->program);
    if (prog == nullptr) return;

    const gpu::IRect scissor = intersect(draw.scissor, area);
    if (scissor.isEmpty()) return;

    // Everything a draw reads must be valid before anything is bound.
    for (uint32_t j = 0; j < draw.storageCount && j < gpu::kMaxDrawStorage; ++j)
        if (uint64_t(draw.storage[j].offset) + draw.storage[j].size > packet.storageData.size()) return;

    if (ps != bound_.pipeline)
    {
        [enc setRenderPipelineState:ps->state.get()];
        bound_.pipeline = ps;
    }
    if (ps->depthStencil && ps->depthStencil.get() != bound_.depthStencil)
    {
        [enc setDepthStencilState:ps->depthStencil.get()];
        bound_.depthStencil = ps->depthStencil.get();
    }
    if (int(draw.stencilRef) != bound_.stencilRef)
    {
        [enc setStencilReferenceValue:draw.stencilRef];
        bound_.stencilRef = draw.stencilRef;
    }
    if (!(scissor == bound_.scissor))
    {
        [enc setScissorRect:(MTLScissorRect) { NSUInteger(scissor.x), NSUInteger(scissor.y),
                                               NSUInteger(scissor.width), NSUInteger(scissor.height) }];
        bound_.scissor = scissor;
    }

    for (uint32_t i = 0; i < draw.textureCount && i < gpu::kMaxDrawTextures; ++i)
    {
        const auto& b = draw.textures[i];
        TextureSlot* t = backend_.texture(b.texture);
        if (t == nullptr || !t->texture) t = backend_.dummyTexture();
        if (t == nullptr) continue;
        t->lastUse = serial;
        const uint8_t ti = prog->textureIndex[i], si = prog->samplerIndex[i];
        id<MTLTexture> texture = t->texture.get();
        if (bound_.textures[ti] != texture)
        {
            [enc setFragmentTexture:texture atIndex:ti];
            bound_.textures[ti] = texture;
        }
        id<MTLSamplerState> sampler = backend_.sampler(b.sampler);
        if (bound_.samplers[si] != sampler)
        {
            [enc setFragmentSamplerState:sampler atIndex:si];
            bound_.samplers[si] = sampler;
        }
    }

    for (uint32_t j = 0; j < draw.storageCount && j < gpu::kMaxDrawStorage; ++j)
    {
        const uint8_t index = prog->storageIndex[j];
        const size_t offset = draw.storage[j].offset;
        if (bound_.buffers[index] != slot.storage.get())
        {
            [enc setFragmentBuffer:slot.storage.get() offset:offset atIndex:index];
            bound_.buffers[index] = slot.storage.get();
            bound_.offsets[index] = offset;
        }
        else if (bound_.offsets[index] != offset)
        {
            [enc setFragmentBufferOffset:offset atIndex:index];
            bound_.offsets[index] = offset;
        }
    }

    // Constants inline: the same bytes to both stages, at least the program's
    // whole block (a shorter range than the shader declares is zero-padded).
    if (prog->constantsBytes > 0 && !draw.constants.isEmpty()
        && uint64_t(draw.constants.offset) + draw.constants.size <= packet.constantData.size())
    {
        alignas(16) std::byte block[512] {};
        const uint32_t length = std::min<uint32_t>((std::max(prog->constantsBytes, draw.constants.size) + 15) & ~15u,
                                                   sizeof(block));
        if (!(draw.constants == bound_.constants) || length > bound_.constantsLength)
        {
            std::memcpy(block, packet.constantData.data() + draw.constants.offset, std::min<uint32_t>(draw.constants.size, length));
            [enc setVertexBytes:block length:length atIndex:kConstantsIndex];
            [enc setFragmentBytes:block length:length atIndex:kConstantsIndex];
            bound_.constants = draw.constants;
            bound_.constantsLength = length;
        }
    }

    uint32_t firstVertex = 0;
    const uint32_t stride = ps->desc.vertexLayout.stride;
    if (stride > 0)
    {
        if (!bound_.vertexBuffer)
        {
            [enc setVertexBuffer:slot.vertices.get() offset:0 atIndex:kVertexBufferIndex];
            bound_.vertexBuffer = true;
        }
        firstVertex = draw.vertices.offset / stride;   // the core aligns ranges to the stride
    }
    [enc drawPrimitives:ps->primitive vertexStart:firstVertex vertexCount:draw.vertexCount
          instanceCount:std::max(1u, draw.instanceCount)];
}

// =============================================================================
// A window's CAMetalLayer
// =============================================================================

LayerSurface::LayerSurface(std::shared_ptr<Backend> backend, CAMetalLayer* layer, gpu::Extent sizeHint)
    : backend_(std::move(backend)), layer_([layer retain]), exec_(*backend_)
{
    inFlight_ = dispatch_semaphore_create(kFrames);
    layer_.device             = backend_->device();
    layer_.pixelFormat        = MTLPixelFormatBGRA8Unorm;
    layer_.framebufferOnly    = YES;   // only ever rendered into (the present copy)
    layer_.maximumDrawableCount = 3;
    layer_.displaySyncEnabled = YES;
    layer_.opaque             = YES;   // as the Vulkan swapchain (opaque composite alpha)
    // sRGB, as the Vulkan swapchain's colour space: on a wide-gamut (P3)
    // display the pixels are colour-matched, not shown as display values.
    CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    layer_.colorspace = srgb;
    CGColorSpaceRelease(srgb);
    swap_ = backend_->createSwapTexture();

    gpu::Extent e = preferredExtent();
    if (e.width == 0 || e.height == 0) e = sizeHint;
    resize(e);
}

LayerSurface::~LayerSurface()
{
    drain(inFlight_, kFrames);
    drawable_ = {};
    backend_->destroyTexture(swap_);
    [layer_ release];
    dispatch_release(inFlight_);
}

gpu::Extent LayerSurface::preferredExtent() const
{
    // The layer's size in physical pixels, as the Vulkan surface reports it
    // (MoltenVK: bounds times contentsScale).
    const CGSize size  = layer_.bounds.size;
    const CGFloat scale = layer_.contentsScale;
    return { static_cast<uint32_t>(std::lround(size.width * scale)), static_cast<uint32_t>(std::lround(size.height * scale)) };
}

void LayerSurface::resize(gpu::Extent extent)
{
    if (extent.width == 0 || extent.height == 0) return;
    width_  = extent.width;
    height_ = extent.height;
    layer_.drawableSize = CGSizeMake(width_, height_);
}

bool LayerSurface::beginFrame(gpu::FrameInfo& info)
{
    if (frameBegun_ || layer_ == nil || backend_->isLost()) return false;
    // A frame slot whose last frame the GPU has finished.
    if (dispatch_semaphore_wait(inFlight_, dispatch_time(DISPATCH_TIME_NOW, (int64_t) NSEC_PER_SEC)) != 0)
        return false;
    id<MTLTexture> texture = nil;
    @autoreleasepool
    {
        id<CAMetalDrawable> drawable = [layer_ nextDrawable];
        if (drawable == nil)
        {
            dispatch_semaphore_signal(inFlight_);
            return false;
        }
        drawable_ = Strong<id<CAMetalDrawable>>::retain(drawable);
        texture = [drawable texture];
        backend_->setSwapTexture(swap_, texture);
    }
    info.output       = swap_;
    info.outputFormat = PixelFormat::BGRA8;
    info.extent       = { static_cast<uint32_t>([texture width]), static_cast<uint32_t>([texture height]) };
    frameBegun_ = true;
    return true;
}

void LayerSurface::submit(const gpu::FramePacket& packet)
{
    if (!frameBegun_) return;
    frameBegun_ = false;
    @autoreleasepool
    {
        std::lock_guard<std::mutex> lk(backend_->submitMutex());
        const uint64_t serial = backend_->beginSubmission();
        id<MTLCommandBuffer> cb = [backend_->queue() commandBuffer];
        backend_->encodePendingWrites(cb, serial);
        exec_.encode(cb, slots_[current_], packet, serial);
        [cb presentDrawable:drawable_.get()];
        backend_->commit(cb, serial, signalOnDone(inFlight_));
    }
    drawable_ = {};          // the command buffer holds it until it is presented
    backend_->setSwapTexture(swap_, nil);
    current_ = (current_ + 1) % kFrames;
}

// =============================================================================
// An offscreen texture
// =============================================================================

OffscreenSurface::OffscreenSurface(std::shared_ptr<Backend> backend, gpu::TextureId target)
    : backend_(std::move(backend)), target_(target), exec_(*backend_)
{
    inFlight_ = dispatch_semaphore_create(1);
}

OffscreenSurface::~OffscreenSurface()
{
    drain(inFlight_, 1);
    dispatch_release(inFlight_);
}

gpu::Extent OffscreenSurface::extent() const
{
    if (auto* t = backend_->texture(target_)) return { t->width, t->height };
    return {};
}

bool OffscreenSurface::beginFrame(gpu::FrameInfo& info)
{
    TextureSlot* t = backend_->texture(target_);
    if (frameBegun_ || t == nullptr || backend_->isLost()) return false;
    if (dispatch_semaphore_wait(inFlight_, dispatch_time(DISPATCH_TIME_NOW, (int64_t) NSEC_PER_SEC)) != 0)
        return false;
    info.output       = target_;
    info.outputFormat = t->pixelFormat;
    info.extent       = { t->width, t->height };
    frameBegun_ = true;
    return true;
}

void OffscreenSurface::submit(const gpu::FramePacket& packet)
{
    if (!frameBegun_) return;
    frameBegun_ = false;
    @autoreleasepool
    {
        std::lock_guard<std::mutex> lk(backend_->submitMutex());
        const uint64_t serial = backend_->beginSubmission();
        id<MTLCommandBuffer> cb = [backend_->queue() commandBuffer];
        backend_->encodePendingWrites(cb, serial);
        exec_.encode(cb, slot_, packet, serial);
        backend_->commit(cb, serial, signalOnDone(inFlight_));
    }
}

} // namespace jvk::metal
