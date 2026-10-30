/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanSurface.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::vulkan
{

namespace
{

// Pins the calling thread to the surface window's DPI-awareness context for
// the duration of any driver call whose result depends on the window's client
// rect: capability queries (currentExtent), acquire and present. On Windows
// the ICD resolves GetClientRect on the CALLING thread, and Win32 coordinate
// APIs DPI-virtualize their results whenever the thread's context differs
// from the window's — so without pinning, the "authoritative" currentExtent
// changes value depending on who asks (message thread running in the host's
// context vs. the render worker in the process-default context), the two
// callers resize the swapchain to different sizes, every acquire reports
// SUBOPTIMAL/OUT_OF_DATE, and the swapchain never converges (black screen in
// hosts that run plugin UIs DPI-virtualized, e.g. Ableton's auto-scale).
// No-op off Windows, without a native window, or pre-Win10-1607.
struct SurfaceDpiScope
{
#ifdef _WIN32
    core::windows::dpi::ScopedWindowContext scope;
    explicit SurfaceDpiScope(void* nativeWindow) : scope(nativeWindow) {}
#else
    explicit SurfaceDpiScope(void*) {}
#endif
};

gpu::IRect intersect(const gpu::IRect& a, const gpu::IRect& b)
{
    const int32_t x0 = std::max(a.x, b.x), y0 = std::max(a.y, b.y);
    const int32_t x1 = std::min(a.x + a.width, b.x + b.width), y1 = std::min(a.y + a.height, b.y + b.height);
    if (x1 <= x0 || y1 <= y0) return {};
    return { x0, y0, x1 - x0, y1 - y0 };
}

// Per-slot command buffer and fence (signalled, so the first wait passes).
bool createSlot(VkDevice d, VkCommandPool pool, FrameSlot& slot, bool withSemaphore)
{
    VkCommandBufferAllocateInfo ai {};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d, &ai, &slot.cmd) != VK_SUCCESS) slot.cmd = VK_NULL_HANDLE;

    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(d, &fci, nullptr, &slot.fence) != VK_SUCCESS) slot.fence = VK_NULL_HANDLE;

    if (withSemaphore)
    {
        VkSemaphoreCreateInfo sci {};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(d, &sci, nullptr, &slot.imageAvailable) != VK_SUCCESS)
            slot.imageAvailable = VK_NULL_HANDLE;
    }
    return slot.cmd != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE
        && (!withSemaphore || slot.imageAvailable != VK_NULL_HANDLE);
}

VkCommandPool createPool(Device& device)
{
    VkCommandPoolCreateInfo ci {};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = device.graphicsFamily();
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device.device(), &ci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pool;
}

bool beginCommands(VkCommandBuffer cmd)
{
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
}

} // namespace

// =============================================================================
// PacketExecutor
// =============================================================================

void PacketExecutor::beginSlot(FrameSlot& slot)
{
    VkDevice d = backend_.device().device();
    for (auto pool : slot.pools) vkResetDescriptorPool(d, pool, 0);
    slot.pool = 0;
    slot.sets.clear();
}

void PacketExecutor::destroySlot(FrameSlot& slot)
{
    VkDevice d = backend_.device().device();
    for (auto pool : slot.pools) vkDestroyDescriptorPool(d, pool, nullptr);
    slot.pools.clear();
    slot.sets.clear();
    destroyHostBuffer(backend_.device().allocator(), slot.vertices);
    destroyHostBuffer(backend_.device().allocator(), slot.storage);
    if (slot.fence != VK_NULL_HANDLE)          vkDestroyFence(d, slot.fence, nullptr);
    if (slot.imageAvailable != VK_NULL_HANDLE) vkDestroySemaphore(d, slot.imageAvailable, nullptr);
    slot.fence = VK_NULL_HANDLE;
    slot.imageAvailable = VK_NULL_HANDLE;
    slot.cmd = VK_NULL_HANDLE;   // freed with its pool
}

bool PacketExecutor::ensure(HostBuffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage)
{
    // Grown, never shrunk: the steady state allocates nothing. The old buffer
    // is free to go now: this slot's fence has signalled (beginFrame waited).
    if (buffer && buffer.size >= size) return true;
    const VkDeviceSize grown = std::max<VkDeviceSize>({ size, buffer.size * 2, VkDeviceSize(1u << 20) });
    destroyHostBuffer(backend_.device().allocator(), buffer);
    buffer = createHostBuffer(backend_.device().allocator(), grown, usage);
    return static_cast<bool>(buffer);
}

void PacketExecutor::record(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet)
{
    // One copy of each stream into this slot's buffers; constants go
    // straight from the packet as push constants.
    if (!ensure(slot.vertices, packet.vertexData.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        || !ensure(slot.storage, std::max<size_t>(packet.storageData.size(), 256),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
        return;
    if (!packet.vertexData.empty())
        std::memcpy(slot.vertices.mapped, packet.vertexData.data(), packet.vertexData.size());
    if (!packet.storageData.empty())
        std::memcpy(slot.storage.mapped, packet.storageData.data(), packet.storageData.size());

    // A buffer a compute stream wrote (its fence waited on by the host) is
    // available; this makes it visible to the frame's shaders.
    bool external = !packet.dispatches.empty();
    for (size_t i = 0; i < packet.draws.size() && !external; ++i)
        for (uint32_t j = 0; j < packet.draws[i].storageCount && j < gpu::kMaxDrawStorage; ++j)
            external |= static_cast<bool>(packet.draws[i].storageBuffer[j]);
    if (external)
    {
        VkMemoryBarrier mb {};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        VkPipelineStageFlags dst = kShaderStages;
        if (backend_.caps().compute.frameDispatch) dst |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    for (const auto& pass : packet.passes)
    {
        if (pass.kind == gpu::PassKind::Compute) recordComputePass(cmd, slot, packet, pass);
        else                                     recordPass(cmd, slot, packet, pass);
    }
}

void PacketExecutor::recordComputePass(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet,
                                       const gpu::Pass& pass)
{
    if (!backend_.caps().compute.frameDispatch
        || pass.firstDispatch > packet.dispatches.size() || pass.dispatchCount > packet.dispatches.size() - pass.firstDispatch)
        return;

    // Dispatches touch buffers alone, so one memory barrier each way orders
    // them against the frame: every draw and dispatch before finishes (and
    // what it wrote is seen) before the first runs; each sees the one before
    // it; everything after sees the last.
    auto barrier = [cmd](VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags dstAccess)
    {
        VkMemoryBarrier mb {};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    constexpr VkAccessFlags reads = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    const VkPipelineStageFlags all = kShaderStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    barrier(all, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, reads | VK_ACCESS_SHADER_WRITE_BIT);
    for (uint32_t k = 0; k < pass.dispatchCount; ++k)
    {
        if (k > 0)
            barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, reads | VK_ACCESS_SHADER_WRITE_BIT);
        dispatcher_.record(backend_, cmd, packet.dispatches[pass.firstDispatch + k], packet.constantData,
                           [this, &slot](const KernelSlot& kernel) { return allocateSet(slot, kernel.setLayout); });
    }
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, all, reads | VK_ACCESS_SHADER_WRITE_BIT);
}

void PacketExecutor::recordPass(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet,
                                const gpu::Pass& pass)
{
    if (pass.firstDraw > packet.draws.size() || pass.drawCount > packet.draws.size() - pass.firstDraw)
        return;

    // ---- Attachments: the render pass's shape, the framebuffer's views ------
    RenderPassKey key;
    key.colorCount = std::min<uint8_t>(pass.colorCount, uint8_t(gpu::kMaxColorAttachments));
    TextureSlot* colors[gpu::kMaxColorAttachments] {};
    uint32_t     handles[gpu::kMaxColorAttachments + 1] {};
    uint32_t     count = 0, width = 0, height = 0;
    for (uint32_t i = 0; i < key.colorCount; ++i)
    {
        const auto& a = pass.color[i];
        if (!a.texture) continue;   // an unused slot (AttachmentFormats::unusedMask)
        colors[i] = backend_.texture(a.texture);
        if (colors[i] == nullptr) return;   // a dead attachment: nothing valid to draw into
        key.color[i]      = colors[i]->format;
        key.colorLoad[i]  = static_cast<uint8_t>(a.load);
        key.colorStore[i] = static_cast<uint8_t>(a.store);
        handles[count++]  = a.texture.value;
        if (width == 0) { width = colors[i]->width; height = colors[i]->height; }
    }
    TextureSlot* ds = nullptr;
    if (pass.depthStencil.texture)
    {
        ds = backend_.texture(pass.depthStencil.texture);
        if (ds == nullptr) return;
        key.depthStencil = ds->format;
        key.dsLoad       = static_cast<uint8_t>(pass.depthStencil.load);
        key.dsStore      = static_cast<uint8_t>(pass.depthStencil.store);
        handles[count++] = pass.depthStencil.texture.value;
        if (width == 0) { width = ds->width; height = ds->height; }
    }
    if (count == 0 || width == 0 || height == 0) return;

    const gpu::IRect area = intersect(pass.renderArea, { 0, 0, int32_t(width), int32_t(height) });
    if (area.isEmpty()) return;

    // ---- Barriers: every attachment and every texture a draw samples --------
    // A Clear/DontCare load may discard the old contents (UNDEFINED) only when
    // the pass covers the whole texture: pixels outside the render area, and
    // other mips and layers, are kept (FramePacket's contract).
    auto discards = [&](const TextureSlot& t, gpu::LoadOp load) {
        return load != gpu::LoadOp::Load && t.mips == 1 && t.layers == 1
            && area.x == 0 && area.y == 0
            && uint32_t(area.width) == t.width && uint32_t(area.height) == t.height;
    };
    batch_.clear();
    for (uint32_t i = 0; i < key.colorCount; ++i)
        if (colors[i] != nullptr)
            Backend::transition(batch_, *colors[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                discards(*colors[i], pass.color[i].load));
    if (ds != nullptr)
        Backend::transition(batch_, *ds, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            discards(*ds, pass.depthStencil.load));
    TextureSlot* dummy = backend_.dummyTexture();
    for (uint32_t k = 0; k < pass.drawCount; ++k)
    {
        const auto& draw = packet.draws[pass.firstDraw + k];
        const uint32_t n = std::min<uint32_t>(draw.textureCount, gpu::kMaxDrawTextures);
        for (uint32_t i = 0; i < n; ++i)
        {
            TextureSlot* t = backend_.texture(draw.textures[i].texture);
            if (t == nullptr || t->kind != gpu::TextureKind::Color) t = dummy;
            if (t != nullptr)
                Backend::transition(batch_, *t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    kShaderStages, VK_ACCESS_SHADER_READ_BIT, false);
        }
    }
    batch_.record(cmd);

    // ---- Render pass and framebuffer -------------------------------------------
    VkRenderPass rp = backend_.renderPass(key);
    RenderPassKey compat = key;   // loads/stores zeroed: the framebuffer's (compatible) pass
    std::fill(std::begin(compat.colorLoad), std::end(compat.colorLoad), uint8_t(0));
    std::fill(std::begin(compat.colorStore), std::end(compat.colorStore), uint8_t(0));
    compat.dsLoad = compat.dsStore = 0;
    VkRenderPass crp = backend_.renderPass(compat);
    VkFramebuffer fb = crp != VK_NULL_HANDLE
        ? backend_.framebuffer(crp, std::span<const uint32_t>(handles, count), width, height)
        : VK_NULL_HANDLE;
    if (rp == VK_NULL_HANDLE || fb == VK_NULL_HANDLE) return;

    // Clear values by ATTACHMENT (unused slots have none).
    VkClearValue clears[gpu::kMaxColorAttachments + 1] {};
    uint32_t c = 0;
    for (uint32_t i = 0; i < key.colorCount; ++i)
    {
        if (colors[i] == nullptr) continue;
        const auto& cc = pass.color[i].clear;
        clears[c++].color = {{ cc.r, cc.g, cc.b, cc.a }};
    }
    if (ds != nullptr)
        clears[c++].depthStencil = { 1.0f, pass.depthStencil.clearStencil };

    VkRenderPassBeginInfo rpbi {};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = rp;
    rpbi.framebuffer     = fb;
    rpbi.renderArea      = { { area.x, area.y }, { uint32_t(area.width), uint32_t(area.height) } };
    rpbi.clearValueCount = c;
    rpbi.pClearValues    = clears;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    // The whole attachment: gl_FragCoord is its pixels.
    VkViewport vp {};
    vp.width    = static_cast<float>(width);
    vp.height   = static_cast<float>(height);
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    bound_ = {};
    for (uint32_t k = 0; k < pass.drawCount; ++k)
        recordDraw(cmd, slot, packet, packet.draws[pass.firstDraw + k], area);

    vkCmdEndRenderPass(cmd);
}

void PacketExecutor::recordDraw(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet,
                                const gpu::Draw& draw, const gpu::IRect& area)
{
    if (draw.vertexCount == 0) return;
    PipelineSlot* ps = backend_.pipeline(draw.pipeline);
    if (ps == nullptr || ps->pipeline == VK_NULL_HANDLE) return;   // failed build: skipped, never bound null
    ProgramSlot* prog = backend_.program(ps->program);
    if (prog == nullptr || prog->setLayouts.size() > std::size(bound_.sets)) return;

    const gpu::IRect scissor = intersect(draw.scissor, area);
    if (scissor.isEmpty()) return;

    if (ps->pipeline != bound_.pipeline)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ps->pipeline);
        bound_.pipeline = ps->pipeline;
    }
    if (prog->layout != bound_.layout)
    {
        // Sets and constants bound under another layout don't carry over.
        bound_.layout = prog->layout;
        std::fill(std::begin(bound_.sets), std::end(bound_.sets), VkDescriptorSet(VK_NULL_HANDLE));
        bound_.constants = { ~0u, 0 };
    }

    for (uint32_t s = 0; s < prog->setLayouts.size(); ++s)
    {
        if (!prog->ownsSetLayout[s]) continue;   // an empty set: nothing to bind
        VkDescriptorSet set = descriptorSet(slot, *prog, ps->program, s, draw);
        if (set == VK_NULL_HANDLE) return;
        if (set != bound_.sets[s])
        {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, prog->layout, s, 1, &set, 0, nullptr);
            bound_.sets[s] = set;
        }
    }

    if (prog->constantsBytes > 0 && !draw.constants.isEmpty()
        && uint64_t(draw.constants.offset) + draw.constants.size <= packet.constantData.size()
        && !(draw.constants == bound_.constants))
    {
        const uint32_t size = std::min<uint32_t>(draw.constants.size, prog->constantsBytes) & ~3u;
        if (size > 0)
            vkCmdPushConstants(cmd, prog->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, size, packet.constantData.data() + draw.constants.offset);
        bound_.constants = draw.constants;
    }

    if (!(scissor == bound_.scissor))
    {
        const VkRect2D sc { { scissor.x, scissor.y }, { uint32_t(scissor.width), uint32_t(scissor.height) } };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        bound_.scissor = scissor;
    }
    // Every pipeline declares the reference dynamic: it must be set before
    // any draw, stencil test or not.
    if (int(draw.stencilRef) != bound_.stencilRef)
    {
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, draw.stencilRef);
        bound_.stencilRef = draw.stencilRef;
    }

    uint32_t firstVertex = 0;
    const uint32_t stride = ps->desc.vertexLayout.stride;
    if (stride > 0)
    {
        if (!bound_.vertexBuffer)
        {
            const VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &slot.vertices.buffer, &zero);
            bound_.vertexBuffer = true;
        }
        firstVertex = draw.vertices.offset / stride;   // the core aligns ranges to the stride
    }
    vkCmdDraw(cmd, draw.vertexCount, std::max(1u, draw.instanceCount), firstVertex, 0);
}

VkDescriptorSet PacketExecutor::allocateSet(FrameSlot& slot, VkDescriptorSetLayout layout)
{
    VkDevice d = backend_.device().device();
    for (;;)
    {
        if (slot.pool == slot.pools.size())
        {
            // Sized for a frame of jvk's draws; another pool joins when one fills.
            const VkDescriptorPoolSize sizes[] = {
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1024 },
                { VK_DESCRIPTOR_TYPE_SAMPLER,                1024 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         512 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         256 },
            };
            VkDescriptorPoolCreateInfo ci {};
            ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets       = 512;
            ci.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
            ci.pPoolSizes    = sizes;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(d, &ci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            slot.pools.push_back(pool);
        }
        VkDescriptorSetAllocateInfo ai {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = slot.pools[slot.pool];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult r = vkAllocateDescriptorSets(d, &ai, &set);
        if (r == VK_SUCCESS) return set;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) return VK_NULL_HANDLE;
        ++slot.pool;
    }
}

VkDescriptorSet PacketExecutor::descriptorSet(FrameSlot& slot, const ProgramSlot& program, uint32_t programHandle,
                                              uint32_t set, const gpu::Draw& draw)
{
    DescriptorKey key;
    key.program      = programHandle;
    key.set          = static_cast<uint8_t>(set);
    key.textureCount = std::min<uint8_t>(draw.textureCount, uint8_t(gpu::kMaxDrawTextures));
    key.storageCount = std::min<uint8_t>(draw.storageCount, uint8_t(gpu::kMaxDrawStorage));
    for (uint32_t i = 0; i < key.textureCount; ++i) key.textures[i] = draw.textures[i];
    for (uint32_t i = 0; i < key.storageCount; ++i)
    {
        key.storage[i]       = draw.storage[i];
        key.storageBuffer[i] = draw.storageBuffer[i];
    }

    // Newest first: consecutive draws mostly bind what the last one did.
    for (size_t i = slot.sets.size(); i-- > 0;)
        if (slot.sets[i].key == key) return slot.sets[i].set;

    VkDescriptorSet ds = allocateSet(slot, program.setLayouts[set]);
    if (ds == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    TextureSlot* dummy = backend_.dummyTexture();
    writes_.clear();
    imageInfos_.clear();
    bufferInfos_.clear();
    // Reserved up front: the writes point into these.
    imageInfos_.reserve(program.bindings.size());
    bufferInfos_.reserve(program.bindings.size());
    for (const auto& b : program.bindings)
    {
        if (b.set != set) continue;
        VkWriteDescriptorSet w {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = ds;
        w.dstBinding      = b.binding;
        w.descriptorCount = 1;

        const bool isBuffer = b.kind == gpu::BindingKind::StorageBuffer || b.kind == gpu::BindingKind::UniformBuffer;
        if (isBuffer)
        {
            VkDescriptorBufferInfo bi {};
            bi.buffer = slot.storage.buffer;
            // A compute buffer bound in place, else the frame's storage stream.
            const BufferSlot* own = b.slot < key.storageCount ? backend_.buffer(key.storageBuffer[b.slot]) : nullptr;
            const VkDeviceSize limit = own != nullptr ? own->size : slot.storage.size;
            if (b.slot < key.storageCount && !key.storage[b.slot].isEmpty()
                && (own != nullptr || !key.storageBuffer[b.slot])
                && uint64_t(key.storage[b.slot].offset) + key.storage[b.slot].size <= limit)
            {
                if (own != nullptr) bi.buffer = own->buffer;
                bi.offset = key.storage[b.slot].offset;
                bi.range  = key.storage[b.slot].size;
            }
            else
            {
                bi.offset = 0;   // nothing named: a small valid range, never read
                bi.range  = 256;
            }
            bufferInfos_.push_back(bi);
            w.descriptorType = b.kind == gpu::BindingKind::UniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                         : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo    = &bufferInfos_.back();
        }
        else
        {
            const gpu::TextureBinding tb = b.slot < key.textureCount ? key.textures[b.slot] : gpu::TextureBinding {};
            TextureSlot* t = backend_.texture(tb.texture);
            if (t == nullptr || t->kind != gpu::TextureKind::Color) t = dummy;
            if (t == nullptr) return VK_NULL_HANDLE;
            VkDescriptorImageInfo ii {};
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            switch (b.kind)
            {
                case gpu::BindingKind::Texture:
                    ii.imageView     = t->view;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    break;
                case gpu::BindingKind::Sampler:
                    ii.sampler       = backend_.sampler(tb.sampler);
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    break;
                default:
                    ii.sampler       = backend_.sampler(tb.sampler);
                    ii.imageView     = t->view;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    break;
            }
            imageInfos_.push_back(ii);
            w.pImageInfo = &imageInfos_.back();
        }
        writes_.push_back(w);
    }
    if (!writes_.empty())
        vkUpdateDescriptorSets(backend_.device().device(), static_cast<uint32_t>(writes_.size()), writes_.data(), 0, nullptr);

    slot.sets.push_back({ key, ds });
    return ds;
}

// =============================================================================
// SwapchainSurface
// =============================================================================

SwapchainSurface::SwapchainSurface(std::shared_ptr<Backend> backend, VkSurfaceKHR surface,
                                   gpu::Extent sizeHint, void* nativeWindow)
    : backend_(std::move(backend)), surface_(surface), nativeWindow_(nativeWindow),
      width_(sizeHint.width), height_(sizeHint.height), exec_(*backend_)
{
    auto& device = backend_->device();
    pool_ = createPool(device);
    if (pool_ != VK_NULL_HANDLE)
        for (auto& slot : slots_)
            createSlot(device.device(), pool_, slot, true);

    const juce::ScopedLock lk(device.queueLock());
    createSwapchain();
}

SwapchainSurface::~SwapchainSurface()
{
    auto& device = backend_->device();
    VkDevice d = device.device();
    const juce::ScopedLock lk(device.queueLock());
    device.waitIdle();
    for (auto& slot : slots_)
    {
        backend_->fenceDone(slot.fence);
        exec_.destroySlot(slot);
    }
    destroySwapchain();
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(d, pool_, nullptr);
    if (surface_ != VK_NULL_HANDLE) vkDestroySurfaceKHR(device.instance(), surface_, nullptr);
}

gpu::Extent SwapchainSurface::preferredExtent() const
{
    SurfaceDpiScope dpiScope(nativeWindow_);
    VkSurfaceCapabilitiesKHR caps {};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(backend_->device().physicalDevice(), surface_, &caps) != VK_SUCCESS)
        return { width_, height_ };
    return { caps.currentExtent.width, caps.currentExtent.height };   // 0xFFFFFFFF: no preference
}

bool SwapchainSurface::canBegin() const
{
    return swapchain_ != VK_NULL_HANDLE && !surfaceLost_ && !backend_->isLost();
}

void SwapchainSurface::revive()
{
    if (swapchain_ == VK_NULL_HANDLE && !surfaceLost_ && !backend_->isLost())
        resize({ width_, height_ });
}

void SwapchainSurface::resize(gpu::Extent extent)
{
    const juce::ScopedLock lk(backend_->device().queueLock());
    rebuild(extent.width, extent.height);
}

void SwapchainSurface::rebuild(uint32_t width, uint32_t height)
{
    // vkDeviceWaitIdle is queue use: the submit lock (held by the caller,
    // reentrant) keeps it from racing another surface's submit.
    backend_->device().waitIdle();
    width_  = width;
    height_ = height;
    destroySwapchain();
    createSwapchain();
}

void SwapchainSurface::createSwapchain()
{
    SurfaceDpiScope dpiScope(nativeWindow_);
    auto& device = backend_->device();
    VkDevice d = device.device();
    VkPhysicalDevice pd = device.physicalDevice();
    if (d == VK_NULL_HANDLE || surface_ == VK_NULL_HANDLE) return;

    // The device's queue must be able to present to this surface (the
    // physical device was picked by its graphics queue alone).
    VkBool32 presentable = VK_FALSE;
    if (vkGetPhysicalDeviceSurfaceSupportKHR(pd, device.presentFamily(), surface_, &presentable) != VK_SUCCESS
        || presentable != VK_TRUE)
    {
        diag::log("vulkan: the device's queue can't present to this window");
        return;
    }

    VkSurfaceCapabilitiesKHR caps {};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface_, &caps) != VK_SUCCESS)
        return;

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface_, &fmtCount, fmts.data());
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (auto& f : fmts)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { colorSpace = f.colorSpace; break; }

    // The surface's currentExtent is the AUTHORITATIVE physical size of the
    // window/layer: on Win32 the child HWND's client area in device pixels,
    // on Metal the CAMetalLayer's drawableSize. It already folds in the host
    // content scale and per-monitor DPI, which getPlatformScaleFactor() does
    // not report reliably. Only the "no preferred size" sentinel (some
    // Wayland) or a transient 0 falls back to the hint, clamped.
    VkExtent2D ext;
    if (caps.currentExtent.width != 0xFFFFFFFFu && caps.currentExtent.width != 0 && caps.currentExtent.height != 0)
    {
        ext = caps.currentExtent;
    }
    else
    {
        ext.width  = std::clamp(width_,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        ext.height = std::clamp(height_, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    // A minimised window has no extent: no swapchain until it comes back
    // (revive retries; the sizes keep their last real values).
    if (ext.width == 0 || ext.height == 0) return;
    width_  = ext.width;
    height_ = ext.height;

    // At least 3 images (MoltenVK guidance): fewer and Direct-to-Display
    // drawable holds stall acquire.
    uint32_t imageCount = std::max(caps.minImageCount + 1, 3u);
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci {};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface_;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = VK_FORMAT_B8G8R8A8_UNORM;
    ci.imageColorSpace  = colorSpace;
    ci.imageExtent      = ext;
    ci.imageArrayLayers = 1;
    // The core's last pass draws into the swap image.
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    const VkCompositeAlphaFlagBitsKHR prefs[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
    };
    for (auto p : prefs) if (caps.supportedCompositeAlpha & p) { ci.compositeAlpha = p; break; }
    ci.presentMode  = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped      = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(d, &ci, nullptr, &swapchain_) != VK_SUCCESS)
    {
        swapchain_ = VK_NULL_HANDLE;   // beginFrame returns false; revive/resize retries
        return;
    }

    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, nullptr);
    swapImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(d, swapchain_, &imageCount, swapImages_.data());

    VkSemaphoreCreateInfo sci {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkImage image : swapImages_)
    {
        swapTextures_.push_back(backend_->adoptImage(image, VK_FORMAT_B8G8R8A8_UNORM, PixelFormat::BGRA8,
                                                     width_, height_));
        VkSemaphore s = VK_NULL_HANDLE;
        vkCreateSemaphore(d, &sci, nullptr, &s);
        renderFinished_.push_back(s);
    }
}

void SwapchainSurface::destroySwapchain()
{
    VkDevice d = backend_->device().device();
    // The GPU is idle (callers wait first), so the swap textures go now,
    // views and framebuffers with them, before their images do.
    for (auto t : swapTextures_) backend_->destroyTextureNow(t);
    swapTextures_.clear();
    for (auto s : renderFinished_) if (s != VK_NULL_HANDLE) vkDestroySemaphore(d, s, nullptr);
    renderFinished_.clear();
    swapImages_.clear();
    if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(d, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
    frameBegun_ = false;
}

bool SwapchainSurface::beginFrame(gpu::FrameInfo& info)
{
    // An image acquired last time and never submitted must still be
    // presented: its semaphore is pending, and acquiring with it again is
    // invalid. An empty frame presents it.
    if (frameBegun_) submit({});

    SurfaceDpiScope dpiScope(nativeWindow_);
    auto& device = backend_->device();
    VkDevice d = device.device();
    if (backend_->isLost() || swapchain_ == VK_NULL_HANDLE) return false;

    FrameSlot& slot = slots_[current_];
    if (slot.cmd == VK_NULL_HANDLE) return false;

    // Finite: a hung driver or a fence that never signals (failed submit)
    // must not wedge the submitting thread forever. One timeout is not a dead
    // device (a DAW holding the GPU, a debugger, a suspended process); five
    // in a row is, and the latch is sticky for everyone sharing it.
    const VkResult waited = vkWaitForFences(d, 1, &slot.fence, VK_TRUE, 2'000'000'000ull);
    if (waited != VK_SUCCESS)
    {
        if (waited == VK_ERROR_DEVICE_LOST || (waited == VK_TIMEOUT && ++fenceTimeouts_ >= kMaxFenceTimeouts))
            device.markLost();
        return false;
    }
    fenceTimeouts_ = 0;
    {
        const juce::ScopedLock lk(device.queueLock());
        backend_->fenceDone(slot.fence);
    }

    uint32_t imageIndex = 0;
    const VkResult acquired = vkAcquireNextImageKHR(d, swapchain_, UINT64_MAX, slot.imageAvailable,
                                                    VK_NULL_HANDLE, &imageIndex);
    // OUT_OF_DATE: nothing was acquired and the semaphore isn't signalled, so
    // dropping the frame is safe; rebuild for the next. SUBOPTIMAL proceeds:
    // an image WAS acquired and must be presented (a pending signal on the
    // semaphore would break the next acquire); submit() rebuilds after.
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize({ width_, height_ });
        return false;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
    {
        if (acquired == VK_ERROR_DEVICE_LOST) device.markLost();
        // The host destroyed our parent: no swapchain on this surface will work
        // again. canBegin() goes false and stays false, so the editor falls back.
        if (acquired == VK_ERROR_SURFACE_LOST_KHR) surfaceLost_ = true;
        return false;   // no image, nothing signalled
    }
    // The fence is reset by submit(), so a frame begun and never submitted
    // leaves it signalled rather than waited on forever.
    exec_.beginSlot(slot);
    imageIndex_ = imageIndex;
    frameBegun_ = true;

    info.output       = swapTextures_[imageIndex];
    info.outputFormat = PixelFormat::BGRA8;
    info.extent       = { width_, height_ };
    return true;
}

void SwapchainSurface::submit(const gpu::FramePacket& packet)
{
    if (!frameBegun_) return;
    frameBegun_ = false;
    SurfaceDpiScope dpiScope(nativeWindow_);
    auto& device = backend_->device();
    FrameSlot& slot = slots_[current_];

    const juce::ScopedLock lk(device.queueLock());
    backend_->collect();
    const uint64_t serial = backend_->beginSubmission();
    if (!beginCommands(slot.cmd)) { device.markLost(); return; }

    backend_->recordPendingWrites(slot.cmd, serial);

    // A fresh acquire: the presentation engine is done with it once the
    // semaphore (waited at COLOR_ATTACHMENT_OUTPUT) signals, and its contents
    // are whatever they were.
    TextureSlot* out = backend_->texture(swapTextures_[imageIndex_]);
    if (out != nullptr) out->state = { VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0 };

    exec_.record(slot.cmd, slot, packet);

    if (out != nullptr && out->state.layout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        // No pass drew into it (an empty frame): present black, never
        // whatever the image held. The barrier's source stage is the one the
        // acquire semaphore is waited at, so the clear chains after it.
        Backend::transition(present_, *out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true);
        present_.record(slot.cmd);
        const VkClearColorValue black {};
        const VkImageSubresourceRange all { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(slot.cmd, out->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    }
    if (out != nullptr)
    {
        Backend::transition(present_, *out, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, false);
        present_.record(slot.cmd);
    }
    vkEndCommandBuffer(slot.cmd);

    vkResetFences(device.device(), 1, &slot.fence);
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphore finished = renderFinished_[imageIndex_];
    VkSubmitInfo si {};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &slot.imageAvailable;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &slot.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &finished;

    // A failed submit leaves the fence unsignalled forever: the device is
    // treated as lost, and beginFrame's guard keeps anything from waiting on it.
    if (vkQueueSubmit(device.graphicsQueue(), 1, &si, slot.fence) != VK_SUCCESS)
    {
        device.markLost();
        current_ = (current_ + 1) % kFrames;
        return;
    }
    backend_->submitted(serial, slot.fence);

    VkPresentInfoKHR pi {};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &finished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &imageIndex_;
    const VkResult presented = vkQueuePresentKHR(device.presentQueue(), &pi);
    current_ = (current_ + 1) % kFrames;

    // After present is the safe place to rebuild: the image and its
    // semaphores are consumed. SUBOPTIMAL rebuilds only when the size really
    // changed (drivers report it for transform hints a rebuild won't cure,
    // and a rebuild is a device-wait-idle).
    if (presented == VK_ERROR_OUT_OF_DATE_KHR)
    {
        rebuild(width_, height_);
    }
    else if (presented == VK_SUBOPTIMAL_KHR)
    {
        const auto cur = preferredExtent();
        if (cur.width != 0xFFFFFFFFu && cur.width != 0 && cur.height != 0
            && (cur.width != width_ || cur.height != height_))
            rebuild(width_, height_);
    }
    else if (presented == VK_ERROR_DEVICE_LOST)
    {
        device.markLost();
    }
    else if (presented == VK_ERROR_SURFACE_LOST_KHR)
    {
        surfaceLost_ = true;
    }
}

// =============================================================================
// OffscreenSurface
// =============================================================================

OffscreenSurface::OffscreenSurface(std::shared_ptr<Backend> backend, gpu::TextureId target)
    : backend_(std::move(backend)), target_(target), exec_(*backend_)
{
    auto& device = backend_->device();
    pool_ = createPool(device);
    if (pool_ != VK_NULL_HANDLE)
        createSlot(device.device(), pool_, slot_, false);
}

OffscreenSurface::~OffscreenSurface()
{
    auto& device = backend_->device();
    VkDevice d = device.device();
    // Its frame may still be on the GPU: never free what it reads before it is
    // done (a timed-out wait falls back to the whole device, which returns even
    // when the device is lost).
    if (slot_.fence != VK_NULL_HANDLE
        && vkWaitForFences(d, 1, &slot_.fence, VK_TRUE, 2'000'000'000ull) != VK_SUCCESS)
    {
        const juce::ScopedLock lk(device.queueLock());
        device.waitIdle();
    }
    const juce::ScopedLock lk(device.queueLock());
    backend_->fenceDone(slot_.fence);
    exec_.destroySlot(slot_);
    if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(d, pool_, nullptr);
}

gpu::Extent OffscreenSurface::extent() const
{
    if (TextureSlot* t = backend_->texture(target_)) return { t->width, t->height };
    return {};
}

bool OffscreenSurface::beginFrame(gpu::FrameInfo& info)
{
    frameBegun_ = false;
    auto& device = backend_->device();
    TextureSlot* t = backend_->texture(target_);
    if (t == nullptr || backend_->isLost() || !valid()) return false;

    const VkResult waited = vkWaitForFences(device.device(), 1, &slot_.fence, VK_TRUE, 2'000'000'000ull);
    if (waited != VK_SUCCESS)
    {
        if (waited == VK_ERROR_DEVICE_LOST) device.markLost();
        return false;
    }
    {
        const juce::ScopedLock lk(device.queueLock());
        backend_->fenceDone(slot_.fence);
    }
    exec_.beginSlot(slot_);   // the fence is reset by submit()
    frameBegun_ = true;

    info.output       = target_;
    info.outputFormat = t->pixelFormat;
    info.extent       = { t->width, t->height };
    return true;
}

void OffscreenSurface::submit(const gpu::FramePacket& packet)
{
    if (!frameBegun_) return;
    frameBegun_ = false;
    auto& device = backend_->device();

    const juce::ScopedLock lk(device.queueLock());
    backend_->collect();
    const uint64_t serial = backend_->beginSubmission();
    if (!beginCommands(slot_.cmd)) { device.markLost(); return; }

    backend_->recordPendingWrites(slot_.cmd, serial);
    exec_.record(slot_.cmd, slot_, packet);

    // Left sampleable: whoever made the target reads it next.
    if (TextureSlot* t = backend_->texture(target_))
    {
        Backend::transition(release_, *t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kShaderStages,
                            VK_ACCESS_SHADER_READ_BIT, false);
        release_.record(slot_.cmd);
    }
    vkEndCommandBuffer(slot_.cmd);

    vkResetFences(device.device(), 1, &slot_.fence);
    VkSubmitInfo si {};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &slot_.cmd;
    if (vkQueueSubmit(device.graphicsQueue(), 1, &si, slot_.fence) != VK_SUCCESS)
    {
        device.markLost();
        return;
    }
    backend_->submitted(serial, slot_.fence);
}

} // namespace jvk::vulkan
