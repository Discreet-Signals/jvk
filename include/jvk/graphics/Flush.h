/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: Flush.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// Upload vertices from a data pointer and draw. Shared by flush() and stencil fan.
//
// Two paths:
// 1. Ring buffer (preferred): write into persistent ring, zero allocations.
// 2. Legacy extBuffer: grow-on-demand HOST_VISIBLE buffer (original path).
inline void uploadAndDraw(VulkanGraphicsContext& ctx, const UIVertex* data, uint32_t count)
{
    if (count == 0) return;
    jassert(ctx.boundPipeline != VK_NULL_HANDLE); // No pipeline bound before draw

    VkDeviceSize needed = sizeof(UIVertex) * count;

    // Per-frame linear allocator: memcpy + pointer bump. Doubles on overflow.
    // If the ring buffer is available, use it exclusively — no fallback needed.
    if (ctx.ringBuffer && ctx.ringBuffer->isValid())
    {
        VkDeviceSize offset = ctx.ringBuffer->write(data, needed);
        VkBuffer buf = ctx.ringBuffer->getBuffer();
        vkCmdBindVertexBuffers(ctx.cmd, 0, 1, &buf, &offset);
        vkCmdDraw(ctx.cmd, count, 1, 0, 0);
        return;
    }

    // Legacy path: grow-on-demand external buffer (used when ring buffer not wired in)
    core::Buffer* buf = ctx.extBuffer;
    void** mapped = ctx.extMappedPtr;
    core::Buffer tempBuffer;
    void* tempMapped = nullptr;
    if (!buf) { buf = &tempBuffer; mapped = &tempMapped; }

    VkDeviceSize totalNeeded = ctx.bufferWriteOffset + needed;
    if (!buf->isValid() || buf->getSize() < totalNeeded)
    {
        if (buf->isValid() && ctx.deletionQueue)
            ctx.deletionQueue->retire(std::move(*buf));

        VkDeviceSize allocSize = std::max(totalNeeded, static_cast<VkDeviceSize>(sizeof(UIVertex) * 65536));
        buf->create({ ctx.physDevice, ctx.device, allocSize,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
        *mapped = buf->map();
        ctx.bufferWriteOffset = 0;
    }

    memcpy(static_cast<char*>(*mapped) + ctx.bufferWriteOffset, data, static_cast<size_t>(needed));

    VkBuffer buffers[] = { buf->getBuffer() };
    VkDeviceSize offsets[] = { ctx.bufferWriteOffset };
    vkCmdBindVertexBuffers(ctx.cmd, 0, 1, buffers, offsets);
    vkCmdDraw(ctx.cmd, count, 1, 0, 0);

    ctx.bufferWriteOffset += needed;
}

// Flush pending vertices with the main color pipeline.
// All normal drawing (fillRect, SDF quads, text, etc.) accumulates into
// vertices[] and is drawn with the main pipeline + default descriptor set.
// Ensure the main pipeline + default descriptor set are bound.
// Call this before any normal drawing that accumulates vertices.
inline void ensureMainPipeline(VulkanGraphicsContext& ctx)
{
    ensurePipeline(ctx, ctx.colorPipeline, ctx.pipelineLayout);
    ensureDescriptorSet(ctx, ctx.pipelineLayout, ctx.defaultDescriptorSet);
}

// Drain the vertex buffer with whatever pipeline is currently bound.
// Callers must ensure the correct pipeline is bound before adding vertices.
inline void flush(VulkanGraphicsContext& ctx)
{
    if (ctx.vertices.empty()) return;

    auto& s = ctx.state();
    VkRect2D scissor;
    scissor.offset = { s.clipBounds.getX(), s.clipBounds.getY() };
    scissor.extent = {
        static_cast<uint32_t>(std::max(0, s.clipBounds.getWidth())),
        static_cast<uint32_t>(std::max(0, s.clipBounds.getHeight()))
    };
    vkCmdSetScissor(ctx.cmd, 0, 1, &scissor);

    uploadAndDraw(ctx, ctx.vertices.data(), static_cast<uint32_t>(ctx.vertices.size()));
    ctx.vertices.clear();
}

} // jvk::graphics
