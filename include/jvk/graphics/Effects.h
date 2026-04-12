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
 File: Effects.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// Draw a fullscreen effect quad using a specific pipeline
inline void drawEffectQuad(VulkanGraphicsContext& ctx, Pipeline* effectPipeline,
                            const glm::vec4& color, juce::Rectangle<float> region = {},
                            const glm::vec4& shapeData = glm::vec4(0))
{
    if (!effectPipeline) return;
    flush(ctx);

    ensurePipeline(ctx, effectPipeline->getInternal(), effectPipeline->getLayout());
    ensureDescriptorSet(ctx, effectPipeline->getLayout(), ctx.defaultDescriptorSet);

    float x = region.isEmpty() ? 0 : region.getX();
    float y = region.isEmpty() ? 0 : region.getY();
    float w = region.isEmpty() ? ctx.vpWidth : region.getWidth();
    float h = region.isEmpty() ? ctx.vpHeight : region.getHeight();

    addVertex(ctx, x,     y,     color, 0, 0, shapeData);
    addVertex(ctx, x + w, y,     color, 0, 0, shapeData);
    addVertex(ctx, x + w, y + h, color, 0, 0, shapeData);
    addVertex(ctx, x,     y,     color, 0, 0, shapeData);
    addVertex(ctx, x + w, y + h, color, 0, 0, shapeData);
    addVertex(ctx, x,     y + h, color, 0, 0, shapeData);

    flush(ctx);
    ensureMainPipeline(ctx);
}

// Blur implementation — ends render pass, copies framebuffer, restarts, draws blur quads
inline void executeBlur(VulkanGraphicsContext& ctx, juce::Rectangle<float> region, float radius)
{
    if (!ctx.blurPipeline || !ctx.blurTempImage || ctx.rpInfo.renderPass == VK_NULL_HANDLE) return;

    flush(ctx); // render all pending geometry first

    // 1. End render pass
    vkCmdEndRenderPass(ctx.cmd);

    // 2. Transition MSAA image for copy source
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = ctx.rpInfo.msaaImage;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // 3. Transition temp image for copy dest
    VkImageMemoryBarrier barrier2 = barrier;
    barrier2.image = ctx.blurTempImage->getImage();
    barrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier2.srcAccessMask = 0;
    barrier2.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

    // 4. Copy framebuffer to temp
    VkImageCopy copyRegion = {};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { ctx.rpInfo.extent.width, ctx.rpInfo.extent.height, 1 };
    vkCmdCopyImage(ctx.cmd, ctx.rpInfo.msaaImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    ctx.blurTempImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &copyRegion);

    // 5. Transition images back
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

    // 6. Restart render pass (no clear — LOAD existing content)
    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = ctx.rpInfo.renderPass;
    rpBegin.framebuffer = ctx.rpInfo.framebuffer;
    rpBegin.renderArea = { {0, 0}, ctx.rpInfo.extent };
    std::array<VkClearValue, 3> clearValues = {};
    rpBegin.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBegin.pClearValues = clearValues.data();
    vkCmdBeginRenderPass(ctx.cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // 7. Set viewport/scissor
    VkViewport vp = {};
    vp.width = static_cast<float>(ctx.rpInfo.extent.width);
    vp.height = static_cast<float>(ctx.rpInfo.extent.height);
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(ctx.cmd, 0, 1, &vp);
    VkRect2D sc = { {0, 0}, ctx.rpInfo.extent };
    vkCmdSetScissor(ctx.cmd, 0, 1, &sc);

    // 8. Draw horizontal blur pass
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx.blurPipeline->getInternal());
    float pushData[2] = { ctx.vpWidth, ctx.vpHeight };
    vkCmdPushConstants(ctx.cmd, ctx.blurPipeline->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

    if (ctx.blurDescSet != VK_NULL_HANDLE)
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx.blurPipeline->getLayout(),
                                 0, 1, &ctx.blurDescSet, 0, nullptr);

    float x = region.isEmpty() ? 0 : region.getX();
    float y = region.isEmpty() ? 0 : region.getY();
    float w = region.isEmpty() ? ctx.vpWidth : region.getWidth();
    float h = region.isEmpty() ? ctx.vpHeight : region.getHeight();

    // Horizontal pass: direction=(1,0), radius in z
    glm::vec4 hDir(1.0f, 0.0f, radius, 1.0f);
    glm::vec4 noShape(0);
    addVertex(ctx, x, y, hDir, 0, 0, noShape);
    addVertex(ctx, x+w, y, hDir, 0, 0, noShape);
    addVertex(ctx, x+w, y+h, hDir, 0, 0, noShape);
    addVertex(ctx, x, y, hDir, 0, 0, noShape);
    addVertex(ctx, x+w, y+h, hDir, 0, 0, noShape);
    addVertex(ctx, x, y+h, hDir, 0, 0, noShape);
    flush(ctx);

    // For a proper two-pass blur, we'd need to copy again for the vertical pass.
    // For now, single-pass gives a reasonable blur effect.
    // TODO: two-pass separable gaussian (horizontal -> copy -> vertical)

    // 9. Rebind main pipeline (invalidate tracker — blur ended/restarted render pass)
    ctx.boundPipeline = VK_NULL_HANDLE;
    ctx.boundDescriptorSet = VK_NULL_HANDLE;
    ensurePipeline(ctx, ctx.colorPipeline, ctx.pipelineLayout);
    ensureDescriptorSet(ctx, ctx.pipelineLayout, ctx.defaultDescriptorSet);
}

} // jvk::graphics
