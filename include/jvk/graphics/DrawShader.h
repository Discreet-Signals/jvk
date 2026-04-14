/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: DrawShader.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// Draw a user's custom fragment shader into the active render pass.
// The shader pipeline is created lazily on first call using the context's device/render pass.
inline void drawShader(VulkanGraphicsContext& ctx, Shader& shader,
                       juce::Rectangle<float> region = {})
{
    if (!ctx.rpInfo.renderPass) return;

    // Lazy-init the shader's Vulkan resources using the current context
    shader.ensureCreated(ctx.physDevice, ctx.device,
                         ctx.rpInfo.renderPass,
                         VK_SAMPLE_COUNT_4_BIT); // match PaintBridge MSAA
    if (!shader.isReady()) return;

    // Flush pending UI geometry before switching pipelines
    flush(ctx);

    // Update storage buffer data
    for (auto& dsi : shader.descriptorSets)
    {
        if (dsi.hasStorageBuffer && dsi.mappedStoragePtr && shader.storageSize > 0)
            std::memcpy(dsi.mappedStoragePtr, shader.storageData.data(),
                        sizeof(float) * shader.storageSize);
    }

    // Bind the shader's pipeline
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipeline);

    // Compute region in physical pixels
    float x = region.isEmpty() ? 0 : region.getX() * ctx.scale;
    float y = region.isEmpty() ? 0 : region.getY() * ctx.scale;
    float w = region.isEmpty() ? ctx.vpWidth : region.getWidth() * ctx.scale;
    float h = region.isEmpty() ? ctx.vpHeight : region.getHeight() * ctx.scale;

    // Push constants: [resX, resY, time, vpW, vpH, regX, regY]
    struct PushConstants {
        float resX, resY, time, vpW, vpH, regX, regY;
    } pc;
    pc.resX = w;
    pc.resY = h;
    pc.time = static_cast<float>(juce::Time::getMillisecondCounterHiRes() / 1000.0 - shader.startTime);
    pc.vpW = ctx.vpWidth;
    pc.vpH = ctx.vpHeight;
    pc.regX = x;
    pc.regY = y;
    vkCmdPushConstants(ctx.cmd, shader.pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pc), &pc);

    // Set viewport
    VkViewport vp = {};
    vp.width = ctx.vpWidth;
    vp.height = ctx.vpHeight;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(ctx.cmd, 0, 1, &vp);

    // Scissor to region
    VkRect2D scissor = {};
    scissor.offset = { static_cast<int32_t>(x), static_cast<int32_t>(y) };
    scissor.extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    vkCmdSetScissor(ctx.cmd, 0, 1, &scissor);

    // Bind descriptor sets
    for (auto& dsi : shader.descriptorSets)
    {
        if (dsi.descriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shader.pipelineLayout, dsi.set, 1, &dsi.descriptorSet, 0, nullptr);
    }

    // Draw fullscreen triangle (vertex shader generates geometry from gl_VertexIndex)
    vkCmdDraw(ctx.cmd, 3, 1, 0, 0);

    // Restore main pipeline state
    ctx.boundPipeline = VK_NULL_HANDLE;
    ctx.boundDescriptorSet = VK_NULL_HANDLE;
    ensureMainPipeline(ctx);

    // Restore full viewport scissor
    VkRect2D fullScissor = { {0, 0}, { static_cast<uint32_t>(ctx.vpWidth),
                                        static_cast<uint32_t>(ctx.vpHeight) } };
    vkCmdSetScissor(ctx.cmd, 0, 1, &fullScissor);
}

} // jvk::graphics
