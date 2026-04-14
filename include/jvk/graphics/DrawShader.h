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
// Respects the JUCE transform stack and stencil-based path clipping.
inline void drawShader(VulkanGraphicsContext& ctx, Shader& shader,
                       juce::Rectangle<float> region = {})
{
    if (!ctx.rpInfo.renderPass) return;

    // Lazy-init the shader's Vulkan resources using the current context
    shader.ensureCreated(ctx.physDevice, ctx.device,
                         ctx.rpInfo.renderPass,
                         ctx.rpInfo.msaaSamples);
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

    // Apply the accumulated JUCE transform (component → physical pixels)
    auto transform = getFullTransform(ctx);
    float x1 = region.getX(), y1 = region.getY();
    float x2 = region.getRight(), y2 = region.getBottom();
    transform.transformPoint(x1, y1);
    transform.transformPoint(x2, y2);

    float x = std::min(x1, x2);
    float y = std::min(y1, y2);
    float w = std::abs(x2 - x1);
    float h = std::abs(y2 - y1);

    // Bind the shader's pipeline
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipeline);

    // Set stencil reference to match current clip depth (for path clipping)
    vkCmdSetStencilReference(ctx.cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                              ctx.state().stencilClipDepth);

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

    // Scissor to the intersection of the transformed region and the context's clip bounds
    auto& clipBounds = ctx.state().clipBounds;
    int sx = std::max(static_cast<int>(x), clipBounds.getX());
    int sy = std::max(static_cast<int>(y), clipBounds.getY());
    int sx2 = std::min(static_cast<int>(x + w), clipBounds.getRight());
    int sy2 = std::min(static_cast<int>(y + h), clipBounds.getBottom());

    VkRect2D scissor = {};
    scissor.offset = { sx, sy };
    scissor.extent = { static_cast<uint32_t>(std::max(0, sx2 - sx)),
                       static_cast<uint32_t>(std::max(0, sy2 - sy)) };
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
}

} // jvk::graphics
