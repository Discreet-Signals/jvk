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
 File: Helpers.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// sRGB→linear conversion for physically correct (sRGB) pipeline
inline float srgbToLinear(float c)
{
    return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Apply color space conversion based on pipeline mode
inline glm::vec4 convertColor(VulkanGraphicsContext& ctx, const glm::vec4& c)
{
    if (ctx.srgbMode)
        return { srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b), c.a };
    return c; // UNORM: pass sRGB values through (matches JUCE)
}

// Compose a local (per-call) transform with the accumulated context transform,
// then apply DPI scaling. This is the single source of truth for coordinate mapping.
// Matches JUCE's getTransformWith(): local is innermost (applied first to points).
inline juce::AffineTransform getFullTransform(VulkanGraphicsContext& ctx,
                                               const juce::AffineTransform& local = {})
{
    return local.followedBy(ctx.state().complexTransform)
                .scaled(ctx.scale);
}

// Check if the accumulated transform is non-trivial (rotation/skew/non-uniform-scale).
// SDF shapes (rounded rect, ellipse) only work axis-aligned.
inline bool hasNonTrivialTransform(VulkanGraphicsContext& ctx)
{
    auto& t = ctx.state().complexTransform;
    return !t.isIdentity() && !t.isOnlyTranslation();
}

inline void addVertex(VulkanGraphicsContext& ctx, float x, float y, const glm::vec4& color,
                       float u = 0, float v = 0, const glm::vec4& shape = glm::vec4(0))
{
    ctx.vertices.push_back({ { x, y }, convertColor(ctx, color), { u, v }, shape });
}

// Flat colored quad (shapeType = 0, no SDF)
inline void addQuad(VulkanGraphicsContext& ctx, float x, float y, float w, float h, const glm::vec4& color)
{
    glm::vec4 flat(0); // type 0 = flat
    addVertex(ctx, x,     y,     color, 0, 0, flat);
    addVertex(ctx, x + w, y,     color, 0, 0, flat);
    addVertex(ctx, x + w, y + h, color, 0, 0, flat);
    addVertex(ctx, x,     y,     color, 0, 0, flat);
    addVertex(ctx, x + w, y + h, color, 0, 0, flat);
    addVertex(ctx, x,     y + h, color, 0, 0, flat);
}

// SDF quad — GPU evaluates the shape per-pixel
inline void addSDFQuad(VulkanGraphicsContext& ctx, float x, float y, float w, float h,
                        const glm::vec4& color, float shapeType,
                        float halfW, float halfH, float param)
{
    // Expand quad slightly for AA fringe (1px padding)
    float pad = 1.0f;
    float ex = x - pad, ey = y - pad;
    float ew = w + pad * 2, eh = h + pad * 2;

    // UV maps the padded quad. SDF uses ORIGINAL half-extents —
    // the padding gives room for the AA fringe outside the shape boundary.
    float uPad = pad / ew, vPad = pad / eh;
    float u0 = -uPad, v0 = -vPad;
    float u1 = 1.0f + uPad, v1 = 1.0f + vPad;

    glm::vec4 shape(shapeType, halfW, halfH, param);

    addVertex(ctx, ex,      ey,      color, u0, v0, shape);
    addVertex(ctx, ex + ew, ey,      color, u1, v0, shape);
    addVertex(ctx, ex + ew, ey + eh, color, u1, v1, shape);
    addVertex(ctx, ex,      ey,      color, u0, v0, shape);
    addVertex(ctx, ex + ew, ey + eh, color, u1, v1, shape);
    addVertex(ctx, ex,      ey + eh, color, u0, v1, shape);
}

inline glm::vec4 getColorForFill(VulkanGraphicsContext& ctx);
inline glm::vec4 getColorAt(VulkanGraphicsContext& ctx, float x, float y);

inline glm::vec4 getColorForFill(VulkanGraphicsContext& ctx)
{
    auto& s = ctx.state();
    if (s.fillType.isColour())
    {
        auto c = s.fillType.colour;
        return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                 c.getFloatAlpha() * s.opacity };
    }
    if (s.fillType.isGradient())
        return getColorAt(ctx, 0, 0);
    return { 0, 0, 0, s.opacity };
}

inline glm::vec4 getColorAt(VulkanGraphicsContext& ctx, float x, float y)
{
    auto& s = ctx.state();
    if (s.fillType.isColour())
    {
        auto c = s.fillType.colour;
        return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                 c.getFloatAlpha() * s.opacity };
    }
    if (s.fillType.isGradient())
    {
        auto& g = *s.fillType.gradient;
        float tx = x, ty = y;
        if (!s.fillType.transform.isIdentity())
        {
            auto inv = s.fillType.transform.inverted();
            inv.transformPoint(tx, ty);
        }
        float t;
        if (g.isRadial)
        {
            float dx = tx - g.point1.x, dy = ty - g.point1.y;
            float radius = g.point1.getDistanceFrom(g.point2);
            t = (radius > 0) ? std::sqrt(dx * dx + dy * dy) / radius : 0.0f;
        }
        else
        {
            float dx = g.point2.x - g.point1.x, dy = g.point2.y - g.point1.y;
            float len2 = dx * dx + dy * dy;
            t = (len2 > 0) ? ((tx - g.point1.x) * dx + (ty - g.point1.y) * dy) / len2 : 0.0f;
        }
        t = juce::jlimit(0.0f, 1.0f, t);
        auto c = g.getColourAtPosition(static_cast<double>(t));
        return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                 c.getFloatAlpha() * s.opacity };
    }
    return { 0, 0, 0, s.opacity };
}

// ==== Pipeline state tracking ====
// Avoids redundant vkCmdBindPipeline / push constant / descriptor set calls.

inline void ensurePipeline(VulkanGraphicsContext& ctx, VkPipeline pipeline, VkPipelineLayout layout)
{
    if (ctx.boundPipeline != pipeline)
    {
        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        float pushData[2] = { ctx.vpWidth, ctx.vpHeight };
        vkCmdPushConstants(ctx.cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
        ctx.boundPipeline = pipeline;
        // Pipeline layout change may invalidate descriptor set bindings
        ctx.boundDescriptorSet = VK_NULL_HANDLE;
    }
}

inline void ensureDescriptorSet(VulkanGraphicsContext& ctx, VkPipelineLayout layout, VkDescriptorSet ds)
{
    if (ctx.boundDescriptorSet != ds && ds != VK_NULL_HANDLE)
    {
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                                 0, 1, &ds, 0, nullptr);
        ctx.boundDescriptorSet = ds;
    }
}

} // jvk::graphics
