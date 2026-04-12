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
 File: FillRect.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// Emit a gradient-filled quad using the GPU gradient LUT pipeline
// Falls back to 4-corner CPU-sampled interpolation if no gradient cache
inline void emitGradientQuad(VulkanGraphicsContext& ctx, float x, float y, float w, float h)
{
    auto& s = ctx.state();
    if (!s.fillType.isGradient() || !s.fillType.gradient) return;

    auto& g = *s.fillType.gradient;

    // GPU path: bind 1D gradient LUT texture, emit quad with gradient-space UVs
    if (ctx.gradientCache)
    {
        VkDescriptorSet gradDescSet = ctx.gradientCache->getOrCreate(g);
        if (gradDescSet != VK_NULL_HANDLE)
        {
            flush(ctx);
            ensureDescriptorSet(ctx, ctx.pipelineLayout, gradDescSet);

            // Compute UV coordinates in gradient space for each corner.
            // Map gradient endpoints from logical to physical space through full transform.
            auto physT = s.transform.scaled(ctx.scale)
                         .translated(static_cast<float>(s.origin.x),
                                     static_cast<float>(s.origin.y));
            glm::vec4 color(1.0f, 1.0f, 1.0f, s.opacity);
            float shapeType = g.isRadial ? 6.0f : 5.0f;
            glm::vec4 shape(shapeType, 0, 0, 0);

            if (g.isRadial)
            {
                // Radial: UV = normalized offset from center, length(UV)=1 at radius edge
                float p1x = g.point1.x, p1y = g.point1.y;
                float p2x = g.point2.x, p2y = g.point2.y;
                physT.transformPoint(p1x, p1y);
                physT.transformPoint(p2x, p2y);
                float cx = p1x, cy = p1y;
                float radius = std::sqrt((p2x - p1x) * (p2x - p1x) + (p2y - p1y) * (p2y - p1y));
                float invR = (radius > 0) ? 1.0f / radius : 0.0f;

                auto uv = [&](float px, float py) -> std::pair<float, float> {
                    return { (px - cx) * invR, (py - cy) * invR };
                };

                auto [u00, v00] = uv(x, y);
                auto [u10, v10] = uv(x + w, y);
                auto [u11, v11] = uv(x + w, y + h);
                auto [u01, v01] = uv(x, y + h);

                addVertex(ctx, x,     y,     color, u00, v00, shape);
                addVertex(ctx, x + w, y,     color, u10, v10, shape);
                addVertex(ctx, x + w, y + h, color, u11, v11, shape);
                addVertex(ctx, x,     y,     color, u00, v00, shape);
                addVertex(ctx, x + w, y + h, color, u11, v11, shape);
                addVertex(ctx, x,     y + h, color, u01, v01, shape);
            }
            else
            {
                // Linear: UV.x = t along gradient axis
                float gx1 = g.point1.x, gy1 = g.point1.y;
                float gx2 = g.point2.x, gy2 = g.point2.y;
                physT.transformPoint(gx1, gy1);
                physT.transformPoint(gx2, gy2);
                float dx = gx2 - gx1, dy = gy2 - gy1;
                float len2 = dx * dx + dy * dy;
                float invLen2 = (len2 > 0) ? 1.0f / len2 : 0.0f;

                auto tAt = [&](float px, float py) -> float {
                    return ((px - gx1) * dx + (py - gy1) * dy) * invLen2;
                };

                float t00 = tAt(x, y),     t10 = tAt(x + w, y);
                float t11 = tAt(x + w, y + h), t01 = tAt(x, y + h);

                addVertex(ctx, x,     y,     color, t00, 0, shape);
                addVertex(ctx, x + w, y,     color, t10, 0, shape);
                addVertex(ctx, x + w, y + h, color, t11, 0, shape);
                addVertex(ctx, x,     y,     color, t00, 0, shape);
                addVertex(ctx, x + w, y + h, color, t11, 0, shape);
                addVertex(ctx, x,     y + h, color, t01, 0, shape);
            }

            flush(ctx);
            ensureDescriptorSet(ctx, ctx.pipelineLayout, ctx.defaultDescriptorSet);
            return;
        }
    }

    // Fallback: CPU-sampled 4-corner interpolation
    auto c00 = getColorAt(ctx, x,     y);
    auto c10 = getColorAt(ctx, x + w, y);
    auto c11 = getColorAt(ctx, x + w, y + h);
    auto c01 = getColorAt(ctx, x,     y + h);
    glm::vec4 flat(0);
    addVertex(ctx, x,     y,     c00, 0, 0, flat);
    addVertex(ctx, x + w, y,     c10, 0, 0, flat);
    addVertex(ctx, x + w, y + h, c11, 0, 0, flat);
    addVertex(ctx, x,     y,     c00, 0, 0, flat);
    addVertex(ctx, x + w, y + h, c11, 0, 0, flat);
    addVertex(ctx, x,     y + h, c01, 0, 0, flat);
}

// ==== Drawing: fillRect ====
// JUCE passes logical point rects. Scale to physical pixels.
inline void fillRect(VulkanGraphicsContext& ctx, const juce::Rectangle<float>& r)
{
    if (isClipEmpty(ctx)) return;
    auto& s = ctx.state();

    if (hasNonTrivialTransform(ctx))
    {
        // Non-trivial transform: emit a transformed quad
        auto t = getFullTransform(ctx);
        auto p00 = juce::Point<float>(r.getX(), r.getY()).transformedBy(t);
        auto p10 = juce::Point<float>(r.getRight(), r.getY()).transformedBy(t);
        auto p11 = juce::Point<float>(r.getRight(), r.getBottom()).transformedBy(t);
        auto p01 = juce::Point<float>(r.getX(), r.getBottom()).transformedBy(t);
        auto c = getColorForFill(ctx);
        glm::vec4 flat(0);
        addVertex(ctx, p00.x, p00.y, c, 0, 0, flat);
        addVertex(ctx, p10.x, p10.y, c, 0, 0, flat);
        addVertex(ctx, p11.x, p11.y, c, 0, 0, flat);
        addVertex(ctx, p00.x, p00.y, c, 0, 0, flat);
        addVertex(ctx, p11.x, p11.y, c, 0, 0, flat);
        addVertex(ctx, p01.x, p01.y, c, 0, 0, flat);
        return;
    }

    // Fast path: identity or translation-only transform. Apply any translation before DPI scale.
    auto adjusted = r;
    if (!s.transform.isIdentity())
        adjusted = adjusted.translated(s.transform.getTranslationX(), s.transform.getTranslationY());

    auto phys = juce::Rectangle<float>(adjusted.getX() * ctx.scale, adjusted.getY() * ctx.scale,
                                        adjusted.getWidth() * ctx.scale, adjusted.getHeight() * ctx.scale);
    auto translated = phys.translated(static_cast<float>(s.origin.x),
                                       static_cast<float>(s.origin.y));

    if (s.fillType.isGradient())
    {
        emitGradientQuad(ctx, translated.getX(), translated.getY(),
                         translated.getWidth(), translated.getHeight());
    }
    else
    {
        addQuad(ctx, translated.getX(), translated.getY(),
                translated.getWidth(), translated.getHeight(), getColorForFill(ctx));
    }
}

inline void fillRectInt(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    fillRect(ctx, r.toFloat());
}

inline void fillRectList(VulkanGraphicsContext& ctx, const juce::RectangleList<float>& list)
{
    for (auto& r : list) fillRect(ctx, r);
}

} // jvk::graphics
