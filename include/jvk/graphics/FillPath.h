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
 File: FillPath.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

inline void fillPath_EdgeTable(VulkanGraphicsContext& ctx, const juce::Path& path,
                                const juce::AffineTransform& combined)
{
    auto& s = ctx.state();
    juce::EdgeTable et(s.clipBounds, path, combined);

    if (s.fillType.isGradient())
    {
        GradientEdgeTableCallback callback(ctx);
        et.iterate(callback);
    }
    else
    {
        SolidEdgeTableCallback callback(ctx, getColorForFill(ctx));
        et.iterate(callback);
    }
}

// GPU stencil-then-cover path rendering.
inline void fillPathStencil(VulkanGraphicsContext& ctx, const juce::Path& path,
                             const juce::AffineTransform& combined)
{
    if (!ctx.stencilPipeline || !ctx.stencilCoverPipeline)
    {
        fillPath_EdgeTable(ctx, path, combined);
        return;
    }
    // Flatten path to triangle fan, computing bounds in one pass.
    // Uses persistent scratch buffers to avoid per-path heap allocation.
    ctx.scratchFanVerts.clear();

    float fanMinX = std::numeric_limits<float>::max(), fanMinY = fanMinX;
    float fanMaxX = -fanMinX, fanMaxY = -fanMinY;

    {
        juce::PathFlatteningIterator iter(path, combined);
        glm::vec2 fanCenter(0);
        glm::vec2 lastPt(std::numeric_limits<float>::max());
        glm::vec2 prevPt(0);
        bool hasSubpath = false;
        glm::vec4 dummy(0);

        auto updateBounds = [&](float x, float y) {
            fanMinX = std::min(fanMinX, x); fanMinY = std::min(fanMinY, y);
            fanMaxX = std::max(fanMaxX, x); fanMaxY = std::max(fanMaxY, y);
        };

        while (iter.next())
        {
            glm::vec2 p1 { iter.x1, iter.y1 };
            glm::vec2 p2 { iter.x2, iter.y2 };

            // Detect new subpath: first segment, or a discontinuity
            if (!hasSubpath ||
                std::abs(p1.x - lastPt.x) > 0.01f ||
                std::abs(p1.y - lastPt.y) > 0.01f)
            {
                fanCenter = p1;
                prevPt = p1;
                hasSubpath = true;
                updateBounds(p1.x, p1.y);
            }

            // Emit triangle: center, prevPt, p2
            ctx.scratchFanVerts.push_back({ fanCenter, dummy, {}, dummy });
            ctx.scratchFanVerts.push_back({ prevPt,    dummy, {}, dummy });
            ctx.scratchFanVerts.push_back({ p2,        dummy, {}, dummy });

            updateBounds(p2.x, p2.y);
            prevPt = p2;
            lastPt = p2;
        }
    }

    if (ctx.scratchFanVerts.empty()) return;

    // Flush pending main-pipeline vertices
    flush(ctx);

    // Stencil write pass — full viewport scissor
    {
        VkRect2D fullScissor = { { 0, 0 }, { static_cast<uint32_t>(ctx.vpWidth), static_cast<uint32_t>(ctx.vpHeight) } };
        vkCmdSetScissor(ctx.cmd, 0, 1, &fullScissor);
    }
    ensurePipeline(ctx, ctx.stencilPipeline->getInternal(), ctx.stencilPipeline->getLayout());
    uploadAndDraw(ctx, ctx.scratchFanVerts.data(), static_cast<uint32_t>(ctx.scratchFanVerts.size()));

    // Cover pass — clip scissor, stencil test NOT_EQUAL 0, zeros on pass
    ensurePipeline(ctx, ctx.stencilCoverPipeline->getInternal(), ctx.stencilCoverPipeline->getLayout());
    ensureDescriptorSet(ctx, ctx.stencilCoverPipeline->getLayout(), ctx.defaultDescriptorSet);

    float bx = fanMinX, by = fanMinY;
    float bw = fanMaxX - fanMinX, bh = fanMaxY - fanMinY;

    auto& s = ctx.state();
    ctx.vertices.clear();

    if (s.fillType.isGradient() && s.fillType.gradient && ctx.gradientCache)
    {
        auto& g = *s.fillType.gradient;
        VkDescriptorSet gradDescSet = ctx.gradientCache->getOrCreate(g);
        ensureDescriptorSet(ctx, ctx.stencilCoverPipeline->getLayout(), gradDescSet);

        glm::vec4 color(1.0f, 1.0f, 1.0f, s.opacity);
        float shapeType = g.isRadial ? 6.0f : 5.0f;
        glm::vec4 shape(shapeType, 0, 0, 0);

        if (g.isRadial)
        {
            float cx = g.point1.x * ctx.scale + static_cast<float>(s.origin.x);
            float cy = g.point1.y * ctx.scale + static_cast<float>(s.origin.y);
            float radius = g.point1.getDistanceFrom(g.point2) * ctx.scale;
            float invR = (radius > 0) ? 1.0f / radius : 0.0f;

            auto uv = [&](float px, float py) { return std::pair{ (px-cx)*invR, (py-cy)*invR }; };
            auto [u00,v00] = uv(bx, by);       auto [u10,v10] = uv(bx+bw, by);
            auto [u11,v11] = uv(bx+bw, by+bh); auto [u01,v01] = uv(bx, by+bh);

            addVertex(ctx, bx,      by,      color, u00, v00, shape);
            addVertex(ctx, bx+bw,   by,      color, u10, v10, shape);
            addVertex(ctx, bx+bw,   by+bh,   color, u11, v11, shape);
            addVertex(ctx, bx,      by,      color, u00, v00, shape);
            addVertex(ctx, bx+bw,   by+bh,   color, u11, v11, shape);
            addVertex(ctx, bx,      by+bh,   color, u01, v01, shape);
        }
        else
        {
            float gx1 = g.point1.x * ctx.scale + static_cast<float>(s.origin.x);
            float gy1 = g.point1.y * ctx.scale + static_cast<float>(s.origin.y);
            float gx2 = g.point2.x * ctx.scale + static_cast<float>(s.origin.x);
            float gy2 = g.point2.y * ctx.scale + static_cast<float>(s.origin.y);
            float dx = gx2-gx1, dy = gy2-gy1;
            float len2 = dx*dx + dy*dy;
            float invLen2 = (len2 > 0) ? 1.0f / len2 : 0.0f;
            auto tAt = [&](float px, float py) { return ((px-gx1)*dx + (py-gy1)*dy) * invLen2; };

            float t00 = tAt(bx,by), t10 = tAt(bx+bw,by), t11 = tAt(bx+bw,by+bh), t01 = tAt(bx,by+bh);
            addVertex(ctx, bx,    by,    color, t00, 0, shape);
            addVertex(ctx, bx+bw, by,    color, t10, 0, shape);
            addVertex(ctx, bx+bw, by+bh, color, t11, 0, shape);
            addVertex(ctx, bx,    by,    color, t00, 0, shape);
            addVertex(ctx, bx+bw, by+bh, color, t11, 0, shape);
            addVertex(ctx, bx,    by+bh, color, t01, 0, shape);
        }
    }
    else
    {
        auto fillColor = getColorForFill(ctx);
        glm::vec4 flat(0);
        addVertex(ctx, bx,      by,      fillColor, 0, 0, flat);
        addVertex(ctx, bx + bw, by,      fillColor, 0, 0, flat);
        addVertex(ctx, bx + bw, by + bh, fillColor, 0, 0, flat);
        addVertex(ctx, bx,      by,      fillColor, 0, 0, flat);
        addVertex(ctx, bx + bw, by + bh, fillColor, 0, 0, flat);
        addVertex(ctx, bx,      by + bh, fillColor, 0, 0, flat);
    }
    flush(ctx);
    ensureMainPipeline(ctx);
}

// ==== Drawing: fillPath ====
inline void fillPath(VulkanGraphicsContext& ctx, const juce::Path& path,
                      const juce::AffineTransform& t)
{
    if (isClipEmpty(ctx)) return;
    auto combined = getFullTransform(ctx, t);

    switch (ctx.pathBackend)
    {
        case VulkanGraphicsContext::PathBackend::Stencil:
            fillPathStencil(ctx, path, combined);
            return;
        case VulkanGraphicsContext::PathBackend::EdgeTable:
            fillPath_EdgeTable(ctx, path, combined);
            return;
    }
}

} // jvk::graphics
