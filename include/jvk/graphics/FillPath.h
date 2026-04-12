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

// GPU stencil-then-cover path rendering.
// Uses Path::Iterator for explicit subpath boundaries (startNewSubPath / closePath)
// instead of heuristic position-based discontinuity detection. Curves are
// flattened inline to line segments before building the stencil triangle fan.
inline void fillPathStencil(VulkanGraphicsContext& ctx, const juce::Path& path,
                             const juce::AffineTransform& combined)
{
    if (!ctx.stencilPipeline || !ctx.stencilCoverPipeline)
        return;

    // Flatten path to triangle fan, computing bounds in one pass.
    // Uses persistent scratch buffers to avoid per-path heap allocation.
    ctx.scratchFanVerts.clear();
    ctx.scratchPoints.clear();

    float fanMinX = std::numeric_limits<float>::max(), fanMinY = fanMinX;
    float fanMaxX = -fanMinX, fanMaxY = -fanMinY;

    {
        // Phase 1: Collect transformed, flattened points with subpath markers.
        // scratchPoints stores all points; negative-infinity X marks subpath starts.
        const float SUBPATH_MARKER = -std::numeric_limits<float>::infinity();
        constexpr float flatTol = 0.5f; // flattening tolerance in pixels

        juce::Path::Iterator iter(path);
        glm::vec2 lastPt(0);
        glm::vec2 subpathStart(0);

        auto transformPt = [&](float x, float y) -> glm::vec2 {
            combined.transformPoint(x, y);
            return { x, y };
        };

        // Subdivide a cubic bezier to line segments
        auto flattenCubic = [&](glm::vec2 p0, glm::vec2 c1, glm::vec2 c2, glm::vec2 p3)
        {
            // Adaptive subdivision: split until flat enough
            struct Segment { glm::vec2 a, b, c, d; };
            // Use scratchPoints as a stack (will be consumed later)
            std::vector<Segment> stack;
            stack.push_back({ p0, c1, c2, p3 });

            while (!stack.empty())
            {
                auto [a, b, c, d] = stack.back();
                stack.pop_back();

                // Flatness test: max distance of control points from line a→d
                float dx = d.x - a.x, dy = d.y - a.y;
                float len2 = dx * dx + dy * dy;
                float d1, d2;
                if (len2 > 0.0001f)
                {
                    float inv = 1.0f / len2;
                    float t1 = ((b.x - a.x) * dx + (b.y - a.y) * dy) * inv;
                    float t2 = ((c.x - a.x) * dx + (c.y - a.y) * dy) * inv;
                    float px1 = a.x + t1 * dx - b.x, py1 = a.y + t1 * dy - b.y;
                    float px2 = a.x + t2 * dx - c.x, py2 = a.y + t2 * dy - c.y;
                    d1 = px1 * px1 + py1 * py1;
                    d2 = px2 * px2 + py2 * py2;
                }
                else
                {
                    d1 = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
                    d2 = (c.x - a.x) * (c.x - a.x) + (c.y - a.y) * (c.y - a.y);
                }

                if (d1 <= flatTol * flatTol && d2 <= flatTol * flatTol)
                {
                    ctx.scratchPoints.push_back(d);
                }
                else
                {
                    // De Casteljau split at t=0.5
                    auto mid = [](glm::vec2 u, glm::vec2 v) { return (u + v) * 0.5f; };
                    auto ab = mid(a, b), bc = mid(b, c), cd = mid(c, d);
                    auto abc = mid(ab, bc), bcd = mid(bc, cd);
                    auto abcd = mid(abc, bcd);
                    // Push right half first so left is processed first
                    stack.push_back({ abcd, bcd, cd, d });
                    stack.push_back({ a, ab, abc, abcd });
                }
            }
        };

        // Subdivide a quadratic bezier to line segments
        auto flattenQuad = [&](glm::vec2 p0, glm::vec2 c, glm::vec2 p2)
        {
            // Promote to cubic: C1 = P0 + 2/3*(C-P0), C2 = P2 + 2/3*(C-P2)
            glm::vec2 c1 = p0 + (2.0f / 3.0f) * (c - p0);
            glm::vec2 c2 = p2 + (2.0f / 3.0f) * (c - p2);
            flattenCubic(p0, c1, c2, p2);
        };

        while (iter.next())
        {
            switch (iter.elementType)
            {
                case juce::Path::Iterator::startNewSubPath:
                {
                    auto pt = transformPt(iter.x1, iter.y1);
                    ctx.scratchPoints.push_back({ SUBPATH_MARKER, 0 }); // marker
                    ctx.scratchPoints.push_back(pt);
                    subpathStart = pt;
                    lastPt = pt;
                    break;
                }
                case juce::Path::Iterator::lineTo:
                {
                    auto pt = transformPt(iter.x1, iter.y1);
                    ctx.scratchPoints.push_back(pt);
                    lastPt = pt;
                    break;
                }
                case juce::Path::Iterator::quadraticTo:
                {
                    auto c = transformPt(iter.x1, iter.y1);
                    auto p = transformPt(iter.x2, iter.y2);
                    flattenQuad(lastPt, c, p);
                    lastPt = p;
                    break;
                }
                case juce::Path::Iterator::cubicTo:
                {
                    auto c1 = transformPt(iter.x1, iter.y1);
                    auto c2 = transformPt(iter.x2, iter.y2);
                    auto p  = transformPt(iter.x3, iter.y3);
                    flattenCubic(lastPt, c1, c2, p);
                    lastPt = p;
                    break;
                }
                case juce::Path::Iterator::closePath:
                {
                    // Emit explicit closing line back to subpath start
                    ctx.scratchPoints.push_back(subpathStart);
                    lastPt = subpathStart;
                    break;
                }
            }
        }

        // Phase 2: Build triangle fan from flattened points.
        // Each subpath marker resets the fan center. The fan center is the
        // first point of the subpath (on the boundary).
        glm::vec4 dummy(0);
        glm::vec2 fanCenter(0);
        glm::vec2 prevPt2(0);
        bool inSubpath = false;

        auto updateBounds = [&](float x, float y) {
            fanMinX = std::min(fanMinX, x); fanMinY = std::min(fanMinY, y);
            fanMaxX = std::max(fanMaxX, x); fanMaxY = std::max(fanMaxY, y);
        };

        for (size_t i = 0; i < ctx.scratchPoints.size(); i++)
        {
            if (ctx.scratchPoints[i].x == SUBPATH_MARKER)
            {
                // Next point is the subpath start / fan center
                i++;
                if (i >= ctx.scratchPoints.size()) break;
                fanCenter = ctx.scratchPoints[i];
                prevPt2 = fanCenter;
                inSubpath = true;
                updateBounds(fanCenter.x, fanCenter.y);
                continue;
            }

            if (!inSubpath) continue;

            glm::vec2 pt = ctx.scratchPoints[i];
            updateBounds(pt.x, pt.y);

            // Emit fan triangle: center, previous edge point, current edge point
            ctx.scratchFanVerts.push_back({ fanCenter, dummy, {}, dummy });
            ctx.scratchFanVerts.push_back({ prevPt2,   dummy, {}, dummy });
            ctx.scratchFanVerts.push_back({ pt,        dummy, {}, dummy });

            prevPt2 = pt;
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

        // Map gradient endpoints to physical space through full transform
        auto physT = getFullTransform(ctx);

        glm::vec4 color(1.0f, 1.0f, 1.0f, s.opacity);
        float shapeType = g.isRadial ? 6.0f : 5.0f;
        glm::vec4 shape(shapeType, 0, 0, 0);

        if (g.isRadial)
        {
            float p1x = g.point1.x, p1y = g.point1.y;
            float p2x = g.point2.x, p2y = g.point2.y;
            physT.transformPoint(p1x, p1y);
            physT.transformPoint(p2x, p2y);
            float cx = p1x, cy = p1y;
            float radius = std::sqrt((p2x - p1x) * (p2x - p1x) + (p2y - p1y) * (p2y - p1y));
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
            float gx1 = g.point1.x, gy1 = g.point1.y;
            float gx2 = g.point2.x, gy2 = g.point2.y;
            physT.transformPoint(gx1, gy1);
            physT.transformPoint(gx2, gy2);
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
    fillPathStencil(ctx, path, combined);
}

} // jvk::graphics
