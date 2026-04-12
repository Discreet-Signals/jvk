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
 File: EdgeTableCallbacks.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// ==== EdgeTable Callbacks (emit vertices in physical pixels) ====

struct SolidEdgeTableCallback
{
    VulkanGraphicsContext& ctx;
    glm::vec4 color;
    int currentY = 0;

    SolidEdgeTableCallback(VulkanGraphicsContext& c, glm::vec4 col) : ctx(c), color(col) {}

    void setEdgeTableYPos(int y) { currentY = y; }

    void handleEdgeTablePixelFull(int x) const
    { addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, color); }

    void handleEdgeTablePixel(int x, int level) const
    {
        auto c = color;
        c.a *= correctAlpha(ctx, level);
        addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
    }

    void handleEdgeTableLineFull(int x, int width) const
    { addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), static_cast<float>(width), 1.0f, color); }

    void handleEdgeTableLine(int x, int width, int level) const
    {
        auto c = color;
        c.a *= correctAlpha(ctx, level);
        addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), static_cast<float>(width), 1.0f, c);
    }
};

struct GradientEdgeTableCallback
{
    VulkanGraphicsContext& ctx;
    int currentY = 0;

    GradientEdgeTableCallback(VulkanGraphicsContext& c) : ctx(c) {}

    void setEdgeTableYPos(int y) { currentY = y; }

    void handleEdgeTablePixelFull(int x) const
    {
        auto c = getColorAt(ctx, static_cast<float>(x) + 0.5f, static_cast<float>(currentY) + 0.5f);
        addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
    }

    void handleEdgeTablePixel(int x, int level) const
    {
        auto c = getColorAt(ctx, static_cast<float>(x) + 0.5f, static_cast<float>(currentY) + 0.5f);
        c.a *= correctAlpha(ctx, level);
        addQuad(ctx, static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
    }

    void handleEdgeTableLineFull(int x, int width) const
    {
        // GPU gradient: emit single quad with corner colors, hardware interpolates
        float fx = static_cast<float>(x), fy = static_cast<float>(currentY);
        float fw = static_cast<float>(width);
        auto c0 = getColorAt(ctx, fx + 0.5f, fy + 0.5f);
        auto c1 = getColorAt(ctx, fx + fw - 0.5f, fy + 0.5f);
        glm::vec4 flat(0);
        addVertex(ctx, fx,      fy,       c0, 0, 0, flat);
        addVertex(ctx, fx + fw, fy,       c1, 0, 0, flat);
        addVertex(ctx, fx + fw, fy + 1.f, c1, 0, 0, flat);
        addVertex(ctx, fx,      fy,       c0, 0, 0, flat);
        addVertex(ctx, fx + fw, fy + 1.f, c1, 0, 0, flat);
        addVertex(ctx, fx,      fy + 1.f, c0, 0, 0, flat);
    }

    void handleEdgeTableLine(int x, int width, int level) const
    {
        float fx = static_cast<float>(x), fy = static_cast<float>(currentY);
        float fw = static_cast<float>(width);
        auto c0 = getColorAt(ctx, fx + 0.5f, fy + 0.5f);
        auto c1 = getColorAt(ctx, fx + fw - 0.5f, fy + 0.5f);
        c0.a *= correctAlpha(ctx, level);
        c1.a *= correctAlpha(ctx, level);
        glm::vec4 flat(0);
        addVertex(ctx, fx,      fy,       c0, 0, 0, flat);
        addVertex(ctx, fx + fw, fy,       c1, 0, 0, flat);
        addVertex(ctx, fx + fw, fy + 1.f, c1, 0, 0, flat);
        addVertex(ctx, fx,      fy,       c0, 0, 0, flat);
        addVertex(ctx, fx + fw, fy + 1.f, c1, 0, 0, flat);
        addVertex(ctx, fx,      fy + 1.f, c0, 0, 0, flat);
    }
};

} // jvk::graphics
