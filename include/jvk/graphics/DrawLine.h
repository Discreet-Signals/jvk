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
 File: DrawLine.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

inline void drawLineWithThickness(VulkanGraphicsContext& ctx,
                                   const juce::Line<float>& line, float lineThickness)
{
    if (isClipEmpty(ctx)) return;
    auto t = getFullTransform(ctx);
    auto c = getColorForFill(ctx);

    auto p1 = line.getStart().transformedBy(t);
    auto p2 = line.getEnd().transformedBy(t);
    float dx = p2.x - p1.x, dy = p2.y - p1.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 0.001f) return;
    float hw = lineThickness * ctx.scale * 0.5f;
    float nx = -dy / len * hw, ny = dx / len * hw;

    addVertex(ctx, p1.x+nx, p1.y+ny, c); addVertex(ctx, p2.x+nx, p2.y+ny, c); addVertex(ctx, p2.x-nx, p2.y-ny, c);
    addVertex(ctx, p1.x+nx, p1.y+ny, c); addVertex(ctx, p2.x-nx, p2.y-ny, c); addVertex(ctx, p1.x-nx, p1.y-ny, c);
}

inline void drawLine(VulkanGraphicsContext& ctx, const juce::Line<float>& line)
{
    drawLineWithThickness(ctx, line, 1.0f);
}

} // jvk::graphics
