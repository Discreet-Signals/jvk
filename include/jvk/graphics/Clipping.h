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
 File: Clipping.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// ==== Transform ====
// JUCE passes logical point offsets. Scale to physical pixels.
inline void setOrigin(VulkanGraphicsContext& ctx, juce::Point<int> o)
{
    ctx.state().origin += juce::Point<int>(
        static_cast<int>(o.x * ctx.scale),
        static_cast<int>(o.y * ctx.scale));
}

inline void addTransform(VulkanGraphicsContext& ctx, const juce::AffineTransform& t)
{
    ctx.state().transform = ctx.state().transform.followedBy(t);
}

// ==== Clipping ====
// JUCE passes logical point rects. Scale to physical pixels.
inline bool clipToRectangle(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    flush(ctx);
    auto& s = ctx.state();
    auto scaled = scaleRect(ctx, r);
    s.clipBounds = s.clipBounds.getIntersection(scaled.translated(s.origin.x, s.origin.y));
    return !s.clipBounds.isEmpty();
}

// Clips to bounding box of the rectangle list.
// This is the standard GPU approach (matches JUCE's OpenGL renderer).
inline bool clipToRectangleList(VulkanGraphicsContext& ctx, const juce::RectangleList<int>& r)
{
    return clipToRectangle(ctx, r.getBounds());
}

inline void excludeClipRectangle(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    flush(ctx);
    auto& s = ctx.state();
    auto excluded = scaleRect(ctx, r).translated(s.origin.x, s.origin.y);
    if (!s.clipBounds.intersects(excluded)) return;

    // With a single scissor rect, we can only exclude regions that touch
    // an edge of the clip bounds. Pick the largest remaining rectangle.
    bool spansWidth  = excluded.getX() <= s.clipBounds.getX() && excluded.getRight() >= s.clipBounds.getRight();
    bool spansHeight = excluded.getY() <= s.clipBounds.getY() && excluded.getBottom() >= s.clipBounds.getBottom();

    if (spansWidth)
    {
        // Exclusion spans full width — shrink vertically
        if (excluded.getY() <= s.clipBounds.getY())
            s.clipBounds.setTop(excluded.getBottom());
        else if (excluded.getBottom() >= s.clipBounds.getBottom())
            s.clipBounds.setBottom(excluded.getY());
    }
    else if (spansHeight)
    {
        // Exclusion spans full height — shrink horizontally
        if (excluded.getX() <= s.clipBounds.getX())
            s.clipBounds.setLeft(excluded.getRight());
        else if (excluded.getRight() >= s.clipBounds.getRight())
            s.clipBounds.setRight(excluded.getX());
    }
    // For arbitrary interior exclusions, we'd need stencil clipping.
    // The remaining visible area is approximated by the unchanged clip bounds.
}

inline void clipToPath(VulkanGraphicsContext& ctx, const juce::Path& path, const juce::AffineTransform& t)
{
    auto& s = ctx.state();
    auto combined = t.scaled(ctx.scale).translated(
        static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
    auto pathBounds = path.getBoundsTransformed(combined).getSmallestIntegerContainer();
    s.clipBounds = s.clipBounds.getIntersection(pathBounds);
}

// Image-based alpha clipping is not supported in the GPU pipeline.
// Falls back to bounding-box clip (matches JUCE's OpenGL renderer behavior).
inline void clipToImageAlpha(VulkanGraphicsContext& ctx, const juce::Image& img, const juce::AffineTransform& t)
{
    if (img.isValid())
    {
        auto bounds = img.getBounds().toFloat().transformedBy(t).getSmallestIntegerContainer();
        clipToRectangle(ctx, bounds);
    }
}

inline bool clipRegionIntersects(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    auto scaled = scaleRect(ctx, r);
    return ctx.state().clipBounds.intersects(scaled.translated(ctx.state().origin.x, ctx.state().origin.y));
}

// Return logical point bounds (JUCE expects this)
inline juce::Rectangle<int> getClipBounds(const VulkanGraphicsContext& ctx)
{
    auto physBounds = ctx.state().clipBounds.translated(-ctx.state().origin.x, -ctx.state().origin.y);
    float inv = 1.0f / ctx.scale;
    return juce::Rectangle<int>(
        static_cast<int>(physBounds.getX() * inv),
        static_cast<int>(physBounds.getY() * inv),
        static_cast<int>(std::ceil(physBounds.getWidth() * inv)),
        static_cast<int>(std::ceil(physBounds.getHeight() * inv)));
}

inline bool isClipEmpty(const VulkanGraphicsContext& ctx)
{
    return ctx.state().clipBounds.isEmpty();
}

// ==== State stack ====
// Flush pending vertices before state changes so they use the correct scissor.
inline void saveState(VulkanGraphicsContext& ctx)
{
    flush(ctx);
    ctx.stateStack.push_back(ctx.stateStack.back());
}

inline void restoreState(VulkanGraphicsContext& ctx)
{
    flush(ctx);
    if (ctx.stateStack.size() > 1) ctx.stateStack.pop_back();
}

// ==== Transparency layers ====
inline void beginTransparencyLayer(VulkanGraphicsContext& ctx, float opacity)
{
    saveState(ctx);
    ctx.state().opacity *= opacity;
}

inline void endTransparencyLayer(VulkanGraphicsContext& ctx)
{
    restoreState(ctx);
}

} // jvk::graphics
