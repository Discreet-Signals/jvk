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
// JUCE passes logical point offsets in the current coordinate space.
// Transform through accumulated state transform, then scale to physical pixels.
inline void setOrigin(VulkanGraphicsContext& ctx, juce::Point<int> o)
{
    auto& s = ctx.state();
    float fx = static_cast<float>(o.x);
    float fy = static_cast<float>(o.y);
    s.transform.transformPoint(fx, fy);
    s.origin += juce::Point<int>(
        static_cast<int>(fx * ctx.scale),
        static_cast<int>(fy * ctx.scale));
}

inline void addTransform(VulkanGraphicsContext& ctx, const juce::AffineTransform& t)
{
    ctx.state().transform = ctx.state().transform.followedBy(t);
}

// ==== Clipping ====
// JUCE passes logical point rects. Transform through accumulated state + DPI scale.
inline bool clipToRectangle(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    flush(ctx);
    auto& s = ctx.state();
    auto physTransform = s.transform.scaled(ctx.scale)
                         .translated(static_cast<float>(s.origin.x),
                                     static_cast<float>(s.origin.y));
    auto physRect = r.toFloat().transformedBy(physTransform).getSmallestIntegerContainer();
    s.clipBounds = s.clipBounds.getIntersection(physRect);
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
    auto physTransform = s.transform.scaled(ctx.scale)
                         .translated(static_cast<float>(s.origin.x),
                                     static_cast<float>(s.origin.y));
    auto excluded = r.toFloat().transformedBy(physTransform).getSmallestIntegerContainer();
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
    auto combined = getFullTransform(ctx, t);
    auto pathBounds = path.getBoundsTransformed(combined).getSmallestIntegerContainer();
    ctx.state().clipBounds = ctx.state().clipBounds.getIntersection(pathBounds);
}

// Image-based alpha clipping is not supported in the GPU pipeline.
// Falls back to bounding-box clip (matches JUCE's OpenGL renderer behavior).
inline void clipToImageAlpha(VulkanGraphicsContext& ctx, const juce::Image& img, const juce::AffineTransform& t)
{
    if (img.isValid())
    {
        auto combined = getFullTransform(ctx, t);
        auto bounds = img.getBounds().toFloat().transformedBy(combined).getSmallestIntegerContainer();
        flush(ctx);
        ctx.state().clipBounds = ctx.state().clipBounds.getIntersection(bounds);
    }
}

inline bool clipRegionIntersects(VulkanGraphicsContext& ctx, const juce::Rectangle<int>& r)
{
    auto& s = ctx.state();
    auto physTransform = s.transform.scaled(ctx.scale)
                         .translated(static_cast<float>(s.origin.x),
                                     static_cast<float>(s.origin.y));
    auto physRect = r.toFloat().transformedBy(physTransform).getSmallestIntegerContainer();
    return s.clipBounds.intersects(physRect);
}

// Return logical point bounds (JUCE expects this).
// Invert the full transform (accumulated + DPI + origin) to map physical clip back to logical.
inline juce::Rectangle<int> getClipBounds(const VulkanGraphicsContext& ctx)
{
    auto& s = ctx.state();
    auto physTransform = s.transform.scaled(ctx.scale)
                         .translated(static_cast<float>(s.origin.x),
                                     static_cast<float>(s.origin.y));
    auto inv = physTransform.inverted();
    return s.clipBounds.toFloat().transformedBy(inv).getSmallestIntegerContainer();
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
