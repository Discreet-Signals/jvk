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
 File: FillShapes.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// ==== SDF Shape Overrides (bypass fillPath stencil) ====

inline void fillRoundedRectangle(VulkanGraphicsContext& ctx,
                                  const juce::Rectangle<float>& r, float cornerSize)
{
    if (isClipEmpty(ctx)) return;
    // SDF rounded rect only works axis-aligned; fall back to path for transforms
    if (hasNonTrivialTransform(ctx)) return; // caller handles fallback
    auto& s = ctx.state();
    auto adjusted = r;
    if (!s.transform.isIdentity())
        adjusted = adjusted.translated(s.transform.getTranslationX(), s.transform.getTranslationY());
    auto phys = juce::Rectangle<float>(adjusted.getX() * ctx.scale, adjusted.getY() * ctx.scale,
                                        adjusted.getWidth() * ctx.scale, adjusted.getHeight() * ctx.scale);
    auto tr = phys.translated(static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
    float hw = tr.getWidth() * 0.5f, hh = tr.getHeight() * 0.5f;
    addSDFQuad(ctx, tr.getX(), tr.getY(), tr.getWidth(), tr.getHeight(),
                getColorForFill(ctx), 1.0f, hw, hh, cornerSize * ctx.scale);
}

inline void fillEllipse(VulkanGraphicsContext& ctx, const juce::Rectangle<float>& area)
{
    if (isClipEmpty(ctx)) return;
    // SDF ellipse only works axis-aligned; fall back to path for transforms
    if (hasNonTrivialTransform(ctx)) return; // caller handles fallback
    auto& s = ctx.state();
    auto adjusted = area;
    if (!s.transform.isIdentity())
        adjusted = adjusted.translated(s.transform.getTranslationX(), s.transform.getTranslationY());
    auto phys = juce::Rectangle<float>(adjusted.getX() * ctx.scale, adjusted.getY() * ctx.scale,
                                        adjusted.getWidth() * ctx.scale, adjusted.getHeight() * ctx.scale);
    auto tr = phys.translated(static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
    float hw = tr.getWidth() * 0.5f, hh = tr.getHeight() * 0.5f;
    addSDFQuad(ctx, tr.getX(), tr.getY(), tr.getWidth(), tr.getHeight(),
                getColorForFill(ctx), 2.0f, hw, hh, 0.0f);
}

} // jvk::graphics
