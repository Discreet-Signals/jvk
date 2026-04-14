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
    auto& ct = ctx.state().complexTransform;
    auto adjusted = r.translated(ct.getTranslationX(), ct.getTranslationY());
    auto phys = juce::Rectangle<float>(adjusted.getX() * ctx.scale, adjusted.getY() * ctx.scale,
                                        adjusted.getWidth() * ctx.scale, adjusted.getHeight() * ctx.scale);
    float hw = phys.getWidth() * 0.5f, hh = phys.getHeight() * 0.5f;
    addSDFQuad(ctx, phys.getX(), phys.getY(), phys.getWidth(), phys.getHeight(),
                getColorForFill(ctx), 1.0f, hw, hh, cornerSize * ctx.scale);
}

inline void fillEllipse(VulkanGraphicsContext& ctx, const juce::Rectangle<float>& area)
{
    if (isClipEmpty(ctx)) return;
    if (hasNonTrivialTransform(ctx)) return; // caller handles fallback
    auto& ct = ctx.state().complexTransform;
    auto adjusted = area.translated(ct.getTranslationX(), ct.getTranslationY());
    auto phys = juce::Rectangle<float>(adjusted.getX() * ctx.scale, adjusted.getY() * ctx.scale,
                                        adjusted.getWidth() * ctx.scale, adjusted.getHeight() * ctx.scale);
    float hw = phys.getWidth() * 0.5f, hh = phys.getHeight() * 0.5f;
    addSDFQuad(ctx, phys.getX(), phys.getY(), phys.getWidth(), phys.getHeight(),
                getColorForFill(ctx), 2.0f, hw, hh, 0.0f);
}

} // jvk::graphics
