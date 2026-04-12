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
 File: DrawGlyphs.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// ==== Drawing: drawGlyphs ====
inline void drawGlyphs(VulkanGraphicsContext& ctx,
                        juce::Span<const uint16_t> glyphs,
                        juce::Span<const juce::Point<float>> positions,
                        const juce::AffineTransform& t)
{
    if (isClipEmpty(ctx) || glyphs.empty()) return;
    auto& s = ctx.state();

    // GPU MSDF atlas path: each glyph = 1 textured quad, resolution-independent
    if (ctx.glyphAtlas)
    {
        auto* typeface = s.font.getTypefacePtr().get();
        auto fontHeight = s.font.getHeight();
        auto baseColor = getColorForFill(ctx);
        float hScale = s.font.getHorizontalScale();

        // Physical font size determines quad size on screen
        float physFontHeight = fontHeight * ctx.scale;

        VkDescriptorSet currentAtlasDescSet = VK_NULL_HANDLE;

        for (size_t idx = 0; idx < glyphs.size(); ++idx)
        {
            // MSDF atlas is resolution-independent: one entry per glyph per typeface
            GlyphAtlas::GlyphKey key { typeface, glyphs[idx] };
            auto* entry = ctx.glyphAtlas->getGlyph(key, s.font);
            if (!entry) continue;

            auto descSet = ctx.glyphAtlas->getDescriptorSet(entry->atlasIndex);
            if (descSet == VK_NULL_HANDLE) continue;

            if (descSet != currentAtlasDescSet)
            {
                flush(ctx);
                ensureDescriptorSet(ctx, ctx.pipelineLayout, descSet);
                currentAtlasDescSet = descSet;
            }

            // Glyph position in physical pixels (applies addTransform + scale + origin)
            auto fullT = getFullTransform(ctx, t);
            auto glyphPos = positions[idx].transformedBy(fullT);

            // The entry bounds are in normalized font units (height=1.0).
            // Scale to physical pixels for screen positioning.
            float quadX = glyphPos.x + entry->boundsX * physFontHeight * hScale
                        - GlyphAtlas::GLYPH_PADDING * (entry->boundsW * physFontHeight * hScale / static_cast<float>(entry->w - GlyphAtlas::GLYPH_PADDING * 2));
            float quadY = glyphPos.y + entry->boundsY * physFontHeight
                        - GlyphAtlas::GLYPH_PADDING * (entry->boundsH * physFontHeight / static_cast<float>(entry->h - GlyphAtlas::GLYPH_PADDING * 2));

            // Quad size: scale the MSDF texel dimensions to screen pixels
            // The inner glyph region maps to boundsW*fontHeight x boundsH*fontHeight
            // Plus padding on each side
            float texelsPerUnit = static_cast<float>(entry->w) / (entry->boundsW + 2.0f * GlyphAtlas::GLYPH_PADDING / (entry->boundsW > 0 ? (static_cast<float>(entry->w - GlyphAtlas::GLYPH_PADDING * 2) / entry->boundsW) : 1.0f));
            float gw = static_cast<float>(entry->w) * (entry->boundsW * physFontHeight * hScale) / static_cast<float>(entry->w - GlyphAtlas::GLYPH_PADDING * 2);
            float gh = static_cast<float>(entry->h) * (entry->boundsH * physFontHeight) / static_cast<float>(entry->h - GlyphAtlas::GLYPH_PADDING * 2);

            juce::ignoreUnused(texelsPerUnit);

            // Screen pixel range for MSDF: how many screen pixels the distance range covers
            float screenPxRange = std::max(1.0f,
                static_cast<float>(GlyphAtlas::MSDF_PIXEL_RANGE) * (gw / static_cast<float>(entry->w)));

            // Type 4: MSDF text, shapeInfo.y = screenPxRange
            glm::vec4 msdfType(4.0f, screenPxRange, 0.0f, 0);
            addVertex(ctx, quadX,      quadY,      baseColor, entry->u0, entry->v0, msdfType);
            addVertex(ctx, quadX + gw, quadY,      baseColor, entry->u1, entry->v0, msdfType);
            addVertex(ctx, quadX + gw, quadY + gh, baseColor, entry->u1, entry->v1, msdfType);
            addVertex(ctx, quadX,      quadY,      baseColor, entry->u0, entry->v0, msdfType);
            addVertex(ctx, quadX + gw, quadY + gh, baseColor, entry->u1, entry->v1, msdfType);
            addVertex(ctx, quadX,      quadY + gh, baseColor, entry->u0, entry->v1, msdfType);
        }

        flush(ctx);
        ensureDescriptorSet(ctx, ctx.pipelineLayout, ctx.defaultDescriptorSet);
    }
}

} // jvk::graphics
