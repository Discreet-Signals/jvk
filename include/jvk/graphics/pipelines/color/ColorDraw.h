#pragma once

namespace jvk::pipelines {

// =============================================================================
// Vertex helpers
// =============================================================================

inline glm::vec4 colorToVec4(juce::Colour c, float opacity)
{
    return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), c.getFloatAlpha() * opacity };
}

inline void emitQuad(State& state, const DrawCommand& cmd,
                     float x, float y, float w, float h,
                     glm::vec4 color, glm::vec4 shape = glm::vec4(0))
{
    UIVertex verts[6] = {
        { {x,     y    }, color, {0, 0}, shape },
        { {x + w, y    }, color, {1, 0}, shape },
        { {x + w, y + h}, color, {1, 1}, shape },
        { {x,     y    }, color, {0, 0}, shape },
        { {x + w, y + h}, color, {1, 1}, shape },
        { {x,     y + h}, color, {0, 1}, shape },
    };
    state.draw(cmd, verts, 6);
}

// Transform a rect from component-local coords to physical pixel coords
inline juce::Rectangle<float> toPixels(const juce::Rectangle<float>& rect,
                                        const juce::AffineTransform& t, float scale)
{
    auto transformed = rect.transformedBy(t);
    return { transformed.getX() * scale, transformed.getY() * scale,
             transformed.getWidth() * scale, transformed.getHeight() * scale };
}

// =============================================================================
// ColorPipeline execute dispatch
// =============================================================================

inline void ColorPipeline::execute(Renderer& r, const Arena& arena, const DrawCommand& cmd)
{
    auto& state = r.state();

    switch (cmd.op) {
        case DrawOp::FillRect: {
            auto& p = arena.read<FillRectParams>(cmd.dataOffset);
            auto phys = toPixels(p.rect, p.transform, p.scale);

            state.setResource(r.caches().defaultDescriptor());
            emitQuad(state, cmd, phys.getX(), phys.getY(), phys.getWidth(), phys.getHeight(),
                     colorToVec4(r.getFill(p.fillIndex).colour, p.opacity));
            break;
        }

        case DrawOp::FillRoundedRect: {
            auto& p = arena.read<FillRoundedRectParams>(cmd.dataOffset);
            auto phys = toPixels(p.rect, p.transform, p.scale);
            float cs = p.cornerSize * p.scale;
            state.setResource(r.caches().defaultDescriptor());
            glm::vec4 shape(1.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f, cs);
            emitQuad(state, cmd, phys.getX(), phys.getY(), phys.getWidth(), phys.getHeight(),
                     colorToVec4(r.getFill(p.fillIndex).colour, p.opacity), shape);
            break;
        }

        case DrawOp::FillEllipse: {
            auto& p = arena.read<FillEllipseParams>(cmd.dataOffset);
            auto phys = toPixels(p.area, p.transform, p.scale);
            state.setResource(r.caches().defaultDescriptor());
            glm::vec4 shape(2.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f, 0);
            emitQuad(state, cmd, phys.getX(), phys.getY(), phys.getWidth(), phys.getHeight(),
                     colorToVec4(r.getFill(p.fillIndex).colour, p.opacity), shape);
            break;
        }

        case DrawOp::DrawImage: {
            auto& p = arena.read<DrawImageParams>(cmd.dataOffset);
            auto* cached = r.caches().textures().find(p.imageHash);
            auto desc = cached ? cached->descriptorSet : r.caches().defaultDescriptor();
            state.setResource(desc);
            // Apply transform to get image position in window coords, then scale to pixels
            float tx = p.transform.getTranslationX() * p.scale;
            float ty = p.transform.getTranslationY() * p.scale;
            float w = static_cast<float>(p.imageWidth) * p.scale;
            float h = static_cast<float>(p.imageHeight) * p.scale;
            emitQuad(state, cmd, tx, ty, w, h, glm::vec4(1, 1, 1, p.opacity), glm::vec4(3.0f, 0, 0, 0));
            break;
        }

        case DrawOp::DrawGlyphs: {
            auto& p = arena.read<DrawGlyphsParams>(cmd.dataOffset);
            auto& atlas = atlas_;
            uint32_t afterParams = cmd.dataOffset + static_cast<uint32_t>(sizeof(DrawGlyphsParams));
            auto glyphIds = arena.readSpan<uint16_t>(afterParams, p.glyphCount);
            auto positions = arena.readSpan<juce::Point<float>>(
                afterParams + p.glyphCount * sizeof(uint16_t), p.glyphCount);

            auto& font = r.getFont(p.fontIndex);
            auto color = colorToVec4(r.getFill(p.fillIndex).colour, p.opacity);
            float fontSize = font.getHeight() * p.scale;
            float originX = p.transform.getTranslationX() * p.scale;
            float originY = p.transform.getTranslationY() * p.scale;

            for (uint32_t i = 0; i < p.glyphCount; i++) {
                auto* typeface = font.getTypefacePtr().get();
                GlyphAtlas::GlyphKey key { typeface, glyphIds[i] };
                auto* entry = atlas.getGlyph(key, font);
                if (!entry) continue;

                float px = originX + positions[i].x * p.scale;
                float py = originY + positions[i].y * p.scale;
                float gw = entry->boundsW * fontSize;
                float gh = entry->boundsH * fontSize;
                float gx = px + entry->boundsX * fontSize;
                float gy = py + entry->boundsY * fontSize;

                VkDescriptorSet glyphDesc = atlas.getDescriptorSet(entry->atlasIndex);
                if (glyphDesc == VK_NULL_HANDLE) continue;
                state.setResource(glyphDesc);

                glm::vec4 shape(4.0f, 4.0f, 0, 0); // MSDF text: type=4, screenPxRange=4
                UIVertex verts[6] = {
                    { {gx,      gy     }, color, {entry->u0, entry->v0}, shape },
                    { {gx + gw, gy     }, color, {entry->u1, entry->v0}, shape },
                    { {gx + gw, gy + gh}, color, {entry->u1, entry->v1}, shape },
                    { {gx,      gy     }, color, {entry->u0, entry->v0}, shape },
                    { {gx + gw, gy + gh}, color, {entry->u1, entry->v1}, shape },
                    { {gx,      gy + gh}, color, {entry->u0, entry->v1}, shape },
                };
                state.draw(cmd, verts, 6);
            }
            break;
        }

        case DrawOp::DrawLine: {
            auto& p = arena.read<DrawLineParams>(cmd.dataOffset);
            state.setResource(r.caches().defaultDescriptor());
            auto color = colorToVec4(r.getFill(p.fillIndex).colour, p.opacity);
            // Apply transform then scale
            float tx = p.transform.getTranslationX();
            float ty = p.transform.getTranslationY();
            float x1 = (p.line.getStartX() + tx) * p.scale;
            float y1 = (p.line.getStartY() + ty) * p.scale;
            float x2 = (p.line.getEndX() + tx) * p.scale;
            float y2 = (p.line.getEndY() + ty) * p.scale;
            float hw = p.lineWidth * p.scale * 0.5f;

            float dx = x2 - x1, dy = y2 - y1;
            float len = std::sqrt(dx * dx + dy * dy);
            if (len < 0.001f) break;
            float nx = -dy / len * hw, ny = dx / len * hw;

            UIVertex verts[6] = {
                { {x1 + nx, y1 + ny}, color, {0, 0}, glm::vec4(0) },
                { {x1 - nx, y1 - ny}, color, {1, 0}, glm::vec4(0) },
                { {x2 - nx, y2 - ny}, color, {1, 1}, glm::vec4(0) },
                { {x1 + nx, y1 + ny}, color, {0, 0}, glm::vec4(0) },
                { {x2 - nx, y2 - ny}, color, {1, 1}, glm::vec4(0) },
                { {x2 + nx, y2 + ny}, color, {0, 1}, glm::vec4(0) },
            };
            state.draw(cmd, verts, 6);
            break;
        }

        default: break;
    }
}

} // namespace jvk::pipelines
