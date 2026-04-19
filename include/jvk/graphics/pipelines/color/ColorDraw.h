#pragma once

namespace jvk::pipelines {

// =============================================================================
// Unified fill model — "shape × color" composition
//
// Every draw op is the product of two independent sources:
//   - A color source: solid vertex color, or a 256x1 gradient LUT sampled by a
//     per-vertex `gradientT`. Gradient LUTs are built at record time from
//     juce::ColourGradient via getColourAtPosition(t), so N-stop gradients
//     (start, middle colours added via addColour, end) work natively.
//   - A shape source: analytical SDF (rect / rounded rect / ellipse), MSDF
//     atlas (text), or sampled image.
//
// Descriptor set 0 holds the colour LUT (or the 1x1 default for solid fills —
// unused when gradientInfo.z == 0), set 1 holds the shape texture. Each op
// computes per-vertex gradientT on CPU so radial/linear behave correctly under
// arbitrary transforms.
// =============================================================================

struct GradientCtx {
    float mode    = 0.0f;         // 0 = solid, 1 = linear, 2 = radial
    float rowNorm = 0.0f;         // normalised y into the GradientAtlas (row center)
    float gx1 = 0, gy1 = 0;       // gradient origin in physical pixels
    // linear projection
    float dx = 0, dy = 0, invLen2 = 0;
    // radial normalization
    float invRadius = 0;

    bool active() const { return mode > 0.0f; }
};

inline GradientCtx makeGradientCtx(Renderer& r, const juce::FillType& fill,
                                   const juce::AffineTransform& physical)
{
    GradientCtx c;
    if (!fill.isGradient() || !fill.gradient) return c;
    auto& g = *fill.gradient;

    // Idempotent w.r.t. the hash — if Graphics.h already registered the
    // gradient during recording, this just re-reads the cached row.
    c.rowNorm = r.caches().registerGradient(g);

    auto gradT = fill.transform.followedBy(physical);
    float x1 = g.point1.x, y1 = g.point1.y;
    float x2 = g.point2.x, y2 = g.point2.y;
    gradT.transformPoint(x1, y1);
    gradT.transformPoint(x2, y2);

    c.gx1 = x1;
    c.gy1 = y1;
    if (g.isRadial) {
        c.mode = 2.0f;
        float radius = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        c.invRadius = (radius > 0) ? 1.0f / radius : 0.0f;
    } else {
        c.mode = 1.0f;
        c.dx = x2 - x1;
        c.dy = y2 - y1;
        float len2 = c.dx * c.dx + c.dy * c.dy;
        c.invLen2 = (len2 > 0) ? 1.0f / len2 : 0.0f;
    }
    return c;
}

// Produces the vertex gradientInfo: (t.x, t.y, mode, rowNorm).
// Shader reads t.x for linear, length(t.xy) for radial; samples the colour
// atlas at (t, rowNorm).
inline glm::vec4 gradientAt(const GradientCtx& c, float px, float py)
{
    if (c.mode == 2.0f) {
        return glm::vec4((px - c.gx1) * c.invRadius,
                         (py - c.gy1) * c.invRadius, 2.0f, c.rowNorm);
    }
    if (c.mode == 1.0f) {
        float t = ((px - c.gx1) * c.dx + (py - c.gy1) * c.dy) * c.invLen2;
        return glm::vec4(t, 0.0f, 1.0f, c.rowNorm);
    }
    return glm::vec4(0.0f);
}

// Effective color attribute: for solid, carries the colour + opacity; for
// gradient, white with opacity (the LUT sample already supplies RGB, opacity
// multiplies in the shader).
inline glm::vec4 fillColorAttr(const GradientCtx& grad, juce::Colour solid, float opacity)
{
    if (grad.active())
        return glm::vec4(1.0f, 1.0f, 1.0f, opacity);
    return { solid.getFloatRed(), solid.getFloatGreen(),
             solid.getFloatBlue(), solid.getFloatAlpha() * opacity };
}

inline VkDescriptorSet colorDescFor(const GradientCtx& grad, Renderer& r)
{
    // All active gradients share the atlas; solids get the 1x1 default which
    // the shader doesn't sample when gradientInfo.z == 0.
    return grad.active() ? r.caches().gradientDescriptor()
                         : r.caches().defaultDescriptor();
}

// =============================================================================
// Quad emitters — each vertex gets its own gradientT based on its physical pos
// =============================================================================

inline void emitQuad(State& state, const DrawCommand& cmd,
                     float x, float y, float w, float h,
                     glm::vec4 color, glm::vec4 shape = glm::vec4(0),
                     const GradientCtx* grad = nullptr)
{
    auto gi = [&](float px, float py) {
        return (grad && grad->active()) ? gradientAt(*grad, px, py) : glm::vec4(0.0f);
    };
    UIVertex verts[6] = {
        { {x,     y    }, color, {0, 0}, shape, gi(x,     y    ) },
        { {x + w, y    }, color, {1, 0}, shape, gi(x + w, y    ) },
        { {x + w, y + h}, color, {1, 1}, shape, gi(x + w, y + h) },
        { {x,     y    }, color, {0, 0}, shape, gi(x,     y    ) },
        { {x + w, y + h}, color, {1, 1}, shape, gi(x + w, y + h) },
        { {x,     y + h}, color, {0, 1}, shape, gi(x,     y + h) },
    };
    state.draw(cmd, verts, 6);
}

// Emit a quad with custom UVs (e.g. for SDF AA-fringe padding)
inline void emitQuadUV(State& state, const DrawCommand& cmd,
                       float x, float y, float w, float h,
                       float u0, float v0, float u1, float v1,
                       glm::vec4 color, glm::vec4 shape,
                       const GradientCtx* grad = nullptr)
{
    auto gi = [&](float px, float py) {
        return (grad && grad->active()) ? gradientAt(*grad, px, py) : glm::vec4(0.0f);
    };
    UIVertex verts[6] = {
        { {x,     y    }, color, {u0, v0}, shape, gi(x,     y    ) },
        { {x + w, y    }, color, {u1, v0}, shape, gi(x + w, y    ) },
        { {x + w, y + h}, color, {u1, v1}, shape, gi(x + w, y + h) },
        { {x,     y    }, color, {u0, v0}, shape, gi(x,     y    ) },
        { {x + w, y + h}, color, {u1, v1}, shape, gi(x + w, y + h) },
        { {x,     y + h}, color, {u0, v1}, shape, gi(x,     y + h) },
    };
    state.draw(cmd, verts, 6);
}

inline void emitTransformedQuad(State& state, const DrawCommand& cmd,
                                juce::Point<float> p00, juce::Point<float> p10,
                                juce::Point<float> p11, juce::Point<float> p01,
                                glm::vec4 color, glm::vec4 shape = glm::vec4(0),
                                const GradientCtx* grad = nullptr)
{
    auto gi = [&](juce::Point<float> p) {
        return (grad && grad->active()) ? gradientAt(*grad, p.x, p.y) : glm::vec4(0.0f);
    };
    UIVertex verts[6] = {
        { {p00.x, p00.y}, color, {0, 0}, shape, gi(p00) },
        { {p10.x, p10.y}, color, {1, 0}, shape, gi(p10) },
        { {p11.x, p11.y}, color, {1, 1}, shape, gi(p11) },
        { {p00.x, p00.y}, color, {0, 0}, shape, gi(p00) },
        { {p11.x, p11.y}, color, {1, 1}, shape, gi(p11) },
        { {p01.x, p01.y}, color, {0, 1}, shape, gi(p01) },
    };
    state.draw(cmd, verts, 6);
}

inline juce::AffineTransform toPhysical(const juce::AffineTransform& t, float scale)
{
    return t.scaled(scale);
}

// Pack two floats into a single float via half-precision encoding.
// Lossless for values whose magnitude fits in fp16 (~65k). Used by stroke
// shape codes (type 8 / 9) to carry cornerSize + strokeWidth in shapeInfo.w,
// which the shader recovers with unpackHalf2x16(floatBitsToUint(w)).
inline float packHalf2(float x, float y)
{
    uint32_t packed = glm::packHalf2x16(glm::vec2(x, y));
    float result;
    memcpy(&result, &packed, sizeof(float));
    return result;
}

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
    auto  def   = r.caches().defaultDescriptor();

    switch (cmd.op) {
        case DrawOp::FillRect: {
            auto& p    = arena.read<FillRectParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);
            glm::vec4 shape(0.0f); // flat fill — no SDF

            if (tx.isOnlyTranslation()) {
                auto phys = toPixels(p.rect, p.transform, p.scale);
                emitQuad(state, cmd, phys.getX(), phys.getY(),
                         phys.getWidth(), phys.getHeight(), color, shape, &grad);
            } else {
                auto p00 = juce::Point<float>(p.rect.getX(),     p.rect.getY()   ).transformedBy(tx);
                auto p10 = juce::Point<float>(p.rect.getRight(), p.rect.getY()   ).transformedBy(tx);
                auto p11 = juce::Point<float>(p.rect.getRight(), p.rect.getBottom()).transformedBy(tx);
                auto p01 = juce::Point<float>(p.rect.getX(),     p.rect.getBottom()).transformedBy(tx);
                emitTransformedQuad(state, cmd, p00, p10, p11, p01, color, shape, &grad);
            }
            break;
        }

        case DrawOp::FillRoundedRect: {
            auto& p    = arena.read<FillRoundedRectParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            auto  phys = toPixels(p.rect, p.transform, p.scale);
            // cornerSize is in LOGICAL pixels — scale it by the composed
            // transform (affine × displayScale) so corners grow with the
            // rect under any transform zoom. Using p.scale alone only
            // covers displayScale and leaves corners under-sized whenever
            // the caller set a scale > 1 on Graphics::addTransform.
            float cs   = p.cornerSize * tx.getScaleFactor();
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);
            glm::vec4 shape(1.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f, cs);

            // Expand quad by `pad` on each side. UV must stay anchored to the
            // unpadded extents so that UV=(0,0)/(1,1) map to the rect corners
            // in the shader's local frame. That means uPad = pad / phys.w, not
            // pad / (phys.w + 2*pad) — the shader's halfSize is unpadded.
            float pad = 1.0f;
            float ex = phys.getX() - pad, ey = phys.getY() - pad;
            float ew = phys.getWidth() + pad * 2, eh = phys.getHeight() + pad * 2;
            float uPad = pad / phys.getWidth(), vPad = pad / phys.getHeight();
            emitQuadUV(state, cmd, ex, ey, ew, eh,
                       -uPad, -vPad, 1.0f + uPad, 1.0f + vPad,
                       color, shape, &grad);
            break;
        }

        case DrawOp::FillEllipse: {
            auto& p    = arena.read<FillEllipseParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            auto  phys = toPixels(p.area, p.transform, p.scale);
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);
            glm::vec4 shape(2.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f, 0.0f);

            float pad = 1.0f;
            float ex = phys.getX() - pad, ey = phys.getY() - pad;
            float ew = phys.getWidth() + pad * 2, eh = phys.getHeight() + pad * 2;
            float uPad = pad / phys.getWidth(), vPad = pad / phys.getHeight();
            emitQuadUV(state, cmd, ex, ey, ew, eh,
                       -uPad, -vPad, 1.0f + uPad, 1.0f + vPad,
                       color, shape, &grad);
            break;
        }

        case DrawOp::StrokeRoundedRect: {
            auto& p    = arena.read<StrokeRoundedRectParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            auto  phys = toPixels(p.rect, p.transform, p.scale);
            // Both cornerSize and lineWidth are logical pixels — scale by
            // the composed transform so they track the rect under zoom.
            float cs   = p.cornerSize * tx.getScaleFactor();
            float sw   = p.lineWidth  * tx.getScaleFactor();
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);
            // Type 8 = stroked rounded rect. Shader: abs(roundedRectSDF) - sw/2.
            glm::vec4 shape(8.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f,
                            packHalf2(cs, sw));

            // Expand quad by sw/2 (to contain outer stroke edge) + 1px AA fringe.
            // UV stays anchored to the unpadded rect so the shader sees the
            // outer edge at localP = ±(halfW + sw/2).
            float pad = sw * 0.5f + 1.0f;
            float ex = phys.getX() - pad, ey = phys.getY() - pad;
            float ew = phys.getWidth() + pad * 2, eh = phys.getHeight() + pad * 2;
            float uPad = pad / phys.getWidth(), vPad = pad / phys.getHeight();
            emitQuadUV(state, cmd, ex, ey, ew, eh,
                       -uPad, -vPad, 1.0f + uPad, 1.0f + vPad,
                       color, shape, &grad);
            break;
        }

        case DrawOp::StrokeEllipse: {
            auto& p    = arena.read<StrokeEllipseParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            auto  phys = toPixels(p.area, p.transform, p.scale);
            // lineWidth is in logical pixels — use the composed transform
            // scale (matches DrawLine's convention).
            float sw   = p.lineWidth * tx.getScaleFactor();
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);
            // Type 9 = stroked ellipse. Shader: abs(ellipseSDF) - sw/2.
            glm::vec4 shape(9.0f, phys.getWidth() * 0.5f, phys.getHeight() * 0.5f,
                            packHalf2(0.0f, sw));

            float pad = sw * 0.5f + 1.0f;
            float ex = phys.getX() - pad, ey = phys.getY() - pad;
            float ew = phys.getWidth() + pad * 2, eh = phys.getHeight() + pad * 2;
            float uPad = pad / phys.getWidth(), vPad = pad / phys.getHeight();
            emitQuadUV(state, cmd, ex, ey, ew, eh,
                       -uPad, -vPad, 1.0f + uPad, 1.0f + vPad,
                       color, shape, &grad);
            break;
        }

        case DrawOp::DrawImage: {
            auto& p     = arena.read<DrawImageParams>(cmd.dataOffset);
            auto* tcache = r.caches().textures().find(p.imageHash);
            auto  shapeDesc = tcache ? tcache->descriptorSet : def;
            // Images are self-colored; color LUT is ignored (solid white tint).
            state.setResources(def, shapeDesc);
            auto  tx = toPhysical(p.transform, p.scale);
            float w = static_cast<float>(p.imageWidth);
            float h = static_cast<float>(p.imageHeight);
            auto p00 = juce::Point<float>(0, 0).transformedBy(tx);
            auto p10 = juce::Point<float>(w, 0).transformedBy(tx);
            auto p11 = juce::Point<float>(w, h).transformedBy(tx);
            auto p01 = juce::Point<float>(0, h).transformedBy(tx);
            emitTransformedQuad(state, cmd, p00, p10, p11, p01,
                                glm::vec4(1, 1, 1, p.opacity),
                                glm::vec4(3.0f, 0, 0, 0));
            break;
        }

        case DrawOp::DrawGlyphs: {
            // ================================================================
            // MSDF extensions roadmap (jvk::Graphics surface, beyond juce::Graphics)
            //
            // The atlas stores a true multi-channel signed distance field per
            // glyph, which gives us several rendering knobs for free — no
            // atlas re-rasterization, just per-draw shader parameters. All
            // of these currently live in shaders/ui2d.frag (type 4 branch);
            // exposing them requires extending DrawGlyphsParams + Graphics.h:
            //
            //   - Weight / stroke width
            //       shift the MSDF 0.5 threshold (already wired as
            //       WEIGHT_SHIFT in the shader) to render variable-weight
            //       text: light/regular/bold/black from ONE atlas page.
            //   - Outline / stroke
            //       render `abs(signedDist - threshold) - halfStroke < 0`
            //       to draw a stroked glyph outline of arbitrary thickness.
            //   - Drop shadow / glow
            //       sample the MSDF at an offset UV with a smoother
            //       falloff; composite shadow then glyph in one draw.
            //   - Glyph-space gradient fills
            //       already works via the unified gradient path — per-glyph
            //       gradient shading with no extra code.
            //
            // These are all juce::Graphics-surface-incompatible (JUCE has no
            // concept of variable text weight or glyph outline), so they'd
            // ship as jvk::Graphics extensions callable via
            // Graphics::create(g)->drawTextWeighted(...) etc., matching the
            // darken/brighten/tint/blur extension pattern.
            // ================================================================
            auto& p = arena.read<DrawGlyphsParams>(cmd.dataOffset);
            auto& atlas = atlas_;
            uint32_t afterParams = cmd.dataOffset + static_cast<uint32_t>(sizeof(DrawGlyphsParams));
            auto glyphIds = arena.readSpan<uint16_t>(afterParams, p.glyphCount);
            uint32_t posOffset = afterParams + p.glyphCount * static_cast<uint32_t>(sizeof(uint16_t));
            posOffset = (posOffset + 3u) & ~3u;
            auto positions = arena.readSpan<juce::Point<float>>(posOffset, p.glyphCount);

            auto& font = r.getFont(p.fontIndex);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            // Color LUT is shared across every glyph of this draw; shape
            // descriptor varies per glyph (atlas page).
            state.setColorResource(colorDescFor(grad, r));

            auto color = fillColorAttr(grad, fill.colour, p.opacity);

            float physFontH = font.getHeight() * tx.getScaleFactor();
            float hScale = font.getHorizontalScale();
            constexpr int PAD = GlyphAtlas::GLYPH_PADDING;

            for (uint32_t i = 0; i < p.glyphCount; i++) {
                auto* typeface = font.getTypefacePtr().get();
                GlyphAtlas::GlyphKey key { typeface, glyphIds[i] };
                auto* entry = atlas.getGlyph(key, font);
                if (!entry) continue;

                VkDescriptorSet glyphDesc = atlas.getDescriptorSet(entry->atlasIndex);
                if (glyphDesc == VK_NULL_HANDLE) continue;
                state.setShapeResource(glyphDesc);

                auto glyphPos = positions[i].transformedBy(tx);
                float innerW = entry->boundsW * physFontH * hScale;
                float innerH = entry->boundsH * physFontH;
                int innerTexW = entry->w - PAD * 2;
                int innerTexH = entry->h - PAD * 2;

                float gw = static_cast<float>(entry->w) * innerW / static_cast<float>(innerTexW);
                float gh = static_cast<float>(entry->h) * innerH / static_cast<float>(innerTexH);
                float padScreenX = PAD * (innerW / static_cast<float>(innerTexW));
                float padScreenY = PAD * (innerH / static_cast<float>(innerTexH));
                float gx = glyphPos.x + entry->boundsX * physFontH * hScale - padScreenX;
                float gy = glyphPos.y + entry->boundsY * physFontH - padScreenY;

                float screenPxRange = std::max(1.0f,
                    static_cast<float>(GlyphAtlas::MSDF_PIXEL_RANGE) * (gw / static_cast<float>(entry->w)));
                glm::vec4 shape(4.0f, screenPxRange, 0, 0);

                auto gi = [&](float px, float py) {
                    return grad.active() ? gradientAt(grad, px, py) : glm::vec4(0.0f);
                };
                UIVertex verts[6] = {
                    { {gx,      gy     }, color, {entry->u0, entry->v0}, shape, gi(gx,      gy     ) },
                    { {gx + gw, gy     }, color, {entry->u1, entry->v0}, shape, gi(gx + gw, gy     ) },
                    { {gx + gw, gy + gh}, color, {entry->u1, entry->v1}, shape, gi(gx + gw, gy + gh) },
                    { {gx,      gy     }, color, {entry->u0, entry->v0}, shape, gi(gx,      gy     ) },
                    { {gx + gw, gy + gh}, color, {entry->u1, entry->v1}, shape, gi(gx + gw, gy + gh) },
                    { {gx,      gy + gh}, color, {entry->u0, entry->v1}, shape, gi(gx,      gy + gh) },
                };
                state.draw(cmd, verts, 6);
            }
            break;
        }

        case DrawOp::DrawLine: {
            auto& p    = arena.read<DrawLineParams>(cmd.dataOffset);
            auto& fill = r.getFill(p.fillIndex);
            auto  tx   = toPhysical(p.transform, p.scale);
            GradientCtx grad = makeGradientCtx(r, fill, tx);

            state.setResources(colorDescFor(grad, r), def);
            auto color = fillColorAttr(grad, fill.colour, p.opacity);

            float x1 = p.line.getStartX(), y1 = p.line.getStartY();
            float x2 = p.line.getEndX(),   y2 = p.line.getEndY();
            tx.transformPoint(x1, y1);
            tx.transformPoint(x2, y2);

            float dx = x2 - x1, dy = y2 - y1;
            float len = std::sqrt(dx * dx + dy * dy);
            if (len < 0.001f) break;

            // Capsule (rounded line): quad oriented along the line direction.
            // Local axes: along = +X, perp = +Y (in the shader's local frame).
            float halfLen = len * 0.5f;
            float radius  = p.lineWidth * tx.getScaleFactor() * 0.5f;
            float pad     = 1.0f; // AA fringe only; quad already contains the caps

            float ax = dx / len,  ay = dy / len;    // along
            float px = -ay,       py = ax;          // perpendicular
            float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;

            // Quad corners in physical pixels, extended by pad on each axis.
            float ah = halfLen + pad;
            float rh = radius  + pad;
            auto at = [&](float a, float s) {
                return juce::Point<float> { cx + ax * a + px * s, cy + ay * a + py * s };
            };
            auto p00 = at(-ah, -rh);
            auto p10 = at(+ah, -rh);
            auto p11 = at(+ah, +rh);
            auto p01 = at(-ah, +rh);

            // Type 10 = capsule. Shader: length(vec2(max(abs(p.x)-halfLen, 0), p.y)) - radius
            // shapeInfo.yz = (halfLen, radius) in the capsule's local frame. UV is
            // anchored to the unpadded extents — uPad = pad / (2*halfLen) — so the
            // shader's localP = (UV-0.5)*halfSize*2 reaches ±(halfLen+pad, radius+pad)
            // exactly at the quad corners.
            glm::vec4 shape(10.0f, halfLen, radius, 0.0f);
            float uPad = pad / (halfLen * 2.0f);
            float vPad = pad / (radius  * 2.0f);

            auto gi = [&](juce::Point<float> q) {
                return grad.active() ? gradientAt(grad, q.x, q.y) : glm::vec4(0.0f);
            };
            UIVertex verts[6] = {
                { {p00.x, p00.y}, color, {-uPad,         -vPad        }, shape, gi(p00) },
                { {p10.x, p10.y}, color, {1.0f + uPad,   -vPad        }, shape, gi(p10) },
                { {p11.x, p11.y}, color, {1.0f + uPad,   1.0f + vPad  }, shape, gi(p11) },
                { {p00.x, p00.y}, color, {-uPad,         -vPad        }, shape, gi(p00) },
                { {p11.x, p11.y}, color, {1.0f + uPad,   1.0f + vPad  }, shape, gi(p11) },
                { {p01.x, p01.y}, color, {-uPad,         1.0f + vPad  }, shape, gi(p01) },
            };
            state.draw(cmd, verts, 6);
            break;
        }

        default: break;
    }
}

} // namespace jvk::pipelines
