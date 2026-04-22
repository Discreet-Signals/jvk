#pragma once

namespace jvk {

class Shader; // forward declaration

class Graphics : public juce::LowLevelGraphicsContext {
public:
    Graphics(Renderer& renderer, float displayScale)
        : renderer_(renderer), displayScale_(displayScale)
    {
        stateStack_.push_back({});
        auto& s = stateStack_.back();
        s.clipBounds = { 0, 0,
            static_cast<int>(renderer_.target().width()),
            static_cast<int>(renderer_.target().height()) };
    }

    ~Graphics() override = default;

    void setZOrder(float z) { state().zOrder = z; }

    // Extract a jvk::Graphics* from a juce::Graphics (for effects API)
    static Graphics* create(juce::Graphics& g)
    {
        auto* ctx = dynamic_cast<Graphics*>(&g.getInternalContext());
        return ctx;
    }

    // ===== LowLevelGraphicsContext overrides =====
    bool isVectorDevice() const override { return false; }

    void setOrigin(juce::Point<int> o) override
    {
        auto& ct = state().transform;
        ct = juce::AffineTransform::translation(static_cast<float>(o.x),
                                                 static_cast<float>(o.y))
                 .followedBy(ct);
    }

    void addTransform(const juce::AffineTransform& t) override
    {
        state().transform = t.followedBy(state().transform);
    }

    float getPhysicalPixelScaleFactor() const override { return displayScale_; }

    bool clipToRectangle(const juce::Rectangle<int>& r) override
    {
        auto& s = state();
        auto transformed = r.transformedBy(s.transform);
        // Scale logical→pixel for Vulkan scissor
        auto pixel = juce::Rectangle<int>(
            static_cast<int>(transformed.getX() * displayScale_),
            static_cast<int>(transformed.getY() * displayScale_),
            static_cast<int>(transformed.getWidth() * displayScale_),
            static_cast<int>(transformed.getHeight() * displayScale_));
        s.clipBounds = s.clipBounds.getIntersection(pixel);
        // Open a new scheduler scope whose bounds match the new scissor. All
        // commands pushed until the matching popScope() inherit this scopeId.
        renderer_.pushScope(s.clipBounds);
        renderer_.push(DrawOp::PushClipRect, s.zOrder, s.clipBounds,
                               s.stencilDepth, s.scopeDepth,
                               PushClipRectParams { transformed },
                               /*writesPx*/ s.clipBounds,
                               /*readsPx*/  s.clipBounds,
                               /*stateKey*/ state_key::make(DrawOp::PushClipRect),
                               /*isOpaque*/ false);
        s.scopeDepth++;
        return !s.clipBounds.isEmpty();
    }

    bool clipToRectangleList(const juce::RectangleList<int>& list) override
    {
        return clipToRectangle(list.getBounds());
    }

    void excludeClipRectangle(const juce::Rectangle<int>&) override {}

    // Axis-aligned rectangle detection — SVG clip-path paths are almost
    // always a <rect>, which shows up here as a 4-or-5 vertex path of
    // horizontal+vertical line segments. Returning true lets clipToPath
    // promote the clip to a simple scissor via clipToRectangle, which
    // leaves the stencil free for the inner fillPath's winding counter
    // (non-zero winding can't tolerate an outer stencil clip summing into
    // its counter — every pixel inside the outer rect would then test as
    // non-zero regardless of the inner path).
    static bool isAxisAlignedRect(const juce::Path& path,
                                  juce::Rectangle<float>& out)
    {
        juce::Path::Iterator iter(path);
        juce::Point<float> verts[6] {};
        int count = 0;
        while (iter.next())
        {
            if (count >= 6) return false;
            switch (iter.elementType)
            {
                case juce::Path::Iterator::startNewSubPath:
                case juce::Path::Iterator::lineTo:
                    verts[count++] = { iter.x1, iter.y1 };
                    break;
                case juce::Path::Iterator::closePath:
                    break;
                default:
                    return false; // any curve segment → not a simple rect
            }
        }
        if (count < 4) return false;
        // Rectangle: 4 corners, optional explicit close-vertex equal to start.
        int n = (count == 5 && verts[4] == verts[0]) ? 4 : count;
        if (n != 4) return false;
        for (int i = 0; i < 4; ++i)
        {
            auto a = verts[i];
            auto b = verts[(i + 1) % 4];
            if (a.x != b.x && a.y != b.y) return false; // not axis-aligned
        }
        float xMin = std::min({verts[0].x, verts[1].x, verts[2].x, verts[3].x});
        float xMax = std::max({verts[0].x, verts[1].x, verts[2].x, verts[3].x});
        float yMin = std::min({verts[0].y, verts[1].y, verts[2].y, verts[3].y});
        float yMax = std::max({verts[0].y, verts[1].y, verts[2].y, verts[3].y});
        if (xMax <= xMin || yMax <= yMin) return false;
        out = { xMin, yMin, xMax - xMin, yMax - yMin };
        return true;
    }

    void clipToPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        // Fast path for rectangular clip paths (common: SVG <clipPath><rect/>
        // wrappers on Drawables). Preserved transforms that stay axis-aligned
        // keep the rectangle axis-aligned, so we can route through the
        // scissor-based clipToRectangle and avoid the stencil altogether.
        auto& s0 = state();
        auto combinedForRectCheck = t.followedBy(s0.transform);
        juce::Rectangle<float> rectLocal;
        if (combinedForRectCheck.mat01 == 0.0f && combinedForRectCheck.mat10 == 0.0f
            && isAxisAlignedRect(path, rectLocal))
        {
            auto transformed = rectLocal.transformedBy(t);
            clipToRectangle(transformed.getSmallestIntegerContainer());
            return;
        }

        // Non-rectangular path — use the geometry-abstracted clip-stencil
        // pipeline. Flatten to segments, upload into the shared path-segment
        // SSBO, emit a GeometryPrimitive (tag=Path), upload to ClipPipeline's
        // primitive SSBO, and push DrawOp::PushClipPath. Fragments inside
        // the path's winding test INCR the stencil; matching pop DECRs with
        // the SAME primitive range.
        auto* pp = renderer_.pathPipeline();
        auto* cp = renderer_.clipPipeline();
        if (!pp || !cp) return;

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        const uint32_t segStart = pp->uploadSegments(
            scratchSegments_.data(),
            static_cast<uint32_t>(scratchSegments_.size()));
        const uint32_t segCount = static_cast<uint32_t>(scratchSegments_.size());
        const uint32_t fillRule = path.isUsingNonZeroWinding() ? 0u : 1u;

        auto px = pathBounds.transformedBy(combined).expanded(1.0f);
        int bx  = static_cast<int>(std::floor(px.getX()));
        int by  = static_cast<int>(std::floor(px.getY()));
        int bx2 = static_cast<int>(std::ceil(px.getRight()));
        int by2 = static_cast<int>(std::ceil(px.getBottom()));
        juce::Rectangle<int>  pathBoundsPx { bx, by, bx2 - bx, by2 - by };
        juce::Rectangle<float> coverRect { static_cast<float>(bx),
                                           static_cast<float>(by),
                                           static_cast<float>(bx2 - bx),
                                           static_cast<float>(by2 - by) };

        auto outerBounds = s.clipBounds.getIntersection(pathBoundsPx);

        // Build the clip primitive. Segments are in physical pixels (baked
        // through `combined`), so invXform is identity — shader walks
        // segments in the same coord space as fragCoord.
        GeometryPrimitive prim {};
        fillBbox(prim, pathBoundsPx);
        prim.invXform01[0] = 1.0f; prim.invXform01[1] = 0.0f;
        prim.invXform01[2] = 0.0f; prim.invXform01[3] = 1.0f;
        prim.flags[0] = static_cast<uint32_t>(GeometryTag::Path);
        prim.flags[1] = packShapeFlags(false, BlurEdge::Centered, fillRule,
                                       /*stroke*/ false, ColorSource::Solid);
        prim.flags[2] = segStart;
        prim.flags[3] = segCount;

        const uint32_t arenaOffset = renderer_.arena_push(prim);
        // Push + matching Pop share this ref via pathClipStack. The upload
        // pass uploads the primitive ONCE for the push command and ONCE
        // for the pop command (since both carry the same arenaOffset in
        // their ref copies); each gets its own firstInstance pointing at
        // the same primitive data in the SSBO. INCR_WRAP and DECR_WRAP
        // therefore touch identical fragments.
        ClipDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u,
                              arenaOffset, /*_pad*/ 0u, coverRect };

        // Remember for the matching pop — same primitive range so DECR
        // cancels INCR exactly.
        s.pathClipStack.push_back(ref);

        renderer_.pushScope(outerBounds);
        renderer_.push(DrawOp::PushClipPath, s.zOrder, outerBounds,
                               s.stencilDepth, s.scopeDepth, ref,
                               /*writesPx*/ outerBounds,
                               /*readsPx*/  outerBounds,
                               /*stateKey*/ state_key::make(DrawOp::PushClipPath),
                               /*isOpaque*/ false);

        s.clipBounds = outerBounds;
        s.stencilDepth++;
        s.scopeDepth++;
    }

    void clipToImageAlpha(const juce::Image&, const juce::AffineTransform&) override {}

    bool clipRegionIntersects(const juce::Rectangle<int>& r) override
    {
        // r is in logical coords (current context space), clipBounds is physical pixels.
        // Transform r to physical to compare in the same space.
        auto& s = state();
        auto physRect = r.toFloat().transformedBy(s.transform.scaled(displayScale_)).getSmallestIntegerContainer();
        return s.clipBounds.intersects(physRect);
    }

    juce::Rectangle<int> getClipBounds() const override
    {
        // JUCE expects the clip bounds in the CURRENT coord space — that's
        // what Component::paintChildren compares each child's local bounds
        // against when deciding whether to paint. `s.clipBounds` is in root
        // pixels, so invert the local→pixel transform to get back to the
        // caller's local coord space.
        auto& s = state();
        if (s.clipBounds.isEmpty()) return {};
        auto pixelToLocal = s.transform.scaled(displayScale_).inverted();
        auto local = s.clipBounds.toFloat().transformedBy(pixelToLocal);
        return local.getSmallestIntegerContainer();
    }
    bool isClipEmpty() const override { return state().clipBounds.isEmpty(); }

    void saveState() override { stateStack_.push_back(stateStack_.back()); }

    void restoreState() override
    {
        if (stateStack_.size() <= 1) return;
        auto& old = stateStack_.back();
        auto& prev = stateStack_[stateStack_.size() - 2];
        while (old.scopeDepth > prev.scopeDepth) {
            bool isPathClip = old.stencilDepth > prev.stencilDepth;
            if (isPathClip) {
                recordPopClip();
            } else {
                // Rect clip — no GPU work, just a CPU-side scissor pop at
                // replay. Payload is empty; op type conveys the kind.
                struct Empty {};
                renderer_.push(DrawOp::PopClipRect, old.zOrder, old.clipBounds,
                                       old.stencilDepth, old.scopeDepth, Empty {},
                                       /*writesPx*/ old.clipBounds,
                                       /*readsPx*/  old.clipBounds,
                                       /*stateKey*/ state_key::make(DrawOp::PopClipRect),
                                       /*isOpaque*/ false);
                // Close the scheduler scope opened by the matching clipToRectangle.
                renderer_.popScope();
                old.scopeDepth--;
            }
        }
        stateStack_.pop_back();
    }

    void beginTransparencyLayer(float opacity) override { state().opacity *= opacity; }
    void endTransparencyLayer() override {}

    void setFill(const juce::FillType& fill) override { state().fill = fill; }
    void setOpacity(float opacity) override { state().opacity = opacity; }
    void setInterpolationQuality(juce::Graphics::ResamplingQuality) override {}

    void fillRect(const juce::Rectangle<int>& r, bool) override { fillRect(r.toFloat()); }

    void fillRect(const juce::Rectangle<float>& r) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);
        auto writes = toWritesPx(r);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        auto anchor    = r.getCentre();
        auto shapeHalf = juce::Point<float>(r.getWidth() * 0.5f, r.getHeight() * 0.5f);
        fillInverseTransform(p, anchor, shapeHalf);

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        p.flags[0] = static_cast<uint32_t>(GeometryTag::Rect);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, /*stroke*/ false, cs);

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::FillRect, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::FillRect, 0, static_cast<uint16_t>(fi)),
            /*isOpaque*/ currentFillIsOpaque());
    }

    void fillRectList(const juce::RectangleList<float>& list) override
    {
        for (auto& r : list) fillRect(r);
    }

    // Analytical-SDF path fill (vger / Slug-style). CPU flattens the path's
    // curves to line segments in physical-pixel space, uploads them to
    // PathPipeline's per-frame storage buffer, and emits a single draw
    // covering the path's bounds. The fragment shader evaluates the SDF
    // analytically per pixel — no atlas, no rasterisation preprocessing.
    //
    // Colour is sourced the same way as every other 2D op: per-vertex
    // `color` + `gradientInfo` populated via the shared GradientCtx /
    // gradientAt() / fillColorAttr helpers from ColorDraw.h. Solid fills
    // land in `color`; gradient fills set mode=1/2 in `gradientInfo.z` and
    // the fragment shader samples the gradient atlas row.
    void fillPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || path.isEmpty()) return;
        auto* pp = renderer_.pathPipeline();
        auto* fp = renderer_.fillPipeline();
        if (!pp || !fp) return;

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        const uint32_t segStart = pp->uploadSegments(
            scratchSegments_.data(),
            static_cast<uint32_t>(scratchSegments_.size()));
        const uint32_t segCount = static_cast<uint32_t>(scratchSegments_.size());
        const uint32_t fillRule = path.isUsingNonZeroWinding() ? 0u : 1u;
        const uint32_t fi       = renderer_.captureFill(s.fill);

        // Path bounds in physical pixels + 1px AA margin for the edge kernel.
        auto pxBounds = pathBounds.transformedBy(combined).expanded(1.0f);
        int bx  = static_cast<int>(std::floor(pxBounds.getX()));
        int by  = static_cast<int>(std::floor(pxBounds.getY()));
        int bx2 = static_cast<int>(std::ceil(pxBounds.getRight()));
        int by2 = static_cast<int>(std::ceil(pxBounds.getBottom()));
        juce::Rectangle<int> quadPxRect { bx, by, bx2 - bx, by2 - by };
        auto clipRect = s.clipBounds.getIntersection(quadPxRect);
        if (clipRect.isEmpty()) return;

        // Path SDF lives in PHYSICAL pixels (segments already baked through
        // `combined`), so invXform is identity — the shader's fragCoord and
        // segment coords share the same space. Gradient coefs live in extraB
        // as usual.
        GeometryPrimitive p {};
        fillBbox(p, clipRect);
        p.invXform01[0] = 1.0f; p.invXform01[1] = 0.0f;
        p.invXform01[2] = 0.0f; p.invXform01[3] = 1.0f;
        auto cs = fillColorFields(s.fill, combined, s.opacity, p);
        p.flags[0] = static_cast<uint32_t>(GeometryTag::Path);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, fillRule, /*stroke*/ false, cs);
        p.flags[2] = segStart;
        p.flags[3] = segCount;

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::FillPath, s.zOrder, clipRect,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ clipRect,
            /*readsPx*/  clipRect,
            /*stateKey*/ state_key::make(DrawOp::FillPath,
                                         /*pipelineHint*/ fillRule,
                                         static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);   // SDF fills have AA edges → never opaque
    }

    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !img.isValid()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s = state();
        uint64_t hash = ResourceCaches::hashImage(img);
        renderer_.caches().getTexture(hash, img);

        juce::Rectangle<float> imgRectLogical {
            0.0f, 0.0f,
            static_cast<float>(img.getWidth()),
            static_cast<float>(img.getHeight())
        };
        auto imgRectTransformed = imgRectLogical.transformedBy(t);
        auto writes = toWritesPx(imgRectTransformed);

        // Image quad lives in its own local frame (w × h logical). The vertex
        // shader expands bbox → physical corners; the fragment shader samples
        // the bound image via atlasUV = mix((0,0),(1,1), vQuadUV). So we just
        // stash the UV rect (0..1) in extraB. invXform is identity-ish since
        // tag=6 doesn't evaluate an SDF.
        GeometryPrimitive p {};
        fillBbox(p, writes);
        p.invXform01[0] = 1.0f; p.invXform01[1] = 0.0f;
        p.invXform01[2] = 0.0f; p.invXform01[3] = 1.0f;
        p.flags[0] = static_cast<uint32_t>(GeometryTag::Image);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, false, ColorSource::Solid);
        p.extraB[0] = 0.0f; p.extraB[1] = 0.0f;
        p.extraB[2] = 1.0f; p.extraB[3] = 1.0f;          // atlas UV = full image
        p.color[0]  = 1.0f; p.color[1]  = 1.0f;
        p.color[2]  = 1.0f; p.color[3]  = s.opacity;     // sampler supplies RGB

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillImageDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u,
                                    arenaOffset, /*_pad*/ 0u, hash };

        renderer_.push(DrawOp::DrawImage, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::DrawImage, 0, state_key::truncHash(hash)),
            /*isOpaque*/ false);   // can't cheaply know the image is opaque
    }

    void drawLine(const juce::Line<float>& line) override
    {
        drawLineWithThickness(line, 1.0f);
    }

    void drawLineWithThickness(const juce::Line<float>& line, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);

        // Capsule SDF: A at shape-local origin, B at a 2D offset, radius =
        // half-thickness. All in logical units — shader's invXform maps
        // physical → shape-local.
        auto a = line.getStart();
        auto b = line.getEnd();
        auto bRel = juce::Point<float>(b.x - a.x, b.y - a.y);
        float radius = lineThickness * 0.5f;

        juce::Rectangle<float> lineBox {
            juce::jmin(a.x, b.x) - radius,
            juce::jmin(a.y, b.y) - radius,
            std::abs(b.x - a.x) + radius * 2.0f,
            std::abs(b.y - a.y) + radius * 2.0f
        };
        auto writes = toWritesPx(lineBox, /*expand physical*/ 1.0f);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        fillInverseTransform(p, a, juce::Point<float>(0, 0));

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        p.flags[0] = static_cast<uint32_t>(GeometryTag::Capsule);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, false, cs);
        p.extraA[0] = 0.0f;
        p.extraA[1] = radius;
        p.extraA[2] = bRel.x;
        p.extraA[3] = bRel.y;

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::DrawLine, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::DrawLine, 0, static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
    }

    void setFont(const juce::Font& f) override { state().font = f; }
    const juce::Font& getFont() override { return state().font; }

    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                    juce::Span<const juce::Point<float>> positions,
                    const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || glyphs.empty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;

        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);

        // Resolve glyph atlas entries HERE (paint thread) so each glyph's
        // MSDF rasterisation + page index is known at record time. Worker's
        // FillPipeline::prepareFrame() then stages the dirty atlas pages to
        // GPU before dispatch samples them.
        auto& atlas = fp->atlas();
        auto  tx    = t.followedBy(s.transform).scaled(displayScale_);
        float physFontH = s.font.getHeight() * tx.getScaleFactor();
        float hScale    = s.font.getHorizontalScale();
        constexpr int PAD = GlyphAtlas::GLYPH_PADDING;

        auto toPhys = s.transform.scaled(displayScale_);
        // Build one GeometryPrimitive per glyph; collect atlas page indices
        // for the dispatcher's per-glyph shape-sampler rebind.
        scratchGlyphPrims_.clear();
        scratchGlyphPages_.clear();
        scratchGlyphPrims_.reserve(glyphs.size());
        scratchGlyphPages_.reserve(glyphs.size());

        juce::Rectangle<int> unionWrites {};

        auto* typeface = s.font.getTypefacePtr().get();
        for (size_t i = 0; i < glyphs.size(); i++) {
            GlyphAtlas::GlyphKey key { typeface, glyphs[i] };
            auto* entry = atlas.getGlyph(key, s.font);
            if (!entry) continue;
            auto glyphPos = positions[i].transformedBy(tx);

            float innerW = entry->boundsW * physFontH * hScale;
            float innerH = entry->boundsH * physFontH;
            int   innerTexW = entry->w - PAD * 2;
            int   innerTexH = entry->h - PAD * 2;
            float gw = static_cast<float>(entry->w) * innerW / static_cast<float>(innerTexW);
            float gh = static_cast<float>(entry->h) * innerH / static_cast<float>(innerTexH);
            float padScreenX = PAD * (innerW / static_cast<float>(innerTexW));
            float padScreenY = PAD * (innerH / static_cast<float>(innerTexH));
            float gx = glyphPos.x + entry->boundsX * physFontH * hScale - padScreenX;
            float gy = glyphPos.y + entry->boundsY * physFontH - padScreenY;
            float screenPxRange = std::max(1.0f,
                static_cast<float>(GlyphAtlas::MSDF_PIXEL_RANGE) * (gw / static_cast<float>(entry->w)));

            juce::Rectangle<int> glyphPx {
                static_cast<int>(std::floor(gx)),
                static_cast<int>(std::floor(gy)),
                static_cast<int>(std::ceil(gx + gw) - std::floor(gx)),
                static_cast<int>(std::ceil(gy + gh) - std::floor(gy))
            };
            glyphPx = s.clipBounds.getIntersection(glyphPx);
            if (glyphPx.isEmpty()) continue;

            GeometryPrimitive p {};
            fillBbox(p, glyphPx);
            // Glyph quad has no shape-local SDF — tag=4 reads the MSDF atlas
            // via vQuadUV, not invXform. Set identity for determinism.
            p.invXform01[0] = 1.0f; p.invXform01[1] = 0.0f;
            p.invXform01[2] = 0.0f; p.invXform01[3] = 1.0f;

            // Gradient coefficients for MSDF live in extraA (extraB carries
            // atlas UV). Write colour fields into p via the solid branch,
            // then if it was a gradient overwrite extraA with gradient coefs
            // and move the result out of extraB.
            auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);
            if (cs != ColorSource::Solid) {
                // Gradient coefs went into extraB; relocate to extraA for MSDF
                // tag (where extraB is atlas UV).
                for (int j = 0; j < 4; j++) {
                    p.extraA[j] = p.extraB[j];
                    p.extraB[j] = 0.0f;
                }
            }
            p.extraB[0] = entry->u0; p.extraB[1] = entry->v0;
            p.extraB[2] = entry->u1; p.extraB[3] = entry->v1;
            // Preserve invLen2 + rowNorm in payload.xy; stash screenPxRange
            // in payload.z (MSDF-only slot).
            p.payload[2] = screenPxRange;

            p.flags[0] = static_cast<uint32_t>(GeometryTag::MSDFGlyph);
            p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, false, cs);

            scratchGlyphPrims_.push_back(p);
            scratchGlyphPages_.push_back(entry->atlasIndex);
            unionWrites = unionWrites.getUnion(glyphPx);
        }

        if (scratchGlyphPrims_.empty()) return;

        // Stash the primitive array in the arena. The upload pass after
        // the scheduler runs will consume this range, concatenate it with
        // any same-stateKey neighbour under Fuse mode, and upload the
        // result into FillPipeline's SSBO in scheduled order.
        const uint32_t arenaOffset = renderer_.arena_pushSpanReturn(
            std::span<const GeometryPrimitive>(scratchGlyphPrims_.data(),
                                               scratchGlyphPrims_.size()));

        FillGlyphsDispatchRef ref {
            /*firstInstance*/ 0u,
            /*primCount*/ static_cast<uint32_t>(scratchGlyphPrims_.size()),
            arenaOffset,
            fi
        };

        renderer_.push(DrawOp::DrawGlyphs, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ unionWrites,
            /*readsPx*/  unionWrites,
            /*stateKey*/ state_key::make(DrawOp::DrawGlyphs,
                                         /*pipelineHint*/ renderer_.captureFont(s.font),
                                         static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
        // Per-glyph atlas page indices follow the ref in arena so the
        // dispatcher can iterate glyphs and rebind the shape sampler when
        // the page changes.
        renderer_.arena_pushSpan(std::span<const uint32_t>(
            scratchGlyphPages_.data(), scratchGlyphPages_.size()));
    }

    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);
        // Expand by 1px AA margin — the SDF smoothstep bleeds one pixel
        // outside the geometric rect at each edge.
        auto writes = toWritesPx(r, /*expand physical*/ 1.0f);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        auto anchor    = r.getCentre();
        auto shapeHalf = juce::Point<float>(r.getWidth() * 0.5f, r.getHeight() * 0.5f);
        fillInverseTransform(p, anchor, shapeHalf);

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        p.flags[0] = static_cast<uint32_t>(GeometryTag::RoundRect);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, /*stroke*/ false, cs);
        p.extraA[0] = cornerSize;   // logical px — SDF lives in logical space
        p.extraA[1] = 0.0f;
        p.extraA[2] = 0.0f;
        p.extraA[3] = 0.0f;

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::FillRoundedRect, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::FillRoundedRect, 0,
                                         static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);
        auto writes = toWritesPx(area, /*expand physical*/ 1.0f);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        auto anchor    = area.getCentre();
        auto shapeHalf = juce::Point<float>(area.getWidth() * 0.5f, area.getHeight() * 0.5f);
        fillInverseTransform(p, anchor, shapeHalf);

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        p.flags[0] = static_cast<uint32_t>(GeometryTag::Ellipse);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, false, cs);

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::FillEllipse, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::FillEllipse, 0, static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
    }

    void drawRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                              float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);
        auto writes = toWritesPx(rect,
            /*expand physical*/ lineThickness * 0.5f * displayScale_ + 1.0f);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        auto anchor    = rect.getCentre();
        auto shapeHalf = juce::Point<float>(rect.getWidth() * 0.5f, rect.getHeight() * 0.5f);
        fillInverseTransform(p, anchor, shapeHalf);

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        // Stroke SDF: tag=RRect with stroke bit; extraA.y = lineThickness (logical).
        // cornerSize is also logical; shader applies abs(filledSDF) - thick/2.
        p.flags[0] = static_cast<uint32_t>(GeometryTag::RoundRect);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, /*stroke*/ true, cs);
        p.extraA[0] = cornerSize;
        p.extraA[1] = lineThickness;
        p.extraA[2] = 0.0f;
        p.extraA[3] = 0.0f;

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::StrokeRoundedRect, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::StrokeRoundedRect, 0,
                                         static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
    }

    // Outline rectangle — stroked rounded rect with cornerSize = 0.
    void drawRect(const juce::Rectangle<float>& rect, float lineThickness) override
    {
        drawRoundedRectangle(rect, 0.0f, lineThickness);
    }

    void drawEllipse(const juce::Rectangle<float>& area, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto* fp = renderer_.fillPipeline();
        if (!fp) return;
        auto& s  = state();
        auto  fi = renderer_.captureFill(s.fill);
        auto writes = toWritesPx(area,
            /*expand physical*/ lineThickness * 0.5f * displayScale_ + 1.0f);

        GeometryPrimitive p {};
        fillBbox(p, writes);
        auto anchor    = area.getCentre();
        auto shapeHalf = juce::Point<float>(area.getWidth() * 0.5f, area.getHeight() * 0.5f);
        fillInverseTransform(p, anchor, shapeHalf);

        auto toPhys = s.transform.scaled(displayScale_);
        auto cs = fillColorFields(s.fill, toPhys, s.opacity, p);

        p.flags[0] = static_cast<uint32_t>(GeometryTag::Ellipse);
        p.flags[1] = packShapeFlags(false, BlurEdge::Centered, 0u, /*stroke*/ true, cs);
        p.extraA[0] = 0.0f;
        p.extraA[1] = lineThickness;

        const uint32_t arenaOffset = renderer_.arena_push(p);
        FillDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u, arenaOffset, fi };

        renderer_.push(DrawOp::StrokeEllipse, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::StrokeEllipse, 0, static_cast<uint16_t>(fi)),
            /*isOpaque*/ false);
    }

    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const override
    {
        return std::make_unique<juce::SoftwareImageType>();
    }

    uint64_t getFrameId() const override { return frameId_++; }

    // ===== GPU Effects =====
    void darken(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        float v = 1.0f - amount;
        // EffectBlend is pointwise; writes the region only. region arrives in
        // pre-transformed coords already (clipBounds is physical), so we don't
        // re-apply s.transform — just clip.
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            EffectBlendParams { v, v, v, region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::EffectBlend),
            /*isOpaque*/ false);
    }

    void brighten(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        float v = 1.0f + amount;
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            EffectBlendParams { v, v, v, region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::EffectBlend),
            /*isOpaque*/ false);
    }

    void tint(juce::Colour c, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            EffectBlendParams { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                                region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::EffectBlend),
            /*isOpaque*/ false);
    }

    void warmth(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            EffectBlendParams { 1.0f + amount * 0.2f, 1.0f, 1.0f - amount * 0.1f,
                                region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,
            /*stateKey*/ state_key::make(DrawOp::EffectBlend),
            /*isOpaque*/ false);
    }

    void blur(float radius, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        // Legacy full-region Gaussian. Writes region; reads region dilated by
        // radius (kernel reach). radius is in the same space as region —
        // typically physical already since region comes from clipBounds.
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        int reach = static_cast<int>(std::ceil(radius * displayScale_)) + 1;
        auto reads = writes.expanded(reach, reach).getIntersection(s.clipBounds);
        renderer_.push(DrawOp::EffectKernel, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            BlurParams { radius, region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  reads,
            /*stateKey*/ state_key::make(DrawOp::EffectKernel),
            /*isOpaque*/ false);
    }

    // =========================================================================
    // Perceptual-colour transforms — one primitive, a handful of specialised
    // helpers. The fragment shader (hsv.frag) runs in Oklch (polar Oklab)
    // rather than classical HSV so desaturation/brightness ops respect human
    // luminance perception; the H/S/V parameter names are kept for source
    // compatibility and map to Oklch's (h, C, L).
    //
    // `hsv` is the universal form: `hsv *= (scaleH, scaleS, scaleV);
    //                                hsv += (deltaH, deltaS, deltaV);`
    //
    // Defaults are identity (scales 1.0, deltas 0.0). The specialised wrappers
    // below set only the relevant fields — they exist purely for readability
    // at call sites, they all compile down to the same DrawOp::EffectHSV.
    // =========================================================================

    void hsv(float scaleH, float scaleS, float scaleV,
             float deltaH, float deltaS, float deltaV,
             juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        auto writes = s.clipBounds.getIntersection(region.getSmallestIntegerContainer());
        renderer_.push(DrawOp::EffectHSV, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            HSVParams { scaleH, scaleS, scaleV, deltaH, deltaS, deltaV,
                        region, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,   // pointwise — reads only its own pixel
            /*stateKey*/ state_key::make(DrawOp::EffectHSV),
            /*isOpaque*/ false);
    }

    // amount: 0 = grayscale, 1 = original, >1 = boosted saturation.
    void saturate(float amount, juce::Rectangle<float> region = {})
    {
        hsv(1.0f, amount, 1.0f, 0.0f, 0.0f, 0.0f, region);
    }

    // turns: hue rotation in 0..1 (1 = full 360°).
    void shiftHue(float turns, juce::Rectangle<float> region = {})
    {
        hsv(1.0f, 1.0f, 1.0f, turns, 0.0f, 0.0f, region);
    }

    // =========================================================================
    // Shape-aware variable-radius blur — JUCE-style draw/fill split.
    //
    //   draw*Blurred…  — stroke variants; `lineThickness` controls the ring
    //                    width. The blur band is centred on the shape's edge
    //                    (the stroke's zero-crossing) rather than the interior.
    //   fill*Blurred…  — fill variants; blur ramps from `blurRadius` inside
    //                    the shape down to 0 across `falloffRadius`. An
    //                    optional trailing `inverted` flips the fill to the
    //                    complement ("blur everything OUTSIDE the shape").
    //
    // `edge` controls where the falloff band sits relative to the shape's
    // boundary (the ring boundary for strokes, SDF=0 for fills).
    // All distances are logical pixels; displayScale is folded in per-pixel.
    // =========================================================================

    // Stroke variants --------------------------------------------------------

    void drawBlurredLine(const juce::Line<float>& line, float lineThickness,
                         float blurRadius, float falloffRadius,
                         BlurEdge edge = BlurEdge::Centered,
                         BlurMode mode = BlurMode::Low)
    {
        // Shape-local anchor at endpoint A; B is stored relative to A.
        auto a = line.getStart();
        auto b = line.getEnd();
        juce::Rectangle<float> anchor { a.x, a.y, 0.0f, 0.0f };
        juce::Point<float> bRel { b.x - a.x, b.y - a.y };
        // Capsule's `lineThickness` shader field is cross-section radius.
        pushBlurShape(anchor, 0.0f, /*shapeType*/ 3,
                      blurRadius, falloffRadius, /*inverted*/ false, edge,
                      { bRel.x, bRel.y }, lineThickness * 0.5f, mode);
    }

    void drawBlurredRectangle(const juce::Rectangle<float>& rect, float lineThickness,
                              float blurRadius, float falloffRadius,
                              BlurEdge edge = BlurEdge::Centered,
                              BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(rect, 0.0f, /*shapeType*/ 0,
                      blurRadius, falloffRadius, /*inverted*/ false, edge,
                      {0, 0}, strokeFloor(lineThickness), mode);
    }

    void drawBlurredRoundedRectangle(const juce::Rectangle<float>& rect,
                                     float cornerSize, float lineThickness,
                                     float blurRadius, float falloffRadius,
                                     BlurEdge edge = BlurEdge::Centered,
                                     BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(rect, cornerSize, /*shapeType*/ 1,
                      blurRadius, falloffRadius, /*inverted*/ false, edge,
                      {0, 0}, strokeFloor(lineThickness), mode);
    }

    void drawBlurredEllipse(const juce::Rectangle<float>& area, float lineThickness,
                            float blurRadius, float falloffRadius,
                            BlurEdge edge = BlurEdge::Centered,
                            BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(area, 0.0f, /*shapeType*/ 2,
                      blurRadius, falloffRadius, /*inverted*/ false, edge,
                      {0, 0}, strokeFloor(lineThickness), mode);
    }

    // Fill variants ----------------------------------------------------------

    void fillBlurredRectangle(const juce::Rectangle<float>& rect,
                              float blurRadius, float falloffRadius,
                              bool inverted = false,
                              BlurEdge edge = BlurEdge::Centered,
                              BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(rect, 0.0f, /*shapeType*/ 0,
                      blurRadius, falloffRadius, inverted, edge,
                      {0, 0}, 0.0f, mode);
    }

    void fillBlurredRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                                     float blurRadius, float falloffRadius,
                                     bool inverted = false,
                                     BlurEdge edge = BlurEdge::Centered,
                                     BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(rect, cornerSize, /*shapeType*/ 1,
                      blurRadius, falloffRadius, inverted, edge,
                      {0, 0}, 0.0f, mode);
    }

    void fillBlurredEllipse(const juce::Rectangle<float>& area,
                            float blurRadius, float falloffRadius,
                            bool inverted = false,
                            BlurEdge edge = BlurEdge::Centered,
                            BlurMode mode = BlurMode::Low)
    {
        pushBlurShape(area, 0.0f, /*shapeType*/ 2,
                      blurRadius, falloffRadius, inverted, edge,
                      {0, 0}, 0.0f, mode);
    }

    // Path variants — route through BlurPipeline (ping-pong effect pass
    // on the scene target), which walks the same per-frame segment SSBO
    // PathPipeline owns for fillPath. The path is flattened to physical-px
    // line segments here and uploaded for the GPU's per-fragment SDF loop.

    void drawBlurredPath(const juce::Path& path, float lineThickness,
                         float blurRadius, float falloffRadius,
                         BlurEdge edge = BlurEdge::Centered,
                         BlurMode mode = BlurMode::Low,
                         const juce::AffineTransform& t = {})
    {
        pushBlurPath(path, t,
                     blurRadius, falloffRadius,
                     /*inverted*/ false, edge,
                     /*strokeHalfWidth*/ strokeFloor(lineThickness) * 0.5f,
                     mode);
    }

    void fillBlurredPath(const juce::Path& path,
                         float blurRadius, float falloffRadius,
                         bool inverted = false,
                         BlurEdge edge = BlurEdge::Centered,
                         BlurMode mode = BlurMode::Low,
                         const juce::AffineTransform& t = {})
    {
        pushBlurPath(path, t,
                     blurRadius, falloffRadius,
                     inverted, edge,
                     /*strokeHalfWidth*/ 0.0f,
                     mode);
    }


    void drawShader(Shader& shader, juce::Rectangle<float> region = {})
    {
        if (isClipEmpty()) return;
        // Pin the Shader for this frame. Renderer holds the pin until the
        // GPU is done with the command buffer that will reference the
        // shader's VkPipeline / VkDescriptorSet — at which point Shader's
        // dtor (waiting on FrameRetained::waitUntilUnretained) is free to
        // proceed. Lets the user reset the owning component from anywhere
        // on the message thread without coordinating with the worker.
        renderer_.retain(&shader);
        auto& s = state();
        // Fold the full paint transform (setOrigin + addTransform stack) and
        // displayScale into the region at record time, matching the convention
        // used by fillRect / darken / tint — the pipeline receives a
        // physical-pixel rect and the shader_region.vert math lines up with
        // the physical-pixel viewport. For rotated/skewed transforms this uses
        // the AABB of the transformed rect (toPixels semantics); axis-aligned
        // transforms are exact.
        juce::Rectangle<float> regionPx;
        if (region.isEmpty()) {
            regionPx = s.clipBounds.toFloat();
        } else {
            auto transformed = region.transformedBy(s.transform);
            regionPx = { transformed.getX()      * displayScale_,
                         transformed.getY()      * displayScale_,
                         transformed.getWidth()  * displayScale_,
                         transformed.getHeight() * displayScale_ };
        }
        auto regionBox = regionPx.getSmallestIntegerContainer();
        auto writes = s.clipBounds.getIntersection(regionBox);
        // Shader identity is the pipelineHint so two ops using the same Shader
        // can fuse once DrawShader batching lands. Collision across different
        // Shader* is fine: DAG independence still enforces correctness.
        uint32_t shaderHint = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&shader) & 0xFFFFFFULL);
        renderer_.push(DrawOp::DrawShader, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth,
            DrawShaderParams { &shader, regionPx, displayScale_ },
            /*writesPx*/ writes,
            /*readsPx*/  writes,   // shaders don't sample the scene buffer
            /*stateKey*/ state_key::make(DrawOp::DrawShader, shaderHint),
            /*isOpaque*/ false);
    }

    Renderer& getRenderer() { return renderer_; }

private:
    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        float                 zOrder = 0.0f;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
        std::vector<ClipDispatchRef> pathClipStack;
    };

    RecordState& state() { return stateStack_.back(); }
    const RecordState& state() const { return stateStack_.back(); }

    // Record a PopClip — same ClipDispatchRef as the paired PushClipPath so
    // the DECR_WRAP at the GPU matches the INCR_WRAP exactly. Stencil
    // reference at replay = stencilDepth BEFORE the pop (current depth),
    // so the hardware decrements only the pixels the push incremented.
    void recordPopClip()
    {
        auto& s = state();
        if (!s.pathClipStack.empty()) {
            ClipDispatchRef ref = s.pathClipStack.back();
            // writesPx/readsPx must be the pop's OWN cover rect, not the
            // parent scope. Using s.clipBounds here would make every sibling
            // pop share the enclosing scope's rect, spuriously WAW-edging
            // them together and defeating disjoint-sibling clustering.
            auto coverPx = ref.coverRect.getSmallestIntegerContainer();
            auto popRegion = s.clipBounds.getIntersection(coverPx);
            renderer_.push(DrawOp::PopClipPath, s.zOrder, s.clipBounds,
                                   s.stencilDepth, s.scopeDepth, ref,
                                   /*writesPx*/ popRegion,
                                   /*readsPx*/  popRegion,
                                   /*stateKey*/ state_key::make(DrawOp::PopClipPath),
                                   /*isOpaque*/ false);
            s.pathClipStack.pop_back();
        }
        // Close the scheduler scope opened by the matching clipToPath (stencil
        // branch). Rect-clip scopes are closed in restoreState's rect branch.
        renderer_.popScope();
        s.scopeDepth--;
        if (s.stencilDepth > 0) s.stencilDepth--;
    }

    // Clamp the stroke thickness to one physical pixel. Anything thinner
    // aliases badly inside the `abs(d) - thickness*0.5` stroke SDF (the ring
    // becomes sub-pixel and the falloff smoothstep degenerates).
    float strokeFloor(float lineThickness) const
    {
        return juce::jmax(lineThickness, 1.0f / juce::jmax(displayScale_, 1.0f));
    }

    // Logical rect + context transform → physical-pixel integer AABB.
    // Used by every push site to compute a tight writesPx for the scheduler;
    // expand-by-N for AA margins where relevant. Intersects with the caller's
    // active scissor so the result is never larger than what actually paints.
    juce::Rectangle<int> toWritesPx(const juce::Rectangle<float>& logical,
                                    float expandPhysicalPx = 0.0f) const
    {
        auto& s = state();
        auto px = logical.transformedBy(s.transform.scaled(displayScale_));
        if (expandPhysicalPx > 0.0f) px = px.expanded(expandPhysicalPx);
        int bx  = static_cast<int>(std::floor(px.getX()));
        int by  = static_cast<int>(std::floor(px.getY()));
        int bx2 = static_cast<int>(std::ceil(px.getRight()));
        int by2 = static_cast<int>(std::ceil(px.getBottom()));
        juce::Rectangle<int> box { bx, by, bx2 - bx, by2 - by };
        return s.clipBounds.getIntersection(box);
    }

    // True iff the current fill is a solid colour with full alpha AND global
    // opacity is 1. Conservative — returns false for gradients, images, or
    // anything with partial alpha. Used to mark FillRect/FillRoundedRect
    // commands as isOpaque for the Step-4 DCE pass.
    bool currentFillIsOpaque() const
    {
        auto& s = state();
        if (s.opacity < 1.0f) return false;
        if (!s.fill.isColour()) return false;
        return s.fill.colour.getAlpha() == 255;
    }

    // Populate a primitive's colour-source fields from `fill`. Writes into
    // extraB/payload/color; returns the ColorSource tag to pack into
    // shapeFlags. For solid fills the RGBA colour lands in `color`; for
    // gradients, `color` carries (1,1,1,opacity) and the gradient
    // coefficients (origin, dir-or-invRadius, invLen2, rowNorm) go in
    // extraB + payload so the fragment shader can evaluate t per-pixel.
    //
    // `toPhys` is the full record-time transform (paint transform × displayScale)
    // — gradient endpoints are pre-transformed into PHYSICAL pixel space here
    // so the shader doesn't need the forward affine at runtime.
    ColorSource fillColorFields(const juce::FillType& fill,
                                const juce::AffineTransform& toPhys,
                                float opacity,
                                GeometryPrimitive& p) const
    {
        if (!fill.isGradient() || !fill.gradient) {
            p.color[0] = fill.colour.getFloatRed();
            p.color[1] = fill.colour.getFloatGreen();
            p.color[2] = fill.colour.getFloatBlue();
            p.color[3] = fill.colour.getFloatAlpha() * opacity;
            p.extraB[0] = p.extraB[1] = p.extraB[2] = p.extraB[3] = 0.0f;
            p.payload[0] = p.payload[1] = p.payload[2] = p.payload[3] = 0.0f;
            return ColorSource::Solid;
        }
        auto& g = *fill.gradient;
        auto gradT = fill.transform.followedBy(toPhys);
        float x1 = g.point1.x, y1 = g.point1.y;
        float x2 = g.point2.x, y2 = g.point2.y;
        gradT.transformPoint(x1, y1);
        gradT.transformPoint(x2, y2);

        const float rowNorm = renderer_.caches().registerGradient(g);

        p.color[0] = p.color[1] = p.color[2] = 1.0f;
        p.color[3] = opacity;

        if (g.isRadial) {
            float radius = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            float invRadius = (radius > 0.0f) ? 1.0f / radius : 0.0f;
            p.extraB[0] = x1;   p.extraB[1] = y1;
            p.extraB[2] = invRadius; p.extraB[3] = 0.0f;
            p.payload[0] = 0.0f;
            p.payload[1] = rowNorm;
            p.payload[2] = p.payload[3] = 0.0f;
            return ColorSource::Radial;
        }

        float dx = x2 - x1, dy = y2 - y1;
        float len2 = dx * dx + dy * dy;
        float invLen2 = (len2 > 0.0f) ? 1.0f / len2 : 0.0f;
        p.extraB[0] = x1; p.extraB[1] = y1;
        p.extraB[2] = dx; p.extraB[3] = dy;
        p.payload[0] = invLen2;
        p.payload[1] = rowNorm;
        p.payload[2] = p.payload[3] = 0.0f;
        return ColorSource::Linear;
    }

    // Populate invXform01 / invXform23_half. `anchor` is the shape-local origin
    // in LOGICAL coordinates (typically the shape's centre); shapeHalf is the
    // shape's half-extent in logical units.
    void fillInverseTransform(GeometryPrimitive& p,
                              const juce::Point<float>& anchor,
                              const juce::Point<float>& shapeHalf) const
    {
        auto& s = state();
        auto M    = s.transform.scaled(displayScale_);
        auto invM = M.inverted().translated(-anchor.x, -anchor.y);
        p.invXform01[0] = invM.mat00; p.invXform01[1] = invM.mat10;
        p.invXform01[2] = invM.mat01; p.invXform01[3] = invM.mat11;
        p.invXform23_half[0] = invM.mat02;
        p.invXform23_half[1] = invM.mat12;
        p.invXform23_half[2] = shapeHalf.x;
        p.invXform23_half[3] = shapeHalf.y;
    }

    // Populate bbox from a physical-pixel integer rect.
    static void fillBbox(GeometryPrimitive& p, const juce::Rectangle<int>& box)
    {
        p.bbox[0] = static_cast<float>(box.getX());
        p.bbox[1] = static_cast<float>(box.getY());
        p.bbox[2] = static_cast<float>(box.getRight());
        p.bbox[3] = static_cast<float>(box.getBottom());
    }

    // Pack a BlurShape draw command. Builds a GeometryPrimitive carrying
    // the inverse affine (physical → shape-local logical) + shape geometry,
    // uploads it to BlurPipeline's per-frame primitive SSBO, and emits a
    // DrawOp::BlurShape whose arena payload is a BlurDispatchRef pointing
    // at that primitive. Single-primitive (count=1) today; the scheduler's
    // Fuse mode will later pack N compatible primitives into one range and
    // collapse them into a single vkCmdDraw(6, N, ...) dispatch.
    void pushBlurShape(const juce::Rectangle<float>& boundsRect,
                       float cornerSize,
                       uint32_t shapeType,
                       float blurRadius, float falloffRadius,
                       bool inverted, BlurEdge edge,
                       juce::Point<float> lineB, float lineThickness,
                       BlurMode mode)
    {
        if (isClipEmpty()) return;
        if (blurRadius <= 0.0f && falloffRadius <= 0.0f) return;
        auto* bp = renderer_.blurPipeline();
        if (!bp) return;
        auto& s = state();

        // Shape-local anchor: rect/rrect/ellipse are origin-centred with
        // halfSize; lines have A at origin, B at `lineB`; for either we
        // translate so shape-local origin lands at the anchor point.
        juce::Point<float> anchor = (shapeType == 3)
            ? juce::Point<float>{ boundsRect.getX(), boundsRect.getY() }
            : boundsRect.getCentre();

        juce::AffineTransform M    = s.transform.scaled(displayScale_);
        juce::AffineTransform invM = M.inverted();
        // Subtract the anchor (in logical context coords) AFTER the inverse
        // of the physical→logical mapping, so the shader's frag.xy lands at
        // the shape-local origin.
        juce::AffineTransform invMshift = invM.translated(-anchor.x, -anchor.y);

        const float shapeHalfX = boundsRect.getWidth()  * 0.5f;
        const float shapeHalfY = boundsRect.getHeight() * 0.5f;

        // blurStep = physical texels per user-logical pixel. Folds in both
        // the user's transform scale AND displayScale so the shader's kernel
        // step converts 1 user-logical pixel to the right number of physical
        // texels — respecting addTransform(scale(...)) AND keeping loop
        // count fixed across displayScale (2x retina just samples every
        // other physical pixel, same tap count).
        const float transformScale = s.transform.getScaleFactor();
        const float blurStep = transformScale * displayScale_;
        const float falloffC = juce::jmax(0.001f, falloffRadius);

        // Scheduler bbox: shape AABB + falloff (where output visibly differs
        // from source). Reads extend further by maxRadius (kernel reach).
        // Both in physical pixels, intersected with scissor.
        juce::Rectangle<float> shapeLogical;
        if (shapeType == 3) {
            // Capsule: boundsRect encodes A at (x,y); lineB is B relative to A;
            // lineThickness is the cross-section half-width. Expand the AB
            // segment by lineThickness.
            float ax = boundsRect.getX(), ay = boundsRect.getY();
            float bx = ax + lineB.x,      by = ay + lineB.y;
            float xmin = juce::jmin(ax, bx) - lineThickness;
            float ymin = juce::jmin(ay, by) - lineThickness;
            float xmax = juce::jmax(ax, bx) + lineThickness;
            float ymax = juce::jmax(ay, by) + lineThickness;
            shapeLogical = { xmin, ymin, xmax - xmin, ymax - ymin };
        } else {
            shapeLogical = boundsRect;
            // Outline/stroke variants spill by lineThickness/2 outside the
            // rect — include it in the shape AABB to keep writesPx tight.
            if (lineThickness > 0.0f)
                shapeLogical = shapeLogical.expanded(lineThickness * 0.5f);
        }
        const float falloffPhys = falloffC * blurStep;
        const float radiusPhys  = blurRadius * blurStep;
        auto writes = toWritesPx(shapeLogical, falloffPhys);
        // readsPx is where the fragment shader samples from. Not bounded by
        // scissor — clamp only to the framebuffer so we don't claim reads
        // beyond the GPU's texture bounds.
        const juce::Rectangle<int> vpBounds {
            0, 0,
            static_cast<int>(renderer_.target().width()),
            static_cast<int>(renderer_.target().height())
        };
        auto reads = writes.expanded(static_cast<int>(std::ceil(radiusPhys)))
                           .getIntersection(vpBounds);

        // Build the GeometryPrimitive. Layout matches shaders/geometry.glsl.
        // Stroke flag is true for non-capsule shapes with lineThickness > 0;
        // capsule consumes lineThickness as its own cross-section, so the
        // shader treats it as fill.
        const bool isStroke = (shapeType != 3) && (lineThickness > 0.0f);
        GeometryPrimitive prim {};
        prim.bbox[0] = static_cast<float>(writes.getX());
        prim.bbox[1] = static_cast<float>(writes.getY());
        prim.bbox[2] = static_cast<float>(writes.getRight());
        prim.bbox[3] = static_cast<float>(writes.getBottom());
        prim.flags[0] = shapeType;
        prim.flags[1] = packShapeFlags(inverted, edge, /*fillRule*/ 0u, isStroke);
        prim.flags[2] = 0;
        prim.flags[3] = 0;
        prim.invXform01[0] = invMshift.mat00;
        prim.invXform01[1] = invMshift.mat10;
        prim.invXform01[2] = invMshift.mat01;
        prim.invXform01[3] = invMshift.mat11;
        prim.invXform23_half[0] = invMshift.mat02;
        prim.invXform23_half[1] = invMshift.mat12;
        prim.invXform23_half[2] = shapeHalfX;
        prim.invXform23_half[3] = shapeHalfY;
        prim.extraA[0] = cornerSize;          // cornerRadius (tag 1)
        prim.extraA[1] = lineThickness;       // thickness (tags 0..2 outline, 3 capsule half)
        prim.extraA[2] = lineB.x;             // capsule B.x
        prim.extraA[3] = lineB.y;             // capsule B.y
        prim.extraB[0] = blurRadius;          // logical
        prim.extraB[1] = falloffC;            // logical
        prim.extraB[2] = blurStep;
        prim.extraB[3] = 0.0f;                // strokeHalfWidth — path-only

        const uint32_t arenaOffset = renderer_.arena_push(prim);
        BlurDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u,
                              arenaOffset, static_cast<uint32_t>(mode) };

        // pipelineHint bundles shape + mode + edge + inverted so two blurs
        // can fuse only when they share shader configuration.
        uint32_t hint = static_cast<uint32_t>(shapeType)
                      | (static_cast<uint32_t>(mode)  <<  4)
                      | (static_cast<uint32_t>(edge)  <<  8)
                      | ((inverted ? 1u : 0u)         << 12);
        renderer_.push(DrawOp::BlurShape, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  reads,
            /*stateKey*/ state_key::make(DrawOp::BlurShape, hint),
            /*isOpaque*/ false);
    }

    // Pack a BlurPath draw command. Flattens the path to line segments
    // in PHYSICAL pixels, uploads them to PathPipeline's per-frame SSBO
    // (shared with FillPath + clip-path), and emits a GeometryPrimitive
    // (tag=5) whose flags[2..3] hold (segmentStart, segmentCount). All
    // distances on the primitive are in physical pixels — blurStep=1.0
    // — because the segments are pre-baked physical; a later phase flips
    // this to local-coord segments + non-unity blurStep.
    void pushBlurPath(const juce::Path& path,
                      const juce::AffineTransform& t,
                      float blurRadius, float falloffRadius,
                      bool inverted, BlurEdge edge,
                      float strokeHalfWidth,
                      BlurMode mode)
    {
        if (isClipEmpty() || path.isEmpty()) return;
        if (blurRadius <= 0.0f && falloffRadius <= 0.0f) return;

        auto* pp = renderer_.pathPipeline();
        auto* bp = renderer_.blurPipeline();
        if (!pp || !bp) return;

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        const uint32_t segStart = pp->uploadSegments(
            scratchSegments_.data(),
            static_cast<uint32_t>(scratchSegments_.size()));
        const uint32_t segCount = static_cast<uint32_t>(scratchSegments_.size());
        const uint32_t fillRule = path.isUsingNonZeroWinding() ? 0u : 1u;

        // Pre-multiply radius params to physical pixels so the shader runs
        // entirely in one coord space (segments are already physical too).
        // blurStep == transformScale × displayScale on the logical side;
        // for BlurPath we collapse it into the radius/falloff/stroke and
        // set the primitive's blurStep to 1.0 so the shader sees physical.
        const float physStep       = s.transform.getScaleFactor() * displayScale_;
        const float maxRadiusPhys  = blurRadius      * physStep;
        const float falloffPhys    = juce::jmax(0.001f, falloffRadius * physStep);
        const float strokeHWPhys   = strokeHalfWidth * physStep;

        // Path AABB → physical px, expanded by falloff for the dirty rect,
        // further expanded by maxRadius for the kernel read footprint.
        auto px = pathBounds.transformedBy(combined).expanded(falloffPhys + 1.0f);
        int bx  = static_cast<int>(std::floor(px.getX()));
        int by  = static_cast<int>(std::floor(px.getY()));
        int bx2 = static_cast<int>(std::ceil(px.getRight()));
        int by2 = static_cast<int>(std::ceil(px.getBottom()));
        juce::Rectangle<int> writes { bx, by, bx2 - bx, by2 - by };
        writes = s.clipBounds.getIntersection(writes);
        const juce::Rectangle<int> vpBounds {
            0, 0,
            static_cast<int>(renderer_.target().width()),
            static_cast<int>(renderer_.target().height())
        };
        auto reads = writes.expanded(static_cast<int>(std::ceil(maxRadiusPhys)))
                           .getIntersection(vpBounds);

        GeometryPrimitive prim {};
        prim.bbox[0] = static_cast<float>(writes.getX());
        prim.bbox[1] = static_cast<float>(writes.getY());
        prim.bbox[2] = static_cast<float>(writes.getRight());
        prim.bbox[3] = static_cast<float>(writes.getBottom());
        prim.flags[0] = static_cast<uint32_t>(GeometryTag::Path);
        prim.flags[1] = packShapeFlags(inverted, edge, fillRule,
                                       /*stroke*/ strokeHalfWidth > 0.0f);
        prim.flags[2] = segStart;
        prim.flags[3] = segCount;
        // Identity inverse — path segments are already in physical pixels,
        // so the shader's fragCoord is already in the right coord space.
        prim.invXform01[0] = 1.0f; prim.invXform01[1] = 0.0f;
        prim.invXform01[2] = 0.0f; prim.invXform01[3] = 1.0f;
        prim.invXform23_half[0] = 0.0f;
        prim.invXform23_half[1] = 0.0f;
        prim.invXform23_half[2] = 0.0f;
        prim.invXform23_half[3] = 0.0f;
        prim.extraA[0] = 0.0f;
        prim.extraA[1] = 0.0f;
        prim.extraA[2] = 0.0f;
        prim.extraA[3] = 0.0f;
        prim.extraB[0] = maxRadiusPhys;   // physical
        prim.extraB[1] = falloffPhys;     // physical
        prim.extraB[2] = 1.0f;            // blurStep — radius already physical
        prim.extraB[3] = strokeHWPhys;    // physical (0 for fill)

        const uint32_t arenaOffset = renderer_.arena_push(prim);
        BlurDispatchRef ref { /*firstInstance*/ 0u, /*primCount*/ 1u,
                              arenaOffset, static_cast<uint32_t>(mode) };

        uint32_t hint = 0xFu                                               // path (distinguish from shape)
                      | (static_cast<uint32_t>(mode)         <<  4)
                      | (static_cast<uint32_t>(edge)         <<  8)
                      | ((inverted ? 1u : 0u)                << 12)
                      | ((strokeHalfWidth > 0.0f ? 1u : 0u)  << 13);       // stroke vs fill
        renderer_.push(DrawOp::BlurPath, s.zOrder, s.clipBounds,
            s.stencilDepth, s.scopeDepth, ref,
            /*writesPx*/ writes,
            /*readsPx*/  reads,
            /*stateKey*/ state_key::make(DrawOp::BlurPath, hint, 0,
                                         static_cast<uint16_t>(fillRule)),
            /*isOpaque*/ false);
    }

    // Flatten a path to a flat list of line segments in the supplied
    // transform's target space, packed as vec4 (p0.xy, p1.xy) ready for the
    // PathPipeline storage buffer. Shares the same subdivision strategy as
    // flattenPathToFan but emits segments between consecutive flattened
    // points (not triangle-fan triples), and skips the orientation
    // normalisation pass — analytical winding in the fragment shader does
    // not depend on contour orientation.
    void flattenPathToSegments(const juce::Path& path,
                               const juce::AffineTransform& combined,
                               std::vector<glm::vec4>& segs)
    {
        scratchPoints_.clear();
        constexpr float flatTol = 0.5f;
        const float SUBPATH_MARKER = -std::numeric_limits<float>::infinity();

        auto transformPt = [&](float x, float y) -> glm::vec2 {
            combined.transformPoint(x, y);
            return { x, y };
        };

        auto flattenCubic = [&](glm::vec2 p0, glm::vec2 c1, glm::vec2 c2, glm::vec2 p3) {
            scratchSegStack_.clear();
            scratchSegStack_.push_back({ p0, c1, c2, p3 });
            while (!scratchSegStack_.empty()) {
                auto [a, b, c, d] = scratchSegStack_.back();
                scratchSegStack_.pop_back();
                float dx = d.x - a.x, dy = d.y - a.y;
                float len2 = dx * dx + dy * dy;
                float d1, d2;
                if (len2 > 0.0001f) {
                    float inv = 1.0f / len2;
                    float t1 = ((b.x-a.x)*dx + (b.y-a.y)*dy) * inv;
                    float t2 = ((c.x-a.x)*dx + (c.y-a.y)*dy) * inv;
                    d1 = (a.x+t1*dx-b.x)*(a.x+t1*dx-b.x) + (a.y+t1*dy-b.y)*(a.y+t1*dy-b.y);
                    d2 = (a.x+t2*dx-c.x)*(a.x+t2*dx-c.x) + (a.y+t2*dy-c.y)*(a.y+t2*dy-c.y);
                } else {
                    d1 = (b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y);
                    d2 = (c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y);
                }
                if (d1 <= flatTol*flatTol && d2 <= flatTol*flatTol) {
                    scratchPoints_.push_back(d);
                } else {
                    auto mid = [](glm::vec2 u, glm::vec2 v) { return (u+v)*0.5f; };
                    auto ab = mid(a,b), bc = mid(b,c), cd = mid(c,d);
                    auto abc = mid(ab,bc), bcd = mid(bc,cd), abcd = mid(abc,bcd);
                    scratchSegStack_.push_back({ abcd, bcd, cd, d });
                    scratchSegStack_.push_back({ a, ab, abc, abcd });
                }
            }
        };

        auto flattenQuad = [&](glm::vec2 p0, glm::vec2 c, glm::vec2 p2) {
            flattenCubic(p0, p0 + (2.0f/3.0f)*(c-p0), p2 + (2.0f/3.0f)*(c-p2), p2);
        };

        juce::Path::Iterator iter(path);
        glm::vec2 lastPt(0), subpathStart(0);

        while (iter.next()) {
            switch (iter.elementType) {
                case juce::Path::Iterator::startNewSubPath: {
                    auto pt = transformPt(iter.x1, iter.y1);
                    scratchPoints_.push_back({ SUBPATH_MARKER, 0 });
                    scratchPoints_.push_back(pt);
                    subpathStart = pt; lastPt = pt; break;
                }
                case juce::Path::Iterator::lineTo: {
                    auto pt = transformPt(iter.x1, iter.y1);
                    scratchPoints_.push_back(pt); lastPt = pt; break;
                }
                case juce::Path::Iterator::quadraticTo: {
                    auto c = transformPt(iter.x1, iter.y1);
                    auto p = transformPt(iter.x2, iter.y2);
                    flattenQuad(lastPt, c, p); lastPt = p; break;
                }
                case juce::Path::Iterator::cubicTo: {
                    auto c1 = transformPt(iter.x1, iter.y1);
                    auto c2 = transformPt(iter.x2, iter.y2);
                    auto p  = transformPt(iter.x3, iter.y3);
                    flattenCubic(lastPt, c1, c2, p); lastPt = p; break;
                }
                case juce::Path::Iterator::closePath:
                    scratchPoints_.push_back(subpathStart);
                    lastPt = subpathStart; break;
            }
        }

        // Walk the flattened point stream and emit a segment for each
        // consecutive pair within a subpath. SUBPATH_MARKER breaks the chain
        // so segments never cross contour boundaries. Zero-length segments
        // are discarded — they contribute nothing to either distance or
        // winding and would waste SSBO slots.
        glm::vec2 prev(0); bool havePrev = false;
        for (size_t i = 0; i < scratchPoints_.size(); ++i) {
            if (scratchPoints_[i].x == SUBPATH_MARKER) {
                havePrev = false;
                continue;
            }
            if (!havePrev) {
                prev = scratchPoints_[i];
                havePrev = true;
                continue;
            }
            glm::vec2 cur = scratchPoints_[i];
            if (cur != prev)
                segs.push_back({ prev.x, prev.y, cur.x, cur.y });
            prev = cur;
        }
    }


    Renderer& renderer_;
    float     displayScale_;
    std::vector<RecordState> stateStack_;
    mutable uint64_t frameId_ = 0;

    // Flatten scratch buffers — cleared-but-not-freed between paths so the
    // hot path is allocation-free after the first frame. Used by both
    // Graphics::fillPath (segments → PathPipeline SSBO) and clipToPath
    // (segments → ClipPipeline via the same shared SSBO).
    struct Seg { glm::vec2 a, b, c, d; };
    std::vector<glm::vec2> scratchPoints_;
    std::vector<Seg>       scratchSegStack_;
    std::vector<glm::vec4> scratchSegments_;

    // drawGlyphs scratch — built in record phase, uploaded to FillPipeline's
    // primitive SSBO in one go. Per-glyph atlas-page indices land in the
    // arena after the dispatch ref, so the worker can rebind the shape
    // sampler when the page changes.
    std::vector<GeometryPrimitive> scratchGlyphPrims_;
    std::vector<uint32_t>          scratchGlyphPages_;
};

} // namespace jvk
