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
        renderer_.push(DrawOp::PushClipRect, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, PushClipRectParams { transformed });
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

        auto& s = state();
        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        // Pack affine into the 6-float form the stencil shader consumes.
        // Shader: pos = col0*x + col1*y + col2, where
        //   col0 = (m00, m10), col1 = (m01, m11), col2 = (m02, m12)
        float xform[6] = {
            combined.mat00, combined.mat10,   // col0
            combined.mat01, combined.mat11,   // col1
            combined.mat02, combined.mat12,   // col2
        };

        auto& caches = renderer_.caches();
        uint64_t hash = ResourceCaches::hashPath(path);
        caches.markPathSeen(hash);

        const CachedPathMesh* cached  = caches.pathMeshes().find(hash);
        uint32_t              vCount  = 0;
        uint32_t              fanOff  = 0;
        juce::Rectangle<int>  fanBounds;

        if (cached) {
            // Cache hit — skip flatten + skip arena push. Bounds come from
            // the cached local bounds transformed into pixel space.
            vCount    = cached->vertexCount;
            fanBounds = cached->localBounds.transformedBy(combined)
                               .getSmallestIntegerContainer();
        } else {
            // Miss — flatten in LOCAL space so the mesh is transform-independent
            // if we later promote it to the cache.
            juce::AffineTransform identity;
            juce::Rectangle<int> localInt;
            flattenPathToFan(path, identity, scratchFanVerts_, localInt);
            vCount = static_cast<uint32_t>(scratchFanVerts_.size());
            fanBounds = juce::Rectangle<float>(localInt.toFloat())
                               .transformedBy(combined)
                               .getSmallestIntegerContainer();

            // Push local verts into the arena for this frame's draw.
            fanOff = renderer_.arena_offset();
            if (!scratchFanVerts_.empty())
                renderer_.arena_pushSpan(std::span<const UIVertex>(
                    scratchFanVerts_.data(), scratchFanVerts_.size()));

            // Promote to the mesh cache only on the second sighting — avoids
            // caching one-shot animated paths whose hash changes each frame.
            // All cached meshes suballocate into one shared PathMeshPool buffer
            // so the vertex-buffer bind is amortised across the whole frame.
            if (caches.pathWasSeenLastFrame(hash) && vCount > 0) {
                auto slot = caches.pathMeshPool().alloc(vCount);
                if (slot.ok) {
                    VkDeviceSize size   = static_cast<VkDeviceSize>(vCount) * sizeof(UIVertex);
                    VkDeviceSize dstOff = static_cast<VkDeviceSize>(slot.firstVertex) * sizeof(UIVertex);

                    auto staging = renderer_.device().staging().alloc(size);
                    memcpy(staging.mappedPtr, scratchFanVerts_.data(), static_cast<size_t>(size));
                    renderer_.device().upload(staging, caches.pathMeshPool().buffer(), size, dstOff);

                    CachedPathMesh entry;
                    entry.firstVertex = slot.firstVertex;
                    entry.vertexCount = vCount;
                    entry.localBounds = localInt.toFloat();

                    auto& inserted = caches.pathMeshes().insert(hash, std::move(entry));
                    cached = &inserted;
                }
            }
        }

        s.clipBounds = s.clipBounds.getIntersection(fanBounds);

        PushClipPathParams params {};
        params.vertexCount       = vCount;
        params.vertexArenaOffset = fanOff;
        params.cachedMesh        = cached;
        params.pathBounds        = fanBounds;
        memcpy(params.transform, xform, sizeof(xform));
        renderer_.push(DrawOp::PushClipPath, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, params);

        PathClipFan clip;
        clip.fanBounds      = fanBounds;
        clip.fanArenaOffset = fanOff;
        clip.vertexCount    = vCount;
        clip.cachedMesh     = cached;
        memcpy(clip.transform, xform, sizeof(xform));
        s.pathClipStack.push_back(clip);

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
            if (isPathClip)
                recordPopClip();
            else {
                renderer_.push(DrawOp::PopClip, old.zOrder, old.clipBounds,
                               old.stencilDepth, old.scopeDepth, PopClipParams {});
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
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        // Stage gradient LUT upload if this is a gradient fill
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillRect, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            FillRectParams { r, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillRectList(const juce::RectangleList<float>& list) override
    {
        for (auto& r : list) fillRect(r);
    }

    void fillPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        if (isClipEmpty()) return;
        auto bounds = path.getBoundsTransformed(t);
        saveState();
        clipToPath(path, t);
        fillRect(bounds);
        restoreState();
    }

    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !img.isValid()) return;
        auto& s = state();
        uint64_t hash = reinterpret_cast<uint64_t>(img.getPixelData().get());

        renderer_.caches().getTexture(hash, img);

        renderer_.push(DrawOp::DrawImage, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawImageParams { hash, t.followedBy(s.transform), s.opacity, displayScale_,
                              img.getWidth(), img.getHeight() });
    }

    void drawLine(const juce::Line<float>& line) override
    {
        drawLineWithThickness(line, 1.0f);
    }

    void drawLineWithThickness(const juce::Line<float>& line, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::DrawLine, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawLineParams { line, lineThickness, fi, s.transform, s.opacity, displayScale_ });
    }

    void setFont(const juce::Font& f) override { state().font = f; }
    const juce::Font& getFont() override { return state().font; }

    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                    juce::Span<const juce::Point<float>> positions,
                    const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || glyphs.empty()) return;
        auto& s = state();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        DrawGlyphsParams params;
        params.glyphCount = static_cast<uint32_t>(glyphs.size());
        params.transform = t.followedBy(s.transform);
        params.fontIndex = renderer_.captureFont(s.font);
        params.fillIndex = renderer_.captureFill(s.fill);
        params.opacity = s.opacity;
        params.scale = displayScale_;
        renderer_.push(DrawOp::DrawGlyphs, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth, params);
        // Append glyph POD data to arena (align before float data)
        renderer_.arena_pushSpan(std::span<const uint16_t>(glyphs.data(), glyphs.size()));
        renderer_.arena_align(4); // Point<float> requires 4-byte alignment
        renderer_.arena_pushSpan(std::span<const juce::Point<float>>(positions.data(), positions.size()));
    }

    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillRoundedRect, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            FillRoundedRectParams { r, cornerSize, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillEllipse, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            FillEllipseParams { area, fi, s.transform, s.opacity, displayScale_ });
    }

    void drawRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                              float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::StrokeRoundedRect, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            StrokeRoundedRectParams { rect, cornerSize, lineThickness, fi, s.transform,
                                      s.opacity, displayScale_ });
    }

    // Outline rectangle — stroked rounded rect with cornerSize = 0
    void drawRect(const juce::Rectangle<float>& rect, float lineThickness) override
    {
        drawRoundedRectangle(rect, 0.0f, lineThickness);
    }

    void drawEllipse(const juce::Rectangle<float>& area, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.caches().registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::StrokeEllipse, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            StrokeEllipseParams { area, lineThickness, fi, s.transform, s.opacity, displayScale_ });
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
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            EffectBlendParams { v, v, v, region, displayScale_ });
    }

    void brighten(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        float v = 1.0f + amount;
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            EffectBlendParams { v, v, v, region, displayScale_ });
    }

    void tint(juce::Colour c, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            EffectBlendParams { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), region, displayScale_ });
    }

    void warmth(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::EffectBlend, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            EffectBlendParams { 1.0f + amount * 0.2f, 1.0f, 1.0f - amount * 0.1f, region, displayScale_ });
    }

    void blur(float radius, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::EffectKernel, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            BlurParams { radius, region, displayScale_ });
    }

    // =========================================================================
    // Shape-aware variable-radius blur. Inside the shape blurs at `blurRadius`;
    // the effective radius ramps down to 0 across `falloffRadius` logical px.
    // `edge` controls where the falloff band sits relative to the shape edge.
    // `inverted = true` blurs OUTSIDE the shape instead (ramp direction flips).
    // All distances are logical pixels; displayScale is folded in per-pixel.
    // =========================================================================

    void blurRect(const juce::Rectangle<float>& rect,
                  float blurRadius, float falloffRadius,
                  bool inverted = false, BlurEdge edge = BlurEdge::Centered)
    {
        pushBlurShape(rect, 0.0f, /*shapeType*/ 0,
                      blurRadius, falloffRadius, inverted, edge,
                      /*lineB*/ {0,0}, /*lineThickness*/ 0.0f);
    }

    void blurRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                              float blurRadius, float falloffRadius,
                              bool inverted = false, BlurEdge edge = BlurEdge::Centered)
    {
        pushBlurShape(rect, cornerSize, /*shapeType*/ 1,
                      blurRadius, falloffRadius, inverted, edge,
                      {0,0}, 0.0f);
    }

    void blurEllipse(const juce::Rectangle<float>& area,
                     float blurRadius, float falloffRadius,
                     bool inverted = false, BlurEdge edge = BlurEdge::Centered)
    {
        pushBlurShape(area, 0.0f, /*shapeType*/ 2,
                      blurRadius, falloffRadius, inverted, edge,
                      {0,0}, 0.0f);
    }

    void blurLine(const juce::Line<float>& line, float thickness,
                  float blurRadius, float falloffRadius,
                  bool inverted = false, BlurEdge edge = BlurEdge::Centered)
    {
        // Shape-local anchor at endpoint A; B is stored relative to A.
        auto a = line.getStart();
        auto b = line.getEnd();
        juce::Rectangle<float> anchor { a.x, a.y, 0.0f, 0.0f };
        juce::Point<float> bRel { b.x - a.x, b.y - a.y };
        pushBlurShape(anchor, 0.0f, /*shapeType*/ 3,
                      blurRadius, falloffRadius, inverted, edge,
                      { bRel.x, bRel.y }, thickness * 0.5f);
    }

    void drawShader(Shader& shader, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::DrawShader, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawShaderParams { &shader, region, displayScale_ });
    }

    Renderer& getRenderer() { return renderer_; }

private:
    // Per-path-clip fan bookkeeping. Either `cachedMesh` holds a stable GPU
    // buffer (cache hit / freshly promoted) or `fanArenaOffset` points at the
    // per-frame arena range from PushClipPath.
    struct PathClipFan {
        uint32_t               fanArenaOffset = 0;
        uint32_t               vertexCount    = 0;
        const CachedPathMesh*  cachedMesh     = nullptr;
        float                  transform[6]   = {1,0,0,1,0,0};
        juce::Rectangle<int>   fanBounds;
    };

    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        float                 zOrder = 0.0f;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
        std::vector<PathClipFan> pathClipStack;
    };

    RecordState& state() { return stateStack_.back(); }
    const RecordState& state() const { return stateStack_.back(); }

    // Record a PopClip for the most recent path clip. Uses the same cached
    // mesh (or arena vertex range) and transform as the paired PushClipPath —
    // second INVERT stencil pass toggles the bit back to zero.
    void recordPopClip()
    {
        auto& s = state();
        PopClipParams params {};
        if (!s.pathClipStack.empty()) {
            auto& fan = s.pathClipStack.back();
            params.vertexCount       = fan.vertexCount;
            params.vertexArenaOffset = fan.fanArenaOffset;
            params.cachedMesh        = fan.cachedMesh;
            memcpy(params.transform, fan.transform, sizeof(fan.transform));
            params.fanBounds         = fan.fanBounds;
            renderer_.push(DrawOp::PopClip, s.zOrder, s.clipBounds,
                           s.stencilDepth, s.scopeDepth, params);
            s.pathClipStack.pop_back();
        } else {
            renderer_.push(DrawOp::PopClip, s.zOrder, s.clipBounds,
                           s.stencilDepth, s.scopeDepth, params);
        }
        s.scopeDepth--;
        if (s.stencilDepth > 0) s.stencilDepth--;
    }

    // Pack a BlurShape draw command. Handles the inverse-affine computation
    // that maps physical fragment coords back into shape-local logical space.
    void pushBlurShape(const juce::Rectangle<float>& boundsRect,
                       float cornerSize,
                       uint32_t shapeType,
                       float blurRadius, float falloffRadius,
                       bool inverted, BlurEdge edge,
                       juce::Point<float> lineB, float lineThickness)
    {
        if (isClipEmpty()) return;
        if (blurRadius <= 0.0f && falloffRadius <= 0.0f) return;
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

        BlurShapeParams p {};
        p.invXform[0] = invMshift.mat00; p.invXform[1] = invMshift.mat10;
        p.invXform[2] = invMshift.mat01; p.invXform[3] = invMshift.mat11;
        p.invXform[4] = invMshift.mat02; p.invXform[5] = invMshift.mat12;

        p.shapeHalf[0] = boundsRect.getWidth()  * 0.5f;
        p.shapeHalf[1] = boundsRect.getHeight() * 0.5f;
        p.lineB[0]     = lineB.x;
        p.lineB[1]     = lineB.y;

        p.maxRadius     = blurRadius;
        p.falloff       = juce::jmax(0.001f, falloffRadius);
        p.displayScale  = displayScale_;
        p.cornerRadius  = cornerSize;
        p.lineThickness = lineThickness;

        p.shapeType     = shapeType;
        p.edgePlacement = static_cast<uint32_t>(edge);
        p.inverted      = inverted ? 1u : 0u;

        renderer_.push(DrawOp::BlurShape, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, p);
    }

    // Flatten a path into a triangle fan for stencil rendering.
    // Uses instance-level scratch buffers — no per-call heap allocations on
    // the steady state.
    void flattenPathToFan(const juce::Path& path,
                          const juce::AffineTransform& combined,
                          std::vector<UIVertex>& fanVerts,
                          juce::Rectangle<int>& bounds)
    {
        fanVerts.clear();
        scratchPoints_.clear();
        // (scratchSegStack_ cleared per curve below)

        float minX =  std::numeric_limits<float>::max(), minY = minX;
        float maxX = -std::numeric_limits<float>::max(), maxY = maxX;

        const float SUBPATH_MARKER = -std::numeric_limits<float>::infinity();
        constexpr float flatTol = 0.5f;

        juce::Path::Iterator iter(path);
        glm::vec2 lastPt(0), subpathStart(0);

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

        // Phase 2: orientation normalisation.
        //
        // The two-sided INCR/DECR stencil trick computes the true signed
        // winding number per pixel only when every contour carries the
        // orientation that matches its containment depth:
        //   even depth (outer, island nested inside a hole) → CCW
        //   odd  depth (hole, hole-inside-an-island)        → CW
        //
        // SVG author orientation conventions are not enforced anywhere in
        // the input path — Chrome/Firefox/Skia scanline-fill by counting
        // edge dy-signs so they don't need this, but a fan emits +1 / -1
        // triangle contributions and relies on contours being oriented
        // correctly for the overlaps to cancel. Without this step,
        // multi-subpath paths (text, SVG logos) accumulate winding where
        // they shouldn't and the whole bounding box fills in.
        //
        // We work in-place over scratchPoints_ using the SUBPATH_MARKER
        // bookends already inserted in phase 1. For each subpath we
        // compute the shoelace signed area + axis-aligned bounds, derive
        // containment depth by bbox nesting against every other subpath,
        // and reverse the point order for any subpath whose signed-area
        // sign doesn't match the expected orientation for its depth.
        scratchSpans_.clear();
        {
            size_t i = 0;
            while (i < scratchPoints_.size()) {
                if (scratchPoints_[i].x != SUBPATH_MARKER) { ++i; continue; }
                size_t start = i + 1;
                size_t end   = start;
                while (end < scratchPoints_.size() &&
                       scratchPoints_[end].x != SUBPATH_MARKER)
                    ++end;
                if (end > start) {
                    SpanInfo s { start, end, 0.0f,
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                -std::numeric_limits<float>::max(),
                                -std::numeric_limits<float>::max() };
                    double area = 0;
                    size_t n = end - start;
                    for (size_t k = 0; k < n; ++k) {
                        auto p  = scratchPoints_[start + k];
                        auto pn = scratchPoints_[start + ((k + 1) % n)];
                        area += double(p.x) * pn.y - double(pn.x) * p.y;
                        s.bl = std::min(s.bl, p.x); s.bt = std::min(s.bt, p.y);
                        s.br = std::max(s.br, p.x); s.bb = std::max(s.bb, p.y);
                    }
                    s.signedArea = static_cast<float>(area * 0.5);
                    scratchSpans_.push_back(s);
                }
                i = end;
            }
        }

        // Containment depth: how many other spans' bboxes strictly enclose
        // this one. Bbox-nesting is cheap and matches font/SVG topology for
        // the overwhelming common case (letter with holes, ring, nested
        // island) — it breaks only for self-intersecting contours where
        // no stencil-fan algorithm can be correct without Skia / Clipper.
        for (size_t i = 0; i < scratchSpans_.size(); ++i) {
            int depth = 0;
            auto& si = scratchSpans_[i];
            for (size_t j = 0; j < scratchSpans_.size(); ++j) {
                if (i == j) continue;
                auto& sj = scratchSpans_[j];
                if (sj.bl <= si.bl && sj.bt <= si.bt &&
                    sj.br >= si.br && sj.bb >= si.bb)
                    ++depth;
            }
            // In Y-down screen space a "CCW-as-drawn" contour has negative
            // shoelace area — we keep that as the outer convention (depth
            // even) and flip holes to match.
            bool wantNegative = ((depth & 1) == 0);
            bool isNegative   = si.signedArea < 0.0f;
            if (wantNegative != isNegative && si.end > si.start + 1) {
                std::reverse(scratchPoints_.begin() + static_cast<ptrdiff_t>(si.start),
                             scratchPoints_.begin() + static_cast<ptrdiff_t>(si.end));
            }
        }

        // Phase 3: build triangle fan from the now-orientation-normalised points
        glm::vec4 dummy(0);
        glm::vec2 fanCenter(0), prevPt(0);
        bool inSubpath = false;

        auto updateBounds = [&](float x, float y) {
            minX = std::min(minX, x); minY = std::min(minY, y);
            maxX = std::max(maxX, x); maxY = std::max(maxY, y);
        };

        for (size_t i = 0; i < scratchPoints_.size(); i++) {
            if (scratchPoints_[i].x == SUBPATH_MARKER) {
                if (++i >= scratchPoints_.size()) break;
                fanCenter = scratchPoints_[i]; prevPt = fanCenter;
                inSubpath = true;
                updateBounds(fanCenter.x, fanCenter.y);
                continue;
            }
            if (!inSubpath) continue;
            glm::vec2 pt = scratchPoints_[i];
            updateBounds(pt.x, pt.y);
            fanVerts.push_back({ fanCenter, dummy, {}, dummy, dummy });
            fanVerts.push_back({ prevPt,   dummy, {}, dummy, dummy });
            fanVerts.push_back({ pt,        dummy, {}, dummy, dummy });
            prevPt = pt;
        }

        if (fanVerts.empty()) {
            bounds = {};
        } else {
            // Don't clamp to 0 — local-space meshes are centred at the origin
            // and have negative coordinates. Clamping here corrupts the cached
            // localBounds so the transformed scissor cuts the path.
            int bx = static_cast<int>(std::floor(minX)) - 1;
            int by = static_cast<int>(std::floor(minY)) - 1;
            int bx2 = static_cast<int>(std::ceil(maxX)) + 1;
            int by2 = static_cast<int>(std::ceil(maxY)) + 1;
            bounds = { bx, by, bx2 - bx, by2 - by };
        }
    }

    Renderer& renderer_;
    float     displayScale_;
    std::vector<RecordState> stateStack_;
    mutable uint64_t frameId_ = 0;

    // Flatten scratch buffers — cleared-but-not-freed between paths so the
    // hot path is allocation-free after the first frame.
    struct Seg { glm::vec2 a, b, c, d; };
    struct SpanInfo {
        size_t start, end;
        float  signedArea;
        float  bl, bt, br, bb;
    };
    std::vector<UIVertex>  scratchFanVerts_;
    std::vector<glm::vec2> scratchPoints_;
    std::vector<Seg>       scratchSegStack_;
    std::vector<SpanInfo>  scratchSpans_;
};

} // namespace jvk
