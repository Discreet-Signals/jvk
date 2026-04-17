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

    void clipToPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        auto& s = state();
        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        // Flatten path to triangle fan in physical pixel space
        std::vector<UIVertex> fanVerts;
        juce::Rectangle<int> fanBounds;
        flattenPathToFan(path, combined, fanVerts, fanBounds);

        s.clipBounds = s.clipBounds.getIntersection(fanBounds);

        // Push stencil write command with fan vertices
        PushClipPathParams params;
        params.vertexCount = static_cast<uint32_t>(fanVerts.size());
        params.pathBounds = fanBounds;
        renderer_.push(DrawOp::PushClipPath, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, params);
        if (!fanVerts.empty())
            renderer_.arena_pushSpan(std::span<const UIVertex>(fanVerts.data(), fanVerts.size()));

        // Store reversed fan for the matching PopClip
        PathClipFan clip;
        clip.fanBounds = fanBounds;
        clip.reversedFanVerts = std::move(fanVerts);
        // Reverse winding: swap v1/v2 in each triangle for stencil decrement
        for (size_t i = 0; i + 2 < clip.reversedFanVerts.size(); i += 3)
            std::swap(clip.reversedFanVerts[i + 1], clip.reversedFanVerts[i + 2]);
        s.pathClipStack.push_back(std::move(clip));

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
        // Return logical coords to JUCE (internal clipBounds are in pixels)
        auto& cb = state().clipBounds;
        float inv = 1.0f / displayScale_;
        return { static_cast<int>(cb.getX() * inv), static_cast<int>(cb.getY() * inv),
                 static_cast<int>(cb.getWidth() * inv), static_cast<int>(cb.getHeight() * inv) };
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

    void drawShader(Shader& shader, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::DrawShader, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawShaderParams { &shader, region, displayScale_ });
    }

    Renderer& getRenderer() { return renderer_; }

private:
    // Per-path-clip fan data stored during recording, consumed by PopClip
    struct PathClipFan {
        std::vector<UIVertex> reversedFanVerts;
        juce::Rectangle<int> fanBounds;
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

    // Record a PopClip for the most recent path clip, with reversed fan data
    void recordPopClip()
    {
        auto& s = state();
        PopClipParams params;
        if (!s.pathClipStack.empty()) {
            auto& fan = s.pathClipStack.back();
            params.vertexCount = static_cast<uint32_t>(fan.reversedFanVerts.size());
            params.fanBounds = fan.fanBounds;
            renderer_.push(DrawOp::PopClip, s.zOrder, s.clipBounds,
                           s.stencilDepth, s.scopeDepth, params);
            if (!fan.reversedFanVerts.empty())
                renderer_.arena_pushSpan(std::span<const UIVertex>(
                    fan.reversedFanVerts.data(), fan.reversedFanVerts.size()));
            s.pathClipStack.pop_back();
        } else {
            renderer_.push(DrawOp::PopClip, s.zOrder, s.clipBounds,
                           s.stencilDepth, s.scopeDepth, params);
        }
        s.scopeDepth--;
        if (s.stencilDepth > 0) s.stencilDepth--;
    }

    // Flatten a path into a triangle fan for stencil rendering.
    // Adapted from the immediate-mode FillPath.h algorithm.
    static void flattenPathToFan(const juce::Path& path,
                                  const juce::AffineTransform& combined,
                                  std::vector<UIVertex>& fanVerts,
                                  juce::Rectangle<int>& bounds)
    {
        fanVerts.clear();

        float minX =  std::numeric_limits<float>::max(), minY = minX;
        float maxX = -std::numeric_limits<float>::max(), maxY = maxX;

        // Phase 1: collect transformed, flattened points with subpath markers
        std::vector<glm::vec2> points;
        const float SUBPATH_MARKER = -std::numeric_limits<float>::infinity();
        constexpr float flatTol = 0.5f;

        juce::Path::Iterator iter(path);
        glm::vec2 lastPt(0), subpathStart(0);

        auto transformPt = [&](float x, float y) -> glm::vec2 {
            combined.transformPoint(x, y);
            return { x, y };
        };

        auto flattenCubic = [&](glm::vec2 p0, glm::vec2 c1, glm::vec2 c2, glm::vec2 p3) {
            struct Seg { glm::vec2 a, b, c, d; };
            std::vector<Seg> stack;
            stack.push_back({ p0, c1, c2, p3 });
            while (!stack.empty()) {
                auto [a, b, c, d] = stack.back();
                stack.pop_back();
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
                    points.push_back(d);
                } else {
                    auto mid = [](glm::vec2 u, glm::vec2 v) { return (u+v)*0.5f; };
                    auto ab = mid(a,b), bc = mid(b,c), cd = mid(c,d);
                    auto abc = mid(ab,bc), bcd = mid(bc,cd), abcd = mid(abc,bcd);
                    stack.push_back({ abcd, bcd, cd, d });
                    stack.push_back({ a, ab, abc, abcd });
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
                    points.push_back({ SUBPATH_MARKER, 0 });
                    points.push_back(pt);
                    subpathStart = pt; lastPt = pt; break;
                }
                case juce::Path::Iterator::lineTo: {
                    auto pt = transformPt(iter.x1, iter.y1);
                    points.push_back(pt); lastPt = pt; break;
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
                    points.push_back(subpathStart);
                    lastPt = subpathStart; break;
            }
        }

        // Phase 2: build triangle fan from flattened points
        glm::vec4 dummy(0);
        glm::vec2 fanCenter(0), prevPt(0);
        bool inSubpath = false;

        auto updateBounds = [&](float x, float y) {
            minX = std::min(minX, x); minY = std::min(minY, y);
            maxX = std::max(maxX, x); maxY = std::max(maxY, y);
        };

        for (size_t i = 0; i < points.size(); i++) {
            if (points[i].x == SUBPATH_MARKER) {
                if (++i >= points.size()) break;
                fanCenter = points[i]; prevPt = fanCenter;
                inSubpath = true;
                updateBounds(fanCenter.x, fanCenter.y);
                continue;
            }
            if (!inSubpath) continue;
            glm::vec2 pt = points[i];
            updateBounds(pt.x, pt.y);
            fanVerts.push_back({ fanCenter, dummy, {}, dummy, dummy });
            fanVerts.push_back({ prevPt,   dummy, {}, dummy, dummy });
            fanVerts.push_back({ pt,        dummy, {}, dummy, dummy });
            prevPt = pt;
        }

        if (fanVerts.empty()) {
            bounds = {};
        } else {
            int bx = std::max(0, static_cast<int>(std::floor(minX)) - 1);
            int by = std::max(0, static_cast<int>(std::floor(minY)) - 1);
            int bx2 = static_cast<int>(std::ceil(maxX)) + 1;
            int by2 = static_cast<int>(std::ceil(maxY)) + 1;
            bounds = { bx, by, bx2 - bx, by2 - by };
        }
    }

    Renderer& renderer_;
    float     displayScale_;
    std::vector<RecordState> stateStack_;
    mutable uint64_t frameId_ = 0;
};

} // namespace jvk
