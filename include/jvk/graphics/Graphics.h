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
        state().transform = state().transform.translated(static_cast<float>(o.x), static_cast<float>(o.y));
    }

    void addTransform(const juce::AffineTransform& t) override
    {
        state().transform = state().transform.followedBy(t);
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
        auto logBounds = path.getBoundsTransformed(s.transform.followedBy(t)).getSmallestIntegerContainer();
        auto bounds = juce::Rectangle<int>(
            static_cast<int>(logBounds.getX() * displayScale_),
            static_cast<int>(logBounds.getY() * displayScale_),
            static_cast<int>(logBounds.getWidth() * displayScale_),
            static_cast<int>(logBounds.getHeight() * displayScale_));
        s.clipBounds = s.clipBounds.getIntersection(bounds);
        PushClipPathParams params;
        params.vertexCount = 0;
        params.pathBounds = bounds;
        renderer_.push(DrawOp::PushClipPath, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, params);
        s.stencilDepth++;
        s.scopeDepth++;
    }

    void clipToImageAlpha(const juce::Image&, const juce::AffineTransform&) override {}

    bool clipRegionIntersects(const juce::Rectangle<int>& r) override
    {
        return state().clipBounds.intersects(r);
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
            renderer_.push(DrawOp::PopClip, old.zOrder, old.clipBounds,
                           old.stencilDepth, old.scopeDepth, PopClipParams {});
            old.scopeDepth--;
            if (old.stencilDepth > prev.stencilDepth) old.stencilDepth--;
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
        auto& s = state();
        auto bounds = path.getBoundsTransformed(s.transform.followedBy(t));
        clipToPath(path, t);
        fillRect(bounds);
        renderer_.push(DrawOp::PopClip, s.zOrder, s.clipBounds,
                       s.stencilDepth, s.scopeDepth, PopClipParams {});
        s.scopeDepth--;
        if (s.stencilDepth > 0) s.stencilDepth--;
    }

    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !img.isValid()) return;
        auto& s = state();
        uint64_t hash = reinterpret_cast<uint64_t>(img.getPixelData().get());

        renderer_.caches().getTexture(hash, img);

        renderer_.push(DrawOp::DrawImage, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawImageParams { hash, s.transform.followedBy(t), s.opacity, displayScale_,
                              img.getWidth(), img.getHeight() });
    }

    void drawLine(const juce::Line<float>& line) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        renderer_.push(DrawOp::DrawLine, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            DrawLineParams { line, 1.0f, fi, s.transform, s.opacity, displayScale_ });
    }

    void setFont(const juce::Font& f) override { state().font = f; }
    const juce::Font& getFont() override { return state().font; }

    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                    juce::Span<const juce::Point<float>> positions,
                    const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || glyphs.empty()) return;
        auto& s = state();
        DrawGlyphsParams params;
        params.glyphCount = static_cast<uint32_t>(glyphs.size());
        params.transform = s.transform.followedBy(t);
        params.fontIndex = renderer_.captureFont(s.font);
        params.fillIndex = renderer_.captureFill(s.fill);
        params.opacity = s.opacity;
        params.scale = displayScale_;
        renderer_.push(DrawOp::DrawGlyphs, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth, params);
        // Append glyph POD data to arena
        renderer_.arena_pushSpan(std::span<const uint16_t>(glyphs.data(), glyphs.size()));
        renderer_.arena_pushSpan(std::span<const juce::Point<float>>(positions.data(), positions.size()));
    }

    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        renderer_.push(DrawOp::FillRoundedRect, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            FillRoundedRectParams { r, cornerSize, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        renderer_.push(DrawOp::FillEllipse, s.zOrder, s.clipBounds, s.stencilDepth, s.scopeDepth,
            FillEllipseParams { area, fi, s.transform, s.opacity, displayScale_ });
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
    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        float                 zOrder = 0.0f;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
    };

    RecordState& state() { return stateStack_.back(); }
    const RecordState& state() const { return stateStack_.back(); }

    Renderer& renderer_;
    float     displayScale_;
    std::vector<RecordState> stateStack_;
    mutable uint64_t frameId_ = 0;
};

} // namespace jvk
