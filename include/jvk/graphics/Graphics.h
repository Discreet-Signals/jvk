#pragma once

namespace jvk {

class Shader; // forward declaration

class Graphics : public juce::LowLevelGraphicsContext {
public:
    Graphics(Renderer& renderer, float displayScale)
        : renderer_(renderer), displayScale_(displayScale)
    {
        beginFrame(displayScale);
    }

    ~Graphics() override = default;

    // Reset the record state for a new frame WITHOUT freeing any scratch
    // capacity. The editor holds ONE Graphics for its lifetime and calls
    // this each tick — the old stack-constructed-per-frame object threw
    // away every scratch vector's capacity 60×/s, so the "allocation-free
    // after the first frame" property only held within a single frame.
    // frameId_ deliberately survives: juce::Graphics::getFrameId must be
    // monotonic across frames (JUCE keys per-frame caches on it).
    void beginFrame(float displayScale)
    {
        displayScale_ = displayScale;
        stateStack_.clear();
        stateStack_.push_back({});
        pathClipShared_.clear();
        auto& s = stateStack_.back();
        s.clipBounds = { 0, 0,
            static_cast<int>(renderer_.target().width()),
            static_cast<int>(renderer_.target().height()) };
        frameId_++;
    }

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
        auto& s = state();
        s.transform = juce::AffineTransform::translation(static_cast<float>(o.x),
                                                          static_cast<float>(o.y))
                          .followedBy(s.transform);
        s.inverseValid = false;
    }

    void addTransform(const juce::AffineTransform& t) override
    {
        auto& s = state();
        s.transform = t.followedBy(s.transform);
        s.inverseValid = false;
    }

    float getPhysicalPixelScaleFactor() const override { return displayScale_; }

    bool clipToRectangle(const juce::Rectangle<int>& r) override
    {
        auto& s = state();
        auto tf = r.toFloat().transformedBy(s.transform);
        // Scale logical→pixel for the Vulkan scissor. floor/ceil (outward),
        // NOT truncation: at fractional DPI (125%/150% Windows) truncating
        // x*scale and w*scale lost up to a pixel on the right/bottom edge —
        // and rounded toward zero, i.e. the WRONG way for negative origins.
        // Outward rounding never clips away content the caller asked to keep
        // (matches fillPath's bounds convention).
        const int x0 = static_cast<int>(std::floor(tf.getX()      * displayScale_));
        const int y0 = static_cast<int>(std::floor(tf.getY()      * displayScale_));
        const int x1 = static_cast<int>(std::ceil (tf.getRight()  * displayScale_));
        const int y1 = static_cast<int>(std::ceil (tf.getBottom() * displayScale_));
        s.clipBounds = s.clipBounds.getIntersection({ x0, y0, x1 - x0, y1 - y0 });
        struct Empty {};   // replay uses cmd.clipBounds only
        renderer_.push(DrawOp::PushClipRect, s.clipBounds, s.stencilDepth, Empty {});
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

        // Non-rectangular path — use the analytical-SDF clip stencil pipeline.
        // Flatten the path to line segments, upload to the shared segment
        // SSBO (same ring PathPipeline owns), emit a PushClipPath op. The
        // GPU then rasterises the cover quad at replay; surviving fragments
        // INCR the stencil, and subsequent draws pass only where stencil
        // == current depth.
        auto* pp = renderer_.pathPipeline();
        if (!pp) return; // no path pipeline wired — clip is a no-op

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        // Cover rect in physical pixels, expanded by 1 pixel so the clip
        // fragment shader runs for every pixel the path's SDF could mark
        // as inside.
        auto px = pathBounds.transformedBy(combined).expanded(1.0f);

        // Y-strip binned upload — clip is winding-only (hard edge), so a
        // small pad suffices.
        uint32_t stripCount; float stripMinY, invStripH;
        uint32_t segStart = binAndUploadSegments(pp, px.getY(), px.getBottom(),
                                                 2.0f, stripCount, stripMinY,
                                                 invStripH);
        int bx  = static_cast<int>(std::floor(px.getX()));
        int by  = static_cast<int>(std::floor(px.getY()));
        int bx2 = static_cast<int>(std::ceil(px.getRight()));
        int by2 = static_cast<int>(std::ceil(px.getBottom()));
        juce::Rectangle<int>  pathBoundsPx { bx, by, bx2 - bx, by2 - by };
        juce::Rectangle<float> coverRect { static_cast<float>(bx),
                                           static_cast<float>(by),
                                           static_cast<float>(bx2 - bx),
                                           static_cast<float>(by2 - by) };

        // Intersect clipBounds BEFORE incrementing stencilDepth — the push
        // op runs at the current (parent) depth; the pop in restoreState
        // will run against the new depth.
        auto outerBounds = s.clipBounds.getIntersection(pathBoundsPx);

        ClipShapeParams params {};
        params.shapeType    = 2u; // path
        params.segmentStart = segStart;
        params.segmentCount = static_cast<uint32_t>(scratchSegments_.size());
        params.fillRule     = path.isUsingNonZeroWinding() ? 0u : 1u;
        params.coverRect    = coverRect;
        params.stripCount   = stripCount;
        params.stripMinY    = stripMinY;
        params.invStripH    = invStripH;

        // Remember for the matching pop — same segment range + cover rect
        // so DECR cancels INCR exactly. Shared stack + per-state watermark:
        // the old per-RecordState vector was deep-copied on EVERY saveState
        // (JUCE brackets every component paint with one) — a heap alloc per
        // component per frame for data that is strictly stack-shaped.
        pathClipShared_.push_back(params);

        renderer_.push(DrawOp::PushClipPath, outerBounds, s.stencilDepth, params);

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
        // caller's local coord space. The inverse is cached on the state:
        // JUCE calls this once per CHILD, and a matrix inversion per call
        // added up (invalidated by setOrigin/addTransform).
        auto& s = state();
        if (s.clipBounds.isEmpty()) return {};
        if (!s.inverseValid) {
            s.pixelToLocal = s.transform.scaled(displayScale_).inverted();
            s.inverseValid = true;
        }
        auto local = s.clipBounds.toFloat().transformedBy(s.pixelToLocal);
        return local.getSmallestIntegerContainer();
    }
    bool isClipEmpty() const override { return state().clipBounds.isEmpty(); }

    void saveState() override { stateStack_.push_back(stateStack_.back()); }

    void restoreState() override
    {
        if (stateStack_.size() <= 1) {
            jassertfalse; // unbalanced restoreState — dropped silently before
            return;
        }
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
                renderer_.push(DrawOp::PopClipRect, old.clipBounds,
                               old.stencilDepth, Empty {});
                old.scopeDepth--;
            }
        }
        stateStack_.pop_back();
    }

    // Transparency layers are approximated by an alpha multiplier on every
    // draw inside the layer (true offscreen layers are the Phase 6 rebuild).
    // The multiplier lives in its own scope so it unwinds at end — it used
    // to be written into the CURRENT state with an empty end, leaking the
    // layer alpha into everything painted after the layer.
    void beginTransparencyLayer(float opacity) override
    {
        saveState();
        state().opacity *= opacity;
    }
    void endTransparencyLayer() override { restoreState(); }

    void setFill(const juce::FillType& fill) override { state().fill = fill; }

    // juce contract (FillType::setOpacity → colour.withAlpha): setOpacity
    // REPLACES the current fill colour's alpha in place. Modeling it as a
    // separate sticky multiplier broke the standard idiom
    //     g.setColour(c.withAlpha(0.06f)); ...; g.setOpacity(1.0f); g.drawImage(...)
    // — the image kept the stale 0.06 fill alpha (melatonin::DropShadow does
    // exactly this before compositing, which made pedal shadows vanish after
    // any glass-panel paint). state().opacity is the transparency-LAYER
    // multiplier only; plain setOpacity must never touch it.
    void setOpacity(float opacity) override { state().fill.setOpacity(opacity); }
    void setInterpolationQuality(juce::Graphics::ResamplingQuality) override {}

    void fillRect(const juce::Rectangle<int>& r, bool) override { fillRect(r.toFloat()); }

    void fillRect(const juce::Rectangle<float>& r) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        // Stage gradient LUT upload if this is a gradient fill
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillRect, s.clipBounds, s.stencilDepth,
            FillRectParams { r, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillRectList(const juce::RectangleList<float>& list) override
    {
        if (isClipEmpty() || list.isEmpty()) return;
        if (list.getNumRectangles() == 1) { fillRect(*list.begin()); return; }

        // ONE command + one arena span for the whole list. The old loop
        // exploded an N-rect list (common from JUCE text/clip internals)
        // into N commands + N FillType captures + N draw calls.
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        FillRectListParams p {};
        p.rectCount = static_cast<uint32_t>(list.getNumRectangles());
        p.fillIndex = fi;
        p.transform = s.transform;
        p.opacity   = s.opacity;
        p.scale     = displayScale_;
        renderer_.push(DrawOp::FillRectList, s.clipBounds, s.stencilDepth, p);
        renderer_.arena_align(4);
        // RectangleList's iterators walk a contiguous juce::Array — a span
        // over [begin, end) is valid.
        renderer_.arena_pushSpan(std::span<const juce::Rectangle<float>>(
            list.begin(), static_cast<size_t>(list.getNumRectangles())));
    }

    // Analytical-SDF path fill, TILE mode (piet-gpu style). CPU flattens
    // the path's curves to line segments in physical-pixel space, then
    // decomposes the (clip-restricted) bounds into tiles:
    //   - tiles the path never touches are NOT drawn;
    //   - interior tiles carry only a constant winding ("backdrop") and
    //     merge into single row-run quads — flat fill, zero segment work;
    //   - edge tiles carry a small tile-local segment list; the fragment
    //     shader reconstructs the EXACT winding as backdrop + local
    //     crossings (derivation in path_sdf.frag) and the exact SDF within
    //     the AA band.
    // This replaced (a) one bounds-covering quad whose every fragment
    // walked the whole segment list — quadratic on big paths and ~90%
    // wasted fragments on waveform-like content — and (b) the Y-strip
    // binning that fixed neither overdraw nor horizontal-heavy paths and
    // whose fwidth() AA streaked at strip boundaries.
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
        if (!pp) return;

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        // Segment-bbox pad: covers the 1px AA band (a segment farther than
        // this from a tile cannot influence any of its fragments' alpha).
        constexpr float pad = 2.0f;

        auto pxBounds = pathBounds.transformedBy(combined).expanded(pad);

        // The grid only needs to cover what's visible: winding at a point
        // depends solely on crossings of that point's own horizontal line
        // (accounted exactly via backdrop/overflow below), so restricting
        // the grid to clip ∩ bounds stays exact — off-clip tiles are never
        // rasterised anyway (scissor = clipBounds).
        int bx  = static_cast<int>(std::floor(pxBounds.getX()));
        int by  = static_cast<int>(std::floor(pxBounds.getY()));
        int bx2 = static_cast<int>(std::ceil(pxBounds.getRight()));
        int by2 = static_cast<int>(std::ceil(pxBounds.getBottom()));
        auto clipRect = s.clipBounds.getIntersection(
            juce::Rectangle<int> { bx, by, bx2 - bx, by2 - by });
        if (clipRect.isEmpty()) return;

        const float gx0 = static_cast<float>(clipRect.getX());
        const float gy0 = static_cast<float>(clipRect.getY());

        // Tile size: 16px nominal, doubled while the grid would exceed the
        // scratch budget (huge transforms / 4K-filling paths degrade toward
        // the old one-big-tile behaviour instead of exploding memory).
        float tileW = kPathTileSize;
        constexpr int kMaxTiles = 1 << 16;
        int nx = 0, ny = 0;
        for (;;) {
            nx = std::max(1, (int) std::ceil((float) clipRect.getWidth()  / tileW));
            ny = std::max(1, (int) std::ceil((float) clipRect.getHeight() / tileW));
            if ((int64_t) nx * ny <= kMaxTiles) break;
            tileW *= 2.0f;
        }
        const int nTiles = nx * ny;

        // --- Pass 1: per-tile local-list counts + horizontal-crossing
        // deltas. A segment lands in every tile its pad-expanded bbox
        // touches (winding needs only true overlap; pad keeps the SDF exact
        // through the AA band). Its crossings of each row's top line are
        // binned by x-column so pass 2 can turn them into per-tile backdrop
        // windings by suffix-summing right-to-left.
        scratchTileCounts_.assign((size_t) nTiles, 0u);
        scratchTileWind_.assign((size_t) nTiles, 0);
        scratchRowOverflow_.assign((size_t) ny, 0);

        const float gridRight  = gx0 + (float) nx * tileW;
        const float gridBottom = gy0 + (float) ny * tileW;
        const float invTileW   = 1.0f / tileW;

        auto tileRange = [&](float lo, float hi, int n, float g0) {
            int i0 = std::max(0,     (int) std::floor((lo - g0) * invTileW));
            int i1 = std::min(n - 1, (int) std::floor((hi - g0) * invTileW));
            return std::pair<int, int> { i0, i1 };
        };

        for (auto& seg : scratchSegments_) {
            const float xLo = std::min(seg.x, seg.z), xHi = std::max(seg.x, seg.z);
            const float yLo = std::min(seg.y, seg.w), yHi = std::max(seg.y, seg.w);

            // Local lists (pad-expanded bbox ∩ grid).
            if (xHi + pad >= gx0 && xLo - pad < gridRight
             && yHi + pad >= gy0 && yLo - pad < gridBottom) {
                auto [tx0, tx1] = tileRange(xLo - pad, xHi + pad, nx, gx0);
                auto [ty0, ty1] = tileRange(yLo - pad, yHi + pad, ny, gy0);
                for (int ty = ty0; ty <= ty1; ty++)
                    for (int tx = tx0; tx <= tx1; tx++)
                        scratchTileCounts_[(size_t)(ty * nx + tx)]++;
            }

            // Row-top crossings. Half-open in y (a.y > T) != (b.y > T),
            // matching the shader's ray convention exactly.
            int r0 = std::max(0, (int) std::ceil((yLo - gy0) * invTileW));
            int r1 = std::min(ny - 1, (int) std::floor((yHi - gy0) * invTileW));
            for (int r = r0; r <= r1; r++) {
                const float T = gy0 + (float) r * tileW;
                const bool aAbove = seg.y > T, bAbove = seg.w > T;
                if (aAbove == bAbove) continue;
                // Statement-split so clang can't contract to FMA — must
                // stay bit-identical to the shader's `precise` evaluation
                // (see path_sdf.frag's top term) or a crossing exactly on a
                // tile edge could be counted in both B and the local term.
                const float tt = (T - seg.y) / (seg.w - seg.y);
                const float dx = tt * (seg.z - seg.x);
                const float xc = seg.x + dx;
                const int sign = bAbove ? 1 : -1;
                if (xc <= gx0) continue;   // left of every tile's ray origin
                // Column bucket, half-open (colLeft, colRight] — pairs with
                // the shader's `xc <= tileRight` so B excludes exactly the
                // crossings the local top-term adds.
                const float fc = std::ceil((xc - gx0) * invTileW) - 1.0f;
                if (fc >= (float) nx)
                    scratchRowOverflow_[(size_t) r] += sign; // right of grid
                else
                    scratchTileWind_[(size_t)(r * nx + std::max(0, (int) fc))] += sign;
            }
        }

        // --- Pass 2: suffix-sum each row right-to-left; the array turns
        // from per-column deltas into B = winding at (tileRight, tileTop).
        for (int r = 0; r < ny; r++) {
            int acc = scratchRowOverflow_[(size_t) r];
            for (int c = nx - 1; c >= 0; c--) {
                const size_t idx = (size_t)(r * nx + c);
                const int h = scratchTileWind_[idx];
                scratchTileWind_[idx] = acc;
                acc += h;
            }
        }

        // --- Pass 3: place segments tile-major. Offsets are an exclusive
        // prefix over counts and double as write cursors; the emission loop
        // recovers each tile's start as cursor − count.
        scratchTileOffsets_.resize((size_t) nTiles);
        uint32_t total = 0;
        for (int i = 0; i < nTiles; i++) {
            scratchTileOffsets_[(size_t) i] = total;
            total += scratchTileCounts_[(size_t) i];
        }
        scratchBinned_.resize(total);
        for (auto& seg : scratchSegments_) {
            const float xLo = std::min(seg.x, seg.z), xHi = std::max(seg.x, seg.z);
            const float yLo = std::min(seg.y, seg.w), yHi = std::max(seg.y, seg.w);
            if (xHi + pad < gx0 || xLo - pad >= gridRight
             || yHi + pad < gy0 || yLo - pad >= gridBottom) continue;
            auto [tx0, tx1] = tileRange(xLo - pad, xHi + pad, nx, gx0);
            auto [ty0, ty1] = tileRange(yLo - pad, yHi + pad, ny, gy0);
            for (int ty = ty0; ty <= ty1; ty++)
                for (int tx = tx0; tx <= tx1; tx++)
                    scratchBinned_[scratchTileOffsets_[(size_t)(ty * nx + tx)]++] = seg;
        }
        const uint32_t segStart =
            (total > 0) ? pp->uploadSegments(scratchBinned_.data(), total) : 0u;

        // Register the gradient row up-front so the atlas upload happens in
        // this frame's staging pass. makeGradientCtx below also registers
        // idempotently — we mirror the fillRect convention here anyway.
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);

        // Per-vertex colour source, same as ColorDraw's primitive ops:
        // solid colour folded into the vertex attribute, or gradient
        // mode/row packed into gradientInfo via gradientAt().
        pipelines::GradientCtx grad = pipelines::makeGradientCtx(renderer_, s.fill, combined);
        glm::vec4 color = pipelines::fillColorAttr(grad, s.fill.colour, s.opacity);

        auto gi = [&](float px, float py) -> glm::vec4 {
            return grad.active() ? pipelines::gradientAt(grad, px, py)
                                 : glm::vec4(0.0f);
        };

        // --- Pass 4: emit quads. Edge tiles (count > 0) get their own
        // quad; consecutive INSIDE interior tiles with equal backdrop merge
        // into one wide quad (interior fragments only read the backdrop, so
        // uv/tile geometry doesn't matter there); outside tiles emit
        // nothing at all.
        const uint32_t fillRule = path.isUsingNonZeroWinding() ? 0u : 1u;
        scratchTileVerts_.clear();

        auto emitQuad = [&](float x, float y, float w,
                            uint32_t localStart, uint32_t localCount, int backdrop) {
            const glm::vec2 uv { x, y };
            const glm::vec4 shape { (float) localStart, (float) localCount,
                                    (float) backdrop, 0.0f };
            const float x2 = x + w, y2 = y + tileW;
            UIVertex v0 { { x,  y  }, color, uv, shape, gi(x,  y ) };
            UIVertex v1 { { x2, y  }, color, uv, shape, gi(x2, y ) };
            UIVertex v2 { { x2, y2 }, color, uv, shape, gi(x2, y2) };
            UIVertex v3 { { x,  y2 }, color, uv, shape, gi(x,  y2) };
            scratchTileVerts_.push_back(v0);
            scratchTileVerts_.push_back(v1);
            scratchTileVerts_.push_back(v2);
            scratchTileVerts_.push_back(v0);
            scratchTileVerts_.push_back(v2);
            scratchTileVerts_.push_back(v3);
        };

        for (int r = 0; r < ny; r++) {
            const float ty = gy0 + (float) r * tileW;
            int runStart = -1;
            int runB     = 0;
            auto flushRun = [&](int cEnd) {
                if (runStart < 0) return;
                emitQuad(gx0 + (float) runStart * tileW, ty,
                         (float)(cEnd - runStart) * tileW, 0u, 0u, runB);
                runStart = -1;
            };
            for (int c = 0; c < nx; c++) {
                const size_t idx  = (size_t)(r * nx + c);
                const uint32_t cnt = scratchTileCounts_[idx];
                const int B        = scratchTileWind_[idx];
                if (cnt == 0) {
                    const bool inside = (fillRule == 0u) ? (B != 0)
                                                         : ((B & 1) != 0);
                    if (!inside)            { flushRun(c); continue; }
                    if (runStart >= 0 && runB == B) continue;
                    flushRun(c);
                    runStart = c;
                    runB     = B;
                } else {
                    flushRun(c);
                    emitQuad(gx0 + (float) c * tileW, ty, tileW,
                             scratchTileOffsets_[idx] - cnt, cnt, B);
                }
            }
            flushRun(nx);
        }
        if (scratchTileVerts_.empty()) return;

        FillPathParams p {};
        p.vertexCount  = static_cast<uint32_t>(scratchTileVerts_.size());
        p.segmentStart = segStart;
        p.fillRule     = fillRule;
        p.fillIndex    = renderer_.captureFill(s.fill);
        p.tileSize     = tileW;

        renderer_.push(DrawOp::FillPath, clipRect, s.stencilDepth, p);
        renderer_.arena_align(4);
        renderer_.arena_pushSpan(std::span<const UIVertex>(
            scratchTileVerts_.data(), scratchTileVerts_.size()));
    }

    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !img.isValid()) return;
        auto& s = state();
        uint64_t hash = ResourceCaches::hashImage(img);

        // Resolve the texture now (message thread — safe around the shared
        // cache's inserts/evictions), pin the entry on this Renderer so it
        // survives sibling eviction, and capture the descriptor directly
        // into the draw command. The worker thread never touches the cache
        // map for this draw.
        auto desc = renderer_.caches().getTexture(hash, img, renderer_);
        if (desc == VK_NULL_HANDLE) return;   // staging/descriptor OOM

        // juce semantics: the current fill colour's ALPHA modulates image
        // draws (g.setColour(c.withAlpha(0.5f)); g.drawImage(...) renders at
        // 50%) — UNCONDITIONALLY: the software renderer reads
        // fillType.colour.getAlpha() for gradient/image fills too
        // (juce_RenderingHelpers.h renderImage). s.opacity is the
        // transparency-layer multiplier on top.
        const float alpha = s.opacity * s.fill.colour.getFloatAlpha();
        renderer_.push(DrawOp::DrawImage, s.clipBounds, s.stencilDepth,
            DrawImageParams { desc, t.followedBy(s.transform), alpha, displayScale_,
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
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::DrawLine, s.clipBounds, s.stencilDepth,
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
            renderer_.registerGradient(*s.fill.gradient);

        // Rasterize missing glyphs NOW, at record time on the message
        // thread. The worker's prepare pass (stageDirtyPages) then stages +
        // uploads the touched atlas rects BEFORE this frame's render pass,
        // so replay only ever samples initialized pages. The old flow
        // rasterized during REPLAY — after the upload flush had already run
        // — so a page created for a new glyph was bound and sampled in
        // UNDEFINED layout, and its pixels arrived one frame late.
        // (rasterizeBudget inside the atlas caps worst-case first-paint
        // hitches; over-budget glyphs simply appear next frame.)
        if (auto* cp = renderer_.colorPipeline())
        {
            auto& atlas = cp->atlas();
            // ONE key, glyphId mutated per iteration — constructing a key
            // per glyph paid an atomic Typeface::Ptr refcount pair per
            // glyph per frame (real cost on 10k+ glyph scenes).
            GlyphAtlas::GlyphKey key { s.font.getTypefacePtr(), 0,
                                       static_cast<int>(s.font.getMetricsKind()) };
            for (auto gid : glyphs)
            {
                key.glyphId = gid;
                atlas.getGlyph(key, s.font);
            }
        }

        DrawGlyphsParams params;
        params.glyphCount = static_cast<uint32_t>(glyphs.size());
        params.transform = t.followedBy(s.transform);
        params.fontIndex = renderer_.captureFont(s.font);
        params.fillIndex = renderer_.captureFill(s.fill);
        params.opacity = s.opacity;
        params.scale = displayScale_;
        renderer_.push(DrawOp::DrawGlyphs, s.clipBounds, s.stencilDepth, params);
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
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillRoundedRect, s.clipBounds, s.stencilDepth,
            FillRoundedRectParams { r, cornerSize, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::FillEllipse, s.clipBounds, s.stencilDepth,
            FillEllipseParams { area, fi, s.transform, s.opacity, displayScale_ });
    }

    void drawRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                              float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = renderer_.captureFill(s.fill);
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::StrokeRoundedRect, s.clipBounds, s.stencilDepth,
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
            renderer_.registerGradient(*s.fill.gradient);
        renderer_.push(DrawOp::StrokeEllipse, s.clipBounds, s.stencilDepth,
            StrokeEllipseParams { area, lineThickness, fi, s.transform, s.opacity, displayScale_ });
    }

    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const override
    {
        return std::make_unique<juce::SoftwareImageType>();
    }

    // Monotonic across frames (incremented by beginFrame): JUCE keys
    // per-frame caches on this, and the old per-frame Graphics object reset
    // it to zero every tick.
    uint64_t getFrameId() const override { return frameId_; }

    // ===== GPU Effects =====
    void darken(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        float v = 1.0f - amount;
        renderer_.push(DrawOp::EffectBlend, s.clipBounds, s.stencilDepth,
            EffectBlendParams { v, v, v, region, displayScale_ });
    }

    void brighten(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        float v = 1.0f + amount;
        renderer_.push(DrawOp::EffectBlend, s.clipBounds, s.stencilDepth,
            EffectBlendParams { v, v, v, region, displayScale_ });
    }

    void tint(juce::Colour c, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::EffectBlend, s.clipBounds, s.stencilDepth,
            EffectBlendParams { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), region, displayScale_ });
    }

    void warmth(float amount, juce::Rectangle<float> region = {})
    {
        if (region.isEmpty()) region = state().clipBounds.toFloat();
        auto& s = state();
        renderer_.push(DrawOp::EffectBlend, s.clipBounds, s.stencilDepth,
            EffectBlendParams { 1.0f + amount * 0.2f, 1.0f, 1.0f - amount * 0.1f, region, displayScale_ });
    }

    // Procedural white-noise overlay — GPU, pixel-perfect (a per-pixel hash of
    // the device coordinate; no texture, no tiling). `amount` blends toward
    // white noise: 0 = untouched, 1 = full overwrite, 0.1 = a fine grain. When
    // `staticHash` is true the pattern is identical every frame; otherwise it
    // animates. Honours the active clip (scissor + path stencil).
    void drawNoise(float amount, bool staticHash, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        // state().clipBounds is in PHYSICAL (root) pixels; an explicit region is
        // in the current logical space. Resolve both to physical and pass
        // scale = 1 so the pipeline doesn't scale again — otherwise an offset
        // region (e.g. a child component's clip) lands off-screen at HiDPI.
        const juce::Rectangle<float> physRegion = region.isEmpty()
            ? s.clipBounds.toFloat()
            : region.transformedBy(s.transform.scaled(displayScale_));
        const float timeOffset = staticHash
            ? 0.0f
            : static_cast<float>(juce::Time::getMillisecondCounter() & 0xFFFFu);
        renderer_.push(DrawOp::EffectNoise, s.clipBounds, s.stencilDepth,
            NoiseParams { amount, timeOffset, physRegion, 1.0f });
    }

    void blur(float radius, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        // Default region = current clip bounds (already physical). An
        // explicit region is in the caller's LOGICAL space — resolve to
        // physical (transform + displayScale), same convention as drawNoise.
        // The replay-side ROI walk scissors every pass of the blur chain to
        // this rect (padded per pass by the kernel's read margin).
        const juce::Rectangle<float> physRegion = region.isEmpty()
            ? s.clipBounds.toFloat()
            : region.transformedBy(s.transform.scaled(displayScale_));
        renderer_.push(DrawOp::EffectKernel, s.clipBounds, s.stencilDepth,
            BlurParams { radius, physRegion, displayScale_ });
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
        auto& s = state();
        // Same region convention as blur() — default = clip bounds
        // (physical), explicit = logical resolved to physical here.
        const juce::Rectangle<float> physRegion = region.isEmpty()
            ? s.clipBounds.toFloat()
            : region.transformedBy(s.transform.scaled(displayScale_));
        renderer_.push(DrawOp::EffectHSV, s.clipBounds, s.stencilDepth,
            HSVParams { scaleH, scaleS, scaleV, deltaH, deltaS, deltaV,
                        physRegion, displayScale_ });
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

    // Path variants — route through PathBlurPipeline (ping-pong effect pass
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
        renderer_.push(DrawOp::DrawShader, s.clipBounds, s.stencilDepth,
            DrawShaderParams { &shader, regionPx, displayScale_ });
    }

    Renderer& getRenderer() { return renderer_; }

private:
    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        // Transparency-LAYER alpha only (beginTransparencyLayer multiplies it
        // in its own saved scope). Plain setOpacity follows the juce contract
        // instead: it rewrites fill.colour's alpha (FillType::setOpacity).
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
        // Cached inverse of transform.scaled(displayScale) for getClipBounds
        // (JUCE calls it per child). Copied validly on saveState (same
        // transform); invalidated by setOrigin/addTransform.
        mutable juce::AffineTransform pixelToLocal;
        mutable bool                  inverseValid = false;
        // Path clips live in Graphics::pathClipShared_ (stack-shaped across
        // states); no per-state vector to deep-copy on every saveState.
    };

    RecordState& state() { return stateStack_.back(); }
    const RecordState& state() const { return stateStack_.back(); }

    // Record a PopClip — same ClipShapeParams as the paired PushClipPath so
    // the DECR_WRAP at the GPU matches the INCR_WRAP exactly. Stencil
    // reference at replay = stencilDepth BEFORE the pop (current depth),
    // so the hardware decrements only the pixels the push incremented.
    void recordPopClip()
    {
        auto& s = state();
        if (!pathClipShared_.empty()) {
            ClipShapeParams params = pathClipShared_.back();
            renderer_.push(DrawOp::PopClipPath, s.clipBounds, s.stencilDepth, params);
            pathClipShared_.pop_back();
        }
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

    // Pack a BlurShape draw command. Handles the inverse-affine computation
    // that maps physical fragment coords back into shape-local logical space.
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

        // blurStep = physical texels per user-logical pixel. Folds in both
        // the user's transform scale AND displayScale so the shader's kernel
        // step converts 1 user-logical pixel to the right number of physical
        // texels — which means the blur respects addTransform(scale(...))
        // AND the loop count stays fixed regardless of displayScale (2x
        // retina just samples every other physical pixel, same tap count).
        const float transformScale = s.transform.getScaleFactor();

        p.maxRadius     = blurRadius;
        p.falloff       = juce::jmax(0.001f, falloffRadius);
        p.blurStep      = transformScale * displayScale_;
        p.cornerRadius  = cornerSize;
        p.lineThickness = lineThickness;

        p.shapeType     = shapeType;
        p.edgePlacement = static_cast<uint32_t>(edge);
        p.inverted      = inverted ? 1u : 0u;
        p.mode          = static_cast<uint32_t>(mode);

        // ROI: every pixel this blur can touch — the shape's bounds expanded
        // by the blur reach (+stroke ring), in LOGICAL space, then mapped to
        // physical. For lines the anchor rect is degenerate at A, so fold in
        // endpoint B first. Inverted blurs touch everything OUTSIDE the
        // shape: leave the region empty = replay uses the full clip.
        if (!inverted) {
            auto reach = blurRadius + falloffRadius + lineThickness + 2.0f;
            juce::Rectangle<float> logical = boundsRect;
            if (shapeType == 3)
                logical = juce::Rectangle<float>::leftTopRightBottom(
                    juce::jmin(boundsRect.getX(), boundsRect.getX() + lineB.x),
                    juce::jmin(boundsRect.getY(), boundsRect.getY() + lineB.y),
                    juce::jmax(boundsRect.getX(), boundsRect.getX() + lineB.x),
                    juce::jmax(boundsRect.getY(), boundsRect.getY() + lineB.y));
            p.region = logical.expanded(reach)
                           .transformedBy(s.transform.scaled(displayScale_));
        }

        renderer_.push(DrawOp::BlurShape, s.clipBounds, s.stencilDepth, p);
    }

    // Pack a BlurPath draw command. Flattens the path to line segments
    // (physical pixels), uploads them to PathPipeline's per-frame SSBO,
    // then emits a BlurPath op carrying the segment range + blur params
    // (all distances in physical pixels — the caller's logical units
    // were multiplied by displayScale here so the shader stays in one
    // coord space).
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
        if (!pp) return;

        auto& s = state();
        auto pathBounds = path.getBounds();
        if (pathBounds.isEmpty()) return;

        auto combined = t.followedBy(s.transform).scaled(displayScale_);

        scratchSegments_.clear();
        flattenPathToSegments(path, combined, scratchSegments_);
        if (scratchSegments_.empty()) return;

        // Pre-multiply radius params to physical pixels so the shader runs
        // entirely in one coord space (segments are already physical too).
        // `blurStep` folds both the user's transform scale and displayScale,
        // so the on-screen blur respects addTransform(scale(...)) as well
        // as retina.
        const float blurStep = s.transform.getScaleFactor() * displayScale_;

        BlurPathParams p {};
        p.segmentCount    = static_cast<uint32_t>(scratchSegments_.size());
        p.fillRule        = path.isUsingNonZeroWinding() ? 0u : 1u;
        p.maxRadius       = blurRadius       * blurStep;
        p.falloff         = juce::jmax(0.001f, falloffRadius * blurStep);
        p.strokeHalfWidth = strokeHalfWidth  * blurStep;
        p.edgePlacement   = static_cast<uint32_t>(edge);
        p.inverted        = inverted ? 1u : 0u;
        p.mode            = static_cast<uint32_t>(mode);

        // Y-strip binned upload. The pad must cover the blur's FULL reach:
        // the falloff curve needs true distances out to
        // maxRadius + falloff + strokeHalfWidth; beyond that a segment
        // cannot influence the result. Inverted blurs sample everywhere,
        // but their SDF is exact under the same pad (distance beyond reach
        // saturates the band either way).
        const float reach = p.maxRadius + p.falloff + p.strokeHalfWidth + 2.0f;
        const auto  pxB   = pathBounds.transformedBy(combined).expanded(reach);
        p.segmentStart = binAndUploadSegments(pp, pxB.getY(), pxB.getBottom(),
                                              reach, p.stripCount,
                                              p.stripMinY, p.invStripH);

        // ROI: path bounds (already transformed to physical by `combined`)
        // expanded by the blur reach + stroke ring, all physical px. Inverted
        // blurs leave it empty = full clip at replay.
        if (!inverted)
            p.region = pathBounds.transformedBy(combined)
                           .expanded(p.maxRadius + p.falloff + p.strokeHalfWidth + 2.0f);

        renderer_.push(DrawOp::BlurPath, s.clipBounds, s.stencilDepth, p);
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
        //
        // Every subpath is IMPLICITLY CLOSED (a closing segment back to its
        // first point when the author didn't closePath) — juce fills treat
        // open subpaths as closed, and winding is only well-defined over
        // closed contours: the tile decomposition in fillPath (and the ray
        // counts in clip/blur) assume the segment set forms closed curves.
        glm::vec2 prev(0), chainStart(0); bool havePrev = false;
        auto closeChain = [&] {
            if (havePrev && prev != chainStart)
                segs.push_back({ prev.x, prev.y, chainStart.x, chainStart.y });
            havePrev = false;
        };
        for (size_t i = 0; i < scratchPoints_.size(); ++i) {
            if (scratchPoints_[i].x == SUBPATH_MARKER) {
                closeChain();
                continue;
            }
            if (!havePrev) {
                prev = scratchPoints_[i];
                chainStart = prev;
                havePrev = true;
                continue;
            }
            glm::vec2 cur = scratchPoints_[i];
            if (cur != prev)
                segs.push_back({ prev.x, prev.y, cur.x, cur.y });
            prev = cur;
        }
        closeChain();
    }


    Renderer& renderer_;
    float     displayScale_;
    std::vector<RecordState> stateStack_;
    uint64_t frameId_ = 0;

    // Active path-clip params, shared across the state stack (strictly
    // stack-shaped: pushes in clipToPath, pops in recordPopClip; saveState
    // copies nothing).
    std::vector<ClipShapeParams> pathClipShared_;

    // Flatten scratch buffers — cleared-but-not-freed between paths AND
    // between frames (the editor reuses one Graphics via beginFrame), so
    // the hot path is genuinely allocation-free after warmup. Used by both
    // Graphics::fillPath (segments → PathPipeline SSBO) and clipToPath
    // (segments → ClipPipeline via the same shared SSBO).
    struct Seg { glm::vec2 a, b, c, d; };
    std::vector<glm::vec2> scratchPoints_;
    std::vector<Seg>       scratchSegStack_;
    std::vector<glm::vec4> scratchSegments_;
    std::vector<glm::vec4> scratchBinned_;
    std::vector<uint32_t>  scratchStripCounts_;

    // Tile-mode fill scratch (Graphics::fillPath). kPathTileSize must match
    // nothing on the GPU side — the actual tile width travels in the push
    // constants (it doubles when a huge path would blow the tile budget).
    static constexpr float kPathTileSize = 16.0f;
    std::vector<uint32_t>  scratchTileCounts_;
    std::vector<uint32_t>  scratchTileOffsets_;
    std::vector<int32_t>   scratchTileWind_;
    std::vector<int32_t>   scratchRowOverflow_;
    std::vector<UIVertex>  scratchTileVerts_;

    // ------------------------------------------------------------------
    // Y-strip binning — used by clipToPath (clip.frag) and pushBlurPath
    // (path_blur.frag); FILLS use the exact tile decomposition in fillPath
    // instead. The path fragment shaders were O(pixels × segments): every
    // fragment of the cover quad walked EVERY segment for winding +
    // distance. Binning reorders the upload as
    //   [strip table: stripCount vec4 entries][segments, strip-major]
    // where a segment is duplicated into every strip its y-range (± pad)
    // overlaps. A fragment then walks only its own strip:
    //   - winding stays EXACT — a segment whose y-range straddles frag.y
    //     necessarily overlaps frag's strip;
    //   - distance stays exact within `pad` — a segment farther than pad
    //     in y can't be nearer than pad, and pad covers the full reach
    //     (1px AA for fills, the blur band for path blurs).
    // A 2000-segment path over a 400px quad drops from 2000 to ~2000/25
    // segment evaluations per fragment. Table entries are vec4s in the
    // SAME SSBO (entry.x = offset after the table, entry.y = count), so
    // no descriptor/layout changes anywhere.
    //
    // Fills scratchBinned_ from scratchSegments_ and uploads it. Returns
    // the absolute SSBO index of the strip table, and the binning params
    // for the push constants. Returns stripCount 0 (flat upload) when the
    // path is too small to benefit.
    uint32_t binAndUploadSegments(PathPipeline* pp,
                                  float minYf, float maxYf, float pad,
                                  uint32_t& outStripCount,
                                  float& outStripMinY, float& outInvStripH)
    {
        const auto segCount = static_cast<uint32_t>(scratchSegments_.size());

        // Small paths: the per-fragment strip indirection costs more than
        // it saves. Upload flat.
        constexpr uint32_t kMinSegsToBin = 24;
        const float height = maxYf - minYf;
        if (segCount < kMinSegsToBin || height <= 1.0f) {
            outStripCount = 0;
            outStripMinY  = 0.0f;
            outInvStripH  = 0.0f;
            return pp->uploadSegments(scratchSegments_.data(), segCount);
        }

        // Strip height ≥ pad keeps duplication bounded (~≤3 strips/segment);
        // 16px floor keeps the table small for tall thin paths.
        const float stripH = std::max(16.0f, pad + 1.0f);
        const uint32_t stripCount = static_cast<uint32_t>(
            juce::jlimit(1, 512, (int) std::ceil(height / stripH)));

        auto stripFor = [&](float y) -> int {
            return juce::jlimit(0, (int) stripCount - 1,
                                (int) std::floor((y - minYf) / stripH));
        };

        scratchStripCounts_.assign(stripCount, 0u);
        for (auto& s : scratchSegments_) {
            const int s0 = stripFor(std::min(s.y, s.w) - pad);
            const int s1 = stripFor(std::max(s.y, s.w) + pad);
            for (int i = s0; i <= s1; i++) scratchStripCounts_[(size_t) i]++;
        }

        uint32_t total = 0;
        scratchBinned_.resize(stripCount);   // table first
        for (uint32_t i = 0; i < stripCount; i++) {
            scratchBinned_[i] = { (float) total, (float) scratchStripCounts_[i], 0.0f, 0.0f };
            total += scratchStripCounts_[i];
        }
        scratchBinned_.resize(stripCount + total);

        // Second pass: place segments strip-major, reusing the counts
        // vector as per-strip write cursors.
        for (uint32_t i = 0; i < stripCount; i++)
            scratchStripCounts_[i] = static_cast<uint32_t>(scratchBinned_[i].x);
        for (auto& s : scratchSegments_) {
            const int s0 = stripFor(std::min(s.y, s.w) - pad);
            const int s1 = stripFor(std::max(s.y, s.w) + pad);
            for (int i = s0; i <= s1; i++)
                scratchBinned_[stripCount + scratchStripCounts_[(size_t) i]++] = s;
        }

        outStripCount = stripCount;
        outStripMinY  = minYf;
        outInvStripH  = 1.0f / stripH;
        return pp->uploadSegments(scratchBinned_.data(),
                                  static_cast<uint32_t>(scratchBinned_.size()));
    }
};

} // namespace jvk
