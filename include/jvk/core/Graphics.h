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
        stencilClips_.clear();
        auto& s = stateStack_.back();
        s.clipBounds = { 0, 0,
            static_cast<int>(renderer_.extent().width),
            static_cast<int>(renderer_.extent().height) };
        syncTarget();   // the main colour
        values_.removeAllChildren(nullptr);   // every frame starts with no values
        values_.removeAllProperties(nullptr);
        frameId_++;
    }

    // Extract a jvk::Graphics* from a juce::Graphics (for effects API)
    static Graphics* create(juce::Graphics& g)
    {
        auto* ctx = dynamic_cast<Graphics*>(&g.getInternalContext());
        return ctx;
    }

    // ===== Frame values ======================================================
    //
    // A tree of values painters leave for the painters after them in the same
    // frame (paint order is the only order: a reader sees what painted before
    // it). Emptied at the start of every frame. Set a property, or append a
    // child of plain properties, so several painters can each add one of a
    // kind; a reader takes what is there, with its own defaults for what isn't.
    // A position is best stored in framebuffer pixels (getPhysicalTransform),
    // where every painter's values meet.
    juce::ValueTree& values() { return values_; }

    // The same, from a juce::Graphics: null when jvk isn't painting it.
    static juce::ValueTree* values(juce::Graphics& g)
    {
        auto* ctx = create(g);
        return ctx != nullptr ? &ctx->values_ : nullptr;
    }

    // The transform from the current paint's local coordinates to physical
    // framebuffer pixels: every origin and AffineTransform above it, times the
    // display scale. Where a shader that works in framebuffer pixels
    // (gl_FragCoord) needs to know where local things land.
    juce::AffineTransform getPhysicalTransform() const { return state().transform.scaled(displayScale_); }

    // ===== Draw target (render targets) =====================================
    //
    // Where this context's draws land: the main colour (the default; "main" or
    // a null Identifier) or a render target declared with
    // AudioProcessorEditor::setTargets. Every built-in draw — fills, paths,
    // text, images — goes to the target exactly as it would to the main
    // colour, so an app can draw an image, then select another target and
    // draw a matching image, shape or text into it. The
    // target is part of the saved state: saveState/restoreState (and so
    // juce::Graphics::ScopedSaveState and every component boundary) restore
    // it, so a component that selects a target never affects the next one.
    //
    // A target the frame doesn't have (one that was never declared, or
    // was dropped) records NO draws until the target changes, so they never
    // land in the main colour by mistake. Effects that run as passes (blur,
    // the blurred shapes and paths, saturate and every hsv transform) work on
    // whatever the target is, a scaled one included, in its own pixels and
    // within the clip's bounds (a target has no stencil); noise and the blend
    // modes work on the main colour only. An hsv transform expects a DISPLAY
    // colour (0..1, sRGB-encoded: it decodes, clamps lightness and re-encodes),
    // so on a linear HDR target it crushes and double-encodes; blur on any
    // format. A SCALED target (Target::scale) takes no ordinary draws: it
    // isn't attached to the scene pass. jvk::Shader draws write the locations
    // their shader declares, whatever the target.
    // Returns false if `name` isn't a target here.
    static inline const juce::Identifier mainTarget { "main" };

    bool setTarget(const juce::Identifier& name)
    {
        auto& s = state();
        if (name.isNull() || name == mainTarget) {
            s.target = 0;
        } else {
            const int i = renderer_.targetIndex(name);
            s.target = i < 0 ? Renderer::kNoTarget : static_cast<uint8_t>(i + 1);
        }
        s.targetName = name;
        syncTarget();
        return s.target != Renderer::kNoTarget;
    }
    juce::Identifier getTarget() const { return state().target == 0 ? mainTarget : state().targetName; }

    // The same, from a juce::Graphics: false when jvk isn't painting it (a
    // software or OpenGL paint) or the target doesn't exist.
    static bool setTarget(juce::Graphics& g, const juce::Identifier& name)
    {
        auto* ctx = create(g);
        return ctx != nullptr && ctx->setTarget(name);
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

    // Logical to physical, the transform included (juce: the root of the
    // transform's determinant): what sizes a buffered or effect-filtered
    // component's image, so one under Component::setTransform(scale) renders
    // at its drawn size rather than at 1x upscaled.
    float getPhysicalPixelScaleFactor() const override
    {
        return displayScale_ * std::sqrt(std::abs(state().transform.getDeterminant()));
    }

    // A transform that rotates or shears (juce's TranslationOrTransform::isRotated):
    // rectangles through it are no longer axis-aligned, so juce takes them as paths.
    static bool isRotated(const juce::AffineTransform& t)
    {
        return t.mat01 != 0.0f || t.mat10 != 0.0f;
    }

    bool clipToRectangle(const juce::Rectangle<int>& r) override
    {
        materializeMask();
        auto& s = state();
        if (isRotated(s.transform)) {
            // As juce: a rotated rectangle clips as a path, not as its bounding box.
            juce::Path p;
            p.addRectangle(r.toFloat());
            clipToPath(p, {});
            return !state().clipBounds.isEmpty();
        }
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
        if (list.getNumRectangles() <= 1)
            return clipToRectangle(list.getBounds());
        // The exact union, as juce (a TabbedComponent's outline is the content
        // minus its page): the rectangles as one path into the stencil. The
        // bounding box let the whole content area paint.
        clipToPath(list.toPath(), {});
        return !state().clipBounds.isEmpty();
    }

    // Excluded rects are stencil-PARKED: their drawable pixels (== current
    // depth) are rewritten to 192+depth, so every draw's EQUAL(depth) test
    // fails inside them, and restoreState un-parks (see ClipPipeline's
    // exclusion variant for the exact protocol and why it survives nested
    // clip push/pop in any order). Rect is taken through the current
    // transform's bounding box — same axis-aligned treatment as rect clips.
    // Known edge: two overlapping exclusions at the SAME clip depth in
    // nested scopes un-park each other's overlap early (192+d collides);
    // juce call sites (viewport repaint carving, melatonin) never do this.
    void excludeClipRectangle(const juce::Rectangle<int>& r) override
    {
        auto& s = state();
        if (isClipEmpty()) return;
        materializeMask();
        if (isRotated(s.transform)) {
            // As juce: the clip's bounds minus the rotated rectangle, as an
            // even-odd path clip (it unwinds with the state like the park would).
            juce::Path p;
            p.addRectangle(r.toFloat());
            p.addRectangle(getClipBounds().toFloat());
            p.setUsingNonZeroWinding(false);
            clipToPath(p, {});
            return;
        }
        auto phys = r.toFloat().transformedBy(s.transform.scaled(displayScale_));
        auto clip = s.clipBounds.getIntersection(phys.getSmallestIntegerContainer());
        if (clip.isEmpty()) return;      // outside the clip — nothing to exclude
        jassert(s.stencilDepth < 64);    // park encoding limit (ref = 192+d)
        exclusionShared_.push_back({ phys, s.stencilDepth });
        s.exclusions++;
        renderer_.push(DrawOp::ExcludeClipRect, clip, s.stencilDepth,
                       ExcludeRectParams { phys });
        // clipBounds stays the (conservative) enclosing rect — juce allows
        // getClipBounds to over-report; draws are culled by the stencil.
    }

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
        materializeMask();
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
        // Flatten the path to line segments, append them to the frame's
        // segment storage, emit a PushClipPath op. The GPU then rasterises
        // the cover quad; surviving fragments INCR the stencil, and
        // subsequent draws pass only where stencil == current depth.
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
        uint32_t segStart = binAndUploadSegments(px.getY(), px.getBottom(),
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
        stencilClips_.push_back({ params, outerBounds, s.scopeDepth });

        renderer_.push(DrawOp::PushClipPath, outerBounds, s.stencilDepth, params);

        s.clipBounds = outerBounds;
        s.stencilDepth++;
        s.scopeDepth++;
    }

    // Alpha-mask clip. What juce funnels through here is mostly one idiom —
    // drawImage*(img, ..., true) = save / clipToImageAlpha / fillAll /
    // restore — and THAT renders exactly: the fill is one quad tinted with
    // the brush and gated per pixel by the mask's alpha (shader shape type
    // 5). Anything else drawn under the mask is clipped by a STENCIL clip
    // of the mask (shape 3, clip_mask.frag: in where its alpha is at least
    // one half — hard edges, like every stencil clip), which enters the
    // stencil only when such a draw (or another clip) arrives — so the
    // idiom never pays for it (materializeMask). Like juce, the clip uses
    // the transform and resampling quality current when it is made, scopes
    // with save/restore, and intersects every other clip, masks included.
    void clipToImageAlpha(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (!img.isValid()) return;
        materializeMask();   // a mask still pending clips this one: both apply
        auto& s = state();
        s.alphaMask          = img;
        s.alphaMaskTransform = t.followedBy(s.transform);
        s.maskNearest        = s.nearest;
        s.maskPending        = true;
        s.maskOwner          = static_cast<uint32_t>(stateStack_.size() - 1);
        // Nothing outside the image can pass an alpha clip: fold its
        // transformed bounds into the rect clip.
        auto phys = img.getBounds().toFloat()
                        .transformedBy(s.alphaMaskTransform.scaled(displayScale_))
                        .getSmallestIntegerContainer();
        s.clipBounds = s.clipBounds.getIntersection(phys);
    }

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
        // Un-park this scope's exclusions FIRST (LIFO): each ref is the
        // depth recorded at exclusion time, so ordering vs the clip pops
        // below is actually immaterial (the park protocol is depth-tagged),
        // but LIFO keeps the stencil timeline easy to reason about.
        while (old.exclusions > prev.exclusions && !exclusionShared_.empty()) {
            auto e = exclusionShared_.back();
            exclusionShared_.pop_back();
            renderer_.push(DrawOp::RestoreClipExclude,
                           e.rect.getSmallestIntegerContainer(),
                           e.depth, ExcludeRectParams { e.rect });
            old.exclusions--;
        }
        // Pop this scope's clips in reverse order of their pushes: a stencil
        // clip is the latest when it was pushed at the scope depth just
        // below the current one; any other push was a rect clip.
        while (old.scopeDepth > prev.scopeDepth) {
            if (!stencilClips_.empty() && stencilClips_.back().scope == old.scopeDepth - 1) {
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
        syncTarget();   // the draw target unwinds with the rest of the state
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
    // juce's LOW quality is point sampling; medium and high are both smooth.
    // Every built-in texture binding samples LINEAR, so nearest is carried
    // per DRAW instead: the flag rides the image
    // quad and ui2d.frag snaps the UV to the texel centre, which a linear
    // sampler resolves to that exact texel (single mip level, no LOD blend).
    // High renders as bilinear — there is no mip chain for it to refine.
    void setInterpolationQuality(juce::Graphics::ResamplingQuality q) override
    {
        state().nearest = (q == juce::Graphics::lowResamplingQuality);
    }

    void fillRect(const juce::Rectangle<int>& r, bool) override { fillRect(r.toFloat()); }

    void fillRect(const juce::Rectangle<float>& r) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        // Solid fill under an active clipToImageAlpha mask → the juce
        // fillAlphaChannelWithCurrentBrush idiom. Render it exactly as one
        // brush-tinted, alpha-gated image quad.
        if (s.alphaMask.isValid() && (s.fill.isColour() || s.fill.isGradient()))
        {
            fillThroughAlphaMask(r);
            return;
        }
        // juce setTiledImageFill: render as a wrapped-UV image quad (shape
        // type 6). Without this the colour pipelines fell back to the fill's
        // colour member — solid black.
        if (s.fill.isTiledImage() && s.fill.image.isValid())
        {
            fillWithTiledImage(r);
            return;
        }
        if (isRotated(s.transform))
        {
            // As juce (fillRectAsPath): a rotated rectangle is a path, with a
            // path's anti-aliased edges, not a quad with jagged ones.
            juce::Path p;
            p.addRectangle(r);
            fillPath(p, {});
            return;
        }
        auto fi = captureFill();
        // Stage gradient LUT upload if this is a gradient fill
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::FillRect, s.clipBounds,
            FillRectParams { r, fi, s.transform, s.opacity, displayScale_ });
    }

    void fillRectList(const juce::RectangleList<float>& list) override
    {
        if (isClipEmpty() || list.isEmpty()) return;
        if (list.getNumRectangles() == 1) { fillRect(*list.begin()); return; }

        auto& s = state();
        if (isRotated(s.transform)) { fillPath(list.toPath(), {}); return; }   // as fillRect

        // ONE command + one arena span for the whole list. The old loop
        // exploded an N-rect list (common from JUCE text/clip internals)
        // into N commands + N FillType captures + N draw calls.
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        FillRectListParams p {};
        p.rectCount = static_cast<uint32_t>(list.getNumRectangles());
        p.fillIndex = fi;
        p.transform = s.transform;
        p.opacity   = s.opacity;
        p.scale     = displayScale_;
        record(DrawOp::FillRectList, s.clipBounds, p);
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
        // windings by suffix-summing right-to-left, and each one is OWNED by
        // the tile whose top edge it lands on.
        scratchTileCounts_.assign((size_t) nTiles, 0u);
        scratchTileOwners_.assign((size_t) nTiles, 0u);
        scratchTileWind_.assign((size_t) nTiles, 0);
        scratchRowOverflow_.assign((size_t) ny, 0);
        scratchOwners_.clear();
        scratchCrossings_.clear();

        const float gridRight  = gx0 + (float) nx * tileW;
        const float gridBottom = gy0 + (float) ny * tileW;
        const float invTileW   = 1.0f / tileW;

        auto tileRange = [&](float lo, float hi, int n, float g0) {
            int i0 = std::max(0,     (int) std::floor((lo - g0) * invTileW));
            int i1 = std::min(n - 1, (int) std::floor((hi - g0) * invTileW));
            return std::pair<int, int> { i0, i1 };
        };

        // The tiles whose local list a segment joins (pad-expanded bbox ∩
        // grid), or false for none. Passes 1 and 3 both ask HERE, so they
        // cannot disagree about a single tile.
        struct TileSpan { int x0, x1, y0, y1; };
        auto localTiles = [&](const glm::vec4& seg, TileSpan& span) {
            const float xLo = std::min(seg.x, seg.z), xHi = std::max(seg.x, seg.z);
            const float yLo = std::min(seg.y, seg.w), yHi = std::max(seg.y, seg.w);
            if (!(xHi + pad >= gx0 && xLo - pad < gridRight
               && yHi + pad >= gy0 && yLo - pad < gridBottom))
                return false;
            const auto [x0, x1] = tileRange(xLo - pad, xHi + pad, nx, gx0);
            const auto [y0, y1] = tileRange(yLo - pad, yHi + pad, ny, gy0);
            span = { x0, x1, y0, y1 };
            return true;
        };

        for (uint32_t si = 0; si < (uint32_t) scratchSegments_.size(); ++si) {
            const auto& seg = scratchSegments_[si];
            const float yLo = std::min(seg.y, seg.w), yHi = std::max(seg.y, seg.w);

            // Local lists.
            TileSpan span { 0, -1, 0, -1 };
            if (localTiles(seg, span))
                for (int ty = span.y0; ty <= span.y1; ty++)
                    for (int tx = span.x0; tx <= span.x1; tx++)
                        scratchTileCounts_[(size_t)(ty * nx + tx)]++;

            // Row-top crossings. Half-open in y (a.y > T) != (b.y > T),
            // matching the shader's ray convention exactly.
            int r0 = std::max(0, (int) std::ceil((yLo - gy0) * invTileW));
            int r1 = std::min(ny - 1, (int) std::floor((yHi - gy0) * invTileW));
            for (int r = r0; r <= r1; r++) {
                const float T = gy0 + (float) r * tileW;
                const bool aAbove = seg.y > T, bAbove = seg.w > T;
                if (aAbove == bAbove) continue;
                // Only this side computes xc: wherever rounding puts a
                // crossing on a tile edge, it is in exactly one of B and its
                // owner's list, and the shader adds it from the list. The
                // shader never re-derives it either — every list entry of
                // this segment in row r carries this xc (pass 3), and both
                // halves of path_sdf.frag's corner split compare against it.
                const float tt = (T - seg.y) / (seg.w - seg.y);
                const float dx = tt * (seg.z - seg.x);
                const float xc = seg.x + dx;
                const int sign = bAbove ? 1 : -1;
                scratchCrossings_.push_back({ si, r, xc });
                if (xc <= gx0) continue;   // left of every tile's ray origin
                // Column bucket, half-open (colLeft, colRight].
                const float fc = std::ceil((xc - gx0) * invTileW) - 1.0f;
                if (fc >= (float) nx) {
                    scratchRowOverflow_[(size_t) r] += sign; // right of grid
                    continue;
                }
                // On tile (r, c)'s top edge: every tile left of it counts the
                // crossing in its backdrop (pass 2), and tile (r, c) OWNS it —
                // its fragments add it wherever it lies right of them. Which
                // tile owns a crossing is decided HERE, once, and travels to
                // the shader with the tile (owners first in its list, their
                // count in shapeInfo.w). The shader used to re-derive it as
                // `xc <= tileRight` from its own division, which matched this
                // one only where the GPU rounds OpFDiv exactly like the CPU —
                // Vulkan allows 2.5 ULP, and drivers that use it counted a
                // crossing on a tile edge twice or not at all, inverting the
                // whole tile.
                const int c = std::max(0, (int) fc);
                const size_t tile = (size_t)(r * nx + c);
                scratchTileWind_[tile] += sign;
                scratchTileOwners_[tile]++;
                scratchOwners_.push_back({ si, (uint32_t) tile, xc });
                // The crossing is a point of the segment, so its tile is one
                // of the segment's own local tiles; were rounding ever to say
                // otherwise, the owner still joins the tile's list, and the
                // list size has to say so.
                if (r < span.y0 || r > span.y1 || c < span.x0 || c > span.x1) {
                    jassertfalse;
                    scratchTileCounts_[tile]++;
                }
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

        // --- Pass 3: place segments tile-major, each tile's OWNERS first:
        // the shader applies the top term to exactly the first `owners`
        // entries of a tile's list. Two write cursors per tile — owners from
        // the tile's start, the rest from start + owners — so the rest
        // cursor ends at start + count and the emission loop recovers each
        // tile's start as cursor − count.
        //
        // An entry is TWO vec4s: the segment, then (xc, 0, 0, 0) — where it
        // crosses its tile's top line, from pass 1, when it does (0
        // otherwise: the shader reads xc only for a segment that crosses).
        scratchTileOffsets_.resize((size_t) nTiles);
        scratchOwnerCursor_.resize((size_t) nTiles);
        uint32_t total = 0;
        for (int i = 0; i < nTiles; i++) {
            scratchOwnerCursor_[(size_t) i] = total;
            scratchTileOffsets_[(size_t) i] = total + scratchTileOwners_[(size_t) i];
            total += scratchTileCounts_[(size_t) i];
        }
        scratchBinned_.resize((size_t) total * 2);
        auto place = [&](uint32_t entry, const glm::vec4& seg, float xc) {
            scratchBinned_[(size_t) entry * 2]     = seg;
            scratchBinned_[(size_t) entry * 2 + 1] = { xc, 0.0f, 0.0f, 0.0f };
        };
        for (const auto& o : scratchOwners_)
            place(scratchOwnerCursor_[o.tile]++, scratchSegments_[o.segment], o.xc);

        // Then every local (segment, tile) pair that is not one of those.
        // Pass 1 recorded owners and crossings segment by segment, rows
        // ascending, so a segment's owners — and its crossings — are each one
        // run in the same ascending order this loop visits its tiles in:
        // sorted merges over the runs skip the owners and find each row's xc.
        size_t ownersEnd = 0, crossingsEnd = 0;
        for (uint32_t si = 0; si < (uint32_t) scratchSegments_.size(); ++si) {
            size_t next = ownersEnd;
            while (ownersEnd < scratchOwners_.size() && scratchOwners_[ownersEnd].segment == si)
                ++ownersEnd;
            size_t crossing = crossingsEnd;
            while (crossingsEnd < scratchCrossings_.size() && scratchCrossings_[crossingsEnd].segment == si)
                ++crossingsEnd;
            TileSpan span;
            if (!localTiles(scratchSegments_[si], span)) continue;
            for (int ty = span.y0; ty <= span.y1; ty++) {
                while (crossing < crossingsEnd && scratchCrossings_[crossing].row < ty) ++crossing;
                const float xc = (crossing < crossingsEnd && scratchCrossings_[crossing].row == ty)
                               ? scratchCrossings_[crossing].xc : 0.0f;
                for (int tx = span.x0; tx <= span.x1; tx++) {
                    const size_t idx = (size_t)(ty * nx + tx);
                    while (next < ownersEnd && scratchOwners_[next].tile < idx) ++next;
                    if (next < ownersEnd && scratchOwners_[next].tile == idx) continue;   // placed above
                    place(scratchTileOffsets_[idx]++, scratchSegments_[si], xc);
                }
            }
        }
        const uint32_t segStart =
            (total > 0) ? renderer_.pushSegments(scratchBinned_.data(), total * 2) : 0u;

        // Register the gradient row up-front so the atlas upload happens in
        // this frame's staging pass. makeGradientCtx below also registers
        // idempotently — we mirror the fillRect convention here anyway.
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);

        // Per-vertex colour source, same as ColorDraw's primitive ops:
        // solid colour folded into the vertex attribute, or gradient
        // mode/row (or a tiled image's UV) packed into gradientInfo via
        // gradientAt(). Placed by the paint's transform: `t` moves the path,
        // never its fill (juce's fillShape).
        const auto fillIndex = captureFill();
        ops::GradientCtx grad = ops::makeFillCtx(renderer_, fillIndex, s.transform.scaled(displayScale_));
        glm::vec4 color = ops::fillColorAttr(grad, s.fill.colour, s.opacity);

        auto gi = [&](float px, float py) -> glm::vec4 {
            return grad.active() ? ops::gradientAt(grad, px, py)
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
                            uint32_t localStart, uint32_t localCount, int backdrop,
                            uint32_t owners) {
            const glm::vec2 uv { x, y };
            const glm::vec4 shape { (float) localStart, (float) localCount,
                                    (float) backdrop, (float) owners };
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
                         (float)(cEnd - runStart) * tileW, 0u, 0u, runB, 0u);
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
                             scratchTileOffsets_[idx] - cnt, cnt, B,
                             scratchTileOwners_[idx]);
                }
            }
            flushRun(nx);
        }
        if (scratchTileVerts_.empty()) return;

        FillPathParams p {};
        p.vertexCount  = static_cast<uint32_t>(scratchTileVerts_.size());
        p.segmentStart = segStart;
        p.fillRule     = fillRule;
        p.fillIndex    = fillIndex;
        p.tileSize     = tileW;

        record(DrawOp::FillPath, clipRect, p);
        renderer_.arena_align(4);
        renderer_.arena_pushSpan(std::span<const UIVertex>(
            scratchTileVerts_.data(), scratchTileVerts_.size()));
    }

    // Emit a solid fill gated by the active clipToImageAlpha mask: one quad
    // over the mask's placement, colour = brush (straight alpha, layer
    // opacity folded in), coverage = mask image alpha (shape type 5). The
    // fill rect restricts via the command's clip; the mask bounds were
    // already folded into clipBounds by clipToImageAlpha.
    void fillThroughAlphaMask(const juce::Rectangle<float>& r)
    {
        auto& s = state();
        const gpu::TextureId desc = renderer_.texture(s.alphaMask);
        if (!desc) return;   // the texture couldn't be made

        auto phys = r.transformedBy(s.transform.scaled(displayScale_))
                        .getSmallestIntegerContainer();
        auto clipRect = s.clipBounds.getIntersection(phys);
        if (clipRect.isEmpty()) return;

        DrawImageParams p { desc, s.alphaMaskTransform, 1.0f, displayScale_,
                            s.alphaMask.getWidth(), s.alphaMask.getHeight(),
                            s.maskNearest };
        if (s.fill.isGradient() && s.fill.gradient != nullptr)
        {
            // Gradient through the mask: replay rebuilds the gradient ctx
            // (sampleColor shades the quad, the mask alpha gates it). The
            // fill's opacity (FillType::setOpacity) rides the colour's alpha.
            renderer_.registerGradient(*s.fill.gradient);
            p.tint              = { 1.0f, 1.0f, 1.0f, s.fill.colour.getFloatAlpha() * s.opacity };
            p.alphaMaskFill     = 2;
            p.fillIndex         = captureFill();
            p.gradientTransform = s.transform;
        }
        else
        {
            p.tint = { s.fill.colour.getFloatRed(), s.fill.colour.getFloatGreen(),
                       s.fill.colour.getFloatBlue(),
                       s.fill.colour.getFloatAlpha() * s.opacity };
            p.alphaMaskFill = 1;
        }
        renderer_.push(DrawOp::DrawImage, clipRect, s.stencilDepth, p);
    }

    // The state's fill, captured for the frame. A tiled image fill takes its
    // texture with it, resolved now on the message thread, so every colour
    // pipeline draw can sample it (colour mode 3: ColorDraw makeFillCtx).
    uint32_t captureFill()
    {
        auto& s = state();
        return renderer_.captureFill(s.fill, s.fill.isTiledImage() ? tiledFillTexture() : gpu::TextureId {}, s.nearest);
    }

    // A tiled image fill's texture: null for no image or a singular placement,
    // which juce draws as nothing.
    gpu::TextureId tiledFillTexture()
    {
        auto& s = state();
        if (!s.fill.image.isValid() || s.fill.transform.isSingularity())
            return {};
        return renderer_.texture(s.fill.image);
    }

    // Tiled image fill: one quad over the fill rect, UVs mapped through the
    // inverse of the fill's image-placement transform, wrapped by fract()
    // in the shader (shape type 6). Exact for affine placements (UVs
    // interpolate linearly). Known divergence vs true REPEAT sampling: a
    // one-texel seam blend at tile edges (the cached sampler is CLAMP).
    void fillWithTiledImage(const juce::Rectangle<float>& r)
    {
        auto& s = state();
        const gpu::TextureId desc = tiledFillTexture();
        if (!desc) return;

        auto physRect = r.transformedBy(s.transform.scaled(displayScale_))
                            .getSmallestIntegerContainer();
        auto clipRect = s.clipBounds.getIntersection(physRect);
        if (clipRect.isEmpty()) return;

        const auto inv = s.fill.transform.inverted();   // user → image px
        const float iw = static_cast<float>(s.fill.image.getWidth());
        const float ih = static_cast<float>(s.fill.image.getHeight());
        const auto  tx = s.transform.scaled(displayScale_);

        FillTiledImageParams p {};
        p.texture = desc;
        const juce::Point<float> corners[4] = {
            { r.getX(),     r.getY()      }, { r.getRight(), r.getY()      },
            { r.getRight(), r.getBottom() }, { r.getX(),     r.getBottom() },
        };
        for (int i = 0; i < 4; i++)
        {
            const auto pp = corners[i].transformedBy(tx);
            const auto ip = corners[i].transformedBy(inv);
            p.pos[i] = { pp.x, pp.y };
            p.uv[i]  = { ip.x / iw, ip.y / ih };
        }
        // juce contract: image fills are modulated by the fill colour's
        // alpha (FillType::setOpacity writes it), layer opacity on top.
        p.tint = { 1.0f, 1.0f, 1.0f,
                   s.fill.colour.getFloatAlpha() * s.opacity };
        p.nearest = s.nearest;
        record(DrawOp::FillTiledImage, clipRect, p);
    }

    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !img.isValid()) return;
        auto& s = state();

        // jvk-backed images (jvk::ImageType) draw zero-copy: the PixelData
        // owns a resident texture with its own dirty tracking, so there is
        // no hash, no shared-cache lookup, and no re-upload unless the
        // pixels actually changed. Null (foreign image, or GPU failure)
        // falls through to the shared cache, which reads pixels via
        // BitmapData — jvk images answer that from their CPU mirror, so
        // the draw still happens either way.
        // Resolve the texture now (message thread — safe around the shared
        // cache's inserts/evictions) and capture it into the draw command:
        // the worker never touches the cache for this draw.
        const gpu::TextureId desc = renderer_.texture(img);
        if (!desc) return;   // the texture couldn't be made

        // juce semantics: the current fill colour's ALPHA modulates image
        // draws (g.setColour(c.withAlpha(0.5f)); g.drawImage(...) renders at
        // 50%) — UNCONDITIONALLY: the software renderer reads
        // fillType.colour.getAlpha() for gradient/image fills too
        // (juce_RenderingHelpers.h renderImage). s.opacity is the
        // transparency-layer multiplier on top.
        const float alpha = s.opacity * s.fill.colour.getFloatAlpha();
        record(DrawOp::DrawImage, s.clipBounds,
            DrawImageParams { desc, t.followedBy(s.transform), alpha, displayScale_,
                              img.getWidth(), img.getHeight(), s.nearest });
    }

    void drawLine(const juce::Line<float>& line) override
    {
        drawLineWithThickness(line, 1.0f);
    }

    void drawLineWithThickness(const juce::Line<float>& line, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::DrawLine, s.clipBounds,
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
        // thread. The worker's lowering (stageDirtyPages) then writes the
        // touched atlas rects BEFORE this frame's passes, so lowering only
        // ever samples written pages. Rasterizing during lowering instead
        // would bind a page the frame never wrote, and its pixels would
        // arrive one frame late. (rasterizeBudget inside the atlas caps
        // worst-case first-paint hitches; over-budget glyphs simply appear
        // next frame.)
        {
            auto& atlas = renderer_.glyphAtlas();
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

        DrawGlyphsParams params {};
        params.glyphCount = static_cast<uint32_t>(glyphs.size());
        params.transform = t.followedBy(s.transform);
        params.fillTransform = s.transform;   // juce fills glyphs in user space: `t` moves the glyphs only
        params.fontIndex = renderer_.captureFont(s.font);
        params.fillIndex = captureFill();
        params.opacity = s.opacity;
        params.scale = displayScale_;
        record(DrawOp::DrawGlyphs, s.clipBounds, params);
        // Append glyph POD data to arena (align before float data)
        renderer_.arena_pushSpan(std::span<const uint16_t>(glyphs.data(), glyphs.size()));
        renderer_.arena_align(4); // Point<float> requires 4-byte alignment
        renderer_.arena_pushSpan(std::span<const juce::Point<float>>(positions.data(), positions.size()));
    }

    // A corner no bigger than half a side, as juce's Path::addRoundedRectangle
    // clamps it: a thin bar at a large radius is a pill, not a pinched one.
    static float clampedCorner(const juce::Rectangle<float>& r, float cornerSize)
    {
        return juce::jmax(0.0f, juce::jmin(cornerSize, r.getWidth() * 0.5f, r.getHeight() * 0.5f));
    }

    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::FillRoundedRect, s.clipBounds,
            FillRoundedRectParams { r, clampedCorner(r, cornerSize), fi, s.transform, s.opacity, displayScale_ });
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::FillEllipse, s.clipBounds,
            FillEllipseParams { area, fi, s.transform, s.opacity, displayScale_ });
    }

    void drawRoundedRectangle(const juce::Rectangle<float>& rect, float cornerSize,
                              float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::StrokeRoundedRect, s.clipBounds,
            StrokeRoundedRectParams { rect, clampedCorner(rect, cornerSize), lineThickness, fi, s.transform,
                                      s.opacity, displayScale_ });
    }

    // drawRect is NOT overridden: juce's default fills four rectangles INSIDE
    // the bounds (fillRectList). The stroke centred on the edge it used to be
    // drew every 1 px component border half outside its clip, at half alpha.

    void drawEllipse(const juce::Rectangle<float>& area, float lineThickness) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto fi = captureFill();
        if (s.fill.isGradient() && s.fill.gradient)
            renderer_.registerGradient(*s.fill.gradient);
        record(DrawOp::StrokeEllipse, s.clipBounds,
            StrokeEllipseParams { area, lineThickness, fi, s.transform, s.opacity, displayScale_ });
    }

    // jvk backing for temporaries: setBufferedToImage caches paint on the
    // CPU mirror and composite zero-copy with dirty tracking (which also
    // retires the address-keyed-hash staleness hazard of repaint-in-place
    // images). Per-paint create-draw-destroy scratch (DropShadowEffect,
    // LookAndFeel_V2/V4 temps) is safe since PixelData's TextureKeeper made
    // destruction non-blocking — a mid-flight destroy just orphans the
    // texture until its frame's fence, no stall.
    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const override
    {
        return std::make_unique<jvk::ImageType>();
    }

    // Monotonic across frames (incremented by beginFrame): JUCE keys
    // per-frame caches on this, and the old per-frame Graphics object reset
    // it to zero every tick.
    uint64_t getFrameId() const override { return frameId_; }

    // ===== GPU Effects =====
    //
    // An effect's region in PHYSICAL pixels: the clip's bounds (already
    // physical) when none is given, else the caller's rect in its own logical
    // space through the transform and the display scale. Recorded with scale
    // 1, so the pipeline scales nothing again: the blend effects used to
    // record the physical clip bounds WITH the display scale, which drew
    // them at twice their place and size at 2x — the top-left of any
    // component not at the origin stayed untouched — and ignored
    // setOrigin/addTransform on explicit regions.
    juce::Rectangle<float> physicalRegion(juce::Rectangle<float> region) const
    {
        auto& s = state();
        return region.isEmpty() ? s.clipBounds.toFloat()
                                : region.transformedBy(s.transform.scaled(displayScale_));
    }

    void darken(float amount, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        float v = 1.0f - amount;
        record(DrawOp::EffectBlend, s.clipBounds,
            EffectBlendParams { v, v, v, physicalRegion(region), 1.0f });
    }

    void brighten(float amount, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        float v = 1.0f + amount;
        record(DrawOp::EffectBlend, s.clipBounds,
            EffectBlendParams { v, v, v, physicalRegion(region), 1.0f });
    }

    void tint(juce::Colour c, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        record(DrawOp::EffectBlend, s.clipBounds,
            EffectBlendParams { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), physicalRegion(region), 1.0f });
    }

    void warmth(float amount, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        record(DrawOp::EffectBlend, s.clipBounds,
            EffectBlendParams { 1.0f + amount * 0.2f, 1.0f, 1.0f - amount * 0.1f, physicalRegion(region), 1.0f });
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
        record(DrawOp::EffectNoise, s.clipBounds,
            NoiseParams { amount, timeOffset, physRegion, 1.0f });
    }

    // `radius` is in the caller's LOGICAL pixels, like every other distance
    // here (the blurred shapes fold the same factor into their step): it
    // reaches the shader in physical pixels, so a blur reads the same at 1x
    // and 2x. The replay-side ROI walk scissors every pass of the blur chain
    // to the region (padded per pass by the kernel's read margin).
    void blur(float radius, juce::Rectangle<float> region = {})
    {
        auto& s = state();
        record(DrawOp::EffectKernel, s.clipBounds,
            BlurParams { radius * s.transform.getScaleFactor() * displayScale_, physicalRegion(region), 1.0f });
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
        record(DrawOp::EffectHSV, s.clipBounds,
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

    // Path variants — the path blur program (ping-pong effect pass on the
    // scene target), which walks the same per-frame segment storage
    // fillPath uses. The path is flattened to physical-px
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
        // The draw captures everything lowering needs of the shader (its
        // program, bindings, uniform values and per-draw constants: Shader.h),
        // so the Shader may go at any time after. A reader also keeps the
        // targets it reads alive up to this draw (Recorder::noteTargetsRead).
        DrawShaderParams params { ~0u, regionPx, displayScale_ };
        prepareShaderDraw(shader, params);
        if (record(DrawOp::DrawShader, s.clipBounds, params))
            renderer_.noteTargetsRead(params.targetsRead);
    }

    Renderer& getRenderer() { return renderer_; }

    // The Shader for `spirv` and its `msl` from this window's shader cache
    // (Renderer::shader): built on first ask, kept with the context, shared by
    // every caller.
    Shader& shader(std::span<const uint32_t> spirv, std::string_view msl = {}) { return renderer_.shader(spirv, msl); }

private:
    // drawShader's record-time step: captures the shader (Renderer::
    // captureShader) and its per-draw constants. Defined in Shader.cpp:
    // this header only forward-declares Shader.
    void prepareShaderDraw(Shader& shader, DrawShaderParams& params);

    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        // Transparency-LAYER alpha only (beginTransparencyLayer multiplies it
        // in its own saved scope). Plain setOpacity follows the juce contract
        // instead: it rewrites fill.colour's alpha (FillType::setOpacity).
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        // clipToImageAlpha state (refcounted handle — cheap to copy on
        // saveState; restore pops it like any other clip). Invalid = none.
        juce::Image           alphaMask;
        juce::AffineTransform alphaMaskTransform;
        // Not yet in the stencil (materializeMask), and the scope it belongs
        // to (stateStack_ index); the resampling it was made with.
        bool                  maskPending = false;
        bool                  maskNearest = false;
        uint32_t              maskOwner = 0;
        // setInterpolationQuality(low) — point-sample every image draw.
        bool                  nearest = false;
        // Count of excludeClipRectangle parks taken in scopes at or below
        // this state; restoreState un-parks the difference (rects live in
        // exclusionShared_, stack-shaped like the path-clip params).
        uint32_t              exclusions = 0;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
        // Cached inverse of transform.scaled(displayScale) for getClipBounds
        // (JUCE calls it per child). Copied validly on saveState (same
        // transform); invalidated by setOrigin/addTransform.
        mutable juce::AffineTransform pixelToLocal;
        mutable bool                  inverseValid = false;
        // Stencil clips live in Graphics::stencilClips_ (stack-shaped across
        // states); no per-state vector to deep-copy on every saveState.

        // Draw target (setTarget): colour attachment index (0 = main colour,
        // i + 1 = target i, Renderer::kNoTarget = a missing target), its
        // name.
        uint8_t               target = 0;
        juce::Identifier      targetName;
    };

    RecordState& state() { return stateStack_.back(); }
    const RecordState& state() const { return stateStack_.back(); }

    // The renderer stamps every recorded command with the current target.
    void syncTarget() { renderer_.setRecordTarget(state().target); }

    // Record a PopClip — same ClipShapeParams and scissor as the paired
    // PushClipPath so the DECR_WRAP at the GPU matches the INCR_WRAP
    // exactly. Stencil reference at replay = stencilDepth BEFORE the pop
    // (current depth), so the hardware decrements only the pixels the push
    // incremented.
    void recordPopClip()
    {
        auto& s = state();
        const auto c = stencilClips_.back();
        stencilClips_.pop_back();
        renderer_.push(DrawOp::PopClipPath, c.bounds, s.stencilDepth, c.params);
        s.scopeDepth--;
        if (s.stencilDepth > 0) s.stencilDepth--;
    }

    // Records a draw. A pending clipToImageAlpha mask enters the stencil
    // first: everything but a fill through the mask itself must be clipped
    // by it.
    template <typename Params>
    bool record(DrawOp op, const juce::Rectangle<int>& bounds, const Params& params)
    {
        materializeMask();
        return renderer_.push(op, bounds, state().stencilDepth, params);
    }

    // Pushes the pending mask as a stencil clip (shape 3) of the scope that
    // made it: every state from that one up takes the push, so it pops when
    // that scope restores. No clip was pushed since it was made (a clip
    // materializes it first), so they all stand at the same depth.
    void materializeMask()
    {
        auto& s = state();
        if (!s.maskPending) return;
        for (auto i = static_cast<size_t>(s.maskOwner); i < stateStack_.size(); ++i)
            stateStack_[i].maskPending = false;
        if (s.clipBounds.isEmpty()) return;   // nothing can draw in this scope
        const gpu::TextureId desc = renderer_.texture(s.alphaMask);
        if (!desc) return;   // the texture couldn't be made: its bounds still clip

        const auto toMask = s.alphaMaskTransform.scaled(displayScale_).inverted();
        ClipShapeParams params {};
        params.shapeType   = 3u;
        params.coverRect   = s.clipBounds.toFloat();
        params.mask        = desc;
        params.toMask[0]   = toMask.mat00; params.toMask[1] = toMask.mat01;   // x row
        params.toMask[2]   = toMask.mat10; params.toMask[3] = toMask.mat11;   // y row
        params.toMask[4]   = toMask.mat02; params.toMask[5] = toMask.mat12;   // offset
        params.maskW       = static_cast<float>(s.alphaMask.getWidth());
        params.maskH       = static_cast<float>(s.alphaMask.getHeight());
        params.maskNearest = s.maskNearest ? 1u : 0u;
        stencilClips_.push_back({ params, s.clipBounds, s.scopeDepth });
        renderer_.push(DrawOp::PushClipPath, s.clipBounds, s.stencilDepth, params);
        for (auto i = static_cast<size_t>(s.maskOwner); i < stateStack_.size(); ++i) {
            stateStack_[i].stencilDepth++;
            stateStack_[i].scopeDepth++;
        }
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

        record(DrawOp::BlurShape, s.clipBounds, p);
    }

    // Pack a BlurPath draw command. Flattens the path to line segments
    // (physical pixels), appends them to the frame's segment storage,
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
        p.segmentStart = binAndUploadSegments(pxB.getY(), pxB.getBottom(),
                                              reach, p.stripCount,
                                              p.stripMinY, p.invStripH);

        // ROI: path bounds (already transformed to physical by `combined`)
        // expanded by the blur reach + stroke ring, all physical px. Inverted
        // blurs leave it empty = full clip at replay.
        if (!inverted)
            p.region = pathBounds.transformedBy(combined)
                           .expanded(p.maxRadius + p.falloff + p.strokeHalfWidth + 2.0f);

        record(DrawOp::BlurPath, s.clipBounds, p);
    }

    // Flatten a path to a flat list of line segments in the supplied
    // transform's target space, packed as vec4 (p0.xy, p1.xy) ready for the
    // frame's segment storage. Shares the same subdivision strategy as
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
    juce::ValueTree values_ { "Frame" };   // values(): emptied every frame

    // Active stencil clips (paths, masks), shared across the state stack
    // (strictly stack-shaped: pushes in clipToPath / materializeMask, pops in
    // recordPopClip; saveState copies nothing). Each keeps its push's
    // scissor for its pop, and the scope depth it was pushed at (restoreState
    // pops a scope's rect and stencil clips in reverse push order by it).
    struct StencilClip { ClipShapeParams params; juce::Rectangle<int> bounds; uint32_t scope; };
    std::vector<StencilClip> stencilClips_;

    // Active exclusion parks (excludeClipRectangle), strictly stack-shaped
    // across the state stack like stencilClips_.
    struct Exclusion { juce::Rectangle<float> rect; uint8_t depth; };
    std::vector<Exclusion> exclusionShared_;

    // Flatten scratch buffers — cleared-but-not-freed between paths AND
    // between frames (the editor reuses one Graphics via beginFrame), so
    // the hot path is genuinely allocation-free after warmup. Used by both
    // Graphics::fillPath and clipToPath (both into the frame's segment
    // storage).
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
    std::vector<uint32_t>  scratchTileOwners_;    // per tile: crossings of its top edge it owns
    std::vector<uint32_t>  scratchOwnerCursor_;
    struct TileOwner { uint32_t segment, tile; float xc; };
    std::vector<TileOwner> scratchOwners_;        // segment order, rows ascending within one
    struct RowCrossing { uint32_t segment; int row; float xc; };
    std::vector<RowCrossing> scratchCrossings_;   // every row-top crossing, same order
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
    uint32_t binAndUploadSegments(float minYf, float maxYf, float pad,
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
            return renderer_.pushSegments(scratchSegments_.data(), segCount);
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
        return renderer_.pushSegments(scratchBinned_.data(),
                                  static_cast<uint32_t>(scratchBinned_.size()));
    }
};

} // namespace jvk
