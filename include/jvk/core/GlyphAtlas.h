/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: GlyphAtlas.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include <msdfgen.h>

namespace jvk
{

// The glyph atlas: every glyph a frame draws, made once on the message thread
// at record time and written to its page before the frame that samples it.
//
// An OUTLINE glyph is a multi-channel signed distance field (msdfgen, from
// JUCE's outline): one per glyph, sharp at any size, in any fill.
//
// A COLOUR glyph (an emoji: a bitmap font's picture, a COLR font's coloured
// layers) is a picture, which a distance field can't hold. JUCE's own image
// renderer paints it (a juce::Image's: CoreGraphics, Direct2D or JUCE's
// software renderer) into a premultiplied RGBA tile as the CPU paint draws
// it: the run's own font under the scale from its space to the device (an
// emoji's size is the font's, not the pixels': CoreText draws one at 12 pt
// under 2x bigger than at 24 pt), and for axis-aligned text at the 1/64
// pixel it falls on, split as the CPU paint splits it: the run's offset in
// the transform, the glyph's own as its position (CoreText snaps a colour
// glyph's position to a whole pixel and resamples a transform's fraction;
// JUCE's software renderer resamples both), so the tile lands pixel for
// pixel. Which glyphs are colour, and whether they take the text's alpha or
// colour, is what that renderer does: each glyph of a colour font is painted
// once in white and once in half-transparent black to see. A glyph with no
// outline JUCE can read that the renderer still paints (macOS 26's PingFang)
// is drawn the same way, as a tile in the text's colour.
//
// Outline glyphs are the `entries`, colour glyphs `colourGlyphs`, and a glyph
// with nothing to draw (a space, an outline-less emoji) is in `empties`, so it
// is never made again. Kept apart, a lookup that finds an entry needs no
// question: text without colour glyphs runs the same lookups as ever.
//
// Glyphs are cached under a typeface's family and style, not the Typeface
// object: JUCE makes a NEW Typeface for a system fallback font each time it
// shapes text (CoreTextTypeface::createSystemFallback), so keyed by pointer
// every re-shaped run was a font never seen before, made again and held for
// good. JUCE's typeface cache treats a family and style as one font; so does
// the atlas (canonicalTypeface).
class GlyphAtlas
{
public:
    static constexpr int ATLAS_SIZE = 2048;
    static constexpr int MSDF_GLYPH_SIZE = 32;   // texels per glyph (resolution-independent)
    // Padding must cover the full distance-field reach — if GLYPH_PADDING is
    // smaller than MSDF_PIXEL_RANGE the field gets truncated inside the atlas
    // tile, which shows up as pinhole artifacts in the counters of glyphs
    // with thin interior features (e.g. 'e', 'a').
    static constexpr double MSDF_PIXEL_RANGE = 4.0; // distance range in atlas pixels
    static constexpr int GLYPH_PADDING = static_cast<int>(MSDF_PIXEL_RANGE);

    // Colour tiles: 1024² RGBA pages, at most kMaxColourPages of them. When a
    // tile no longer fits, every colour tile goes at the next frame and the
    // ones still drawn are painted again (the frame that overflowed skips it).
    static constexpr int COLOUR_PAGE_SIZE = 1024;
    static constexpr int kMaxColourPages = 4;
    static constexpr float kMaxTileHeight = 256.0f;   // device px of font; a larger glyph draws its tile scaled

    struct GlyphKey
    {
        // The CANONICAL typeface (canonicalTypeface), which the atlas holds
        // for as long as it holds entries: its address can't be recycled
        // under a key, and a raw pointer costs a key no refcount traffic.
        const juce::Typeface* typeface = nullptr;
        uint16_t glyphId = 0;
        // Part of the outline identity: rasterization passes it to
        // getOutlineForGlyph, so two Fonts differing only in metrics kind
        // must not collide on one atlas entry.
        int metricsKind = 0;

        bool operator==(const GlyphKey& o) const
        {
            return typeface == o.typeface && glyphId == o.glyphId
                && metricsKind == o.metricsKind;
        }
    };

    struct GlyphKeyHash
    {
        size_t operator()(const GlyphKey& k) const
        {
            size_t h = reinterpret_cast<size_t>(k.typeface);
            h ^= std::hash<uint16_t>{}(k.glyphId) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.metricsKind) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct AtlasEntry
    {
        float u0, v0, u1, v1; // UV coordinates in atlas
        int w, h;              // MSDF dimensions in atlas pixels
        int atlasIndex;
        // Glyph metrics for positioning (in normalized font units, height=1.0)
        float boundsX, boundsY, boundsW, boundsH;
    };

    // A colour glyph as the image renderer paints it: faded by the text
    // colour's alpha (juce's software renderer fades a bitmap glyph, not a
    // COLR one), and partly IN the text colour (a COLR layer in the
    // foreground colour), when its tiles are painted in the run's colour.
    // Its own type, so the colour map shares no code with the entries': the
    // compiler keeps inlining the outline lookup into the glyph loops.
    struct ColourGlyph
    {
        bool fillAlpha = false, fillColour = false;
    };

    // A colour glyph's tile: the glyph, the run font's height (1/64 px) and
    // horizontal scale (1/1024), the scale it is painted under (1/65536), the
    // phases (1/64 px) of the run's offset and of the glyph's own, and the
    // text colour (ARGB) for a glyph that takes it.
    struct ColourKey
    {
        GlyphKey glyph;
        uint32_t height = 0, scaleX = 0, scaleY = 0, colour = 0;
        uint16_t hscale = 0;
        uint8_t  runX = 0, runY = 0, glyphX = 0, glyphY = 0;
        bool operator==(const ColourKey&) const = default;
    };

    struct ColourKeyHash
    {
        size_t operator()(const ColourKey& k) const
        {
            size_t h = GlyphKeyHash{}(k.glyph);
            for (const uint64_t v : { (uint64_t(k.height) << 32) | k.colour, (uint64_t(k.scaleX) << 32) | k.scaleY,
                                      (uint64_t(k.hscale) << 32) | (uint64_t(k.runX) << 24) | (uint64_t(k.runY) << 16)
                                          | (uint64_t(k.glyphX) << 8) | k.glyphY })
                h ^= std::hash<uint64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // A tile in its page, a transparent pixel round what it covers; w == 0:
    // the glyph paints nothing at this size.
    struct ColourEntry
    {
        float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
        int w = 0, h = 0, page = -1;
        float originX = 0, originY = 0;   // the glyph's origin in the tile, px
    };

    // How a run's colour glyphs stand on the device, from its font and `tx`
    // (the run's space to device pixels): the scale their tiles are painted
    // under, which is also the tile pixels per unit of the run's space
    // (axis-aligned: tx's own; else its area scale, and the tile turns with
    // the quad). `exact`: axis-aligned at tx's own scale, so each tile stands
    // pixel for pixel at its phases (`runX/Y`: of the run's origin, tx's
    // translation, in 1/64 px); otherwise it is drawn transformed.
    struct ColourRun
    {
        uint32_t height = 0, scaleX = 0, scaleY = 0;
        uint16_t hscale = 0;
        float kx = 0, ky = 0;
        bool exact = false;
        int runX = 0, runY = 0;
        uint32_t colour = 0;   // the run's fill colour, ARGB: what a glyph that takes it is painted in
    };

    static ColourRun colourRun(const juce::Font& font, const juce::AffineTransform& tx, juce::Colour fill)
    {
        ColourRun r;
        const float h = font.getHeight(), hs = font.getHorizontalScale();
        const bool aligned = tx.mat01 == 0.0f && tx.mat10 == 0.0f && tx.mat00 > 0.0f && tx.mat11 > 0.0f;
        const float ay = aligned ? tx.mat11 : std::sqrt(std::abs(tx.getDeterminant()));
        const float ax = aligned ? tx.mat00 : ay;
        if (!(h > 0.0f && hs > 0.0f && ay > 0.0f)) return r;
        const float cap = std::min(1.0f, kMaxTileHeight / (h * ay));
        r.height = static_cast<uint32_t>(std::lround(h * 64.0f));
        r.hscale = static_cast<uint16_t>(juce::jlimit(1L, 65535L, std::lround(hs * 1024.0f)));
        r.scaleX = static_cast<uint32_t>(std::lround(ax * cap * 65536.0f));
        r.scaleY = static_cast<uint32_t>(std::lround(ay * cap * 65536.0f));
        r.kx = static_cast<float>(r.scaleX) / 65536.0f;
        r.ky = static_cast<float>(r.scaleY) / 65536.0f;
        r.exact = aligned && std::abs(r.kx - ax) <= 1e-4f * ax && std::abs(r.ky - ay) <= 1e-4f * ay;
        if (r.exact)
        {
            r.runX = sixtyFourths(tx.mat02);
            r.runY = sixtyFourths(tx.mat12);
        }
        r.colour = fill.getARGB();
        return r;
    }

    // The tile key of colour glyph `base` (`key`) at `origin` (the run's space).
    static ColourKey colourKey(const ColourGlyph& base, const GlyphKey& key, const ColourRun& run,
                               juce::Point<float> origin, const juce::AffineTransform& tx)
    {
        ColourKey k { key, run.height, run.scaleX, run.scaleY, base.fillColour ? run.colour : 0u, run.hscale, 0, 0, 0, 0 };
        if (run.exact)
        {
            k.runX = static_cast<uint8_t>(run.runX & 63);
            k.runY = static_cast<uint8_t>(run.runY & 63);
            k.glyphX = static_cast<uint8_t>(sixtyFourths(tx.mat00 * origin.x) & 63);
            k.glyphY = static_cast<uint8_t>(sixtyFourths(tx.mat11 * origin.y) & 63);
        }
        return k;
    }

    // Where tile `e` of the glyph at `origin` lands: its corners in device px
    // (top-left, top-right, bottom-right, bottom-left).
    static std::array<juce::Point<float>, 4> colourQuad(const ColourRun& run, const ColourEntry& e,
                                                        juce::Point<float> origin, const juce::AffineTransform& tx)
    {
        const float w = static_cast<float>(e.w), h = static_cast<float>(e.h);
        if (run.exact)
        {
            auto at = [](int q) { return (static_cast<float>(q) + 0.5f) / 64.0f; };
            const float x = std::round(at(run.runX) + at(sixtyFourths(tx.mat00 * origin.x)) - e.originX);
            const float y = std::round(at(run.runY) + at(sixtyFourths(tx.mat11 * origin.y)) - e.originY);
            return { { { x, y }, { x + w, y }, { x + w, y + h }, { x, y + h } } };
        }
        auto at = [&](float u, float v) {
            return (origin + juce::Point<float>((u - e.originX) / run.kx, (v - e.originY) / run.ky)).transformedBy(tx);
        };
        return { { at(0, 0), at(w, 0), at(w, h), at(0, h) } };
    }

    void init(gpu::Backend& backend)
    {
        this->backend = &backend;
    }

    // The typeface a run's glyphs are cached under: the first one seen with
    // its family and style. Record time (message thread); the draw carries it
    // to lowering (DrawGlyphsParams::typeface). Consecutive runs mostly share
    // a font, so the last one is answered without a lookup (`last_` holds it:
    // its address can't come back as another typeface).
    const juce::Typeface* canonicalTypeface(const juce::Typeface::Ptr& t)
    {
        if (t == nullptr) return nullptr;
        if (t == last_) return lastCanonical_;
        auto it = aliases_.find(t.get());
        if (it == aliases_.end())
        {
            auto& canonical = byName_[t->getName() + "\n" + t->getStyle()];
            if (canonical == nullptr) canonical = t;
            // An alias held by nothing but this map can't be drawn again: let
            // it go (once freed, a new typeface at its address is looked up anew).
            if (aliases_.size() >= kMaxAliases)
                std::erase_if(aliases_, [](const auto& a) { return a.second.typeface->getReferenceCount() == 1; });
            it = aliases_.emplace(t.get(), Alias { t, canonical }).first;
        }
        last_ = t;
        return lastCanonical_ = it->second.canonical.get();
    }

    // The pages' textures, handed to the owner (Renderer) to retire through
    // the Context: a frame still unsubmitted may name them.
    template <typename Retire>
    void clear(Retire&& retire)
    {
        entries.clear();
        empties.clear();
        colourGlyphs.clear();
        colourEntries.clear();
        aliases_.clear();
        byName_.clear();
        last_ = nullptr;
        lastCanonical_ = nullptr;
        for (auto& page : atlasPages)
            if (page.texture) retire(page.texture);
        for (auto& page : colourPages)
            if (page.texture) retire(page.texture);
        atlasPages.clear();
        colourPages.clear();
    }

    // Lookup + rasterize-on-miss. Called at RECORD time (message thread,
    // Graphics::drawGlyphs) so pages dirty here are written by the worker's
    // stageDirtyPages BEFORE the frame that samples them. An outline glyph's
    // entry; null for one over budget, empty, or colour (colourGlyph), which
    // it notes (takeColourGlyphSeen).
    const AtlasEntry* getGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto it = entries.find(key);
        if (it != entries.end())
            return &it->second;
        if (empties.contains(key))
            return nullptr;
        if (colourGlyphs.contains(key))
        {
            colourGlyphSeen_ = true;
            return nullptr;
        }
        // Budget the expensive MSDF generation per frame so the first paint
        // of a big new font can't hitch the message thread unboundedly —
        // over-budget glyphs rasterize next frame (invisible for one frame,
        // which replay handles by skipping the missing entry).
        if (rasterizedThisFrame_ >= kRasterBudgetPerFrame)
        {
            deferred_ = true;
            return nullptr;
        }
        return rasterizeGlyph(key, font);
    }

    // True once after getGlyph met a colour glyph: the run being recorded
    // needs its tiles (getColourGlyph).
    bool takeColourGlyphSeen() { return std::exchange(colourGlyphSeen_, false); }

    // A colour glyph's entry (its probe's findings), or null: find only, so
    // record and lowering both use it.
    const ColourGlyph* colourGlyph(const GlyphKey& key) const
    {
        auto it = colourGlyphs.find(key);
        return it != colourGlyphs.end() ? &it->second : nullptr;
    }

    // Find-only lookup for LOWERING (worker thread): never rasterizes — a miss
    // means the glyph was over budget this frame and simply isn't drawn.
    // Rasterizing here would sample a page this frame never wrote (the pages
    // are written before lowering).
    const AtlasEntry* peekGlyph(const GlyphKey& key) const
    {
        auto it = entries.find(key);
        return it != entries.end() ? &it->second : nullptr;
    }

    // A colour glyph's tile, painted on a miss: record time, as getGlyph.
    const ColourEntry* getColourGlyph(const ColourGlyph& base, const ColourKey& key, const juce::Font& font)
    {
        if (colourResetPending_ && frame_ != colourOverflowFrame_) resetColourPages();
        if (auto it = colourEntries.find(key); it != colourEntries.end()) return &it->second;
        if (rasterizedThisFrame_ >= kRasterBudgetPerFrame || colourResetPending_) { deferred_ = true; return nullptr; }
        rasterizedThisFrame_++;
        return rasterizeColour(base, key, font);
    }

    const ColourEntry* peekColourGlyph(const ColourKey& key) const
    {
        auto it = colourEntries.find(key);
        return it != colourEntries.end() ? &it->second : nullptr;
    }

    gpu::TextureId pageTexture(int atlasIndex) const
    {
        if (atlasIndex < 0 || atlasIndex >= static_cast<int>(atlasPages.size()))
            return {};
        return atlasPages[static_cast<size_t>(atlasIndex)].texture;
    }

    gpu::TextureId colourPageTexture(int page) const
    {
        if (page < 0 || page >= static_cast<int>(colourPages.size())) return {};
        return colourPages[static_cast<size_t>(page)].texture;
    }

    // Writes the dirty pages to their textures (Backend::writeTexture: before
    // the frame being lowered, which samples them). True when the frame just
    // recorded left glyphs for a later one (over budget, or a page not yet
    // written): a renderer that renders only on change must render again,
    // or they never appear.
    bool stageDirtyPages()
    {
        // New frame — refill the record-side rasterization budget. Runs on
        // the worker, but never concurrently with record (isBusy gate).
        rasterizedThisFrame_ = 0;
        ++frame_;
        const bool again = std::exchange(deferred_, false);

        for (auto& page : atlasPages)
            if (!stagePage(page)) return true;   // OOM — retry next frame
        for (auto& page : colourPages)
            if (!stagePage(page)) return true;
        return again;
    }

private:
    struct AtlasPage
    {
        gpu::TextureId texture;
        int size = ATLAS_SIZE;
        std::vector<uint32_t> pixels;
        int cursorX = 0;
        int cursorY = 0;
        int shelfHeight = 0;
        bool needsUpload = false;
        bool uploadedOnce = false;
        // Dirty rect accumulated by rasterization, consumed (and reset) by
        // stageDirtyPages. Empty when x1 <= x0.
        int dirtyX0 = ATLAS_SIZE, dirtyY0 = ATLAS_SIZE, dirtyX1 = 0, dirtyY1 = 0;

        void dirty(int x, int y, int w, int h)
        {
            needsUpload = true;
            dirtyX0 = std::min(dirtyX0, x);
            dirtyY0 = std::min(dirtyY0, y);
            dirtyX1 = std::max(dirtyX1, x + w);
            dirtyY1 = std::max(dirtyY1, y + h);
        }
    };

    bool stagePage(AtlasPage& page)
    {
        if (!page.needsUpload)
            return true;
        const int size = page.size;

        if (!page.uploadedOnce)
        {
            // First upload of a page: the whole page (lands the black
            // border everywhere).
            const auto bytes = std::as_bytes(std::span<const uint32_t>(page.pixels));
            if (!backend->writeTexture(page.texture, 0, 0, { 0, 0, size, size }, bytes))
                return false;
            page.uploadedOnce = true;
        }
        else if (page.dirtyX1 > page.dirtyX0 && page.dirtyY1 > page.dirtyY0)
        {
            // Incremental: only the dirty rect. A full page would cost a
            // 16 MB copy for ONE new glyph.
            const int dw = page.dirtyX1 - page.dirtyX0;
            const int dh = page.dirtyY1 - page.dirtyY0;
            rectScratch_.resize(static_cast<size_t>(dw) * static_cast<size_t>(dh));
            for (int y = 0; y < dh; y++)
                memcpy(rectScratch_.data() + static_cast<size_t>(y) * static_cast<size_t>(dw),
                       page.pixels.data() + static_cast<size_t>(page.dirtyY0 + y) * static_cast<size_t>(size)
                                          + static_cast<size_t>(page.dirtyX0),
                       static_cast<size_t>(dw) * 4);
            if (!backend->writeTexture(page.texture, 0, 0, { page.dirtyX0, page.dirtyY0, dw, dh },
                                       std::as_bytes(std::span<const uint32_t>(rectScratch_))))
                return false;
        }

        page.needsUpload = false;
        page.dirtyX0 = page.dirtyY0 = size;
        page.dirtyX1 = page.dirtyY1 = 0;
        return true;
    }

    // The 1/64 pixel a position falls in: q >> 6 is the pixel, q & 63 the
    // phase. A tile is painted at its phase's centre, never on a boundary a
    // renderer rounds at (a whole or fractional pixel), so it rounds the way
    // the position itself does.
    static int sixtyFourths(float v) { return static_cast<int>(std::floor(v * 64.0f)); }

    // Remembers that `key` draws nothing, or is a colour glyph (no entry:
    // getGlyph answers null; colourGlyph finds it).
    const AtlasEntry* emptyEntry(const GlyphKey& key)
    {
        empties.insert(key);
        return nullptr;
    }
    const AtlasEntry* colourEntry(const GlyphKey& key, const ColourGlyph& seen)
    {
        colourGlyphs.emplace(key, seen);
        colourGlyphSeen_ = true;
        return nullptr;
    }

    const AtlasEntry* rasterizeGlyph(const GlyphKey& key, const juce::Font& font)
    {
        const auto* typeface = key.typeface;
        if (typeface == nullptr) return nullptr;

        // A colour font's glyph is whatever the image renderer paints. Painted
        // in one colour (the fill's) it is an outline glyph like any other.
        std::optional<Probe> seen;
        if (typeface->getColourGlyphFormats() != 0)
        {
            seen = probe(font, key.glyphId);
            if (seen->kind == Kind::Empty) return emptyEntry(key);
            rasterizedThisFrame_++;
            if (seen->kind == Kind::Colour) return colourEntry(key, seen->glyph);
        }

        // Get glyph outline (normalized to height=1.0 by JUCE). JUCE applies
        // scale(s, -s) internally → Y is down, matching Vulkan's V-down
        // texture storage convention so we leave inverseYAxis=false.
        juce::Path glyphPath;
        typeface->getOutlineForGlyph(
            static_cast<juce::TypefaceMetricsKind>(key.metricsKind),
            key.glyphId, glyphPath);

        // Delegate the entire MSDF pipeline (path → shape → normalise →
        // orient → edge-colour → generate → sign-correct → error-correct)
        // to the shared jvk::msdf module. Config matches the tuned values
        // we previously baked into this class:
        //   MSDF_GLYPH_SIZE  → Config::maxDimension
        //   GLYPH_PADDING    → Config::padding
        //   MSDF_PIXEL_RANGE → Config::pxRange
        // Glyphs are always non-zero winding (font standard).
        msdf::Config cfg;
        cfg.maxDimension = MSDF_GLYPH_SIZE;
        cfg.padding      = GLYPH_PADDING;
        cfg.pxRange      = MSDF_PIXEL_RANGE;
        cfg.fillRule     = msdf::FillRule::NonZero;

        auto result = glyphPath.isEmpty() ? msdf::Result {} : msdf::rasteriseJucePath(glyphPath, cfg);
        if (!result.valid || result.bitmap.width() > ATLAS_SIZE / 2 || result.bitmap.height() > ATLAS_SIZE / 2)
        {
            // No outline JUCE can read (a space; or a face its HarfBuzz can't
            // draw, as macOS 26's PingFang): whatever the image renderer
            // paints, as tiles, in the run's colour where it takes the fill's.
            if (!seen) seen = probe(font, key.glyphId);
            if (seen->kind == Kind::Empty) return emptyEntry(key);
            rasterizedThisFrame_++;
            return colourEntry(key, seen->glyph);
        }

        int msdfW = result.bitmap.width();
        int msdfH = result.bitmap.height();

        auto pathBounds = result.shapeBounds;
        auto& msdfBitmap = result.bitmap;

        // Find space in atlas
        auto* page = findSpace(atlasPages, ATLAS_SIZE, msdfW + 1, msdfH + 1, 0xFF000000u, "jvk glyph atlas page");
        if (!page) return nullptr;

        int atlasIdx = static_cast<int>(page - atlasPages.data());
        int px = page->cursorX;
        int py = page->cursorY;

        // Belt-and-braces bounds guard: the pixel loop below writes
        // [px, px+msdfW) × [py, py+msdfH) with unchecked indices — a
        // packing-logic slip must fail a lookup, not scribble the heap.
        if (px + msdfW > ATLAS_SIZE || py + msdfH > ATLAS_SIZE)
            return nullptr;

        // Copy MSDF to atlas pixels (RGB = distance channels, A = 255)
        for (int y = 0; y < msdfH; y++)
        {
            for (int x = 0; x < msdfW; x++)
            {
                // msdfgen output is float [0,1] after normalization
                // Clamp and convert to 8-bit
                const float* pixel = msdfBitmap(x, y);
                auto toU8 = [](float v) -> uint8_t {
                    return static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, v * 255.0f));
                };

                size_t idx = static_cast<size_t>((py + y) * ATLAS_SIZE + (px + x));
                page->pixels[idx] =
                    static_cast<uint32_t>(toU8(pixel[0]))
                  | (static_cast<uint32_t>(toU8(pixel[1])) << 8)
                  | (static_cast<uint32_t>(toU8(pixel[2])) << 16)
                  | (0xFFu << 24); // full alpha
            }
        }
        page->dirty(px, py, msdfW, msdfH);

        // Advance shelf cursor
        page->cursorX = px + msdfW + 1;
        page->shelfHeight = std::max(page->shelfHeight, msdfH + 1);

        // Store entry with normalized glyph metrics
        float invSize = 1.0f / static_cast<float>(ATLAS_SIZE);
        AtlasEntry entry;
        entry.u0 = static_cast<float>(px) * invSize;
        entry.v0 = static_cast<float>(py) * invSize;
        entry.u1 = static_cast<float>(px + msdfW) * invSize;
        entry.v1 = static_cast<float>(py + msdfH) * invSize;
        entry.w = msdfW;
        entry.h = msdfH;
        entry.atlasIndex = atlasIdx;
        // Store bounds in normalized font units (height=1.0) for positioning
        entry.boundsX = pathBounds.getX();
        entry.boundsY = pathBounds.getY();
        entry.boundsW = pathBounds.getWidth();
        entry.boundsH = pathBounds.getHeight();

        rasterizedThisFrame_++;
        auto [insertIt, _] = entries.emplace(key, entry);
        return &insertIt->second;
    }

    // ---- Colour glyphs -----------------------------------------------------

    // Glyph `glyph` of `f` painted in `colour` by a juce::Image's renderer
    // under `scale` (device px per unit of the font's space) in an image
    // covering `area` of the device (px, the origin at 0, y down), its origin
    // at `run` + `own` (fractions of a device px) passed as the CPU paint
    // passes them: the run's in the transform, the glyph's as its position.
    static juce::Image paintGlyph(const juce::Font& f, uint16_t glyph, juce::Colour colour, juce::Point<float> scale,
                                  juce::Rectangle<int> area, juce::Point<float> run, juce::Point<float> own)
    {
        juce::Image image(juce::Image::ARGB, area.getWidth(), area.getHeight(), true);
        juce::Graphics g(image);
        g.setColour(colour);
        g.addTransform(juce::AffineTransform::scale(scale.x, scale.y).translated(run - area.getPosition().toFloat()));
        auto& ctx = g.getInternalContext();
        ctx.setFont(f);
        const juce::Point<float> at(own.x / scale.x, own.y / scale.y);
        ctx.drawGlyphs({ &glyph, 1 }, { &at, 1 }, {});
        return image;
    }

    // The box (device px, origin at 0) a glyph of `f` can paint in under
    // `scale`: its bounds, never less than an em either way, with a margin for
    // anti-aliasing.
    static juce::Rectangle<int> paintArea(const juce::Font& f, uint16_t glyph, juce::Point<float> scale)
    {
        const float h = f.getHeight() * scale.y, w = f.getHeight() * f.getHorizontalScale() * scale.x;
        auto box = juce::Rectangle<float>(-0.25f * w, -1.25f * h, 1.5f * w, 1.75f * h);
        if (auto* tf = f.getTypefacePtr().get())
            box = box.getUnion(tf->getGlyphBounds(f.getMetricsKind(), glyph).transformedBy(juce::AffineTransform::scale(w, h)));
        return box.expanded(2.0f + 0.125f * h).getSmallestIntegerContainer();
    }

    // The pixels with any coverage.
    static juce::Rectangle<int> coverage(const juce::Image& image)
    {
        const juce::Image::BitmapData bd(image, juce::Image::BitmapData::readOnly);
        int x0 = bd.width, y0 = bd.height, x1 = 0, y1 = 0;
        for (int y = 0; y < bd.height; ++y)
        {
            const auto* line = reinterpret_cast<const juce::PixelARGB*>(bd.getLinePointer(y));
            for (int x = 0; x < bd.width; ++x)
                if (line[x].getAlpha() != 0) { x0 = std::min(x0, x); x1 = std::max(x1, x + 1); y0 = std::min(y0, y); y1 = std::max(y1, y + 1); }
        }
        return x1 > x0 ? juce::Rectangle<int>(x0, y0, x1 - x0, y1 - y0) : juce::Rectangle<int>();
    }

    // What the image renderer makes of a colour font's glyph, painted at a
    // probe size in opaque white (a) and half-transparent black (b): nothing;
    // white where it shows and black in b (it takes the fill: an outline
    // glyph); else a colour glyph, faded in b if it takes the fill's alpha,
    // with white pixels gone black if parts of it take the fill's colour.
    enum class Kind : uint8_t { Outline, Colour, Empty };
    struct Probe { Kind kind; ColourGlyph glyph; };
    Probe probe(const juce::Font& font, uint16_t glyph) const
    {
        const float s = kProbeHeight / std::max(1.0f, font.getHeight());
        const juce::Point<float> scale(s, s);
        const auto area = paintArea(font, glyph, scale).expanded(static_cast<int>(kProbeHeight) / 2);
        const auto a = paintGlyph(font, glyph, juce::Colours::white, scale, area, {}, {});
        const auto b = paintGlyph(font, glyph, juce::Colours::black.withAlpha(0.5f), scale, area, {}, {});
        const juce::Image::BitmapData da(a, juce::Image::BitmapData::readOnly), db(b, juce::Image::BitmapData::readOnly);

        uint64_t sumA = 0, sumB = 0;
        int solid = 0, coloured = 0, followed = 0;
        for (int y = 0; y < da.height; ++y)
        {
            const auto* la = reinterpret_cast<const juce::PixelARGB*>(da.getLinePointer(y));
            const auto* lb = reinterpret_cast<const juce::PixelARGB*>(db.getLinePointer(y));
            for (int x = 0; x < da.width; ++x)
            {
                sumA += la[x].getAlpha();
                sumB += lb[x].getAlpha();
                if (la[x].getAlpha() < 128) continue;
                ++solid;
                auto pa = la[x]; pa.unpremultiply();
                if (std::min({ pa.getRed(), pa.getGreen(), pa.getBlue() }) < 224) { ++coloured; continue; }
                auto pb = lb[x]; pb.unpremultiply();
                if (lb[x].getAlpha() >= 32 && std::max({ pb.getRed(), pb.getGreen(), pb.getBlue() }) < 64) ++followed;
            }
        }

        return { sumA == 0 ? Kind::Empty : (coloured == 0 && followed == solid ? Kind::Outline : Kind::Colour),
                 { static_cast<double>(sumB) < 0.75 * static_cast<double>(sumA), followed > 0 } };
    }

    const ColourEntry* rasterizeColour(const ColourGlyph& base, const ColourKey& key, const juce::Font& font)
    {
        const juce::Point<float> scale(static_cast<float>(key.scaleX) / 65536.0f, static_cast<float>(key.scaleY) / 65536.0f);
        const auto colour = base.fillColour ? juce::Colour(key.colour) : juce::Colours::white;
        auto at = [](uint8_t q) { return (static_cast<float>(q) + 0.5f) / 64.0f; };
        const juce::Point<float> run(at(key.runX), at(key.runY)), own(at(key.glyphX), at(key.glyphY)), origin = run + own;

        // A glyph reaching past its box is painted again in one twice the size.
        auto area = paintArea(font, key.glyph.glyphId, scale);
        auto image = paintGlyph(font, key.glyph.glyphId, colour, scale, area, run, own);
        auto covered = coverage(image);
        if (!covered.isEmpty() && !image.getBounds().reduced(1).contains(covered))
        {
            area = area.expanded(area.getWidth() / 2, area.getHeight() / 2);
            image = paintGlyph(font, key.glyph.glyphId, colour, scale, area, run, own);
            covered = coverage(image);
        }

        ColourEntry e;
        if (!covered.isEmpty())
        {
            // The tile: what it covers and a transparent pixel round it (the
            // page is clear there), so filtering never reaches a neighbour.
            const int w = covered.getWidth() + 2, h = covered.getHeight() + 2;
            auto* page = findSpace(colourPages, COLOUR_PAGE_SIZE, w + 1, h + 1, 0u, "jvk colour glyph page");
            if (page == nullptr)
            {
                if (colourPages.size() < kMaxColourPages) return nullptr;   // no texture now: next frame
                colourResetPending_ = true;
                colourOverflowFrame_ = frame_;
                deferred_ = true;
                return nullptr;
            }
            const int px = page->cursorX, py = page->cursorY;
            const juce::Image::BitmapData bd(image, juce::Image::BitmapData::readOnly);
            for (int y = 0; y < covered.getHeight(); ++y)
            {
                const auto* src = reinterpret_cast<const juce::PixelARGB*>(bd.getLinePointer(covered.getY() + y)) + covered.getX();
                auto* dst = page->pixels.data() + static_cast<size_t>(py + 1 + y) * static_cast<size_t>(COLOUR_PAGE_SIZE) + static_cast<size_t>(px + 1);
                for (int x = 0; x < covered.getWidth(); ++x)
                    dst[x] = uint32_t(src[x].getRed()) | (uint32_t(src[x].getGreen()) << 8)
                           | (uint32_t(src[x].getBlue()) << 16) | (uint32_t(src[x].getAlpha()) << 24);
            }
            page->dirty(px, py, w, h);
            page->cursorX = px + w + 1;
            page->shelfHeight = std::max(page->shelfHeight, h + 1);

            const float inv = 1.0f / static_cast<float>(COLOUR_PAGE_SIZE);
            e.u0 = static_cast<float>(px) * inv;
            e.v0 = static_cast<float>(py) * inv;
            e.u1 = static_cast<float>(px + w) * inv;
            e.v1 = static_cast<float>(py + h) * inv;
            e.w = w;
            e.h = h;
            e.page = static_cast<int>(page - colourPages.data());
            e.originX = origin.x - static_cast<float>(area.getX() + covered.getX() - 1);
            e.originY = origin.y - static_cast<float>(area.getY() + covered.getY() - 1);
        }
        return &colourEntries.emplace(key, e).first->second;
    }

    // Every colour tile goes; the pages stay, cleared, written whole again.
    void resetColourPages()
    {
        colourEntries.clear();
        for (auto& page : colourPages)
        {
            std::fill(page.pixels.begin(), page.pixels.end(), 0u);
            page.cursorX = page.cursorY = page.shelfHeight = 0;
            page.needsUpload = true;
            page.uploadedOnce = false;
        }
        colourResetPending_ = false;
    }

    // ---- Packing -------------------------------------------------------------

    AtlasPage* findSpace(std::vector<AtlasPage>& pages, int size, int w, int h, uint32_t clearPixel, const char* label)
    {
        if (w > size || h > size) return nullptr;
        for (auto& page : pages)
        {
            if (page.cursorX + w <= size && page.cursorY + h <= size)
                return &page;
            // Try wrapping to a new shelf — but commit the wrap ONLY if the
            // glyph then fits. The old code mutated cursorY/shelfHeight on
            // pages it went on to reject, so a full page kept being pushed
            // further past its end on every subsequent lookup.
            if (page.cursorX + w > size)
            {
                const int wrappedY = page.cursorY + page.shelfHeight;
                if (wrappedY + h <= size)
                {
                    page.cursorY = wrappedY;
                    page.cursorX = 0;
                    page.shelfHeight = 0;
                    return &page;
                }
            }
        }
        if (&pages == &colourPages && pages.size() >= kMaxColourPages) return nullptr;
        return createPage(pages, size, clearPixel, label);
    }

    AtlasPage* createPage(std::vector<AtlasPage>& pages, int size, uint32_t clearPixel, const char* label)
    {
        if (!backend) return nullptr;

        AtlasPage page;
        page.size = size;
        // An MSDF page starts black (sd=0.0 per channel = "far outside glyph")
        // so linear filtering near quad edges smoothly transitions to
        // transparent; a colour page starts transparent.
        page.pixels.resize(static_cast<size_t>(size) * static_cast<size_t>(size), clearPixel);
        page.dirtyX0 = page.dirtyY0 = size;

        gpu::TextureDesc d;
        d.format = PixelFormat::RGBA8;
        d.width = d.height = static_cast<uint32_t>(size);
        d.label = label;
        page.texture = backend->createTexture(d);
        if (!page.texture)
            return nullptr;

        // Written by stageDirtyPages once painting completes (a whole page the
        // first time), before the frame that samples it.
        page.needsUpload = true;

        pages.push_back(std::move(page));
        return &pages.back();
    }

    struct Alias { juce::Typeface::Ptr typeface, canonical; };
    static constexpr size_t kMaxAliases = 256;
    std::unordered_map<const juce::Typeface*, Alias> aliases_;
    std::map<juce::String, juce::Typeface::Ptr> byName_;
    juce::Typeface::Ptr last_;
    const juce::Typeface* lastCanonical_ = nullptr;

    std::unordered_map<GlyphKey, AtlasEntry, GlyphKeyHash> entries;
    std::unordered_set<GlyphKey, GlyphKeyHash> empties;
    std::unordered_map<GlyphKey, ColourGlyph, GlyphKeyHash> colourGlyphs;
    std::vector<AtlasPage> atlasPages;
    std::unordered_map<ColourKey, ColourEntry, ColourKeyHash> colourEntries;
    std::vector<AtlasPage> colourPages;

    static constexpr int kRasterBudgetPerFrame = 64;
    static constexpr float kProbeHeight = 32.0f;   // device px the probe paints a glyph's font at
    int rasterizedThisFrame_ = 0;
    // Frames staged (worker); record reads it between frames, never during one.
    uint64_t frame_ = 0, colourOverflowFrame_ = 0;
    bool colourResetPending_ = false;
    bool deferred_ = false;   // this frame's record left glyphs for a later frame
    bool colourGlyphSeen_ = false;

    gpu::Backend* backend = nullptr;
    std::vector<uint32_t> rectScratch_;   // a dirty rect on its way to writeTexture (capacity kept)
};

} // jvk
