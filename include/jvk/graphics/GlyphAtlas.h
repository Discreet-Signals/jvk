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

// GPU MSDF glyph atlas for resolution-independent text rendering.
// Generates multi-channel signed distance fields from JUCE glyph outlines
// via msdfgen, then renders at any size with a simple fragment shader.
// One atlas per typeface serves all font sizes with sharp corners preserved.
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

    struct GlyphKey
    {
        juce::Typeface* typeface;
        uint16_t glyphId;

        bool operator==(const GlyphKey& o) const
        {
            return typeface == o.typeface && glyphId == o.glyphId;
        }
    };

    struct GlyphKeyHash
    {
        size_t operator()(const GlyphKey& k) const
        {
            size_t h = reinterpret_cast<size_t>(k.typeface);
            h ^= std::hash<uint16_t>{}(k.glyphId) + 0x9e3779b9 + (h << 6) + (h >> 2);
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

    void init(Device& dev)
    {
        device = &dev;
    }

    const AtlasEntry* getGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto it = entries.find(key);
        if (it != entries.end())
            return &it->second;
        return rasterizeGlyph(key, font);
    }

    VkDescriptorSet getDescriptorSet(int atlasIndex) const
    {
        if (atlasIndex < 0 || atlasIndex >= static_cast<int>(atlasPages.size()))
            return VK_NULL_HANDLE;
        return atlasPages[static_cast<size_t>(atlasIndex)].descriptorSet;
    }

    // Stage dirty atlas pages for deferred upload via Device staging.
    // Pixels are copied to staging memory; actual GPU transfer commands
    // are recorded in the next frame's flushUploads() before the render pass.
    void stageDirtyPages()
    {
        for (auto& page : atlasPages)
        {
            if (page.needsUpload)
            {
                auto byteSize = static_cast<VkDeviceSize>(ATLAS_SIZE * ATLAS_SIZE * 4);
                auto staging = device->staging().alloc(byteSize);
                memcpy(staging.mappedPtr, page.pixels.data(), static_cast<size_t>(byteSize));
                device->upload(staging, page.texture.image(),
                    static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE));
                page.needsUpload = false;
            }
        }
    }

    void clear()
    {
        entries.clear();
        for (auto& page : atlasPages)
        {
            if (page.descriptorSet != VK_NULL_HANDLE)
                device->bindings().free(page.descriptorSet);
        }
        atlasPages.clear();
    }

private:
    struct AtlasPage
    {
        Image texture;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        std::vector<uint32_t> pixels;
        int cursorX = 0;
        int cursorY = 0;
        int shelfHeight = 0;
        bool needsUpload = false;
    };

    // Convert a JUCE Path to an msdfgen Shape.
    //
    // msdfgen requires CLOSED contours — orientContours() and generateMSDF()
    // both rely on scanline inside/outside tests, which are undefined for
    // open contours. JUCE's Path::Iterator emits `closePath` as a marker
    // only (no coordinates), so we track each subpath's start point and
    // emit an explicit lineTo back to it on closePath. We also close any
    // still-open contour when a new subpath starts or the path ends, which
    // matches JUCE's rendering behaviour for un-terminated subpaths.
    static msdfgen::Shape jucePathToShape(const juce::Path& path)
    {
        msdfgen::Shape shape;
        msdfgen::Contour* contour = nullptr;

        juce::Path::Iterator it(path);
        msdfgen::Point2 lastPoint(0, 0);
        msdfgen::Point2 subpathStart(0, 0);
        bool haveSubpath = false;

        auto closeIfOpen = [&]() {
            if (!contour || !haveSubpath) return;
            // Avoid a zero-length closing edge — msdfgen rejects those.
            if (lastPoint.x != subpathStart.x || lastPoint.y != subpathStart.y)
                contour->addEdge(msdfgen::EdgeHolder(lastPoint, subpathStart));
            haveSubpath = false;
        };

        while (it.next())
        {
            switch (it.elementType)
            {
                case juce::Path::Iterator::startNewSubPath:
                    // A new subpath without an explicit closePath implicitly
                    // terminates the previous one.
                    closeIfOpen();
                    contour = &shape.addContour();
                    lastPoint = msdfgen::Point2(it.x1, it.y1);
                    subpathStart = lastPoint;
                    haveSubpath = true;
                    break;

                case juce::Path::Iterator::lineTo:
                    if (contour)
                    {
                        msdfgen::Point2 p(it.x1, it.y1);
                        contour->addEdge(msdfgen::EdgeHolder(lastPoint, p));
                        lastPoint = p;
                    }
                    break;

                case juce::Path::Iterator::quadraticTo:
                    if (contour)
                    {
                        msdfgen::Point2 c(it.x1, it.y1);
                        msdfgen::Point2 p(it.x2, it.y2);
                        contour->addEdge(msdfgen::EdgeHolder(lastPoint, c, p));
                        lastPoint = p;
                    }
                    break;

                case juce::Path::Iterator::cubicTo:
                    if (contour)
                    {
                        msdfgen::Point2 c1(it.x1, it.y1);
                        msdfgen::Point2 c2(it.x2, it.y2);
                        msdfgen::Point2 p(it.x3, it.y3);
                        contour->addEdge(msdfgen::EdgeHolder(lastPoint, c1, c2, p));
                        lastPoint = p;
                    }
                    break;

                case juce::Path::Iterator::closePath:
                    closeIfOpen();
                    break;
            }
        }

        // Close any trailing subpath that had no explicit closePath marker.
        closeIfOpen();

        return shape;
    }

    const AtlasEntry* rasterizeGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto* typeface = key.typeface;
        if (!typeface) return nullptr;

        // Get glyph outline (normalized to height=1.0 by JUCE)
        juce::Path glyphPath;
        typeface->getOutlineForGlyph(font.getMetricsKind(), key.glyphId, glyphPath);
        if (glyphPath.isEmpty()) return nullptr;

        auto pathBounds = glyphPath.getBounds();
        if (pathBounds.isEmpty()) return nullptr;

        // Convert JUCE path to msdfgen shape
        msdfgen::Shape shape = jucePathToShape(glyphPath);
        if (shape.contours.empty()) return nullptr;

        // JUCE's getOutlineForGlyph applies scale(s, -s) which flips Y.
        // This makes the Y-down shape coords match Vulkan's V-down texture convention,
        // so we leave inverseYAxis=false for correct atlas storage order.
        shape.normalize();

        // Winding normalization. msdfgen's Shape::orientContours() uses a
        // scanline sampled at ~0.618 of each contour's vertical extent — for
        // non-convex single-contour glyphs (e.g. the letter 't') that ray can
        // hit the cross-bar and produce intersections whose dy signs cancel,
        // leaving the contour unreversed. Those glyphs then render inverted
        // relative to every other glyph that did get normalized.
        //
        // Shoelace-based winding() is robust for any closed polygon, so we
        // compute a containment depth per contour (how many other contours'
        // bounding boxes enclose it) and force the expected winding:
        //   even depth (outer / island)   → CCW (winding = +1)
        //   odd  depth (hole  / hole-in-) → CW  (winding = -1)
        // Y-flipped glyph coords from JUCE put "font interior" on the CCW
        // side, matching the fragment shader's `sd > 0.5` inside convention.
        {
            std::vector<msdfgen::Shape::Bounds> bounds;
            bounds.reserve(shape.contours.size());
            for (auto& c : shape.contours) {
                msdfgen::Shape::Bounds b { +INFINITY, +INFINITY, -INFINITY, -INFINITY };
                c.bound(b.l, b.b, b.r, b.t);
                bounds.push_back(b);
            }
            for (size_t i = 0; i < shape.contours.size(); ++i) {
                int depth = 0;
                for (size_t j = 0; j < shape.contours.size(); ++j) {
                    if (i == j) continue;
                    const auto& bi = bounds[i];
                    const auto& bj = bounds[j];
                    if (bj.l <= bi.l && bj.b <= bi.b && bj.r >= bi.r && bj.t >= bi.t)
                        ++depth;
                }
                const int expected = (depth & 1) ? -1 : +1;
                if (shape.contours[i].winding() != expected)
                    shape.contours[i].reverse();
            }
        }

        // edgeColoringByDistance picks edge colours so that the three MSDF
        // channels disagree only in regions where their median still resolves
        // to the correct distance. edgeColoringSimple can miscolour chains of
        // short edges around tight interior features (e.g. the counter of
        // 'e' or the bowl of 'a'), producing pinpoint median inversions.
        msdfgen::edgeColoringByDistance(shape, 3.0);

        // Compute MSDF dimensions — scale glyph to fit in MSDF_GLYPH_SIZE texels
        // with padding for the distance field spread
        float glyphMaxDim = std::max(pathBounds.getWidth(), pathBounds.getHeight());
        if (glyphMaxDim <= 0) return nullptr;

        // Scale so the glyph fits in (MSDF_GLYPH_SIZE - 2*GLYPH_PADDING) texels
        int innerSize = MSDF_GLYPH_SIZE - GLYPH_PADDING * 2;
        double texelScale = static_cast<double>(innerSize) / static_cast<double>(glyphMaxDim);

        // Actual output dimensions based on glyph aspect ratio
        int msdfW = static_cast<int>(std::ceil(pathBounds.getWidth() * texelScale)) + GLYPH_PADDING * 2;
        int msdfH = static_cast<int>(std::ceil(pathBounds.getHeight() * texelScale)) + GLYPH_PADDING * 2;
        msdfW = std::max(msdfW, 4);
        msdfH = std::max(msdfH, 4);

        if (msdfW > ATLAS_SIZE / 2 || msdfH > ATLAS_SIZE / 2) return nullptr;

        // Generate MSDF using msdfgen
        msdfgen::Bitmap<float, 3> msdf(msdfW, msdfH);

        // Transform: scale glyph coordinates to texel coordinates, with padding offset
        msdfgen::Vector2 translate(
            -static_cast<double>(pathBounds.getX()) + GLYPH_PADDING / texelScale,
            -static_cast<double>(pathBounds.getY()) + GLYPH_PADDING / texelScale);

        // Generate MSDF: Projection maps shape coords -> pixel coords.
        // Range must be in shape units — convert pixel range by dividing by scale.
        // msdfgen computes distances in shape space, so Range(px/scale) ensures
        // MSDF_PIXEL_RANGE pixels of distance map to the full [0,1] output.
        msdfgen::Projection projection(msdfgen::Vector2(texelScale), translate);
        double rangeInShapeUnits = MSDF_PIXEL_RANGE / texelScale;

        // Canonical non-Skia pipeline used by msdf-atlas-gen:
        //
        //   1. generateMSDF  — distance per pixel, per-contour combiner
        //   2. distanceSignCorrection — scanline pass under the real non-zero
        //      fill rule; rewrites the sign of any pixel whose combined-
        //      contour distance disagrees with the true inside/outside test.
        //      This is what fixes the pinhole / stripe artifacts inside
        //      counters of glyphs with overlapping same-winding contours
        //      (common in sans-serif — the bowls of e/a, the crossbar of A).
        //   3. msdfErrorCorrection with CHECK_DISTANCE disabled — a fast
        //      median-coherence pass that cleans up any leftover edge-
        //      coloring seams without re-running the exact distance solve.
        //
        // Step 1's built-in error correction is disabled here so the
        // scanlinePass (step 2) owns sign correction. Running both would
        // fight each other — the exact-distance fallback in generateMSDF
        // can't see overlapping contours as filled.
        msdfgen::MSDFGeneratorConfig cfg;
        cfg.errorCorrection.mode = msdfgen::ErrorCorrectionConfig::DISABLED;
        msdfgen::generateMSDF(msdf, shape, projection, msdfgen::Range(rangeInShapeUnits), cfg);

        msdfgen::distanceSignCorrection(msdf, shape, projection, msdfgen::FILL_NONZERO);

        msdfgen::MSDFGeneratorConfig ecCfg;
        ecCfg.errorCorrection.distanceCheckMode = msdfgen::ErrorCorrectionConfig::DO_NOT_CHECK_DISTANCE;
        msdfgen::msdfErrorCorrection(msdf, shape, projection, msdfgen::Range(rangeInShapeUnits), ecCfg);

        // Find space in atlas
        auto* page = findSpace(msdfW + 1, msdfH + 1);
        if (!page) return nullptr;

        int atlasIdx = static_cast<int>(page - atlasPages.data());
        int px = page->cursorX;
        int py = page->cursorY;

        // Copy MSDF to atlas pixels (RGB = distance channels, A = 255)
        for (int y = 0; y < msdfH; y++)
        {
            for (int x = 0; x < msdfW; x++)
            {
                // msdfgen output is float [0,1] after normalization
                // Clamp and convert to 8-bit
                const float* pixel = msdf(x, y);
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
        page->needsUpload = true;

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

        auto [insertIt, _] = entries.emplace(key, entry);
        return &insertIt->second;
    }

    AtlasPage* findSpace(int w, int h)
    {
        for (auto& page : atlasPages)
        {
            if (page.cursorX + w <= ATLAS_SIZE && page.cursorY + h <= ATLAS_SIZE)
                return &page;
            if (page.cursorX + w > ATLAS_SIZE)
            {
                page.cursorY += page.shelfHeight;
                page.cursorX = 0;
                page.shelfHeight = 0;
                if (page.cursorY + h <= ATLAS_SIZE)
                    return &page;
            }
        }
        return createPage();
    }

    AtlasPage* createPage()
    {
        if (!device) return nullptr;

        AtlasPage page;
        // Initialize to black (sd=0.0 per channel = "far outside glyph") so that
        // linear filtering near quad edges smoothly transitions to transparent.
        // Alpha=0xFF is unused by MSDF but keeps the texture valid.
        page.pixels.resize(static_cast<size_t>(ATLAS_SIZE * ATLAS_SIZE), 0xFF000000);

        page.texture = Image(device->pool(), device->device(),
            static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_FILTER_LINEAR);

        if (!page.texture.valid())
            return nullptr;

        // Don't upload here — createPage() may be called during command buffer
        // recording (inside paintEntireComponent). Uploading would submit a
        // separate command buffer from the same pool/queue, which conflicts
        // with MoltenVK. Instead, mark as dirty and let stageDirtyPages()
        // handle it after painting completes.
        page.needsUpload = true;

        page.descriptorSet = device->bindings().alloc();
        if (page.descriptorSet == VK_NULL_HANDLE)
            return nullptr;

        Memory::M::writeImage(device->device(), page.descriptorSet, 0,
            page.texture.view(), page.texture.sampler());

        atlasPages.push_back(std::move(page));
        return &atlasPages.back();
    }

    std::unordered_map<GlyphKey, AtlasEntry, GlyphKeyHash> entries;
    std::vector<AtlasPage> atlasPages;

    Device* device = nullptr;
};

} // jvk
