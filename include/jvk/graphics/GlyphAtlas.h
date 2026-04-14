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
    static constexpr int GLYPH_PADDING = 2;
    static constexpr double MSDF_PIXEL_RANGE = 4.0; // distance range in atlas pixels

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

    void init(VkPhysicalDevice pd, VkDevice d,
              VkCommandPool cp, VkQueue gq,
              core::DescriptorHelper* dh)
    {
        physDevice = pd;
        device = d;
        commandPool = cp;
        graphicsQueue = gq;
        descriptorHelper = dh;
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

    // Stage dirty atlas pages for deferred upload via StagingBelt.
    // Pixels are copied to staging memory; actual GPU transfer commands
    // are recorded in the next frame's prepareFrame() before the render pass.
    void stageDirtyPages(core::StagingBelt& belt, std::vector<core::PendingUpload>& uploads)
    {
        for (auto& page : atlasPages)
        {
            if (page.needsUpload)
            {
                belt.stageImageUpload(page.texture, uploads,
                    page.pixels.data(),
                    static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE));
                page.needsUpload = false;
            }
        }
    }

    void clear()
    {
        entries.clear();
        for (auto& page : atlasPages)
            page.texture.destroy();
        atlasPages.clear();
    }

private:
    struct AtlasPage
    {
        core::Image texture;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        std::vector<uint32_t> pixels;
        int cursorX = 0;
        int cursorY = 0;
        int shelfHeight = 0;
        bool needsUpload = false;
    };

    // Convert a JUCE Path to an msdfgen Shape
    static msdfgen::Shape jucePathToShape(const juce::Path& path)
    {
        msdfgen::Shape shape;
        msdfgen::Contour* contour = nullptr;

        juce::Path::Iterator it(path);
        msdfgen::Point2 lastPoint(0, 0);

        while (it.next())
        {
            switch (it.elementType)
            {
                case juce::Path::Iterator::startNewSubPath:
                    contour = &shape.addContour();
                    lastPoint = msdfgen::Point2(it.x1, it.y1);
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
                    // msdfgen contours auto-close
                    break;
            }
        }

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
        // The Y-flip reverses winding, which inverts inside/outside sign —
        // we compensate in the fragment shader with (0.5 - sd) instead of (sd - 0.5).
        shape.normalize();
        msdfgen::edgeColoringSimple(shape, 3.0);

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

        // Generate MSDF: Projection maps shape coords → pixel coords.
        // Range must be in shape units — convert pixel range by dividing by scale.
        // msdfgen computes distances in shape space, so Range(px/scale) ensures
        // MSDF_PIXEL_RANGE pixels of distance map to the full [0,1] output.
        msdfgen::Projection projection(msdfgen::Vector2(texelScale), translate);
        double rangeInShapeUnits = MSDF_PIXEL_RANGE / texelScale;
        msdfgen::generateMSDF(msdf, shape, projection, msdfgen::Range(rangeInShapeUnits));

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
        if (!device || !descriptorHelper) return nullptr;

        AtlasPage page;
        // Initialize to white (sd=1.0 per channel = "far outside glyph") so that
        // linear filtering near quad edges doesn't bleed opaque border artifacts.
        page.pixels.resize(static_cast<size_t>(ATLAS_SIZE * ATLAS_SIZE), 0xFFFFFFFF);

        page.texture.create({ physDevice, device,
            static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
        page.texture.createSampler(VK_FILTER_LINEAR);

        // Don't upload here — createPage() may be called during command buffer
        // recording (inside paintEntireComponent). Uploading would submit a
        // separate command buffer from the same pool/queue, which conflicts
        // with MoltenVK. Instead, mark as dirty and let uploadDirtyPages()
        // handle it after painting completes.
        page.needsUpload = true;

        page.descriptorSet = descriptorHelper->allocateSet();
        core::DescriptorHelper::writeImage(device, page.descriptorSet, 0,
            page.texture.getView(), page.texture.getSampler());

        atlasPages.push_back(std::move(page));
        return &atlasPages.back();
    }

    std::unordered_map<GlyphKey, AtlasEntry, GlyphKeyHash> entries;
    std::vector<AtlasPage> atlasPages;

    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    core::DescriptorHelper* descriptorHelper = nullptr;
};

} // jvk
