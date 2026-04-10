/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: GlyphAtlas.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// GPU SDF glyph atlas for resolution-independent text rendering.
// Generates a signed distance field per glyph at a fixed resolution,
// then renders at any size via the SDF fragment shader (type 4).
// One small atlas per typeface serves all font sizes.
class GlyphAtlas
{
public:
    static constexpr int ATLAS_SIZE = 1024;
    static constexpr int SDF_GLYPH_SIZE = 48;     // fixed render height for SDF generation
    static constexpr int SDF_PADDING = 4;          // padding for distance field spread
    static constexpr float SDF_SPREAD = 6.0f;      // max distance in pixels for SDF encoding

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
        float u0, v0, u1, v1;      // UV coordinates in atlas
        int w, h;                   // SDF glyph dimensions in atlas pixels
        int atlasIndex;             // which atlas page
        // Metrics for positioning (in SDF-space, scale by fontHeight/SDF_GLYPH_SIZE)
        float offsetX, offsetY;     // path bounds origin offset
        float pathW, pathH;         // original path bounds size at SDF_GLYPH_SIZE
    };

    void init(VkPhysicalDevice physDevice, VkDevice device,
              VkCommandPool commandPool, VkQueue graphicsQueue,
              core::DescriptorHelper* descriptorHelper)
    {
        this->physDevice = physDevice;
        this->device = device;
        this->commandPool = commandPool;
        this->graphicsQueue = graphicsQueue;
        this->descriptorHelper = descriptorHelper;
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

    void clear()
    {
        entries.clear();
        for (auto& page : atlasPages)
            page.texture.destroy();
        atlasPages.clear();
    }

    // Upload any dirty atlas pages to GPU before rendering
    void uploadDirtyPages()
    {
        for (auto& page : atlasPages)
        {
            if (page.needsUpload)
            {
                page.texture.upload(physDevice, commandPool, graphicsQueue,
                    page.pixels.data(),
                    static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE),
                    VK_FORMAT_R8G8B8A8_UNORM);
                page.needsUpload = false;
            }
        }
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

    // Generate SDF from a binary image using the dead reckoning algorithm
    static void generateSDF(const std::vector<bool>& bitmap, int bw, int bh,
                            std::vector<float>& sdf, int sw, int sh,
                            int padX, int padY, float spread)
    {
        sdf.resize(static_cast<size_t>(sw * sh), -spread);

        // For each SDF texel, compute distance to nearest edge in the bitmap
        for (int sy = 0; sy < sh; sy++)
        {
            for (int sx = 0; sx < sw; sx++)
            {
                // Map SDF coordinate back to bitmap coordinate
                int bx = sx - padX;
                int by = sy - padY;

                // Check if this texel is inside the glyph
                bool inside = (bx >= 0 && bx < bw && by >= 0 && by < bh)
                              && bitmap[static_cast<size_t>(by * bw + bx)];

                // Brute force nearest edge search within spread radius
                float minDist = spread;
                int searchR = static_cast<int>(spread) + 1;

                for (int dy = -searchR; dy <= searchR; dy++)
                {
                    for (int dx = -searchR; dx <= searchR; dx++)
                    {
                        int nx = bx + dx;
                        int ny = by + dy;
                        bool nInside = (nx >= 0 && nx < bw && ny >= 0 && ny < bh)
                                       && bitmap[static_cast<size_t>(ny * bw + nx)];

                        if (nInside != inside)
                        {
                            float d = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                            minDist = std::min(minDist, d);
                        }
                    }
                }

                // Signed: positive inside, negative outside
                float signedDist = inside ? minDist : -minDist;

                sdf[static_cast<size_t>(sy * sw + sx)] = signedDist;
            }
        }
    }

    const AtlasEntry* rasterizeGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto* typeface = key.typeface;
        if (!typeface) return nullptr;

        // Get glyph outline path
        juce::Path glyphPath;
        typeface->getOutlineForGlyph(font.getMetricsKind(), key.glyphId, glyphPath);
        if (glyphPath.isEmpty()) return nullptr;

        // Scale path to fixed SDF render size
        float sdfHeight = static_cast<float>(SDF_GLYPH_SIZE);
        auto scaledPath = glyphPath;
        scaledPath.applyTransform(juce::AffineTransform::scale(sdfHeight, sdfHeight));

        auto pathBounds = scaledPath.getBounds();
        int bw = static_cast<int>(std::ceil(pathBounds.getWidth())) + 1;
        int bh = static_cast<int>(std::ceil(pathBounds.getHeight())) + 1;

        if (bw <= 0 || bh <= 0) return nullptr;

        // Rasterize to binary bitmap using JUCE software renderer
        juce::Image binaryImg(juce::Image::SingleChannel, bw, bh, true);
        {
            juce::Graphics g(binaryImg);
            g.setColour(juce::Colours::white);
            g.fillPath(scaledPath, juce::AffineTransform::translation(
                -pathBounds.getX(), -pathBounds.getY()));
        }

        // Extract binary bitmap
        std::vector<bool> bitmap(static_cast<size_t>(bw * bh), false);
        {
            juce::Image::BitmapData bmp(binaryImg, juce::Image::BitmapData::readOnly);
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++)
                    bitmap[static_cast<size_t>(y * bw + x)] = (bmp.getPixelColour(x, y).getAlpha() > 127);
        }

        // Generate SDF with padding
        int sw = bw + SDF_PADDING * 2;
        int sh = bh + SDF_PADDING * 2;

        if (sw > ATLAS_SIZE || sh > ATLAS_SIZE) return nullptr;

        std::vector<float> sdf;
        generateSDF(bitmap, bw, bh, sdf, sw, sh, SDF_PADDING, SDF_PADDING, SDF_SPREAD);

        // Find space in atlas
        auto* page = findSpace(sw, sh);
        if (!page) return nullptr;

        int atlasIdx = static_cast<int>(page - atlasPages.data());
        int px = page->cursorX;
        int py = page->cursorY;

        // Write SDF to atlas pixels (distance encoded in alpha channel)
        for (int y = 0; y < sh; y++)
        {
            for (int x = 0; x < sw; x++)
            {
                float d = sdf[static_cast<size_t>(y * sw + x)];
                // Normalize distance to 0-1 range: 0.5 = edge, >0.5 = inside, <0.5 = outside
                float normalized = (d / SDF_SPREAD) * 0.5f + 0.5f;
                uint8_t alpha = static_cast<uint8_t>(juce::jlimit(0.0f, 255.0f, normalized * 255.0f));
                size_t idx = static_cast<size_t>((py + y) * ATLAS_SIZE + (px + x));
                // Store in RGBA with alpha = SDF value, RGB = white (color comes from vertex)
                page->pixels[idx] = 0x00FFFFFF | (static_cast<uint32_t>(alpha) << 24);
            }
        }
        page->needsUpload = true;

        // Advance cursor
        page->cursorX = px + sw + 1;
        page->shelfHeight = std::max(page->shelfHeight, sh + 1);

        // Store entry with positioning metrics
        float invSize = 1.0f / static_cast<float>(ATLAS_SIZE);
        AtlasEntry entry;
        entry.u0 = static_cast<float>(px) * invSize;
        entry.v0 = static_cast<float>(py) * invSize;
        entry.u1 = static_cast<float>(px + sw) * invSize;
        entry.v1 = static_cast<float>(py + sh) * invSize;
        entry.w = sw;
        entry.h = sh;
        entry.atlasIndex = atlasIdx;
        entry.offsetX = pathBounds.getX();
        entry.offsetY = pathBounds.getY();
        entry.pathW = pathBounds.getWidth();
        entry.pathH = pathBounds.getHeight();

        auto [it, _] = entries.emplace(key, entry);
        return &it->second;
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
        page.pixels.resize(static_cast<size_t>(ATLAS_SIZE * ATLAS_SIZE), 0);

        page.texture.create({ physDevice, device,
            static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
        page.texture.createSampler(VK_FILTER_LINEAR);

        page.texture.upload(physDevice, commandPool, graphicsQueue,
            page.pixels.data(),
            static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE),
            VK_FORMAT_R8G8B8A8_UNORM);

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
