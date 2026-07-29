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
        // Typeface::Ptr, not a raw pointer: entries never evict, so a raw
        // key outliving its Typeface meant a recycled allocation address
        // silently served the WRONG glyph outlines forever. Holding the ref
        // also pins the address, keeping pointer-identity hashing valid.
        juce::Typeface::Ptr typeface;
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
            size_t h = reinterpret_cast<size_t>(k.typeface.get());
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

    void init(Device& dev)
    {
        device = &dev;
    }

    // Without this, every editor open/close leaked one descriptor set per
    // atlas page (clear() existed but had no caller), and Memory::M's pool
    // `used` counts climbed monotonically — after ~256 leaked sets every
    // alloc grew a fresh 256-set pool. The owner (ColorPipeline) is
    // destroyed after the editor's vkDeviceWaitIdle, so the GPU is done
    // with the pages.
    ~GlyphAtlas() { clear(); }

    // Lookup + rasterize-on-miss. Called at RECORD time (message thread,
    // Graphics::drawGlyphs) so pages dirty here get staged by the worker's
    // prepare pass BEFORE the frame that samples them.
    const AtlasEntry* getGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto it = entries.find(key);
        if (it != entries.end())
            return &it->second;
        // Budget the expensive MSDF generation per frame so the first paint
        // of a big new font can't hitch the message thread unboundedly —
        // over-budget glyphs rasterize next frame (invisible for one frame,
        // which replay handles by skipping the missing entry).
        if (rasterizedThisFrame_ >= kRasterBudgetPerFrame)
            return nullptr;
        auto* e = rasterizeGlyph(key, font);
        if (e != nullptr) rasterizedThisFrame_++;
        return e;
    }

    // Find-only lookup for REPLAY (worker thread): never rasterizes — a miss
    // means the glyph was over budget this frame and simply isn't drawn.
    // Rasterizing here would recreate the sampled-in-UNDEFINED-layout bug
    // (the upload flush has already run by replay time).
    const AtlasEntry* peekGlyph(const GlyphKey& key) const
    {
        auto it = entries.find(key);
        return it != entries.end() ? &it->second : nullptr;
    }

    VkDescriptorSet getDescriptorSet(int atlasIndex) const
    {
        if (atlasIndex < 0 || atlasIndex >= static_cast<int>(atlasPages.size()))
            return VK_NULL_HANDLE;
        return atlasPages[static_cast<size_t>(atlasIndex)].descriptorSet;
    }

    // Stage dirty atlas pages for deferred upload. Pixels are copied into
    // the Renderer's staging allocator (per-Renderer, not shared via Device)
    // and the copy-to-image command is queued on the Renderer's upload list.
    // flushUploads() at the top of execute() records both into the frame's
    // command buffer before the scene render pass.
    void stageDirtyPages(Renderer& r)
    {
        // New frame — refill the record-side rasterization budget. Runs on
        // the worker, but never concurrently with record (isBusy gate).
        rasterizedThisFrame_ = 0;

        for (auto& page : atlasPages)
        {
            if (!page.needsUpload)
                continue;

            if (!page.uploadedOnce)
            {
                // First upload of a page: full image (transitions the texture
                // out of UNDEFINED and lands the black border everywhere).
                auto byteSize = static_cast<VkDeviceSize>(ATLAS_SIZE * ATLAS_SIZE * 4);
                auto staging = r.staging().alloc(byteSize);
                if (staging.mappedPtr == nullptr) return;   // OOM — retry next frame
                memcpy(staging.mappedPtr, page.pixels.data(), static_cast<size_t>(byteSize));
                r.upload(staging, page.texture.image(),
                    static_cast<uint32_t>(ATLAS_SIZE), static_cast<uint32_t>(ATLAS_SIZE));
                page.uploadedOnce = true;
            }
            else if (page.dirtyX1 > page.dirtyX0 && page.dirtyY1 > page.dirtyY0)
            {
                // Incremental: stage only the dirty rect. The old full-page
                // path cost a 16 MB alloc + memcpy + copy for ONE new glyph.
                const int dw = page.dirtyX1 - page.dirtyX0;
                const int dh = page.dirtyY1 - page.dirtyY0;
                auto byteSize = static_cast<VkDeviceSize>(dw) * dh * 4;
                auto staging = r.staging().alloc(byteSize);
                if (staging.mappedPtr == nullptr) return;
                auto* dst = static_cast<uint8_t*>(staging.mappedPtr);
                for (int y = 0; y < dh; y++)
                    memcpy(dst + static_cast<size_t>(y) * dw * 4,
                           page.pixels.data() + static_cast<size_t>(page.dirtyY0 + y) * ATLAS_SIZE
                                              + page.dirtyX0,
                           static_cast<size_t>(dw) * 4);
                r.uploadRect(staging, page.texture.image(),
                             page.dirtyX0, page.dirtyY0,
                             static_cast<uint32_t>(dw), static_cast<uint32_t>(dh));
            }

            page.needsUpload = false;
            page.dirtyX0 = page.dirtyY0 = ATLAS_SIZE;
            page.dirtyX1 = page.dirtyY1 = 0;
        }
    }

    void clear()
    {
        entries.clear();
        if (device != nullptr)
            for (auto& page : atlasPages)
                if (page.descriptorSet != VK_NULL_HANDLE)
                    device->bindings().free(page.descriptorSet);
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
        bool uploadedOnce = false;
        // Dirty rect accumulated by rasterizeGlyph, consumed (and reset) by
        // stageDirtyPages. Empty when x1 <= x0.
        int dirtyX0 = ATLAS_SIZE, dirtyY0 = ATLAS_SIZE, dirtyX1 = 0, dirtyY1 = 0;
    };

    const AtlasEntry* rasterizeGlyph(const GlyphKey& key, const juce::Font& font)
    {
        auto typeface = key.typeface;
        if (typeface == nullptr) return nullptr;
        juce::ignoreUnused(font);

        // Get glyph outline (normalized to height=1.0 by JUCE). JUCE applies
        // scale(s, -s) internally → Y is down, matching Vulkan's V-down
        // texture storage convention so we leave inverseYAxis=false.
        juce::Path glyphPath;
        typeface->getOutlineForGlyph(
            static_cast<juce::TypefaceMetricsKind>(key.metricsKind),
            key.glyphId, glyphPath);
        if (glyphPath.isEmpty()) return nullptr;

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

        auto result = msdf::rasteriseJucePath(glyphPath, cfg);
        if (!result.valid) return nullptr;

        int msdfW = result.bitmap.width();
        int msdfH = result.bitmap.height();
        if (msdfW > ATLAS_SIZE / 2 || msdfH > ATLAS_SIZE / 2) return nullptr;

        auto pathBounds = result.shapeBounds;
        auto& msdfBitmap = result.bitmap;
        double texelScale = result.texelScale;

        // Find space in atlas
        auto* page = findSpace(msdfW + 1, msdfH + 1);
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
        page->needsUpload = true;
        page->dirtyX0 = std::min(page->dirtyX0, px);
        page->dirtyY0 = std::min(page->dirtyY0, py);
        page->dirtyX1 = std::max(page->dirtyX1, px + msdfW);
        page->dirtyY1 = std::max(page->dirtyY1, py + msdfH);

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
            // Try wrapping to a new shelf — but commit the wrap ONLY if the
            // glyph then fits. The old code mutated cursorY/shelfHeight on
            // pages it went on to reject, so a full page kept being pushed
            // further past its end on every subsequent lookup.
            if (page.cursorX + w > ATLAS_SIZE)
            {
                const int wrappedY = page.cursorY + page.shelfHeight;
                if (wrappedY + h <= ATLAS_SIZE)
                {
                    page.cursorY = wrappedY;
                    page.cursorX = 0;
                    page.shelfHeight = 0;
                    return &page;
                }
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

    static constexpr int kRasterBudgetPerFrame = 64;
    int rasterizedThisFrame_ = 0;

    Device* device = nullptr;
};

} // jvk
