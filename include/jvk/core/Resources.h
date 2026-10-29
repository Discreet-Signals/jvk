/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Resources.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Context;

// =============================================================================
// GradientAtlas — one texture, N rows of 256-sample colour LUTs
//
// All gradients share this one texture; each occupies one row. The row index
// is packed into the vertex `gradientInfo.w` and the shader samples the atlas
// at (t, rowNorm). One per Renderer: mutated on the message thread during
// record (getRow), uploaded from the worker (stageUploads), never both at once
// (Renderer::isBusy).
//
// Rows PERSIST across frames: a gradient's hash covers its full content, so a
// cached row can never go stale — steady-state frames do zero rasterization
// and zero upload. When the atlas fills, it resets wholesale (rare: >4096
// distinct gradients) rather than evicting piecemeal.
// =============================================================================

class GradientAtlas {
public:
    static constexpr uint32_t WIDTH  = 256;   // colour stops per gradient
    static constexpr uint32_t HEIGHT = 4096;  // max unique gradients

    void init(gpu::Backend& backend)
    {
        if (backend_) return;
        backend_ = &backend;
        gpu::TextureDesc d;
        d.format = PixelFormat::RGBA8;
        d.width  = WIDTH;
        d.height = HEIGHT;
        d.label  = "jvk gradient atlas";
        texture_ = backend.createTexture(d);
        cpuBuffer_.resize(static_cast<size_t>(WIDTH) * HEIGHT * 4);
        hashToRow_.reserve(HEIGHT);
    }

    // The texture goes through the Context (deferred until no frame names it).
    gpu::TextureId release() { auto t = texture_; texture_ = {}; return t; }

    // Returns normalized row y-coord (row center) for this gradient.
    // Sampling the atlas at (t, rowNorm) yields the gradient's colour at t.
    // On overflow (>= HEIGHT), the atlas resets and starts refilling; the
    // brief re-rasterization burst beats reusing one row for every
    // overflowed gradient.
    float getRow(const juce::ColourGradient& g, uint64_t hash)
    {
        auto it = hashToRow_.find(hash);
        uint32_t row;
        if (it != hashToRow_.end()) {
            row = it->second;
        } else {
            if (cursor_ >= HEIGHT) {
                hashToRow_.clear();
                cursor_ = 0;
                uploadedRows_ = 0;
            }
            row = cursor_++;
            hashToRow_.emplace(hash, row);
            rasterizeRow(g, row);
        }
        return (static_cast<float>(row) + 0.5f) / static_cast<float>(HEIGHT);
    }

    // Writes the NEW rows (since the last upload) to the texture: a region
    // write of just the added band, not the whole [0, cursor) prefix.
    void stageUploads()
    {
        if (!backend_ || !texture_ || cursor_ <= uploadedRows_) return;

        const uint32_t first = uploadedRows_;
        const uint32_t count = cursor_ - first;
        const size_t bytes = static_cast<size_t>(WIDTH) * count * 4;
        const auto* src = reinterpret_cast<const std::byte*>(cpuBuffer_.data() + static_cast<size_t>(first) * WIDTH * 4);
        if (backend_->writeTexture(texture_, 0, 0,
                gpu::IRect { 0, static_cast<int32_t>(first), static_cast<int32_t>(WIDTH), static_cast<int32_t>(count) },
                { src, bytes }))
            uploadedRows_ = cursor_;
    }

    gpu::TextureId texture()    const { return texture_; }
    uint32_t       activeRows() const { return cursor_; }

    static uint64_t hash(const juce::ColourGradient& g)
    {
        uint64_t h = g.isRadial ? 1ULL : 0ULL;
        h ^= static_cast<uint64_t>(g.getNumColours()) * 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < g.getNumColours(); i++) {
            auto c = g.getColour(i).getARGB();
            h ^= (static_cast<uint64_t>(c) + 0x517cc1b727220a95ULL) + (h << 6) + (h >> 2);
            double pos = g.getColourPosition(i);
            uint64_t posBytes;
            memcpy(&posBytes, &pos, sizeof(posBytes));
            h ^= posBytes * 0x6c62272e07bb0142ULL;
        }
        return h;
    }

private:
    // As juce's ColourGradient::createLookupTable: between two stops the
    // STRAIGHT (non-premultiplied) components interpolate linearly and the
    // result is premultiplied afterwards (here by ui2d.frag, which samples
    // the row straight). Colour::interpolatedWith tweens PREMULTIPLIED
    // pixels instead, so a red@1 → blue@0 gradient read red at its middle
    // where juce paints purple at half alpha.
    void rasterizeRow(const juce::ColourGradient& g, uint32_t row)
    {
        uint8_t* dst = cpuBuffer_.data() + static_cast<size_t>(row) * WIDTH * 4;
        const int stops = g.getNumColours();
        int j = 0;   // the stop at or before t
        for (uint32_t i = 0; i < WIDTH; i++) {
            const double t = static_cast<double>(i) / static_cast<double>(WIDTH - 1);
            while (j + 1 < stops && g.getColourPosition(j + 1) <= t) ++j;
            juce::Colour c;
            if (stops <= 0)          c = juce::Colours::transparentBlack;
            else if (j + 1 >= stops) c = g.getColour(j);
            else {
                const auto p0 = g.getColourPosition(j), p1 = g.getColourPosition(j + 1);
                const auto a = g.getColour(j), b = g.getColour(j + 1);
                const float f = p1 > p0 ? static_cast<float>((t - p0) / (p1 - p0)) : 0.0f;
                auto lerp8 = [f](juce::uint8 x, juce::uint8 y) {
                    return static_cast<juce::uint8>(juce::roundToInt(x + (static_cast<float>(y) - x) * f));
                };
                c = juce::Colour(lerp8(a.getRed(), b.getRed()), lerp8(a.getGreen(), b.getGreen()),
                                 lerp8(a.getBlue(), b.getBlue()), lerp8(a.getAlpha(), b.getAlpha()));
            }
            dst[i * 4 + 0] = c.getRed();
            dst[i * 4 + 1] = c.getGreen();
            dst[i * 4 + 2] = c.getBlue();
            dst[i * 4 + 3] = c.getAlpha();
        }
    }

    gpu::Backend*  backend_ = nullptr;
    gpu::TextureId texture_;
    std::vector<uint8_t>                   cpuBuffer_;
    std::unordered_map<uint64_t, uint32_t> hashToRow_;
    uint32_t cursor_ = 0;
    uint32_t uploadedRows_ = 0;   // rows already written to the texture
};

// =============================================================================
// ImageSource — a CPU image of ANY pixel format, as the image cache takes it.
//
// `key` names its CONTENT: every draw of the same key, in every window, shares
// one upload, kept by the cache's LRU like any texture. `fill` writes its texels
// (tightly packed rows, formatInfo(format) bytes a texel) into `dst`; it runs on
// the message thread inside the call that resolves the source
// (ImageCache::get), and only when the key isn't cached, reading `pixels`,
// which the caller keeps alive for that call.
//
// A juce::Image is one kind (of: RGBA8, premultiplied as juce stores it, keyed by
// its pixel buffer); raw texels another (of: keyed by their address).
//
// A key is an ADDRESS, so the cache HOLDS the memory behind it while it keeps
// the upload (heldImage / heldTexels): memory freed under a cached key would
// hand it to the next buffer allocated there, of the same size, which would
// then draw the old texels (a texture rebuilt at the same size, every time).
// Once nothing else holds that memory, the upload goes (orphaned).
// =============================================================================

// jvk::ImageType's id: a jvk image keeps its own texture, never cached.
inline constexpr int imageTypeID = 0x6A766B31;   // 'jvk1'

struct ImageSource
{
    uint64_t    key    = 0;
    uint32_t    width  = 0, height = 0;
    PixelFormat format = PixelFormat::RGBA8;
    const void* pixels = nullptr;
    void      (*fill)(const ImageSource& self, uint8_t* dst) = nullptr;
    const juce::Image*          image = nullptr;   // a juce::Image's own (of), held while cached
    std::shared_ptr<const void> owner;             // what keeps raw texels (of), held the same way

    bool isValid() const { return key != 0 && width > 0 && height > 0 && fill != nullptr; }

    // A juce::Image's key: the address of its top-left pixel plus its size and
    // line stride. Stable across the ephemeral SubsectionPixelData wrappers JUCE
    // builds for 9-arg drawImage calls (those sit on the stack and are reused at
    // the same address, which makes getPixelData().get() useless as a key).
    // BitmapData::data for a subsection points into the parent's buffer at the
    // subsection origin, so different source offsets key differently while
    // identical draws of the same subsection key the same.
    static uint64_t keyOf(const juce::Image& img)
    {
        if (!img.isValid()) return 0;
        juce::Image::BitmapData bd(img, juce::Image::BitmapData::readOnly);
        uint64_t h = reinterpret_cast<uint64_t>(bd.data);
        h ^= (static_cast<uint64_t>(static_cast<uint32_t>(img.getWidth())) << 32)
           |  static_cast<uint64_t>(static_cast<uint32_t>(img.getHeight()));
        h ^= static_cast<uint64_t>(bd.lineStride) * 0x9e3779b97f4a7c15ULL;
        return h;
    }

    // Writes `image`'s pixels as RGBA8, PREMULTIPLIED, exactly as juce stores
    // them — the sampler has to interpolate premultiplied values or edge texels
    // bleed the undefined colour of their fully-transparent neighbours; ui2d.frag
    // and the paint blend (premultiplied) are premultiplied end-to-end to match.
    static void writeRGBA(const juce::Image& image, uint8_t* dst)
    {
        const uint32_t w = static_cast<uint32_t>(image.getWidth()), h = static_cast<uint32_t>(image.getHeight());
        juce::Image::BitmapData bmp(image, juce::Image::BitmapData::readOnly);
        // Fast path for JUCE ARGB (byte order B,G,R,A on little-endian): row
        // copy + in-register swizzle to the texture's R,G,B,A. Per-pixel
        // getPixelColour was ~1M virtual format-dispatch calls for a 1024²
        // image, on the message thread, inside paint. The slow path agrees on
        // the alpha convention: getPixelColour yields (0,0,0,a) for
        // SingleChannel and a=255 for RGB, both valid premultiplied values.
        if (image.getFormat() == juce::Image::ARGB) {
            for (uint32_t y = 0; y < h; y++) {
                const auto* src = bmp.getLinePointer(static_cast<int>(y));
                auto* out = dst + static_cast<size_t>(y) * w * 4;
                for (uint32_t x = 0; x < w; x++) {
                    out[x * 4 + 0] = src[x * 4 + 2]; // R
                    out[x * 4 + 1] = src[x * 4 + 1]; // G
                    out[x * 4 + 2] = src[x * 4 + 0]; // B
                    out[x * 4 + 3] = src[x * 4 + 3]; // A
                }
            }
        } else {
            for (uint32_t y = 0; y < h; y++)
                for (uint32_t x = 0; x < w; x++) {
                    auto c = bmp.getPixelColour(static_cast<int>(x), static_cast<int>(y));
                    auto idx = (y * w + x) * 4;
                    dst[idx + 0] = c.getRed();
                    dst[idx + 1] = c.getGreen();
                    dst[idx + 2] = c.getBlue();
                    dst[idx + 3] = c.getAlpha();
                }
        }
    }

    static ImageSource of(const juce::Image& img, uint64_t key)
    {
        ImageSource s;
        if (!img.isValid()) return s;
        s.key = key;
        s.width = static_cast<uint32_t>(img.getWidth());
        s.height = static_cast<uint32_t>(img.getHeight());
        s.format = PixelFormat::RGBA8;
        s.pixels = &img;
        s.image = &img;
        s.fill = [](const ImageSource& self, uint8_t* dst)
        {
            writeRGBA(*static_cast<const juce::Image*>(self.pixels), dst);
        };
        return s;
    }
    static ImageSource of(const juce::Image& img) { return of(img, keyOf(img)); }

    // Texels already in `format`, tightly packed (`texels` alive while it resolves),
    // keyed by their address: a buffer that keeps its texels keeps its key. `owner`,
    // what keeps them, is held while the upload is cached; without one, a buffer
    // freed meanwhile can hand its key to the next one at its address.
    static ImageSource of(const void* texels, uint32_t width, uint32_t height, PixelFormat format,
                          std::shared_ptr<const void> owner = {})
    {
        ImageSource s;
        if (texels == nullptr || width == 0 || height == 0) return s;
        s.owner = std::move(owner);
        s.key = reinterpret_cast<uint64_t>(texels) ^ ((static_cast<uint64_t>(width) << 32) | height)
              ^ (static_cast<uint64_t>(format) + 1) * 0x9e3779b97f4a7c15ULL;
        s.width = width;
        s.height = height;
        s.format = format;
        s.pixels = texels;
        s.fill = [](const ImageSource& self, uint8_t* dst)
        {
            std::memcpy(dst, self.pixels, static_cast<size_t>(formatInfo(self.format).bytesFor(self.width, self.height)));
        };
        return s;
    }
};

// =============================================================================
// ImageCache — every window's uploads of CPU images, shared (Context)
//
// Message thread only (record: Graphics::drawImage, a Shader's per-draw
// images), so it never races: JUCE serializes the message thread. An entry
// used by a frame is stamped with that frame's record serial; an entry not
// used for kMaxAge serials, or orphaned, is retired, and the Context destroys
// its texture once no unsubmitted frame can name it. No pinning, no waits.
// =============================================================================

class ImageCache
{
public:
    static constexpr uint64_t kMaxAge = 120;

    explicit ImageCache(Context& context) : context_(context) {}
    ~ImageCache();

    // The texture for `src`, uploaded on a miss. Null when it can't be made.
    gpu::TextureId get(const ImageSource& src, uint64_t serial);
    gpu::TextureId get(const juce::Image& img, uint64_t serial) { return get(ImageSource::of(img), serial); }

    // Retires stale and orphaned entries; every 16 serials (the sweep walks the
    // whole map).
    void sweep(uint64_t serial);

private:
    struct Entry
    {
        gpu::TextureId              texture;
        uint64_t                    lastUsed = 0;
        juce::Image                 heldImage;
        std::shared_ptr<const void> heldTexels;

        // Nothing but this entry holds what it came from: nobody can draw it
        // again, so the next sweep frees it.
        bool orphaned() const
        {
            if (heldImage.isValid()) return heldImage.getReferenceCount() <= 1;
            return heldTexels != nullptr && heldTexels.use_count() <= 1;
        }
    };

    Context& context_;
    std::unordered_map<uint64_t, Entry> entries_;
    std::vector<std::byte>              scratch_;   // texels on their way to writeTexture (capacity kept)
    uint64_t lastSweep_ = 0;
};

} // namespace jvk
