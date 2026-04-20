#pragma once

namespace jvk {

// =============================================================================
// GradientAtlas — one image, N rows of 256-sample color LUTs
//
// Replaces per-gradient VkImage+descriptor-set allocation. All gradients in a
// frame share this one image and one descriptor set; each occupies one row.
// The row index is packed into the vertex `gradientInfo.w` and the shader
// samples the atlas at (t, rowNorm).
//
// Lifecycle per frame:
//   - beginFrame(): cursor = 0, hash→row map cleared.
//   - getRow(g): looks up the gradient by hash; on miss, rasterizes a new row
//     into the CPU mirror and returns cursor++. O(1) per call.
//   - stageUploads(): bulk-copies rows [0, cursor) from CPU mirror to L2
//     staging and queues one GPU copy. Called once before the render pass.
//
// Designed for the degenerate case of thousands of unique, time-varying
// gradients per frame — no descriptor churn, one upload per frame.
// =============================================================================

class GradientAtlas {
public:
    static constexpr uint32_t WIDTH  = 256;   // colour stops per gradient
    static constexpr uint32_t HEIGHT = 4096;  // max unique gradients per frame

    void init(Device& device)
    {
        if (device_) return;
        device_ = &device;
        image_ = Image(device.pool(), device.device(), WIDTH, HEIGHT,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        descriptorSet_ = device.bindings().alloc();
        Memory::M::writeImage(device.device(), descriptorSet_, 0,
            image_.view(), image_.sampler());

        cpuBuffer_.resize(static_cast<size_t>(WIDTH) * HEIGHT * 4);
        hashToRow_.reserve(HEIGHT);
    }

    ~GradientAtlas()
    {
        if (device_ && descriptorSet_ != VK_NULL_HANDLE)
            device_->bindings().free(descriptorSet_);
    }

    void beginFrame()
    {
        cursor_ = 0;
        hashToRow_.clear();
    }

    // Returns normalized row y-coord (row center) for this gradient.
    // Sampling the atlas at (t, rowNorm) yields the gradient's colour at t.
    // On overflow (>= HEIGHT), the last row is reused — visible artifacts are
    // preferable to silent black fills.
    float getRow(const juce::ColourGradient& g, uint64_t hash)
    {
        auto it = hashToRow_.find(hash);
        uint32_t row;
        if (it != hashToRow_.end()) {
            row = it->second;
        } else {
            row = (cursor_ < HEIGHT) ? cursor_++ : (HEIGHT - 1);
            hashToRow_.emplace(hash, row);
            rasterizeRow(g, row);
        }
        return (static_cast<float>(row) + 0.5f) / static_cast<float>(HEIGHT);
    }

    void stageUploads()
    {
        if (cursor_ == 0 || !device_) return;
        VkDeviceSize byteSize = static_cast<VkDeviceSize>(WIDTH) * cursor_ * 4;
        auto staging = device_->staging().alloc(byteSize);
        memcpy(staging.mappedPtr, cpuBuffer_.data(), static_cast<size_t>(byteSize));
        device_->upload(staging, image_.image(), WIDTH, cursor_);
    }

    VkDescriptorSet descriptorSet() const { return descriptorSet_; }
    uint32_t        activeRows()    const { return cursor_; }

private:
    void rasterizeRow(const juce::ColourGradient& g, uint32_t row)
    {
        uint8_t* dst = cpuBuffer_.data() + static_cast<size_t>(row) * WIDTH * 4;
        for (uint32_t i = 0; i < WIDTH; i++) {
            double t = static_cast<double>(i) / static_cast<double>(WIDTH - 1);
            auto c = g.getColourAtPosition(t);
            dst[i * 4 + 0] = c.getRed();
            dst[i * 4 + 1] = c.getGreen();
            dst[i * 4 + 2] = c.getBlue();
            dst[i * 4 + 3] = c.getAlpha();
        }
    }

    Device*         device_        = nullptr;
    Image           image_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    std::vector<uint8_t>              cpuBuffer_;
    std::unordered_map<uint64_t, uint32_t> hashToRow_;
    uint32_t        cursor_ = 0;
};

// =============================================================================
// Generic frame-based LRU cache
// =============================================================================

template <typename K, typename V>
class Cache {
public:
    explicit Cache(uint64_t maxAge = 120) : maxAge_(maxAge) {}

    V* find(const K& key)
    {
        auto it = entries_.find(key);
        if (it == entries_.end()) return nullptr;
        it->second.lastUsedFrame = currentFrame_;
        return &it->second.value;
    }

    V& insert(const K& key, V&& value)
    {
        auto [it, _] = entries_.emplace(key, Entry { std::move(value), currentFrame_ });
        return it->second.value;
    }

    void evict(uint64_t frame)
    {
        currentFrame_ = frame;
        for (auto it = entries_.begin(); it != entries_.end(); ) {
            if (frame - it->second.lastUsedFrame > maxAge_)
                it = entries_.erase(it);
            else
                ++it;
        }
    }

    void clear() { entries_.clear(); }
    bool contains(const K& key) const { return entries_.count(key) > 0; }
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        V        value;
        uint64_t lastUsedFrame = 0;
    };
    std::unordered_map<K, Entry> entries_;
    uint64_t maxAge_;
    uint64_t currentFrame_ = 0;
};


// =============================================================================
// Cached resource types — pair a Resource with its descriptor set
// =============================================================================

struct CachedImage {
    Image           image;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Memory::M*      bindings = nullptr;

    ~CachedImage() { if (bindings && descriptorSet) bindings->free(descriptorSet); }
    CachedImage() = default;
    CachedImage(CachedImage&& o) noexcept
        : image(std::move(o.image)), descriptorSet(o.descriptorSet), bindings(o.bindings)
    { o.descriptorSet = VK_NULL_HANDLE; }
    CachedImage& operator=(CachedImage&& o) noexcept {
        if (this != &o) {
            if (bindings && descriptorSet) bindings->free(descriptorSet);
            image = std::move(o.image); descriptorSet = o.descriptorSet;
            bindings = o.bindings; o.descriptorSet = VK_NULL_HANDLE;
        }
        return *this;
    }
};

struct CachedBuffer {
    Buffer          buffer;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Memory::M*      bindings = nullptr;

    ~CachedBuffer() { if (bindings && descriptorSet) bindings->free(descriptorSet); }
    CachedBuffer() = default;
    CachedBuffer(CachedBuffer&& o) noexcept
        : buffer(std::move(o.buffer)), descriptorSet(o.descriptorSet), bindings(o.bindings)
    { o.descriptorSet = VK_NULL_HANDLE; }
    CachedBuffer& operator=(CachedBuffer&& o) noexcept {
        if (this != &o) {
            if (bindings && descriptorSet) bindings->free(descriptorSet);
            buffer = std::move(o.buffer); descriptorSet = o.descriptorSet;
            bindings = o.bindings; o.descriptorSet = VK_NULL_HANDLE;
        }
        return *this;
    }
};

// =============================================================================
// ResourceCaches — shared across all plugin instances
// Generic GPU resource caches. Does not know about specific resource types.
// =============================================================================

class ResourceCaches {
public:
    explicit ResourceCaches(Device& dev)
        : device_(dev),
          textures_(120)
    {
        createBlackPixel();
        gradientAtlas_.init(dev);
    }

    ~ResourceCaches() = default;

    GradientAtlas&                     gradientAtlas() { return gradientAtlas_; }
    Cache<uint64_t, CachedImage>&      textures()      { return textures_; }

    // Stable 64-bit hash of a juce::Path's geometry. Still used by path
    // de-duplication within the frame (e.g. segment-upload caching) even
    // though the old path-mesh cache is gone.
    static uint64_t hashPath(const juce::Path& p)
    {
        uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint32_t v) {
            h ^= v; h *= 0x100000001b3ULL;
        };
        auto mixFloat = [&](float f) {
            uint32_t bits; memcpy(&bits, &f, sizeof(bits)); mix(bits);
        };
        juce::Path::Iterator it(p);
        while (it.next()) {
            mix(static_cast<uint32_t>(it.elementType));
            switch (it.elementType) {
                case juce::Path::Iterator::startNewSubPath:
                case juce::Path::Iterator::lineTo:
                    mixFloat(it.x1); mixFloat(it.y1); break;
                case juce::Path::Iterator::quadraticTo:
                    mixFloat(it.x1); mixFloat(it.y1);
                    mixFloat(it.x2); mixFloat(it.y2); break;
                case juce::Path::Iterator::cubicTo:
                    mixFloat(it.x1); mixFloat(it.y1);
                    mixFloat(it.x2); mixFloat(it.y2);
                    mixFloat(it.x3); mixFloat(it.y3); break;
                case juce::Path::Iterator::closePath: break;
            }
        }
        return h;
    }

    // Hash a juce::Image by the actual pixel-buffer address of its top-left
    // pixel plus dimensions and line stride. Stable across the ephemeral
    // SubsectionPixelData wrappers that JUCE builds for 9-arg drawImage calls
    // (those wrappers sit on the stack and get reused at the same address,
    // which makes getPixelData().get() useless as a cache key). BitmapData::data
    // for a subsection points into the parent's buffer at the subsection origin,
    // so different source-Y offsets hash differently while identical draws of
    // the same subsection hash the same.
    static uint64_t hashImage(const juce::Image& img)
    {
        if (!img.isValid()) return 0;
        juce::Image::BitmapData bd(img, juce::Image::BitmapData::readOnly);
        uint64_t h = reinterpret_cast<uint64_t>(bd.data);
        h ^= (static_cast<uint64_t>(static_cast<uint32_t>(img.getWidth())) << 32)
           |  static_cast<uint64_t>(static_cast<uint32_t>(img.getHeight()));
        h ^= static_cast<uint64_t>(bd.lineStride) * 0x9e3779b97f4a7c15ULL;
        return h;
    }

    VkDescriptorSet getTexture(uint64_t hash, const juce::Image& img)
    {
        if (auto* cached = textures_.find(hash))
            return cached->descriptorSet;

        auto w = static_cast<uint32_t>(img.getWidth());
        auto h = static_cast<uint32_t>(img.getHeight());

        CachedImage ci;
        ci.image = Image(device_.pool(), device_.device(), w, h,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ci.descriptorSet = device_.bindings().alloc();
        ci.bindings = &device_.bindings();

        Memory::M::writeImage(device_.device(), ci.descriptorSet, 0,
            ci.image.view(), ci.image.sampler());

        // Stage pixel upload (JUCE → RGBA8)
        VkDeviceSize byteSize = static_cast<VkDeviceSize>(w) * h * 4;
        auto staging = device_.staging().alloc(byteSize);
        auto* dst = static_cast<uint8_t*>(staging.mappedPtr);
        juce::Image::BitmapData bmp(img, juce::Image::BitmapData::readOnly);
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                auto c = bmp.getPixelColour(static_cast<int>(x), static_cast<int>(y));
                auto idx = (y * w + x) * 4;
                dst[idx + 0] = c.getRed();
                dst[idx + 1] = c.getGreen();
                dst[idx + 2] = c.getBlue();
                dst[idx + 3] = c.getAlpha();
            }
        }
        device_.upload(staging, ci.image.image(), w, h);

        auto& inserted = textures_.insert(hash, std::move(ci));
        return inserted.descriptorSet;
    }

    static uint64_t hashGradient(const juce::ColourGradient& g)
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

    // Stage a gradient for rendering this frame. Returns the normalized row
    // y-coord (pass via vertex gradientInfo.w). All gradients share the atlas
    // descriptor set (ResourceCaches::gradientDescriptor()), so there is no
    // per-gradient descriptor switch during draws.
    float registerGradient(const juce::ColourGradient& g)
    {
        return gradientAtlas_.getRow(g, hashGradient(g));
    }

    VkDescriptorSet gradientDescriptor() const { return gradientAtlas_.descriptorSet(); }

    VkDescriptorSet defaultDescriptor() const { return blackPixel_.descriptorSet; }
    VkImageView     defaultImageView() const { return blackPixel_.image.view(); }
    VkSampler       defaultSampler()   const { return blackPixel_.image.sampler(); }

    void beginFrame(uint64_t frameId)
    {
        currentFrame_ = frameId;
        textures_.evict(frameId);
        gradientAtlas_.beginFrame();
    }

    Device& device() { return device_; }

private:
    Device& device_;
    GradientAtlas                   gradientAtlas_;
    Cache<uint64_t, CachedImage>    textures_;
    CachedImage                     blackPixel_;
    uint64_t currentFrame_ = 0;

    void createBlackPixel()
    {
        auto& pool = device_.pool();
        auto& m = device_.bindings();
        VkDevice vkd = device_.device();

        blackPixel_.image = Image(pool, vkd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        blackPixel_.descriptorSet = m.alloc();
        blackPixel_.bindings = &m;

        Memory::M::writeImage(vkd, blackPixel_.descriptorSet, 0,
            blackPixel_.image.view(), blackPixel_.image.sampler());

        // Upload 1x1 black pixel
        uint32_t black = 0xFF000000;
        auto staging = device_.staging().alloc(4);
        memcpy(staging.mappedPtr, &black, 4);
        device_.upload(staging, blackPixel_.image.image(), 1, 1);

        // Flush immediately so it's ready before first frame
        device_.submitImmediate([&](VkCommandBuffer cmd) {
            device_.flushUploads(cmd);
        });
    }
};

} // namespace jvk
