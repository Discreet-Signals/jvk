#pragma once

namespace jvk {

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
          gradients_(120),
          textures_(120)
    {
        createBlackPixel();
    }

    ~ResourceCaches() = default;

    Cache<uint64_t, CachedImage>& gradients() { return gradients_; }
    Cache<uint64_t, CachedImage>& textures()  { return textures_; }

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

        // Stage pixel upload (JUCE → RGBA)
        VkDeviceSize byteSize = static_cast<VkDeviceSize>(w) * h * 4;
        auto staging = device_.staging().alloc(byteSize);
        auto* dst = static_cast<uint8_t*>(staging.mappedPtr);
        juce::Image::BitmapData bmp(img, juce::Image::BitmapData::readOnly);
        for (uint32_t y = 0; y < h; y++) {
            auto* src = bmp.getLinePointer(static_cast<int>(y));
            for (uint32_t x = 0; x < w; x++) {
                auto c = juce::Colour(static_cast<uint32_t>(
                    *reinterpret_cast<const uint32_t*>(src + x * static_cast<uint32_t>(bmp.pixelStride))));
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

    VkDescriptorSet defaultDescriptor() const { return blackPixel_.descriptorSet; }

    void beginFrame(uint64_t frameId)
    {
        currentFrame_ = frameId;
        gradients_.evict(frameId);
        textures_.evict(frameId);
    }

    Device& device() { return device_; }

private:
    Device& device_;
    Cache<uint64_t, CachedImage> gradients_;
    Cache<uint64_t, CachedImage> textures_;
    CachedImage                  blackPixel_;
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
