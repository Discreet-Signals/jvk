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

    // Push this frame's new rows to the GPU. Both the staging allocation
    // and the upload-queue entry come from the Renderer (per-Renderer so
    // two editors' workers never share either structure). The copy command
    // is recorded into the frame's command buffer by flushUploads().
    void stageUploads(Renderer& r)
    {
        if (cursor_ == 0 || !device_) return;
        VkDeviceSize byteSize = static_cast<VkDeviceSize>(WIDTH) * cursor_ * 4;
        auto staging = r.staging().alloc(byteSize);
        memcpy(staging.mappedPtr, cpuBuffer_.data(), static_cast<size_t>(byteSize));
        r.upload(staging, image_.image(), WIDTH, cursor_);
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
//
// Entries are stored as `unique_ptr<V>` so they have stable addresses and V
// itself does not need to be movable — important because cached resource
// types inherit FrameRetained (which carries a std::atomic and is therefore
// non-movable) to block cross-instance destruction races.
//
// LRU uses a frame counter that the OWNER of the cache advances. For the
// shared `ResourceCaches::textures_` that counter is the process-wide
// monotonic tick from AudioProcessorEditor — otherwise two editors with
// different per-instance counters corrupt the eviction math. See
// `AudioProcessorEditor::nextGlobalFrameTick()`.
// =============================================================================

template <typename K, typename V>
class Cache {
public:
    explicit Cache(uint64_t maxAge = 120) : maxAge_(maxAge) {}

    // Every mutator (find, insert, evict) takes lock_. Even though the
    // intended use for `ResourceCaches::textures_` is message-thread-only
    // today, the lock makes the cache safe to share with any worker-thread
    // caller that might be added later and keeps concurrent access from
    // cross-instance forceDrainAll paths harmless.
    V* find(const K& key)
    {
        const juce::ScopedLock lk(lock_);
        auto it = entries_.find(key);
        if (it == entries_.end()) return nullptr;
        it->second.lastUsedFrame = currentFrame_;
        return it->second.value.get();
    }

    // Combined find+pin under a single lock acquisition. Use this instead of
    // `find` followed by a separate `retain` when V inherits FrameRetained
    // and concurrent eviction (on any thread) could otherwise destroy the
    // entry between the find's lock release and the caller pinning it.
    //
    // Returns nullptr on miss. On hit, the entry's pin count is incremented
    // before the lock is released; caller is responsible for pushing the
    // returned object into whatever unpin mechanism will eventually balance
    // it (Renderer::recordingRetains_ via retain(), etc.).
    template <typename PinFn>
    V* findAndPin(const K& key, PinFn&& pinFn)
    {
        static_assert(std::is_base_of_v<FrameRetained, V>,
            "Cache::findAndPin requires V to inherit FrameRetained");
        const juce::ScopedLock lk(lock_);
        auto it = entries_.find(key);
        if (it == entries_.end()) return nullptr;
        it->second.lastUsedFrame = currentFrame_;
        V* v = it->second.value.get();
        pinFn(v);   // runs under lock — evict can't erase this entry here
        return v;
    }

    V& insert(const K& key, std::unique_ptr<V> value)
    {
        const juce::ScopedLock lk(lock_);
        auto [it, _] = entries_.emplace(key, Entry { std::move(value), currentFrame_ });
        return *it->second.value;
    }

    // Combined insert+pin, same rationale as findAndPin.
    template <typename PinFn>
    V& insertAndPin(const K& key, std::unique_ptr<V> value, PinFn&& pinFn)
    {
        static_assert(std::is_base_of_v<FrameRetained, V>,
            "Cache::insertAndPin requires V to inherit FrameRetained");
        const juce::ScopedLock lk(lock_);
        auto [it, _] = entries_.emplace(key, Entry { std::move(value), currentFrame_ });
        V* v = it->second.value.get();
        pinFn(v);
        return *v;
    }

    void evict(uint64_t frame)
    {
        const juce::ScopedLock lk(lock_);
        currentFrame_ = frame;
        for (auto it = entries_.begin(); it != entries_.end(); ) {
            // Only destroy entries that (a) are past the age threshold and
            // (b) no Renderer currently pins. Skipping pinned entries keeps
            // this path non-blocking — if the same entry is still stale on a
            // later frame, it'll be caught then. Prevents a multi-editor
            // session from stalling the message thread on every eviction.
            //
            // If V doesn't inherit FrameRetained (e.g. pure POD caches we
            // might add later), the `if constexpr` skips the pin check at
            // compile time.
            bool stale = (frame - it->second.lastUsedFrame) > maxAge_;
            bool pinned = false;
            if constexpr (std::is_base_of_v<FrameRetained, V>) {
                pinned = it->second.value && it->second.value->isPinned();
            }
            if (stale && !pinned)
                it = entries_.erase(it);
            else
                ++it;
        }
    }

    void clear()
    {
        const juce::ScopedLock lk(lock_);
        entries_.clear();
    }
    bool contains(const K& key) const
    {
        const juce::ScopedLock lk(lock_);
        return entries_.count(key) > 0;
    }
    size_t size() const
    {
        const juce::ScopedLock lk(lock_);
        return entries_.size();
    }

private:
    struct Entry {
        std::unique_ptr<V> value;
        uint64_t           lastUsedFrame = 0;
    };
    std::unordered_map<K, Entry> entries_;
    uint64_t maxAge_;
    uint64_t currentFrame_ = 0;
    mutable juce::CriticalSection lock_;
};


// =============================================================================
// Cached resource types — pair a Resource with its descriptor set
// =============================================================================

// Inherits FrameRetained so every Renderer that reads this entry during
// record can pin it via Renderer::retain, and eviction (which runs on one
// editor's message thread) blocks on any other editor's in-flight pin
// before destroying the VkImage / descriptor. Inherited atomic makes this
// type non-movable — Cache stores unique_ptr<CachedImage> for stable
// addresses, and construction is heap-allocated at getTexture().
struct CachedImage : public FrameRetained {
    Image           image;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Memory::M*      bindings      = nullptr;

    CachedImage() = default;
    ~CachedImage() override
    {
        // MUST run before ~Image / descriptor free. If any Renderer has an
        // outstanding pin (recorded draw still in flight, upload-queue entry
        // not yet flushed), this drives Renderer::forceDrainAll() until all
        // pins drop — then the VkImage + descriptor are safe to destroy.
        waitUntilUnretained();
        if (bindings && descriptorSet) bindings->free(descriptorSet);
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
// ResourceCaches — shared across all plugin instances. Holds state that is
// genuinely shareable because it is mutated only from the (single) JUCE
// message thread: the texture cache (image-dedup across editors) and the
// black-pixel default descriptor. Per-frame mutable structures (GradientAtlas,
// upload queue, deletion queues) live on each Renderer instead.
// =============================================================================

class ResourceCaches {
public:
    explicit ResourceCaches(Device& dev)
        : device_(dev),
          textures_(120)
    {
        createBlackPixel();
    }

    ~ResourceCaches() = default;

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

    // The shared texture cache. Called from the message thread during
    // record (Graphics::drawImage), so inserts/lookups never race another
    // editor — JUCE serializes the message thread.
    //
    // Per-Renderer safety against eviction: every access (hit or miss)
    // pins the entry via Renderer::retain so another editor's message
    // thread can't destroy the VkImage while THIS editor has an in-flight
    // upload or a recorded draw referencing it. The pin is released when
    // the pinning Renderer's slot rolls around and the fence has signalled
    // — i.e. when the GPU provably no longer reads the image.
    //
    // Cross-instance eviction: if ~CachedImage fires on editor A's beginFrame
    // while editor B still has a pin, FrameRetained blocks and forceDrainAll
    // idles every Renderer + vkDeviceWaitIdle's before the image is freed.
    VkDescriptorSet getTexture(uint64_t hash, const juce::Image& img, Renderer& r)
    {
        // Hit path: atomic find+pin. Pin is added while still holding the
        // cache lock, so no concurrent eviction can free this entry between
        // the map lookup and the retain.
        if (auto* cached = textures_.findAndPin(hash,
                [&](CachedImage* v) { r.retain(v); }))
        {
            return cached->descriptorSet;
        }

        auto w = static_cast<uint32_t>(img.getWidth());
        auto h = static_cast<uint32_t>(img.getHeight());

        auto ci = std::make_unique<CachedImage>();
        ci->image = Image(device_.pool(), device_.device(), w, h,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ci->descriptorSet = device_.bindings().alloc();
        ci->bindings = &device_.bindings();

        Memory::M::writeImage(device_.device(), ci->descriptorSet, 0,
            ci->image.view(), ci->image.sampler());

        // Stage pixel upload (JUCE → RGBA8) into the Renderer's per-instance
        // staging allocator — keeps all per-frame staging off the shared
        // Device L2.
        VkDeviceSize byteSize = static_cast<VkDeviceSize>(w) * h * 4;
        auto staging = r.staging().alloc(byteSize);
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

        // Snapshot the VkImage handle from the freshly-created Image while
        // `ci` still owns it. We need this handle for the upload queue entry
        // that records into the worker's command buffer — capturing it here
        // keeps the code correct even if some future refactor changes the
        // order of upload() vs insert().
        VkImage dstImage = ci->image.image();

        // Miss path: atomic insert+pin. Pinning inside the Cache lock
        // guarantees no other thread can observe the entry as unpinned
        // before this Renderer has a pin — so the eviction skip-if-pinned
        // rule covers this entry from the moment it's visible.
        auto& inserted = textures_.insertAndPin(hash, std::move(ci),
                [&](CachedImage* v) { r.retain(v); });

        // Queue the pixel upload into the Renderer's command buffer. Order
        // relative to insertAndPin doesn't matter for correctness anymore —
        // the entry is pinned from the moment it enters the cache, and the
        // pin outlives the worker's flushUploads + the frame's GPU fence.
        r.upload(staging, dstImage, w, h);

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

    VkDescriptorSet defaultDescriptor() const { return blackPixel_.descriptorSet; }
    VkImageView     defaultImageView() const { return blackPixel_.image.view(); }
    VkSampler       defaultSampler()   const { return blackPixel_.image.sampler(); }

    void beginFrame(uint64_t frameId)
    {
        currentFrame_ = frameId;
        textures_.evict(frameId);
    }

    Device& device() { return device_; }

private:
    Device& device_;
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

        // Upload 1x1 black pixel via a one-shot command buffer. Runs during
        // Device setup (before any Renderer exists), so it can't use the
        // Renderer-owned upload queue — record the copy directly using
        // Renderer's static helper so the barrier sequence stays in one place.
        uint32_t black = 0xFF000000;
        auto staging = device_.staging().alloc(4);
        memcpy(staging.mappedPtr, &black, 4);
        VkImage dst = blackPixel_.image.image();
        device_.submitImmediate([&](VkCommandBuffer cmd) {
            Renderer::recordImageUpload(cmd, staging, dst, 1, 1);
        });
    }
};

} // namespace jvk
