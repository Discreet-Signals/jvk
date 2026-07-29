#pragma once

namespace jvk {

// =============================================================================
// jvk::ImageType / jvk::PixelData — a Vulkan-resident juce::Image backend
// =============================================================================
//
//     juce::Image img (juce::Image::ARGB, w, h, true, jvk::ImageType{});
//
// behaves like any other juce::Image — full BitmapData access, Graphics
// contexts, clone, ImageType::convert — while keeping a GPU texture that
// jvk::Graphics::drawImage samples ZERO-COPY (no hash, no shared-cache
// lookup, no per-frame upload unless the pixels actually changed).
//
// The design is mirror-first (the model JUCE 8's Direct2D images use for
// device-loss backup, made primary):
//
//   - A SOFTWARE image (the "mirror") is the source of truth. Every CPU
//     surface — BitmapData reads/writes, juce::Graphics painting via
//     createLowLevelContext, clone, convert — operates on the mirror
//     directly, so correctness never depends on the GPU, readbacks never
//     happen, and the image keeps working headless (no editor open, device
//     lost, GPU rendering toggled off in settings).
//
//   - The VkImage is a lazily-created accelerator, refreshed from the
//     mirror only when a draw happens AFTER the content changed (dirty
//     flag). A buffered component (setBufferedToImage) therefore uploads
//     once per repaint and composites zero-copy every frame after. This
//     also fixes a staleness hazard of the shared texture cache: its hash
//     is address-keyed, so repainting an image in place and redrawing it
//     could sample the OLD texture forever; PixelData tracks its own
//     dirty state instead.
//
// Costs stated honestly: pixels exist twice (CPU + GPU — the price of
// headless correctness and instant device-loss recovery), and paint-INTO
// the image runs on the CPU software renderer (a GPU render-to-texture
// context can slot in behind createLowLevelContext later without touching
// this API).
//
// LIFETIME RULES (the Cubase DllMain/D3D capture applies verbatim here,
// with VkDevice in the D3D role):
//
//   - PixelData holds the shared Device (Device::acquire()) from the
//     moment its texture exists, so the device outlives the last image —
//     which means a jvk-backed image with STATIC storage duration would
//     hold the whole Vulkan device until DLL unload. Never do that: hold
//     images through finec::ImageResources / members, exactly like every
//     other image asset. Do not park jvk images in juce::ImageCache (a
//     process-global static holder).
//
//   - Draw-time safety uses the FrameRetained contract on the KEEPER, not
//     the PixelData: drawImage pins the refcounted TextureKeeper for the
//     frame, and ~PixelData just drops its owner ref — never blocks. A
//     texture whose image died mid-flight is orphaned until its frame's
//     fence, then freed by the last unpin (on whichever thread that is).
//
// Threading: prepareForDraw / createLowLevelContext / BitmapData access are
// message-thread only, like all jvk::Graphics record-time surfaces.
// =============================================================================

class PixelData final : public juce::ImagePixelData
{
public:
    PixelData(juce::Image::PixelFormat fmt, int w, int h, bool clear)
        : juce::ImagePixelData(fmt, w, h),
          mirror_(fmt, juce::jmax(1, w), juce::jmax(1, h), clear,
                  juce::SoftwareImageType{})
    {
    }

    ~PixelData() override
    {
        // NON-BLOCKING: the GPU bundle lives in a refcounted TextureKeeper.
        // Dropping the owner ref here never waits — if a draw is still in
        // flight, the keeper's per-frame pins each hold a ref and the last
        // unpin (after that frame's fence) deletes it. This is what makes
        // per-paint create-draw-destroy temporaries (DropShadowEffect,
        // LookAndFeel scratch, buffered-image resizes) stall-free.
        if (keeper_ != nullptr)
            keeper_->release();
    }

    // ---- juce::ImagePixelData ------------------------------------------

    std::unique_ptr<juce::LowLevelGraphicsContext> createLowLevelContext() override
    {
        // Paint on the CPU into the mirror; the texture refreshes on the
        // next draw. Dirty is set NOW (not at context destruction — juce
        // gives no hook there): safe because the upload reads the mirror at
        // draw-record time, which in any sane paint flow is after the
        // context that painted it has been destroyed.
        markDirty();
        return std::make_unique<juce::LowLevelGraphicsSoftwareRenderer>(mirror_);
    }

    void initialiseBitmapData(juce::Image::BitmapData& bd, int x, int y,
                              juce::Image::BitmapData::ReadWriteMode mode) override
    {
        mirror_.getPixelData()->initialiseBitmapData(bd, x, y, mode);
        if (mode != juce::Image::BitmapData::readOnly)
            markDirty();
    }

    juce::ImagePixelData::Ptr clone() override
    {
        auto c = new PixelData(pixelFormat, width, height, false);
        c->mirror_ = mirror_.createCopy();   // deep pixel copy, stays software
        return c;
    }

    std::unique_ptr<juce::ImageType> createType() const override;

    // ---- jvk fast path --------------------------------------------------

    /** Message thread, record time. Ensures the GPU texture exists and is
        current, pins this PixelData for the frame, and returns the
        descriptor jvk::Graphics captures into the DrawImage op. Returns
        VK_NULL_HANDLE on any GPU failure (staging OOM, device lost) — the
        caller falls back to the shared texture cache, which reads the
        mirror, so the draw still happens. */
    VkDescriptorSet prepareForDraw(Renderer& r)
    {
        if (r.device().isLost())
            return VK_NULL_HANDLE;

        auto& dev = r.device();
        const auto w = static_cast<uint32_t>(width);
        const auto h = static_cast<uint32_t>(height);
        if (w == 0 || h == 0)
            return VK_NULL_HANDLE;

        if (keeper_ == nullptr || !keeper_->texture.valid()) {
            auto* k = new TextureKeeper();
            // The keeper holds the shared device until IT dies, so the
            // VkDevice outlives the VkImage even if this PixelData (and
            // every editor) is gone while the last frame drains.
            k->device  = Device::acquire();
            k->texture = Image(dev.pool(), dev.device(), w, h,
                               VK_FORMAT_R8G8B8A8_UNORM,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            if (!k->texture.valid()) {
                k->release();
                return VK_NULL_HANDLE;
            }
            k->descriptorSet = dev.bindings().alloc();
            if (k->descriptorSet == VK_NULL_HANDLE) {
                k->release();
                return VK_NULL_HANDLE;
            }
            k->bindings = &dev.bindings();
            Memory::M::writeImage(dev.device(), k->descriptorSet, 0,
                                  k->texture.view(), k->texture.sampler());
            if (keeper_ != nullptr)
                keeper_->release();
            keeper_ = k;
            dirty_  = true;
        }

        if (dirty_) {
            if (!stageMirror(r, w, h))
                return VK_NULL_HANDLE;   // staging OOM — cache fallback
            dirty_ = false;
        }

        // Pin-and-ref move together: Renderer::retain guarantees exactly one
        // unpin (after this frame's slot fence), and the keeper's unpin
        // override drops the matching ref.
        keeper_->addRef();
        r.retain(keeper_);
        return keeper_->descriptorSet;
    }

private:
    void markDirty()
    {
        dirty_ = true;
        sendDataChangeMessage();
    }

    // Stage the mirror's pixels and queue the copy into this frame's
    // command stream. Same conversion contract as the shared texture cache:
    // ARGB rows swizzle B,G,R,A → R,G,B,A (premultiplied both sides); other
    // formats go through the format-dispatching slow path.
    bool stageMirror(Renderer& r, uint32_t w, uint32_t h)
    {
        const VkDeviceSize byteSize = static_cast<VkDeviceSize>(w) * h * 4;
        auto staging = r.staging().alloc(byteSize);
        if (staging.mappedPtr == nullptr)
            return false;
        auto* dst = static_cast<uint8_t*>(staging.mappedPtr);

        juce::Image::BitmapData bmp(mirror_, juce::Image::BitmapData::readOnly);
        if (mirror_.getFormat() == juce::Image::ARGB) {
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
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    auto c = bmp.getPixelColour(static_cast<int>(x), static_cast<int>(y));
                    auto idx = (static_cast<size_t>(y) * w + x) * 4;
                    dst[idx + 0] = c.getRed();
                    dst[idx + 1] = c.getGreen();
                    dst[idx + 2] = c.getBlue();
                    dst[idx + 3] = c.getAlpha();
                }
            }
        }

        r.upload(staging, keeper_->texture.image(), w, h);
        return true;
    }

    // The GPU bundle, lifetime-decoupled from the PixelData. refs starts at
    // 1 (the owning PixelData); every Renderer::retain adds one, dropped by
    // the drain's unpin after that frame's fence. Deletion happens on
    // whichever thread drops the last ref — safe: vkDestroyImage/VMA are
    // externally-synchronized per object and nothing else touches this
    // bundle by then; Memory::M::free takes its own lock. Never blocks.
    struct TextureKeeper final : FrameRetained {
        std::shared_ptr<Device> device;     // declared first → dies last
        Image                   texture;
        Memory::M*              bindings      = nullptr;
        VkDescriptorSet         descriptorSet = VK_NULL_HANDLE;
        std::atomic<int>        refs { 1 };

        void addRef() noexcept { refs.fetch_add(1, std::memory_order_relaxed); }
        void release() noexcept
        {
            if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
                delete this;
        }
        void unpin() noexcept override
        {
            FrameRetained::unpin();
            release();
        }
        ~TextureKeeper() override
        {
            // refs hit zero → every pin's unpin already ran, so the base
            // dtor's waitUntilUnretained is a no-op single load.
            if (bindings && descriptorSet != VK_NULL_HANDLE)
                bindings->free(descriptorSet);
        }
    };

    juce::Image    mirror_;              // CPU source of truth (software)
    TextureKeeper* keeper_ = nullptr;    // owner ref; released non-blocking
    bool           dirty_  = true;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PixelData)
};

class ImageType final : public juce::ImageType
{
public:
    juce::ImagePixelData::Ptr create(juce::Image::PixelFormat fmt,
                                     int w, int h, bool clear) const override
    {
        return *new PixelData(fmt, w, h, clear);
    }

    int getTypeID() const override { return 0x6A766B31; }   // 'jvk1'
};

inline std::unique_ptr<juce::ImageType> PixelData::createType() const
{
    return std::make_unique<ImageType>();
}

// Record-time helper for jvk::Graphics::drawImage: the zero-copy path for
// jvk-backed images, VK_NULL_HANDLE for everything else (or on GPU
// failure), in which case the caller uses the shared texture cache.
inline VkDescriptorSet tryImageFastPath(const juce::Image& img, Renderer& r)
{
    if (auto* pd = dynamic_cast<PixelData*>(img.getPixelData().get()))
        return pd->prepareForDraw(r);
    return VK_NULL_HANDLE;
}

} // namespace jvk
