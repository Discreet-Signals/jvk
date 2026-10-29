#pragma once

namespace jvk {

// =============================================================================
// jvk::ImageType / jvk::PixelData — a GPU-resident juce::Image backend
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
//   - The texture is a lazily-created accelerator, refreshed from the
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
// with the GPU device in the D3D role):
//
//   - PixelData holds the shared Context (and so the backend) from the
//     moment its texture exists, so the device outlives the last image —
//     which means a jvk-backed image with STATIC storage duration would
//     hold the whole GPU device until DLL unload. Never do that: hold
//     images through finec::ImageResources / members, exactly like every
//     other image asset. Do not park jvk images in juce::ImageCache (a
//     process-global static holder).
//
//   - ~PixelData never blocks: it hands its texture to the Context's retire
//     list, which destroys it once every frame that named it is submitted
//     (and the backend once the GPU is done).
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
        // Frames already recorded may still name the texture: the Context
        // destroys it once they are submitted, the backend once the GPU is done.
        if (context_ != nullptr && texture_)
            context_->retire(texture_, lastUsed_);
    }

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

    /** Message thread, record time. Ensures the texture exists on the
        Renderer's backend and is current, and returns it for jvk::Graphics
        to capture into the draw. Null on any failure (a texture that can't be
        made, a second backend) — the caller then falls back to the shared
        image cache, which reads the mirror, so the draw still happens. */
    gpu::TextureId prepareForDraw(Renderer& r)
    {
        const auto w = static_cast<uint32_t>(width);
        const auto h = static_cast<uint32_t>(height);
        if (w == 0 || h == 0)
            return {};
        if (context_ != nullptr && context_.get() != &r.context())
            return {};   // drawn on another backend: the cache serves that one

        if (!texture_) {
            gpu::TextureDesc d;
            d.format = PixelFormat::RGBA8;
            d.width = w;
            d.height = h;
            d.label = "jvk image";
            texture_ = r.backend().createTexture(d);
            if (!texture_)
                return {};
            context_ = r.context().shared_from_this();
            dirty_ = true;
        }

        if (dirty_) {
            scratch_.resize(static_cast<size_t>(w) * h * 4);
            ImageSource::writeRGBA(mirror_, scratch_.data());
            if (!r.backend().writeTexture(texture_, 0, 0, { 0, 0, (int32_t) w, (int32_t) h },
                                          std::as_bytes(std::span<const uint8_t>(scratch_))))
                return {};
            dirty_ = false;
        }

        lastUsed_ = r.serial();
        return texture_;
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
    juce::Image              mirror_;          // CPU source of truth (software)
    std::shared_ptr<Context> context_;         // the backend the texture lives on
    gpu::TextureId           texture_;
    uint64_t                 lastUsed_ = 0;    // the record serial of the last frame that drew it
    std::vector<uint8_t>     scratch_;         // the mirror as RGBA, on its way to writeTexture
    bool                     dirty_  = true;

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

    int getTypeID() const override { return imageTypeID; }   // 'jvk1'
};

inline std::unique_ptr<juce::ImageType> PixelData::createType() const
{
    return std::make_unique<ImageType>();
}

} // namespace jvk
