/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Recorder.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// DrawOp — what to draw or what state change to make
// =============================================================================

enum class DrawOp : uint8_t {
    FillRect, FillRectList, FillRoundedRect, FillEllipse,
    StrokeRoundedRect, StrokeEllipse,
    DrawImage, DrawGlyphs, DrawLine,
    FillPath,           // analytical SDF path fill (vger-style)
    DrawShader,
    EffectBlend,
    EffectNoise,        // procedural white-noise overlay (Graphics::drawNoise)
    EffectKernel,
    EffectHSV,          // full-screen HSV scale/delta (saturate, shiftHue, etc.)
    BlurShape,
    BlurPath,          // analytical path SDF blur (fill or stroke ring)
    PushClipRect, PopClipRect,      // scissor-only clips (Emitter-side stack)
    PushClipPath, PopClipPath,      // stencil INCR/DECR via the clip program
    FillTiledImage,                 // tiled image fill (setTiledImageFill)
    ExcludeClipRect,                // stencil-park the rect out of the clip
    RestoreClipExclude,             // un-park it on restoreState
    Dispatch,                       // a compute kernel in paint order (Graphics::dispatch)
    COUNT
};

// =============================================================================
// DrawCommand — one entry in the command vector
// =============================================================================

// Slimmed to what lowering actually reads. Commands lower in recording order
// (JUCE's painter's algorithm order, which is already correct).
struct DrawCommand {
    DrawOp               op;
    uint8_t              stencilDepth;
    uint32_t             dataOffset;
    juce::Rectangle<int> clipBounds;
    // Where a built-in draw lands (Graphics::setTarget): colour attachment
    // 0 = main colour, i + 1 = render target i. Captured at record.
    uint8_t              target = 0;
};

// =============================================================================
// Arena — flat byte allocator for command parameter data
// =============================================================================

class Arena {
public:
    template <typename T>
    uint32_t push(const T& data)
    {
        static_assert(std::is_trivially_copyable_v<T>, "Arena is POD-only. Use the Recorder's side vectors for non-POD types.");
        // At T's alignment: read<T>() hands out a reference into the buffer.
        constexpr uint32_t a = static_cast<uint32_t>(alignof(T));
        uint32_t offset = (head_ + a - 1u) & ~(a - 1u);
        uint32_t needed = static_cast<uint32_t>(sizeof(T));
        ensureCapacity(offset + needed);
        memcpy(buffer_.data() + offset, &data, sizeof(T));
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    uint32_t pushSpan(std::span<const T> data)
    {
        uint32_t offset = head_;
        uint32_t needed = static_cast<uint32_t>(data.size_bytes());
        ensureCapacity(offset + needed);
        if (needed > 0)
            memcpy(buffer_.data() + offset, data.data(), data.size_bytes());
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    const T& read(uint32_t offset) const
    {
        return *reinterpret_cast<const T*>(buffer_.data() + offset);
    }

    template <typename T>
    std::span<const T> readSpan(uint32_t offset, uint32_t count) const
    {
        return { reinterpret_cast<const T*>(buffer_.data() + offset), count };
    }

    // Advance head to the next alignment boundary (e.g., align(4) for float data after uint16 data)
    void align(uint32_t alignment)
    {
        uint32_t mask = alignment - 1;
        head_ = (head_ + mask) & ~mask;
        ensureCapacity(head_);
    }

    void reset() { head_ = 0; }
    uint32_t size() const { return head_; }
    const std::byte* data() const { return buffer_.data(); }

private:
    std::vector<std::byte> buffer_;
    uint32_t head_ = 0;

    void ensureCapacity(uint32_t needed)
    {
        if (needed > buffer_.size())
            buffer_.resize(std::max(needed, static_cast<uint32_t>(buffer_.size() * 2 + 1024)));
    }
};

// =============================================================================
// Recorder — one frame of juce::Graphics calls, as data
//
// The record half of a frame (TRANSLATION_LAYER.md, "The frame"): written by
// jvk::Graphics on the message thread while the frame is painted, read by the
// Renderer's lowering on its worker after submit, reset before the next
// paint. The two never overlap (Renderer::isBusy gates recording). Names no
// GPU: what a command draws with is captured as gpu handles or CPU data.
// =============================================================================

class Recorder {
public:
    // Records a command; false when the current draw target doesn't take it
    // (see recordsInTarget).
    template <typename Params>
    bool push(DrawOp op, const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, const Params& params)
    {
        if (!recordsInTarget(op)) return false;
        uint32_t offset = arena_.push(params);
        commands_.push_back({ op, stencilDepth, offset, clip, recordTarget_ });
        // An effect on a target reads it: the target lives up to here.
        if (recordTarget_ > 0 && recordTarget_ != kNoTarget && isPassEffect(op))
            noteTargetsRead(static_cast<TargetMask>(1u << (recordTarget_ - 1)));
        return true;
    }

    // ---- Draw target (render targets) --------------------------------------
    //
    // Where the draws recorded from now on land: colour attachment 0 (the
    // main colour) or i + 1 (render target i). jvk::Graphics sets
    // this from its saved state, so a target chosen inside one component's
    // paint is restored with the graphics state like a transform or a clip.
    // kNoTarget (a target this frame doesn't have) records no draws at all,
    // so a draw meant for a missing target never lands in the main colour.
    static constexpr uint8_t kNoTarget = 0xFF;
    void setRecordTarget(uint8_t attachment) { recordTarget_ = attachment; }
    uint8_t recordTarget() const { return recordTarget_; }

    // ---- Target lifetimes --------------------------------------------------
    //
    // Targets are cleared at the start of every frame and their only readers
    // are shaders and effects drawn later in the same frame (the final copy
    // reads the main colour alone), so a target is LIVE up to and including its
    // last reader and DEAD after it: nothing written to it then is ever read. The
    // lowering drops that work: draws into a dead target, the clear-under-paint
    // of main-colour draws (Target::clearedByMain), and the load and store of
    // its image at every scene-pass boundary. A frame pays for its targets up
    // to the point it uses them, and a frame that reads none pays nothing.
    // Masks: bit i = target i.
    //
    // The command just recorded samples `targets` (jvk::Graphics::drawShader,
    // for a shader that reads them; push, for an effect on a target).
    void noteTargetsRead(TargetMask targets)
    {
        for (uint32_t i = 0; i < kMaxTargets; ++i)
            if (targets & (1u << i))
                targetReadEnd_[i] = static_cast<uint32_t>(commands_.size());
    }
    // The targets command `index` or a later one reads.
    TargetMask liveTargets(size_t index) const
    {
        TargetMask live = 0;
        for (uint32_t i = 0; i < kMaxTargets; ++i)
            if (index < targetReadEnd_[i])
                live |= static_cast<TargetMask>(1u << i);
        return live;
    }

    // ---- Captures ----------------------------------------------------------
    //
    // Non-POD types go into side vectors; returns the index. Consecutive
    // duplicates dedupe: JUCE paints whole components with one fill/font, so
    // most captures repeat the previous entry — and a gradient FillType copy
    // heap-allocates a fresh ColourGradient every time. One equality check
    // saves an allocation per draw in the common case.
    uint32_t captureFont(const juce::Font& f)
    {
        if (!fonts_.empty() && fonts_.back() == f)
            return static_cast<uint32_t>(fonts_.size() - 1);
        fonts_.push_back(f);
        return static_cast<uint32_t>(fonts_.size() - 1);
    }

    // Where a captured fill's colour comes from, beside juce's FillType: its
    // colour as jvk keeps it (floats, premultiplied: Colour.h), and a tiled
    // image fill's texture, resolved when the draw was recorded
    // (Graphics::captureFill), with its point sampling: what colour mode 3
    // samples. Null for every other fill.
    struct FillColour
    {
        Colour         colour;
        gpu::TextureId texture;
        bool           nearest = false;
        bool operator==(const FillColour&) const = default;
    };

    uint32_t captureFill(const juce::FillType& f, const FillColour& c)
    {
        if (!fills_.empty() && fills_.back() == f && fillColours_.back() == c)
            return static_cast<uint32_t>(fills_.size() - 1);
        fills_.push_back(f);
        fillColours_.push_back(c);
        return static_cast<uint32_t>(fills_.size() - 1);
    }

    const juce::Font&     getFont(uint32_t i) const { return fonts_[i]; }
    const juce::FillType& getFill(uint32_t i) const { return fills_[i]; }
    const FillColour&     getFillColour(uint32_t i) const { return fillColours_[i]; }

    // Append POD data after a command's params (read back at a known offset).
    void arena_align(uint32_t alignment) { arena_.align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { arena_.pushSpan(data); }

    // ---- Path segments -----------------------------------------------------
    //
    // Flattened path segments (16 bytes each: vec4 p0.xy, p1.xy, or the tile
    // and strip tables Graphics builds over them), for the frame's one
    // segment storage buffer: every path fill, path clip and path blur reads
    // its own range of it by index. Returns the first segment's index.
    static constexpr uint32_t kSegmentStride = 16;
    uint32_t pushSegments(const void* segmentData, uint32_t segmentCount)
    {
        const size_t bytes = static_cast<size_t>(segmentCount) * kSegmentStride;
        const uint32_t first = static_cast<uint32_t>(segments_.size() / kSegmentStride);
        auto* src = static_cast<const std::byte*>(segmentData);
        segments_.insert(segments_.end(), src, src + bytes);
        return first;
    }
    std::span<const std::byte> segments() const { return segments_; }

    // ---- The frame ---------------------------------------------------------

    // Per-frame reset: drop the command list, arena, captures and segments
    // (keeping their capacity). `scaledTargets`: the frame's scaled targets,
    // which ordinary paint can't land on.
    void reset(TargetMask scaledTargets)
    {
        commands_.clear();
        arena_.reset();
        fonts_.clear();
        fills_.clear();
        fillColours_.clear();
        segments_.clear();
        recordTarget_ = 0;   // every frame starts drawing into the main colour
        scaledTargets_ = scaledTargets;
        std::fill(std::begin(targetReadEnd_), std::end(targetReadEnd_), 0u);   // no target read yet
    }

    const std::vector<DrawCommand>& commands() const { return commands_; }
    const Arena&                    arena() const    { return arena_; }

    // Clip ops change the stencil, whatever the draw target.
    static bool isClipOp(DrawOp op)
    {
        switch (op) {
            case DrawOp::PushClipRect: case DrawOp::PopClipRect:
            case DrawOp::PushClipPath: case DrawOp::PopClipPath:
            case DrawOp::ExcludeClipRect: case DrawOp::RestoreClipExclude:
                return true;
            default:
                return false;
        }
    }

    // Effects that run as passes of their own: on the main colour or on any target.
    static bool isPassEffect(DrawOp op)
    {
        return op == DrawOp::EffectKernel || op == DrawOp::EffectHSV
            || op == DrawOp::BlurShape || op == DrawOp::BlurPath;
    }

private:
    // Clip ops always record (they keep the stencil balanced whatever the
    // target), and dispatches (they draw into no target). Nothing else
    // records into a missing target. Pass effects run
    // on the main colour or any target; a shader writes what it declares,
    // whatever the target; the blend-mode draws land on the main colour; and
    // ordinary paint on the main colour or a target at scale 1 (a scaled one
    // isn't attached to the scene pass).
    bool recordsInTarget(DrawOp op) const
    {
        if (isClipOp(op) || op == DrawOp::Dispatch)
            return true;
        if (recordTarget_ == kNoTarget)
            return false;
        if (isPassEffect(op) || op == DrawOp::DrawShader)
            return true;
        if (op == DrawOp::EffectBlend || op == DrawOp::EffectNoise)
            return recordTarget_ == 0;
        return recordTarget_ == 0 || (scaledTargets_ & (1u << (recordTarget_ - 1))) == 0;
    }

    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    uint8_t    recordTarget_  = 0;
    TargetMask scaledTargets_ = 0;

    // Target lifetimes (noteTargetsRead): per target, the index of its last
    // reader + 1 this frame (0 = never read).
    uint32_t targetReadEnd_[kMaxTargets] {};

    // Non-POD captures (proper RAII, cleared each frame)
    std::vector<juce::Font>     fonts_;
    std::vector<juce::FillType> fills_;
    std::vector<FillColour>     fillColours_;    // beside fills_, one each

    std::vector<std::byte>      segments_;
};

} // namespace jvk
