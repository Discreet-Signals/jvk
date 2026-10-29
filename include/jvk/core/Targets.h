/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Targets.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// Targets
//
// By default a frame draws into ONE colour target, the main colour. An
// application can declare more, named TARGETS, that the same render pass
// draws into alongside it: a normal buffer, a depth or height buffer,
// material parameters, motion, anything in a format that can be rendered,
// sampled and blended.
//
//   - jvk::Graphics::setTarget(name) selects where the following draws land;
//     every built-in draw (fills, text, images, paths) goes to the selected
//     target exactly as it would to the main colour. The selection is part of
//     the saved graphics state.
//   - A jvk::Shader writes target i by declaring output `layout(location =
//     i + 1)`; location 0 is always the main colour. A SCALED target (below)
//     it may write instead by declaring an output named after it, at any
//     location from 1. Each write combines with what the target holds through
//     the target's Blend.
//   - A jvk::Shader reads targets as inputs by declaring samplers with their
//     names. Drawn inline (Graphics::drawShader), it runs as its own pass at
//     that point in paint order, sees everything drawn into the targets
//     before it, and writes the main colour: lighting, fog, outlines, any
//     "resolve" of the targets is just a draw placed where it belongs.
//
// Targets are cleared to their clear value at the start of every frame, and
// only shaders drawn later in the same frame read them: after a target's
// last reader, the frame does no more work on it (Recorder::liveTargets).
// Declare them with AudioProcessorEditor::setTargets, before the GPU starts:
// they are fixed while it runs. They need per-attachment blending
// (gpu::Caps::independentBlend).
//
// Up to kMaxTargets in all. The ones at scale 1 are the scene passes' colour
// attachments, target i at location i + 1, so there are at most
// kMaxSceneTargets of them (fewer where the device allows fewer colour
// attachments), each among the first that many declared.
// =============================================================================

// Targets as a mask: bit i = target i.
using TargetMask = uint16_t;
inline constexpr size_t kMaxTargets = 16, kMaxSceneTargets = 7;

struct Target
{
    // How a target stores each pixel: a format that can be rendered, sampled
    // and BLENDED (every built-in draw blends, and the paint shaders write
    // floats), so not an integer (UI) or block-compressed (BCn) format.
    using Format = PixelFormat;

    // How a draw's output combines with what a target already holds
    // (s = shader output, d = target, a = s.a).
    enum class Blend
    {
        Over,           // straight alpha:  rgb = s*a + d*(1-a),        a = a + d.a*(1-a)
        Premultiplied,  // premultiplied:   rgb = s + d*(1-a),          a = a + d.a*(1-a)
        Replace,        // rgba = s
        Add,            // rgb = s*a + d,                               a = a + d.a
        Max,            // rgba = max(s, d)
        None            // nothing is written
    };

    juce::Identifier  name;                       // Graphics::setTarget(name); a shader's sampler2D of this name reads it
    Format            format = Format::RGBA8;
    gpu::ClearColor   clear {};                   // its value where nothing drew this frame
    Blend             blend  = Blend::Over;       // how a Shader's write to it combines

    // True: every ordinary draw into the MAIN colour (fills, strokes, paths,
    // images, text) also sets this target to its `clear` value under it,
    // blended by the draw's alpha: rgb = clear.rgb * a + d * (1 - a); the
    // alpha becomes covered (clear alpha 1) or erased (clear alpha 0). The
    // target then always describes the top-most paint at each pixel: data
    // written for something is replaced by the clear value wherever ordinary
    // paint lands over it later (a label, a panel). Draws INTO the target
    // (Graphics::setTarget) are unaffected; effects, Multiply blends and
    // jvk::Shader draws don't clear it.
    bool              clearedByMain = false;

    // Its size relative to the frame's, per side (ceil(frame x scale)). At 1 it is an
    // attachment of the scene passes, as above. A SCALED target (down- or upsampled) is
    // not: jvk::Shader draws write it (by output location, or by an output named after
    // it), each as its own TARGET PASS with it as the attachment, and effects run on it
    // (Graphics::setTarget, then a blur or an hsv transform), both in its own pixels and
    // without path clips (Renderer); ordinary paint doesn't land on it and clearedByMain
    // doesn't apply. It holds its clear value until its first write in a frame.
    float             scale = 1.0f;

    // A FIXED HEIGHT in pixels (0: none), in place of `scale`: the target is this tall at
    // any frame size and keeps the frame's shape (its scale is height / the frame's
    // height), so filling it costs the same in any window, each of its pixels spreading
    // over more of the frame as the window grows. Scaled, as above.
    int               height = 0;

    bool isScaled() const { return scale != 1.0f || height > 0; }

    // Whether it is always the size of `o`: what one target pass can write together.
    bool sameSize(const Target& o) const { return scale == o.scale && height == o.height; }

    bool operator==(const Target& o) const
    {
        return name == o.name && format == o.format && blend == o.blend
            && clearedByMain == o.clearedByMain && sameSize(o) && clear == o.clear;
    }
};

// The colour slots of the scene passes: the main colour, then one per target up to the
// last at scale 1 (slot i + 1 = target i; a scaled target's among them is unused).
inline size_t sceneSlots(const std::vector<Target>& targets)
{
    size_t n = 1;
    for (size_t i = 0; i < targets.size(); ++i)
        if (!targets[i].isScaled()) n = i + 2;
    return n;
}

// The colour slots of a TARGET PASS: byte L = 1 + the target its shader writes at output
// location L (0: none; location 0, the main colour, is never one).
using TargetSlots = uint64_t;
inline int slotTarget(TargetSlots slots, uint32_t location) { return static_cast<int>((slots >> (8 * location)) & 0xFFu) - 1; }
inline TargetMask slotTargets(TargetSlots slots)
{
    TargetMask mask = 0;
    for (uint32_t l = 1; l < 8; ++l)
        if (const int t = slotTarget(slots, l); t >= 0) mask |= static_cast<TargetMask>(1u << t);
    return mask;
}
inline uint32_t slotCount(TargetSlots slots)
{
    uint32_t n = 0;
    for (uint32_t l = 1; l < 8; ++l)
        if (slotTarget(slots, l) >= 0) n = l + 1;
    return n;
}

// How a target blends, as pipeline state.
inline gpu::BlendState targetBlend(Target::Blend mode)
{
    using Blend = Target::Blend;
    switch (mode)
    {
        case Blend::Over:          return gpu::BlendState::over();
        case Blend::Premultiplied: return gpu::BlendState::premultiplied();
        case Blend::Replace:       return gpu::BlendState::replace();
        case Blend::Add:           return gpu::BlendState::add();
        case Blend::Max:           return gpu::BlendState::maximum();
        case Blend::None:          return gpu::BlendState::none();
    }
    return gpu::BlendState::none();
}

// The blend of a draw into the main colour on a target it clears
// (Target::clearedByMain): the draw writes (clear.rgb, its alpha) there.
inline gpu::BlendState clearedByMainBlend(const Target& t)
{
    using F = gpu::BlendFactor;
    gpu::BlendState b = gpu::BlendState::over();
    b.srcColor = F::SrcAlpha;
    b.dstColor = F::OneMinusSrcAlpha;
    b.srcAlpha = t.clear.a >= 0.5f ? F::One : F::Zero;
    b.dstAlpha = F::OneMinusSrcAlpha;
    return b;
}

// The targets a BUILT-IN draw (fills, strokes, paths, images, text) sets to
// their clear values under its paint, as a mask (bit i = target i): PAINT
// into the main colour clears the targets declared clearedByMain that are
// still `live` (read later this frame: Recorder::liveTargets; clearing one
// nothing reads any more is a dead store). `paint`: the program is a paint
// program (ui2d, path: it writes every output location and reads FrameData).
// Every other draw clears none.
inline TargetMask paintClears(const std::vector<Target>& targets, uint8_t attachment,
                              bool paint, TargetMask live)
{
    if (!paint || attachment != 0) return 0;
    TargetMask clears = 0;
    for (size_t i = 0; i < targets.size(); ++i)
        if (targets[i].clearedByMain)
            clears |= static_cast<TargetMask>(1u << i);
    return clears & live;
}

// How a BUILT-IN draw lands in a pass with render targets: the one rule
// every built-in pipeline builds from. Per colour attachment of the scene
// pass (sceneSlots): the draw's `blend` on the attachment it lands on (0 =
// main colour, i + 1 = target i; overwritten, not blended, on a target
// declared Replace), nothing written on the others, except the targets in
// `clears` (paintClears), set to their clear values under the paint.
// Returns whether the paint shader must write the clear values (its
// paintsMain constant).
inline bool drawTargets(const std::vector<Target>& targets, uint8_t attachment,
                        gpu::BlendState blend, TargetMask clears, gpu::PipelineDesc& desc)
{
    const bool paintsMain = attachment == 0 && clears != 0 && blend.writeMask != 0;
    const size_t slots = sceneSlots(targets);
    desc.attachments.colorCount = static_cast<uint8_t>(slots);
    for (size_t i = 0; i < slots; ++i)
        desc.blend[i] = gpu::BlendState::none();

    if (attachment > 0 && attachment <= targets.size()
        && targets[attachment - 1].blend == Target::Blend::Replace)
        blend.enabled = false;
    if (attachment < slots)
        desc.blend[attachment] = blend;
    if (paintsMain)
        for (size_t i = 0; 1 + i < slots; ++i)
            if (clears & (1u << i))
                desc.blend[1 + i] = clearedByMainBlend(targets[i]);
    return paintsMain;
}

} // namespace jvk
