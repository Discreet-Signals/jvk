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
 File: PipelineConfig.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

enum class BlendMode
{
    Opaque,
    AlphaBlend,
    Additive,
    Premultiplied,
    Multiply,  // result = src * dst (color grading)
    Screen     // result = 1 - (1-src)*(1-dst) (lighten)
};

struct PipelineConfig
{
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    BlendMode blendMode = BlendMode::AlphaBlend;
    std::vector<VkPushConstantRange> pushConstantRanges;

    // Stencil (front face)
    bool stencilTestEnable = false;
    VkStencilOp stencilFailOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilPassOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilDepthFailOp = VK_STENCIL_OP_KEEP;
    VkCompareOp stencilCompareOp = VK_COMPARE_OP_ALWAYS;
    uint32_t stencilWriteMask = 0xFF;
    uint32_t stencilCompareMask = 0xFF;
    uint32_t stencilReference = 0;

    // Stencil (back face) — if not set, mirrors front face
    bool separateBackStencil = false;
    VkStencilOp stencilBackFailOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilBackPassOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilBackDepthFailOp = VK_STENCIL_OP_KEEP;
    VkCompareOp stencilBackCompareOp = VK_COMPARE_OP_ALWAYS;

    // Color write
    bool colorWriteEnable = true;
};

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
//     i + 1)`; location 0 is always the main colour. Each write combines with
//     what the target holds through the target's Blend.
//   - A jvk::Shader reads targets as inputs by declaring samplers with their
//     names. Drawn inline (Graphics::drawShader), it runs as its own pass at
//     that point in paint order, sees everything drawn into the targets
//     before it, and writes the main colour: lighting, fog, outlines, any
//     "resolve" of the targets is just a draw placed where it belongs.
//
// Targets are cleared to their clear value at the start of every frame, and
// only shaders drawn later in the same frame read them: after a target's
// last reader, the frame does no more work on it (Renderer::liveTargets).
// Declare them with AudioProcessorEditor::setTargets, before Vulkan starts:
// they are fixed while it runs. They need the independentBlend device
// feature (Device::supportsIndependentBlend).
// =============================================================================

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
    VkClearColorValue clear {};                   // its value where nothing drew this frame
    Blend             blend  = Blend::Over;       // how a Shader's write to it combines

    // True: every ordinary draw into the MAIN colour (fills, strokes, paths,
    // images, text) also sets this target to its `clear` value under it,
    // blended by the draw's alpha: rgb = clear.rgb * a + d * (1 - a); the
    // alpha becomes covered (clear alpha 1) or erased (clear alpha 0). The
    // target then always describes the top-most paint at each pixel: data
    // written for something is replaced by the clear value wherever ordinary
    // paint lands over it later (a label, a panel). Draws INTO the target
    // (Graphics::setTarget) are unaffected; effects, Multiply blends and
    // jvk::Shader draws don't clear it. Up to 7 targets.
    bool              clearedByMain = false;

    // Its size relative to the frame's, per side (ceil(frame x scale)). At 1 it is an
    // attachment of the scene passes, as above. A SCALED target (down- or upsampled) is
    // not: only jvk::Shader draws write it (by output location, as any target), each as
    // its own TARGET PASS with it as the attachment (Renderer), in its own pixels and
    // without path clips; ordinary draws can't select it and clearedByMain doesn't apply.
    // It holds its clear value until its first write in a frame.
    float             scale = 1.0f;

    bool isScaled() const { return scale != 1.0f; }

    bool operator==(const Target& o) const
    {
        return name == o.name && format == o.format && blend == o.blend
            && clearedByMain == o.clearedByMain && scale == o.scale
            && std::memcmp(&clear, &o.clear, sizeof(clear)) == 0;
    }
};

// The blend of a draw into the main colour on a target it clears
// (Target::clearedByMain): the draw writes (clear.rgb, its alpha) there.
inline VkPipelineColorBlendAttachmentState clearedByMainAttachment(const Target& t)
{
    VkPipelineColorBlendAttachmentState b {};
    b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                     | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    b.blendEnable = VK_TRUE;
    b.colorBlendOp = VK_BLEND_OP_ADD;
    b.alphaBlendOp = VK_BLEND_OP_ADD;
    b.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    b.srcAlphaBlendFactor = t.clear.float32[3] >= 0.5f ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
    b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    return b;
}

// The fragment specialization of the paint shaders (ui2d.frag, path_sdf.frag):
// constant 0 = 1 for a draw into the main colour (each output location i + 1
// then writes target i's clear rgb with the draw's alpha), 0 for a draw into
// a target (every location writes the colour); constants 1 + 3i .. 3 + 3i =
// target i's clear rgb.
struct PaintSpecialization
{
    int32_t paintsMain = 0;
    float   clear[21] {};
    VkSpecializationMapEntry entries[22] {};
    VkSpecializationInfo     info {};

    PaintSpecialization(bool intoMain, const std::vector<Target>& targets)
    {
        paintsMain = intoMain ? 1 : 0;
        for (size_t i = 0; i < targets.size() && i < 7; ++i)
            for (int c = 0; c < 3; ++c)
                clear[i * 3 + (size_t) c] = targets[i].clear.float32[c];
        entries[0] = { 0, 0, sizeof(int32_t) };
        for (uint32_t k = 0; k < 21; ++k)
            entries[1 + k] = { 1 + k, static_cast<uint32_t>(offsetof(PaintSpecialization, clear) + k * sizeof(float)), sizeof(float) };
        info.mapEntryCount = 22;
        info.pMapEntries = entries;
        info.dataSize = offsetof(PaintSpecialization, entries);
        info.pData = this;
    }
    PaintSpecialization(const PaintSpecialization&) = delete;
};

inline VkPipelineColorBlendAttachmentState blendAttachment(Target::Blend mode)
{
    using Blend = Target::Blend;
    VkPipelineColorBlendAttachmentState b {};
    b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                     | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    b.blendEnable  = VK_TRUE;
    b.colorBlendOp = VK_BLEND_OP_ADD;
    b.alphaBlendOp = VK_BLEND_OP_ADD;
    auto set = [&b](VkBlendFactor sc, VkBlendFactor dc, VkBlendFactor sa, VkBlendFactor da)
    {
        b.srcColorBlendFactor = sc; b.dstColorBlendFactor = dc;
        b.srcAlphaBlendFactor = sa; b.dstAlphaBlendFactor = da;
    };
    switch (mode)
    {
        case Blend::Over:          set(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                       VK_BLEND_FACTOR_ONE,       VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA); break;
        case Blend::Premultiplied: set(VK_BLEND_FACTOR_ONE,       VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                       VK_BLEND_FACTOR_ONE,       VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA); break;
        case Blend::Replace:       b.blendEnable = VK_FALSE; break;
        case Blend::Add:           set(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE,
                                       VK_BLEND_FACTOR_ONE,       VK_BLEND_FACTOR_ONE); break;
        case Blend::Max:           set(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE);
                                   b.colorBlendOp = VK_BLEND_OP_MAX; b.alphaBlendOp = VK_BLEND_OP_MAX; break;
        case Blend::None:          b.blendEnable = VK_FALSE; b.colorWriteMask = 0; break;
    }
    return b;
}

// The targets a BUILT-IN draw (fills, strokes, paths, images, text) sets to
// their clear values under its paint, as a mask (bit i = target i): PAINT
// into the main colour clears the targets declared clearedByMain that are
// still `live` (read later this frame: Renderer::liveTargets; clearing one
// nothing reads any more is a dead store). `paint`: the pipeline's fragment
// shader is a paint shader (ui2d.frag, path_sdf.frag: it writes every output
// location and takes PaintSpecialization). Every other draw clears none.
inline uint8_t paintClears(const std::vector<Target>& targets, uint8_t attachment,
                           bool paint, uint8_t live)
{
    if (!paint || attachment != 0) return 0;
    uint8_t clears = 0;
    for (size_t i = 0; i < targets.size() && i < 8; ++i)
        if (targets[i].clearedByMain)
            clears |= static_cast<uint8_t>(1u << i);
    return clears & live;
}

// How a BUILT-IN draw lands in a pass with render targets: the one rule
// every built-in pipeline builds from (Pipeline, PathPipeline). Per colour
// attachment of the pass: the draw's `blend` on the attachment it lands on
// (0 = main colour, i + 1 = target i; overwritten, not blended, on a target
// declared Replace), nothing written on the others, except the targets in
// `clears` (paintClears), set to their clear values under the paint. Holds
// pointers into itself: build it where the pipeline is created, never copy it.
struct DrawTargets
{
    DrawTargets(const std::vector<Target>& targets, uint8_t attachment,
                VkPipelineColorBlendAttachmentState blend, uint8_t clears)
        : paintsMain(attachment == 0 && clears != 0 && blend.colorWriteMask != 0),
          specialization(paintsMain, targets),
          states(1 + targets.size(), blendAttachment(Target::Blend::None))
    {
        if (attachment > 0 && attachment <= targets.size()
            && targets[attachment - 1].blend == Target::Blend::Replace)
            blend.blendEnable = VK_FALSE;
        if (attachment < states.size())
            states[attachment] = blend;
        if (paintsMain)
            for (size_t i = 0; i < targets.size() && i < 8; ++i)
                if (clears & (1u << i))
                    states[1 + i] = clearedByMainAttachment(targets[i]);

        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = static_cast<uint32_t>(states.size());
        colorBlend.pAttachments = states.data();
    }
    DrawTargets(const DrawTargets&) = delete;

    const bool                                       paintsMain;
    PaintSpecialization                              specialization;   // the fragment stage's
    std::vector<VkPipelineColorBlendAttachmentState> states;
    VkPipelineColorBlendStateCreateInfo              colorBlend {};    // the pipeline's
};

} // jvk
