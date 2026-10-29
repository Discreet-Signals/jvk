/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: PipelineState.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::gpu
{

// =============================================================================
// Pipeline state as data
//
// Blend factors, stencil ops and attachment formats change pixels, so the
// core decides them; a PipelineDesc says exactly what a draw needs, in terms
// every API shares. The backend turns one into a native pipeline object
// (a Vulkan pipeline, a Metal render pipeline state plus depth/stencil state,
// a WebGPU render pipeline, an eacp RenderPipeline) and keeps it.
//
// The core interns descs: the first time it needs a desc it calls
// Backend::createPipeline and keeps the PipelineId in its own table, so a
// draw carries a 4-byte id and the backend never hashes a desc per frame.
// Only what is BAKED into a pipeline on every API is here; scissor, stencil
// reference and bindings change per draw and travel in gpu::Draw.
// =============================================================================

inline constexpr uint32_t kMaxColorAttachments = 8;   // every API's floor
inline constexpr uint32_t kMaxVertexAttributes = 8;

// ---- Blending ---------------------------------------------------------------
//
// No constant-colour factors and no dual-source blending: eacp has neither,
// and nothing jvk draws needs them.

enum class BlendFactor : uint8_t
{
    Zero, One,
    SrcColor, OneMinusSrcColor,
    SrcAlpha, OneMinusSrcAlpha,
    DstColor, OneMinusDstColor,
    DstAlpha, OneMinusDstAlpha,
};

enum class BlendOp : uint8_t { Add, Subtract, ReverseSubtract, Min, Max };

enum ColorWrite : uint8_t
{
    WriteR = 1 << 0, WriteG = 1 << 1, WriteB = 1 << 2, WriteA = 1 << 3,
    WriteAll = WriteR | WriteG | WriteB | WriteA,
};

struct BlendState
{
    bool        enabled  = false;          // false: the output replaces the attachment
    BlendFactor srcColor = BlendFactor::One;
    BlendFactor dstColor = BlendFactor::Zero;
    BlendOp     colorOp  = BlendOp::Add;
    BlendFactor srcAlpha = BlendFactor::One;
    BlendFactor dstAlpha = BlendFactor::Zero;
    BlendOp     alphaOp  = BlendOp::Add;
    uint8_t     writeMask = WriteAll;

    constexpr bool operator==(const BlendState&) const = default;

    // The blends jvk::Target::Blend names (s = shader output, d = attachment, a = s.a).
    static constexpr BlendState replace() { return {}; }
    static constexpr BlendState none()    { BlendState b; b.writeMask = 0; return b; }
    // rgb = s*a + d*(1-a), a = a + d.a*(1-a)
    static constexpr BlendState over()
    {
        return { true, BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendOp::Add,
                       BlendFactor::One,      BlendFactor::OneMinusSrcAlpha, BlendOp::Add, WriteAll };
    }
    // rgb = s + d*(1-a), a = a + d.a*(1-a)
    static constexpr BlendState premultiplied()
    {
        return { true, BlendFactor::One, BlendFactor::OneMinusSrcAlpha, BlendOp::Add,
                       BlendFactor::One, BlendFactor::OneMinusSrcAlpha, BlendOp::Add, WriteAll };
    }
    // rgb = s*a + d, a = a + d.a
    static constexpr BlendState add()
    {
        return { true, BlendFactor::SrcAlpha, BlendFactor::One, BlendOp::Add,
                       BlendFactor::One,      BlendFactor::One, BlendOp::Add, WriteAll };
    }
    // rgba = max(s, d). Not max(): <windows.h> without NOMINMAX defines a max macro.
    static constexpr BlendState maximum()
    {
        return { true, BlendFactor::One, BlendFactor::One, BlendOp::Max,
                       BlendFactor::One, BlendFactor::One, BlendOp::Max, WriteAll };
    }
};

// ---- Stencil ----------------------------------------------------------------
//
// One state for both faces, and one read and one write mask: what eacp and
// WebGPU offer, and all the clip stack needs (2D geometry has no back faces
// worth treating differently). The reference is dynamic: gpu::Draw::stencilRef.

enum class CompareOp : uint8_t
{
    Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always,
};

enum class StencilOp : uint8_t
{
    Keep, Zero, Replace,
    IncrementClamp, DecrementClamp,
    Invert,
    IncrementWrap, DecrementWrap,
};

struct StencilState
{
    bool      enabled   = false;
    CompareOp compare   = CompareOp::Always;
    StencilOp passOp    = StencilOp::Keep;
    StencilOp failOp    = StencilOp::Keep;
    uint8_t   readMask  = 0xFF;   // ANDed into both sides of the compare
    uint8_t   writeMask = 0xFF;

    constexpr bool operator==(const StencilState&) const = default;
};

// ---- Vertex input -----------------------------------------------------------

enum class VertexFormat : uint8_t { Float, Float2, Float3, Float4, UInt, UByte4Norm };

struct VertexAttribute
{
    uint8_t      location = 0;
    VertexFormat format   = VertexFormat::Float4;
    uint16_t     offset   = 0;   // bytes into the vertex

    constexpr bool operator==(const VertexAttribute&) const = default;
};

// stride 0 = no vertex buffer: the program pulls what it needs (a fullscreen
// triangle from its vertex index, segments from a storage buffer).
struct VertexLayout
{
    uint16_t        stride = 0;
    uint8_t         attributeCount = 0;
    VertexAttribute attributes[kMaxVertexAttributes] {};

    constexpr bool operator==(const VertexLayout&) const = default;
};

enum class Topology : uint8_t { TriangleList, TriangleStrip };

// ---- The pipeline -----------------------------------------------------------

// What a pass's attachments are, which a pipeline is built against.
// colorFormats[i] is attachment i's format; FrameInfo::outputFormat when an
// attachment is the frame's output. Bit i of unusedMask: slot i has no
// attachment (its Pass ColorAttachment's texture is null), so fragment output
// location i goes nowhere; its format is ignored. Location i always feeds slot
// i, which is why a slot can't simply be left out.
struct AttachmentFormats
{
    uint8_t     colorCount = 1;
    PixelFormat colorFormats[kMaxColorAttachments] {};
    uint8_t     unusedMask = 0;
    bool        depthStencil = false;

    constexpr bool operator==(const AttachmentFormats&) const = default;
};

struct PipelineDesc
{
    ProgramId         program;
    // A permutation of the program the shader bundle compiled ahead of time
    // (0 = the default). This is where specialisation constants go: D3D12 and
    // eacp have none, so a variant is a separately compiled shader.
    uint16_t          variant  = 0;
    Topology          topology = Topology::TriangleList;
    VertexLayout      vertexLayout;
    AttachmentFormats attachments;
    BlendState        blend[kMaxColorAttachments] {};   // per colour attachment
    StencilState      stencil;                          // ignored without attachments.depthStencil

    constexpr bool operator==(const PipelineDesc&) const = default;
};

// For the core's intern table (std::unordered_map<PipelineDesc, PipelineId,
// PipelineDescHash>). Field-wise rather than over the raw bytes: padding is
// indeterminate, so two equal descs could hash apart.
struct PipelineDescHash
{
    size_t operator()(const PipelineDesc& d) const noexcept
    {
        uint64_t h = 1469598103934665603ull;                 // FNV-1a offset basis
        auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };

        mix(d.program.value);
        mix(d.variant);
        mix(static_cast<uint64_t>(d.topology));
        mix(d.vertexLayout.stride);
        mix(d.vertexLayout.attributeCount);
        for (uint32_t i = 0; i < d.vertexLayout.attributeCount && i < kMaxVertexAttributes; ++i)
        {
            const auto& a = d.vertexLayout.attributes[i];
            mix((uint64_t(a.location) << 24) | (uint64_t(a.format) << 16) | a.offset);
        }
        mix(d.attachments.colorCount);
        mix(d.attachments.unusedMask);
        mix(d.attachments.depthStencil ? 1u : 0u);
        for (uint32_t i = 0; i < d.attachments.colorCount && i < kMaxColorAttachments; ++i)
        {
            const auto& b = d.blend[i];
            mix(static_cast<uint64_t>(d.attachments.colorFormats[i]));
            mix((uint64_t(b.enabled) << 56)
              | (uint64_t(b.srcColor) << 48) | (uint64_t(b.dstColor) << 40) | (uint64_t(b.colorOp) << 32)
              | (uint64_t(b.srcAlpha) << 24) | (uint64_t(b.dstAlpha) << 16) | (uint64_t(b.alphaOp) << 8)
              | b.writeMask);
        }
        const auto& s = d.stencil;
        mix((uint64_t(s.enabled) << 40) | (uint64_t(s.compare) << 32) | (uint64_t(s.passOp) << 24)
          | (uint64_t(s.failOp) << 16) | (uint64_t(s.readMask) << 8) | s.writeMask);
        return static_cast<size_t>(h);
    }
};

} // namespace jvk::gpu
