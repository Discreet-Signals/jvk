/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: FramePacket.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::gpu
{

// =============================================================================
// FramePacket — one frame, fully decided, handed to a backend
//
// The core records a frame (juce::Graphics calls), lowers it (CPU geometry,
// pass planning, interned pipelines) and hands the backend the result: a list
// of passes, each a run of draws. The backend's job is to execute it
// faithfully: begin each pass with the loads and clears it names, bind and
// draw in order, end it with the stores it names, and present
// FrameInfo::output when the surface has one. It never reorders, merges or
// drops draws, and it never needs to know what a draw is for (a blur is
// passes of fullscreen draws of the blur program; the backend can't tell).
//
// Texture writes are not in the packet: Backend::writeTexture runs them before
// the next frame submitted on ANY surface, which is what lets several surfaces
// share one image cache safely.
//
// What the backend works out for itself: barriers and layout transitions
// (from each texture's use in consecutive passes), native render passes or
// framebuffers or encoders, descriptor sets or bind groups or argument
// buffers, and where the three data streams below live in GPU memory.
//
// The core guarantees:
//   - no texture is both sampled and an attachment in one pass (a resource
//     read and written together is versioned through a second texture);
//   - every PipelineId was created by this backend, with attachment formats
//     matching the pass its draws are in;
//   - every Range lies inside its stream; a draw's vertex range starts at a
//     multiple of its pipeline's vertex stride, its storage ranges at
//     multiples of Caps::storageAlignment; constants fit the program's
//     ProgramLayout::constantsBytes.
//
// Memory: every span points into the core's per-frame arenas, valid until
// Surface::submit() returns. The backend copies what it needs into its own
// transient memory; it never writes to the packet.
// =============================================================================

inline constexpr uint32_t kMaxDrawTextures = 8;   // eacp's codegen limit; every API allows more
inline constexpr uint32_t kMaxDrawStorage  = 4;

// A byte range in one of the packet's data streams.
struct Range
{
    uint32_t offset = 0, size = 0;

    constexpr bool isEmpty() const { return size == 0; }
    constexpr bool operator==(const Range&) const = default;
};

enum class LoadOp : uint8_t
{
    Load,       // keep what the attachment holds
    Clear,      // to the attachment's clear value
    DontCare,   // every pixel the pass reads is written first (a fullscreen effect)
};

enum class StoreOp : uint8_t
{
    Store,
    DontCare,   // nothing reads it after this pass (a dead render target, a transient stencil)
};

struct ColorAttachment
{
    TextureId  texture;      // FrameInfo::output for the frame's own output; null: an unused
                             // slot (AttachmentFormats::unusedMask), load/store ignored
    LoadOp     load  = LoadOp::Load;
    StoreOp    store = StoreOp::Store;
    ClearColor clear;
};

struct DepthStencilAttachment
{
    TextureId texture;       // null: the pass has none (and its pipelines no stencil)
    LoadOp    load  = LoadOp::Load;
    StoreOp   store = StoreOp::Store;
    uint8_t   clearStencil = 0;
};

struct TextureBinding
{
    TextureId   texture;
    SamplerDesc sampler;

    constexpr bool operator==(const TextureBinding&) const = default;
};

// One draw. Slot i of `textures` and `storage` is binding slot i in the
// program's layout (TRANSLATION_LAYER.md, "Binding convention"); a backend maps
// slots to its own binding model.
struct Draw
{
    PipelineId pipeline;
    IRect      scissor;                 // already clamped to the pass's render area
    uint8_t    stencilRef = 0;          // the pipeline's StencilState compares against this

    uint8_t        textureCount = 0;
    uint8_t        storageCount = 0;
    TextureBinding textures[kMaxDrawTextures] {};
    Range          storage[kMaxDrawStorage] {};   // in storageData

    Range    constants;                 // per-draw constants, in constantData
    Range    vertices;                  // in vertexData; empty when the program pulls its own
    uint32_t vertexCount   = 0;
    uint32_t instanceCount = 1;
};

// The viewport of every draw in a pass is the whole of its attachments
// (gl_FragCoord is attachment pixels); the render area and scissors bound
// what is touched.
struct Pass
{
    uint8_t                colorCount = 0;
    ColorAttachment        color[kMaxColorAttachments] {};
    DepthStencilAttachment depthStencil;
    IRect                  renderArea;  // pixels outside it are neither loaded nor stored
    uint32_t               firstDraw = 0, drawCount = 0;   // into FramePacket::draws
    const char*            label = nullptr;
};

struct FramePacket
{
    uint64_t frame = 0;                 // monotonic per surface

    std::span<const Pass> passes;
    std::span<const Draw> draws;        // passes index into this

    std::span<const std::byte> vertexData;
    std::span<const std::byte> constantData;
    std::span<const std::byte> storageData;
};

} // namespace jvk::gpu
