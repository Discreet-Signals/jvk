/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Types.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::gpu
{

// =============================================================================
// jvk::gpu — what the core asks of a GPU (TRANSLATION_LAYER.md)
//
// Everything a backend implements or consumes lives under core/gpu/: these
// value types, PipelineState.h, FramePacket.h and Backend.h. None of it names
// JUCE or a GPU API, so a backend (Vulkan, Metal, D3D12, WebGPU, eacp) needs
// only these headers to know what it is being asked to do.
//
// The rule for what lands on which side: the core owns anything that changes
// pixels or encodes JUCE semantics, and any policy that holds on every API
// (caching, batching, atlas packing, pass planning). The backend owns anything
// whose only effect is validity or speed on one API (memory, native objects,
// synchronisation, submission).
// =============================================================================

// =============================================================================
// Handles
//
// The core refers to a backend's objects through opaque 32-bit handles. The
// backend chooses the encoding (e.g. 24-bit slot + 8-bit generation, so a
// stale handle is caught instead of aliasing a reused slot); the core only
// stores, compares and passes them back. 0 is null.
//
// Handles, not pointers, are what make the frame packet self-contained: the
// core can drop an object the moment it is done with it (destroyTexture) and
// the backend keeps the native object alive until the GPU has finished every
// frame that names it. No pinning on the core's side.
// =============================================================================

template <typename Tag>
struct Handle
{
    uint32_t value = 0;

    constexpr explicit operator bool() const { return value != 0; }
    constexpr bool operator==(const Handle&) const = default;
};

struct TextureTag;
struct ProgramTag;
struct PipelineTag;

using TextureId  = Handle<TextureTag>;
using ProgramId  = Handle<ProgramTag>;
using PipelineId = Handle<PipelineTag>;

// =============================================================================
// Geometry
//
// JUCE-free stand-ins for juce::Rectangle<int>: the backend headers must not
// need JUCE. Pixels, top-left origin, y down, in the space of whatever is
// being rendered into.
// =============================================================================

struct IRect
{
    int32_t x = 0, y = 0;
    int32_t width = 0, height = 0;

    constexpr bool isEmpty() const { return width <= 0 || height <= 0; }
    constexpr bool operator==(const IRect&) const = default;
};

struct Extent
{
    uint32_t width = 0, height = 0;

    constexpr bool operator==(const Extent&) const = default;
};

struct ClearColor
{
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

    constexpr bool operator==(const ClearColor&) const = default;
};

// =============================================================================
// Textures
//
// The core describes what a texture is for; the backend allocates it in its
// own memory and picks the native format. A depth/stencil texture names no
// PixelFormat: the core only needs an 8-bit stencil plane (clip stacks), and
// which combined format provides one is the backend's call (D24S8 is missing
// on Apple silicon, D32FS8 elsewhere costs more).
// =============================================================================

enum class TextureUsage : uint8_t
{
    Sampled      = 1 << 0,   // read by shaders
    RenderTarget = 1 << 1,   // a pass's colour or depth/stencil attachment
    Upload       = 1 << 2,   // written by Backend::writeTexture / clearTexture
    Readback     = 1 << 3,   // read back by Backend::readTexture
};

constexpr TextureUsage operator|(TextureUsage a, TextureUsage b)
{
    return static_cast<TextureUsage>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
constexpr bool hasUsage(TextureUsage set, TextureUsage bit)
{
    return (static_cast<uint8_t>(set) & static_cast<uint8_t>(bit)) != 0;
}

enum class TextureKind : uint8_t
{
    Color,
    DepthStencil,            // `format` ignored; the backend picks one with 8 stencil bits
};

struct TextureDesc
{
    TextureKind  kind   = TextureKind::Color;
    PixelFormat  format = PixelFormat::RGBA8;
    TextureUsage usage  = TextureUsage::Sampled | TextureUsage::Upload;
    uint32_t     width  = 0, height = 0;
    uint32_t     layers = 1, mips = 1;
    const char*  label  = nullptr;   // debug name (validation layers, GPU captures); may be null
};

// How a draw samples a texture. A property of the BINDING, not the texture:
// WebGPU, Metal and eacp all separate the two, and one texture may be read
// point-sampled by one draw and filtered by the next. Backends cache their
// native samplers by this value.
enum class Filter : uint8_t { Nearest, Linear };
enum class AddressMode : uint8_t { ClampToEdge, Repeat, MirroredRepeat };

struct SamplerDesc
{
    Filter      filter  = Filter::Linear;
    AddressMode address = AddressMode::ClampToEdge;

    constexpr bool operator==(const SamplerDesc&) const = default;
};

} // namespace jvk::gpu
