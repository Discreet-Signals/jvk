#pragma once

namespace jvk
{

// =============================================================================
// PixelFormat — how a texture or render target stores each pixel
//
// jvk's own names for the GPU formats an application chooses (render
// targets, textures it creates and uploads). Like juce::Image::PixelFormat,
// but for GPU-side storage. Names list components in memory order with their
// width and kind:
//
//   (none)  normalised: stored as integers, read as 0..1 floats
//   sRGB    normalised, sRGB-encoded: decoded to linear when sampled,
//           encoded when rendered to (and blended in linear)
//   F       floating point
//   UI      unsigned integer, read as integers (ids; a `usampler2D`). For
//           textures only: not blendable, so never a render target
//   BCn     block-compressed (4x4 blocks), for sampled textures only: a GPU
//           cannot render to them. Needs the textureCompressionBC feature
//           (Device::supportsTextureCompressionBC)
//
// toVkFormat() is the only place these meet Vulkan.
// =============================================================================

enum class PixelFormat
{
    // 8 bits per component
    R8, RG8, RGBA8, BGRA8,                // BGRA8 = juce::Image::ARGB's byte order
    RGBA8sRGB, BGRA8sRGB,

    // 16 bits per component
    R16, RG16, RGBA16,                    // normalised
    R16F, RG16F, RGBA16F,                 // half float

    // 32 bits per component
    R32F, RG32F, RGBA32F,

    // Packed, 32 bits per pixel
    RGB10A2,                              // 10-bit colour, 2-bit alpha (normals, detail)
    R11G11B10F,                           // unsigned floats, no alpha (HDR colour)

    // Unsigned integers
    R8UI, R16UI, R32UI, RG32UI,

    // Block-compressed (sampled only)
    BC1,                                  // RGB(A) 5:6:5, 4 bits/px
    BC3,                                  // RGBA, 8 bits/px
    BC4,                                  // one component, 4 bits/px
    BC5,                                  // two components, 8 bits/px (normal xy)
    BC6H,                                 // HDR RGB half float, 8 bits/px, no alpha
    BC7, BC7sRGB                          // RGBA, 8 bits/px, high quality
};

inline VkFormat toVkFormat(PixelFormat f)
{
    switch (f)
    {
        case PixelFormat::R8:          return VK_FORMAT_R8_UNORM;
        case PixelFormat::RG8:         return VK_FORMAT_R8G8_UNORM;
        case PixelFormat::RGBA8:       return VK_FORMAT_R8G8B8A8_UNORM;
        case PixelFormat::BGRA8:       return VK_FORMAT_B8G8R8A8_UNORM;
        case PixelFormat::RGBA8sRGB:   return VK_FORMAT_R8G8B8A8_SRGB;
        case PixelFormat::BGRA8sRGB:   return VK_FORMAT_B8G8R8A8_SRGB;
        case PixelFormat::R16:         return VK_FORMAT_R16_UNORM;
        case PixelFormat::RG16:        return VK_FORMAT_R16G16_UNORM;
        case PixelFormat::RGBA16:      return VK_FORMAT_R16G16B16A16_UNORM;
        case PixelFormat::R16F:        return VK_FORMAT_R16_SFLOAT;
        case PixelFormat::RG16F:       return VK_FORMAT_R16G16_SFLOAT;
        case PixelFormat::RGBA16F:     return VK_FORMAT_R16G16B16A16_SFLOAT;
        case PixelFormat::R32F:        return VK_FORMAT_R32_SFLOAT;
        case PixelFormat::RG32F:       return VK_FORMAT_R32G32_SFLOAT;
        case PixelFormat::RGBA32F:     return VK_FORMAT_R32G32B32A32_SFLOAT;
        case PixelFormat::RGB10A2:     return VK_FORMAT_A2B10G10R10_UNORM_PACK32;   // R in the low bits
        case PixelFormat::R11G11B10F:  return VK_FORMAT_B10G11R11_UFLOAT_PACK32;    // R in the low bits
        case PixelFormat::R8UI:        return VK_FORMAT_R8_UINT;
        case PixelFormat::R16UI:       return VK_FORMAT_R16_UINT;
        case PixelFormat::R32UI:       return VK_FORMAT_R32_UINT;
        case PixelFormat::RG32UI:      return VK_FORMAT_R32G32_UINT;
        case PixelFormat::BC1:         return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case PixelFormat::BC3:         return VK_FORMAT_BC3_UNORM_BLOCK;
        case PixelFormat::BC4:         return VK_FORMAT_BC4_UNORM_BLOCK;
        case PixelFormat::BC5:         return VK_FORMAT_BC5_UNORM_BLOCK;
        case PixelFormat::BC6H:        return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case PixelFormat::BC7:         return VK_FORMAT_BC7_UNORM_BLOCK;
        case PixelFormat::BC7sRGB:     return VK_FORMAT_BC7_SRGB_BLOCK;
    }
    return VK_FORMAT_UNDEFINED;
}

// What an upload needs to know: the data is a grid of blocks, `blockBytes`
// each, every block covering blockWidth x blockHeight pixels (1 x 1 for
// uncompressed formats, 4 x 4 for BCn). A w x h region is
// ceil(w / blockWidth) * ceil(h / blockHeight) * blockBytes bytes.
struct PixelFormatInfo
{
    uint32_t blockBytes  = 4;
    uint32_t blockWidth  = 1;
    uint32_t blockHeight = 1;
    bool     compressed  = false;   // BCn: sampled only, never a render target
    bool     integer     = false;   // UI: not blendable, never a render target

    uint64_t bytesFor(uint32_t width, uint32_t height) const
    {
        return uint64_t((width + blockWidth - 1) / blockWidth)
             * uint64_t((height + blockHeight - 1) / blockHeight) * blockBytes;
    }
};

inline PixelFormatInfo formatInfo(PixelFormat f)
{
    auto plain = [](uint32_t bytes, bool integer = false) { return PixelFormatInfo { bytes, 1, 1, false, integer }; };
    auto block = [](uint32_t bytes) { return PixelFormatInfo { bytes, 4, 4, true, false }; };
    switch (f)
    {
        case PixelFormat::R8:          return plain(1);
        case PixelFormat::RG8:         return plain(2);
        case PixelFormat::RGBA8:
        case PixelFormat::BGRA8:
        case PixelFormat::RGBA8sRGB:
        case PixelFormat::BGRA8sRGB:   return plain(4);
        case PixelFormat::R16:
        case PixelFormat::R16F:        return plain(2);
        case PixelFormat::RG16:
        case PixelFormat::RG16F:       return plain(4);
        case PixelFormat::RGBA16:
        case PixelFormat::RGBA16F:     return plain(8);
        case PixelFormat::R32F:        return plain(4);
        case PixelFormat::RG32F:       return plain(8);
        case PixelFormat::RGBA32F:     return plain(16);
        case PixelFormat::RGB10A2:
        case PixelFormat::R11G11B10F:  return plain(4);
        case PixelFormat::R8UI:        return plain(1, true);
        case PixelFormat::R16UI:       return plain(2, true);
        case PixelFormat::R32UI:       return plain(4, true);
        case PixelFormat::RG32UI:      return plain(8, true);
        case PixelFormat::BC1:
        case PixelFormat::BC4:         return block(8);
        case PixelFormat::BC3:
        case PixelFormat::BC5:
        case PixelFormat::BC6H:
        case PixelFormat::BC7:
        case PixelFormat::BC7sRGB:     return block(16);
    }
    return {};
}

} // namespace jvk
