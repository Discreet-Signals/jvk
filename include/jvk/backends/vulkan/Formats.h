#pragma once

namespace jvk
{

// The only place jvk's PixelFormat (core/PixelFormat.h) meets Vulkan.
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

} // namespace jvk
