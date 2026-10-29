/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Program.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::shaders
{

// =============================================================================
// The shader bundle's shape (TRANSLATION_LAYER.md, "Shaders")
//
// Every built-in program is compiled ahead of time from shaders/*.vert|frag
// (shaders/compile.sh) into Bundle.h: SPIR-V for Vulkan, and MSL translated
// from it by SPIRV-Cross. A backend takes the form its Caps::shaderFormat
// names. The layout (what the program binds, by slot) comes from reflecting
// the SPIR-V at build time, so no backend reflects at run time.
//
// Binding convention, one set (set 0):
//   texture i   binding i        (i < 8)
//   sampler i   binding 8 + i
//   storage j   binding 16 + j   (read-only)
//   constants   push constants in the Vulkan SPIR-V; a uniform block at
//               binding 24 in the SPIR-V the translations start from (MSL:
//               buffer(24), the same bytes bound to both stages)
// =============================================================================

struct Stage
{
    std::span<const uint32_t> spirv;   // Vulkan SPIR-V
    std::string_view          msl;     // Metal Shading Language, entry point "main0"; empty if not translated
};

// The two stages in `format` (a backend's Caps::shaderFormat) into `desc`:
// false when either has no form in it.
inline bool setStages(gpu::ProgramDesc& desc, gpu::ShaderFormat format, const Stage& vertex, const Stage& fragment)
{
    desc.format = format;
    const auto text = [](std::string_view s) { return std::as_bytes(std::span<const char>(s.data(), s.size())); };
    if (format == gpu::ShaderFormat::SPIRV && !vertex.spirv.empty() && !fragment.spirv.empty())
    {
        desc.vertex   = std::as_bytes(vertex.spirv);
        desc.fragment = std::as_bytes(fragment.spirv);
        return true;
    }
    if (format == gpu::ShaderFormat::MSL && !vertex.msl.empty() && !fragment.msl.empty())
    {
        desc.vertex   = text(vertex.msl);
        desc.fragment = text(fragment.msl);
        desc.vertexEntry = desc.fragmentEntry = "main0";
        return true;
    }
    return false;   // the bundle has no HLSL or WGSL yet
}

struct Program
{
    const char* name = nullptr;
    Stage       vertex, fragment;
    uint8_t     textures       = 0;    // slots 0..textures-1
    uint8_t     storageBuffers = 0;    // slots 0..storageBuffers-1
    uint16_t    constantsBytes = 0;    // the constants block, the larger of the two stages'
    uint8_t     colorOutputs   = 1;    // fragment output locations
};

} // namespace jvk::shaders
