/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: SpirvToMsl.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace jvk::metal
{

// =============================================================================
// SPIR-V → MSL for a jvk::Shader's program (the user contract: Vulkan sets,
// push constants), so the Metal backend draws the SPIR-V applications compile.
// SPIRV-Cross does the translation (vendor/spirv_cross, built in its own
// translation unit, jvk_spirv_cross.cpp: this header names nothing of it).
//
// Each resource goes where the backend binds that slot; the push-constant
// block to [[buffer(constantsIndex)]], the same bytes on both stages. A vertex
// stage flips y: Metal's clip space is y-up, the shaders are written y-down.
// =============================================================================

struct MslResource
{
    uint8_t set = 0, binding = 0;
    bool    buffer = false;   // a uniform or storage block, else a combined image sampler
    uint8_t index  = 0;       // [[buffer(index)]], or [[texture(index)]] with [[sampler(index)]]
};

struct MslStage
{
    std::string source;   // MSL 2.1 for macOS
    std::string entry;    // the entry point's name in `source`
    std::string error;    // non-empty: the translation failed
};

MslStage spirvToMsl(const uint32_t* words, size_t wordCount, bool vertex,
                    const MslResource* resources, size_t resourceCount, uint32_t constantsIndex);

} // namespace jvk::metal
