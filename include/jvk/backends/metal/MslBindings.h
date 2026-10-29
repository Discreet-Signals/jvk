/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MslBindings.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

#include <cstddef>
#include <cstdint>

namespace jvk::metal
{

// =============================================================================
// Where a jvk::Shader's bindings live in Metal: the one rule the Metal backend
// binds by and jvk_msl (tools/jvk_msl.cpp) translates by, so the MSL made at
// build time binds where the backend binds.
//
//   a texture (combined image sampler)   [[texture(n)]] [[sampler(n)]]
//   a uniform or storage block           [[buffer(kBlockBase + n)]]
//   the constants (push constants)       [[buffer(kConstantsIndex)]], both stages
//
// n is the binding's rank among the program's bindings of its kind, ordered by
// (set, binding): a rule of what the SPIR-V declares alone.
// =============================================================================

inline constexpr uint32_t kBlockBase = 16, kConstantsIndex = 24;

struct MslBinding
{
    uint32_t set = 0, binding = 0;
    bool     block = false;   // a uniform or storage block, else a texture
};

inline uint32_t mslIndex(const MslBinding* bindings, size_t count, const MslBinding& b)
{
    uint32_t rank = 0;
    for (size_t i = 0; i < count; ++i)
        if (bindings[i].block == b.block && (bindings[i].set < b.set || (bindings[i].set == b.set && bindings[i].binding < b.binding))) ++rank;
    return b.block ? kBlockBase + rank : rank;
}

} // namespace jvk::metal
