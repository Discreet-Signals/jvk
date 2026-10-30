/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: spirv_cross.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------

 The vendored SPIRV-Cross (spirv_cross/), compiled once for jvk's build-machine
 tools (CMake: jvk_spirv_cross): jvk_msl and jvk_kernel link it. Nothing of
 it ships.
 ------------------------------------------------------------------------------
*/

#if defined(__clang__)
 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Weverything"
#elif defined(_MSC_VER)
 #pragma warning(push, 0)
#endif
#include "spirv_cross/spirv_cross_parsed_ir.cpp"
#include "spirv_cross/spirv_parser.cpp"
#include "spirv_cross/spirv_cfg.cpp"
#include "spirv_cross/spirv_cross.cpp"
#include "spirv_cross/spirv_glsl.cpp"
#include "spirv_cross/spirv_msl.cpp"
#if defined(__clang__)
 #pragma clang diagnostic pop
#elif defined(_MSC_VER)
 #pragma warning(pop)
#endif
