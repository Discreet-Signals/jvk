/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: jvk_spirv_cross.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------

 The Metal backend's SPIR-V → MSL translation (backends/metal/SpirvToMsl.h):
 the vendored SPIRV-Cross (parser, GLSL and MSL back ends) and the one call
 jvk makes of it, in a translation unit of its own. It includes nothing of
 jvk.h, so the module's main TU never parses SPIRV-Cross. Apple only.
 ------------------------------------------------------------------------------
*/

#if defined(__APPLE__)

// Its own namespace: a host that links another SPIRV-Cross keeps its own.
#define SPIRV_CROSS_NAMESPACE_OVERRIDE jvk_spirv_cross

#if defined(__clang__)
 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Weverything"
#endif
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_cross_parsed_ir.cpp"
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_parser.cpp"
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_cfg.cpp"
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_cross.cpp"
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_glsl.cpp"
#include "include/jvk/backends/metal/vendor/spirv_cross/spirv_msl.cpp"
#if defined(__clang__)
 #pragma clang diagnostic pop
#endif

#include "include/jvk/backends/metal/SpirvToMsl.h"

namespace jvk::metal
{

MslStage spirvToMsl(const uint32_t* words, size_t wordCount, bool vertex,
                    const MslResource* resources, size_t resourceCount, uint32_t constantsIndex)
{
    namespace sc = SPIRV_CROSS_NAMESPACE;
    MslStage out;
    try
    {
        sc::CompilerMSL msl(words, wordCount);

        auto options = msl.get_msl_options();
        options.platform = sc::CompilerMSL::Options::macOS;
        options.set_msl_version(2, 1);   // Metal 2.1: macOS 10.14, under the 10.15 floor
        options.enable_decoration_binding = false;
        msl.set_msl_options(options);

        auto common = msl.get_common_options();
        common.vertex.flip_vert_y = vertex;
        msl.set_common_options(common);

        // Every resource on both stages: a binding a stage doesn't use is ignored.
        for (const auto model : { spv::ExecutionModelVertex, spv::ExecutionModelFragment })
        {
            for (size_t i = 0; i < resourceCount; ++i)
            {
                const auto& r = resources[i];
                sc::MSLResourceBinding b;
                b.stage    = model;
                b.desc_set = r.set;
                b.binding  = r.binding;
                b.count    = 1;
                if (r.buffer) b.msl_buffer = r.index;
                else          b.msl_texture = b.msl_sampler = r.index;
                msl.add_msl_resource_binding(b);
            }
            sc::MSLResourceBinding pc;
            pc.stage      = model;
            pc.desc_set   = sc::kPushConstDescSet;
            pc.binding    = sc::kPushConstBinding;
            pc.count      = 1;
            pc.msl_buffer = constantsIndex;
            msl.add_msl_resource_binding(pc);
        }

        out.source = msl.compile();
        out.entry  = msl.get_cleansed_entry_point_name("main", vertex ? spv::ExecutionModelVertex
                                                                      : spv::ExecutionModelFragment);
    }
    catch (const std::exception& e)
    {
        out.error = e.what();
        if (out.error.empty()) out.error = "SPIRV-Cross failed";
    }
    return out;
}

} // namespace jvk::metal

#endif
