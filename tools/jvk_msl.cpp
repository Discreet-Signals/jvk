/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: jvk_msl.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------

 jvk_msl <fragment.spv> <out.metal>

 The MSL the Metal backend draws a jvk::Shader with, translated at build time
 (CMake: jvk_msl) from its fragment stage's SPIR-V, so no translator ships. Its
 bindings go where the backend binds them (MslBindings.h), its push constants
 to [[buffer(24)]], where jvk's vertex stage reads them too (shader_region.vert,
 in the bundle). The vendored SPIRV-Cross (spirv_cross/, built by
 spirv_cross.cpp) is the build machine's alone.
 ------------------------------------------------------------------------------
*/

#if defined(__clang__)
 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Weverything"
#endif
#include "spirv_cross/spirv_msl.hpp"   // compiled once for the tools: spirv_cross.cpp
#if defined(__clang__)
 #pragma clang diagnostic pop
#endif

#include "../include/jvk/backends/metal/MslBindings.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3) { std::cerr << "usage: jvk_msl <fragment.spv> <out.metal>\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<char> bytes { std::istreambuf_iterator<char>(in), {} };
    if (bytes.empty() || bytes.size() % 4 != 0) { std::cerr << "jvk_msl: " << argv[1] << " is not SPIR-V\n"; return 1; }
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());

    namespace sc = spirv_cross;
    using jvk::metal::MslBinding;
    try
    {
        sc::CompilerMSL msl(std::move(words));
        auto options = msl.get_msl_options();
        options.platform = sc::CompilerMSL::Options::macOS;
        options.set_msl_version(2, 1);   // Metal 2.1: macOS 10.14, under the 10.15 floor
        msl.set_msl_options(options);

        // Every binding the stage declares, used or not, as jvk's reflection sees it.
        const auto resources = msl.get_shader_resources();
        std::vector<MslBinding> all;
        const auto add = [&](const auto& list, bool block)
        {
            for (const auto& r : list)
                all.push_back({ msl.get_decoration(r.id, spv::DecorationDescriptorSet), msl.get_decoration(r.id, spv::DecorationBinding), block });
        };
        add(resources.sampled_images, false);
        add(resources.uniform_buffers, true);
        add(resources.storage_buffers, true);
        for (const auto& b : all)
        {
            sc::MSLResourceBinding r;
            r.stage    = spv::ExecutionModelFragment;
            r.desc_set = b.set;
            r.binding  = b.binding;
            r.count    = 1;
            const auto index = jvk::metal::mslIndex(all.data(), all.size(), b);
            if (b.block) r.msl_buffer = index;
            else         r.msl_texture = r.msl_sampler = index;
            msl.add_msl_resource_binding(r);
        }
        sc::MSLResourceBinding constants;
        constants.stage      = spv::ExecutionModelFragment;
        constants.desc_set   = sc::kPushConstDescSet;
        constants.binding    = sc::kPushConstBinding;
        constants.count      = 1;
        constants.msl_buffer = jvk::metal::kConstantsIndex;
        msl.add_msl_resource_binding(constants);

        const auto source = msl.compile();
        if (msl.get_cleansed_entry_point_name("main", spv::ExecutionModelFragment) != "main0")
        {
            std::cerr << "jvk_msl: " << argv[1] << ": its entry point is not main\n";
            return 1;
        }
        std::ofstream out(argv[2], std::ios::binary);
        if (!(out << source)) { std::cerr << "jvk_msl: can't write " << argv[2] << '\n'; return 1; }
    }
    catch (const std::exception& e)
    {
        std::cerr << "jvk_msl: " << argv[1] << ": " << e.what() << '\n';
        return 1;
    }
    return 0;
}
