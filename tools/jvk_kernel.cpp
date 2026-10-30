/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: jvk_kernel.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------

 jvk_kernel <out.h> <namespace> <name>=<kernel.spv> ...

 The header jvk_compute_shaders() makes of a project's kernels (COMPUTE.md),
 at build time, so nothing reflects or translates at run time. For each
 kernel:
   - its SPIR-V (what the Vulkan backend runs), under #if JVK_VULKAN;
   - its MSL (what the Metal backend runs), translated here with its buffers
     where the backend binds them (MslBindings.h), under #if JVK_METAL;
   - its interface, reflected from the SPIR-V: the buffers it binds, its
     push-constant block's size, its workgroup size and the constants that
     set it, its specialization constants and their types;
   - <namespace>::<name>, a jvk::compute::KernelSource over all of it.
 The header is rewritten only when its text changes. A kernel jvk can't
 bind (a texture, a set other than 0, a binding past 15) fails the build.
 The vendored SPIRV-Cross (spirv_cross/, built by spirv_cross.cpp) is the
 build machine's alone.
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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace
{

namespace sc = spirv_cross;

constexpr uint32_t kMaxBindings = 16;   // gpu::kMaxKernelBindings

struct Binding  { uint32_t binding; bool uniform; bool writes; };
struct Constant { uint32_t id; const char* type; uint32_t bits; };

struct Kernel
{
    std::string           name;
    std::vector<uint32_t> spirv;
    std::string           msl;
    std::vector<Binding>  bindings;
    uint32_t              constantsBytes = 0;
    uint32_t              groupSize[3] { 1, 1, 1 };
    uint32_t              groupSizeId[3] { ~0u, ~0u, ~0u };
    std::vector<Constant> constants;
    bool                  bufferSizes = false;
};

[[noreturn]] void fail(const std::string& what)
{
    std::cerr << "jvk_kernel: " << what << '\n';
    std::exit(1);
}

std::vector<uint32_t> readSpirv(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes { std::istreambuf_iterator<char>(in), {} };
    if (bytes.empty() || bytes.size() % 4 != 0) fail(path + " is not SPIR-V");
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

Kernel reflect(const std::string& name, const std::string& path)
{
    Kernel k;
    k.name  = name;
    k.spirv = readSpirv(path);
    const std::string where = path + " (" + name + ")";
    try
    {
        sc::CompilerMSL msl(k.spirv);
        if (msl.get_execution_model() != spv::ExecutionModelGLCompute) fail(where + ": not a compute shader");
        auto options = msl.get_msl_options();
        options.platform = sc::CompilerMSL::Options::macOS;
        options.set_msl_version(2, 1);   // Metal 2.1: macOS 10.14, under the 10.15 floor
        options.buffer_size_buffer_index = jvk::metal::kBufferSizesIndex;
        msl.set_msl_options(options);

        const auto resources = msl.get_shader_resources();
        if (!resources.sampled_images.empty() || !resources.separate_images.empty() || !resources.separate_samplers.empty()
            || !resources.storage_images.empty() || !resources.atomic_counters.empty() || !resources.acceleration_structures.empty())
            fail(where + ": binds a texture or an image; kernels bind buffers alone for now (COMPUTE.md)");

        auto addBuffer = [&](const sc::Resource& r, bool uniform)
        {
            const uint32_t set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
            const uint32_t binding = msl.get_decoration(r.id, spv::DecorationBinding);
            if (set != 0) fail(where + ": " + r.name + " is in set " + std::to_string(set) + "; kernels bind set 0 alone");
            if (binding >= kMaxBindings) fail(where + ": " + r.name + " is at binding " + std::to_string(binding) + "; kernels bind 0 to 15");
            for (const auto& b : k.bindings)
                if (b.binding == binding) fail(where + ": two buffers at binding " + std::to_string(binding));
            const bool readonly = msl.has_decoration(r.id, spv::DecorationNonWritable)
                               || msl.get_buffer_block_flags(r.id).get(spv::DecorationNonWritable);
            k.bindings.push_back({ binding, uniform, !uniform && !readonly });

            sc::MSLResourceBinding mb;
            mb.stage      = spv::ExecutionModelGLCompute;
            mb.desc_set   = set;
            mb.binding    = binding;
            mb.count      = 1;
            mb.msl_buffer = binding;   // [[buffer(b)]] (MslBindings.h)
            msl.add_msl_resource_binding(mb);
        };
        for (const auto& r : resources.storage_buffers) addBuffer(r, false);
        for (const auto& r : resources.uniform_buffers) addBuffer(r, true);
        std::sort(k.bindings.begin(), k.bindings.end(), [](const Binding& a, const Binding& b) { return a.binding < b.binding; });

        if (resources.push_constant_buffers.size() > 1) fail(where + ": more than one push-constant block");
        if (!resources.push_constant_buffers.empty())
        {
            const auto& block = resources.push_constant_buffers.front();
            k.constantsBytes = (static_cast<uint32_t>(msl.get_declared_struct_size(msl.get_type(block.base_type_id))) + 3u) & ~3u;
            sc::MSLResourceBinding mb;
            mb.stage      = spv::ExecutionModelGLCompute;
            mb.desc_set   = sc::kPushConstDescSet;
            mb.binding    = sc::kPushConstBinding;
            mb.count      = 1;
            mb.msl_buffer = jvk::metal::kConstantsIndex;
            msl.add_msl_resource_binding(mb);
        }

        // The workgroup's size, and the constants that set it (local_size_x_id).
        sc::SpecializationConstant wx, wy, wz;
        msl.get_work_group_size_specialization_constants(wx, wy, wz);
        const sc::SpecializationConstant* w[3] = { &wx, &wy, &wz };
        for (uint32_t i = 0; i < 3; ++i)
        {
            k.groupSize[i] = std::max(1u, msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, i));
            if (w[i]->id != 0)
            {
                k.groupSizeId[i] = w[i]->constant_id;
                k.groupSize[i]   = std::max(1u, msl.get_constant(w[i]->id).scalar());
            }
            else if (const uint32_t id = msl.get_execution_mode_argument(spv::ExecutionModeLocalSizeId, i); id != 0)
            {
                if (msl.get_constant(id).specialization && msl.has_decoration(id, spv::DecorationSpecId))
                    k.groupSizeId[i] = msl.get_decoration(id, spv::DecorationSpecId);
            }
        }

        for (const auto& c : msl.get_specialization_constants())
        {
            const auto& constant = msl.get_constant(c.id);
            const auto& type = msl.get_type(constant.constant_type);
            const char* t = nullptr;
            switch (type.basetype)
            {
                case sc::SPIRType::Boolean: t = "Bool";  break;
                case sc::SPIRType::Int:     t = "Int";   break;
                case sc::SPIRType::UInt:    t = "UInt";  break;
                case sc::SPIRType::Float:   t = "Float"; break;
                default: fail(where + ": specialization constant " + std::to_string(c.constant_id)
                              + " is not a bool, int, uint or float");
            }
            if (type.vecsize != 1 || type.columns != 1) fail(where + ": specialization constant " + std::to_string(c.constant_id) + " is not a scalar");
            bool seen = false;
            for (const auto& e : k.constants) seen |= e.id == c.constant_id;
            if (!seen) k.constants.push_back({ c.constant_id, t, constant.scalar() });
        }
        std::sort(k.constants.begin(), k.constants.end(), [](const Constant& a, const Constant& b) { return a.id < b.id; });

        k.msl = msl.compile();
        if (msl.get_cleansed_entry_point_name("main", spv::ExecutionModelGLCompute) != "main0")
            fail(where + ": its entry point is not main");
        k.bufferSizes = msl.needs_buffer_size_buffer();
    }
    catch (const std::exception& e)
    {
        fail(where + ": " + e.what());
    }
    return k;
}

std::string words(const std::vector<uint32_t>& w)
{
    std::string out;
    char hex[16];
    for (size_t i = 0; i < w.size(); ++i)
    {
        if (i % 8 == 0) out += i == 0 ? "    " : "\n    ";
        std::snprintf(hex, sizeof(hex), "0x%08x,", w[i]);
        out += hex;
        if (i % 8 != 7) out += ' ';
    }
    return out;
}

// Adjacent raw strings of at most ~2 KB, split at lines: MSVC caps one string
// literal at 16380 bytes (shaders/compile.sh does the same for the bundle).
std::string mslLiteral(const std::string& where, const std::string& text)
{
    static const std::string delimiter = "jvkmsl";
    if (text.find(")" + delimiter + "\"") != std::string::npos) fail(where + ": its MSL holds the raw-string delimiter");
    if (text.size() > 60000) fail(where + ": its MSL is too long for one string literal on MSVC");
    std::string out, chunk;
    std::istringstream lines(text);
    std::string line;
    auto flush = [&]
    {
        if (chunk.empty()) return;
        out += (out.empty() ? "" : "\n") + std::string("R\"") + delimiter + "(" + chunk + ")" + delimiter + "\"";
        chunk.clear();
    };
    while (std::getline(lines, line))
    {
        if (chunk.size() + line.size() + 1 > 2000) flush();
        chunk += line + '\n';
    }
    flush();
    return out.empty() ? "\"\"" : out;
}

std::string header(const std::string& ns, const std::vector<Kernel>& kernels)
{
    std::string out = "// Generated by jvk_kernel (jvk_compute_shaders) from GLSL compute shaders: do not edit.\n"
                      "// Each kernel's SPIR-V and MSL are built where the build has a backend that\n"
                      "// takes them (JVK_VULKAN, JVK_METAL); its interface was reflected at build time.\n"
                      "#pragma once\n#include <jvk/jvk.h>\n\nnamespace " + ns + "\n{\n";
    for (const auto& k : kernels)
    {
        const std::string& n = k.name;
        out += "\n// ---- " + n + " ----\n";
        out += "#if JVK_VULKAN\nalignas(4) inline constexpr uint32_t " + n + "_spirv[] = {\n" + words(k.spirv) + "\n};\n#endif\n";
        out += "#if JVK_METAL\ninline constexpr char " + n + "_msl[] =\n" + mslLiteral(n, k.msl) + ";\n#endif\n";
        if (!k.bindings.empty())
        {
            out += "inline constexpr ::jvk::gpu::KernelBinding " + n + "_bindings[] = {\n";
            for (const auto& b : k.bindings)
                out += "    { " + std::to_string(b.binding) + ", ::jvk::gpu::BindingKind::" + (b.uniform ? "UniformBuffer" : "StorageBuffer")
                     + ", " + (b.writes ? "true" : "false") + " },\n";
            out += "};\n";
        }
        if (!k.constants.empty())
        {
            out += "inline constexpr ::jvk::gpu::SpecConstant " + n + "_constants[] = {\n";
            for (const auto& c : k.constants)
                out += "    { " + std::to_string(c.id) + ", ::jvk::gpu::ScalarType::" + c.type + ", " + std::to_string(c.bits) + "u },\n";
            out += "};\n";
        }
        auto id = [](uint32_t v) { return v == ~0u ? std::string("0xffffffffu") : std::to_string(v) + "u"; };
        out += "inline constexpr ::jvk::compute::KernelSource " + n + " {\n"
               "    \"" + n + "\",\n"
               "    JVK_KERNEL_SPIRV(" + n + "),\n"
               "    JVK_KERNEL_MSL(" + n + "),\n"
               "    " + (k.bindings.empty() ? std::string("{}") : n + "_bindings") + ",\n"
               "    " + std::to_string(k.constantsBytes) + ",\n"
               "    { " + std::to_string(k.groupSize[0]) + "u, " + std::to_string(k.groupSize[1]) + "u, " + std::to_string(k.groupSize[2]) + "u },\n"
               "    { " + id(k.groupSizeId[0]) + ", " + id(k.groupSizeId[1]) + ", " + id(k.groupSizeId[2]) + " },\n"
               "    " + (k.constants.empty() ? std::string("{}") : n + "_constants") + ",\n"
               "    " + (k.bufferSizes ? "true" : "false") + "\n};\n";
    }
    out += "\n} // namespace " + ns + "\n";
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) { std::cerr << "usage: jvk_kernel <out.h> <namespace> <name>=<kernel.spv> ...\n"; return 2; }
    const std::string output = argv[1], ns = argv[2];
    std::vector<Kernel> kernels;
    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq == std::string::npos || eq == 0) fail("expected <name>=<kernel.spv>, not " + arg);
        kernels.push_back(reflect(arg.substr(0, eq), arg.substr(eq + 1)));
    }
    const std::string text = header(ns, kernels);

    {
        std::ifstream in(output, std::ios::binary);
        const std::string old { std::istreambuf_iterator<char>(in), {} };
        if (old == text) return 0;   // unchanged: nothing that includes it rebuilds
    }
    std::ofstream out(output, std::ios::binary);
    if (!(out << text)) fail("can't write " + output);
    return 0;
}
