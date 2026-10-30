/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: PluginEditor.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "PluginEditor.h"
#include "BinaryData.h"
#include "ComputeKernels.h"   // jvk_compute_shaders (CMakeLists.txt): Source/kernels/*.comp

namespace
{
constexpr uint32_t kBins = 16;   // histogram.comp's constant 0 (its default is 8)

juce::String verdict(bool ok, const juce::String& what)
{
    return (ok ? "pass  " : "FAIL  ") + what;
}
}

ComputeEditor::ComputeEditor(juce::AudioProcessor& p) : jvk::AudioProcessorEditor(p)
{
    juce::String why;
    compute = jvk::compute::Context::create(&why);
    if (compute == nullptr)
        report.add("no compute: " + why);
    else
    {
        selfTest();
        waveform = compute->kernel(kernels::waveform);
        peaks = compute->buffer(kColumns * sizeof(float) * 2, jvk::compute::Usage::Device);
    }

    // waveform.frag: SPIR-V (what Vulkan draws, and what jvk::Shader reflects)
    // and its MSL (what Metal draws; empty in a build without Metal).
    std::vector<uint32_t> spirv(BinaryData::waveform_frag_spvSize / 4);
    std::memcpy(spirv.data(), BinaryData::waveform_frag_spv, spirv.size() * 4);
    peaksShader.load(spirv, std::string_view(BinaryData::waveform_frag_metal, size_t(BinaryData::waveform_frag_metalSize)));

    setSize(760, 520);
}

void ComputeEditor::selfTest()
{
    using namespace jvk::compute;
    auto& gpu = *compute;
    const auto& limits = gpu.limits();
    report.add("GPU: " + limits.device + (limits.unifiedMemory ? ", unified memory" : "")
               + (limits.ownQueue ? ", a compute queue of its own" : ", compute behind the frames")
               + ", subgroups of " + juce::String(limits.subgroupSize));

    // saxpy's workgroup (constant 0) and scale (constant 1); histogram's bins (constant 0).
    auto saxpy     = gpu.kernel(kernels::saxpy, { Constant::of(0, 128u), Constant::of(1, 0.5f) });
    auto histogram = gpu.kernel(kernels::histogram, { Constant::of(0, kBins) });
    auto prefix    = gpu.kernel(kernels::prefix);
    if (!saxpy.valid() || !histogram.valid() || !prefix.valid())
    {
        report.add("FAIL  a kernel could not be made (the log says why)");
        return;
    }
    report.add(verdict(saxpy.groupSize()[0] == 128, "saxpy's workgroup set by a constant: " + juce::String(saxpy.groupSize()[0])));

    constexpr uint32_t n = 100000, m = 1000;
    std::vector<float> x(n);
    std::vector<uint32_t> values(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        x[i] = float(i % 1000) * 0.25f;
        values[i] = (i * 7u) % 37u;
    }
    auto xs     = gpu.buffer(n * sizeof(float), Usage::Device);
    auto ys     = gpu.buffer(n * sizeof(float), Usage::Device);
    auto vs     = gpu.buffer(n * sizeof(uint32_t), Usage::Device);
    auto counts = gpu.buffer(kBins * sizeof(uint32_t), Usage::Device);
    auto scan   = gpu.buffer(m * sizeof(uint32_t), Usage::Shared);
    auto from   = gpu.buffer(64, Usage::Shared);
    auto to     = gpu.buffer(64, Usage::Shared);
    if (!xs.valid() || !ys.valid() || !vs.valid() || !counts.valid() || !scan.valid() || !from.valid() || !to.valid())
    {
        report.add("FAIL  a buffer could not be had");
        return;
    }
    std::fill_n(scan.as<uint32_t>(), m, 1u);
    scan.written();
    for (int i = 0; i < 64; ++i)
    {
        from.as<uint8_t>()[i] = uint8_t(i);
        to.as<uint8_t>()[i] = 0xee;
    }
    from.written();
    to.written();

    std::vector<float> y(n);
    std::vector<uint32_t> tally(kBins);
    struct SaxpyPush { uint32_t n; float a; };
    struct PrefixPush { uint32_t count; };

    auto stream = gpu.stream();
    if (stream == nullptr)
    {
        report.add("FAIL  no stream");
        return;
    }
    stream->time("start");
    stream->upload(xs.range(), x.data());
    stream->upload(vs.range(), values.data());
    stream->fill(ys.range(), 0x3f800000u);   // 1.0f: a word no byte fill makes
    stream->fill(counts.range());
    stream->barrier();
    stream->time("uploads and fills");
    stream->dispatch(saxpy, Groups::cover(n, saxpy.groupSize()[0], limits), { xs.range(), ys.range() }, SaxpyPush { n, 2.0f });
    stream->dispatch(histogram, Groups::cover(n, histogram.groupSize()[0], limits), { vs.range(), counts.range() });
    stream->dispatch(prefix, Groups { 1, 1, 1 }, { scan.range() }, PrefixPush { m });
    stream->copy(from.range(2, 6), to.range(3, 6));   // off a 4-byte boundary
    stream->barrier();
    stream->time("kernels");
    stream->download(ys.range(), y.data());
    stream->download(counts.range(), tally.data());
    stream->time("downloads");
    stream->submit();
    const auto result = stream->wait(5.0);
    if (result != Stream::Result::done)
    {
        report.add(result == Stream::Result::lost ? "FAIL  the GPU was lost" : "FAIL  the stream took over 5 s");
        return;
    }
    scan.toRead();
    to.toRead();

    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i)
        ok = y[i] == x[i] + 1.0f;   // 2 x 0.5 + 1: scale's constant taken, every element reached
    report.add(verdict(ok, "saxpy over " + juce::String(n) + " (an upload, a word fill, a float constant)"));

    std::vector<uint32_t> expected(kBins);
    for (auto v : values) ++expected[v % kBins];
    report.add(verdict(tally == expected, "histogram into " + juce::String(kBins) + " bins (shared memory sized by a constant, atomics, length())"));

    ok = true;
    for (uint32_t k = 0; k < m && ok; ++k)
        ok = scan.as<uint32_t>()[k] == k;
    report.add(verdict(ok, "exclusive scan of " + juce::String(m) + " (jvk/scan.glsl, shared memory)"));

    ok = true;
    for (int i = 0; i < 64 && ok; ++i)
        ok = to.as<uint8_t>()[i] == (i >= 3 && i < 9 ? uint8_t(i - 1) : uint8_t(0xee));
    report.add(verdict(ok, "a copy off 4-byte boundaries"));

    juce::String times;
    for (const auto& t : stream->times())
        times << t.label << " " << juce::String(t.ms, 3) << " ms   ";
    report.add(limits.timestampNs > 0.0 ? "GPU times: " + times : juce::String("no GPU timestamps on this device"));
}

void ComputeEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff0c0e12));
    auto area = getLocalBounds().reduced(16);
    auto wave = area.removeFromTop(area.getHeight() / 2);

    // The kernel writes this frame's peaks, then the shader draws them: one
    // frame, in paint order, the buffer never leaving the GPU.
    auto* jg = jvk::Graphics::create(g);
    if (jg != nullptr && waveform.valid() && peaks.valid())
    {
        struct Push { uint32_t columns; float time; };
        const auto time = float((juce::Time::getMillisecondCounterHiRes() - started) * 0.001);
        jg->dispatch(waveform, jvk::compute::Groups::cover(kColumns, waveform.groupSize()[0], compute->limits()),
                     { peaks.range() }, Push { kColumns, time });
        peaksShader.set("Peaks", peaks.range());
        jg->drawShader(peaksShader, wave.toFloat());
    }
    else
    {
        g.setColour(juce::Colours::grey);
        g.drawText(jg == nullptr ? "the software renderer: no kernels in frames" : "no waveform kernel",
                   wave, juce::Justification::centred);
    }
    g.setColour(juce::Colours::white.withAlpha(0.25f));
    g.drawRect(wave, 1);

    area.removeFromTop(12);
    g.setFont(juce::FontOptions(14.0f));
    for (const auto& line : report)
    {
        g.setColour(line.startsWith("FAIL") ? juce::Colours::orangered : juce::Colours::white);
        g.drawText(line, area.removeFromTop(22), juce::Justification::centredLeft);
    }
}
