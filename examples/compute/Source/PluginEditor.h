/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: PluginEditor.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include "PluginProcessor.h"
#include <jvk/jvk.h>

// jvk compute on whichever backend the build runs (COMPUTE.md):
//   - a self-test on a stream, once: kernels with specialization constants
//     (a workgroup size and an array's size among them), atomics, a runtime
//     array's length, jvk/scan.glsl, fills, copies, uploads, downloads and
//     timestamps, each checked against the CPU;
//   - a waveform a kernel writes in paint (Graphics::dispatch) and a
//     jvk::Shader draws from that buffer, every frame, with no copy back.
class ComputeEditor : public jvk::AudioProcessorEditor
{
public:
    static constexpr uint32_t kColumns = 512;   // waveform.frag's

    explicit ComputeEditor(juce::AudioProcessor&);

    void paint(juce::Graphics&) override;   // every frame: jvk repaints at 60 Hz

private:
    void selfTest();

    std::unique_ptr<jvk::compute::Context> compute;
    jvk::compute::Kernel waveform;
    jvk::compute::Buffer peaks;
    jvk::Shader          peaksShader;
    juce::StringArray    report;
    const double         started = juce::Time::getMillisecondCounterHiRes();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ComputeEditor)
};
