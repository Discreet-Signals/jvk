/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC
 
 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝
 
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

class CompressorEditor : public jvk::AudioProcessorEditor,
                          private juce::Timer
{
public:
    CompressorEditor(CompressorProcessor&);
    ~CompressorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    // Benchmark
    void runBenchmark();
    void paintBenchmarkScene(juce::Graphics& g, int frame);
    void paintBenchmarkResults(juce::Graphics& g);

    CompressorProcessor& processor;

    juce::Slider thresholdSlider, ratioSlider, attackSlider, releaseSlider, makeupSlider;
    juce::Label thresholdLabel, ratioLabel, attackLabel, releaseLabel, makeupLabel;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
        thresholdAttach, ratioAttach, attackAttach, releaseAttach, makeupAttach;

    juce::TextButton vulkanToggle;
    juce::TextButton benchmarkButton;

    // Benchmark state
    bool benchmarking = false;
    bool showResults = false;
    int benchFrame = 0;
    double benchPhaseStart = 0;
    static constexpr double benchDurationMs = 5000.0;
    int vulkanRenderedFrames = 0, juceRenderedFrames = 0;
    double vulkanFPS = 0, juceFPS = 0;
    std::vector<double> vulkanFrameTimes, juceFrameTimes;
    double lastPaintTime = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CompressorEditor)
};
