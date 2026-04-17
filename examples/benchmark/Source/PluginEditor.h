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

class BenchmarkEditor : public jvk::AudioProcessorEditor,
                         private juce::Timer
{
public:
    BenchmarkEditor(BenchmarkProcessor&);
    ~BenchmarkEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    // Each mode exercises a specific pipeline or group of pipelines.
    // Modes marked (Vulkan-only) compare effect-on vs effect-off, both in Vulkan.
    enum class Mode {
        All,              // Mixed workload — all pipelines
        Fills,            // mainPipeline — flat rects, SDF rounded rects, ellipses
        Text,             // mainPipeline — MSDF text via glyph atlas
        Paths,            // stencilPipeline + stencilCoverPipeline — complex filled paths
        Images,           // mainPipeline — textured quads via texture cache
        Gradients,        // mainPipeline — gradient LUT descriptor set switching
        Effects,          // multiplyPipeline — hardware blend darken/tint (Vulkan-only)
        Blur,             // postProcess blurPipeline — 2-pass Gaussian (Vulkan-only)
        TextStatic        // mainPipeline — 20K static text volume stress test
    };

    // Returns true if this mode compares Vulkan-on vs Vulkan-off (not vs JUCE)
    static bool isVulkanOnlyMode(Mode m) { return m == Mode::Effects || m == Mode::Blur; }

    void runBenchmark(Mode mode);
    void paintBenchmarkScene(juce::Graphics& g, int frame);
    void paintBenchmarkResults(juce::Graphics& g);

    // Scene painters
    void sceneAll(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneText(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneTextStatic(juce::Graphics& g, float w, float h);
    void sceneComplexPaths(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneFills(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneImages(juce::Graphics& g, float w, float h);
    void sceneGradients(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneEffects(juce::Graphics& g, float w, float h, float t, float phase);
    void sceneBlur(juce::Graphics& g, float w, float h, float t, float phase);

    // Buttons — Row 1: primary pipeline benchmarks, Row 2: specialized benchmarks
    juce::TextButton btnAll, btnFills, btnText, btnPaths;
    juce::TextButton btnImages, btnGradients, btnEffects, btnBlur;
    juce::TextButton btnTextStatic;
    juce::TextButton btnToggleRenderer;

    void setBenchButtonsVisible(bool visible);
    void updateToggleButtonText();

    // Pregenerated complex paths (built once at startup)
    std::vector<juce::Path> waveformPaths;
    std::vector<juce::Path> spectrumPaths;
    std::vector<juce::Path> gearPaths;
    void generatePaths(float w, float h);
    bool pathsGenerated = false;

    // --- Image benchmark ---
    juce::Image barkImage;
    struct Sprite { float x, y, vx, vy; };
    std::vector<Sprite> sprites;
    void initSprites(float w, float h);
    bool spritesInitialized = false;

    // Benchmark state
    Mode activeMode = Mode::All;
    bool benchmarking = false;
    bool showResults = false;
    bool benchVulkanPhase = false;  // true = phase 1, false = phase 2
    bool effectEnabled = true;      // for Vulkan-only modes: true in phase 1, false in phase 2
    int benchFrame = 0;
    double benchPhaseStart = 0;
    static constexpr double benchDurationMs = 5000.0;
    int vulkanRenderedFrames = 0, juceRenderedFrames = 0;
    double vulkanFPS = 0, juceFPS = 0;
    juce::String phase1Label, phase2Label;  // "Vulkan"/"JUCE" or "Effect ON"/"Effect OFF"
    std::vector<double> vulkanFrameTimes, juceFrameTimes;
    double lastPaintTime = 0;
    VkPresentModeKHR savedPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    juce::String modeName;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BenchmarkEditor)
};
