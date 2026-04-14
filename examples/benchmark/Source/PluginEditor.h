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

    enum class Mode { All, Text, TextStatic, ComplexPaths, Fills, Images, BackendComparison };

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

    juce::TextButton btnAll, btnText, btnTextStatic, btnPaths, btnFills, btnImages, btnBackends;

    juce::TextButton btnStencil;

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

    // --- Backend comparison test ---
    std::vector<juce::Path> testPaths;
    std::vector<int> testPathSegCounts;
    void generateTestPaths();
    bool testPathsGenerated = false;

    struct BackendTestState
    {
        bool running = false;
        bool complete = false;
        int currentBackendIdx = 0;
        int currentComplexityIdx = 0;
        int framesRendered = 0;
        static constexpr int framesPerTest = 10;
        static constexpr int pathsPerFrame = 10;

        std::vector<int> complexities;

        struct BackendResult
        {
            juce::String name;
            juce::Colour color;
            std::vector<double> frameTimes;
        };
        std::vector<BackendResult> backends;

        double frameTimeAccum = 0;
        int globalFrame = 0;
    };
    BackendTestState backendTest;

    void startBackendComparison();
    void advanceBackendTest();
    void paintBackendResults(juce::Graphics& g);

    // Benchmark state
    Mode activeMode = Mode::All;
    bool benchmarking = false;
    bool showResults = false;
    bool benchVulkanPhase = false;
    int benchFrame = 0;
    double benchPhaseStart = 0;
    static constexpr double benchDurationMs = 5000.0;
    int vulkanRenderedFrames = 0, juceRenderedFrames = 0;
    double vulkanFPS = 0, juceFPS = 0;
    std::vector<double> vulkanFrameTimes, juceFrameTimes;
    double lastPaintTime = 0;
    VkPresentModeKHR savedPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    juce::String modeName;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BenchmarkEditor)
};
