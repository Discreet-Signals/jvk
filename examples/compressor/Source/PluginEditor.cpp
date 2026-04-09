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
 File: PluginEditor.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

static void styleKnob(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 18);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
}

static void styleLabel(juce::Label& label, const juce::String& text)
{
    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.setColour(juce::Label::textColourId, juce::Colours::white);
}

CompressorEditor::CompressorEditor(CompressorProcessor& p)
    : jvk::AudioProcessorEditor(p), processor(p)
{
    styleKnob(thresholdSlider); addAndMakeVisible(thresholdSlider);
    styleKnob(ratioSlider);     addAndMakeVisible(ratioSlider);
    styleKnob(attackSlider);    addAndMakeVisible(attackSlider);
    styleKnob(releaseSlider);   addAndMakeVisible(releaseSlider);
    styleKnob(makeupSlider);    addAndMakeVisible(makeupSlider);

    thresholdSlider.setTextValueSuffix(" dB");
    ratioSlider.setTextValueSuffix(":1");
    attackSlider.setTextValueSuffix(" ms");
    releaseSlider.setTextValueSuffix(" ms");
    makeupSlider.setTextValueSuffix(" dB");

    styleLabel(thresholdLabel, "THRESHOLD"); addAndMakeVisible(thresholdLabel);
    styleLabel(ratioLabel, "RATIO");         addAndMakeVisible(ratioLabel);
    styleLabel(attackLabel, "ATTACK");       addAndMakeVisible(attackLabel);
    styleLabel(releaseLabel, "RELEASE");     addAndMakeVisible(releaseLabel);
    styleLabel(makeupLabel, "MAKEUP");       addAndMakeVisible(makeupLabel);

    thresholdAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "threshold", thresholdSlider);
    ratioAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "ratio", ratioSlider);
    attackAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "attack", attackSlider);
    releaseAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "release", releaseSlider);
    makeupAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "makeup", makeupSlider);

    vulkanToggle.setButtonText("VULKAN");
    vulkanToggle.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF6C63FF));
    vulkanToggle.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    vulkanToggle.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    vulkanToggle.onClick = [this]
    {
        bool nowEnabled = !isVulkanEnabled();
        setVulkanEnabled(nowEnabled);
        vulkanToggle.setButtonText(nowEnabled ? "VULKAN" : "JUCE");
        vulkanToggle.setColour(juce::TextButton::buttonColourId,
            nowEnabled ? juce::Colour(0xFF6C63FF) : juce::Colour(0xFF444444));
    };
    addAndMakeVisible(vulkanToggle);

    benchmarkButton.setButtonText("BENCH");
    benchmarkButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF2A9D8F));
    benchmarkButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    benchmarkButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    benchmarkButton.onClick = [this] { runBenchmark(); };
    addAndMakeVisible(benchmarkButton);

    setSize(550, 400);
}

// =============================================================================
// Benchmark
// =============================================================================

void CompressorEditor::runBenchmark()
{
    if (benchmarking) return;

    showResults = false;
    vulkanFrameTimes.clear();
    juceFrameTimes.clear();
    vulkanRenderedFrames = 0;
    juceRenderedFrames = 0;
    lastPaintTime = 0;

    setVulkanEnabled(true);
    benchmarking = true;
    benchFrame = 0;
    benchPhaseStart = juce::Time::getMillisecondCounterHiRes();
    startTimerHz(1000);
}

void CompressorEditor::timerCallback()
{
    if (!benchmarking) { stopTimer(); return; }
    repaint(); // request a paint — actual counting happens in paint()
}

void CompressorEditor::paintBenchmarkScene(juce::Graphics& g, int frame)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());
    float t = static_cast<float>(frame % 300) / 300.0f;
    float phase = t * juce::MathConstants<float>::twoPi * 3.0f;

    g.fillAll(juce::Colour(0xFF0E0E1A));

    // --- 200 filled rectangles ---
    for (int i = 0; i < 200; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::sin(phase + fi * 0.3f) * w * 0.45f;
        float y = h * 0.5f + std::cos(phase + fi * 0.5f) * h * 0.45f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 200.0f + t, 1.0f), 0.8f, 0.9f, 0.4f));
        g.fillRect(x - 10.0f, y - 10.0f, 20.0f, 20.0f);
    }

    // --- 150 rounded rectangles (SDF vs EdgeTable) ---
    for (int i = 0; i < 150; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 23.0f + t * w * 2.0f, w + 100.0f) - 50.0f;
        float y = std::fmod(fi * 17.0f + t * h, h);
        float cornerR = 4.0f + std::fmod(fi * 3.0f, 12.0f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 150.0f + t * 0.5f, 1.0f), 0.7f, 0.85f, 0.5f));
        g.fillRoundedRectangle(x, y, 50.0f + fi * 0.2f, 25.0f + fi * 0.1f, cornerR);
    }

    // --- 100 ellipses ---
    for (int i = 0; i < 100; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::cos(phase * 0.8f + fi * 0.7f) * w * 0.4f;
        float y = h * 0.5f + std::sin(phase * 0.6f + fi * 0.9f) * h * 0.4f;
        float rx = 5.0f + fi * 0.3f, ry = 5.0f + fi * 0.2f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 100.0f + t * 2.0f, 1.0f), 0.9f, 1.0f, 0.35f));
        g.fillEllipse(x - rx, y - ry, rx * 2, ry * 2);
    }

    // --- 50 complex bezier paths ---
    for (int i = 0; i < 50; i++)
    {
        float fi = static_cast<float>(i);
        juce::Path p;
        float sx = std::fmod(fi * 53.0f + t * 200.0f, w);
        float sy = std::fmod(fi * 37.0f + t * 100.0f, h);
        p.startNewSubPath(sx, sy);
        for (int seg = 0; seg < 4; seg++)
        {
            float fs = static_cast<float>(seg);
            p.cubicTo(
                sx + (30.0f + fs * 20.0f) * std::sin(phase + fi + fs),
                sy + (40.0f + fs * 15.0f) * std::cos(phase + fi + fs * 0.7f),
                sx + (60.0f + fs * 25.0f) * std::cos(phase * 0.5f + fi + fs),
                sy + (20.0f + fs * 30.0f) * std::sin(phase * 0.7f + fi + fs),
                sx + (90.0f + fs * 30.0f), sy + (30.0f + fs * 20.0f));
        }
        p.closeSubPath();
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 50.0f + t, 1.0f), 0.6f, 0.95f, 0.3f));
        g.fillPath(p);
    }

    // --- 30 stroked stars ---
    for (int i = 0; i < 30; i++)
    {
        float fi = static_cast<float>(i);
        juce::Path p;
        p.addStar(juce::Point<float>(
            std::fmod(fi * 67.0f + t * 300.0f, w),
            std::fmod(fi * 43.0f + t * 150.0f, h)),
            5 + (i % 6), 6.0f + fi * 0.5f, 15.0f + fi);
        g.setColour(juce::Colours::white.withAlpha(0.5f));
        g.strokePath(p, juce::PathStrokeType(1.5f + fi * 0.1f));
    }

    // --- 20 gradient rectangles ---
    for (int i = 0; i < 20; i++)
    {
        float fi = static_cast<float>(i);
        float gx = std::fmod(fi * 57.0f, w);
        float gy = std::fmod(fi * 41.0f + t * 100.0f, h);
        juce::ColourGradient grad(
            juce::Colour::fromHSV(std::fmod(t + fi / 20.0f, 1.0f), 0.9f, 1.0f, 0.7f),
            gx, gy,
            juce::Colour::fromHSV(std::fmod(t + fi / 20.0f + 0.4f, 1.0f), 0.9f, 0.4f, 0.7f),
            gx + 80.0f, gy + 40.0f, (i % 3 == 0)); // every 3rd is radial
        g.setGradientFill(grad);
        g.fillRoundedRectangle(gx, gy, 80.0f, 40.0f, 6.0f);
    }

    // --- 60 text labels at various sizes ---
    juce::String texts[] = {
        "Vulkan GPU", "SDF Shapes", "Stencil Paths", "2x Super",
        "sRGB Gamma", "Lock Free", "Zero Alloc", "EdgeTable",
        "Bezier Cubic", "Anti-Alias", "Cross Plat", "Real-Time",
        "MoltenVK", "Pipeline", "Descriptor", "Framebuffer"
    };
    for (int i = 0; i < 60; i++)
    {
        float fi = static_cast<float>(i);
        float fontSize = 8.0f + std::fmod(fi * 3.7f, 16.0f);
        g.setFont(juce::FontOptions(fontSize));
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 60.0f + t, 1.0f), 0.3f, 1.0f, 0.8f));
        float tx = std::fmod(fi * 43.0f + t * 200.0f, w + 200.0f) - 100.0f;
        float ty = std::fmod(fi * 29.0f + t * 80.0f, h);
        g.drawText(texts[i % 16], static_cast<int>(tx), static_cast<int>(ty), 150, 20,
                    juce::Justification::centredLeft);
    }

    // --- 20 clipped ellipses ---
    for (int i = 0; i < 20; i++)
    {
        float fi = static_cast<float>(i);
        float cx = std::fmod(fi * 67.0f + t * 150.0f, w);
        float cy = std::fmod(fi * 51.0f + t * 80.0f, h);
        g.saveState();
        g.reduceClipRegion(static_cast<int>(cx), static_cast<int>(cy), 60, 30);
        g.setColour(juce::Colour::fromHSV(std::fmod(t * 3.0f + fi / 20.0f, 1.0f), 1.0f, 1.0f, 0.9f));
        g.fillEllipse(cx - 30.0f, cy - 30.0f, 120.0f, 90.0f);
        g.restoreState();
    }

    // --- 30 thick lines ---
    for (int i = 0; i < 30; i++)
    {
        float fi = static_cast<float>(i);
        float lx1 = std::fmod(fi * 41.0f + t * 300.0f, w);
        float ly1 = std::fmod(fi * 31.0f, h);
        float lx2 = lx1 + std::sin(phase + fi) * 80.0f;
        float ly2 = ly1 + std::cos(phase + fi) * 60.0f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 30.0f + t, 1.0f), 0.8f, 1.0f, 0.6f));
        g.drawLine(lx1, ly1, lx2, ly2, 2.0f + fi * 0.1f);
    }
}

void CompressorEditor::paintBenchmarkResults(juce::Graphics& g)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());

    g.fillAll(juce::Colour(0xFF0E0E1A));

    // Title
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(22.0f));
    g.drawText("Benchmark Results", 0, 10, static_cast<int>(w), 30, juce::Justification::centred);

    // Summary stats
    g.setFont(juce::FontOptions(14.0f));
    float col1 = w * 0.15f, col2 = w * 0.55f;
    float row = 50.0f;

    g.setColour(juce::Colour(0xFF6C63FF));
    g.drawText("VULKAN", static_cast<int>(col1), static_cast<int>(row), 150, 20, juce::Justification::centredLeft);
    g.setColour(juce::Colour(0xFFE07A5F));
    g.drawText("JUCE", static_cast<int>(col2), static_cast<int>(row), 150, 20, juce::Justification::centredLeft);

    row += 25.0f;
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(12.0f));
    g.drawText(juce::String(vulkanFPS, 1) + " FPS (" + juce::String(vulkanRenderedFrames) + " frames)",
               static_cast<int>(col1), static_cast<int>(row), 250, 18, juce::Justification::centredLeft);
    g.drawText(juce::String(juceFPS, 1) + " FPS (" + juce::String(juceRenderedFrames) + " frames)",
               static_cast<int>(col2), static_cast<int>(row), 250, 18, juce::Justification::centredLeft);

    // Speedup
    row += 25.0f;
    double speedup = juceFPS > 0 ? vulkanFPS / juceFPS : 0;
    g.setFont(juce::FontOptions(16.0f));
    g.setColour(speedup > 1.0 ? juce::Colour(0xFF4ECCA3) : juce::Colour(0xFFE84545));
    g.drawText("Vulkan is " + juce::String(speedup, 2) + "x " + (speedup > 1.0 ? "faster" : "slower"),
               0, static_cast<int>(row), static_cast<int>(w), 22, juce::Justification::centred);

    // Frame time chart
    float chartTop = row + 40.0f;
    float chartH = h - chartTop - 30.0f;
    float chartLeft = 50.0f;
    float chartW = w - 80.0f;

    // Chart background
    g.setColour(juce::Colour(0xFF16213E));
    g.fillRoundedRectangle(chartLeft - 5, chartTop - 5, chartW + 10, chartH + 10, 6.0f);

    // Find max frame time for Y axis
    double maxMs = 1.0;
    for (auto d : vulkanFrameTimes) maxMs = std::max(maxMs, d);
    for (auto d : juceFrameTimes) maxMs = std::max(maxMs, d);
    maxMs *= 1.1;

    // Y axis labels
    g.setColour(juce::Colours::grey);
    g.setFont(juce::FontOptions(9.0f));
    for (int i = 0; i <= 4; i++)
    {
        float y = chartTop + chartH * (1.0f - static_cast<float>(i) / 4.0f);
        double ms = maxMs * i / 4.0;
        g.drawText(juce::String(ms, 1) + "ms", 0, static_cast<int>(y) - 6, 45, 12,
                    juce::Justification::centredRight);
        g.setColour(juce::Colour(0x30FFFFFF));
        g.drawHorizontalLine(static_cast<int>(y), chartLeft, chartLeft + chartW);
        g.setColour(juce::Colours::grey);
    }

    // Plot Vulkan frame times
    // Plot with moving average to smooth out spikes
    auto plotLine = [&](const std::vector<double>& times, juce::Colour col)
    {
        if (times.size() < 2) return;
        g.setColour(col);
        juce::Path path;
        constexpr int windowSize = 10;
        for (size_t i = 0; i < times.size(); i++)
        {
            // Moving average
            double sum = 0;
            int count = 0;
            for (int j = -windowSize / 2; j <= windowSize / 2; j++)
            {
                int idx = static_cast<int>(i) + j;
                if (idx >= 0 && idx < static_cast<int>(times.size()))
                {
                    sum += times[static_cast<size_t>(idx)];
                    count++;
                }
            }
            double smoothed = sum / count;
            float x = chartLeft + (static_cast<float>(i) / static_cast<float>(times.size() - 1)) * chartW;
            float y = chartTop + chartH * (1.0f - static_cast<float>(smoothed / maxMs));
            if (i == 0) path.startNewSubPath(x, y);
            else path.lineTo(x, y);
        }
        g.strokePath(path, juce::PathStrokeType(2.0f));
    };

    plotLine(vulkanFrameTimes, juce::Colour(0xFF6C63FF));
    plotLine(juceFrameTimes, juce::Colour(0xFFE07A5F));

    // Legend
    float legendY = h - 25.0f;
    g.setColour(juce::Colour(0xFF6C63FF));
    g.fillRect(chartLeft, legendY, 12.0f, 12.0f);
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(11.0f));
    g.drawText("Vulkan", static_cast<int>(chartLeft + 16), static_cast<int>(legendY), 50, 12,
                juce::Justification::centredLeft);

    g.setColour(juce::Colour(0xFFE07A5F));
    g.fillRect(chartLeft + 80.0f, legendY, 12.0f, 12.0f);
    g.setColour(juce::Colours::white);
    g.drawText("JUCE", static_cast<int>(chartLeft + 96), static_cast<int>(legendY), 50, 12,
                juce::Justification::centredLeft);
}

// =============================================================================
// Paint
// =============================================================================

void CompressorEditor::paint(juce::Graphics& g)
{
    if (showResults && !benchmarking)
    {
        paintBenchmarkResults(g);
        return;
    }

    if (benchmarking)
    {
        double now = juce::Time::getMillisecondCounterHiRes();
        double phaseElapsed = now - benchPhaseStart;

        // Count this rendered frame
        benchFrame++;
        if (isVulkanEnabled())
            vulkanRenderedFrames++;
        else
            juceRenderedFrames++;

        // Record per-frame interval for chart
        if (lastPaintTime > 0)
        {
            double frameMs = now - lastPaintTime;
            if (isVulkanEnabled())
                vulkanFrameTimes.push_back(frameMs);
            else
                juceFrameTimes.push_back(frameMs);
        }
        lastPaintTime = now;

        // Paint the scene
        paintBenchmarkScene(g, benchFrame);

        // Check if phase is done (5 seconds)
        if (phaseElapsed >= benchDurationMs)
        {
            if (isVulkanEnabled())
            {
                // Switch to JUCE phase
                setVulkanEnabled(false);
                benchFrame = 0;
                lastPaintTime = 0;
                benchPhaseStart = juce::Time::getMillisecondCounterHiRes();
            }
            else
            {
                // Done
                stopTimer();
                benchmarking = false;
                setVulkanEnabled(true);

                vulkanFPS = vulkanRenderedFrames / (benchDurationMs / 1000.0);
                juceFPS = juceRenderedFrames / (benchDurationMs / 1000.0);

                showResults = true;
            }
        }
        return;
    }

    // Normal compressor UI
    g.fillAll(juce::Colour(0xFF1A1A2E));

    g.setColour(juce::Colour(0xFF16213E));
    g.fillRect(0, 0, getWidth(), 40);

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(18.0f));
    g.drawText("VULKAN COMPRESSOR", 15, 8, 300, 24, juce::Justification::centredLeft);

    auto meterArea = getLocalBounds().removeFromBottom(60).reduced(20, 5);
    g.setColour(juce::Colour(0xFF0F0F23));
    g.fillRoundedRectangle(meterArea.toFloat(), 6.0f);

    float inLevel = processor.inputLevel.load();
    float outLevel = processor.outputLevel.load();

    auto inBar = meterArea.reduced(4).removeFromLeft(meterArea.getWidth() / 2 - 4);
    auto outBar = meterArea.reduced(4).removeFromRight(meterArea.getWidth() / 2 - 4);

    g.setColour(juce::Colour(0xFF4ECCA3));
    g.fillRect(inBar.removeFromBottom(static_cast<int>(inBar.getHeight() * juce::jlimit(0.0f, 1.0f, inLevel))));

    g.setColour(juce::Colour(0xFF45B7D1));
    g.fillRect(outBar.removeFromBottom(static_cast<int>(outBar.getHeight() * juce::jlimit(0.0f, 1.0f, outLevel))));
}

void CompressorEditor::resized()
{
    vulkanToggle.setBounds(getWidth() - 80, 8, 65, 24);
    benchmarkButton.setBounds(getWidth() - 150, 8, 60, 24);

    auto area = getLocalBounds();
    area.removeFromTop(45);
    area.removeFromBottom(65);

    int knobW = area.getWidth() / 5;

    auto knobArea = area;
    auto labelArea = knobArea.removeFromBottom(20);

    thresholdSlider.setBounds(knobArea.removeFromLeft(knobW));
    ratioSlider.setBounds(knobArea.removeFromLeft(knobW));
    attackSlider.setBounds(knobArea.removeFromLeft(knobW));
    releaseSlider.setBounds(knobArea.removeFromLeft(knobW));
    makeupSlider.setBounds(knobArea);

    thresholdLabel.setBounds(labelArea.removeFromLeft(knobW));
    ratioLabel.setBounds(labelArea.removeFromLeft(knobW));
    attackLabel.setBounds(labelArea.removeFromLeft(knobW));
    releaseLabel.setBounds(labelArea.removeFromLeft(knobW));
    makeupLabel.setBounds(labelArea);
}
