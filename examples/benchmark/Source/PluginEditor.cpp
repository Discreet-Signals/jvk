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

static void styleButton(juce::TextButton& btn, const juce::String& text, juce::Colour col)
{
    btn.setButtonText(text);
    btn.setColour(juce::TextButton::buttonColourId, col);
    btn.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    btn.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
}

BenchmarkEditor::BenchmarkEditor(BenchmarkProcessor& p)
    : jvk::AudioProcessorEditor(p)
{
    styleButton(btnAll,        "ALL",          juce::Colour(0xFF2A9D8F));
    styleButton(btnText,       "TEXT",         juce::Colour(0xFF6C63FF));
    styleButton(btnTextStatic, "TEXT STATIC",  juce::Colour(0xFF5B5EA6));
    styleButton(btnPaths,      "COMPLEX PATHS",juce::Colour(0xFFE07A5F));
    styleButton(btnFills,      "FILLS",        juce::Colour(0xFFF4A261));

    btnAll.onClick        = [this] { runBenchmark(Mode::All); };
    btnText.onClick       = [this] { runBenchmark(Mode::Text); };
    btnTextStatic.onClick = [this] { runBenchmark(Mode::TextStatic); };
    btnPaths.onClick      = [this] { runBenchmark(Mode::ComplexPaths); };
    btnFills.onClick      = [this] { runBenchmark(Mode::Fills); };

    addAndMakeVisible(btnAll);
    addAndMakeVisible(btnText);
    addAndMakeVisible(btnTextStatic);
    addAndMakeVisible(btnPaths);
    addAndMakeVisible(btnFills);

    setSize(800, 600);
}

// =============================================================================
// Path generation — built once, reused every frame
// =============================================================================

void BenchmarkEditor::generatePaths(float w, float h)
{
    if (pathsGenerated) return;
    pathsGenerated = true;

    // 100 waveform paths — sine waves with harmonics
    waveformPaths.resize(100);
    for (int i = 0; i < 100; i++)
    {
        auto& p = waveformPaths[static_cast<size_t>(i)];
        float baseY = h * (static_cast<float>(i) + 0.5f) / 100.0f;
        float amp = 8.0f + static_cast<float>(i) * 0.5f;
        float freq = 2.0f + static_cast<float>(i % 8) * 0.5f;
        p.startNewSubPath(0, baseY);
        for (int x = 0; x <= static_cast<int>(w); x += 2)
        {
            float fx = static_cast<float>(x);
            float y = baseY + amp * std::sin(fx * freq * juce::MathConstants<float>::twoPi / w)
                             + amp * 0.3f * std::sin(fx * freq * 3.0f * juce::MathConstants<float>::twoPi / w)
                             + amp * 0.15f * std::sin(fx * freq * 7.0f * juce::MathConstants<float>::twoPi / w);
            p.lineTo(fx, y);
        }
        p.lineTo(w, baseY + amp);
        p.lineTo(w, baseY - amp);
        p.closeSubPath();
    }

    // 80 spectrum analyzer paths — jagged peaks
    spectrumPaths.resize(80);
    for (int i = 0; i < 80; i++)
    {
        auto& p = spectrumPaths[static_cast<size_t>(i)];
        float baseY = h * 0.8f;
        int numBins = 64 + i * 4;
        float binW = w / static_cast<float>(numBins);
        p.startNewSubPath(0, baseY);
        for (int b = 0; b < numBins; b++)
        {
            float fb = static_cast<float>(b);
            float binH = (std::sin(fb * 0.3f + static_cast<float>(i)) * 0.5f + 0.5f)
                        * h * 0.6f * std::exp(-fb / static_cast<float>(numBins) * 2.0f);
            p.lineTo(fb * binW, baseY - binH);
            p.lineTo((fb + 0.8f) * binW, baseY - binH);
            p.lineTo((fb + 0.9f) * binW, baseY);
        }
        p.lineTo(w, baseY);
        p.closeSubPath();
    }

    // 50 gear/mechanical shapes — concentric paths with teeth
    gearPaths.resize(50);
    for (int i = 0; i < 50; i++)
    {
        auto& p = gearPaths[static_cast<size_t>(i)];
        float cx = w * (0.05f + static_cast<float>(i % 10) * 0.1f);
        float cy = h * (0.1f + static_cast<float>(i / 10) * 0.2f);
        int teeth = 8 + i * 2;
        float outerR = 20.0f + static_cast<float>(i) * 3.0f;
        float innerR = outerR * 0.7f;
        float toothH = outerR * 0.25f;

        // Outer ring with teeth
        for (int t = 0; t < teeth; t++)
        {
            float a0 = static_cast<float>(t) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
            float a1 = (static_cast<float>(t) + 0.3f) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
            float a2 = (static_cast<float>(t) + 0.5f) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
            float a3 = (static_cast<float>(t) + 0.7f) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
            float a4 = (static_cast<float>(t) + 1.0f) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;

            if (t == 0) p.startNewSubPath(cx + outerR * std::cos(a0), cy + outerR * std::sin(a0));
            p.lineTo(cx + outerR * std::cos(a1), cy + outerR * std::sin(a1));
            p.lineTo(cx + (outerR + toothH) * std::cos(a1), cy + (outerR + toothH) * std::sin(a1));
            p.lineTo(cx + (outerR + toothH) * std::cos(a3), cy + (outerR + toothH) * std::sin(a3));
            p.lineTo(cx + outerR * std::cos(a3), cy + outerR * std::sin(a3));
            p.lineTo(cx + outerR * std::cos(a4), cy + outerR * std::sin(a4));
        }
        p.closeSubPath();

        // Inner hole (opposite winding for cutout)
        float holeR = innerR * 0.4f;
        p.startNewSubPath(cx + holeR, cy);
        for (int s = 1; s <= 32; s++)
        {
            float a = -static_cast<float>(s) / 32.0f * juce::MathConstants<float>::twoPi;
            p.lineTo(cx + holeR * std::cos(a), cy + holeR * std::sin(a));
        }
        p.closeSubPath();
    }
}

// =============================================================================
// Benchmark control
// =============================================================================

void BenchmarkEditor::runBenchmark(Mode mode)
{
    if (benchmarking) return;

    activeMode = mode;
    switch (mode)
    {
        case Mode::All:          modeName = "ALL"; break;
        case Mode::Text:         modeName = "TEXT"; break;
        case Mode::TextStatic:   modeName = "TEXT STATIC"; break;
        case Mode::ComplexPaths: modeName = "COMPLEX PATHS"; break;
        case Mode::Fills:        modeName = "FILLS"; break;
    }

    generatePaths(static_cast<float>(getWidth()), static_cast<float>(getHeight()));

    showResults = false;
    vulkanFrameTimes.clear();
    juceFrameTimes.clear();
    vulkanRenderedFrames = 0;
    juceRenderedFrames = 0;
    lastPaintTime = 0;

    setVulkanEnabled(true);
    benchmarking = true;
    benchVulkanPhase = true;
    benchFrame = 0;
    benchPhaseStart = juce::Time::getMillisecondCounterHiRes();

    savedPresentMode = getVulkanRenderer().getSettings().presentMode;
    getVulkanRenderer().setPresentMode(VK_PRESENT_MODE_IMMEDIATE_KHR);
}

void BenchmarkEditor::timerCallback()
{
    if (!benchmarking || benchVulkanPhase) { stopTimer(); return; }
    repaint();
}

// =============================================================================
// Scene: ALL — everything at once
// =============================================================================

void BenchmarkEditor::sceneAll(juce::Graphics& g, float w, float h, float t, float phase)
{
    // 800 filled rectangles
    for (int i = 0; i < 800; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::sin(phase + fi * 0.3f) * w * 0.45f;
        float y = h * 0.5f + std::cos(phase + fi * 0.5f) * h * 0.45f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 800.0f + t, 1.0f), 0.8f, 0.9f, 0.4f));
        g.fillRect(x - 10.0f, y - 10.0f, 20.0f, 20.0f);
    }

    // 600 rounded rectangles
    for (int i = 0; i < 600; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 23.0f + t * w * 2.0f, w + 100.0f) - 50.0f;
        float y = std::fmod(fi * 17.0f + t * h, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 600.0f + t * 0.5f, 1.0f), 0.7f, 0.85f, 0.5f));
        g.fillRoundedRectangle(x, y, 50.0f + fi * 0.05f, 25.0f + fi * 0.03f, 4.0f + std::fmod(fi * 3.0f, 12.0f));
    }

    // 400 ellipses
    for (int i = 0; i < 400; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::cos(phase * 0.8f + fi * 0.7f) * w * 0.4f;
        float y = h * 0.5f + std::sin(phase * 0.6f + fi * 0.9f) * h * 0.4f;
        float rx = 5.0f + fi * 0.08f, ry = 5.0f + fi * 0.05f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 400.0f + t * 2.0f, 1.0f), 0.9f, 1.0f, 0.35f));
        g.fillEllipse(x - rx, y - ry, rx * 2, ry * 2);
    }

    // 200 bezier paths
    for (int i = 0; i < 200; i++)
    {
        float fi = static_cast<float>(i);
        juce::Path p;
        float sx = std::fmod(fi * 53.0f + t * 200.0f, w);
        float sy = std::fmod(fi * 37.0f + t * 100.0f, h);
        p.startNewSubPath(sx, sy);
        for (int seg = 0; seg < 6; seg++)
        {
            float fs = static_cast<float>(seg);
            p.cubicTo(sx + (30.0f + fs * 20.0f) * std::sin(phase + fi + fs),
                      sy + (40.0f + fs * 15.0f) * std::cos(phase + fi + fs * 0.7f),
                      sx + (60.0f + fs * 25.0f) * std::cos(phase * 0.5f + fi + fs),
                      sy + (20.0f + fs * 30.0f) * std::sin(phase * 0.7f + fi + fs),
                      sx + (90.0f + fs * 30.0f), sy + (30.0f + fs * 20.0f));
        }
        p.closeSubPath();
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 200.0f + t, 1.0f), 0.6f, 0.95f, 0.3f));
        g.fillPath(p);
    }

    // 120 stroked stars
    for (int i = 0; i < 120; i++)
    {
        float fi = static_cast<float>(i);
        juce::Path p;
        p.addStar({ std::fmod(fi * 67.0f + t * 300.0f, w), std::fmod(fi * 43.0f + t * 150.0f, h) },
                  5 + (i % 6), 6.0f + fi * 0.2f, 15.0f + fi * 0.3f);
        g.setColour(juce::Colours::white.withAlpha(0.5f));
        g.strokePath(p, juce::PathStrokeType(1.5f + fi * 0.05f));
    }

    // 80 gradient rectangles
    for (int i = 0; i < 80; i++)
    {
        float fi = static_cast<float>(i);
        float gx = std::fmod(fi * 57.0f, w), gy = std::fmod(fi * 41.0f + t * 100.0f, h);
        juce::ColourGradient grad(
            juce::Colour::fromHSV(std::fmod(t + fi / 80.0f, 1.0f), 0.9f, 1.0f, 0.7f), gx, gy,
            juce::Colour::fromHSV(std::fmod(t + fi / 80.0f + 0.4f, 1.0f), 0.9f, 0.4f, 0.7f), gx + 80.0f, gy + 40.0f, (i % 3 == 0));
        g.setGradientFill(grad);
        g.fillRoundedRectangle(gx, gy, 80.0f, 40.0f, 6.0f);
    }

    // 200 text labels
    juce::String texts[] = { "Vulkan GPU", "SDF Shapes", "Stencil Paths", "2x Super",
        "sRGB Gamma", "Lock Free", "Zero Alloc", "EdgeTable", "Bezier Cubic",
        "Anti-Alias", "Cross Plat", "Real-Time", "MoltenVK", "Pipeline", "Descriptor", "Framebuffer" };
    for (int i = 0; i < 200; i++)
    {
        float fi = static_cast<float>(i);
        g.setFont(juce::FontOptions(8.0f + std::fmod(fi * 3.7f, 16.0f)));
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 200.0f + t, 1.0f), 0.3f, 1.0f, 0.8f));
        g.drawText(texts[i % 16], static_cast<int>(std::fmod(fi * 43.0f + t * 200.0f, w + 200.0f) - 100.0f),
                   static_cast<int>(std::fmod(fi * 29.0f + t * 80.0f, h)), 150, 20, juce::Justification::centredLeft);
    }

    // 120 thick lines
    for (int i = 0; i < 120; i++)
    {
        float fi = static_cast<float>(i);
        float lx1 = std::fmod(fi * 41.0f + t * 300.0f, w), ly1 = std::fmod(fi * 31.0f, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 120.0f + t, 1.0f), 0.8f, 1.0f, 0.6f));
        g.drawLine(lx1, ly1, lx1 + std::sin(phase + fi) * 80.0f, ly1 + std::cos(phase + fi) * 60.0f, 2.0f + fi * 0.05f);
    }
}

// =============================================================================
// Scene: TEXT — moving text everywhere
// =============================================================================

void BenchmarkEditor::sceneText(juce::Graphics& g, float w, float h, float t, float phase)
{
    juce::String words[] = {
        "The quick brown fox", "jumps over the lazy dog", "VULKAN RENDERING",
        "GPU Accelerated", "Real-Time Audio", "SDF Text Atlas",
        "Multi-channel signed distance", "Sub-pixel precision",
        "Glyph cache hit rate", "Font rasterization", "Kerning pairs",
        "Baseline alignment", "Ascender/Descender", "Anti-aliased edges",
        "Resolution independent", "Dynamic font sizing"
    };

    // 3000 moving text labels — all on screen, overlapping, varied sizes
    float sizes[] = { 6, 7, 8, 9, 10, 11, 12, 14, 16, 18 };
    for (int i = 0; i < 3000; i++)
    {
        float fi = static_cast<float>(i);
        float fontSize = sizes[i % 10] + std::fmod(fi * 1.7f, 8.0f);
        // Position wraps within viewport — always on screen
        float x = std::fmod(fi * 37.0f + t * 100.0f + std::sin(phase + fi * 0.05f) * 50.0f, w * 0.85f);
        float y = std::fmod(fi * 11.3f + std::cos(phase * 0.7f + fi * 0.03f) * 20.0f, h - fontSize);

        g.setFont(juce::FontOptions(fontSize));
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 3000.0f + t * 0.3f, 1.0f), 0.5f, 1.0f, 0.85f));
        g.drawText(words[i % 16], static_cast<int>(x), static_cast<int>(y),
                   static_cast<int>(w * 0.3f), static_cast<int>(fontSize * 1.5f),
                   juce::Justification::centredLeft);
    }
}

// =============================================================================
// Scene: TEXT STATIC — wall of text, no animation
// =============================================================================

void BenchmarkEditor::sceneTextStatic(juce::Graphics& g, float w, float h)
{
    juce::String paragraph =
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur";

    auto words = juce::StringArray::fromTokens(paragraph, " ", "");

    // Brute force: 20000 words placed in a dense grid, all on screen, overlapping layers.
    // 10 layers at different font sizes stacked on top of each other.
    float sizes[] = { 7, 8, 9, 10, 11, 12, 13, 14, 16, 18 };
    int wordIdx = 0;

    for (int layer = 0; layer < 10; layer++)
    {
        float fontSize = sizes[layer];
        g.setFont(juce::FontOptions(fontSize));
        float lineH = fontSize * 1.2f;
        float charW = fontSize * 0.55f;
        int wordsPerRow = static_cast<int>(w / (charW * 6.0f)); // ~6 chars avg word
        if (wordsPerRow < 1) wordsPerRow = 1;
        int numRows = static_cast<int>(h / lineH);
        int wordsThisLayer = wordsPerRow * numRows;

        float curX = 0, curY = static_cast<float>(layer) * 1.5f; // slight offset per layer

        for (int i = 0; i < wordsThisLayer && i < 2000; i++)
        {
            auto& word = words[wordIdx % words.size()];
            wordIdx++;
            float wordW = charW * static_cast<float>(word.length());

            if (curX + wordW > w) { curX = 0; curY += lineH; }
            if (curY > h) break;

            g.setColour(juce::Colour::fromHSV(
                std::fmod(static_cast<float>(layer) * 0.1f + static_cast<float>(i) * 0.0005f, 1.0f),
                0.15f + static_cast<float>(layer) * 0.05f, 0.95f, 0.9f));
            g.drawText(word, static_cast<int>(curX), static_cast<int>(curY),
                       static_cast<int>(wordW + 2.0f), static_cast<int>(lineH),
                       juce::Justification::centredLeft);
            curX += wordW + charW * 0.5f;
        }
    }
}

// =============================================================================
// Scene: COMPLEX PATHS — pregenerated waveforms, spectrums, gears with transforms
// =============================================================================

void BenchmarkEditor::sceneComplexPaths(juce::Graphics& g, float w, float h, float t, float phase)
{
    // Waveforms with vertical scrolling and scale pulsing
    for (size_t i = 0; i < waveformPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        float yOff = std::sin(phase * 0.5f + fi * 0.3f) * 15.0f;
        float scaleX = 0.9f + 0.1f * std::sin(phase * 0.3f + fi * 0.2f);

        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 100.0f + t * 0.5f, 1.0f), 0.7f, 0.9f, 0.4f));
        g.fillPath(waveformPaths[i], juce::AffineTransform::translation(0.0f, yOff).scaled(scaleX, 1.0f));
    }

    // Spectrums with horizontal slide
    for (size_t i = 0; i < spectrumPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        float xOff = std::sin(phase * 0.7f + fi * 0.4f) * 20.0f;
        float yScale = 0.5f + 0.5f * std::sin(phase * 0.4f + fi * 0.15f);

        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 80.0f + t, 1.0f), 0.9f, 1.0f, 0.3f));
        g.fillPath(spectrumPaths[i], juce::AffineTransform::translation(xOff, 0).scaled(1.0f, yScale, w * 0.5f, h * 0.8f));
    }

    // Gears rotating
    for (size_t i = 0; i < gearPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        float cx = w * (0.05f + static_cast<float>(i % 10) * 0.1f);
        float cy = h * (0.1f + static_cast<float>(i / 10) * 0.2f);
        float angle = phase * (0.5f + fi * 0.1f) * ((i % 2 == 0) ? 1.0f : -1.0f);

        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 50.0f + t * 0.3f, 1.0f), 0.6f, 0.85f, 0.5f));
        g.fillPath(gearPaths[i], juce::AffineTransform::rotation(angle, cx, cy));
    }

    // 700 additional dynamic cubic paths
    for (int i = 0; i < 700; i++)
    {
        float fi = static_cast<float>(i);
        juce::Path p;
        float sx = std::fmod(fi * 47.0f + t * 150.0f, w);
        float sy = std::fmod(fi * 31.0f + t * 80.0f, h);
        p.startNewSubPath(sx, sy);
        for (int seg = 0; seg < 8; seg++)
        {
            float fs = static_cast<float>(seg);
            p.cubicTo(sx + (20.0f + fs * 15.0f) * std::sin(phase * 1.5f + fi + fs),
                      sy + (30.0f + fs * 10.0f) * std::cos(phase * 1.2f + fi + fs * 0.8f),
                      sx + (50.0f + fs * 20.0f) * std::cos(phase * 0.8f + fi + fs),
                      sy + (15.0f + fs * 25.0f) * std::sin(phase * 0.6f + fi + fs),
                      sx + (70.0f + fs * 25.0f), sy + (25.0f + fs * 15.0f));
        }
        p.closeSubPath();
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 700.0f + t * 2.0f, 1.0f), 0.8f, 0.95f, 0.25f));
        g.fillPath(p);
    }
}

// =============================================================================
// Scene: FILLS — massive fill workload
// =============================================================================

void BenchmarkEditor::sceneFills(juce::Graphics& g, float w, float h, float t, float phase)
{
    // 20000 filled rectangles
    for (int i = 0; i < 20000; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 13.0f + t * 200.0f, w + 40.0f) - 20.0f;
        float y = std::fmod(fi * 7.0f + t * 100.0f, h + 40.0f) - 20.0f;
        float s = 3.0f + std::fmod(fi * 3.7f, 15.0f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 20000.0f + t, 1.0f), 0.8f, 0.9f, 0.3f));
        g.fillRect(x, y, s, s);
    }

    // 15000 rounded rectangles
    for (int i = 0; i < 15000; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 19.0f + t * w, w + 60.0f) - 30.0f;
        float y = std::fmod(fi * 11.0f + t * h * 0.7f, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 15000.0f + t * 0.7f, 1.0f), 0.7f, 0.85f, 0.4f));
        g.fillRoundedRectangle(x, y, 20.0f + fi * 0.001f, 10.0f + fi * 0.0005f, 3.0f + std::fmod(fi, 6.0f));
    }

    // 10000 ellipses
    for (int i = 0; i < 10000; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::cos(phase + fi * 0.3f) * w * 0.48f;
        float y = h * 0.5f + std::sin(phase * 0.7f + fi * 0.5f) * h * 0.48f;
        float r = 2.0f + fi * 0.002f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 10000.0f + t * 1.5f, 1.0f), 0.9f, 1.0f, 0.2f));
        g.fillEllipse(x - r, y - r, r * 2, r * 2);
    }

    // 1000 gradient fills
    for (int i = 0; i < 1000; i++)
    {
        float fi = static_cast<float>(i);
        float gx = std::fmod(fi * 37.0f + t * 80.0f, w);
        float gy = std::fmod(fi * 29.0f + t * 50.0f, h);
        juce::ColourGradient grad(
            juce::Colour::fromHSV(std::fmod(t + fi / 1000.0f, 1.0f), 0.9f, 1.0f, 0.6f), gx, gy,
            juce::Colour::fromHSV(std::fmod(t + fi / 1000.0f + 0.5f, 1.0f), 0.9f, 0.5f, 0.6f),
            gx + 40.0f, gy + 20.0f, (i % 4 == 0));
        g.setGradientFill(grad);
        g.fillRoundedRectangle(gx, gy, 40.0f, 20.0f, 4.0f);
    }

    // 4000 lines
    for (int i = 0; i < 4000; i++)
    {
        float fi = static_cast<float>(i);
        float lx = std::fmod(fi * 31.0f + t * 200.0f, w);
        float ly = std::fmod(fi * 23.0f, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 4000.0f + t, 1.0f), 0.7f, 1.0f, 0.4f));
        g.drawLine(lx, ly, lx + std::sin(phase + fi) * 40.0f, ly + std::cos(phase + fi) * 30.0f, 1.0f + fi * 0.001f);
    }
}

// =============================================================================
// Scene dispatch
// =============================================================================

void BenchmarkEditor::paintBenchmarkScene(juce::Graphics& g, int frame)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());
    float t = static_cast<float>(frame % 300) / 300.0f;
    float phase = t * juce::MathConstants<float>::twoPi * 3.0f;

    g.fillAll(juce::Colour(0xFF0E0E1A));

    switch (activeMode)
    {
        case Mode::All:          sceneAll(g, w, h, t, phase); break;
        case Mode::Text:         sceneText(g, w, h, t, phase); break;
        case Mode::TextStatic:   sceneTextStatic(g, w, h); break;
        case Mode::ComplexPaths: sceneComplexPaths(g, w, h, t, phase); break;
        case Mode::Fills:        sceneFills(g, w, h, t, phase); break;
    }
}

// =============================================================================
// Results
// =============================================================================

void BenchmarkEditor::paintBenchmarkResults(juce::Graphics& g)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());

    g.fillAll(juce::Colour(0xFF0E0E1A));

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(22.0f));
    g.drawText("Benchmark: " + modeName, 0, 10, static_cast<int>(w), 30, juce::Justification::centred);

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

    row += 25.0f;
    double speedup = juceFPS > 0 ? vulkanFPS / juceFPS : 0;
    g.setFont(juce::FontOptions(16.0f));
    g.setColour(speedup > 1.0 ? juce::Colour(0xFF4ECCA3) : juce::Colour(0xFFE84545));
    g.drawText("Vulkan is " + juce::String(speedup, 2) + "x " + (speedup > 1.0 ? "faster" : "slower"),
               0, static_cast<int>(row), static_cast<int>(w), 22, juce::Justification::centred);

    // Frame time chart
    float chartTop = row + 40.0f;
    float chartH = h - chartTop - 30.0f;
    float chartLeft = 50.0f, chartW = w - 80.0f;

    g.setColour(juce::Colour(0xFF16213E));
    g.fillRoundedRectangle(chartLeft - 5, chartTop - 5, chartW + 10, chartH + 10, 6.0f);

    double maxMs = 1.0;
    for (auto d : vulkanFrameTimes) maxMs = std::max(maxMs, d);
    for (auto d : juceFrameTimes) maxMs = std::max(maxMs, d);
    maxMs *= 1.1;

    g.setColour(juce::Colours::grey);
    g.setFont(juce::FontOptions(9.0f));
    for (int i = 0; i <= 4; i++)
    {
        float y = chartTop + chartH * (1.0f - static_cast<float>(i) / 4.0f);
        g.drawText(juce::String(maxMs * i / 4.0, 1) + "ms", 0, static_cast<int>(y) - 6, 45, 12, juce::Justification::centredRight);
        g.setColour(juce::Colour(0x30FFFFFF));
        g.drawHorizontalLine(static_cast<int>(y), chartLeft, chartLeft + chartW);
        g.setColour(juce::Colours::grey);
    }

    auto plotLine = [&](const std::vector<double>& times, juce::Colour col)
    {
        if (times.size() < 2) return;
        g.setColour(col);
        juce::Path path;
        constexpr int win = 10;
        for (size_t i = 0; i < times.size(); i++)
        {
            double sum = 0; int count = 0;
            for (int j = -win/2; j <= win/2; j++)
            {
                int idx = static_cast<int>(i) + j;
                if (idx >= 0 && idx < static_cast<int>(times.size())) { sum += times[static_cast<size_t>(idx)]; count++; }
            }
            float x = chartLeft + (static_cast<float>(i) / static_cast<float>(times.size() - 1)) * chartW;
            float y = chartTop + chartH * (1.0f - static_cast<float>((sum / count) / maxMs));
            if (i == 0) path.startNewSubPath(x, y); else path.lineTo(x, y);
        }
        g.strokePath(path, juce::PathStrokeType(2.0f));
    };

    plotLine(vulkanFrameTimes, juce::Colour(0xFF6C63FF));
    plotLine(juceFrameTimes, juce::Colour(0xFFE07A5F));

    float legendY = h - 25.0f;
    g.setColour(juce::Colour(0xFF6C63FF)); g.fillRect(chartLeft, legendY, 12.0f, 12.0f);
    g.setColour(juce::Colours::white); g.setFont(juce::FontOptions(11.0f));
    g.drawText("Vulkan", static_cast<int>(chartLeft + 16), static_cast<int>(legendY), 50, 12, juce::Justification::centredLeft);
    g.setColour(juce::Colour(0xFFE07A5F)); g.fillRect(chartLeft + 80.0f, legendY, 12.0f, 12.0f);
    g.setColour(juce::Colours::white);
    g.drawText("JUCE", static_cast<int>(chartLeft + 96), static_cast<int>(legendY), 50, 12, juce::Justification::centredLeft);
}

// =============================================================================
// Paint
// =============================================================================

void BenchmarkEditor::paint(juce::Graphics& g)
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

        benchFrame++;
        if (benchVulkanPhase)
        {
            vulkanRenderedFrames++;
            if (lastPaintTime > 0) vulkanFrameTimes.push_back(now - lastPaintTime);
        }
        else
        {
            juceRenderedFrames++;
            if (lastPaintTime > 0) juceFrameTimes.push_back(now - lastPaintTime);
        }
        lastPaintTime = now;

        paintBenchmarkScene(g, benchFrame);

        if (phaseElapsed >= benchDurationMs)
        {
            if (benchVulkanPhase)
            {
                setVulkanEnabled(false);
                benchVulkanPhase = false;
                benchFrame = 0;
                lastPaintTime = 0;
                benchPhaseStart = juce::Time::getMillisecondCounterHiRes();
                startTimerHz(1000);
            }
            else
            {
                stopTimer();
                benchmarking = false;
                getVulkanRenderer().setPresentMode(savedPresentMode);
                setVulkanEnabled(true);
                vulkanFPS = vulkanRenderedFrames / (benchDurationMs / 1000.0);
                juceFPS = juceRenderedFrames / (benchDurationMs / 1000.0);
                showResults = true;
            }
        }
        return;
    }

    // Idle screen
    g.fillAll(juce::Colour(0xFF0E0E1A));
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(28.0f));
    g.drawText("JVK Benchmark", 0, getHeight() / 2 - 80, getWidth(), 35, juce::Justification::centred);
    g.setFont(juce::FontOptions(14.0f));
    g.setColour(juce::Colour(0xFFAAAAAA));
    g.drawText("Select a benchmark to compare Vulkan vs JUCE rendering", 0, getHeight() / 2 - 40, getWidth(), 20, juce::Justification::centred);
}

void BenchmarkEditor::resized()
{
    int btnW = 130, btnH = 32, gap = 10;
    int totalW = 5 * btnW + 4 * gap;
    int startX = (getWidth() - totalW) / 2;
    int y = getHeight() / 2;

    btnAll.setBounds(startX, y, btnW, btnH); startX += btnW + gap;
    btnText.setBounds(startX, y, btnW, btnH); startX += btnW + gap;
    btnTextStatic.setBounds(startX, y, btnW, btnH); startX += btnW + gap;
    btnPaths.setBounds(startX, y, btnW, btnH); startX += btnW + gap;
    btnFills.setBounds(startX, y, btnW, btnH);
}
