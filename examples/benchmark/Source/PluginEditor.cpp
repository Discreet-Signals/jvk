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
#include "BinaryData.h"
#include <numeric>

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
    styleButton(btnImages,    "IMAGES",       juce::Colour(0xFF45B7D1));
    styleButton(btnBackends,  "BACKENDS",     juce::Colour(0xFFE84545));

    btnAll.onClick        = [this] { runBenchmark(Mode::All); };
    btnText.onClick       = [this] { runBenchmark(Mode::Text); };
    btnTextStatic.onClick = [this] { runBenchmark(Mode::TextStatic); };
    btnPaths.onClick      = [this] { runBenchmark(Mode::ComplexPaths); };
    btnFills.onClick      = [this] { runBenchmark(Mode::Fills); };
    btnImages.onClick     = [this] { runBenchmark(Mode::Images); };
    btnBackends.onClick   = [this] { runBenchmark(Mode::BackendComparison); };

    addAndMakeVisible(btnAll);
    addAndMakeVisible(btnText);
    addAndMakeVisible(btnTextStatic);
    addAndMakeVisible(btnPaths);
    addAndMakeVisible(btnFills);
    addAndMakeVisible(btnImages);
    addAndMakeVisible(btnBackends);

    styleButton(btnStencil, "Stencil", juce::Colour(0xFFE07A5F));
    addAndMakeVisible(btnStencil);

    auto fullBark = juce::ImageFileFormat::loadFrom(BinaryData::bark_png, BinaryData::bark_pngSize);
    barkImage = fullBark.rescaled(48, 48, juce::Graphics::highResamplingQuality);

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
        case Mode::Images:       modeName = "IMAGES"; break;
        case Mode::BackendComparison: modeName = "BACKENDS"; break;
    }

    if (mode == Mode::BackendComparison)
    {
        startBackendComparison();
        return;
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
        "sRGB Gamma", "Lock Free", "Zero Alloc", "GPU Cache", "Bezier Cubic",
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
// Scene: IMAGES — 1000 bouncing sprites
// =============================================================================

void BenchmarkEditor::initSprites(float w, float h)
{
    if (spritesInitialized) return;
    spritesInitialized = true;

    juce::Random rng(42);
    sprites.resize(20000);
    for (auto& s : sprites)
    {
        s.x = rng.nextFloat() * w;
        s.y = rng.nextFloat() * h;
        s.vx = (rng.nextFloat() - 0.5f) * 6.0f;
        s.vy = (rng.nextFloat() - 0.5f) * 6.0f;
    }
}

void BenchmarkEditor::sceneImages(juce::Graphics& g, float w, float h)
{
    if (!barkImage.isValid()) return;
    initSprites(w, h);

    constexpr float spriteSize = 48.0f;

    for (auto& s : sprites)
    {
        s.x += s.vx;
        s.y += s.vy;

        if (s.x < 0)              { s.x = 0;              s.vx = std::abs(s.vx); }
        if (s.x + spriteSize > w) { s.x = w - spriteSize;  s.vx = -std::abs(s.vx); }
        if (s.y < 0)              { s.y = 0;              s.vy = std::abs(s.vy); }
        if (s.y + spriteSize > h) { s.y = h - spriteSize;  s.vy = -std::abs(s.vy); }

        g.drawImage(barkImage, juce::Rectangle<float>(s.x, s.y, spriteSize, spriteSize));
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
        case Mode::Images:       sceneImages(g, w, h); break;
        case Mode::BackendComparison: break; // handled separately
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
    // Backend comparison test — independent state machine
    if (backendTest.running)
    {
        auto& bt = backendTest;
        auto& backend = bt.backends[static_cast<size_t>(bt.currentBackendIdx)];

        double t0 = juce::Time::getMillisecondCounterHiRes();

        int pathIdx = bt.complexities[static_cast<size_t>(bt.currentComplexityIdx)];
        auto& path = testPaths[static_cast<size_t>(pathIdx)];
        int segs = testPathSegCounts[static_cast<size_t>(pathIdx)];
        g.fillAll(juce::Colour(0xFF0E0E1A));
        float w = static_cast<float>(getWidth()), h = static_cast<float>(getHeight());
        float anim = static_cast<float>(bt.globalFrame) * 0.02f;

        for (int i = 0; i < bt.pathsPerFrame; i++)
        {
            float fi = static_cast<float>(i);
            float x = std::fmod(fi * 7.3f + std::sin(anim + fi * 0.1f) * 30.0f, w * 0.85f);
            float y = std::fmod(fi * 5.1f + std::cos(anim * 0.7f + fi * 0.07f) * 20.0f, h * 0.85f);
            float scale = 0.8f + 0.2f * std::sin(anim * 0.5f + fi * 0.05f);
            g.setColour(juce::Colour::fromHSV(std::fmod(fi / static_cast<float>(bt.pathsPerFrame) + anim * 0.1f, 1.0f),
                                               0.7f, 0.9f, 0.4f));
            g.fillPath(path, juce::AffineTransform::scale(scale).translated(x, y));
        }

        double frameMs = juce::Time::getMillisecondCounterHiRes() - t0;
        bt.frameTimeAccum += frameMs;
        bt.framesRendered++;
        bt.globalFrame++;

        g.setColour(juce::Colour(0xCC000000));
        g.fillRect(0, 0, getWidth(), 30);
        g.setColour(juce::Colours::white);
        g.setFont(juce::FontOptions(12.0f));
        g.drawText(backend.name + " | " + juce::String(segs) + " segs | "
                   + juce::String(bt.pathsPerFrame) + " paths | "
                   + juce::String(frameMs, 1) + "ms | "
                   + juce::String(bt.currentComplexityIdx + 1) + "/" + juce::String(static_cast<int>(bt.complexities.size()))
                   + " x " + juce::String(bt.currentBackendIdx + 1) + "/" + juce::String(static_cast<int>(bt.backends.size())),
                   10, 5, getWidth() - 20, 20, juce::Justification::centredLeft);

        if (bt.framesRendered >= bt.framesPerTest)
            advanceBackendTest();

        return;
    }

    if (backendTest.complete)
    {
        paintBackendResults(g);
        // Click anywhere to dismiss
        if (juce::ModifierKeys::currentModifiers.isLeftButtonDown())
        {
            backendTest.complete = false;
            showResults = false;
            repaint();
        }
        return;
    }

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
    // Backend selector row at top-right
    int beW = 95, beH = 24, beGap = 4;
    int beStartX = getWidth() - (beW + beGap) - 6;
    int beY = 4;
    btnStencil.setBounds(beStartX, beY, beW, beH);

    // Benchmark buttons centered in two rows
    int btnW = 120, btnH = 40, gap = 10;
    int totalW = 4 * btnW + 3 * gap;
    int startX = (getWidth() - totalW) / 2;
    int row1Y = getHeight() / 2;
    int row2Y = row1Y + btnH + gap;

    btnAll.setBounds(startX, row1Y, btnW, btnH);
    btnText.setBounds(startX + btnW + gap, row1Y, btnW, btnH);
    btnTextStatic.setBounds(startX + 2 * (btnW + gap), row1Y, btnW, btnH);
    btnPaths.setBounds(startX + 3 * (btnW + gap), row1Y, btnW, btnH);
    btnFills.setBounds(startX, row2Y, btnW, btnH);
    btnImages.setBounds(startX + btnW + gap, row2Y, btnW, btnH);
    btnBackends.setBounds(startX + 2 * (btnW + gap), row2Y, btnW, btnH);
}

// =============================================================================
// Backend comparison test
// =============================================================================

void BenchmarkEditor::generateTestPaths()
{
    if (testPathsGenerated) return;
    testPathsGenerated = true;

    std::vector<int> targetSegments = { 10, 100, 1000, 10000 };

    testPaths.clear();
    testPaths.reserve(targetSegments.size());
    testPathSegCounts = targetSegments;

    for (int target : targetSegments)
    {
        juce::Path p;
        float cx = 100.0f, cy = 100.0f;
        p.startNewSubPath(cx, cy);

        for (int i = 0; i < target; i++)
        {
            float angle = static_cast<float>(i) * 0.31f;
            float r = 30.0f + static_cast<float>(i % 200) * 0.5f;

            // Every 40 segments, start a new subpath (compound shape with holes)
            if (i > 0 && i % 40 == 0)
            {
                p.closeSubPath();
                float holeAngle = static_cast<float>(i) * 0.7f;
                p.startNewSubPath(cx + std::cos(holeAngle) * 20.0f,
                                  cy + std::sin(holeAngle) * 20.0f);
            }

            if (i % 3 == 0)
            {
                p.cubicTo(
                    cx + r * std::cos(angle) * 1.3f,
                    cy + r * std::sin(angle) * 0.8f,
                    cx + r * std::cos(angle + 0.5f) * 0.7f,
                    cy + r * std::sin(angle + 0.5f) * 1.2f,
                    cx + r * std::cos(angle + 1.0f),
                    cy + r * std::sin(angle + 1.0f));
            }
            else
            {
                p.lineTo(cx + r * std::cos(angle),
                         cy + r * std::sin(angle));
            }
        }
        p.closeSubPath();
        testPaths.push_back(std::move(p));
    }
}

void BenchmarkEditor::startBackendComparison()
{
    if (backendTest.running) return;

    generateTestPaths();

    backendTest = {};

    backendTest.complexities.resize(static_cast<int>(testPaths.size()));
    std::iota(backendTest.complexities.begin(), backendTest.complexities.end(), 0);

    backendTest.backends.clear();
    backendTest.backends.push_back({ "Stencil", juce::Colour(0xFFE07A5F), {} });

    for (auto& b : backendTest.backends)
        b.frameTimes.resize(backendTest.complexities.size(), 0.0);

    backendTest.currentBackendIdx = 0;
    backendTest.currentComplexityIdx = 0;
    backendTest.framesRendered = 0;
    backendTest.frameTimeAccum = 0;

    setVulkanEnabled(true);
    savedPresentMode = getVulkanRenderer().getSettings().presentMode;
    getVulkanRenderer().setPresentMode(VK_PRESENT_MODE_IMMEDIATE_KHR);
    backendTest.running = true;
    benchmarking = true;
    showResults = false;
    backendTest.complete = false;
}

void BenchmarkEditor::advanceBackendTest()
{
    auto& bt = backendTest;

    double avgMs = bt.frameTimeAccum / static_cast<double>(bt.framesRendered);
    bt.backends[static_cast<size_t>(bt.currentBackendIdx)]
        .frameTimes[static_cast<size_t>(bt.currentComplexityIdx)] = avgMs;

    bt.frameTimeAccum = 0;
    bt.framesRendered = 0;

    bt.currentComplexityIdx++;
    if (bt.currentComplexityIdx >= static_cast<int>(bt.complexities.size()))
    {
        bt.currentComplexityIdx = 0;
        bt.currentBackendIdx++;

        if (bt.currentBackendIdx >= static_cast<int>(bt.backends.size()))
        {
            bt.running = false;
            bt.complete = true;
            benchmarking = false;
            getVulkanRenderer().setPresentMode(savedPresentMode);
            return;
        }
    }
}

void BenchmarkEditor::paintBackendResults(juce::Graphics& g)
{
    float w = static_cast<float>(getWidth());
    float h = static_cast<float>(getHeight());

    g.fillAll(juce::Colour(0xFF0E0E1A));

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(22.0f));
    g.drawText("Backend Comparison — Frame Time vs Path Complexity",
               0, 10, static_cast<int>(w), 30, juce::Justification::centred);

    g.setFont(juce::FontOptions(11.0f));
    g.setColour(juce::Colour(0xFFAAAAAA));
    g.drawText(juce::String(backendTest.pathsPerFrame) + " paths/frame, "
               + juce::String(backendTest.framesPerTest) + " frames/sample",
               0, 38, static_cast<int>(w), 16, juce::Justification::centred);

    float chartLeft = 70.0f, chartTop = 70.0f;
    float chartW = w - 100.0f, chartH = h - 140.0f;

    g.setColour(juce::Colour(0xFF16213E));
    g.fillRoundedRectangle(chartLeft - 5, chartTop - 5, chartW + 10, chartH + 10, 6.0f);

    auto& bt = backendTest;
    if (bt.backends.empty() || bt.complexities.empty()) return;

    double maxMs = 1.0;
    for (auto& b : bt.backends)
        for (auto t : b.frameTimes)
            maxMs = std::max(maxMs, t);
    maxMs *= 1.15;

    // Y axis
    g.setFont(juce::FontOptions(9.0f));
    for (int i = 0; i <= 5; i++)
    {
        float y = chartTop + chartH * (1.0f - static_cast<float>(i) / 5.0f);
        double ms = maxMs * static_cast<double>(i) / 5.0;
        g.setColour(juce::Colours::grey);
        g.drawText(juce::String(ms, 1) + "ms", 0, static_cast<int>(y) - 6, 65, 12,
                    juce::Justification::centredRight);
        g.setColour(juce::Colour(0x20FFFFFF));
        g.drawHorizontalLine(static_cast<int>(y), chartLeft, chartLeft + chartW);
    }

    // X axis
    g.setFont(juce::FontOptions(8.0f));
    int numLabels = std::min(10, static_cast<int>(bt.complexities.size()));
    for (int li = 0; li < numLabels; li++)
    {
        size_t ci = static_cast<size_t>(li) * (bt.complexities.size() - 1) / static_cast<size_t>(numLabels - 1);
        float x = chartLeft + (static_cast<float>(ci) / static_cast<float>(bt.complexities.size() - 1)) * chartW;
        int segs = testPathSegCounts[static_cast<size_t>(bt.complexities[ci])];
        juce::String label = segs >= 1000 ? juce::String(segs / 1000) + "K" : juce::String(segs);
        g.setColour(juce::Colours::grey);
        g.drawText(label, static_cast<int>(x) - 20, static_cast<int>(chartTop + chartH + 4), 40, 12,
                    juce::Justification::centred);
    }
    g.setColour(juce::Colour(0xFFAAAAAA));
    g.setFont(juce::FontOptions(10.0f));
    g.drawText("Path segments", static_cast<int>(chartLeft), static_cast<int>(chartTop + chartH + 18),
               static_cast<int>(chartW), 14, juce::Justification::centred);

    // Plot each backend
    for (auto& backend : bt.backends)
    {
        g.setColour(backend.color);
        juce::Path line;

        for (size_t ci = 0; ci < bt.complexities.size(); ci++)
        {
            float x = chartLeft + (static_cast<float>(ci) / static_cast<float>(bt.complexities.size() - 1)) * chartW;
            float y = chartTop + chartH * (1.0f - static_cast<float>(backend.frameTimes[ci] / maxMs));
            if (ci == 0) line.startNewSubPath(x, y);
            else line.lineTo(x, y);

            g.fillEllipse(x - 3, y - 3, 6, 6);
        }
        g.strokePath(line, juce::PathStrokeType(2.0f));
    }

    // Legend
    float legendX = chartLeft + 10, legendY = chartTop + 10;
    g.setFont(juce::FontOptions(11.0f));
    for (auto& backend : bt.backends)
    {
        g.setColour(backend.color);
        g.fillRoundedRectangle(legendX, legendY, 12, 12, 2.0f);
        g.setColour(juce::Colours::white);
        g.drawText(backend.name, static_cast<int>(legendX + 16), static_cast<int>(legendY), 100, 12,
                    juce::Justification::centredLeft);
        legendY += 18;
    }
}
