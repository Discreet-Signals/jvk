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

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "BinaryData.h"

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
    // Row 1: core pipeline benchmarks
    styleButton(btnAll,        "ALL",           juce::Colour(0xFF2A9D8F));
    styleButton(btnFills,      "FILLS",         juce::Colour(0xFFF4A261));
    styleButton(btnText,       "TEXT",          juce::Colour(0xFF6C63FF));
    styleButton(btnPaths,      "PATHS",         juce::Colour(0xFFE07A5F));

    // Row 2: specialized pipeline benchmarks
    styleButton(btnImages,     "IMAGES",        juce::Colour(0xFF45B7D1));
    styleButton(btnGradients,  "GRADIENTS",     juce::Colour(0xFF9B59B6));
    styleButton(btnEffects,    "EFFECTS",       juce::Colour(0xFFE84545));
    styleButton(btnBlur,       "BLUR",          juce::Colour(0xFF2ECC71));

    // Row 3: stress test
    styleButton(btnTextStatic, "TEXT STATIC",   juce::Colour(0xFF5B5EA6));

    btnAll.onClick        = [this] { runBenchmark(Mode::All); };
    btnFills.onClick      = [this] { runBenchmark(Mode::Fills); };
    btnText.onClick       = [this] { runBenchmark(Mode::Text); };
    btnPaths.onClick      = [this] { runBenchmark(Mode::Paths); };
    btnImages.onClick     = [this] { runBenchmark(Mode::Images); };
    btnGradients.onClick  = [this] { runBenchmark(Mode::Gradients); };
    btnEffects.onClick    = [this] { runBenchmark(Mode::Effects); };
    btnBlur.onClick       = [this] { runBenchmark(Mode::Blur); };
    btnTextStatic.onClick = [this] { runBenchmark(Mode::TextStatic); };
    for (auto* btn : { &btnAll, &btnFills, &btnText, &btnPaths,
                       &btnImages, &btnGradients, &btnEffects, &btnBlur,
                       &btnTextStatic })
        addAndMakeVisible(*btn);

    // Renderer toggle button (top-right)
    btnToggleRenderer.onClick = [this] {
        setVulkanEnabled(!isVulkanEnabled());
        updateToggleButtonText();
    };
    btnToggleRenderer.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF333355));
    btnToggleRenderer.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    btnToggleRenderer.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    addAndMakeVisible(btnToggleRenderer);
    updateToggleButtonText();

    auto fullBark = juce::ImageFileFormat::loadFrom(BinaryData::bark_png, BinaryData::bark_pngSize);
    barkImage = fullBark.rescaled(48, 48, juce::Graphics::highResamplingQuality);

    setResizable(true, true);
    setResizeLimits(400, 300, 2560, 1600);
    setSize(800, 600);
}

// =============================================================================
// Path generation — built once, reused every frame
// =============================================================================

void BenchmarkEditor::generatePaths(float w, float h)
{
    if (pathsGenerated) return;
    pathsGenerated = true;

    // 300 waveform paths — sine waves with harmonics
    waveformPaths.resize(300);
    for (int i = 0; i < 300; i++)
    {
        auto& p = waveformPaths[static_cast<size_t>(i)];
        float baseY = h * (static_cast<float>(i) + 0.5f) / 300.0f;
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

    // 300 spectrum analyzer paths — jagged peaks (static shape, animated transforms)
    spectrumPaths.resize(300);
    for (int i = 0; i < 300; i++)
    {
        auto& p = spectrumPaths[static_cast<size_t>(i)];
        float baseY = h * 0.8f;
        int numBins = 64 + (i % 16) * 4;
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

    // 300 gear/mechanical shapes — concentric paths with teeth
    gearPaths.resize(300);
    for (int i = 0; i < 300; i++)
    {
        auto& p = gearPaths[static_cast<size_t>(i)];
        // Local coords — gear lives at origin, positioned via transform at draw time.
        float cx = 0.0f;
        float cy = 0.0f;
        int teeth = 8 + (i % 12) * 2;
        float outerR = 20.0f + static_cast<float>(i % 20) * 3.0f;
        float innerR = outerR * 0.7f;
        float toothH = outerR * 0.25f;

        for (int t = 0; t < teeth; t++)
        {
            float a0 = static_cast<float>(t) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
            float a1 = (static_cast<float>(t) + 0.3f) / static_cast<float>(teeth) * juce::MathConstants<float>::twoPi;
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
        case Mode::All:               modeName = "ALL"; break;
        case Mode::Fills:             modeName = "FILLS"; break;
        case Mode::Text:              modeName = "TEXT"; break;
        case Mode::Paths:             modeName = "PATHS"; break;
        case Mode::Images:            modeName = "IMAGES"; break;
        case Mode::Gradients:         modeName = "GRADIENTS"; break;
        case Mode::Effects:           modeName = "EFFECTS"; break;
        case Mode::Blur:              modeName = "BLUR"; break;
        case Mode::TextStatic:        modeName = "TEXT STATIC"; break;
    }

    generatePaths(static_cast<float>(getWidth()), static_cast<float>(getHeight()));

    showResults = false;
    vulkanFrameTimes.clear();
    juceFrameTimes.clear();
    vulkanRenderedFrames = 0;
    juceRenderedFrames = 0;
    lastPaintTime = 0;

    bool vulkanOnly = isVulkanOnlyMode(mode);

    if (vulkanOnly)
    {
        // Phase 1: Vulkan with effect ON, Phase 2: Vulkan with effect OFF
        phase1Label = "Effect ON";
        phase2Label = "Effect OFF";
        effectEnabled = true;
    }
    else
    {
        // Phase 1: Vulkan, Phase 2: JUCE
        phase1Label = "VULKAN";
        phase2Label = "JUCE";
    }

    setBenchButtonsVisible(false);
    btnToggleRenderer.setVisible(false);
    setVulkanEnabled(true);
    benchmarking = true;
    benchVulkanPhase = true;
    benchFrame = 0;
    benchPhaseStart = juce::Time::getMillisecondCounterHiRes();

    savedPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    // TODO: present mode switching not yet exposed in new architecture
}

void BenchmarkEditor::timerCallback()
{
    if (!benchmarking || benchVulkanPhase) { stopTimer(); return; }
    repaint();
}

// =============================================================================
// Scene: ALL — mixed workload exercising all pipelines
// =============================================================================

void BenchmarkEditor::sceneAll(juce::Graphics& g, float w, float h, float t, float phase)
{
    // 800 filled rectangles (mainPipeline — flat fills)
    for (int i = 0; i < 800; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::sin(phase + fi * 0.3f) * w * 0.45f;
        float y = h * 0.5f + std::cos(phase + fi * 0.5f) * h * 0.45f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 800.0f + t, 1.0f), 0.8f, 0.9f, 0.4f));
        g.fillRect(x - 10.0f, y - 10.0f, 20.0f, 20.0f);
    }

    // 600 rounded rectangles (mainPipeline — SDF shapes)
    for (int i = 0; i < 600; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 23.0f + t * w * 2.0f, w + 100.0f) - 50.0f;
        float y = std::fmod(fi * 17.0f + t * h, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 600.0f + t * 0.5f, 1.0f), 0.7f, 0.85f, 0.5f));
        g.fillRoundedRectangle(x, y, 50.0f + fi * 0.05f, 25.0f + fi * 0.03f, 4.0f + std::fmod(fi * 3.0f, 12.0f));
    }

    // 400 ellipses (mainPipeline — SDF shapes)
    for (int i = 0; i < 400; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::cos(phase * 0.8f + fi * 0.7f) * w * 0.4f;
        float y = h * 0.5f + std::sin(phase * 0.6f + fi * 0.9f) * h * 0.4f;
        float rx = 5.0f + fi * 0.08f, ry = 5.0f + fi * 0.05f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 400.0f + t * 2.0f, 1.0f), 0.9f, 1.0f, 0.35f));
        g.fillEllipse(x - rx, y - ry, rx * 2, ry * 2);
    }

    // 200 bezier paths (stencilPipeline + stencilCoverPipeline)
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

    // 80 gradient rectangles (mainPipeline — gradient descriptor set switching)
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

    // 200 text labels (mainPipeline — MSDF text)
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
// Scene: TEXT — 3000 moving text labels
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

    float sizes[] = { 6, 7, 8, 9, 10, 11, 12, 14, 16, 18 };
    for (int i = 0; i < 3000; i++)
    {
        float fi = static_cast<float>(i);
        float fontSize = sizes[i % 10] + std::fmod(fi * 1.7f, 8.0f);
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
// Scene: TEXT STATIC — 20K static words
// =============================================================================

void BenchmarkEditor::sceneTextStatic(juce::Graphics& g, float w, float h)
{
    juce::String paragraph =
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur";

    auto words = juce::StringArray::fromTokens(paragraph, " ", "");
    float sizes[] = { 7, 8, 9, 10, 11, 12, 13, 14, 16, 18 };
    int wordIdx = 0;

    for (int layer = 0; layer < 10; layer++)
    {
        float fontSize = sizes[layer];
        g.setFont(juce::FontOptions(fontSize));
        float lineH = fontSize * 1.2f;
        float charW = fontSize * 0.55f;
        int wordsPerRow = std::max(1, static_cast<int>(w / (charW * 6.0f)));
        int numRows = static_cast<int>(h / lineH);
        int wordsThisLayer = wordsPerRow * numRows;

        float curX = 0, curY = static_cast<float>(layer) * 1.5f;

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
// Scene: COMPLEX PATHS — stencil pipeline stress test
// =============================================================================

void BenchmarkEditor::sceneComplexPaths(juce::Graphics& g, float w, float h, float t, float phase)
{
    for (size_t i = 0; i < waveformPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        float yOff = std::sin(phase * 0.5f + fi * 0.3f) * 15.0f;
        float scaleX = 0.9f + 0.1f * std::sin(phase * 0.3f + fi * 0.2f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 100.0f + t * 0.5f, 1.0f), 0.7f, 0.9f, 0.4f));
        g.fillPath(waveformPaths[i], juce::AffineTransform::translation(0.0f, yOff).scaled(scaleX, 1.0f));
    }

    for (size_t i = 0; i < spectrumPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        float xOff = std::sin(phase * 0.7f + fi * 0.4f) * 20.0f;
        float yScale = 0.5f + 0.5f * std::sin(phase * 0.4f + fi * 0.15f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 80.0f + t, 1.0f), 0.9f, 1.0f, 0.3f));
        g.fillPath(spectrumPaths[i], juce::AffineTransform::translation(xOff, 0).scaled(1.0f, yScale, w * 0.5f, h * 0.8f));
    }

    for (size_t i = 0; i < gearPaths.size(); i++)
    {
        float fi = static_cast<float>(i);
        int   cols = 20;
        float cx = w * (0.05f + static_cast<float>(i % cols) / static_cast<float>(cols) * 0.95f);
        float cy = h * (0.05f + static_cast<float>(static_cast<int>(i) / cols) / 15.0f * 0.95f);
        float angle = phase * (0.5f + fi * 0.1f) * ((i % 2 == 0) ? 1.0f : -1.0f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 300.0f + t * 0.3f, 1.0f), 0.6f, 0.85f, 0.5f));
        g.fillPath(gearPaths[i], juce::AffineTransform::rotation(angle).translated(cx, cy));
    }

    // 100 dynamic spectrum-analyzer paths — recomputed per frame with bin
    // heights driven by `phase`. Simulates audio-visualiser workloads where
    // geometry genuinely changes every frame (cache always misses).
    constexpr int kDynSpectrumCount = 100;
    constexpr int kDynBins          = 48;
    for (int i = 0; i < kDynSpectrumCount; i++)
    {
        float fi    = static_cast<float>(i);
        int   cols  = 10;
        float cellW = w / static_cast<float>(cols);
        float cellH = h / 10.0f;
        float x0    = cellW * static_cast<float>(i % cols);
        float y0    = cellH * static_cast<float>(i / cols);
        float baseY = y0 + cellH;
        float binW  = cellW / static_cast<float>(kDynBins);

        juce::Path p;
        p.startNewSubPath(x0, baseY);
        for (int b = 0; b < kDynBins; b++)
        {
            float fb   = static_cast<float>(b);
            float amp  = 0.5f + 0.5f * std::sin(phase * 2.0f + fi * 0.2f + fb * 0.35f);
            float decay = std::exp(-fb / static_cast<float>(kDynBins) * 1.5f);
            float binH = amp * decay * cellH * 0.9f;
            float bx   = x0 + fb * binW;
            p.lineTo(bx,              baseY - binH);
            p.lineTo(bx + binW * 0.8f, baseY - binH);
            p.lineTo(bx + binW * 0.9f, baseY);
        }
        p.lineTo(x0 + cellW, baseY);
        p.closeSubPath();

        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 100.0f + t, 1.0f), 0.9f, 1.0f, 0.35f));
        g.fillPath(p);
    }
}

// =============================================================================
// Scene: FILLS — main pipeline fill workload
// =============================================================================

void BenchmarkEditor::sceneFills(juce::Graphics& g, float w, float h, float t, float phase)
{
    for (int i = 0; i < 20000; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 13.0f + t * 200.0f, w + 40.0f) - 20.0f;
        float y = std::fmod(fi * 7.0f + t * 100.0f, h + 40.0f) - 20.0f;
        float s = 3.0f + std::fmod(fi * 3.7f, 15.0f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 20000.0f + t, 1.0f), 0.8f, 0.9f, 0.3f));
        g.fillRect(x, y, s, s);
    }

    for (int i = 0; i < 15000; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 19.0f + t * w, w + 60.0f) - 30.0f;
        float y = std::fmod(fi * 11.0f + t * h * 0.7f, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 15000.0f + t * 0.7f, 1.0f), 0.7f, 0.85f, 0.4f));
        g.fillRoundedRectangle(x, y, 20.0f + fi * 0.001f, 10.0f + fi * 0.0005f, 3.0f + std::fmod(fi, 6.0f));
    }

    for (int i = 0; i < 10000; i++)
    {
        float fi = static_cast<float>(i);
        float x = w * 0.5f + std::cos(phase + fi * 0.3f) * w * 0.48f;
        float y = h * 0.5f + std::sin(phase * 0.7f + fi * 0.5f) * h * 0.48f;
        float r = 2.0f + fi * 0.002f;
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 10000.0f + t * 1.5f, 1.0f), 0.9f, 1.0f, 0.2f));
        g.fillEllipse(x - r, y - r, r * 2, r * 2);
    }

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
// Scene: IMAGES — 20K bouncing sprites
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
        s.x += s.vx;  s.y += s.vy;
        if (s.x < 0)              { s.x = 0;              s.vx = std::abs(s.vx); }
        if (s.x + spriteSize > w) { s.x = w - spriteSize;  s.vx = -std::abs(s.vx); }
        if (s.y < 0)              { s.y = 0;              s.vy = std::abs(s.vy); }
        if (s.y + spriteSize > h) { s.y = h - spriteSize;  s.vy = -std::abs(s.vy); }
        g.drawImage(barkImage, juce::Rectangle<float>(s.x, s.y, spriteSize, spriteSize));
    }
}

// =============================================================================
// Scene: GRADIENTS — gradient LUT descriptor set switching
// =============================================================================

void BenchmarkEditor::sceneGradients(juce::Graphics& g, float w, float h, float t, float phase)
{
    // 3000 linear gradient fills — each unique gradient causes a descriptor set switch
    for (int i = 0; i < 3000; i++)
    {
        float fi = static_cast<float>(i);
        float gx = std::fmod(fi * 37.0f + t * 80.0f, w);
        float gy = std::fmod(fi * 29.0f + t * 50.0f, h);
        float gw = 30.0f + std::fmod(fi * 7.3f, 50.0f);
        float gh = 15.0f + std::fmod(fi * 4.1f, 25.0f);

        juce::ColourGradient grad(
            juce::Colour::fromHSV(std::fmod(t + fi / 3000.0f, 1.0f), 0.9f, 1.0f, 0.7f), gx, gy,
            juce::Colour::fromHSV(std::fmod(t + fi / 3000.0f + 0.5f, 1.0f), 0.9f, 0.5f, 0.7f),
            gx + gw, gy + gh, false);
        g.setGradientFill(grad);
        g.fillRect(gx, gy, gw, gh);
    }

    // 1500 radial gradient fills
    for (int i = 0; i < 1500; i++)
    {
        float fi = static_cast<float>(i);
        float cx = std::fmod(fi * 43.0f + std::sin(phase + fi * 0.1f) * 40.0f, w);
        float cy = std::fmod(fi * 31.0f + std::cos(phase * 0.7f + fi * 0.07f) * 30.0f, h);
        float radius = 10.0f + std::fmod(fi * 5.7f, 30.0f);

        juce::ColourGradient grad(
            juce::Colour::fromHSV(std::fmod(t * 0.5f + fi / 1500.0f, 1.0f), 1.0f, 1.0f, 0.8f), cx, cy,
            juce::Colour::fromHSV(std::fmod(t * 0.5f + fi / 1500.0f + 0.3f, 1.0f), 0.8f, 0.3f, 0.0f),
            cx + radius, cy, true);
        g.setGradientFill(grad);
        g.fillEllipse(cx - radius, cy - radius, radius * 2, radius * 2);
    }
}

// =============================================================================
// Scene: EFFECTS — multiplyPipeline hardware blend modes (Vulkan-only)
// =============================================================================

void BenchmarkEditor::sceneEffects(juce::Graphics& g, float w, float h, float t, float phase)
{
    // Base scene: 5000 bright colored rectangles
    for (int i = 0; i < 5000; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 17.0f + t * 150.0f, w + 40.0f) - 20.0f;
        float y = std::fmod(fi * 11.0f + t * 80.0f, h + 40.0f) - 20.0f;
        float s = 8.0f + std::fmod(fi * 3.7f, 20.0f);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 5000.0f + t, 1.0f), 0.9f, 1.0f, 0.6f));
        g.fillRoundedRectangle(x, y, s, s * 0.6f, 3.0f);
    }

    // Apply multiply-blend effects if enabled (Vulkan only)
    if (effectEnabled)
    {
        auto vk = jvk::Graphics::create(g);
        if (vk)
        {
            // Darken the left third
            vk->darken(0.4f, juce::Rectangle<float>(0, 0, w / 3.0f, h));

            // Tint the middle third warm
            vk->warmth(0.8f, juce::Rectangle<float>(w / 3.0f, 0, w / 3.0f, h));

            // Tint the right third cool
            vk->tint(juce::Colour(0xFF0000FF), juce::Rectangle<float>(w * 2.0f / 3.0f, 0, w / 3.0f, h));

            // Darken bottom half
            vk->darken(0.2f, juce::Rectangle<float>(0, h * 0.5f, w, h * 0.5f));

            // Multiple overlapping effects to stress the pipeline
            for (int i = 0; i < 20; i++)
            {
                float fi = static_cast<float>(i);
                float rx = std::fmod(fi * 97.0f + t * 200.0f, w);
                float ry = std::fmod(fi * 73.0f + t * 100.0f, h);
                float rw = 60.0f + fi * 8.0f;
                float rh = 40.0f + fi * 5.0f;
                vk->darken(0.15f, juce::Rectangle<float>(rx, ry, rw, rh));
            }
        }
    }
}

// =============================================================================
// Scene: BLUR — post-process blur pipeline (Vulkan-only)
// =============================================================================

void BenchmarkEditor::sceneBlur(juce::Graphics& g, float w, float h, float t, float phase)
{
    // Base scene: 2000 bouncing sprites (texture fill to create complex content to blur)
    if (barkImage.isValid())
    {
        initSprites(w, h);
        constexpr float spriteSize = 48.0f;
        int count = std::min(2000, static_cast<int>(sprites.size()));
        for (int i = 0; i < count; i++)
        {
            auto& s = sprites[static_cast<size_t>(i)];
            s.x += s.vx;  s.y += s.vy;
            if (s.x < 0)              { s.x = 0;              s.vx = std::abs(s.vx); }
            if (s.x + spriteSize > w) { s.x = w - spriteSize;  s.vx = -std::abs(s.vx); }
            if (s.y < 0)              { s.y = 0;              s.vy = std::abs(s.vy); }
            if (s.y + spriteSize > h) { s.y = h - spriteSize;  s.vy = -std::abs(s.vy); }
            g.drawImage(barkImage, juce::Rectangle<float>(s.x, s.y, spriteSize, spriteSize));
        }
    }

    // Also add some colored geometry
    for (int i = 0; i < 500; i++)
    {
        float fi = static_cast<float>(i);
        float x = std::fmod(fi * 23.0f + t * 200.0f, w);
        float y = std::fmod(fi * 17.0f + t * 100.0f, h);
        g.setColour(juce::Colour::fromHSV(std::fmod(fi / 500.0f + t, 1.0f), 0.9f, 1.0f, 0.5f));
        g.fillRoundedRectangle(x, y, 30.0f, 20.0f, 4.0f);
    }

    // Apply blur if enabled — radius pulses 0..128..0 on a cosine, 4 cycles per scene loop.
    constexpr float maxRadius = 128.0f;
    const float blurRadius = (1.0f - std::cos(t * juce::MathConstants<float>::twoPi * 4.0f)) * 0.5f * maxRadius;
    // Phase-inverted so the shape blurs are brightest when the global blur is dimmest.
    const float shapeBlurRadius = maxRadius - blurRadius;

    if (effectEnabled)
    {
        auto vk = jvk::Graphics::create(g);
        if (vk)
        {
            // vk->blur(blurRadius); // temporarily disabled while testing shape blurs

            // Centered circle blur — disc of radius 100 logical px, 24 px falloff.
            float cx = w * 0.5f;
            float cy = h * 0.5f;
            float r  = 100.0f;
            vk->fillBlurredEllipse({ cx - r, cy - r, r * 2.0f, r * 2.0f },
                                   shapeBlurRadius, 24.0f);

            // Vertical line hugging the right edge, 48 px thick, 24 px falloff.
            juce::Line<float> rightEdge { w - 24.0f, 0.0f, w - 24.0f, h };
            vk->drawBlurredLine(rightEdge, 48.0f, shapeBlurRadius, 24.0f);
        }
    }

    // Draw text on top (to show that post-blur drawing works)
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(20.0f));
    juce::String label = effectEnabled
        ? juce::String::formatted("Blur ON  -  radius %.1f", blurRadius)
        : juce::String("Blur OFF");
    g.drawText(label,
               0, static_cast<int>(h * 0.5f) - 15, static_cast<int>(w), 30,
               juce::Justification::centred);
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
        case Mode::All:               sceneAll(g, w, h, t, phase); break;
        case Mode::Fills:             sceneFills(g, w, h, t, phase); break;
        case Mode::Text:              sceneText(g, w, h, t, phase); break;
        case Mode::Paths:             sceneComplexPaths(g, w, h, t, phase); break;
        case Mode::Images:            sceneImages(g, w, h); break;
        case Mode::Gradients:         sceneGradients(g, w, h, t, phase); break;
        case Mode::Effects:           sceneEffects(g, w, h, t, phase); break;
        case Mode::Blur:              sceneBlur(g, w, h, t, phase); break;
        case Mode::TextStatic:        sceneTextStatic(g, w, h); break;
    }
}

// =============================================================================
// Results display
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
    g.drawText(phase1Label, static_cast<int>(col1), static_cast<int>(row), 150, 20, juce::Justification::centredLeft);
    g.setColour(juce::Colour(0xFFE07A5F));
    g.drawText(phase2Label, static_cast<int>(col2), static_cast<int>(row), 150, 20, juce::Justification::centredLeft);

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

    bool vulkanOnly = isVulkanOnlyMode(activeMode);
    if (vulkanOnly)
    {
        // For effect-on/off comparison, show overhead percentage
        double overhead = juceFPS > 0 ? (1.0 - vulkanFPS / juceFPS) * 100.0 : 0;
        g.setColour(overhead < 20.0 ? juce::Colour(0xFF4ECCA3) : juce::Colour(0xFFE84545));
        g.drawText("Effect overhead: " + juce::String(std::abs(overhead), 1) + "% "
                   + (overhead > 0 ? "slower" : "faster"),
                   0, static_cast<int>(row), static_cast<int>(w), 22, juce::Justification::centred);
    }
    else
    {
        g.setColour(speedup > 1.0 ? juce::Colour(0xFF4ECCA3) : juce::Colour(0xFFE84545));
        g.drawText("Vulkan is " + juce::String(speedup, 2) + "x " + (speedup > 1.0 ? "faster" : "slower"),
                   0, static_cast<int>(row), static_cast<int>(w), 22, juce::Justification::centred);
    }

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
    g.drawText(phase1Label, static_cast<int>(chartLeft + 16), static_cast<int>(legendY), 80, 12, juce::Justification::centredLeft);
    g.setColour(juce::Colour(0xFFE07A5F)); g.fillRect(chartLeft + 100.0f, legendY, 12.0f, 12.0f);
    g.setColour(juce::Colours::white);
    g.drawText(phase2Label, static_cast<int>(chartLeft + 116), static_cast<int>(legendY), 80, 12, juce::Justification::centredLeft);
}

// =============================================================================
// Paint — main rendering entry point
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
                // Transition to phase 2
                benchVulkanPhase = false;
                benchFrame = 0;
                lastPaintTime = 0;
                benchPhaseStart = juce::Time::getMillisecondCounterHiRes();

                if (isVulkanOnlyMode(activeMode))
                {
                    // Phase 2: same Vulkan rendering but with effect OFF
                    effectEnabled = false;
                }
                else
                {
                    // Phase 2: JUCE CPU rendering
                    setVulkanEnabled(false);
                    startTimerHz(1000);
                }
            }
            else
            {
                stopTimer();
                benchmarking = false;
                effectEnabled = true;  // reset for next run
                // TODO: restore present mode

                if (!isVulkanOnlyMode(activeMode))
                    setVulkanEnabled(true);

                vulkanFPS = vulkanRenderedFrames / (benchDurationMs / 1000.0);
                juceFPS = juceRenderedFrames / (benchDurationMs / 1000.0);
                showResults = true;
                setBenchButtonsVisible(true);
                btnToggleRenderer.setVisible(true);
            }
        }
        return;
    }

    // Idle screen
    g.fillAll(juce::Colour(0xFF0E0E1A));
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(28.0f));
    g.drawText("JVK Benchmark", 0, getHeight() / 2 - 160, getWidth(), 35, juce::Justification::centred);
    g.setFont(juce::FontOptions(14.0f));
    g.setColour(juce::Colour(0xFFAAAAAA));
    g.drawText("Select a benchmark to compare Vulkan vs JUCE rendering", 0, getHeight() / 2 - 120, getWidth(), 20, juce::Justification::centred);

    g.setFont(juce::FontOptions(11.0f));
    g.setColour(juce::Colour(0xFF666666));
    g.drawText("Row 1: Core pipelines (Vulkan vs JUCE)  |  Row 2: Specialized (EFFECTS/BLUR compare ON vs OFF)",
               0, getHeight() / 2 - 95, getWidth(), 16, juce::Justification::centred);
}

// =============================================================================
// Layout
// =============================================================================

void BenchmarkEditor::updateToggleButtonText()
{
    btnToggleRenderer.setButtonText(isVulkanEnabled() ? "Vulkan" : "JUCE");
    auto col = isVulkanEnabled() ? juce::Colour(0xFF2A9D8F) : juce::Colour(0xFFE07A5F);
    btnToggleRenderer.setColour(juce::TextButton::buttonColourId, col);
}

void BenchmarkEditor::setBenchButtonsVisible(bool visible)
{
    for (auto* btn : { &btnAll, &btnFills, &btnText, &btnPaths,
                       &btnImages, &btnGradients, &btnEffects, &btnBlur,
                       &btnTextStatic })
        btn->setVisible(visible);
}

void BenchmarkEditor::resized()
{
    pathsGenerated = false;
    spritesInitialized = false;

    // Renderer toggle button — top right
    btnToggleRenderer.setBounds(getWidth() - 90, 5, 80, 28);

    // Benchmark buttons centered in three rows
    int btnW = 120, btnH = 40, gap = 10;
    int totalW = 4 * btnW + 3 * gap;
    int startX = (getWidth() - totalW) / 2;
    int row1Y = getHeight() / 2 - btnH - gap / 2;
    int row2Y = row1Y + btnH + gap;
    int row3Y = row2Y + btnH + gap;

    // Row 1: core pipeline benchmarks
    btnAll.setBounds(startX, row1Y, btnW, btnH);
    btnFills.setBounds(startX + btnW + gap, row1Y, btnW, btnH);
    btnText.setBounds(startX + 2 * (btnW + gap), row1Y, btnW, btnH);
    btnPaths.setBounds(startX + 3 * (btnW + gap), row1Y, btnW, btnH);

    // Row 2: specialized pipeline benchmarks
    btnImages.setBounds(startX, row2Y, btnW, btnH);
    btnGradients.setBounds(startX + btnW + gap, row2Y, btnW, btnH);
    btnEffects.setBounds(startX + 2 * (btnW + gap), row2Y, btnW, btnH);
    btnBlur.setBounds(startX + 3 * (btnW + gap), row2Y, btnW, btnH);

    // Row 3: stress test (smaller, centered)
    int smallW = 110, smallH = 30;
    int row3StartX = (getWidth() - smallW) / 2;
    btnTextStatic.setBounds(row3StartX, row3Y, smallW, smallH);
}
