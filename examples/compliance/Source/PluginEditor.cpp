/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: PluginEditor.cpp  (JVKCompliance — golden-image compliance harness)
 ------------------------------------------------------------------------------
*/

#include "PluginEditor.h"

namespace
{
constexpr int kHeaderH = 44;
constexpr int kFooterH = 40;
constexpr int kMargin  = 10;
constexpr int kPaneW   = 460;
constexpr int kPaneH   = 460;
} // namespace

// =============================================================================
// Construction / layout
// =============================================================================

ComplianceEditor::ComplianceEditor(ComplianceProcessor& p)
    : jvk::AudioProcessorEditor(p)
{
    setWantsKeyboardFocus(true);
    setSize(kMargin * 3 + kPaneW * 2, kHeaderH + kPaneH + kFooterH + kMargin * 2);
}

juce::Rectangle<int> ComplianceEditor::headerArea() const
{
    return { 0, 0, getWidth(), kHeaderH };
}

juce::Rectangle<int> ComplianceEditor::footerArea() const
{
    return { 0, getHeight() - kFooterH, getWidth(), kFooterH };
}

juce::Rectangle<int> ComplianceEditor::leftPane() const
{
    return { kMargin, kHeaderH + kMargin, kPaneW, kPaneH };
}

juce::Rectangle<int> ComplianceEditor::rightPane() const
{
    return { kMargin * 2 + kPaneW, kHeaderH + kMargin, kPaneW, kPaneH };
}

void ComplianceEditor::resized()
{
    rebuildReference();
}

void ComplianceEditor::setScene(int index)
{
    const int n = static_cast<int>(scenes().size());
    sceneIndex_ = ((index % n) + n) % n;
    rebuildReference();
    repaint();
}

bool ComplianceEditor::keyPressed(const juce::KeyPress& key)
{
    if (key == juce::KeyPress::rightKey) { setScene(sceneIndex_ + 1); return true; }
    if (key == juce::KeyPress::leftKey)  { setScene(sceneIndex_ - 1); return true; }
    if (key.getTextCharacter() == 'v' || key.getTextCharacter() == 'V')
    {
        vulkanOn_ = !vulkanOn_;
        setVulkanEnabled(vulkanOn_);
        repaint();
        return true;
    }
    return false;
}

// The reference is what the JUCE SOFTWARE renderer produces for the scene —
// the golden image the live pane is judged against.
void ComplianceEditor::rebuildReference()
{
    auto pane = leftPane();
    if (pane.isEmpty()) return;
    reference_ = juce::Image(juce::Image::ARGB, pane.getWidth(), pane.getHeight(), true);
    juce::Graphics ig(reference_);
    ig.fillAll(juce::Colour(0xFF15151D));
    scenes()[static_cast<size_t>(sceneIndex_)].painter(
        ig, juce::Rectangle<float>(0, 0, (float) pane.getWidth(), (float) pane.getHeight()));
}

void ComplianceEditor::paint(juce::Graphics& g)
{
    auto& scene = scenes()[static_cast<size_t>(sceneIndex_)];

    g.fillAll(juce::Colour(0xFF0B0B10));

    // Header
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
    g.drawText(juce::String(sceneIndex_ + 1) + "/" + juce::String((int) scenes().size())
                   + "  " + scene.name,
               headerArea().reduced(kMargin, 0), juce::Justification::centredLeft);
    g.setFont(juce::FontOptions(13.0f));
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.drawText(juce::String("renderer: ") + (vulkanOn_ ? "jvk (Vulkan)" : "JUCE software")
                   + "   [<-/->] scene   [V] toggle renderer",
               headerArea().reduced(kMargin, 0), juce::Justification::centredRight);

    // Pane labels
    g.setColour(juce::Colours::white.withAlpha(0.5f));
    g.setFont(juce::FontOptions(12.0f));
    g.drawText("REFERENCE (software render)", leftPane().withY(kHeaderH - 6).withHeight(14),
               juce::Justification::centredLeft);
    g.drawText("LIVE (active renderer)", rightPane().withY(kHeaderH - 6).withHeight(14),
               juce::Justification::centredLeft);

    // Left: the golden image.
    if (reference_.isValid())
        g.drawImageAt(reference_, leftPane().getX(), leftPane().getY());

    // Right: the same painter, live through the active renderer, clipped and
    // translated so scene code sees an origin-based area in both panes.
    {
        juce::Graphics::ScopedSaveState save(g);
        auto pane = rightPane();
        g.reduceClipRegion(pane);
        g.setOrigin(pane.getPosition());
        g.setColour(juce::Colour(0xFF15151D));
        g.fillRect(0, 0, pane.getWidth(), pane.getHeight());
        scene.painter(g, juce::Rectangle<float>(0, 0, (float) pane.getWidth(),
                                                (float) pane.getHeight()));
    }

    // Pane borders + footer caveat
    g.setColour(juce::Colours::white.withAlpha(0.25f));
    g.drawRect(leftPane());
    g.drawRect(rightPane());
    g.setColour(juce::Colour(0xFFE0B040));
    g.setFont(juce::FontOptions(13.0f));
    g.drawText(scene.caveat != nullptr ? juce::String("expected divergence: ") + scene.caveat
                                       : juce::String("panes should match"),
               footerArea().reduced(kMargin, 0), juce::Justification::centredLeft);
}

// =============================================================================
// Shared bits
// =============================================================================

juce::Image ComplianceEditor::makeTestImage(int w, int h)
{
    juce::Image img(juce::Image::ARGB, w, h, true);
    juce::Graphics g(img);
    for (int y = 0; y < h; y += 16)
        for (int x = 0; x < w; x += 16)
        {
            const bool a = ((x / 16) + (y / 16)) % 2 == 0;
            g.setColour(a ? juce::Colour(0xFFB05070) : juce::Colour(0xFF5070B0));
            g.fillRect(x, y, 16, 16);
        }
    g.setColour(juce::Colours::yellow);
    g.drawEllipse(4.0f, 4.0f, (float) w - 8.0f, (float) h - 8.0f, 3.0f);
    return img;
}

static juce::Path makeStar(juce::Point<float> c, float rOuter, float rInner, int points)
{
    juce::Path p;
    p.addStar(c, points, rInner, rOuter);
    return p;
}

// =============================================================================
// Scenes
// =============================================================================

// Deep clip stack: path clip -> nested path clip -> nested rect clip. Fills at
// every level; the saturated fill at the deepest level must appear ONLY in the
// triple intersection, and the big magenta rect at the end must be fully
// clipped away by the restored outer clip.
void ComplianceEditor::sceneDeepClips(juce::Graphics& g, juce::Rectangle<float> area)
{
    auto c = area.getCentre();

    juce::Graphics::ScopedSaveState s1(g);
    g.reduceClipRegion(makeStar(c, area.getWidth() * 0.45f, area.getWidth() * 0.28f, 9));
    g.setColour(juce::Colour(0xFF2A9D8F));
    g.fillRect(area);   // level 1: star-shaped teal

    {
        juce::Graphics::ScopedSaveState s2(g);
        juce::Path ring;
        ring.addEllipse(area.reduced(area.getWidth() * 0.18f));
        g.reduceClipRegion(ring);
        g.setColour(juce::Colour(0xFFF4A261));
        g.fillRect(area);   // level 2: star ∩ ellipse, orange

        {
            juce::Graphics::ScopedSaveState s3(g);
            g.reduceClipRegion(juce::Rectangle<int>((int) c.x - 200, (int) c.y - 60, 400, 120));
            g.setGradientFill(juce::ColourGradient(juce::Colours::red, area.getX(), c.y,
                                                   juce::Colours::blue, area.getRight(), c.y,
                                                   false));
            g.fillRect(area);   // level 3: star ∩ ellipse ∩ rect, gradient
        }

        g.setColour(juce::Colours::white.withAlpha(0.8f));
        g.drawEllipse(area.reduced(area.getWidth() * 0.18f), 3.0f); // back at level 2
    }

    // Back at level 1: this must still be star-clipped.
    g.setColour(juce::Colour(0x60FF00FF));
    g.fillRect(area.removeFromBottom(area.getHeight() * 0.25f));
}

// The live host pattern: a jvk blend effect (drawNoise) INSIDE a path clip.
// Reference has no jvk extras, so the caveat states the noise only shows on
// the live pane — what must MATCH is the clip shape containing it.
void ComplianceEditor::sceneBlendInClip(juce::Graphics& g, juce::Rectangle<float> area)
{
    auto disc = area.reduced(area.getWidth() * 0.2f);

    juce::Graphics::ScopedSaveState save(g);
    juce::Path p;
    p.addEllipse(disc);
    g.reduceClipRegion(p);

    g.setGradientFill(juce::ColourGradient(juce::Colour(0xFF264653), disc.getX(), disc.getY(),
                                           juce::Colour(0xFF2A9D8F), disc.getRight(), disc.getBottom(),
                                           false));
    g.fillRect(area);

    if (auto vk = jvk::Graphics::create(g))
        vk->drawNoise(0.25f, true);

    g.setColour(juce::Colours::white);
    g.drawEllipse(disc, 4.0f);
}

void ComplianceEditor::scenePaths(juce::Graphics& g, juce::Rectangle<float> area)
{
    // Complex filled path (non-zero winding star)
    g.setColour(juce::Colour(0xFFE07A5F));
    g.fillPath(makeStar({ area.getWidth() * 0.28f, area.getHeight() * 0.32f },
                        area.getWidth() * 0.22f, area.getWidth() * 0.09f, 11));

    // Gradient-filled wave path
    juce::Path wave;
    wave.startNewSubPath(area.getX(), area.getCentreY());
    for (float x = 0; x <= area.getWidth(); x += 4.0f)
        wave.lineTo(area.getX() + x,
                    area.getCentreY() + std::sin(x * 0.05f) * 40.0f
                        + std::sin(x * 0.013f) * 60.0f);
    wave.lineTo(area.getBottomRight());
    wave.lineTo(area.getBottomLeft());
    wave.closeSubPath();
    g.setGradientFill(juce::ColourGradient(juce::Colour(0xC06C63FF), area.getCentreX(), area.getCentreY(),
                                           juce::Colour(0x306C63FF), area.getCentreX(), area.getBottom(),
                                           false));
    g.fillPath(wave);

    // Stroked spiral
    juce::Path spiral;
    auto c = juce::Point<float>(area.getWidth() * 0.72f, area.getHeight() * 0.32f);
    spiral.startNewSubPath(c);
    for (float t = 0; t < 6.0f * juce::MathConstants<float>::pi; t += 0.1f)
        spiral.lineTo(c.x + std::cos(t) * t * 4.0f, c.y + std::sin(t) * t * 4.0f);
    g.setColour(juce::Colours::white);
    g.strokePath(spiral, juce::PathStrokeType(2.5f));
}

void ComplianceEditor::sceneText(juce::Graphics& g, juce::Rectangle<float> area)
{
    g.setColour(juce::Colours::white);
    float y = 16.0f;
    for (float size : { 11.0f, 16.0f, 24.0f, 40.0f })
    {
        g.setFont(juce::FontOptions(size));
        g.drawText("The quick brown fox jumps over 0123456789 !@#$%",
                   juce::Rectangle<float>(10.0f, y, area.getWidth() - 20.0f, size * 1.3f),
                   juce::Justification::centredLeft);
        y += size * 1.5f;
    }

    // Rotated + sheared runs — guardrail scene for the transformed-text fix.
    g.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    {
        juce::Graphics::ScopedSaveState save(g);
        g.addTransform(juce::AffineTransform::rotation(-0.5f,
                       area.getWidth() * 0.3f, area.getHeight() * 0.7f));
        g.setColour(juce::Colour(0xFFF4A261));
        g.drawSingleLineText("ROTATED TEXT", (int) (area.getWidth() * 0.12f),
                             (int) (area.getHeight() * 0.72f));
    }
    {
        juce::Graphics::ScopedSaveState save(g);
        g.addTransform(juce::AffineTransform::shear(0.35f, 0.0f));
        g.setColour(juce::Colour(0xFF2A9D8F));
        g.drawSingleLineText("SHEARED TEXT", (int) (area.getWidth() * 0.35f),
                             (int) (area.getHeight() * 0.9f));
    }
}

void ComplianceEditor::sceneGradients(juce::Graphics& g, juce::Rectangle<float> area)
{
    auto half = area.getWidth() / 2.0f;

    juce::ColourGradient linear(juce::Colour(0xFFE84545), 20.0f, 20.0f,
                                juce::Colour(0xFF2ECC71), half - 20.0f, area.getHeight() - 20.0f,
                                false);
    linear.addColour(0.5, juce::Colour(0xFFF4A261));
    g.setGradientFill(linear);
    g.fillRoundedRectangle(10.0f, 10.0f, half - 20.0f, area.getHeight() - 20.0f, 24.0f);

    juce::ColourGradient radial(juce::Colours::white, half + half / 2.0f, area.getCentreY(),
                                juce::Colour(0xFF264653), half + 20.0f, 20.0f, true);
    radial.addColour(0.35, juce::Colour(0xFF6C63FF));
    g.setGradientFill(radial);
    g.fillEllipse(half + 10.0f, 10.0f, half - 20.0f, area.getHeight() - 20.0f);
}

void ComplianceEditor::sceneRectsAndLists(juce::Graphics& g, juce::Rectangle<float> area)
{
    // RectangleList fill — exercises fillRectList batching.
    juce::RectangleList<float> list;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            if ((i + j) % 2 == 0)
                list.addWithoutMerging({ 10.0f + i * 24.0f, 10.0f + j * 24.0f, 20.0f, 20.0f });
    g.setColour(juce::Colour(0xFF45B7D1));
    g.fillRectList(list);

    // Rounded rects / ellipses / strokes / lines at fractional coords.
    g.setColour(juce::Colour(0xFFE07A5F));
    g.fillRoundedRectangle(area.getWidth() * 0.68f, 12.5f, 140.5f, 90.25f, 18.0f);
    g.setColour(juce::Colours::white);
    g.drawRoundedRectangle(area.getWidth() * 0.68f, 120.5f, 140.5f, 90.25f, 18.0f, 3.0f);
    g.setColour(juce::Colour(0xFF9B59B6));
    g.fillEllipse(area.getWidth() * 0.68f, 228.0f, 140.0f, 90.0f);
    for (int i = 0; i < 8; i++)
    {
        g.setColour(juce::Colours::white.withAlpha(0.3f + 0.1f * i));
        g.drawLine(20.0f, area.getHeight() - 130.0f + i * 14.0f,
                   area.getWidth() * 0.6f, area.getHeight() - 90.0f + i * 10.0f,
                   1.0f + i * 0.5f);
    }
}

void ComplianceEditor::sceneImages(juce::Graphics& g, juce::Rectangle<float> area)
{
    static const juce::Image test = makeTestImage(128, 128);

    // Plain draw
    g.drawImageAt(test, 16, 16);

    // Transformed draw (scale + rotate)
    g.drawImageTransformed(test,
        juce::AffineTransform::rotation(0.4f, 64.0f, 64.0f)
            .scaled(1.4f)
            .translated(area.getWidth() * 0.42f, 24.0f));

    // Alpha-modulated draw — juce semantics: current colour's ALPHA modulates.
    g.setColour(juce::Colours::black.withAlpha(0.35f));
    g.drawImageAt(test, 16, (int) (area.getHeight() * 0.55f));
    g.setColour(juce::Colours::black);

    // Alpha-brush draw (fillAlphaChannelWithCurrentBrush): juce turns this
    // into clipToImageAlpha + fillAll — must render the image's alpha
    // channel in hot pink, NOT a solid rectangle and NOT nothing.
    g.setColour(juce::Colours::hotpink);
    g.drawImageAt(test, (int) (area.getWidth() * 0.45f), 16, true);
    g.setColour(juce::Colours::black);

    // Tiled image fill.
    g.setTiledImageFill(test, 0, 0, 1.0f);
    g.fillRoundedRectangle(area.getWidth() * 0.45f, area.getHeight() * 0.55f,
                           area.getWidth() * 0.5f, area.getHeight() * 0.4f, 16.0f);
}

// The setOpacity / fill-alpha contract (juce_RenderingHelpers.h):
//   - setOpacity REPLACES the current fill colour's alpha (FillType::setOpacity)
//     — it is NOT a separate sticky multiplier;
//   - setColour replaces the whole fill, discarding any prior setOpacity;
//   - image draws are modulated by fillType.colour's alpha unconditionally;
//   - a transparency layer's alpha must not leak past endTransparencyLayer.
// Row 1 is the exact melatonin::DropShadow idiom that regressed pedal
// shadows: a whisper-alpha glass fill, then setOpacity(1.0) + drawImage —
// the image must render at FULL strength.
void ComplianceEditor::sceneOpacityContract(juce::Graphics& g, juce::Rectangle<float> area)
{
    static const juce::Image test = makeTestImage(96, 96);
    const float rowH = area.getHeight() * 0.25f;

    // Row 1 — melatonin idiom: low-alpha colour, setOpacity(1), draw image.
    g.setColour(juce::Colours::white.withAlpha(0.06f));
    g.fillRoundedRectangle(16.0f, 8.0f, 140.0f, rowH - 16.0f, 8.0f);
    g.setOpacity(1.0f);
    g.drawImageAt(test, 180, 8);                        // must be FULL alpha

    // Row 2 — setOpacity then setColour: the colour wins outright.
    g.setOpacity(0.2f);
    g.setColour(juce::Colours::orangered);
    g.fillRect(16.0f, rowH + 8.0f, 140.0f, rowH - 16.0f);   // must be OPAQUE
    // ...and setOpacity alone rewrites the fill's alpha in place.
    g.setOpacity(0.35f);
    g.fillRect(180.0f, rowH + 8.0f, 140.0f, rowH - 16.0f);  // 35% orangered

    // Row 3 — setOpacity applies to image draws directly.
    g.setColour(juce::Colours::black);
    g.setOpacity(0.4f);
    g.drawImageAt(test, 16, (int) (rowH * 2.0f + 8.0f));    // 40% image
    g.setOpacity(1.0f);

    // Row 4 — layer alpha must unwind at endTransparencyLayer.
    g.beginTransparencyLayer(0.3f);
    g.setColour(juce::Colours::cornflowerblue);
    g.fillRect(16.0f, rowH * 3.0f + 8.0f, 140.0f, rowH - 16.0f);  // 30% via layer
    g.endTransparencyLayer();
    g.setColour(juce::Colours::seagreen);
    g.fillRect(180.0f, rowH * 3.0f + 8.0f, 140.0f, rowH - 16.0f); // must be OPAQUE
}

// Region-scissored effects, per guardrail G-B: one region fully interior and
// one abutting the pane edge (exercises the ROI clamp), plus a no-region
// saturate (defaults to clip bounds).
void ComplianceEditor::sceneEffectsRegion(juce::Graphics& g, juce::Rectangle<float> area)
{
    // Busy background so blur/saturation are visible.
    for (int i = 0; i < 14; i++)
    {
        g.setColour(juce::Colour::fromHSV(i / 14.0f, 0.8f, 0.9f, 1.0f));
        g.fillEllipse(20.0f + (i % 5) * 88.0f, 20.0f + (i / 5) * 120.0f, 110.0f, 110.0f);
    }

    if (auto vk = jvk::Graphics::create(g))
    {
        // Interior region blur.
        vk->blur(12.0f, { area.getWidth() * 0.3f, area.getHeight() * 0.3f,
                          area.getWidth() * 0.4f, area.getHeight() * 0.3f });
        // Edge-abutting region blur (clamp path).
        vk->blur(8.0f, { 0.0f, 0.0f, area.getWidth() * 0.25f, area.getHeight() });
        // Desaturate the bottom strip.
        vk->saturate(0.15f, { 0.0f, area.getHeight() * 0.78f,
                              area.getWidth(), area.getHeight() * 0.22f });
    }

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(13.0f));
    g.drawText("blur: interior + left-edge region; saturate: bottom strip",
               area.removeFromBottom(20.0f), juce::Justification::centred);
}

void ComplianceEditor::sceneTransparencyLayer(juce::Graphics& g, juce::Rectangle<float> area)
{
    // Opaque overlapping shapes inside a half-alpha layer: TRUE layer
    // semantics composite the GROUP at 50% (overlap looks the same as the
    // single shapes); the per-primitive approximation double-darkens the
    // overlap. Reference shows the correct result.
    g.setColour(juce::Colour(0xFF264653));
    g.fillRect(area);

    g.beginTransparencyLayer(0.5f);
    g.setColour(juce::Colours::red);
    g.fillEllipse(area.getWidth() * 0.2f, area.getHeight() * 0.25f,
                  area.getWidth() * 0.4f, area.getHeight() * 0.4f);
    g.setColour(juce::Colours::red);
    g.fillEllipse(area.getWidth() * 0.4f, area.getHeight() * 0.35f,
                  area.getWidth() * 0.4f, area.getHeight() * 0.4f);
    g.endTransparencyLayer();

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(13.0f));
    g.drawText("two opaque red ellipses inside one 50% layer — overlap must NOT darken",
               area.removeFromBottom(24.0f), juce::Justification::centred);
}

void ComplianceEditor::sceneExcludeClip(juce::Graphics& g, juce::Rectangle<float> area)
{
    juce::Graphics::ScopedSaveState save(g);
    // Donut: full-pane fill with the centre excluded.
    g.excludeClipRegion(juce::Rectangle<int>((int) (area.getWidth() * 0.3f),
                                             (int) (area.getHeight() * 0.3f),
                                             (int) (area.getWidth() * 0.4f),
                                             (int) (area.getHeight() * 0.4f)));
    g.setColour(juce::Colour(0xFF2A9D8F));
    g.fillRect(area);
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(13.0f));
    g.drawText("centre rectangle must stay background-coloured",
               juce::Rectangle<float>(0, area.getHeight() - 24.0f, area.getWidth(), 20.0f),
               juce::Justification::centred);
}

// =============================================================================
// Scene registry
// =============================================================================

const std::vector<ComplianceEditor::Scene>& ComplianceEditor::scenes()
{
    static const std::vector<Scene> s = {
        { "Deep clip stack (path->path->rect)", &sceneDeepClips,         nullptr },
        { "Blend effect inside path clip",      &sceneBlendInClip,
          "noise is jvk-only: appears on LIVE pane only; clip SHAPE must match" },
        { "Paths (fill / gradient / stroke)",   &scenePaths,             nullptr },
        { "Text (sizes + rotated/sheared)",     &sceneText,
          "rotated/sheared runs render upright until the transformed-text fix lands" },
        { "Gradients (linear / radial)",        &sceneGradients,         nullptr },
        { "Rect lists / shapes / lines",        &sceneRectsAndLists,     nullptr },
        { "Images (draw / transform / alpha / tiled)", &sceneImages,
          "tiled image fills still diverge (setTiledImageFill unimplemented in jvk)" },
        { "Effects with regions",               &sceneEffectsRegion,
          "jvk-only ops: LIVE pane only; regions take effect once ROI scissoring lands" },
        { "Transparency layer",                 &sceneTransparencyLayer,
          "overlap darkens until true layers land (Phase 6)" },
        { "Opacity contract (setOpacity/fill alpha)", &sceneOpacityContract, nullptr },
        { "Exclude clip region",                &sceneExcludeClip,
          "centre not excluded until excludeClipRectangle lands" },
    };
    return s;
}
