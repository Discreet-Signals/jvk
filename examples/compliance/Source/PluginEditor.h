/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: PluginEditor.h  (JVKCompliance — golden-image compliance harness)

 The anti-regression tool for jvk's juce::Graphics compliance work
 (CLEANUP_PLAN.md, "Verification strategy"). Each scene paints the SAME
 juce::Graphics code twice:

   LEFT  — reference: rendered into a juce::Image by the JUCE software
           renderer, shown as a texture. (Caveat: the reference travels
           through jvk's drawImage to reach the screen — but drawImage of a
           plain raster is the most basic op, and toggling Vulkan off ['V']
           removes even that, letting you sanity-check the harness itself.)
   RIGHT — live: painted through whichever renderer is active (jvk by
           default; press 'V' to flip the whole window to the JUCE software
           renderer and confirm the two halves become identical).

 Scenes cover the regression guardrails: deep clip stacks (path→path→rect),
 blend effects inside a path clip, complex paths, text (incl. the rotated/
 sheared runs), gradients, rect lists, images (alpha modulation + tiled
 fills), region-scissored effects (interior + edge-abutting, per G-B), and
 transparency layers. Scenes with known, planned divergences state them in
 the footer so a human diff reads "expected" vs "regression" at a glance.

 Keys: LEFT/RIGHT arrows — switch scene; 'V' — toggle renderer.
 ------------------------------------------------------------------------------
*/

#pragma once
#include "PluginProcessor.h"
#include <jvk/jvk.h>

class ComplianceEditor : public jvk::AudioProcessorEditor
{
public:
    explicit ComplianceEditor(ComplianceProcessor&);
    ~ComplianceEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress&) override;

private:
    using ScenePainter = void (*)(juce::Graphics&, juce::Rectangle<float>);

    struct Scene {
        const char*  name;
        ScenePainter painter;
        const char*  caveat;   // known/expected divergence, or nullptr
    };

    static const std::vector<Scene>& scenes();

    // Scene painters — pure juce::Graphics so both renderers accept them.
    // jvk-only extras (blur/saturate/noise) go through jvk::Graphics::create
    // and no-op under the software renderer (stated in the scene's caveat).
    static void sceneDeepClips        (juce::Graphics&, juce::Rectangle<float>);
    static void sceneBlendInClip      (juce::Graphics&, juce::Rectangle<float>);
    static void scenePaths            (juce::Graphics&, juce::Rectangle<float>);
    static void sceneText             (juce::Graphics&, juce::Rectangle<float>);
    static void sceneGradients        (juce::Graphics&, juce::Rectangle<float>);
    static void sceneRectsAndLists    (juce::Graphics&, juce::Rectangle<float>);
    static void sceneImages           (juce::Graphics&, juce::Rectangle<float>);
    static void sceneEffectsRegion    (juce::Graphics&, juce::Rectangle<float>);
    static void sceneTransparencyLayer(juce::Graphics&, juce::Rectangle<float>);
    static void sceneOpacityContract  (juce::Graphics&, juce::Rectangle<float>);
    static void sceneExcludeClip      (juce::Graphics&, juce::Rectangle<float>);

    static juce::Image makeTestImage(int w, int h);

    void setScene(int index);
    void rebuildReference();

    juce::Rectangle<int> headerArea()  const;
    juce::Rectangle<int> footerArea()  const;
    juce::Rectangle<int> leftPane()    const;
    juce::Rectangle<int> rightPane()   const;

    int         sceneIndex_ = 0;
    bool        vulkanOn_   = true;
    juce::Image reference_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ComplianceEditor)
};
