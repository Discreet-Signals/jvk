#pragma once

#include "PluginProcessor.h"
#include "Triangle.h"

class JuceVulkanAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    JuceVulkanAudioProcessorEditor (JuceVulkanAudioProcessor&);
    ~JuceVulkanAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    JuceVulkanAudioProcessor& audioProcessor;
    jvk::VulkanRenderer jvkRenderer;
    TriangleDemo tlTriangle;
    TriangleDemo brTriangle;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JuceVulkanAudioProcessorEditor)
};
