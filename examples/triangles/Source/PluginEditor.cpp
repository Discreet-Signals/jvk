/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
JuceVulkanAudioProcessorEditor::JuceVulkanAudioProcessorEditor (JuceVulkanAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p), tlTriangle(false), brTriangle(true)
{
    addAndMakeVisible(jvkRenderer);
    jvkRenderer.addChildComponentWithDefaultPipeline(&tlTriangle);
    jvkRenderer.addChildComponentWithDefaultPipeline(&brTriangle);
    setSize (800, 600);
    setResizable(true, true);
}

JuceVulkanAudioProcessorEditor::~JuceVulkanAudioProcessorEditor()
{
}

//==============================================================================
void JuceVulkanAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(juce::Colours::midnightblue);
    g.setColour(juce::Colours::white.withBrightness(0.85f));
    g.setFont(getHeight() * 0.2f);
    g.drawText("JVK", 0, 0, getWidth(), getHeight(), juce::Justification::centred);
}

void JuceVulkanAudioProcessorEditor::resized()
{
    jvkRenderer.setBounds(getBounds());
    tlTriangle.setBounds({0.0f, 0.0f, getWidth() * 0.5f, getHeight() * 0.5f});
    brTriangle.setBounds({getWidth() * 0.5f, getHeight() * 0.5f, getWidth() * 0.5f, getHeight() * 0.5f});
}
