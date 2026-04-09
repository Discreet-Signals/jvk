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
