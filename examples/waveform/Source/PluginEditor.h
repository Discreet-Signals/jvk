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
#include "Waveform.h"

//==============================================================================
/**
*/
class JvkWaveformAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    JvkWaveformAudioProcessorEditor (JvkWaveformAudioProcessor&);
    ~JvkWaveformAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    JvkWaveformAudioProcessor& audioProcessor;
    jvk::VulkanRenderer jvkRenderer;
    WaveformDemo jvkWaveform;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JvkWaveformAudioProcessorEditor)
};
