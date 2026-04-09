/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
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
