/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
JvkWaveformAudioProcessorEditor::JvkWaveformAudioProcessorEditor (JvkWaveformAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    audioProcessor.jvkWaveform = &jvkWaveform;
    addAndMakeVisible(jvkRenderer);
    jvkRenderer.addChildComponent(&jvkWaveform);
    setSize (1024, 256);
}

JvkWaveformAudioProcessorEditor::~JvkWaveformAudioProcessorEditor()
{
    audioProcessor.jvkWaveform = nullptr;
}

//==============================================================================
void JvkWaveformAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::green);
}

void JvkWaveformAudioProcessorEditor::resized()
{
    jvkRenderer.setBounds(getBounds());
    jvkWaveform.setBounds(getBounds().toFloat());
}
