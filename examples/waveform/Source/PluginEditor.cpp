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
