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
 File: PluginProcessor.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

CompressorProcessor::CompressorProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout CompressorProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("threshold", 1), "Threshold", -60.0f, 0.0f, -20.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("ratio", 1), "Ratio", 1.0f, 20.0f, 4.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("attack", 1), "Attack", 0.1f, 100.0f, 10.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("release", 1), "Release", 10.0f, 1000.0f, 100.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("makeup", 1), "Makeup", 0.0f, 24.0f, 0.0f));
    return { params.begin(), params.end() };
}

void CompressorProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec { sampleRate,
                                   static_cast<juce::uint32>(samplesPerBlock),
                                   static_cast<juce::uint32>(getTotalNumOutputChannels()) };
    compressor.prepare(spec);
}

void CompressorProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    compressor.setThreshold(*apvts.getRawParameterValue("threshold"));
    compressor.setRatio(*apvts.getRawParameterValue("ratio"));
    compressor.setAttack(*apvts.getRawParameterValue("attack"));
    compressor.setRelease(*apvts.getRawParameterValue("release"));

    // Measure input level
    float inPeak = buffer.getMagnitude(0, buffer.getNumSamples());
    inputLevel.store(inPeak);

    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    compressor.process(context);

    // Apply makeup gain
    float makeupDb = *apvts.getRawParameterValue("makeup");
    float makeupGain = juce::Decibels::decibelsToGain(makeupDb);
    buffer.applyGain(makeupGain);

    // Measure output level
    float outPeak = buffer.getMagnitude(0, buffer.getNumSamples());
    outputLevel.store(outPeak);
    gainReduction.store(inPeak > 0.0001f ? outPeak / inPeak : 1.0f);
}

juce::AudioProcessorEditor* CompressorProcessor::createEditor()
{
    return new CompressorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CompressorProcessor();
}
