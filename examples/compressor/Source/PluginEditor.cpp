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

static void styleKnob(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 18);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
}

static void styleLabel(juce::Label& label, const juce::String& text)
{
    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.setColour(juce::Label::textColourId, juce::Colours::white);
}

CompressorEditor::CompressorEditor(CompressorProcessor& p)
    : jvk::AudioProcessorEditor(p), processor(p)
{
    styleKnob(thresholdSlider); addAndMakeVisible(thresholdSlider);
    styleKnob(ratioSlider);     addAndMakeVisible(ratioSlider);
    styleKnob(attackSlider);    addAndMakeVisible(attackSlider);
    styleKnob(releaseSlider);   addAndMakeVisible(releaseSlider);
    styleKnob(makeupSlider);    addAndMakeVisible(makeupSlider);

    thresholdSlider.setTextValueSuffix(" dB");
    ratioSlider.setTextValueSuffix(":1");
    attackSlider.setTextValueSuffix(" ms");
    releaseSlider.setTextValueSuffix(" ms");
    makeupSlider.setTextValueSuffix(" dB");

    styleLabel(thresholdLabel, "THRESHOLD"); addAndMakeVisible(thresholdLabel);
    styleLabel(ratioLabel, "RATIO");         addAndMakeVisible(ratioLabel);
    styleLabel(attackLabel, "ATTACK");       addAndMakeVisible(attackLabel);
    styleLabel(releaseLabel, "RELEASE");     addAndMakeVisible(releaseLabel);
    styleLabel(makeupLabel, "MAKEUP");       addAndMakeVisible(makeupLabel);

    thresholdAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "threshold", thresholdSlider);
    ratioAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "ratio", ratioSlider);
    attackAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "attack", attackSlider);
    releaseAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "release", releaseSlider);
    makeupAttach = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "makeup", makeupSlider);

    vulkanToggle.setButtonText("VULKAN");
    vulkanToggle.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF6C63FF));
    vulkanToggle.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    vulkanToggle.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    vulkanToggle.onClick = [this]
    {
        bool nowEnabled = !isVulkanEnabled();
        setVulkanEnabled(nowEnabled);
        vulkanToggle.setButtonText(nowEnabled ? "VULKAN" : "JUCE");
        vulkanToggle.setColour(juce::TextButton::buttonColourId,
            nowEnabled ? juce::Colour(0xFF6C63FF) : juce::Colour(0xFF444444));
    };
    addAndMakeVisible(vulkanToggle);

    setSize(550, 400);
}

// =============================================================================
// Paint
// =============================================================================

void CompressorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF1A1A2E));

    g.setColour(juce::Colour(0xFF16213E));
    g.fillRect(0, 0, getWidth(), 40);

    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(18.0f));
    g.drawText("VULKAN COMPRESSOR", 15, 8, 300, 24, juce::Justification::centredLeft);

    auto meterArea = getLocalBounds().removeFromBottom(60).reduced(20, 5);
    g.setColour(juce::Colour(0xFF0F0F23));
    g.fillRoundedRectangle(meterArea.toFloat(), 6.0f);

    float inLevel = processor.inputLevel.load();
    float outLevel = processor.outputLevel.load();

    auto inBar = meterArea.reduced(4).removeFromLeft(meterArea.getWidth() / 2 - 4);
    auto outBar = meterArea.reduced(4).removeFromRight(meterArea.getWidth() / 2 - 4);

    g.setColour(juce::Colour(0xFF4ECCA3));
    g.fillRect(inBar.removeFromBottom(static_cast<int>(inBar.getHeight() * juce::jlimit(0.0f, 1.0f, inLevel))));

    g.setColour(juce::Colour(0xFF45B7D1));
    g.fillRect(outBar.removeFromBottom(static_cast<int>(outBar.getHeight() * juce::jlimit(0.0f, 1.0f, outLevel))));
}

void CompressorEditor::resized()
{
    vulkanToggle.setBounds(getWidth() - 80, 8, 65, 24);

    auto area = getLocalBounds();
    area.removeFromTop(45);
    area.removeFromBottom(65);

    int knobW = area.getWidth() / 5;

    auto knobArea = area;
    auto labelArea = knobArea.removeFromBottom(20);

    thresholdSlider.setBounds(knobArea.removeFromLeft(knobW));
    ratioSlider.setBounds(knobArea.removeFromLeft(knobW));
    attackSlider.setBounds(knobArea.removeFromLeft(knobW));
    releaseSlider.setBounds(knobArea.removeFromLeft(knobW));
    makeupSlider.setBounds(knobArea);

    thresholdLabel.setBounds(labelArea.removeFromLeft(knobW));
    ratioLabel.setBounds(labelArea.removeFromLeft(knobW));
    attackLabel.setBounds(labelArea.removeFromLeft(knobW));
    releaseLabel.setBounds(labelArea.removeFromLeft(knobW));
    makeupLabel.setBounds(labelArea);
}
