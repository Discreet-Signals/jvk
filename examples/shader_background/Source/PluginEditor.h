#pragma once

#include "PluginProcessor.h"

class ShaderBgEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    ShaderBgEditor(ShaderBgProcessor&);
    ~ShaderBgEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    ShaderBgProcessor& processor;

    // Shader background
    std::unique_ptr<jvk::ShaderImage> background;
    std::unique_ptr<juce::LookAndFeel_V4> lnf;

    // Controls
    juce::Slider driveSlider, toneSlider, levelSlider;
    juce::Label driveLabel, toneLabel, levelLabel;
    std::unique_ptr<juce::Component> tonePainter, levelPainter;

    // Parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> driveAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> toneAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> levelAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ShaderBgEditor)
};
