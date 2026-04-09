#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <BinaryData.h>

namespace
{

struct DarkLookAndFeel : public juce::LookAndFeel_V4
{
    DarkLookAndFeel()
    {
        // Text editor popup (when typing into slider value)
        setColour(juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
        setColour(juce::TextEditor::textColourId, juce::Colour(0xFFFFFFFF));
        setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
        setColour(juce::TextEditor::focusedOutlineColourId, juce::Colours::transparentBlack);
        setColour(juce::TextEditor::highlightColourId, juce::Colour(0x40FFFFFF));
        setColour(juce::TextEditor::highlightedTextColourId, juce::Colour(0xFFFFFFFF));
        setColour(juce::CaretComponent::caretColourId, juce::Colour(0xFFFFFFFF));
    }
};

} // anonymous namespace

static void styleDriveKnob(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 20);
    slider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::thumbColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xFFFFFFFF));
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    slider.setTextValueSuffix(" dB");
}

struct SliderFillPainter : public juce::Component
{
    juce::Slider& slider;

    SliderFillPainter(juce::Slider& s) : slider(s)
    {
        setInterceptsMouseClicks(false, false);
    }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        constexpr float cr = 8.0f;

        juce::Path outline;
        outline.addRoundedRectangle(bounds.reduced(1.0f), cr);

        float pos = (float)slider.valueToProportionOfLength(slider.getValue());
        float fillH = bounds.getHeight() * pos;
        auto fillRect = bounds.withTop(bounds.getBottom() - fillH);

        g.saveState();
        g.reduceClipRegion(outline);
        g.setColour(juce::Colour(0xFFFFFFFF));
        g.fillRect(fillRect);
        g.restoreState();

        g.setColour(juce::Colour(0xFFFFFFFF));
        g.strokePath(outline, juce::PathStrokeType(2.0f));
    }
};

static void styleSmallSlider(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::LinearBarVertical);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    slider.setAlpha(0.0f);
}

static void styleLabel(juce::Label& label, const juce::String& text)
{
    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.setColour(juce::Label::textColourId, juce::Colour(0xFFFFFFFF));
    label.setFont(juce::Font(11.0f));
}

ShaderBgEditor::ShaderBgEditor(ShaderBgProcessor& p)
    : AudioProcessorEditor(&p), processor(p)
{
    lnf = std::make_unique<DarkLookAndFeel>();
    setLookAndFeel(lnf.get());

    background = std::make_unique<jvk::ShaderImage>(
        BinaryData::background_frag_spv,
        BinaryData::background_frag_spvSize,
        400, 300, 1 // 1 float: saturation envelope
    );
    background->setFrameRate(30);

    // Drive knob — large, centered, aligned with sphere
    styleDriveKnob(driveSlider);
    addAndMakeVisible(driveSlider);
    driveAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "drive", driveSlider);

    // Tone slider — small, bottom-left
    tonePainter = std::make_unique<SliderFillPainter>(toneSlider);
    tonePainter->setAlpha(0.5f);
    addAndMakeVisible(*tonePainter);
    styleSmallSlider(toneSlider);
    addAndMakeVisible(toneSlider);
    toneAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "tone", toneSlider);

    // Level slider — small, bottom-right
    levelPainter = std::make_unique<SliderFillPainter>(levelSlider);
    levelPainter->setAlpha(0.5f);
    addAndMakeVisible(*levelPainter);
    styleSmallSlider(levelSlider);
    levelSlider.setTextValueSuffix(" dB");
    addAndMakeVisible(levelSlider);
    levelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processor.apvts, "level", levelSlider);

    // Labels
    styleLabel(driveLabel, "DRIVE");
    addAndMakeVisible(driveLabel);
    styleLabel(toneLabel, "TONE");
    addAndMakeVisible(toneLabel);
    styleLabel(levelLabel, "LEVEL");
    addAndMakeVisible(levelLabel);

    setSize(400, 300);
    startTimerHz(30);
}

ShaderBgEditor::~ShaderBgEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void ShaderBgEditor::timerCallback()
{
    float sat = processor.saturationEnvelope.load();
    background->pushData(&sat, 1);
    repaint();
}

void ShaderBgEditor::paint(juce::Graphics& g)
{
    if (background->isReady())
        g.drawImage(background->getImage(), getLocalBounds().toFloat());
    else
        g.fillAll(juce::Colour(0xFF0E0E14));

    // Draw drive thumb on the sphere surface
    auto centre = getLocalBounds().getCentre().toFloat();
    constexpr float indicatorRadius = 70.0f;

    float sliderPos = (float)driveSlider.valueToProportionOfLength(driveSlider.getValue());
    auto params = driveSlider.getRotaryParameters();
    float angle = params.startAngleRadians + sliderPos * (params.endAngleRadians - params.startAngleRadians);

    float dotX = centre.x + indicatorRadius * std::sin(angle);
    float dotY = centre.y - indicatorRadius * std::cos(angle);

    // Track arc (full range)
    juce::Path arc;
    arc.addCentredArc(centre.x, centre.y, indicatorRadius, indicatorRadius, 0.0f,
                      params.startAngleRadians, params.endAngleRadians, true);
    g.setColour(juce::Colour(0x20000000));
    g.strokePath(arc, juce::PathStrokeType(1.2f, juce::PathStrokeType::curved,
                                            juce::PathStrokeType::rounded));

    // Dot with inner shadow
    float thumbR = 6.0f;

    // Base fill
    g.setColour(juce::Colour(0xDDFFFFFF));
    g.fillEllipse(dotX - thumbR, dotY - thumbR, 2 * thumbR, 2 * thumbR);

    // Inner shadow overlay: transparent center → dark edge
    juce::ColourGradient shadow(
        juce::Colour(0x00000000), dotX, dotY,
        juce::Colour(0x60000000), dotX + thumbR, dotY,
        true);
    shadow.addColour(0.55, juce::Colour(0x00000000));
    shadow.addColour(0.85, juce::Colour(0x40000000));
    g.setGradientFill(shadow);
    g.fillEllipse(dotX - thumbR, dotY - thumbR, 2 * thumbR, 2 * thumbR);

}

void ShaderBgEditor::resized()
{
    background->setSize(getWidth(), getHeight());

    auto area = getLocalBounds();

    // Tone — full-height vertical bar, left corner
    auto toneCol = area.removeFromLeft(55).reduced(6, 10);
    toneLabel.setBounds(toneCol.removeFromBottom(14));
    toneCol.removeFromBottom(4);
    toneSlider.setBounds(toneCol);
    tonePainter->setBounds(toneCol);

    // Level — full-height vertical bar, right corner
    auto levelCol = area.removeFromRight(55).reduced(6, 10);
    levelLabel.setBounds(levelCol.removeFromBottom(14));
    levelCol.removeFromBottom(4);
    levelSlider.setBounds(levelCol);
    levelPainter->setBounds(levelCol);

    // Drive knob fills centre
    auto driveArea = area.reduced(20, 10);
    driveLabel.setBounds(driveArea.removeFromBottom(16));
    driveSlider.setBounds(driveArea);
}
