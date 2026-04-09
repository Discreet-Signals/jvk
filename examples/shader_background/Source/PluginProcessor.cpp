#include "PluginProcessor.h"
#include "PluginEditor.h"

ShaderBgProcessor::ShaderBgProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout ShaderBgProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("drive", 1), "Drive",
        juce::NormalisableRange<float>(0.0f, 36.0f, 0.1f), 0.0f, "dB"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("tone", 1), "Tone",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("level", 1), "Level",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f, "dB"));

    return { params.begin(), params.end() };
}

void ShaderBgProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    // 20ms attack, 200ms release envelope coefficients
    attackCoeff  = 1.0f - std::exp(-1.0f / (float(sampleRate) * 0.020f));
    releaseCoeff = 1.0f - std::exp(-1.0f / (float(sampleRate) * 0.200f));
    envelopeSmoothed = 0.0f;
}

bool ShaderBgProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return layouts.getMainOutputChannelSet() == layouts.getMainInputChannelSet();
}

void ShaderBgProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const float driveDb = apvts.getRawParameterValue("drive")->load();
    const float toneVal = apvts.getRawParameterValue("tone")->load();
    const float levelDb = apvts.getRawParameterValue("level")->load();

    const float driveLinear = juce::Decibels::decibelsToGain(driveDb);
    const float levelLinear = juce::Decibels::decibelsToGain(levelDb);
    const float bias   = toneVal;           // DC bias before tanh
    const float biasDC = std::tanh(bias);   // DC to subtract after tanh
    const float invDrive = 1.0f / std::max(driveLinear, 0.0001f);

    const int numChannels = buffer.getNumChannels();
    const int numSamples  = buffer.getNumSamples();

    // Cache channel pointers
    float* ch[2] = {};
    const int maxCh = juce::jmin(numChannels, 2);
    for (int c = 0; c < maxCh; c++)
        ch[c] = buffer.getWritePointer(c);

    for (int i = 0; i < numSamples; i++)
    {
        float peakLevel = 0.0f;

        for (int c = 0; c < maxCh; c++)
        {
            float x = ch[c][i] * driveLinear;
            peakLevel = std::max(peakLevel, std::fabs(x));

            // tanh with tone bias, analytical DC removal, auto gain correction
            ch[c][i] = (std::tanh(x + bias) - biasDC) * invDrive * levelLinear;
        }

        // Analytical saturation metric: 1 - tanh(x)/x
        // Exactly 0 in the linear region, approaches 1 when hard-clipping
        float satInst = peakLevel > 0.001f
            ? 1.0f - std::tanh(peakLevel) / peakLevel
            : 0.0f;

        float coeff = satInst > envelopeSmoothed ? attackCoeff : releaseCoeff;
        envelopeSmoothed += coeff * (satInst - envelopeSmoothed);
    }

    saturationEnvelope.store(envelopeSmoothed);
}

juce::AudioProcessorEditor* ShaderBgProcessor::createEditor()
{
    return new ShaderBgEditor(*this);
}

void ShaderBgProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void ShaderBgProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ShaderBgProcessor();
}
