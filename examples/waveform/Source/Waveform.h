/*
  ==============================================================================

    Waveform.h
    Created: 19 Oct 2023 2:03:14pm
    Author:  Gavin

  ==============================================================================
*/

#pragma once
#include <BinaryData.h>
#include <juce_audio_basics/juce_audio_basics.h>
#define SIZE 1024

class WaveformDemo : public jvk::ShaderComponent
{
public:
    WaveformDemo() :
        ShaderComponent(jvk::shaders::ShaderGroup()
                        .addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                   jvk::shaders::defaults::basic_vert_spv,
                                   jvk::shaders::defaults::basic_vert_spvSize)
                        .addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                   BinaryData::Waveform_frag_spv,
                                   BinaryData::Waveform_frag_spvSize), SIZE),
        fifo(2 * SIZE, 0.0f)
    { }
    ~WaveformDemo() { }

    void pushAudio(const float* data, int num)
    {
        fifo.push_num(data, num);

        while (fifo.size() > 512)
        {
            float peak = 0.0f;
            for (int i = 0; i < 512; i++)
            {
                float s = std::fabs(fifo.pop());
                if (s > peak)
                    peak = s;
            }
            peak = static_cast<float>((juce::Decibels::gainToDecibels(peak) + 54.0) / 60.0);
            pushData(&peak, 1);
        }
    }
private:
    jvk::FIFO<float> fifo;

    void preProcessData(float *data, int num) override
    {

    }
};
