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
 File: Animator.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// Built-in interpolation functions (phase 0..1 -> curve 0..1)
namespace interp
{
    inline float linear(float t)     { return t; }
    inline float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }
    inline float cosine(float t)     { return 0.5f * (1.0f - std::cos(glm::pi<float>() * t)); }
    inline float easeIn(float t)     { return t * t; }
    inline float easeOut(float t)    { return t * (2.0f - t); }
    inline float easeInOut(float t)  { return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t; }
}

// Base animator — pooled as UpdaterVector<AnimatorBase>
class AnimatorBase : public Updater<AnimatorBase>
{
public:
    virtual ~AnimatorBase() = default;

    float phase = 0.0f;
    float phasePerSecond = 0.0f; // 1.0 / durationSeconds
    bool active = false;
    bool looping = false;

    void update(double dt) override
    {
        if (!active) return;

        phase += phasePerSecond * static_cast<float>(dt);

        if (phase >= 1.0f)
        {
            if (looping)
                phase -= 1.0f;
            else
            {
                phase = 1.0f;
                active = false;
            }
        }
        applyValue();
    }

    virtual void applyValue() = 0;
};

// Typed animator targeting a specific value
template <typename T>
class Animator : public AnimatorBase
{
public:
    using InterpFn = std::function<float(float)>;

    Animator(T* target, T startValue, T endValue, float durationSeconds,
             InterpFn interpolation = interp::linear)
        : target(target), startValue(startValue), endValue(endValue),
          interpolation(std::move(interpolation))
    {
        phasePerSecond = 1.0f / durationSeconds;
        active = true;
        phase = 0.0f;
    }

    void set(T start, T end, float durationSeconds)
    {
        startValue = start;
        endValue = end;
        phasePerSecond = 1.0f / durationSeconds;
        phase = 0.0f;
        active = true;
    }

    void setInterpolation(InterpFn fn) { interpolation = std::move(fn); }
    void setLooping(bool loop) { looping = loop; }

    void applyValue() override
    {
        if (!target) return;
        float t = interpolation(phase);
        *target = glm::mix(startValue, endValue, t);
    }

    T* target = nullptr;
    T startValue {};
    T endValue {};
    InterpFn interpolation = interp::linear;
};

} // jvk
