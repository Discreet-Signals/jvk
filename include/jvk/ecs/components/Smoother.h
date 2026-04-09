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
 File: Smoother.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// Base smoother — pooled as UpdaterVector<SmootherBase>
class SmootherBase : public Updater<SmootherBase>
{
public:
    virtual ~SmootherBase() = default;

    float smoothingMs = 100.0f;
    bool active = true;

    void update(double dt) override
    {
        if (!active) return;
        applySmoothing(dt);
    }

    virtual void applySmoothing(double dt) = 0;
};

// Typed smoother targeting a specific value
template <typename T>
class Smoother : public SmootherBase
{
public:
    Smoother(T* target, float ms)
        : target(target)
    {
        smoothingMs = ms;
        if (target)
            smoothTarget = *target;
    }

    void setTarget(const T& newTarget) { smoothTarget = newTarget; }
    const T& getTarget() const { return smoothTarget; }

    void snap()
    {
        if (target)
            *target = smoothTarget;
    }

    void applySmoothing(double dt) override
    {
        if (!target) return;
        float tau = smoothingMs / 1000.0f;
        float alpha = 1.0f - std::exp(static_cast<float>(-dt) / tau);
        *target = *target + (smoothTarget - *target) * alpha;
    }

    T* target = nullptr;
    T smoothTarget {};
};

} // jvk
