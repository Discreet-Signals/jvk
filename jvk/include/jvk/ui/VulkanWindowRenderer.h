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
 File: VulkanWindowRenderer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class VulkanWindowRenderer : public VulkanComponent,
                              private juce::ComponentListener,
                              private juce::Timer
{
public:
    VulkanWindowRenderer() = default;

    ~VulkanWindowRenderer() override
    {
        detach();
    }

    void attachToComponent(juce::Component& component)
    {
        detach();
        attachedComponent = &component;
        component.addComponentListener(this);
        // TODO: Get native handle via component.getPeer()->getNativeHandle()
        // Create VkSurface from native handle
        // Initialize Vulkan swapchain on that surface
    }

    void detach()
    {
        if (attachedComponent)
        {
            attachedComponent->removeComponentListener(this);
            attachedComponent = nullptr;
        }
        stopTimer();
    }

    // Component hierarchy (same API as VulkanRenderer)
    void addChildComponent(VulkanComponent* child, bool sortChildren = true)
    {
        VulkanComponent::addChildComponent(child, sortChildren);
    }

    void setFrameRate(int fps)
    {
        if (fps > 0)
            startTimerHz(fps);
        else
            stopTimer();
    }

private:
    void componentMovedOrResized(juce::Component& component, bool, bool wasResized) override
    {
        if (wasResized && &component == attachedComponent)
        {
            // TODO: Recreate swapchain for new size
        }
    }

    void componentBeingDeleted(juce::Component&) override
    {
        detach();
    }

    void timerCallback() override
    {
        // TODO: Execute render loop on the attached window's surface
    }

    juce::Component* attachedComponent = nullptr;
};

} // jvk
