/*
  ==============================================================================

    VulkanWindowRenderer.h
    Attaches to an existing JUCE window for full GPU rendering.
    Alternative to VulkanRenderer — uses the DAW/host's window instead of
    creating a new native surface.

  ==============================================================================
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
