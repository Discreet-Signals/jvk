/*
  ==============================================================================

    VulkanNSViewComponent.h
    Created: 11 Oct 2023 1:51:28pm
    Author:  Gavin

  ==============================================================================
*/

#pragma once

namespace jvk::core::macos
{

class VulkanNSViewComponent : public VulkanInstance, public juce::NSViewComponent
{
public:
    virtual void vulkanResized() {}
private:
    std::vector<const char*> getExtensions() override;
    void createSurface() override;
    void resized() final override;
    NSViewGenerator nsView;
};

} // jvk::core::macos
