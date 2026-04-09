/*
  ==============================================================================

    VulkanHWNDComponent.h
    Created: 11 Oct 2023 1:51:28pm
    Author:  Gavin

  ==============================================================================
*/

#pragma once

namespace jvk::core::windows
{

class VulkanHWNDComponent : public VulkanInstance, public juce::HWNDComponent
{
public:
    virtual void vulkanResized() {}
private:
    std::vector<const char*> getExtensions() override;
    void createSurface() override;
    void resized() final override;
    HWNDGenerator hwndGen;
};

} // jvk::core::windows
