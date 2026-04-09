/*
  ==============================================================================

    VulkanNSViewComponent.cpp
    Created: 11 Oct 2023 1:52:05pm
    Author:  Gavin

  ==============================================================================
*/

#include "VulkanNSViewComponent.h"

namespace jvk::core::macos
{

std::vector<const char*> VulkanNSViewComponent::getExtensions()
{
    return {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_MVK_MACOS_SURFACE_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
    };
}
void VulkanNSViewComponent::createSurface()
{
    DBG("Creating Surface...");
    VkMacOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
    surfaceCreateInfo.pNext = nullptr;
    surfaceCreateInfo.flags = 0;
    setView(nsView.create());
    surfaceCreateInfo.pView = getView();
    if (vkCreateMacOSSurfaceMVK(instance, &surfaceCreateInfo, nullptr, &surface) != VK_SUCCESS)
    {
        DBG("Failed to create a Vulkan surface!");
        return;
    }
    DBG("Created Surface!");
}

void VulkanNSViewComponent::resized()
{
    resizeViewToFit();
    // Store pixel dimensions (accounting for contentsScale on the CAMetalLayer)
    float scale = 2.0f; // matches NSViewGenerator contentsScale
    width.store(static_cast<int>(getWidth() * scale));
    height.store(static_cast<int>(getHeight() * scale));
    windowResized.store(true);
    vulkanResized();
}

} // jvk::core::macos
