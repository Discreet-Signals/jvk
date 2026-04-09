/*
  ==============================================================================

    SwapChain.cpp
    Created: 13 Oct 2023 2:38:19pm
    Author:  Gavin

  ==============================================================================
*/

#include "SwapChain.h"

namespace jvk::core
{

SwapChainSupportDetails SwapChain::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, info.surface, &details.capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, info.surface, &formatCount, nullptr);
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, info.surface, &formatCount, details.formats.data());
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, info.surface, &presentModeCount, nullptr);
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, info.surface, &presentModeCount, details.presentModes.data());
    return details;
}

VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    // Default to UNORM for JUCE-identical alpha blending.
    // User can switch to SRGB for physically correct rendering via setSRGBPipeline(true).
    VkFormat preferred = info.useSRGBFormat ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == preferred)
            return availableFormat;
    }
    return availableFormats[0];
}

VkPresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& mode : availablePresentModes)
        if (mode == info.preferredPresentMode) return mode;
    return VK_PRESENT_MODE_FIFO_KHR; // guaranteed by spec
}

VkExtent2D SwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    // Always use our requested size (which may be 2x for supersampling),
    // clamped to the surface's allowed range. Don't blindly use
    // capabilities.currentExtent — on macOS it may be in logical points.
    auto clamp = [](float val, uint32_t lo, uint32_t hi) -> uint32_t {
        return static_cast<uint32_t>(std::fmax(lo, std::fmin(static_cast<float>(hi), val)));
    };
    return {
        clamp(info.size.x, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        clamp(info.size.y, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

SwapChain::SwapChain(SwapChainInfo sc_info, VkSwapchainKHR previous) : info(sc_info)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(info.physicalDevice );
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D swapExtent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = info.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = swapExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = { info.graphicsQueueFamilyIndex, info.presentQueueFamilyIndex };
    if (info.graphicsQueueFamilyIndex != info.presentQueueFamilyIndex)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    // Pick a supported composite alpha mode
    VkCompositeAlphaFlagBitsKHR compositeAlphaPrefs[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
    };
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (auto pref : compositeAlphaPrefs)
    {
        if (swapChainSupport.capabilities.supportedCompositeAlpha & pref)
        {
            createInfo.compositeAlpha = pref;
            break;
        }
    }
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    // If we're recreating the swap chain, reference the old one (useful for window resizing, etc.)
    createInfo.oldSwapchain = previous;

    vkCreateSwapchainKHR(info.device, &createInfo, nullptr, &swapChain);

    // Retrieve swap chain images
    vkGetSwapchainImagesKHR(info.device, swapChain, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(info.device, swapChain, &imageCount, images.data());

    // Store the chosen format and extent for future use
    format = surfaceFormat.format;
    extent = swapExtent;

    createMsaaImages();
    createDepthImages();
    createImageViews();

    if (info.renderPass)
        createFrameBuffers(info.renderPass);
}

void SwapChain::createMsaaImages()
{
    msaaImages.resize(images.size());
    msaaImageMemory.resize(images.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = info.msaaSamples;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vkCreateImage(info.device, &imageInfo, nullptr, &msaaImages[i]);

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(info.device, msaaImages[i], &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(info.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkAllocateMemory(info.device, &allocInfo, nullptr, &msaaImageMemory[i]);

        vkBindImageMemory(info.device, msaaImages[i], msaaImageMemory[i], 0);
    }
}


void SwapChain::createDepthImages()
{
    depthImages.resize(images.size());
    depthImageMemory.resize(images.size());
    depthImageViews.resize(images.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_D32_SFLOAT;
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = info.msaaSamples;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vkCreateImage(info.device, &imageInfo, nullptr, &depthImages[i]);

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(info.device, depthImages[i], &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(info.physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkAllocateMemory(info.device, &allocInfo, nullptr, &depthImageMemory[i]);
        vkBindImageMemory(info.device, depthImages[i], depthImageMemory[i], 0);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = depthImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(info.device, &viewInfo, nullptr, &depthImageViews[i]);
    }
}

void SwapChain::createImageViews()
{
    msaaImageViews.resize(msaaImages.size());
    for (size_t i = 0; i < msaaImages.size(); i++)
    {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = msaaImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(info.device, &createInfo, nullptr, &msaaImageViews[i]);
    }
    imageViews.resize(images.size());
    for (size_t i = 0; i < images.size(); i++)
    {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = images[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(info.device, &createInfo, nullptr, &imageViews[i]);
    }
}

void SwapChain::createFrameBuffers(VkRenderPass renderPass)
{
    frameBuffers.resize(imageViews.size());
    for (size_t i = 0; i < imageViews.size(); i++)
    {
        VkImageView attachments[] = {
            msaaImageViews[i],
            imageViews[i],
            depthImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(std::size(attachments));
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;

        vkCreateFramebuffer(info.device, &framebufferInfo, nullptr, &frameBuffers[i]);
    }
}

SwapChain::~SwapChain()
{
    for (auto framebuffer : frameBuffers)
        vkDestroyFramebuffer(info.device, framebuffer, nullptr);
    for (auto imageView : imageViews)
        vkDestroyImageView(info.device, imageView, nullptr);
    for (auto imageView : msaaImageViews)
        vkDestroyImageView(info.device, imageView, nullptr);
    for (auto image : msaaImages)
        vkDestroyImage(info.device, image, nullptr);
    for (auto memory : msaaImageMemory)
        vkFreeMemory(info.device, memory, nullptr);
    for (auto imageView : depthImageViews)
        vkDestroyImageView(info.device, imageView, nullptr);
    for (auto image : depthImages)
        vkDestroyImage(info.device, image, nullptr);
    for (auto memory : depthImageMemory)
        vkFreeMemory(info.device, memory, nullptr);
    vkDestroySwapchainKHR(info.device, swapChain, nullptr);
}

} // jvk::core
