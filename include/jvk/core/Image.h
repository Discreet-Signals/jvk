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
 File: Image.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

struct ImageInfo
{
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    VkImageUsageFlags usage;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
};

class Image
{
public:
    Image() = default;

    explicit Image(const ImageInfo& info)
    {
        create(info);
    }

    ~Image()
    {
        destroy();
    }

    Image(Image&& other) noexcept
        : device(other.device), image(other.image), memory(other.memory),
          view(other.view), sampler(other.sampler), w(other.w), h(other.h),
          pool(other.pool), poolAlloc(other.poolAlloc)
    {
        other.device = VK_NULL_HANDLE;
        other.image = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
        other.view = VK_NULL_HANDLE;
        other.sampler = VK_NULL_HANDLE;
        other.w = 0;
        other.h = 0;
        other.pool = nullptr;
        other.poolAlloc = {};
    }

    Image& operator=(Image&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            device = other.device;
            image = other.image;
            memory = other.memory;
            view = other.view;
            sampler = other.sampler;
            w = other.w;
            h = other.h;
            pool = other.pool;
            poolAlloc = other.poolAlloc;
            other.device = VK_NULL_HANDLE;
            other.image = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
            other.view = VK_NULL_HANDLE;
            other.sampler = VK_NULL_HANDLE;
            other.w = 0;
            other.h = 0;
            other.pool = nullptr;
            other.poolAlloc = {};
        }
        return *this;
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    // Standalone allocation (original path — one vkAllocateMemory per image)
    bool create(const ImageInfo& info)
    {
        destroy();
        device = info.device;
        w = info.width;
        h = info.height;

        // Create image
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = info.format;
        imageInfo.extent = { info.width, info.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = info.samples;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = info.usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create image!");
            return false;
        }

        // Allocate memory
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(info.physicalDevice,
                                                    memReq.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        {
            DBG("jvk: Failed to allocate image memory!");
            vkDestroyImage(device, image, nullptr);
            image = VK_NULL_HANDLE;
            return false;
        }

        vkBindImageMemory(device, image, memory, 0);

        // Create image view
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = info.format;
        viewInfo.subresourceRange.aspectMask = info.aspectMask;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create image view!");
            return false;
        }

        return true;
    }

    // Pool-backed allocation — sub-allocates from a GPUMemoryPool block.
    bool create(const ImageInfo& info, GPUMemoryPool& memPool)
    {
        destroy();
        device = info.device;
        w = info.width;
        h = info.height;
        pool = &memPool;

        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = info.format;
        imageInfo.extent = { info.width, info.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = info.samples;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = info.usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create image!");
            pool = nullptr;
            return false;
        }

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        poolAlloc = memPool.allocate(memReq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (poolAlloc.memory == VK_NULL_HANDLE)
        {
            DBG("jvk: Pool allocation failed for image!");
            vkDestroyImage(device, image, nullptr);
            image = VK_NULL_HANDLE;
            pool = nullptr;
            return false;
        }

        memory = poolAlloc.memory;
        vkBindImageMemory(device, image, memory, poolAlloc.offset);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = info.format;
        viewInfo.subresourceRange.aspectMask = info.aspectMask;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create image view!");
            return false;
        }

        return true;
    }

    void createSampler(VkFilter filter = VK_FILTER_LINEAR,
                        VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT)
    {
        if (sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
        }

        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = filter;
        samplerInfo.minFilter = filter;
        samplerInfo.addressModeU = addressMode;
        samplerInfo.addressModeV = addressMode;
        samplerInfo.addressModeW = addressMode;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
            DBG("jvk: Failed to create sampler!");
    }

    void upload(VkPhysicalDevice physDevice, VkCommandPool cmdPool, VkQueue queue,
                const void* pixels, uint32_t width, uint32_t height, VkFormat format)
    {
        VkDeviceSize imageSize = width * height * 4; // assumes 4 bytes per pixel

        // Create staging buffer
        Buffer staging;
        staging.create({ physDevice, device, imageSize,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
        staging.upload(pixels, imageSize);

        // Record and submit transfer commands
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &allocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Transition to TRANSFER_DST
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &barrier);

        // Copy buffer to image
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(cmd, staging.getBuffer(), image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Transition to SHADER_READ_ONLY
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &barrier);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    }

    void destroy()
    {
        if (sampler != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroySampler(device, sampler, nullptr);
        if (view != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroyImageView(device, view, nullptr);
        if (image != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroyImage(device, image, nullptr);
        if (pool)
            pool->free(poolAlloc);
        else if (memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkFreeMemory(device, memory, nullptr);
        sampler = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        w = 0;
        h = 0;
        pool = nullptr;
        poolAlloc = {};
    }

    VkImage getImage() const { return image; }
    VkImageView getView() const { return view; }
    VkSampler getSampler() const { return sampler; }
    uint32_t getWidth() const { return w; }
    uint32_t getHeight() const { return h; }
    bool isValid() const { return image != VK_NULL_HANDLE; }
    bool isPoolBacked() const { return pool != nullptr; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    GPUMemoryPool* pool = nullptr;
    SubAllocation poolAlloc {};
};

} // jvk::core
