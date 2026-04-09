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
 File: DescriptorHelper.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

class DescriptorHelper
{
public:
    explicit DescriptorHelper(VkDevice device) : device(device) {}

    ~DescriptorHelper()
    {
        destroy();
    }

    DescriptorHelper(DescriptorHelper&& other) noexcept
        : device(other.device), bindings(std::move(other.bindings)),
          poolSizes(std::move(other.poolSizes)), layout(other.layout), pool(other.pool)
    {
        other.layout = VK_NULL_HANDLE;
        other.pool = VK_NULL_HANDLE;
    }

    DescriptorHelper& operator=(DescriptorHelper&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            device = other.device;
            bindings = std::move(other.bindings);
            poolSizes = std::move(other.poolSizes);
            layout = other.layout;
            pool = other.pool;
            other.layout = VK_NULL_HANDLE;
            other.pool = VK_NULL_HANDLE;
        }
        return *this;
    }

    DescriptorHelper(const DescriptorHelper&) = delete;
    DescriptorHelper& operator=(const DescriptorHelper&) = delete;

    DescriptorHelper& addBinding(uint32_t binding, VkDescriptorType type,
                                  VkShaderStageFlags stages, uint32_t count = 1)
    {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = binding;
        b.descriptorType = type;
        b.descriptorCount = count;
        b.stageFlags = stages;
        bindings.push_back(b);

        // Track pool sizes
        bool found = false;
        for (auto& ps : poolSizes)
        {
            if (ps.type == type) { ps.descriptorCount += count; found = true; break; }
        }
        if (!found)
            poolSizes.push_back({ type, count });

        return *this;
    }

    VkDescriptorSetLayout buildLayout()
    {
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
            DBG("jvk: Failed to create descriptor set layout!");

        return layout;
    }

    bool createPool(uint32_t maxSets = 1)
    {
        // Scale pool sizes by maxSets
        std::vector<VkDescriptorPoolSize> scaledSizes = poolSizes;
        for (auto& ps : scaledSizes)
            ps.descriptorCount *= maxSets;

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(scaledSizes.size());
        poolInfo.pPoolSizes = scaledSizes.data();
        poolInfo.maxSets = maxSets;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create descriptor pool!");
            return false;
        }
        return true;
    }

    VkDescriptorSet allocateSet()
    {
        return allocateSet(layout);
    }

    VkDescriptorSet allocateSet(VkDescriptorSetLayout setLayout)
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &setLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &set) != VK_SUCCESS)
            DBG("jvk: Failed to allocate descriptor set!");

        return set;
    }

    // Static helpers to write descriptor bindings
    static void writeBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                             VkDescriptorType type, VkBuffer buffer,
                             VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE)
    {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = buffer;
        bufferInfo.offset = offset;
        bufferInfo.range = range;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = type;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    static void writeImage(VkDevice device, VkDescriptorSet set, uint32_t binding,
                            VkImageView view, VkSampler sampler,
                            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = imageLayout;
        imageInfo.imageView = view;
        imageInfo.sampler = sampler;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    void destroy()
    {
        if (pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, pool, nullptr);
        if (layout != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        pool = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
    }

    VkDescriptorSetLayout getLayout() const { return layout; }
    VkDescriptorPool getPool() const { return pool; }

private:
    VkDevice device = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorPoolSize> poolSizes;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

} // jvk::core
