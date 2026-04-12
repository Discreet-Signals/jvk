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
 File: ShaderComponent.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

struct Vertex
{
    glm::vec3 position;
    glm::vec4 color;
};

class ShaderComponent : public VulkanComponent
{
public:
    template <typename SG>
    ShaderComponent(SG&& shader_group, int storage_buffer_size) :
    shaderGroup(std::forward<SG>(shader_group)),
    fifo(storage_buffer_size * 3, 0.0f),
    buffer(storage_buffer_size * 2, 0.0f)
    {
        for (int i = 0; i < storage_buffer_size; i++)
            buffer.push(0.0f);
        storageData.resize(storage_buffer_size, 0.0f);
    }

    void pushData(float* data, int num)
    {
        preProcessData(data, num);
        fifo.push_num(data, num);
    }
protected:
    virtual void preProcessData(float* data, int num) { }

private:
    void sendDataToShader()
    {
        int size = fifo.size();
        for (int i = 0; i < size; i++)
            buffer.push(fifo.pop());
        buffer.pop_num(size);
        buffer.read(storageData.data(), static_cast<int>(storageData.size()));

        // Persistent mapping — no map/unmap per frame
        if (storageMappedPtr)
            memcpy(storageMappedPtr, storageData.data(), sizeof(float) * storageData.size());
    }

    shaders::ShaderGroup shaderGroup;
    void addedToRenderer(const VulkanRenderer& renderer) override
    {
        device = renderer.getDevice();

        VkBufferCreateInfo vBuffInfo = {};
        vBuffInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vBuffInfo.size = sizeof(vertices[0]) * vertices.size();
        vBuffInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vBuffInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(renderer.getDevice(), &vBuffInfo, nullptr, &vertexBuffer) != VK_SUCCESS)
        {
            DBG("failed to create vertex buffer!");
            return;
        }

        VkMemoryRequirements vMemReq;
        vkGetBufferMemoryRequirements(renderer.getDevice(), vertexBuffer, &vMemReq);

        VkMemoryAllocateInfo vAllocInfo = {};
        vAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        vAllocInfo.allocationSize = vMemReq.size;
        vAllocInfo.memoryTypeIndex = core::findMemoryType(renderer.getPhysicalDevice(), vMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(renderer.getDevice(), &vAllocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS)
        {
            DBG("failed to allocate vertex buffer memory!");
            return;
        }

        vkBindBufferMemory(renderer.getDevice(), vertexBuffer, vertexBufferMemory, 0);

        void* vData;
        vkMapMemory(renderer.getDevice(), vertexBufferMemory, 0, vBuffInfo.size, 0, &vData);
        memcpy(vData, vertices.data(), (size_t)vBuffInfo.size);
        vkUnmapMemory(renderer.getDevice(), vertexBufferMemory);

        VkBufferCreateInfo iBuffInfo = {};
        iBuffInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        iBuffInfo.size = sizeof(uint32_t) * indices.size();
        iBuffInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        iBuffInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(renderer.getDevice(), &iBuffInfo, nullptr, &indexBuffer) != VK_SUCCESS)
        {
            DBG("failed to create index buffer!");
            return;
        }

        VkMemoryRequirements iMemReq;
        vkGetBufferMemoryRequirements(renderer.getDevice(), indexBuffer, &iMemReq);

        VkMemoryAllocateInfo iAllocInfo = {};
        iAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        iAllocInfo.allocationSize = iMemReq.size;
        iAllocInfo.memoryTypeIndex = core::findMemoryType(renderer.getPhysicalDevice(), iMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(renderer.getDevice(), &iAllocInfo, nullptr, &indexBufferMemory) != VK_SUCCESS)
        {
            DBG("failed to allocate vertex buffer memory!");
            return;
        }

        vkBindBufferMemory(renderer.getDevice(), indexBuffer, indexBufferMemory, 0);

        void* iData;
        vkMapMemory(renderer.getDevice(), indexBufferMemory, 0, iBuffInfo.size, 0, &iData);
        memcpy(iData, indices.data(), (size_t)iBuffInfo.size);
        vkUnmapMemory(renderer.getDevice(), indexBufferMemory);

        VkBufferCreateInfo sBuffInfo = {};
        sBuffInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sBuffInfo.size = sizeof(float) * storageData.size();
        sBuffInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        sBuffInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(renderer.getDevice(), &sBuffInfo, nullptr, &storageBuffer) != VK_SUCCESS)
        {
            DBG("failed to create storage buffer!");
            return;
        }

        VkMemoryRequirements sMemReq;
        vkGetBufferMemoryRequirements(renderer.getDevice(), storageBuffer, &sMemReq);

        VkMemoryAllocateInfo sAllocInfo = {};
        sAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        sAllocInfo.allocationSize = sMemReq.size;
        sAllocInfo.memoryTypeIndex = core::findMemoryType(renderer.getPhysicalDevice(), sMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(renderer.getDevice(), &sAllocInfo, nullptr, &storageBufferMemory) != VK_SUCCESS)
        {
            DBG("failed to allocate storage buffer memory!");
            return;
        }

        vkBindBufferMemory(renderer.getDevice(), storageBuffer, storageBufferMemory, 0);

        // Persistently map storage buffer — no per-frame map/unmap
        size_t memSize = sizeof(float) * storageData.size();
        vkMapMemory(renderer.getDevice(), storageBufferMemory, 0, memSize, 0, &storageMappedPtr);
        memcpy(storageMappedPtr, storageData.data(), memSize);

        VkDescriptorSetLayoutBinding storageBufferBinding = {};
        storageBufferBinding.binding = 0;
        storageBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBufferBinding.descriptorCount = 1;
        storageBufferBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount = 1;
        layoutCreateInfo.pBindings = &storageBufferBinding;

        if (vkCreateDescriptorSetLayout(renderer.getDevice(), &layoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            DBG("failed to create descriptor set layout!");

        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;
        poolCreateInfo.maxSets = 1;

        if (vkCreateDescriptorPool(renderer.getDevice(), &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            DBG("failed to create descriptor pool!");

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(renderer.getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS)
            DBG("failed to allocate descriptor set!");

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = storageBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet descriptorWrite = {};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(renderer.getDevice(), 1, &descriptorWrite, 0, nullptr);

        auto sg = shaderGroup; // copy — original preserved for rebuild
        sg.addDescriptorSetLayout(descriptorSetLayout);
        pipeline = std::make_unique<Pipeline>(renderer.getDevice(), renderer.getRenderPass(),
            std::move(sg), renderer.getMSAASamples(), VertexLayout::positionColor());
        setPipeline(pipeline->getInternal());
    }
    void removedFromRenderer(const VulkanRenderer& renderer) override
    {
        vkDestroyBuffer(renderer.getDevice(), vertexBuffer, nullptr);
        vkFreeMemory(renderer.getDevice(), vertexBufferMemory, nullptr);

        vkDestroyBuffer(renderer.getDevice(), indexBuffer, nullptr);
        vkFreeMemory(renderer.getDevice(), indexBufferMemory, nullptr);

        if (storageMappedPtr)
        {
            vkUnmapMemory(renderer.getDevice(), storageBufferMemory);
            storageMappedPtr = nullptr;
        }
        vkDestroyBuffer(renderer.getDevice(), storageBuffer, nullptr);
        vkFreeMemory(renderer.getDevice(), storageBufferMemory, nullptr);

        vkFreeDescriptorSets(renderer.getDevice(), descriptorPool, 1, &descriptorSet);
        vkDestroyDescriptorSetLayout(renderer.getDevice(), descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(renderer.getDevice(), descriptorPool, nullptr);
    }

    void render(VkCommandBuffer &commandBuffer) final override
    {
        VkDeviceSize offsets[] = {0};

        sendDataToShader();

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getLayout(), 0, 1, &descriptorSet, 0, VK_NULL_HANDLE);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexed(commandBuffer, 6, 1, 0, 0, 0);
    }

    std::vector<Vertex> vertices
    {
        {{ -1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},// Bottom Left
        {{  1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},// Bottom Right
        {{ -1.0f,  1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},// Top Left
        {{  1.0f,  1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}} // Top Right
    };
    std::vector<uint16_t> indices
    {
        0, 1, 2,  // First triangle
        1, 3, 2   // Second triangle
    };

    FIFO<float> fifo;
    FIFO<float> buffer;
    std::vector<float> storageData;
    VkDevice device;
    std::unique_ptr<Pipeline> pipeline;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer storageBuffer;
    VkDeviceMemory storageBufferMemory;
    void* storageMappedPtr = nullptr;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;

    juce::CriticalSection criticalSection;
};

} // jvk
