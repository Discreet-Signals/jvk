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
 File: Buffer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

struct BufferInfo
{
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags properties;
};

class Buffer
{
public:
    Buffer() = default;

    explicit Buffer(const BufferInfo& info)
    {
        create(info);
    }

    ~Buffer()
    {
        destroy();
    }

    Buffer(Buffer&& other) noexcept
        : device(other.device), buffer(other.buffer), memory(other.memory),
          bufferSize(other.bufferSize), mapped(other.mapped),
          pool(other.pool), poolAlloc(other.poolAlloc)
    {
        other.device = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
        other.bufferSize = 0;
        other.mapped = nullptr;
        other.pool = nullptr;
        other.poolAlloc = {};
    }

    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            device = other.device;
            buffer = other.buffer;
            memory = other.memory;
            bufferSize = other.bufferSize;
            mapped = other.mapped;
            pool = other.pool;
            poolAlloc = other.poolAlloc;
            other.device = VK_NULL_HANDLE;
            other.buffer = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
            other.bufferSize = 0;
            other.mapped = nullptr;
            other.pool = nullptr;
            other.poolAlloc = {};
        }
        return *this;
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Standalone allocation (original path — one vkAllocateMemory per buffer)
    bool create(const BufferInfo& info)
    {
        destroy();
        device = info.device;
        bufferSize = info.size;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = info.size;
        bufferInfo.usage = info.usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create buffer!");
            return false;
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(info.physicalDevice,
                                                    memReq.memoryTypeBits,
                                                    info.properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        {
            DBG("jvk: Failed to allocate buffer memory!");
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            return false;
        }

        vkBindBufferMemory(device, buffer, memory, 0);
        return true;
    }

    // Pool-backed allocation — sub-allocates from a GPUMemoryPool block.
    // Reduces vkAllocateMemory count from O(resources) to O(1).
    bool create(const BufferInfo& info, GPUMemoryPool& memPool)
    {
        destroy();
        device = info.device;
        bufferSize = info.size;
        pool = &memPool;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = info.size;
        bufferInfo.usage = info.usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            DBG("jvk: Failed to create buffer!");
            pool = nullptr;
            return false;
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        poolAlloc = memPool.allocate(memReq, info.properties);
        if (poolAlloc.memory == VK_NULL_HANDLE)
        {
            DBG("jvk: Pool allocation failed for buffer!");
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            pool = nullptr;
            return false;
        }

        memory = poolAlloc.memory;
        vkBindBufferMemory(device, buffer, memory, poolAlloc.offset);
        return true;
    }

    void destroy()
    {
        if (mapped)
            unmap();
        if (buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkDestroyBuffer(device, buffer, nullptr);
        if (pool)
            pool->free(poolAlloc);
        else if (memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
            vkFreeMemory(device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        bufferSize = 0;
        pool = nullptr;
        poolAlloc = {};
    }

    void upload(const void* data, VkDeviceSize size, VkDeviceSize offset = 0)
    {
        if (pool)
        {
            // Pool-backed: use the block's persistent mapping
            void* dest = static_cast<char*>(poolAlloc.mappedBase) + poolAlloc.offset + offset;
            memcpy(dest, data, static_cast<size_t>(size));
        }
        else
        {
            void* dest;
            vkMapMemory(device, memory, offset, size, 0, &dest);
            memcpy(dest, data, static_cast<size_t>(size));
            vkUnmapMemory(device, memory);
        }
    }

    void* map()
    {
        if (!mapped)
        {
            if (pool)
                mapped = static_cast<char*>(poolAlloc.mappedBase) + poolAlloc.offset;
            else
                vkMapMemory(device, memory, 0, bufferSize, 0, &mapped);
        }
        return mapped;
    }

    void unmap()
    {
        if (mapped)
        {
            if (!pool)
                vkUnmapMemory(device, memory);
            mapped = nullptr;
        }
    }

    VkBuffer getBuffer() const { return buffer; }
    VkDeviceMemory getMemory() const { return memory; }
    VkDeviceSize getSize() const { return bufferSize; }
    bool isValid() const { return buffer != VK_NULL_HANDLE; }
    bool isPoolBacked() const { return pool != nullptr; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bufferSize = 0;
    void* mapped = nullptr;
    GPUMemoryPool* pool = nullptr;
    SubAllocation poolAlloc {};
};

} // jvk::core
