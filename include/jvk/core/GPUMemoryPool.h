/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: GPUMemoryPool.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

// Sub-allocation handle returned by GPUMemoryPool::allocate().
// Consumers bind their VkBuffer/VkImage to (memory, offset) instead of
// owning a dedicated VkDeviceMemory per resource.
struct SubAllocation
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   offset = 0;
    VkDeviceSize   size   = 0;
    uint32_t       blockIndex = UINT32_MAX;
    uint32_t       memoryTypeIndex = UINT32_MAX;
    void*          mappedBase = nullptr; // non-null if host-visible block
};

// Pool-based GPU memory sub-allocator.
//
// Instead of one vkAllocateMemory per resource (which Vulkan limits to
// ~4096 on many drivers), allocate large blocks and sub-allocate from
// them with a coalescing free-list.
//
// For a 10-100 MB audio plugin UI this typically means 2-3 blocks total
// across the app lifetime — well under any driver limit.
//
// Alignment and bufferImageGranularity are handled automatically.
struct GPUMemoryPool
{
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 64 * 1024 * 1024; // 64 MB

    struct FreeRange
    {
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    struct Block
    {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size = 0;
        void*          mapped = nullptr; // persistently mapped if host-visible
        std::vector<FreeRange> freeList; // sorted by offset, coalesced on free
    };

    // One block list per memory type index
    std::unordered_map<uint32_t, std::vector<Block>> pools;

    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps {};
    VkDeviceSize bufferImageGranularity = 1;
    VkDeviceSize blockSize = DEFAULT_BLOCK_SIZE;

    // Stats
    uint32_t totalAllocations = 0;
    uint32_t totalBlocks = 0;

    void init(VkPhysicalDevice pd, VkDevice d, VkDeviceSize preferredBlockSize = DEFAULT_BLOCK_SIZE)
    {
        physDevice = pd;
        device = d;
        blockSize = preferredBlockSize;
        vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physDevice, &props);
        bufferImageGranularity = props.limits.bufferImageGranularity;
    }

    // Allocate a sub-region from a pooled block.
    // Returns a SubAllocation whose (memory, offset) pair is ready for
    // vkBindBufferMemory / vkBindImageMemory.
    SubAllocation allocate(VkMemoryRequirements req, VkMemoryPropertyFlags properties)
    {
        uint32_t memTypeIndex = findMemoryTypeIndex(req.memoryTypeBits, properties);

        // Ensure alignment respects both the resource requirement and
        // bufferImageGranularity (images and buffers sharing a block
        // need granularity-aligned boundaries).
        VkDeviceSize alignment = std::max(req.alignment, bufferImageGranularity);

        auto& blocks = pools[memTypeIndex];

        // Try to sub-allocate from an existing block (first-fit)
        for (uint32_t bi = 0; bi < blocks.size(); bi++)
        {
            auto& block = blocks[bi];
            for (size_t fi = 0; fi < block.freeList.size(); fi++)
            {
                auto& fr = block.freeList[fi];
                VkDeviceSize alignedOffset = alignUp(fr.offset, alignment);
                VkDeviceSize padding = alignedOffset - fr.offset;

                if (fr.size >= padding + req.size)
                {
                    SubAllocation alloc;
                    alloc.memory = block.memory;
                    alloc.offset = alignedOffset;
                    alloc.size = req.size;
                    alloc.blockIndex = bi;
                    alloc.memoryTypeIndex = memTypeIndex;
                    alloc.mappedBase = block.mapped;

                    // Carve out the allocated region
                    if (padding > 0)
                    {
                        // Keep the padding as a free range before the allocation
                        VkDeviceSize remainAfter = fr.size - padding - req.size;
                        fr.size = padding; // shrink existing range to just the padding

                        if (remainAfter > 0)
                        {
                            FreeRange after { alignedOffset + req.size, remainAfter };
                            block.freeList.insert(block.freeList.begin() + static_cast<ptrdiff_t>(fi + 1), after);
                        }
                    }
                    else
                    {
                        VkDeviceSize remain = fr.size - req.size;
                        if (remain > 0)
                        {
                            fr.offset += req.size;
                            fr.size = remain;
                        }
                        else
                        {
                            block.freeList.erase(block.freeList.begin() + static_cast<ptrdiff_t>(fi));
                        }
                    }

                    totalAllocations++;
                    return alloc;
                }
            }
        }

        // No existing block has space — allocate a new one
        VkDeviceSize newBlockSize = std::max(blockSize, req.size);
        Block newBlock;
        newBlock.size = newBlockSize;

        VkMemoryAllocateInfo allocInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = newBlockSize;
        allocInfo.memoryTypeIndex = memTypeIndex;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &newBlock.memory) != VK_SUCCESS)
        {
            DBG("jvk::GPUMemoryPool: Failed to allocate " << (newBlockSize / (1024*1024)) << " MB block!");
            return {};
        }

        // Persistently map host-visible blocks
        bool isHostVisible = (memProps.memoryTypes[memTypeIndex].propertyFlags
                              & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        if (isHostVisible)
            vkMapMemory(device, newBlock.memory, 0, newBlockSize, 0, &newBlock.mapped);

        // Entire block is free initially
        newBlock.freeList.push_back({ 0, newBlockSize });

        uint32_t bi = static_cast<uint32_t>(blocks.size());
        blocks.push_back(std::move(newBlock));
        totalBlocks++;

        DBG("jvk::GPUMemoryPool: Allocated " << (int)(newBlockSize / (1024*1024))
            << " MB block (type " << (int)memTypeIndex << ", total blocks: " << (int)totalBlocks << ")");

        // Recurse to allocate from the new block (guaranteed to fit)
        return allocate(req, properties);
    }

    // Return a sub-allocation to the pool.
    // Coalesces with adjacent free ranges to prevent fragmentation.
    void free(SubAllocation alloc)
    {
        if (alloc.memory == VK_NULL_HANDLE) return;

        auto poolIt = pools.find(alloc.memoryTypeIndex);
        if (poolIt == pools.end() || alloc.blockIndex >= poolIt->second.size())
            return;

        auto& block = poolIt->second[alloc.blockIndex];
        FreeRange newRange { alloc.offset, alloc.size };

        // Insert in sorted order by offset
        auto insertPos = block.freeList.begin();
        while (insertPos != block.freeList.end() && insertPos->offset < newRange.offset)
            ++insertPos;
        insertPos = block.freeList.insert(insertPos, newRange);

        // Coalesce with next range
        auto next = insertPos + 1;
        if (next != block.freeList.end()
            && insertPos->offset + insertPos->size == next->offset)
        {
            insertPos->size += next->size;
            block.freeList.erase(next);
        }

        // Coalesce with previous range
        if (insertPos != block.freeList.begin())
        {
            auto prev = insertPos - 1;
            if (prev->offset + prev->size == insertPos->offset)
            {
                prev->size += insertPos->size;
                block.freeList.erase(insertPos);
            }
        }

        totalAllocations--;
    }

    // Destroy all blocks and free all GPU memory.
    void destroy()
    {
        for (auto& [typeIdx, blocks] : pools)
        {
            for (auto& block : blocks)
            {
                if (block.mapped)
                    vkUnmapMemory(device, block.memory);
                if (block.memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, block.memory, nullptr);
            }
        }
        pools.clear();
        totalAllocations = 0;
        totalBlocks = 0;
    }

private:
    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    uint32_t findMemoryTypeIndex(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i))
                && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return 0;
    }
};

} // jvk::core
