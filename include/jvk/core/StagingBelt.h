/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: StagingBelt.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

// A texture upload that has been staged into a StagingBelt block
// but not yet recorded as Vulkan commands.
struct PendingUpload
{
    VkImage dstImage;
    uint32_t width, height;
    VkBuffer srcBuffer;
    VkDeviceSize srcOffset;
};

// Pool of reusable HOST_VISIBLE staging buffers for batched GPU uploads.
// Callers write pixel data via allocate(), then the upload commands
// (barriers + vkCmdCopyBufferToImage) are recorded into the main command
// buffer before the render pass by recordUploads().
//
// In steady state: blocks exist from warmup, allocate() just bumps a
// pointer, recycle() moves blocks back to the free list. Zero allocation.
struct StagingBelt
{
    static constexpr VkDeviceSize BLOCK_SIZE = 4 * 1024 * 1024; // 4 MB

    struct Block
    {
        Buffer buffer;
        void* mapped = nullptr;
        VkDeviceSize offset = 0; // write cursor
    };

    struct Allocation
    {
        void* mappedPtr;
        VkBuffer buffer;
        VkDeviceSize offset;
    };

    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    // Blocks currently being written to this frame.
    // Moved to the per-frame usedStagingBlocks after recording uploads.
    std::vector<Block> activeBlocks;

    // Blocks returned after a frame's fence signals — ready for reuse.
    std::vector<Block> freeBlocks;

    void init(VkPhysicalDevice pd, VkDevice d)
    {
        physDevice = pd;
        device = d;
    }

    // Allocate space in a staging buffer. Returns a mapped CPU pointer
    // to write into, plus the VkBuffer + offset for the copy command.
    Allocation allocate(VkDeviceSize size)
    {
        // Try the current active block
        if (!activeBlocks.empty())
        {
            auto& b = activeBlocks.back();
            if (b.offset + size <= b.buffer.getSize())
            {
                Allocation a { static_cast<char*>(b.mapped) + b.offset,
                               b.buffer.getBuffer(), b.offset };
                b.offset += size;
                return a;
            }
        }

        // Need a new block — check free list first (zero allocation in steady state)
        Block block;
        if (!freeBlocks.empty())
        {
            block = std::move(freeBlocks.back());
            freeBlocks.pop_back();
            block.offset = 0;
        }
        else
        {
            VkDeviceSize blockSize = std::max(BLOCK_SIZE, size);
            block.buffer.create({ physDevice, device, blockSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
            block.mapped = block.buffer.map();
        }

        Allocation a { block.mapped, block.buffer.getBuffer(), 0 };
        block.offset = size;
        activeBlocks.push_back(std::move(block));
        return a;
    }

    // After recording upload commands, move active blocks to the per-frame
    // used list so they stay alive until the frame's fence signals.
    void moveActiveTo(std::vector<Block>& usedBlocks)
    {
        for (auto& b : activeBlocks)
            usedBlocks.push_back(std::move(b));
        activeBlocks.clear();
    }

    // Return used blocks from a completed frame to the free list.
    // Called after the frame's fence signals — the GPU is done reading.
    void recycle(std::vector<Block>& usedBlocks)
    {
        for (auto& b : usedBlocks)
            freeBlocks.push_back(std::move(b));
        usedBlocks.clear();
    }

    // Stage pixel data for deferred upload of an Image.
    // Copies pixels into staging memory and appends a PendingUpload record.
    // No GPU submission — purely CPU work.
    void stageImageUpload(Image& img, std::vector<PendingUpload>& uploads,
                          const void* pixels, uint32_t width, uint32_t height)
    {
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
        auto alloc = allocate(imageSize);
        memcpy(alloc.mappedPtr, pixels, static_cast<size_t>(imageSize));
        uploads.push_back({ img.getImage(), width, height, alloc.buffer, alloc.offset });
    }

    // Record all pending upload commands into a command buffer.
    // Must be called BEFORE vkCmdBeginRenderPass.
    static void recordUploads(VkCommandBuffer cmd, std::vector<PendingUpload>& uploads)
    {
        for (auto& u : uploads)
        {
            // Transition: UNDEFINED → TRANSFER_DST_OPTIMAL
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = u.dstImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Copy staging buffer → image
            VkBufferImageCopy region = {};
            region.bufferOffset = u.srcOffset;
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageExtent = { u.width, u.height, 1 };

            vkCmdCopyBufferToImage(cmd, u.srcBuffer, u.dstImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            // Transition: TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
        uploads.clear();
    }

    void destroy()
    {
        for (auto& b : activeBlocks) b.buffer.destroy();
        for (auto& b : freeBlocks) b.buffer.destroy();
        activeBlocks.clear();
        freeBlocks.clear();
    }
};

} // jvk::core
