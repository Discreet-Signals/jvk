/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: VertexRingBuffer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

// Per-frame linear vertex allocator with doubling growth.
//
// Design follows bgfx "transient buffers" and Diligent Engine's approach:
// each frame slot owns a separate buffer. beginFrame() resets the write head
// to 0. write() bumps the pointer linearly. If the buffer is too small,
// it doubles — the old buffer is retired to the DeletionQueue (stays alive
// for in-flight GPU reads) and a new 2x buffer takes its place.
//
// No ring wrap. No cross-frame hazards. No fallback path.
//
// In steady state (after warmup): write() is a memcpy + pointer bump.
// Zero allocations, zero DeletionQueue traffic. Buffers are persistently
// mapped HOST_VISIBLE|HOST_COHERENT.
//
// References:
// - bgfx transient buffers (Branimir Karadzic)
// - Diligent Engine GPURingBuffer (Egor Yusov)
// - "Writing an efficient Vulkan renderer" (Arseny Kapoulkine / zeux)
struct VertexRingBuffer
{
    static constexpr VkDeviceSize INITIAL_CAPACITY = 8 * 1024 * 1024; // 8 MB per slot
    static constexpr int MAX_FRAME_SLOTS = 2; // matches MAX_FRAMES_IN_FLIGHT

    struct Slot
    {
        Buffer buffer;
        void*  mapped = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize writeHead = 0;
    };

    Slot slots[MAX_FRAME_SLOTS];
    int currentSlot = 0;

    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    DeletionQueue* deletionQueue = nullptr;

    bool create(VkPhysicalDevice pd, VkDevice d,
                VkDeviceSize slotCapacity = INITIAL_CAPACITY)
    {
        physDevice = pd;
        device = d;

        for (int i = 0; i < MAX_FRAME_SLOTS; i++)
        {
            slots[i].capacity = slotCapacity;
            bool ok = slots[i].buffer.create({ pd, d, slotCapacity,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
            if (!ok) return false;
            slots[i].mapped = slots[i].buffer.map();
            if (!slots[i].mapped) return false;
        }
        return true;
    }

    // Call at the start of each frame. Resets the slot's write head to 0.
    // Safe because the fence for this slot was just waited on — the GPU
    // finished reading this slot's previous frame data.
    void beginFrame(int frameSlot, DeletionQueue* dq = nullptr)
    {
        currentSlot = frameSlot;
        slots[frameSlot].writeHead = 0;
        if (dq) deletionQueue = dq;
    }

    // Write vertex data. Returns the byte offset for vkCmdBindVertexBuffers.
    // If the slot's buffer is too small, it doubles automatically.
    // Never fails. Never wraps. Never corrupts.
    VkDeviceSize write(const void* data, VkDeviceSize byteCount)
    {
        auto& slot = slots[currentSlot];

        if (slot.writeHead + byteCount > slot.capacity)
            grow(slot, std::max(slot.capacity * 2, slot.writeHead + byteCount));

        VkDeviceSize offset = slot.writeHead;
        memcpy(static_cast<char*>(slot.mapped) + offset, data, static_cast<size_t>(byteCount));
        slot.writeHead += byteCount;
        return offset;
    }

    // Call at the end of each frame. No-op — kept for API consistency.
    void endFrame() {}

    VkBuffer getBuffer() const { return slots[currentSlot].buffer.getBuffer(); }
    bool isValid() const { return slots[0].buffer.isValid() && slots[0].mapped != nullptr; }

    void destroy()
    {
        for (auto& slot : slots)
        {
            slot.buffer.destroy();
            slot.mapped = nullptr;
            slot.capacity = 0;
            slot.writeHead = 0;
        }
    }

private:
    void grow(Slot& slot, VkDeviceSize newCapacity)
    {
        // Earlier draws this frame reference the old buffer — retire it
        // to the DeletionQueue so it stays alive until the fence signals.
        if (deletionQueue)
            deletionQueue->retire(std::move(slot.buffer));

        slot.capacity = newCapacity;
        slot.buffer.create({ physDevice, device, newCapacity,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
        slot.mapped = slot.buffer.map();

        // Reset write head: data from before the grow is in the retired buffer.
        // New draws go into the new buffer from offset 0.
        slot.writeHead = 0;

        DBG("jvk::VertexRingBuffer: Slot grew to "
            << (newCapacity / (1024 * 1024)) << " MB");
    }
};

} // jvk::core
