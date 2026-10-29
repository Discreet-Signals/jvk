/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanSurface.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::vulkan
{

// =============================================================================
// Surfaces and the packet executor
//
// A surface owns what one frame in flight needs (FrameSlot: its command
// buffer, fence, stream buffers and descriptor pools) and runs a
// gpu::FramePacket through PacketExecutor, under the submit lock (see
// VulkanBackend.h). Everything per-slot is reused once the slot's fence has
// signalled, which is the only wait a frame makes.
// =============================================================================

// What a descriptor set was written for: reused within a frame by every draw
// that binds the same things through the same program set.
struct DescriptorKey
{
    uint32_t            program = 0;
    uint8_t             set = 0;
    uint8_t             textureCount = 0, storageCount = 0;
    gpu::TextureBinding textures[gpu::kMaxDrawTextures] {};
    gpu::Range          storage[gpu::kMaxDrawStorage] {};

    bool operator==(const DescriptorKey& o) const
    {
        if (program != o.program || set != o.set || textureCount != o.textureCount || storageCount != o.storageCount)
            return false;
        for (uint32_t i = 0; i < textureCount; ++i) if (!(textures[i] == o.textures[i])) return false;
        for (uint32_t i = 0; i < storageCount; ++i) if (!(storage[i] == o.storage[i])) return false;
        return true;
    }
};

struct FrameSlot
{
    VkCommandBuffer cmd            = VK_NULL_HANDLE;
    VkFence         fence          = VK_NULL_HANDLE;
    VkSemaphore     imageAvailable = VK_NULL_HANDLE;   // swapchain surfaces
    HostBuffer      vertices, storage;
    std::vector<VkDescriptorPool> pools;
    size_t          pool = 0;                          // the one allocating now
    struct SetEntry { DescriptorKey key; VkDescriptorSet set; };
    std::vector<SetEntry> sets;                        // this frame's, for reuse
};

class PacketExecutor
{
public:
    explicit PacketExecutor(Backend& backend) : backend_(backend) {}

    // The slot's fence has signalled: its descriptor sets are free again.
    void beginSlot(FrameSlot& slot);
    // Records the packet's passes into `cmd` (begun, under the submit lock).
    void record(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet);
    // Frees the slot's buffers and pools (its GPU work is done).
    void destroySlot(FrameSlot& slot);

private:
    struct Bound
    {
        VkPipeline       pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout   = VK_NULL_HANDLE;
        VkDescriptorSet  sets[8] {};
        gpu::Range       constants { ~0u, 0 };
        gpu::IRect       scissor { -1, -1, -1, -1 };
        int              stencilRef = -1;
        bool             vertexBuffer = false;
    };

    bool ensure(HostBuffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage);
    void recordPass(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet, const gpu::Pass& pass);
    void recordDraw(VkCommandBuffer cmd, FrameSlot& slot, const gpu::FramePacket& packet,
                    const gpu::Draw& draw, const gpu::IRect& area);
    VkDescriptorSet descriptorSet(FrameSlot& slot, const ProgramSlot& program, uint32_t programHandle,
                                  uint32_t set, const gpu::Draw& draw);
    VkDescriptorSet allocateSet(FrameSlot& slot, VkDescriptorSetLayout layout);

    Backend& backend_;
    Bound    bound_;
    Backend::BarrierBatch batch_;
    std::vector<VkWriteDescriptorSet>   writes_;
    std::vector<VkDescriptorImageInfo>  imageInfos_;
    std::vector<VkDescriptorBufferInfo> bufferInfos_;
};

// ---- A window's swapchain -------------------------------------------------------

class SwapchainSurface final : public gpu::Surface
{
public:
    // Takes ownership of `surface`. `nativeWindow`: the HWND on Windows (for
    // DPI pinning), else null. `sizeHint`: physical pixels, used only where
    // the platform has no size of its own.
    SwapchainSurface(std::shared_ptr<Backend> backend, VkSurfaceKHR surface,
                     gpu::Extent sizeHint, void* nativeWindow);
    ~SwapchainSurface() override;

    bool        beginFrame(gpu::FrameInfo& info) override;
    void        submit(const gpu::FramePacket& packet) override;
    gpu::Extent extent() const override { return { width_, height_ }; }
    gpu::Extent preferredExtent() const override;
    bool        canBegin() const override;
    void        revive() override;
    void        resize(gpu::Extent extent) override;

private:
    static constexpr uint32_t kFrames = 2;

    void createSwapchain();
    void destroySwapchain();   // the GPU must be idle
    void rebuild(uint32_t width, uint32_t height);   // under the submit lock

    std::shared_ptr<Backend> backend_;
    VkSurfaceKHR             surface_      = VK_NULL_HANDLE;
    void*                    nativeWindow_ = nullptr;
    VkSwapchainKHR           swapchain_    = VK_NULL_HANDLE;
    bool                     surfaceLost_  = false;   // VK_ERROR_SURFACE_LOST_KHR: sticky
    std::vector<VkImage>        swapImages_;
    std::vector<gpu::TextureId> swapTextures_;
    std::vector<VkSemaphore>    renderFinished_;   // one per swap image
    uint32_t width_ = 0, height_ = 0;

    VkCommandPool  pool_ = VK_NULL_HANDLE;
    FrameSlot      slots_[kFrames];
    PacketExecutor exec_;
    Backend::BarrierBatch present_;   // the swap image's last transition (capacity kept)
    uint32_t       current_       = 0;
    uint32_t       imageIndex_    = 0;
    bool           frameBegun_    = false;
    int            fenceTimeouts_ = 0;
    static constexpr int kMaxFenceTimeouts = 5;
};

// ---- An offscreen texture ---------------------------------------------------------

class OffscreenSurface final : public gpu::Surface
{
public:
    OffscreenSurface(std::shared_ptr<Backend> backend, gpu::TextureId target);
    ~OffscreenSurface() override;

    bool valid() const { return slot_.cmd != VK_NULL_HANDLE && slot_.fence != VK_NULL_HANDLE; }

    bool        beginFrame(gpu::FrameInfo& info) override;
    void        submit(const gpu::FramePacket& packet) override;
    gpu::Extent extent() const override;

private:
    std::shared_ptr<Backend> backend_;
    gpu::TextureId           target_;
    VkCommandPool            pool_ = VK_NULL_HANDLE;
    FrameSlot                slot_;
    PacketExecutor           exec_;
    Backend::BarrierBatch    release_;   // the target's last transition (capacity kept)
    bool                     frameBegun_ = false;
};

} // namespace jvk::vulkan
