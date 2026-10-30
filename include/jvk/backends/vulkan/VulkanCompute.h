/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanCompute.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::vulkan
{

// =============================================================================
// Compute on Vulkan (COMPUTE.md)
//
// Buffers and kernels are the backend's (VulkanBackend.h: BufferSlot,
// KernelSlot, made in VulkanCompute.cpp). A ComputeStream runs packets on the
// device's compute queue (Device::computeQueue: its own where the GPU has
// one), each in its own command buffer with its own descriptor pools,
// staging and timestamp queries; a frame runs its compute passes in its own
// command buffer (VulkanSurface.cpp). Both bind and run a dispatch the same
// way: DispatchRecorder.
// =============================================================================

// Binds and runs one gpu::Dispatch: its set (from `allocate`, a stream's pools
// or a frame slot's) written with its buffers, its push constants from
// `constants`. Fixed scratch: records allocate nothing.
class DispatchRecorder
{
public:
    // False when it was skipped: a dead kernel or buffer, a range outside its
    // buffer, a set that could not be had (each logged once).
    template <typename Allocate>
    bool record(Backend& backend, VkCommandBuffer cmd, const gpu::Dispatch& d,
                std::span<const std::byte> constants, Allocate&& allocate)
    {
        const KernelSlot* k = backend.kernel(d.kernel);
        if (k == nullptr || k->pipeline == VK_NULL_HANDLE) return skipped("a kernel that no longer exists");
        if (d.bindingCount < k->bindingCount) return skipped("fewer bindings than its kernel declares");

        VkDescriptorSet set = VK_NULL_HANDLE;
        if (k->bindingCount > 0)
        {
            uint32_t written = 0;
            for (uint32_t b = 0; b < k->bindingCount; ++b)
            {
                if (k->types[b] == VK_DESCRIPTOR_TYPE_MAX_ENUM) continue;   // declared by none
                const auto& r = d.bindings[b];
                const BufferSlot* buffer = backend.buffer(r.buffer);
                if (buffer == nullptr) return skipped("a buffer that no longer exists");
                if (r.size == 0 || r.offset + r.size > buffer->size) return skipped("a range outside its buffer");
                infos_[written] = { buffer->buffer, r.offset, r.size };
                auto& w = writes_[written];
                w = {};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstBinding      = b;
                w.descriptorCount = 1;
                w.descriptorType  = k->types[b];
                w.pBufferInfo     = &infos_[written];
                ++written;
            }
            set = allocate(*k);
            if (set == VK_NULL_HANDLE) return skipped("no descriptor set");
            for (uint32_t i = 0; i < written; ++i) writes_[i].dstSet = set;
            vkUpdateDescriptorSets(backend.device().device(), written, writes_.data(), 0, nullptr);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
        if (set != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &set, 0, nullptr);
        if (k->constantsBytes > 0 && !d.constants.isEmpty()
            && uint64_t(d.constants.offset) + d.constants.size <= constants.size())
        {
            const uint32_t size = std::min<uint32_t>(d.constants.size, k->constantsBytes) & ~3u;
            if (size > 0)
                vkCmdPushConstants(cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, constants.data() + d.constants.offset);
        }
        vkCmdDispatch(cmd, std::max(1u, d.groups[0]), std::max(1u, d.groups[1]), std::max(1u, d.groups[2]));
        return true;
    }

private:
    static bool skipped(const char* why)
    {
        static std::atomic<bool> logged { false };
        if (!logged.exchange(true)) diag::log(juce::String("vulkan: a dispatch was skipped: ") + why);
        return false;
    }

    std::array<VkWriteDescriptorSet, gpu::kMaxKernelBindings>   writes_ {};
    std::array<VkDescriptorBufferInfo, gpu::kMaxKernelBindings> infos_ {};
};

// ---- A stream ---------------------------------------------------------------------

class ComputeStream final : public gpu::ComputeStream
{
public:
    explicit ComputeStream(std::shared_ptr<Backend> backend);
    ~ComputeStream() override;

    bool valid() const { return cmd_ != VK_NULL_HANDLE && fence_ != VK_NULL_HANDLE; }

    bool   submit(const gpu::ComputePacket& packet) override;
    Status wait(double seconds) override;
    std::span<const double> timestamps() const override { return times_; }

private:
    static constexpr uint32_t kQueries = 64;   // timestamps a submission, the first its start

    void            finish(bool delivered);
    VkDescriptorSet allocate(const KernelSlot& kernel);
    bool            ensure(HostBuffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool readback);

    std::shared_ptr<Backend> backend_;
    VkDevice        vk_    = VK_NULL_HANDLE;
    VkCommandPool   pool_  = VK_NULL_HANDLE;
    VkCommandBuffer cmd_   = VK_NULL_HANDLE;
    VkFence         fence_ = VK_NULL_HANDLE;
    VkQueryPool     queries_ = VK_NULL_HANDLE;
    uint32_t        queryCount_ = 0;
    uint64_t        serial_ = 0;     // the submission in flight; 0: none
    bool            lost_ = false;   // its submit failed: the next wait says Lost

    struct DescriptorPool { VkDescriptorPool pool; uint32_t sets, storage, uniform; };
    std::vector<DescriptorPool> pools_;
    size_t                      poolInUse_ = 0;

    HostBuffer upload_, readback_;   // grown, never shrunk
    struct Download { VkDeviceSize offset; size_t bytes; void* to; };
    std::vector<Download> downloads_;
    std::vector<double>   times_;
    DispatchRecorder      dispatcher_;
};

} // namespace jvk::vulkan
