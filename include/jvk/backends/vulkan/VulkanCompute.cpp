/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanCompute.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::vulkan
{

namespace
{
constexpr uint32_t kPoolSets = 256, kPoolStorage = 2048, kPoolUniform = 256;   // a stream's descriptor pool

void logComputeOnce(std::atomic<bool>& flag, const juce::String& line)
{
    if (!flag.exchange(true)) diag::log(line);
}

// Everything the stream's commands wrote is the host's to read once its fence signals.
void hostBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb {};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}
} // namespace

// =============================================================================
// Caps
// =============================================================================

void Backend::initComputeCaps()
{
    auto& c = caps_.compute;
    if (device_->computeQueue() == VK_NULL_HANDLE) return;

    const VkPhysicalDevice physical = device_->physicalDevice();
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physical, &props);
    const auto& l = props.limits;
    deviceName_ = props.deviceName;

    c.supported     = true;
    c.frameDispatch = device_->graphicsComputes();
    c.ownQueue      = !device_->computeQueueShared();
    c.unifiedMemory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;   // MoltenVK reports Apple silicon so
    for (int i = 0; i < 3; ++i)
    {
        c.maxGroupSize[i] = l.maxComputeWorkGroupSize[i];
        c.maxGroups[i]    = l.maxComputeWorkGroupCount[i];
    }
    c.maxGroupThreads = l.maxComputeWorkGroupInvocations;
    c.sharedBytes     = l.maxComputeSharedMemorySize;
    c.constantsBytes  = l.maxPushConstantsSize;
    c.maxBindings     = std::min<uint32_t>(gpu::kMaxKernelBindings, l.maxPerStageDescriptorStorageBuffers);
    c.maxBufferBytes  = l.maxStorageBufferRange;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    const uint32_t bits = device_->computeFamily() < familyCount ? families[device_->computeFamily()].timestampValidBits : 0;
    if (bits > 0 && l.timestampPeriod > 0.0f)
    {
        c.timestampNs  = l.timestampPeriod;
        timestampMask_ = bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    }

    if (device_->apiVersion() >= VK_API_VERSION_1_1 && vkGetPhysicalDeviceProperties2 != nullptr)
    {
        VkPhysicalDeviceSubgroupProperties subgroup {};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &props2);
        c.subgroupSize = subgroup.subgroupSize;
    }
    c.deviceName = deviceName_.c_str();
}

// =============================================================================
// Buffers
// =============================================================================

gpu::BufferId Backend::createBuffer(const gpu::BufferDesc& desc)
{
    if (!usable() || !caps_.compute.supported || desc.size == 0) return {};

    VkBufferCreateInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = desc.size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // Frames (the graphics family) and streams (the compute family) both use
    // it as it is: no ownership transfers.
    const uint32_t families[] = { device_->graphicsFamily(), device_->computeFamily() };
    if (families[0] != families[1])
    {
        bi.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2;
        bi.pQueueFamilyIndices   = families;
    }
    else
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai {};
    switch (desc.memory)
    {
        case gpu::BufferMemory::Device:
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        case gpu::BufferMemory::Upload:
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case gpu::BufferMemory::Readback:
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case gpu::BufferMemory::Shared:
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    BufferSlot b;
    VmaAllocationInfo info {};
    if (vmaCreateBuffer(device_->allocator(), &bi, &ai, &b.buffer, &b.allocation, &info) != VK_SUCCESS)
    {
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, "vulkan: a buffer could not be created (" + juce::String((juce::int64) desc.size) + " bytes)");
        return {};
    }
    b.mapped = info.pMappedData;
    b.size   = desc.size;
    b.memory = desc.memory;

    std::lock_guard<std::mutex> lk(tableMutex_);
    BufferSlot* slot = nullptr;
    const uint32_t handle = buffers_.allocate(slot);
    if (handle == 0)
    {
        vmaDestroyBuffer(device_->allocator(), b.buffer, b.allocation);
        return {};
    }
    *slot = b;
    return { handle };
}

void Backend::destroyBuffer(gpu::BufferId buffer)
{
    if (!buffer) return;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (buffers_.resolve(buffer.value) == nullptr) return;
        // Every submission started before now, frame or stream, may use it.
        retired_.push_back({ RetiredKind::Buffer, buffer.value, startedSerial_.load(std::memory_order_acquire) });
    }
    tryCollect();
}

void Backend::freeBuffer(uint32_t handle)
{
    BufferSlot* b = buffers_.resolve(handle);
    if (b == nullptr) return;
    if (b->buffer != VK_NULL_HANDLE) vmaDestroyBuffer(device_->allocator(), b->buffer, b->allocation);
    buffers_.release(handle);
}

void* Backend::bufferData(gpu::BufferId buffer)
{
    const BufferSlot* b = buffers_.resolve(buffer.value);
    return b != nullptr ? b->mapped : nullptr;
}

void Backend::flushBuffer(gpu::BufferId buffer, uint64_t offset, uint64_t size)
{
    const BufferSlot* b = buffers_.resolve(buffer.value);
    if (b != nullptr && b->mapped != nullptr)
        vmaFlushAllocation(device_->allocator(), b->allocation, offset, size);   // a no-op on coherent memory
}

void Backend::invalidateBuffer(gpu::BufferId buffer, uint64_t offset, uint64_t size)
{
    const BufferSlot* b = buffers_.resolve(buffer.value);
    if (b != nullptr && b->mapped != nullptr)
        vmaInvalidateAllocation(device_->allocator(), b->allocation, offset, size);
}

// =============================================================================
// Kernels
// =============================================================================

gpu::KernelId Backend::createKernel(const gpu::KernelDesc& desc)
{
    const juce::String name = desc.name != nullptr ? desc.name : "?";
    auto refused = [&name](const juce::String& why) -> gpu::KernelId
    {
        diag::log("vulkan: kernel " + name + ": " + why);
        return {};
    };
    if (!usable() || !caps_.compute.supported) return {};
    if (desc.format != gpu::ShaderFormat::SPIRV || desc.code.empty() || desc.code.size() % 4 != 0)
        return refused("not SPIR-V");
    VkDevice d = device_->device();

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(device_->physicalDevice(), &props);

    KernelSlot k;
    std::fill(std::begin(k.types), std::end(k.types), VK_DESCRIPTOR_TYPE_MAX_ENUM);
    std::array<VkDescriptorSetLayoutBinding, gpu::kMaxKernelBindings> layoutBindings {};
    uint32_t layoutCount = 0;
    for (const auto& b : desc.bindings)
    {
        if (b.binding >= gpu::kMaxKernelBindings) return refused("binds past binding " + juce::String(gpu::kMaxKernelBindings - 1));
        if (b.kind != gpu::BindingKind::StorageBuffer && b.kind != gpu::BindingKind::UniformBuffer)
            return refused("binds something other than a buffer");
        const VkDescriptorType type = b.kind == gpu::BindingKind::UniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                                 : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        k.types[b.binding] = type;
        k.bindingCount = std::max<uint32_t>(k.bindingCount, uint32_t(b.binding) + 1);
        (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? k.storageCount : k.uniformCount)++;
        layoutBindings[layoutCount++] = { b.binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    if (k.storageCount > props.limits.maxPerStageDescriptorStorageBuffers
        || k.uniformCount > props.limits.maxPerStageDescriptorUniformBuffers)
        return refused("binds more buffers than the GPU takes (" + juce::String(props.limits.maxPerStageDescriptorStorageBuffers)
                       + " storage, " + juce::String(props.limits.maxPerStageDescriptorUniformBuffers) + " uniform)");
    k.constantsBytes = (uint32_t(desc.constantsBytes) + 3u) & ~3u;
    if (k.constantsBytes > props.limits.maxPushConstantsSize)
        return refused("has more push constants than the GPU takes");
    for (int i = 0; i < 3; ++i) k.groupSize[i] = desc.groupSize[i];

    auto destroy = [d](KernelSlot& s)
    {
        if (s.pipeline != VK_NULL_HANDLE)  vkDestroyPipeline(d, s.pipeline, nullptr);
        if (s.layout != VK_NULL_HANDLE)    vkDestroyPipelineLayout(d, s.layout, nullptr);
        if (s.setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, s.setLayout, nullptr);
    };

    VkDescriptorSetLayoutCreateInfo sl {};
    sl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sl.bindingCount = layoutCount;
    sl.pBindings    = layoutBindings.data();
    if (vkCreateDescriptorSetLayout(d, &sl, nullptr, &k.setLayout) != VK_SUCCESS)
        return refused("its descriptor set layout could not be made");

    VkPushConstantRange push { VK_SHADER_STAGE_COMPUTE_BIT, 0, k.constantsBytes };
    VkPipelineLayoutCreateInfo pl {};
    pl.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount         = 1;
    pl.pSetLayouts            = &k.setLayout;
    pl.pushConstantRangeCount = k.constantsBytes > 0 ? 1u : 0u;
    pl.pPushConstantRanges    = &push;
    if (vkCreatePipelineLayout(d, &pl, nullptr, &k.layout) != VK_SUCCESS)
    {
        destroy(k);
        return refused("its pipeline layout could not be made");
    }

    // Copied: a std::byte span promises no uint32_t alignment.
    std::vector<uint32_t> words(desc.code.size() / 4);
    std::memcpy(words.data(), desc.code.data(), desc.code.size());
    VkShaderModuleCreateInfo sm {};
    sm.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = desc.code.size();
    sm.pCode    = words.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(d, &sm, nullptr, &shader) != VK_SUCCESS)
    {
        destroy(k);
        return refused("the driver refused its SPIR-V");
    }

    std::vector<VkSpecializationMapEntry> entries;
    std::vector<uint32_t> values;
    for (const auto& c : desc.constants)
    {
        entries.push_back({ c.id, static_cast<uint32_t>(values.size() * 4), 4 });   // a VkBool32 is 4 bytes too
        values.push_back(c.bits);
    }
    VkSpecializationInfo spec { static_cast<uint32_t>(entries.size()), entries.data(), values.size() * 4, values.data() };

    VkComputePipelineCreateInfo ci {};
    ci.sType                     = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module              = shader;
    ci.stage.pName               = desc.entry != nullptr ? desc.entry : "main";
    ci.stage.pSpecializationInfo = entries.empty() ? nullptr : &spec;
    ci.layout                    = k.layout;
    const VkResult made = vkCreateComputePipelines(d, device_->pipelineCache(), 1, &ci, nullptr, &k.pipeline);
    vkDestroyShaderModule(d, shader, nullptr);
    if (made != VK_SUCCESS)
    {
        k.pipeline = VK_NULL_HANDLE;
        destroy(k);
        return refused("the driver could not build it");
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    KernelSlot* slot = nullptr;
    const uint32_t handle = kernels_.allocate(slot);
    if (handle == 0)
    {
        destroy(k);
        return {};
    }
    *slot = k;
    return { handle };
}

void Backend::destroyKernel(gpu::KernelId kernel)
{
    if (!kernel) return;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (kernels_.resolve(kernel.value) == nullptr) return;
        retired_.push_back({ RetiredKind::Kernel, kernel.value, startedSerial_.load(std::memory_order_acquire) });
    }
    tryCollect();
}

void Backend::freeKernel(uint32_t handle)
{
    KernelSlot* k = kernels_.resolve(handle);
    if (k == nullptr) return;
    VkDevice d = device_->device();
    if (k->pipeline != VK_NULL_HANDLE)  vkDestroyPipeline(d, k->pipeline, nullptr);
    if (k->layout != VK_NULL_HANDLE)    vkDestroyPipelineLayout(d, k->layout, nullptr);
    if (k->setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, k->setLayout, nullptr);
    kernels_.release(handle);
}

// =============================================================================
// Submissions
// =============================================================================

uint64_t Backend::beginComputeSubmission()
{
    // Taken and listed in one step, so collect() never sees a begun serial
    // that isn't listed yet.
    std::lock_guard<std::mutex> lk(computeMutex_);
    const uint64_t serial = startedSerial_.fetch_add(1, std::memory_order_acq_rel) + 1;
    computeActive_.push_back(serial);
    return serial;
}

void Backend::endComputeSubmission(uint64_t serial)
{
    {
        std::lock_guard<std::mutex> lk(computeMutex_);
        std::erase(computeActive_, serial);
    }
    tryCollect();
}

void Backend::tryCollect()
{
    auto& lock = device_->queueLock();
    if (lock.tryEnter())
    {
        collect();
        lock.exit();
    }
}

std::unique_ptr<gpu::ComputeStream> Backend::createComputeStream()
{
    if (!usable() || !caps_.compute.supported) return nullptr;
    auto s = std::make_unique<ComputeStream>(shared_from_this());
    if (!s->valid()) return nullptr;
    return s;
}

// =============================================================================
// ComputeStream
// =============================================================================

ComputeStream::ComputeStream(std::shared_ptr<Backend> backend)
    : backend_(std::move(backend)), vk_(backend_->device().device())
{
    auto& device = backend_->device();
    VkCommandPoolCreateInfo pi {};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = device.computeFamily();
    pi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(vk_, &pi, nullptr, &pool_) != VK_SUCCESS)
        pool_ = VK_NULL_HANDLE;

    VkCommandBufferAllocateInfo ai {};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (pool_ == VK_NULL_HANDLE || vkAllocateCommandBuffers(vk_, &ai, &cmd_) != VK_SUCCESS)
        cmd_ = VK_NULL_HANDLE;

    VkFenceCreateInfo fi {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(vk_, &fi, nullptr, &fence_) != VK_SUCCESS)
        fence_ = VK_NULL_HANDLE;

    if (backend_->caps().compute.timestampNs > 0.0)
    {
        VkQueryPoolCreateInfo qi {};
        qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kQueries;
        if (vkCreateQueryPool(vk_, &qi, nullptr, &queries_) != VK_SUCCESS)
            queries_ = VK_NULL_HANDLE;
    }
}

ComputeStream::~ComputeStream()
{
    if (serial_ != 0)
        wait(-1.0);
    if (serial_ != 0)   // never answered: nothing it holds may go while the GPU can use it
    {
        backend_->device().waitIdle();
        backend_->endComputeSubmission(serial_);
        serial_ = 0;
    }
    auto* allocator = backend_->device().allocator();
    destroyHostBuffer(allocator, upload_);
    destroyHostBuffer(allocator, readback_);
    for (auto& p : pools_) vkDestroyDescriptorPool(vk_, p.pool, nullptr);
    if (queries_ != VK_NULL_HANDLE) vkDestroyQueryPool(vk_, queries_, nullptr);
    if (fence_ != VK_NULL_HANDLE)   vkDestroyFence(vk_, fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE)    vkDestroyCommandPool(vk_, pool_, nullptr);   // and its command buffer
}

bool ComputeStream::ensure(HostBuffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool readback)
{
    // Grown, never shrunk: a stream's steady state allocates nothing. Nothing
    // is in flight here (submit runs after the last wait), so the old one goes now.
    if (size == 0 || (buffer && buffer.size >= size)) return true;
    auto* allocator = backend_->device().allocator();
    const VkDeviceSize grown = std::max<VkDeviceSize>({ size, buffer.size * 2, VkDeviceSize(1u << 20) });
    destroyHostBuffer(allocator, buffer);
    buffer = createHostBuffer(allocator, grown, usage, readback);
    return static_cast<bool>(buffer);
}

VkDescriptorSet ComputeStream::allocate(const KernelSlot& kernel)
{
    for (;; ++poolInUse_)
    {
        if (poolInUse_ == pools_.size())
        {
            const VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kPoolStorage },
                                                   { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kPoolUniform } };
            VkDescriptorPoolCreateInfo pi {};
            pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets       = kPoolSets;
            pi.poolSizeCount = 2;
            pi.pPoolSizes    = sizes;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(vk_, &pi, nullptr, &pool) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            pools_.push_back({ pool, 0, 0, 0 });
        }
        // Counted here, so a pool is never asked for more than it has (MoltenVK has crashed on that).
        auto& p = pools_[poolInUse_];
        if (p.sets + 1 > kPoolSets || p.storage + kernel.storageCount > kPoolStorage || p.uniform + kernel.uniformCount > kPoolUniform)
        {
            if (p.sets == 0) return VK_NULL_HANDLE;   // more than a fresh pool holds
            continue;
        }
        VkDescriptorSetAllocateInfo ai {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = p.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &kernel.setLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vk_, &ai, &set) != VK_SUCCESS)
        {
            if (p.sets == 0) return VK_NULL_HANDLE;   // a fresh pool refused it: no pool will take it
            continue;
        }
        p.sets    += 1;
        p.storage += kernel.storageCount;
        p.uniform += kernel.uniformCount;
        return set;
    }
}

bool ComputeStream::submit(const gpu::ComputePacket& packet)
{
    jassert(serial_ == 0);   // the last submission was waited for
    if (serial_ != 0 || !valid() || backend_->isLost())
    {
        lost_ = backend_->isLost();
        return false;
    }

    // From here on whatever is destroyed waits for this submission: the
    // handles below stay good while it is recorded and runs.
    serial_ = backend_->beginComputeSubmission();

    // Staging: the packet's upload bytes as they are, and a slot per download.
    VkDeviceSize downloadBytes = 0;
    for (const auto& c : packet.commands)
        if (c.op == gpu::ComputeOp::Download)
            downloadBytes = ((downloadBytes + 15) & ~VkDeviceSize(15)) + c.src.size;
    if (!ensure(upload_, packet.uploadData.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false)
        || !ensure(readback_, downloadBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true))
    {
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, "vulkan: a compute stream's staging could not be had");
        backend_->endComputeSubmission(std::exchange(serial_, 0));
        return false;
    }
    if (!packet.uploadData.empty())
        std::memcpy(upload_.mapped, packet.uploadData.data(), packet.uploadData.size());

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &bi);
    queryCount_ = 0;
    if (queries_ != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd_, queries_, 0, kQueries);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, queryCount_++);   // the start
    }

    auto rangeOk = [this](const gpu::BufferBinding& r) -> const BufferSlot*
    {
        const BufferSlot* b = backend_->buffer(r.buffer);
        return b != nullptr && r.size > 0 && r.offset + r.size <= b->size ? b : nullptr;
    };
    VkDeviceSize readbackHead = 0;
    downloads_.clear();
    for (const auto& c : packet.commands)
    {
        switch (c.op)
        {
            case gpu::ComputeOp::Dispatch:
                if (c.dispatch < packet.dispatches.size())
                    dispatcher_.record(*backend_, cmd_, packet.dispatches[c.dispatch], packet.constantData,
                                       [this](const KernelSlot& k) { return allocate(k); });
                break;

            case gpu::ComputeOp::Barrier:
            {
                VkMemoryBarrier mb {};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT
                                 | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                vkCmdPipelineBarrier(cmd_, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
                break;
            }

            case gpu::ComputeOp::Fill:
                if (const auto* b = rangeOk(c.dst); b != nullptr && c.dst.offset % 4 == 0 && c.dst.size % 4 == 0)
                    vkCmdFillBuffer(cmd_, b->buffer, c.dst.offset, c.dst.size, c.word);
                break;

            case gpu::ComputeOp::Copy:
            {
                const auto* from = rangeOk(c.src);
                const auto* to   = rangeOk({ c.dst.buffer, c.dst.offset, c.src.size });
                if (from != nullptr && to != nullptr)
                {
                    const VkBufferCopy copy { c.src.offset, c.dst.offset, c.src.size };
                    vkCmdCopyBuffer(cmd_, from->buffer, to->buffer, 1, &copy);
                }
                break;
            }

            case gpu::ComputeOp::Upload:
            {
                const auto* to = rangeOk({ c.dst.buffer, c.dst.offset, c.data.size });
                if (to != nullptr && uint64_t(c.data.offset) + c.data.size <= packet.uploadData.size())
                {
                    const VkBufferCopy copy { c.data.offset, c.dst.offset, c.data.size };
                    vkCmdCopyBuffer(cmd_, upload_.buffer, to->buffer, 1, &copy);
                }
                break;
            }

            case gpu::ComputeOp::Download:
            {
                readbackHead = (readbackHead + 15) & ~VkDeviceSize(15);
                if (const auto* from = rangeOk(c.src); from != nullptr && c.to != nullptr)
                {
                    const VkBufferCopy copy { c.src.offset, readbackHead, c.src.size };
                    vkCmdCopyBuffer(cmd_, from->buffer, readback_.buffer, 1, &copy);
                    downloads_.push_back({ readbackHead, static_cast<size_t>(c.src.size), c.to });
                }
                readbackHead += c.src.size;
                break;
            }

            case gpu::ComputeOp::Timestamp:
                if (queries_ != VK_NULL_HANDLE && queryCount_ < kQueries)
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, queryCount_++);
                break;
        }
    }
    hostBarrier(cmd_);
    vkEndCommandBuffer(cmd_);

    VkResult r = VK_ERROR_DEVICE_LOST;
    auto& device = backend_->device();
    if (!device.isLost())
    {
        VkSubmitInfo si {};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd_;
        const juce::ScopedLock queueSync(device.computeQueueLock());
        r = vkQueueSubmit(device.computeQueue(), 1, &si, fence_);
    }
    if (r == VK_SUCCESS)
        return true;
    if (r == VK_ERROR_DEVICE_LOST)
        device.markLost();
    lost_ = true;
    finish(false);
    return false;
}

gpu::ComputeStream::Status ComputeStream::wait(double seconds)
{
    if (serial_ == 0)
        return std::exchange(lost_, false) ? Status::Lost : Status::Done;
    const uint64_t ns = seconds < 0.0 ? UINT64_MAX : static_cast<uint64_t>(seconds * 1.0e9);
    const VkResult r = vkWaitForFences(vk_, 1, &fence_, VK_TRUE, ns);
    if (r == VK_TIMEOUT)
        return Status::Pending;
    if (r != VK_SUCCESS)
    {
        backend_->device().markLost();
        // The GPU may still hold the command buffer: everything waits for it.
        backend_->device().waitIdle();
        finish(false);
        return Status::Lost;
    }
    finish(true);
    return Status::Done;
}

void ComputeStream::finish(bool delivered)
{
    times_.clear();
    if (delivered)
    {
        for (const auto& d : downloads_)
            std::memcpy(d.to, static_cast<const std::byte*>(readback_.mapped) + d.offset, d.bytes);
        if (queries_ != VK_NULL_HANDLE && queryCount_ > 1)
        {
            std::array<uint64_t, kQueries> ticks {};
            if (vkGetQueryPoolResults(vk_, queries_, 0, queryCount_, queryCount_ * sizeof(uint64_t), ticks.data(),
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            {
                const double ns = backend_->caps().compute.timestampNs;
                const uint64_t mask = backend_->timestampMask();
                for (uint32_t i = 1; i < queryCount_; ++i)
                    times_.push_back(static_cast<double>((ticks[i] - ticks[0]) & mask) * ns * 1.0e-6);
            }
        }
    }
    downloads_.clear();
    vkResetFences(vk_, 1, &fence_);
    vkResetCommandPool(vk_, pool_, 0);
    for (auto& p : pools_)
    {
        vkResetDescriptorPool(vk_, p.pool, 0);
        p.sets = p.storage = p.uniform = 0;
    }
    poolInUse_ = 0;
    queryCount_ = 0;
    if (serial_ != 0)
        backend_->endComputeSubmission(std::exchange(serial_, 0));
}

} // namespace jvk::vulkan
