namespace jvk::compute {

namespace {
constexpr uint32_t kQueries = 64;                       // timestamps a submit
constexpr uint32_t kPoolSets = 256, kPoolStorage = 2048, kPoolUniform = 256;   // a descriptor pool's
constexpr VkDeviceSize kStagingBlock = 4 * 1024 * 1024; // a stream's staging grows by this, or by what one copy needs
constexpr uint32_t kMaxBindings = 16;                   // ranges a dispatch binds
}

// =============================================================================
// Buffer
// =============================================================================

Buffer::~Buffer()
{
    if (buffer_ != VK_NULL_HANDLE)
        vmaDestroyBuffer(device_->allocator(), buffer_, allocation_);
}

Buffer::Buffer(Buffer&& other) noexcept
    : device_(std::move(other.device_)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr))
{
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other)
    {
        Buffer gone(std::move(*this));   // what this held goes at the end of the scope
        device_     = std::move(other.device_);
        buffer_     = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, nullptr);
        size_       = std::exchange(other.size_, 0);
        mapped_     = std::exchange(other.mapped_, nullptr);
    }
    return *this;
}

void Buffer::written(VkDeviceSize offset, VkDeviceSize bytes) const
{
    if (allocation_ != nullptr)
        vmaFlushAllocation(device_->allocator(), allocation_, offset, bytes);
}

void Buffer::toRead(VkDeviceSize offset, VkDeviceSize bytes) const
{
    if (allocation_ != nullptr)
        vmaInvalidateAllocation(device_->allocator(), allocation_, offset, bytes);
}

// =============================================================================
// Kernel
// =============================================================================

Kernel::~Kernel()
{
    if (device_ == nullptr)
        return;
    const VkDevice vk = device_->device();
    if (pipeline_ != VK_NULL_HANDLE)  vkDestroyPipeline(vk, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)    vkDestroyPipelineLayout(vk, layout_, nullptr);
    if (setLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk, setLayout_, nullptr);
}

Kernel::Kernel(Kernel&& other) noexcept
    : device_(std::move(other.device_)),
      name_(std::move(other.name_)),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)),
      layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
      setLayout_(std::exchange(other.setLayout_, VK_NULL_HANDLE)),
      types_(std::move(other.types_)),
      storageCount_(other.storageCount_),
      uniformCount_(other.uniformCount_),
      pushBytes_(other.pushBytes_),
      groupSize_(other.groupSize_)
{
}

Kernel& Kernel::operator=(Kernel&& other) noexcept
{
    if (this != &other)
    {
        Kernel gone(std::move(*this));
        device_       = std::move(other.device_);
        name_         = std::move(other.name_);
        pipeline_     = std::exchange(other.pipeline_, VK_NULL_HANDLE);
        layout_       = std::exchange(other.layout_, VK_NULL_HANDLE);
        setLayout_    = std::exchange(other.setLayout_, VK_NULL_HANDLE);
        types_        = std::move(other.types_);
        storageCount_ = other.storageCount_;
        uniformCount_ = other.uniformCount_;
        pushBytes_    = other.pushBytes_;
        groupSize_    = other.groupSize_;
    }
    return *this;
}

// =============================================================================
// Context
// =============================================================================

std::unique_ptr<Context> Context::create(std::shared_ptr<Device> device, juce::String* whyNot)
{
    auto no = [whyNot](const juce::String& reason) -> std::unique_ptr<Context>
    {
        if (whyNot != nullptr) *whyNot = reason;
        return nullptr;
    };
    if (device == nullptr)                         return no("no Vulkan device");
    if (! device->usable())                        return no(device->unavailableReason());
    if (device->isLost())                          return no("the GPU was lost");
    if (device->computeQueue() == VK_NULL_HANDLE)  return no("the GPU has no compute queue");
    return std::unique_ptr<Context>(new Context(std::move(device)));
}

Context::Context(std::shared_ptr<Device> device) : device_(std::move(device))
{
    const VkPhysicalDevice physical = device_->physicalDevice();
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physical, &props);
    const auto& l = props.limits;
    limits_.device = props.deviceName;
    limits_.apiVersion = device_->apiVersion();
    for (int i = 0; i < 3; ++i)
    {
        limits_.maxGroupSize[i] = l.maxComputeWorkGroupSize[i];
        limits_.maxGroups[i] = l.maxComputeWorkGroupCount[i];
    }
    limits_.maxGroupThreads = l.maxComputeWorkGroupInvocations;
    limits_.sharedBytes = l.maxComputeSharedMemorySize;
    limits_.pushBytes = l.maxPushConstantsSize;
    limits_.storageAlignment = std::max<VkDeviceSize>({ 4, l.minStorageBufferOffsetAlignment, l.minUniformBufferOffsetAlignment });
    limits_.unifiedMemory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;   // MoltenVK reports Apple silicon so
    limits_.ownQueue = ! device_->computeQueueShared();

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    const uint32_t bits = device_->computeFamily() < familyCount ? families[device_->computeFamily()].timestampValidBits : 0;
    if (bits > 0 && l.timestampPeriod > 0.0f)
    {
        limits_.timestampNs = l.timestampPeriod;
        timestampMask_ = bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    }

    if (limits_.apiVersion >= VK_API_VERSION_1_1 && vkGetPhysicalDeviceProperties2 != nullptr)
    {
        VkPhysicalDeviceSubgroupProperties subgroup {};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &props2);
        limits_.subgroupSize = subgroup.subgroupSize;
    }

    diag::log("compute context: " + limits_.device + ", Vulkan " + juce::String(VK_API_VERSION_MAJOR(limits_.apiVersion)) + "."
              + juce::String(VK_API_VERSION_MINOR(limits_.apiVersion)) + ", subgroups of " + juce::String(limits_.subgroupSize)
              + ", " + juce::String(limits_.sharedBytes / 1024) + " KB shared, " + (limits_.unifiedMemory ? "unified memory" : "own memory")
              + ", " + (limits_.ownQueue ? "own queue" : "graphics queue") + (limits_.timestampNs > 0.0 ? ", timestamps" : ", no timestamps"));
}

Context::~Context()
{
    jassert(streams_.load() == 0);   // every Stream goes before its Context
}

Buffer Context::buffer(VkDeviceSize bytes, Usage usage)
{
    Buffer b;
    VkBufferCreateInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = std::max<VkDeviceSize>(bytes, 4);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai {};
    switch (usage)
    {
        case Usage::Device:
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        case Usage::Upload:
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case Usage::Readback:
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case Usage::Shared:
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    VmaAllocationInfo info {};
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (vmaCreateBuffer(device_->allocator(), &bi, &ai, &handle, &allocation, &info) != VK_SUCCESS)
        return b;
    b.device_ = device_;
    b.buffer_ = handle;
    b.allocation_ = allocation;
    b.size_ = bi.size;
    b.mapped_ = info.pMappedData;
    return b;
}

Kernel Context::kernel(std::span<const uint32_t> spirv, const juce::String& name, std::initializer_list<Constant> constants)
{
    auto refused = [&name](const juce::String& why)
    {
        diag::log("compute kernel " + name + ": " + why);
        jassertfalse;
        return Kernel();
    };

    SpvReflectShaderModule module {};
    if (spvReflectCreateShaderModule(spirv.size_bytes(), spirv.data(), &module) != SPV_REFLECT_RESULT_SUCCESS)
        return refused("not SPIR-V");
    struct Reflected { SpvReflectShaderModule& m; ~Reflected() { spvReflectDestroyShaderModule(&m); } } reflected { module };
    if (module.shader_stage != SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT || module.entry_point_count == 0)
        return refused("not a compute shader");

    Kernel k;
    k.device_ = device_;
    k.name_ = name;
    k.groupSize_ = { module.entry_points[0].local_size.x, module.entry_points[0].local_size.y, module.entry_points[0].local_size.z };

    uint32_t count = 0;
    spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
    std::vector<SpvReflectDescriptorBinding*> found(count);
    spvReflectEnumerateDescriptorBindings(&module, &count, found.data());
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto* b : found)
    {
        const auto type = (VkDescriptorType) b->descriptor_type;
        if (b->set != 0)
            return refused("binds a set other than 0");
        if (type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            return refused("binds something other than a buffer");
        if (b->binding >= kMaxBindings)
            return refused("binds past binding " + juce::String((int) kMaxBindings - 1));
        if (b->binding >= k.types_.size())
            k.types_.resize(b->binding + 1, VK_DESCRIPTOR_TYPE_MAX_ENUM);
        k.types_[b->binding] = type;
        (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? k.storageCount_ : k.uniformCount_)++;
        bindings.push_back({ b->binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr });
    }

    count = 0;
    spvReflectEnumeratePushConstantBlocks(&module, &count, nullptr);
    if (count > 1)
        return refused("has more than one push-constant block");
    if (count == 1)
    {
        SpvReflectBlockVariable* block = nullptr;
        spvReflectEnumeratePushConstantBlocks(&module, &count, &block);
        k.pushBytes_ = (block->offset + block->size + 3u) & ~3u;
        if (k.pushBytes_ > limits_.pushBytes)
            return refused("has more push constants than the GPU takes (" + juce::String((int) limits_.pushBytes) + " bytes)");
    }

    const VkDevice vk = device_->device();
    VkDescriptorSetLayoutCreateInfo sl {};
    sl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sl.bindingCount = (uint32_t) bindings.size();
    sl.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(vk, &sl, nullptr, &k.setLayout_) != VK_SUCCESS)
        return refused("its descriptor set layout could not be made");

    VkPushConstantRange push { VK_SHADER_STAGE_COMPUTE_BIT, 0, k.pushBytes_ };
    VkPipelineLayoutCreateInfo pl {};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &k.setLayout_;
    pl.pushConstantRangeCount = k.pushBytes_ > 0 ? 1u : 0u;
    pl.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(vk, &pl, nullptr, &k.layout_) != VK_SUCCESS)
        return refused("its pipeline layout could not be made");

    VkShaderModuleCreateInfo sm {};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = spirv.size_bytes();
    sm.pCode = spirv.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(vk, &sm, nullptr, &shader) != VK_SUCCESS)
        return refused("the driver refused its SPIR-V");

    std::vector<VkSpecializationMapEntry> entries;
    std::vector<uint32_t> values;
    for (const auto& c : constants)
    {
        entries.push_back({ c.id, (uint32_t) (values.size() * 4), 4 });
        values.push_back(c.bits);
    }
    VkSpecializationInfo spec { (uint32_t) entries.size(), entries.data(), values.size() * 4, values.data() };

    VkComputePipelineCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shader;
    ci.stage.pName = module.entry_points[0].name;
    ci.stage.pSpecializationInfo = entries.empty() ? nullptr : &spec;
    ci.layout = k.layout_;
    const VkResult made = vkCreateComputePipelines(vk, device_->pipelineCache(), 1, &ci, nullptr, &k.pipeline_);
    vkDestroyShaderModule(vk, shader, nullptr);
    if (made != VK_SUCCESS)
    {
        k.pipeline_ = VK_NULL_HANDLE;
        return refused("the driver could not build it");
    }
    return k;
}

std::unique_ptr<Stream> Context::stream()
{
    return std::unique_ptr<Stream>(new Stream(*this));
}

// =============================================================================
// Stream
// =============================================================================

Stream::Stream(Context& context) : context_(context), device_(context.device_), vk_(context.device_->device())
{
    ++context_.streams_;

    VkCommandPoolCreateInfo pi {};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = device_->computeFamily();
    if (vkCreateCommandPool(vk_, &pi, nullptr, &pool_) != VK_SUCCESS)
        pool_ = VK_NULL_HANDLE;

    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (pool_ == VK_NULL_HANDLE || vkAllocateCommandBuffers(vk_, &ai, &cmd_) != VK_SUCCESS)
        cmd_ = VK_NULL_HANDLE;

    VkFenceCreateInfo fi {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(vk_, &fi, nullptr, &fence_) != VK_SUCCESS)
        fence_ = VK_NULL_HANDLE;

    if (context_.limits_.timestampNs > 0.0)
    {
        VkQueryPoolCreateInfo qi {};
        qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kQueries;
        if (vkCreateQueryPool(vk_, &qi, nullptr, &queries_) != VK_SUCCESS)
            queries_ = VK_NULL_HANDLE;
    }
    jassert(cmd_ != VK_NULL_HANDLE && fence_ != VK_NULL_HANDLE);
}

Stream::~Stream()
{
    if (state_ == State::submitted)
        wait();
    for (auto& p : descriptorPools_)
        vkDestroyDescriptorPool(vk_, p.pool, nullptr);
    if (queries_ != VK_NULL_HANDLE) vkDestroyQueryPool(vk_, queries_, nullptr);
    if (fence_ != VK_NULL_HANDLE)   vkDestroyFence(vk_, fence_, nullptr);
    if (pool_ != VK_NULL_HANDLE)    vkDestroyCommandPool(vk_, pool_, nullptr);   // and its command buffer
    --context_.streams_;
}

void Stream::begin()
{
    if (state_ == State::submitted)   // recording again before waiting: wait now
        wait();
    if (state_ != State::idle || cmd_ == VK_NULL_HANDLE)
        return;
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &bi);
    if (queries_ != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd_, queries_, 0, kQueries);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 0);
        queryCount_ = 1;
        labels_.push_back("begin");
    }
    state_ = State::recording;
}

VkDescriptorSet Stream::allocate(const Kernel& kernel)
{
    for (;; ++poolInUse_)
    {
        if (poolInUse_ == descriptorPools_.size())
        {
            const VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kPoolStorage },
                                                   { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kPoolUniform } };
            VkDescriptorPoolCreateInfo pi {};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = kPoolSets;
            pi.poolSizeCount = 2;
            pi.pPoolSizes = sizes;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(vk_, &pi, nullptr, &pool) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            descriptorPools_.push_back({ pool, 0, 0, 0 });
        }
        // Counted here, so a pool is never asked for more than it has (MoltenVK has crashed on that).
        auto& p = descriptorPools_[poolInUse_];
        if (p.sets + 1 > kPoolSets || p.storage + kernel.storageCount_ > kPoolStorage || p.uniform + kernel.uniformCount_ > kPoolUniform)
            continue;
        VkDescriptorSetAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = p.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &kernel.setLayout_;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vk_, &ai, &set) != VK_SUCCESS)
        {
            if (p.sets == 0)   // a fresh pool refused it: no pool will take it
                return VK_NULL_HANDLE;
            continue;
        }
        p.sets += 1;
        p.storage += kernel.storageCount_;
        p.uniform += kernel.uniformCount_;
        return set;
    }
}

void Stream::dispatch(const Kernel& kernel, Groups groups, std::initializer_list<Range> bindings, const void* push, uint32_t pushBytes)
{
    begin();
    jassert(kernel.valid());
    jassert(bindings.size() == kernel.bindings());   // every binding up to the highest the kernel declares, in order
    jassert(pushBytes == kernel.pushBytes());
    if (state_ != State::recording || ! kernel.valid())
        return;

    if (! kernel.types_.empty())
    {
        const VkDescriptorSet set = allocate(kernel);
        jassert(set != VK_NULL_HANDLE);
        if (set == VK_NULL_HANDLE)
            return;
        std::array<VkDescriptorBufferInfo, kMaxBindings> infos {};
        std::array<VkWriteDescriptorSet, kMaxBindings> writes {};
        uint32_t written = 0, binding = 0;
        for (const auto& r : bindings)
        {
            if (binding < kernel.types_.size() && kernel.types_[binding] != VK_DESCRIPTOR_TYPE_MAX_ENUM)
            {
                jassert(r.buffer != VK_NULL_HANDLE && r.bytes > 0);
                jassert(r.offset % context_.limits_.storageAlignment == 0);
                infos[written] = { r.buffer, r.offset, r.bytes };
                auto& w = writes[written];
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = set;
                w.dstBinding = binding;
                w.descriptorCount = 1;
                w.descriptorType = kernel.types_[binding];
                w.pBufferInfo = &infos[written];
                ++written;
            }
            ++binding;
        }
        vkUpdateDescriptorSets(vk_, written, writes.data(), 0, nullptr);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline_);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout_, 0, 1, &set, 0, nullptr);
    }
    else
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline_);

    if (kernel.pushBytes_ > 0 && push != nullptr)
        vkCmdPushConstants(cmd_, kernel.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, std::min(pushBytes, kernel.pushBytes_), push);
    vkCmdDispatch(cmd_, groups.x, groups.y, groups.z);
}

void Stream::barrier()
{
    begin();
    if (state_ != State::recording)
        return;
    VkMemoryBarrier mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT
                     | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cmd_, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Stream::fill(Range range, uint32_t word)
{
    begin();
    jassert(range.offset % 4 == 0 && range.bytes % 4 == 0);
    if (state_ == State::recording)
        vkCmdFillBuffer(cmd_, range.buffer, range.offset, range.bytes, word);
}

void Stream::copy(Range from, Range to)
{
    begin();
    jassert(to.bytes >= from.bytes);
    if (state_ != State::recording)
        return;
    const VkBufferCopy c { from.offset, to.offset, from.bytes };
    vkCmdCopyBuffer(cmd_, from.buffer, to.buffer, 1, &c);
}

Range Stream::stage(Usage usage, VkDeviceSize bytes, void*& mapped)
{
    auto& s = usage == Usage::Upload ? uploads_ : readbacks_;
    const VkDeviceSize at = (s.head + 15) & ~VkDeviceSize(15);
    if (s.block < s.blocks.size() && at + bytes <= s.blocks[s.block].size())
        s.head = at;
    else
    {
        // The next block that holds it, or a new one.
        if (s.block < s.blocks.size() && s.head > 0)
            ++s.block;
        while (s.block < s.blocks.size() && s.blocks[s.block].size() < bytes)
            ++s.block;
        if (s.block >= s.blocks.size())
        {
            s.blocks.push_back(context_.buffer(std::max(bytes, kStagingBlock), usage));
            s.block = s.blocks.size() - 1;
        }
        s.head = 0;
    }
    const auto& block = s.blocks[s.block];
    if (! block.valid() || block.data() == nullptr)
    {
        mapped = nullptr;
        return {};
    }
    mapped = static_cast<char*>(block.data()) + s.head;
    const Range r { block.handle(), s.head, bytes };
    s.head += bytes;
    return r;
}

void Stream::upload(Range to, const void* from)
{
    begin();
    if (state_ != State::recording)
        return;
    void* mapped = nullptr;
    const Range staged = stage(Usage::Upload, to.bytes, mapped);
    jassert(mapped != nullptr);
    if (mapped == nullptr)
        return;
    std::memcpy(mapped, from, (size_t) to.bytes);
    const VkBufferCopy c { staged.offset, to.offset, to.bytes };
    vkCmdCopyBuffer(cmd_, staged.buffer, to.buffer, 1, &c);
}

void Stream::download(Range from, void* to)
{
    begin();
    if (state_ != State::recording)
        return;
    void* mapped = nullptr;
    const Range staged = stage(Usage::Readback, from.bytes, mapped);
    jassert(mapped != nullptr);
    if (mapped == nullptr)
        return;
    const VkBufferCopy c { from.offset, staged.offset, from.bytes };
    vkCmdCopyBuffer(cmd_, from.buffer, staged.buffer, 1, &c);
    downloads_.push_back({ mapped, (size_t) from.bytes, to });
}

void Stream::time(const char* label)
{
    begin();
    if (state_ != State::recording || queries_ == VK_NULL_HANDLE || queryCount_ >= kQueries)
        return;
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, queryCount_++);
    labels_.push_back(label);
}

void Stream::submit()
{
    if (state_ != State::recording)
        return;
    // Everything written is the host's to read once the fence signals.
    VkMemoryBarrier mb {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(cmd_);
    for (size_t b = 0; b < uploads_.blocks.size() && b <= uploads_.block; ++b)
        uploads_.blocks[b].written();

    VkResult r = VK_ERROR_DEVICE_LOST;
    if (! device_->isLost())
    {
        VkSubmitInfo si {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        const juce::ScopedLock queueSync(device_->computeQueueLock());
        r = vkQueueSubmit(device_->computeQueue(), 1, &si, fence_);
    }
    if (r == VK_SUCCESS)
    {
        state_ = State::submitted;
        return;
    }
    if (r == VK_ERROR_DEVICE_LOST)
        device_->markLost();
    lost_ = true;
    finish(false);
}

Stream::Result Stream::wait(double seconds)
{
    if (state_ != State::submitted)
        return std::exchange(lost_, false) ? Result::lost : Result::done;
    const uint64_t ns = seconds < 0.0 ? UINT64_MAX : (uint64_t) (seconds * 1.0e9);
    const VkResult r = vkWaitForFences(vk_, 1, &fence_, VK_TRUE, ns);
    if (r == VK_TIMEOUT)
        return Result::pending;
    if (r != VK_SUCCESS)
    {
        device_->markLost();
        finish(false);
        return Result::lost;
    }
    finish(true);
    return Result::done;
}

void Stream::finish(bool delivered)
{
    if (delivered)
    {
        for (size_t b = 0; b < readbacks_.blocks.size() && b <= readbacks_.block; ++b)
            readbacks_.blocks[b].toRead();
        for (const auto& d : downloads_)
            std::memcpy(d.to, d.from, d.bytes);
        times_.clear();
        if (queries_ != VK_NULL_HANDLE && queryCount_ > 1)
        {
            std::array<uint64_t, kQueries> ticks {};
            if (vkGetQueryPoolResults(vk_, queries_, 0, queryCount_, queryCount_ * sizeof(uint64_t), ticks.data(),
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                for (uint32_t i = 1; i < queryCount_; ++i)
                    times_.push_back({ labels_[i], (double) ((ticks[i] - ticks[0]) & context_.timestampMask_) * context_.limits_.timestampNs * 1.0e-6 });
        }
    }
    vkResetFences(vk_, 1, &fence_);
    vkResetCommandPool(vk_, pool_, 0);
    for (auto& p : descriptorPools_)
    {
        vkResetDescriptorPool(vk_, p.pool, 0);
        p.sets = p.storage = p.uniform = 0;
    }
    poolInUse_ = 0;
    uploads_.block = uploads_.head = 0;
    readbacks_.block = readbacks_.head = 0;
    downloads_.clear();
    labels_.clear();
    queryCount_ = 0;
    state_ = State::idle;
}

} // namespace jvk::compute
