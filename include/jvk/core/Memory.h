#pragma once

namespace jvk {
namespace Memory {

inline uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    return 0;
}

inline VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

// =============================================================================
// L1 — GPU Pool (persistent, device-local, sub-allocated)
// =============================================================================

class L1 {
public:
    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   offset = 0;
        VkDeviceSize   size   = 0;
        uint32_t       blockIndex = 0;
        uint32_t       memoryType = 0;
    };

    L1() = default;

    L1(VkPhysicalDevice pd, VkDevice d)
        : physDevice(pd), device(d)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        bufferImageGranularity = props.limits.bufferImageGranularity;
    }

    ~L1() { destroy(); }

    L1(L1&& o) noexcept
        : pools(std::move(o.pools)), physDevice(o.physDevice),
          device(o.device), bufferImageGranularity(o.bufferImageGranularity)
    {
        o.device = VK_NULL_HANDLE;
    }

    L1& operator=(L1&& o) noexcept
    {
        if (this != &o) {
            destroy();
            pools = std::move(o.pools);
            physDevice = o.physDevice;
            device = o.device;
            bufferImageGranularity = o.bufferImageGranularity;
            o.device = VK_NULL_HANDLE;
        }
        return *this;
    }

    L1(const L1&) = delete;
    L1& operator=(const L1&) = delete;

    Allocation alloc(VkMemoryRequirements req, VkMemoryPropertyFlags properties)
    {
        uint32_t memType = findMemoryType(physDevice, req.memoryTypeBits, properties);
        VkDeviceSize alignment = std::max(req.alignment, bufferImageGranularity);
        auto& blocks = pools[memType];

        for (uint32_t bi = 0; bi < blocks.size(); bi++) {
            auto& block = blocks[bi];
            for (size_t fi = 0; fi < block.freeList.size(); fi++) {
                auto& fr = block.freeList[fi];
                VkDeviceSize aligned = alignUp(fr.offset, alignment);
                VkDeviceSize padding = aligned - fr.offset;

                if (fr.size >= padding + req.size) {
                    Allocation a { block.memory, aligned, req.size, bi, memType };

                    if (padding > 0) {
                        VkDeviceSize remain = fr.size - padding - req.size;
                        fr.size = padding;
                        if (remain > 0)
                            block.freeList.insert(block.freeList.begin() + static_cast<ptrdiff_t>(fi + 1),
                                                  FreeRange { aligned + req.size, remain });
                    } else {
                        VkDeviceSize remain = fr.size - req.size;
                        if (remain > 0) { fr.offset += req.size; fr.size = remain; }
                        else block.freeList.erase(block.freeList.begin() + static_cast<ptrdiff_t>(fi));
                    }
                    return a;
                }
            }
        }

        VkDeviceSize newSize = std::max(BLOCK_SIZE, req.size);
        Block block;
        block.size = newSize;

        VkMemoryAllocateInfo allocInfo {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = newSize;
        allocInfo.memoryTypeIndex = memType;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &block.memory) != VK_SUCCESS)
            return {};

        block.freeList.push_back({ 0, newSize });
        blocks.push_back(std::move(block));
        return alloc(req, properties);
    }

    void free(Allocation a)
    {
        if (a.memory == VK_NULL_HANDLE) return;
        auto it = pools.find(a.memoryType);
        if (it == pools.end() || a.blockIndex >= it->second.size()) return;

        auto& block = it->second[a.blockIndex];
        FreeRange nr { a.offset, a.size };

        auto pos = block.freeList.begin();
        while (pos != block.freeList.end() && pos->offset < nr.offset) ++pos;
        pos = block.freeList.insert(pos, nr);

        auto next = pos + 1;
        if (next != block.freeList.end() && pos->offset + pos->size == next->offset) {
            pos->size += next->size;
            block.freeList.erase(next);
        }
        if (pos != block.freeList.begin()) {
            auto prev = pos - 1;
            if (prev->offset + prev->size == pos->offset) {
                prev->size += pos->size;
                block.freeList.erase(pos);
            }
        }
    }

private:
    static constexpr VkDeviceSize BLOCK_SIZE = 64 * 1024 * 1024;

    struct FreeRange { VkDeviceSize offset, size; };
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size = 0;
        std::vector<FreeRange> freeList;
    };

    std::unordered_map<uint32_t, std::vector<Block>> pools;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceSize bufferImageGranularity = 1;

    void destroy()
    {
        if (device == VK_NULL_HANDLE) return;
        for (auto& [_, blocks] : pools)
            for (auto& block : blocks)
                if (block.memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, block.memory, nullptr);
        pools.clear();
    }
};


// =============================================================================
// L2 — Staging (host-visible, CPU→GPU transfer, linear bump allocator)
// =============================================================================

class L2 {
public:
    struct Allocation {
        void*        mappedPtr;
        VkBuffer     buffer;
        VkDeviceSize offset;
    };

    L2() = default;

    L2(VkPhysicalDevice pd, VkDevice d)
        : physDevice(pd), device(d) {}

    ~L2() { destroy(); }

    L2(L2&& o) noexcept
        : activeBlocks(std::move(o.activeBlocks)), freeBlocks(std::move(o.freeBlocks)),
          physDevice(o.physDevice), device(o.device)
    { o.device = VK_NULL_HANDLE; }

    L2& operator=(L2&& o) noexcept
    {
        if (this != &o) {
            destroy();
            activeBlocks = std::move(o.activeBlocks);
            freeBlocks = std::move(o.freeBlocks);
            physDevice = o.physDevice;
            device = o.device;
            o.device = VK_NULL_HANDLE;
        }
        return *this;
    }

    L2(const L2&) = delete;
    L2& operator=(const L2&) = delete;

    Allocation alloc(VkDeviceSize size)
    {
        if (!activeBlocks.empty()) {
            auto& b = activeBlocks.back();
            if (b.writeHead + size <= b.capacity) {
                Allocation a { static_cast<char*>(b.mapped) + b.writeHead, b.buffer, b.writeHead };
                b.writeHead += size;
                return a;
            }
        }

        Block block {};
        bool found = false;
        for (size_t i = freeBlocks.size(); i-- > 0; ) {
            if (freeBlocks[i].capacity >= size) {
                block = std::move(freeBlocks[i]);
                freeBlocks.erase(freeBlocks.begin() + static_cast<ptrdiff_t>(i));
                block.writeHead = 0;
                found = true;
                break;
            }
        }
        if (!found) {
            VkDeviceSize cap = std::max(BLOCK_SIZE, size);
            createBlock(block, cap);
        }

        Allocation a { block.mapped, block.buffer, 0 };
        block.writeHead = size;
        activeBlocks.push_back(std::move(block));
        return a;
    }

    struct Block {
        VkBuffer     buffer   = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*        mapped   = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize writeHead = 0;
    };

    void moveActiveTo(std::vector<Block>& used)
    {
        for (auto& b : activeBlocks) used.push_back(std::move(b));
        activeBlocks.clear();
    }

    void recycle(std::vector<Block>& used)
    {
        for (auto& b : used) freeBlocks.push_back(std::move(b));
        used.clear();
    }

private:
    static constexpr VkDeviceSize BLOCK_SIZE = 4 * 1024 * 1024;

    std::vector<Block> activeBlocks;
    std::vector<Block> freeBlocks;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    void createBlock(Block& b, VkDeviceSize cap)
    {
        VkBufferCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = cap;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &ci, nullptr, &b.buffer);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);

        VkMemoryAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &ai, nullptr, &b.memory);
        vkBindBufferMemory(device, b.buffer, b.memory, 0);
        vkMapMemory(device, b.memory, 0, cap, 0, &b.mapped);
        b.capacity = cap;
        b.writeHead = 0;
    }

    void destroyBlock(Block& b)
    {
        if (b.mapped) vkUnmapMemory(device, b.memory);
        if (b.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory != VK_NULL_HANDLE) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }

    void destroy()
    {
        if (device == VK_NULL_HANDLE) return;
        for (auto& b : activeBlocks) destroyBlock(b);
        for (auto& b : freeBlocks) destroyBlock(b);
        activeBlocks.clear();
        freeBlocks.clear();
    }
};


// =============================================================================
// M — Bindings (descriptor set pool, multi-layout, growable)
// =============================================================================

class M {
public:
    using LayoutID = uint32_t;
    static constexpr LayoutID IMAGE_SAMPLER = 0;

    M() = default;

    explicit M(VkDevice d) : device(d)
    {
        VkDescriptorSetLayoutBinding binding {};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        registerLayout(&binding, 1);
    }

    ~M() { destroy(); }

    M(M&& o) noexcept
        : device(o.device), layouts(std::move(o.layouts)), pools(std::move(o.pools))
    { o.device = VK_NULL_HANDLE; }

    M& operator=(M&& o) noexcept
    {
        if (this != &o) {
            destroy();
            device = o.device;
            layouts = std::move(o.layouts);
            pools = std::move(o.pools);
            o.device = VK_NULL_HANDLE;
        }
        return *this;
    }

    M(const M&) = delete;
    M& operator=(const M&) = delete;

    LayoutID registerLayout(const VkDescriptorSetLayoutBinding* bindings, uint32_t count)
    {
        LayoutEntry entry;
        for (uint32_t i = 0; i < count; i++) {
            auto& b = bindings[i];
            bool found = false;
            for (auto& ps : entry.poolSizes) {
                if (ps.type == b.descriptorType) { ps.descriptorCount += b.descriptorCount; found = true; break; }
            }
            if (!found) entry.poolSizes.push_back({ b.descriptorType, b.descriptorCount });
        }

        VkDescriptorSetLayoutCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = count;
        ci.pBindings = bindings;
        vkCreateDescriptorSetLayout(device, &ci, nullptr, &entry.layout);

        auto id = static_cast<LayoutID>(layouts.size());
        layouts.push_back(std::move(entry));
        return id;
    }

    VkDescriptorSetLayout getLayout(LayoutID id) const { return layouts[id].layout; }

    VkDescriptorSet alloc(LayoutID id = IMAGE_SAMPLER)
    {
        auto& entry = layouts[id];
        VkDescriptorSetAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &entry.layout;

        // Try existing pools
        for (auto& pool : pools) {
            ai.descriptorPool = pool;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(device, &ai, &set) == VK_SUCCESS)
                return set;
        }

        // Grow: create a new pool
        growPool(entry.poolSizes);
        ai.descriptorPool = pools.back();
        VkDescriptorSet set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(device, &ai, &set);
        return set;
    }

    void free(VkDescriptorSet set)
    {
        for (auto& pool : pools)
            if (vkFreeDescriptorSets(device, pool, 1, &set) == VK_SUCCESS)
                return;
    }

    static void writeImage(VkDevice device, VkDescriptorSet set, uint32_t binding,
                           VkImageView view, VkSampler sampler)
    {
        VkDescriptorImageInfo ii {};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii.imageView = view;
        ii.sampler = sampler;

        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    static void writeBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                            VkDescriptorType type, VkBuffer buffer,
                            VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE)
    {
        VkDescriptorBufferInfo bi {};
        bi.buffer = buffer;
        bi.offset = offset;
        bi.range = range;

        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorType = type;
        w.descriptorCount = 1;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

private:
    static constexpr uint32_t SETS_PER_POOL = 256;

    struct LayoutEntry {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> poolSizes;
    };

    VkDevice device = VK_NULL_HANDLE;
    std::vector<LayoutEntry> layouts;
    std::vector<VkDescriptorPool> pools;

    void growPool(const std::vector<VkDescriptorPoolSize>& sizes)
    {
        std::vector<VkDescriptorPoolSize> scaled = sizes;
        for (auto& ps : scaled) ps.descriptorCount *= SETS_PER_POOL;

        VkDescriptorPoolCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ci.maxSets = SETS_PER_POOL;
        ci.poolSizeCount = static_cast<uint32_t>(scaled.size());
        ci.pPoolSizes = scaled.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;
        vkCreateDescriptorPool(device, &ci, nullptr, &pool);
        pools.push_back(pool);
    }

    void destroy()
    {
        if (device == VK_NULL_HANDLE) return;
        for (auto& pool : pools)
            if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
        for (auto& entry : layouts)
            if (entry.layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, entry.layout, nullptr);
        pools.clear();
        layouts.clear();
        device = VK_NULL_HANDLE;
    }
};


// =============================================================================
// V — Vertices (per-frame, host-visible, linear bump allocator)
// =============================================================================

class V {
public:
    V() = default;

    V(VkPhysicalDevice pd, VkDevice d)
        : physDevice(pd), device(d)
    {
        for (int i = 0; i < MAX_SLOTS; i++)
            createSlot(slots[i], INITIAL_CAPACITY);
    }

    ~V() { destroy(); }

    V(V&& o) noexcept
        : currentSlot(o.currentSlot), physDevice(o.physDevice), device(o.device)
    {
        for (int i = 0; i < MAX_SLOTS; i++) slots[i] = o.slots[i];
        retired = std::move(o.retired);
        o.device = VK_NULL_HANDLE;
        for (int i = 0; i < MAX_SLOTS; i++) o.slots[i] = {};
    }

    V& operator=(V&& o) noexcept
    {
        if (this != &o) {
            destroy();
            physDevice = o.physDevice;
            device = o.device;
            currentSlot = o.currentSlot;
            for (int i = 0; i < MAX_SLOTS; i++) slots[i] = o.slots[i];
            retired = std::move(o.retired);
            o.device = VK_NULL_HANDLE;
            for (int i = 0; i < MAX_SLOTS; i++) o.slots[i] = {};
        }
        return *this;
    }

    V(const V&) = delete;
    V& operator=(const V&) = delete;

    void beginFrame(int frameSlot)
    {
        currentSlot = frameSlot;
        slots[frameSlot].writeHead = 0;
        flushRetired();
    }

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

    VkBuffer getBuffer() const { return slots[currentSlot].buffer; }
    bool isValid() const { return slots[0].buffer != VK_NULL_HANDLE; }

private:
    static constexpr VkDeviceSize INITIAL_CAPACITY = 8 * 1024 * 1024;
    static constexpr int MAX_SLOTS = 2;

    struct Slot {
        VkBuffer       buffer   = VK_NULL_HANDLE;
        VkDeviceMemory memory   = VK_NULL_HANDLE;
        void*          mapped   = nullptr;
        VkDeviceSize   capacity = 0;
        VkDeviceSize   writeHead = 0;
    };

    struct Retired { VkBuffer buffer; VkDeviceMemory memory; };

    Slot slots[MAX_SLOTS] {};
    int  currentSlot = 0;
    std::vector<Retired> retired;

    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    void createSlot(Slot& s, VkDeviceSize cap)
    {
        VkBufferCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = cap;
        ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &ci, nullptr, &s.buffer);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, s.buffer, &req);

        VkMemoryAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &ai, nullptr, &s.memory);
        vkBindBufferMemory(device, s.buffer, s.memory, 0);
        vkMapMemory(device, s.memory, 0, cap, 0, &s.mapped);
        s.capacity = cap;
        s.writeHead = 0;
    }

    void destroySlot(Slot& s)
    {
        if (s.mapped) vkUnmapMemory(device, s.memory);
        if (s.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, s.buffer, nullptr);
        if (s.memory != VK_NULL_HANDLE) vkFreeMemory(device, s.memory, nullptr);
        s = {};
    }

    void grow(Slot& slot, VkDeviceSize newCap)
    {
        retired.push_back({ slot.buffer, slot.memory });
        slot.buffer = VK_NULL_HANDLE;
        slot.memory = VK_NULL_HANDLE;
        slot.mapped = nullptr;
        createSlot(slot, newCap);
        slot.writeHead = 0;
    }

    void flushRetired()
    {
        for (auto& r : retired) {
            if (r.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, r.buffer, nullptr);
            if (r.memory != VK_NULL_HANDLE) vkFreeMemory(device, r.memory, nullptr);
        }
        retired.clear();
    }

    void destroy()
    {
        flushRetired();
        for (auto& s : slots) destroySlot(s);
    }
};


// =============================================================================
// CPU — CPU-side cache memory (simple allocator)
// =============================================================================

class CPU {
public:
    struct Allocation {
        void*  ptr  = nullptr;
        size_t size = 0;
    };

    CPU() = default;

    Allocation alloc(size_t size)
    {
        auto* p = ::operator new(size);
        return { p, size };
    }

    void free(Allocation a)
    {
        ::operator delete(a.ptr);
    }
};

} // namespace Memory
} // namespace jvk
