#pragma once

#include <unordered_map>

namespace jvk {
namespace Memory {

// Sentinel for "no memory type satisfies the request". Callers MUST check —
// the old code returned 0 on failure, which silently selected whatever type 0
// happens to be (typically DEVICE_LOCAL, non-host-visible): vkMapMemory then
// fails and the caller memcpys through a null pointer.
inline constexpr uint32_t kNoMemoryType = 0xFFFFFFFFu;

inline uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    return kNoMemoryType;
}

inline VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    jassert(alignment != 0 && (alignment & (alignment - 1)) == 0); // POT only
    return (value + alignment - 1) & ~(alignment - 1);
}

// =============================================================================
// L1 — GPU Pool (persistent, device-local, sub-allocated)
//
// Backed by the vendored Vulkan Memory Allocator (community-consensus
// replacement for the old hand-rolled block/free-list engine). The external
// API is unchanged — callers still receive a (memory, offset) pair and bind
// with vkBindImageMemory/vkBindBufferMemory — but VMA supplies what the old
// engine got wrong or lacked: bufferImageGranularity handling per spec,
// memory-type fallbacks, the process-wide maxMemoryAllocationCount (4096)
// budget via sub-allocation, dedicated-allocation support, double-free/leak
// detection in debug, and internal thread safety (no external lock needed).
// The VmaAllocator itself is owned by Device; L1 is a thin view.
// =============================================================================

class L1 {
public:
    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   offset = 0;
        VkDeviceSize   size   = 0;
        uint32_t       blockIndex = 0;   // legacy field, unused
        uint32_t       memoryType = 0;
        void*          vma = nullptr;    // VmaAllocation handle
    };

    L1() = default;
    explicit L1(VmaAllocator allocator) : allocator_(allocator) {}

    // Thin view over the Device-owned allocator — trivially movable, no
    // ownership. (The old engine owned VkDeviceMemory blocks and needed
    // careful move/destroy plumbing; VMA frees per-allocation.)
    L1(L1&& o) noexcept : allocator_(o.allocator_) { o.allocator_ = nullptr; }
    L1& operator=(L1&& o) noexcept
    {
        allocator_ = o.allocator_;
        o.allocator_ = nullptr;
        return *this;
    }
    L1(const L1&) = delete;
    L1& operator=(const L1&) = delete;

    Allocation alloc(VkMemoryRequirements req, VkMemoryPropertyFlags properties)
    {
        if (allocator_ == nullptr) return {};
        VmaAllocationCreateInfo ci {};
        ci.requiredFlags = properties;
        VmaAllocation a = nullptr;
        VmaAllocationInfo info {};
        if (vmaAllocateMemory(allocator_, &req, &ci, &a, &info) != VK_SUCCESS)
            return {};
        return { info.deviceMemory, info.offset, req.size, 0, info.memoryType, a };
    }

    void free(Allocation a)
    {
        if (allocator_ != nullptr && a.vma != nullptr)
            vmaFreeMemory(allocator_, static_cast<VmaAllocation>(a.vma));
    }

private:
    VmaAllocator allocator_ = nullptr;
};


// =============================================================================
// L2 — Staging (host-visible, CPU→GPU transfer, linear bump allocator)
//
// Every mutator takes `lock_`. L2 is used in two shapes:
//   1. Per-Renderer (Renderer::staging_): record-phase allocs on the message
//      thread, worker-phase allocs on the render-worker thread. The
//      isBusy() gate *should* keep these disjoint within one instance, but
//      the internal lock is cheap and defends against any future caller
//      that forgets the contract.
//   2. On Device::staging_: used only by ResourceCaches::createBlackPixel
//      at init. Message-thread only today, but the lock keeps the type
//      safe to share more broadly later.
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

    // Locks are not movable. Move the underlying state and let lock_ be
    // default-constructed on the moved-to instance.
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
        const juce::ScopedLock lk(lock_);
        if (!activeBlocks.empty()) {
            auto& b = activeBlocks.back();
            // Align every allocation start: VkBufferImageCopy::bufferOffset
            // must be a multiple of 4 AND of the texel block size. The old
            // raw bump worked only because every caller's size happened to
            // be a multiple of 4 — one odd-sized allocation would silently
            // misalign every subsequent image copy in the block.
            VkDeviceSize head = alignUp(b.writeHead, kOffsetAlign);
            if (head + size <= b.capacity) {
                Allocation a { static_cast<char*>(b.mapped) + head, b.buffer, head };
                b.writeHead = head + size;
                b.uncommitted++;
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
                block.uncommitted = 0;
                block.recordedFrom = false;
                found = true;
                break;
            }
        }
        if (!found) {
            VkDeviceSize cap = std::max(BLOCK_SIZE, size);
            if (!createBlock(block, cap))
                return {};   // caller must null-check mappedPtr
        }

        Allocation a { block.mapped, block.buffer, 0 };
        block.writeHead = size;
        block.uncommitted = 1;
        activeBlocks.push_back(std::move(block));
        return a;
    }

    struct Block {
        VkBuffer     buffer   = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*        mapped   = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize writeHead = 0;
        // Allocations handed out but not yet committed via commit(). A block
        // with uncommitted > 0 must not be parked/recycled: its data hasn't
        // been queued for recording yet (the alloc→memcpy→upload() sequence
        // on the message thread can straddle the worker's flush).
        int          uncommitted = 0;
        // True once a flush has recorded copies while this block was the
        // current write target — its contents may be referenced by an
        // in-flight command buffer, so only the normal fence-keyed park may
        // release it (recycleUnrecorded must skip it).
        bool         recordedFrom = false;
    };

    // Mark an allocation's data as queued (called by Renderer::upload under
    // its own lock). Clears the park hold taken by alloc().
    void commit(const Allocation& a)
    {
        const juce::ScopedLock lk(lock_);
        for (auto& b : activeBlocks)
            if (b.buffer == a.buffer) {
                if (b.uncommitted > 0) b.uncommitted--;
                return;
            }
    }

    // Move every parkable active block into `out`: not the current write
    // target (activeBlocks.back()), and no uncommitted allocations. Called
    // by the worker's flushUploads in the same lock hold as the queue swap;
    // parked blocks recycle when the frame slot's fence next signals. The
    // surviving current block is marked recordedFrom — this flush may have
    // recorded copies sourced from it.
    void parkAllButCurrent(std::vector<Block>& out)
    {
        const juce::ScopedLock lk(lock_);
        if (activeBlocks.empty()) return;
        for (size_t i = 0; i + 1 < activeBlocks.size(); ) {
            if (activeBlocks[i].uncommitted == 0) {
                out.push_back(std::move(activeBlocks[i]));
                activeBlocks.erase(activeBlocks.begin() + static_cast<ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        activeBlocks.back().recordedFrom = true;
    }

    // Skip-path bound (no submission happened): return to the free list any
    // active block that provably was never recorded from and is no longer
    // referenced — safe to reuse immediately without a fence. Keeps a
    // minimised-window dynamic feed from growing the belt without bound.
    void recycleUnrecorded(const std::vector<VkBuffer>& stillReferenced)
    {
        const juce::ScopedLock lk(lock_);
        for (size_t i = 0; i < activeBlocks.size(); ) {
            auto& b = activeBlocks[i];
            const bool referenced = std::find(stillReferenced.begin(), stillReferenced.end(),
                                              b.buffer) != stillReferenced.end();
            const bool isCurrent  = (i + 1 == activeBlocks.size());
            if (!referenced && !isCurrent && !b.recordedFrom && b.uncommitted == 0) {
                b.writeHead = 0;
                freeBlocks.push_back(std::move(b));
                activeBlocks.erase(activeBlocks.begin() + static_cast<ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
    }

    void recycle(std::vector<Block>& used)
    {
        const juce::ScopedLock lk(lock_);
        for (auto& b : used) {
            b.writeHead = 0;
            b.uncommitted = 0;
            b.recordedFrom = false;
            freeBlocks.push_back(std::move(b));
        }
        used.clear();
    }

private:
    static constexpr VkDeviceSize BLOCK_SIZE = 4 * 1024 * 1024;
    // VkBufferImageCopy::bufferOffset must be a multiple of 4 and of the
    // texel block size; 16 covers every format jvk stages.
    static constexpr VkDeviceSize kOffsetAlign = 16;

    std::vector<Block> activeBlocks;
    std::vector<Block> freeBlocks;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    mutable juce::CriticalSection lock_;

    bool createBlock(Block& b, VkDeviceSize cap)
    {
        VkBufferCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = cap;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &ci, nullptr, &b.buffer) != VK_SUCCESS)
            { b = {}; return false; }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);

        VkMemoryAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == kNoMemoryType
            || vkAllocateMemory(device, &ai, nullptr, &b.memory) != VK_SUCCESS
            || vkBindBufferMemory(device, b.buffer, b.memory, 0) != VK_SUCCESS
            || vkMapMemory(device, b.memory, 0, cap, 0, &b.mapped) != VK_SUCCESS)
        {
            destroyBlock(b);
            return false;
        }
        b.capacity = cap;
        b.writeHead = 0;
        b.uncommitted = 0;
        b.recordedFrom = false;
        return true;
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

// Shared across all plugin instances. Every mutating call (registerLayout /
// alloc / free / internal growPool) takes `lock_` so two editors' message
// threads — plus any worker-thread caller we might add later — can't race
// on `pools`, `layouts`, or the VkDescriptorPool itself (which Vulkan
// requires to be externally synchronized per §13.2.3).
class M {
public:
    using LayoutID = uint32_t;
    static constexpr LayoutID IMAGE_SAMPLER = 0;
    static constexpr LayoutID kInvalidLayout = 0xFFFFFFFFu;

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

    // Locks are not movable — but the members we actually care about are.
    // Leave lock_ default-constructed on the moved-to instance.
    M(M&& o) noexcept
        : device(o.device), layouts(std::move(o.layouts)), pools(std::move(o.pools)),
          setToPool(std::move(o.setToPool))
    { o.device = VK_NULL_HANDLE; }

    M& operator=(M&& o) noexcept
    {
        if (this != &o) {
            destroy();
            device = o.device;
            layouts = std::move(o.layouts);
            pools = std::move(o.pools);
            setToPool = std::move(o.setToPool);
            o.device = VK_NULL_HANDLE;
        }
        return *this;
    }

    M(const M&) = delete;
    M& operator=(const M&) = delete;

    LayoutID registerLayout(const VkDescriptorSetLayoutBinding* bindings, uint32_t count)
    {
        const juce::ScopedLock lk(lock_);

        // Dedupe by binding content. Without this, every jvk::Shader
        // registered a fresh layout on ensureCreated — and because alloc()
        // only reuses pools tagged with the SAME LayoutID, each leaked
        // layout also earned its own 256-set VkDescriptorPool. A plugin
        // rebuilding shaders on preset change bled layouts + pools until
        // device destruction. Identical shaders now share one entry; the
        // registry is bounded by the number of DISTINCT layout shapes.
        const uint64_t key = hashBindings(bindings, count);
        for (size_t i = 0; i < layouts.size(); i++) {
            if (layouts[i].contentKey == key) {
                layouts[i].refs++;
                return static_cast<LayoutID>(i);
            }
        }

        LayoutEntry entry;
        entry.contentKey = key;
        entry.refs = 1;
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
        if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &entry.layout) != VK_SUCCESS)
            return kInvalidLayout;

        auto id = static_cast<LayoutID>(layouts.size());
        layouts.push_back(std::move(entry));
        return id;
    }

    // Release one reference taken by registerLayout. The VkDescriptorSetLayout
    // itself lives until M::destroy — pools reference it and reuse is the
    // point — but the refcount keeps the dedup honest and gives leak
    // diagnostics a number to watch.
    void unregisterLayout(LayoutID id)
    {
        const juce::ScopedLock lk(lock_);
        if (id < layouts.size() && layouts[id].refs > 0)
            layouts[id].refs--;
    }

    // Reads take the lock: registerLayout's push_back can REALLOCATE the
    // vector's storage from another thread (editor A's worker lazily builds
    // a Shader while editor B's worker builds a Pipeline) — the entry may be
    // immutable, the buffer holding it is not.
    VkDescriptorSetLayout getLayout(LayoutID id) const
    {
        const juce::ScopedLock lk(lock_);
        return id < layouts.size() ? layouts[id].layout : VK_NULL_HANDLE;
    }

    VkDescriptorSet alloc(LayoutID id = IMAGE_SAMPLER)
    {
        const juce::ScopedLock lk(lock_);
        if (id >= layouts.size())
            return VK_NULL_HANDLE;
        auto& entry = layouts[id];
        VkDescriptorSetAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &entry.layout;

        // Try existing pools that serve this layout AND still have room. A FULL
        // pool is skipped, never handed to vkAllocateDescriptorSets — MoltenVK
        // crashes on that instead of returning VK_ERROR_OUT_OF_POOL_MEMORY.
        for (size_t i = 0; i < pools.size(); ++i) {
            if (pools[i].layout != id || pools[i].used >= SETS_PER_POOL)
                continue;
            ai.descriptorPool = pools[i].pool;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(device, &ai, &set) == VK_SUCCESS) {
                pools[i].used++;
                setToPool[set] = i;
                return set;
            }
        }

        // None had room — grow a fresh pool for this layout and allocate
        // there. growPool failure must NOT push a null pool entry: the old
        // code did, and because the null entry reported used==0 it was
        // selected FIRST on every subsequent alloc — a permanently poisoned
        // allocator handing VK_NULL_HANDLE pools to vkAllocateDescriptorSets
        // (the MoltenVK crash class the used-count fix was written for).
        if (!growPool(id, entry.poolSizes))
            return VK_NULL_HANDLE;
        const size_t last = pools.size() - 1;
        ai.descriptorPool = pools[last].pool;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(device, &ai, &set) == VK_SUCCESS) {
            pools[last].used++;
            setToPool[set] = last;
        }
        return set;
    }

    void free(VkDescriptorSet set)
    {
        const juce::ScopedLock lk(lock_);
        auto it = setToPool.find(set);
        if (it == setToPool.end())
            return;                                  // not ours / already freed
        auto& p = pools[it->second];
        if (vkFreeDescriptorSets(device, p.pool, 1, &set) == VK_SUCCESS && p.used > 0)
            p.used--;
        setToPool.erase(it);
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
        uint64_t contentKey = 0;   // hash of the binding array (dedup)
        uint32_t refs       = 0;   // registerLayout/unregisterLayout balance
    };

    static uint64_t hashBindings(const VkDescriptorSetLayoutBinding* bindings, uint32_t count)
    {
        uint64_t h = 1469598103934665603ull;                 // FNV-1a
        auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix(count);
        for (uint32_t i = 0; i < count; i++) {
            mix(bindings[i].binding);
            mix(static_cast<uint64_t>(bindings[i].descriptorType));
            mix(bindings[i].descriptorCount);
            mix(bindings[i].stageFlags);
        }
        return h;
    }

    // A pool, tagged with the layout it serves and the number of live sets in
    // it. Tracking `used` lets alloc() SKIP a pool that has hit maxSets rather
    // than ask Vulkan to allocate from a full pool — MoltenVK crashes on that
    // (memset past its internal set array) instead of returning an error, and a
    // silently-null set draws as a black frame.
    struct Pool {
        VkDescriptorPool pool   = VK_NULL_HANDLE;
        LayoutID         layout = 0;
        uint32_t         used   = 0;
    };

    VkDevice device = VK_NULL_HANDLE;
    std::vector<LayoutEntry> layouts;
    std::vector<Pool> pools;
    std::unordered_map<VkDescriptorSet, size_t> setToPool;  // set -> index into `pools`
    mutable juce::CriticalSection lock_;

    // Caller holds lock_. Returns false (and pushes nothing) on failure.
    bool growPool(LayoutID layout, const std::vector<VkDescriptorPoolSize>& sizes)
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
        if (vkCreateDescriptorPool(device, &ci, nullptr, &pool) != VK_SUCCESS)
            return false;
        pools.push_back({ pool, layout, 0 });
        return true;
    }

    void destroy()
    {
        if (device == VK_NULL_HANDLE) return;
        for (auto& p : pools)
            if (p.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, p.pool, nullptr);
        for (auto& entry : layouts)
            if (entry.layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, entry.layout, nullptr);
        pools.clear();
        setToPool.clear();
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
        for (int i = 0; i < MAX_SLOTS; i++) retiredBySlot[i] = std::move(o.retiredBySlot[i]);
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
            for (int i = 0; i < MAX_SLOTS; i++) retiredBySlot[i] = std::move(o.retiredBySlot[i]);
            o.device = VK_NULL_HANDLE;
            for (int i = 0; i < MAX_SLOTS; i++) o.slots[i] = {};
        }
        return *this;
    }

    V(const V&) = delete;
    V& operator=(const V&) = delete;

    // Caller contract: the frame slot's fence has been waited immediately
    // before this call (Renderer::execute after target_.beginFrame). Only
    // then is it safe to destroy THIS slot's retired buffers — a buffer
    // retired by grow() during slot S's frame is referenced by slot S's
    // command buffer, whose completion is proven exactly when slot S's
    // fence next signals. The old code kept ONE retired list flushed on
    // every beginFrame: a slot-S buffer died on the very next frame
    // (slot S^1), whose fence only proves frame S-1 finished — destroying
    // a vertex buffer the in-flight frame S was still reading.
    void beginFrame(int frameSlot)
    {
        currentSlot = frameSlot;
        slots[frameSlot].writeHead = 0;
        flushRetired(frameSlot);
    }

    VkDeviceSize write(const void* data, VkDeviceSize byteCount)
    {
        auto& slot = slots[currentSlot];
        if (slot.writeHead + byteCount > slot.capacity)
            grow(slot, std::max(slot.capacity * 2, slot.writeHead + byteCount));

        if (slot.mapped == nullptr)   // createSlot failed (OOM/device lost)
            return 0;

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
    std::vector<Retired> retiredBySlot[MAX_SLOTS];

    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    void createSlot(Slot& s, VkDeviceSize cap)
    {
        VkBufferCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = cap;
        ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &ci, nullptr, &s.buffer) != VK_SUCCESS)
            { s = {}; return; }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, s.buffer, &req);

        VkMemoryAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == kNoMemoryType
            || vkAllocateMemory(device, &ai, nullptr, &s.memory) != VK_SUCCESS
            || vkBindBufferMemory(device, s.buffer, s.memory, 0) != VK_SUCCESS
            || vkMapMemory(device, s.memory, 0, cap, 0, &s.mapped) != VK_SUCCESS)
        {
            destroySlot(s);
            return;
        }
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
        // The old buffer is referenced by draws already recorded into the
        // CURRENT slot's command buffer — key its destruction to this
        // slot's fence (see beginFrame).
        if (slot.mapped) vkUnmapMemory(device, slot.memory);
        retiredBySlot[currentSlot].push_back({ slot.buffer, slot.memory });
        slot.buffer = VK_NULL_HANDLE;
        slot.memory = VK_NULL_HANDLE;
        slot.mapped = nullptr;
        createSlot(slot, newCap);
        slot.writeHead = 0;
    }

    void flushRetired(int frameSlot)
    {
        for (auto& r : retiredBySlot[frameSlot]) {
            if (r.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, r.buffer, nullptr);
            if (r.memory != VK_NULL_HANDLE) vkFreeMemory(device, r.memory, nullptr);
        }
        retiredBySlot[frameSlot].clear();
    }

    void destroy()
    {
        if (device == VK_NULL_HANDLE) return;
        for (int i = 0; i < MAX_SLOTS; i++) flushRetired(i);
        for (auto& s : slots) destroySlot(s);
    }
};


} // namespace Memory
} // namespace jvk
