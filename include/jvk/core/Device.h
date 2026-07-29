#pragma once
#include <set>
#if JUCE_MAC || JUCE_LINUX
#include <dlfcn.h>
#endif

namespace jvk {

class ResourceCaches;  // forward — defined in Cache.h

// Process-wide shared Vulkan device and memory allocators. Refcounted via
// weak_ptr (acquire() returns the same instance to every caller), so the
// instance stays alive until the last shared_ptr holder drops it.
//
// Multi-instance thread contract (every member below falls into exactly one
// category):
//
//   (a) Immutable after init — safe to read from any thread:
//         instance(), physicalDevice(), device(), graphicsQueue(),
//         graphicsFamily(), presentQueue(), presentFamily(), commandPool()
//         (commandPool is only used by submitImmediate; per-render command
//         pools live on RenderTarget), time().
//
//   (b) Internally synchronized — safe to call from any thread, including
//       concurrently from multiple editors' workers:
//         pool() (Memory::L1), bindings() (Memory::M).
//
//   (c) Message-thread only — caller must not invoke from a render worker.
//       Safe across editors because JUCE's message thread is single-threaded:
//         submitImmediate(), caches(), initCaches(). (Per-frame staging
//         lives on each Renderer; Device owns no staging allocator.)
//
// Everything that used to be mutable per-frame state on Device has been
// moved to Renderer: upload queues, deletion queues, gradient atlas, the
// worker-thread staging allocator. See ARCHITECTURE.md for the rationale.
class Device {
public:
    static std::shared_ptr<Device> acquire();

    VkInstance       instance()       const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physDevice_; }
    VkDevice         device()         const { return device_; }
    VkQueue          graphicsQueue()  const { return graphicsQueue_; }
    uint32_t         graphicsFamily() const { return graphicsFamily_; }
    VkQueue          presentQueue()   const { return presentQueue_; }
    uint32_t         presentFamily()  const { return presentFamily_; }
    VkCommandPool    commandPool()    const { return commandPool_; }

    Memory::L1& pool()     { return pool_; }
    Memory::M&  bindings() { return bindings_; }

    // Process-wide VMA allocator (backs Memory::L1; internally thread-safe).
    VmaAllocator allocator() const { return allocator_; }

    void submitImmediate(std::function<void(VkCommandBuffer)> fn);

    // Process-wide pipeline cache, serialized to per-user app data
    // (<userAppData>/jvk/pipeline.cache). Every vkCreateGraphicsPipelines in
    // jvk feeds through it, so editor opens after the first launch reuse
    // compiled pipelines — on MoltenVK the cache also stores the converted
    // MSL, skipping the SPIR-V→MSL translation entirely. Saved at Device
    // teardown via write-to-temp + atomic rename; the file carries a
    // vendor/device/driver/UUID header and falls back to an empty cache on
    // any mismatch.
    VkPipelineCache pipelineCache() const { return pipelineCache_; }

    ResourceCaches& caches();
    void initCaches();

    // ---- Device-lost latch -------------------------------------------------
    // Set when any submit / wait / acquire reports VK_ERROR_DEVICE_LOST (or a
    // fence wait times out — a wedged driver is indistinguishable from a lost
    // one for our purposes). Once set, every RenderTarget::beginFrame returns
    // a null frame, so the render loop degrades to a no-op instead of a
    // permanent failing-submit spin, an infinite fence wait, or a force-
    // killed worker thread. Sticky for the Device's lifetime — recovery from
    // device loss means recreating the Device.
    bool isLost() const noexcept { return lost_.load(std::memory_order_acquire); }
    void markLost() noexcept
    {
        if (!lost_.exchange(true, std::memory_order_acq_rel))
            DBG("jvk: VK_ERROR_DEVICE_LOST (or fence timeout) — rendering disabled for this Device");
    }

    // Process-wide monotonic clock in seconds, anchored when this Device was
    // constructed. Single source of truth for the `time` push-constant slot
    // every shader pipeline reads — Renderer snapshots it once per frame so
    // all shaders within a frame see an identical value, and ShaderImage
    // samples it per-render so the offscreen path stays in lockstep with the
    // direct Vulkan path when the user toggles between them.
    float time() const noexcept
    {
        return static_cast<float>(juce::Time::getMillisecondCounterHiRes() / 1000.0 - epoch_);
    }

    ~Device();

private:
    Device();
    bool createInstance();
    void setupDebugMessenger();
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandPool();
    void loadPipelineCache();
    void savePipelineCache();
    static juce::File pipelineCacheFile();

    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_     = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    uint32_t         graphicsFamily_ = UINT32_MAX;
    uint32_t         presentFamily_  = UINT32_MAX;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkPipelineCache  pipelineCache_  = VK_NULL_HANDLE;
    VmaAllocator     allocator_      = nullptr;

#if JUCE_DEBUG
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
#endif

    Memory::L1 pool_;
    Memory::M  bindings_;

    std::atomic<bool> lost_ { false };

    std::unique_ptr<ResourceCaches> caches_;

    const double epoch_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
};

} // namespace jvk
