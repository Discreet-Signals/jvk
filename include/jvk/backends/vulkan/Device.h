#pragma once
#include <set>
#if JUCE_MAC || JUCE_LINUX
#include <dlfcn.h>
#endif

namespace jvk {

// The Vulkan instance, device, queue and memory allocators. Owned by
// vulkan::Backend (VulkanBackend.h), which is the process-wide shared object;
// nothing else makes a Device.
//
// Multi-instance thread contract (every member below falls into exactly one
// category):
//
//   (a) Immutable after init — safe to read from any thread:
//         instance(), physicalDevice(), device(), graphicsQueue(),
//         graphicsFamily(), presentQueue(), presentFamily(), commandPool()
//         (commandPool is only used by submitImmediate and the backend's own
//         one-off submissions, under queueLock; each surface has its own
//         pool), time(), apiVersion(), computeQueue(), computeFamily(),
//         computeQueueShared().
//
//   (b) Internally synchronized — safe to call from any thread, including
//       concurrently from multiple editors' workers:
//         pool() (Memory::L1), bindings() (Memory::M).
//
//   (c) The queues: every vkQueueSubmit, vkQueuePresentKHR and
//       vkQueueWaitIdle in the process holds queueLock() (Vulkan requires
//       external synchronisation of a VkQueue), or computeQueueLock() for the
//       compute queue. vkDeviceWaitIdle holds both (queueLock first). Both
//       reentrant; one lock when compute shares the graphics queue.
class Device {
public:
    Device();

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

    // The one lock around queue use (see (c) above).
    juce::CriticalSection& queueLock() const noexcept { return queueLock_; }

    // The Vulkan version in use: the lower of the instance's (1.2 where the
    // loader has 1.1 or later, else 1.0) and the GPU's. Rendering uses 1.0's
    // calls whatever it is; compute reads subgroup properties from 1.1.
    uint32_t apiVersion() const { return apiVersion_; }

    // The queue compute streams submit to (COMPUTE.md, "The queues"): a
    // compute family without graphics where the GPU has one (async compute:
    // NVIDIA's and AMD's run it beside the editors' frames), else another
    // family with compute (MoltenVK's), else a second queue of the graphics
    // family, else the graphics queue itself (computeQueueShared()). Null when
    // the GPU has no compute at all. Submit to it only under
    // computeQueueLock(), which is queueLock() when the queue is shared.
    VkQueue  computeQueue()       const { return computeQueue_; }
    uint32_t computeFamily()      const { return computeFamily_; }
    bool     computeQueueShared() const { return computeQueue_ != VK_NULL_HANDLE && computeQueue_ == graphicsQueue_; }
    juce::CriticalSection& computeQueueLock() const noexcept { return computeQueueShared() ? queueLock_ : computeLock_; }
    // Whether the graphics family runs compute too: frames can dispatch.
    bool graphicsComputes() const { return graphicsComputes_; }

    // vkDeviceWaitIdle under both queue locks (queueLock first): every
    // queue is idle after. Any thread; reentrant with a lock already held.
    void waitIdle() const
    {
        const juce::ScopedLock queueSync(queueLock_);
        const juce::ScopedLock computeSync(computeLock_);
        if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
    }
    // The graphics (and present) queue idle, under queueLock: every frame
    // done, a compute stream's own queue left running (a resize never waits
    // for a long kernel). When compute shares the queue, it waits for it too.
    void waitGraphicsIdle() const
    {
        const juce::ScopedLock queueSync(queueLock_);
        if (graphicsQueue_ != VK_NULL_HANDLE) vkQueueWaitIdle(graphicsQueue_);
        if (presentQueue_ != VK_NULL_HANDLE && presentQueue_ != graphicsQueue_) vkQueueWaitIdle(presentQueue_);
    }

    // Whether this Device came up, and why not when it didn't (no Vulkan, no
    // driver, no GPU, ...). An unusable Device makes no Vulkan call; editors
    // fall back to the JUCE renderer with this reason.
    bool usable() const noexcept { return unavailableReason_.isEmpty(); }
    const juce::String& unavailableReason() const noexcept { return unavailableReason_; }

    // Optional features enabled at device creation (when the GPU has them).
    // Per-attachment blend state; render targets need it.
    bool supportsIndependentBlend() const noexcept { return independentBlend_; }
    // The block-compressed PixelFormats (BCn) for sampled textures.
    bool supportsTextureCompressionBC() const noexcept { return textureCompressionBC_; }

    // ---- Device-lost latch -------------------------------------------------
    // Set when any submit / wait / acquire reports VK_ERROR_DEVICE_LOST (or a
    // fence wait times out — a wedged driver is indistinguishable from a lost
    // one for our purposes). Once set, every surface's beginFrame returns
    // false, so the render loop degrades to a no-op instead of a
    // permanent failing-submit spin, an infinite fence wait, or a force-
    // killed worker thread. Sticky for the Device's lifetime — recovery from
    // device loss means recreating the Device.
    bool isLost() const noexcept { return lost_.load(std::memory_order_acquire); }
    void markLost() noexcept
    {
        if (!lost_.exchange(true, std::memory_order_acq_rel))
        {
            DBG("jvk: VK_ERROR_DEVICE_LOST (or fence timeout) — rendering disabled for this Device");
            try { diag::log("VK_ERROR_DEVICE_LOST or fence timeout: rendering disabled for this Device"); } catch (...) {}
        }
    }

    // Monotonic clock in seconds, anchored when this Device was constructed.
    float time() const noexcept
    {
        return static_cast<float>(juce::Time::getMillisecondCounterHiRes() / 1000.0 - epoch_);
    }

    ~Device();

private:
    void fail(const juce::String& reason) { unavailableReason_ = reason; diag::log(reason + ": JUCE renderer"); }
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
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    uint32_t         computeFamily_  = UINT32_MAX;
    uint32_t         computeIndex_   = 0;     // the queue's index in its family
    bool             graphicsComputes_ = false;
    uint32_t         instanceVersion_ = VK_API_VERSION_1_0;
    uint32_t         apiVersion_      = VK_API_VERSION_1_0;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkPipelineCache  pipelineCache_  = VK_NULL_HANDLE;
    VmaAllocator     allocator_      = nullptr;

#if JUCE_DEBUG
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
#endif

    Memory::L1 pool_;
    Memory::M  bindings_;

    std::atomic<bool> lost_ { false };
    juce::String unavailableReason_;
    bool independentBlend_ = false;
    bool textureCompressionBC_ = false;

    mutable juce::CriticalSection queueLock_;
    mutable juce::CriticalSection computeLock_;   // the compute queue's, when it is its own

    const double epoch_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
};

} // namespace jvk
