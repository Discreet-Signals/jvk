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
//         staging() (retained for the one-shot createBlackPixel bootstrap;
//         per-frame staging lives on each Renderer), submitImmediate(),
//         caches(), initCaches().
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
    Memory::L2& staging()  { return staging_; }
    Memory::M&  bindings() { return bindings_; }

    void submitImmediate(std::function<void(VkCommandBuffer)> fn);

    ResourceCaches& caches();
    void initCaches();

    VkSampleCountFlagBits maxMSAA() const { return maxMSAA_; }

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

    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_     = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    uint32_t         graphicsFamily_ = UINT32_MAX;
    uint32_t         presentFamily_  = UINT32_MAX;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkSampleCountFlagBits maxMSAA_   = VK_SAMPLE_COUNT_1_BIT;

#if JUCE_DEBUG
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
#endif

    Memory::L1 pool_;
    Memory::L2 staging_;
    Memory::M  bindings_;

    std::unique_ptr<ResourceCaches> caches_;

    const double epoch_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
};

} // namespace jvk
