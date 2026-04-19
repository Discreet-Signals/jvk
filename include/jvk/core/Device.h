#pragma once
#include <set>
#if JUCE_MAC || JUCE_LINUX
#include <dlfcn.h>
#endif

namespace jvk {

class ResourceCaches;  // forward — defined in Cache.h

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

    void upload(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height);
    // Buffer-to-buffer staged upload. Queues a copy that flushUploads issues
    // before the render pass begins. Used by path-mesh cache inserts.
    void upload(Memory::L2::Allocation src, VkBuffer dst,
                VkDeviceSize size, VkDeviceSize dstOffset = 0);
    void flushUploads(VkCommandBuffer cmd);

    void submitImmediate(std::function<void(VkCommandBuffer)> fn);

    void retire(Image&& image);
    void retire(Buffer&& buffer);
    void flushRetired(int frameSlot);

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

    struct PendingUpload {
        VkImage      dstImage;
        uint32_t     width, height;
        VkBuffer     srcBuffer;
        VkDeviceSize srcOffset;
    };
    std::vector<PendingUpload> pendingUploads_;

    struct PendingBufferUpload {
        VkBuffer     dstBuffer;
        VkDeviceSize dstOffset;
        VkDeviceSize size;
        VkBuffer     srcBuffer;
        VkDeviceSize srcOffset;
    };
    std::vector<PendingBufferUpload> pendingBufferUploads_;

    static constexpr int MAX_FRAMES = 2;
    struct RetiredResources {
        std::vector<Image>  images;
        std::vector<Buffer> buffers;
    };
    RetiredResources retired_[MAX_FRAMES];

    std::unique_ptr<ResourceCaches> caches_;

    const double epoch_ = juce::Time::getMillisecondCounterHiRes() / 1000.0;
};

} // namespace jvk
