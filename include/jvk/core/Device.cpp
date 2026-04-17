namespace jvk {

// =============================================================================
// ICD discovery (MoltenVK on macOS)
// =============================================================================

static inline void ensureICDDiscoverable()
{
#if JUCE_MAC
    if (getenv("VK_ICD_FILENAMES") != nullptr) return;
    Dl_info info;
    if (dladdr((void*)ensureICDDiscoverable, &info) && info.dli_fname) {
        juce::File binary(info.dli_fname);
        juce::File contents = binary.getParentDirectory().getParentDirectory();
        juce::File icd = contents.getChildFile("Resources/vulkan/icd.d/MoltenVK_icd.json");
        if (icd.existsAsFile())
            setenv("VK_ICD_FILENAMES", icd.getFullPathName().toRawUTF8(), 1);
    }
#endif
}

// =============================================================================
// Debug callback
// =============================================================================

#if JUCE_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL deviceDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        DBG("Vulkan: " << data->pMessage);
    return VK_FALSE;
}
#endif

// =============================================================================
// Singleton
// =============================================================================

static std::weak_ptr<Device> g_device;

std::shared_ptr<Device> Device::acquire()
{
    auto d = g_device.lock();
    if (d) return d;
    d = std::shared_ptr<Device>(new Device());
    g_device = d;
    return d;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

Device::Device()
{
    if (!createInstance()) return;
    setupDebugMessenger();
    if (!selectPhysicalDevice()) return;
    if (!createLogicalDevice()) return;
    if (!createCommandPool()) return;

    pool_    = Memory::L1(physDevice_, device_);
    staging_ = Memory::L2(physDevice_, device_);
    bindings_ = Memory::M(device_);
}

Device::~Device()
{
    if (device_) vkDeviceWaitIdle(device_);

    caches_.reset();

    for (auto& r : retired_) { r.images.clear(); r.buffers.clear(); }

    // Destroy memory tiers BEFORE the VkDevice — their destructors call vkDestroy*
    bindings_ = Memory::M();
    staging_  = Memory::L2();
    pool_     = Memory::L1();

    if (commandPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);

#if JUCE_DEBUG
    if (debugMessenger_ && instance_) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance_, debugMessenger_, nullptr);
    }
#endif
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

// =============================================================================
// Instance creation
// =============================================================================

bool Device::createInstance()
{
    ensureICDDiscoverable();

    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "jvk";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "jvk";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    std::vector<const char*> extensions = { VK_KHR_SURFACE_EXTENSION_NAME };
#if JUCE_MAC
    extensions.push_back("VK_MVK_macos_surface");
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#elif JUCE_WINDOWS
    extensions.push_back("VK_KHR_win32_surface");
#elif JUCE_LINUX
    extensions.push_back("VK_KHR_xlib_surface");
#endif

#if JUCE_DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
#if JUCE_MAC
    ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

#if JUCE_DEBUG
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
    bool hasValidation = false;
    for (auto& l : layers)
        if (strcmp(l.layerName, validationLayer) == 0) { hasValidation = true; break; }
    if (hasValidation) {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &validationLayer;
    }
#endif

    return vkCreateInstance(&ci, nullptr, &instance_) == VK_SUCCESS;
}

// =============================================================================
// Debug messenger
// =============================================================================

void Device::setupDebugMessenger()
{
#if JUCE_DEBUG
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) return;

    VkDebugUtilsMessengerCreateInfoEXT ci {};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                   | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = deviceDebugCallback;
    fn(instance_, &ci, nullptr, &debugMessenger_);
#endif
}

// =============================================================================
// Physical device selection
// =============================================================================

static int scorePhysicalDevice(VkPhysicalDevice d)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d, &props);
    int score = 0;
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score += 1000; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 100;  break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score += 10;   break;
        default: break;
    }
    score += static_cast<int>(props.limits.maxImageDimension2D / 1000);
    return score;
}

static VkSampleCountFlagBits getMaxSampleCount(VkPhysicalDevice d)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d, &props);
    VkSampleCountFlags c = props.limits.framebufferColorSampleCounts;
    if (c & VK_SAMPLE_COUNT_8_BIT)  return VK_SAMPLE_COUNT_8_BIT;
    if (c & VK_SAMPLE_COUNT_4_BIT)  return VK_SAMPLE_COUNT_4_BIT;
    if (c & VK_SAMPLE_COUNT_2_BIT)  return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

bool Device::selectPhysicalDevice()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return false;

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    int bestScore = -1;
    for (auto& d : devices) {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfs.data());

        uint32_t gf = UINT32_MAX;
        for (uint32_t i = 0; i < qfCount; i++) {
            if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { gf = i; break; }
        }
        if (gf == UINT32_MAX) continue;

        // Check swapchain extension
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());
        bool hasSwapchain = false;
        for (auto& e : exts)
            if (strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) { hasSwapchain = true; break; }
        if (!hasSwapchain) continue;

        int score = scorePhysicalDevice(d);
        if (score > bestScore) {
            bestScore = score;
            physDevice_ = d;
            graphicsFamily_ = gf;
            presentFamily_ = gf; // assume same family — validated at surface creation
        }
    }
    if (physDevice_ == VK_NULL_HANDLE) return false;

    maxMSAA_ = getMaxSampleCount(physDevice_);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice_, &props);
    DBG("jvk::Device: " << props.deviceName);
    return true;
}

// =============================================================================
// Logical device
// =============================================================================

bool Device::createLogicalDevice()
{
    float priority = 1.0f;
    std::set<uint32_t> families = { graphicsFamily_, presentFamily_ };
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (auto f : families) {
        VkDeviceQueueCreateInfo qci {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = f;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        qcis.push_back(qci);
    }

    std::vector<const char*> exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(physDevice_, nullptr, &extCount, available.data());
    for (auto& e : available)
        if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) { exts.push_back("VK_KHR_portability_subset"); break; }

    VkPhysicalDeviceFeatures features {};
    VkDeviceCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    ci.pQueueCreateInfos = qcis.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    ci.pEnabledFeatures = &features;

    if (vkCreateDevice(physDevice_, &ci, nullptr, &device_) != VK_SUCCESS) return false;

    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
    return true;
}

// =============================================================================
// Command pool
// =============================================================================

bool Device::createCommandPool()
{
    VkCommandPoolCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = graphicsFamily_;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    return vkCreateCommandPool(device_, &ci, nullptr, &commandPool_) == VK_SUCCESS;
}

// =============================================================================
// Upload (L2 staging → L1)
// =============================================================================

void Device::upload(Memory::L2::Allocation src, VkImage dst, uint32_t width, uint32_t height)
{
    pendingUploads_.push_back({ dst, width, height, src.buffer, src.offset });
}

void Device::upload(Memory::L2::Allocation src, VkBuffer dst,
                    VkDeviceSize size, VkDeviceSize dstOffset)
{
    pendingBufferUploads_.push_back({ dst, dstOffset, size, src.buffer, src.offset });
}

void Device::flushUploads(VkCommandBuffer cmd)
{
    for (auto& u : pendingUploads_) {
        VkImageMemoryBarrier barrier {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = u.dstImage;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy region {};
        region.bufferOffset = u.srcOffset;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { u.width, u.height, 1 };

        vkCmdCopyBufferToImage(cmd, u.srcBuffer, u.dstImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    pendingUploads_.clear();

    // Buffer-to-buffer copies (path-mesh cache inserts, etc.)
    for (auto& u : pendingBufferUploads_) {
        VkBufferCopy region {};
        region.srcOffset = u.srcOffset;
        region.dstOffset = u.dstOffset;
        region.size      = u.size;
        vkCmdCopyBuffer(cmd, u.srcBuffer, u.dstBuffer, 1, &region);

        // Make the write visible to vertex-input reads (downstream path
        // stencil draws sample this buffer as a vertex attribute source).
        VkBufferMemoryBarrier bb {};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = u.dstBuffer;
        bb.offset = u.dstOffset;
        bb.size   = u.size;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 0, nullptr, 1, &bb, 0, nullptr);
    }
    pendingBufferUploads_.clear();
}

// =============================================================================
// Immediate submission
// =============================================================================

void Device::submitImmediate(std::function<void(VkCommandBuffer)> fn)
{
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    fn(cmd);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue_);

    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}

// =============================================================================
// Deferred destruction
// =============================================================================

void Device::retire(Image&& image) { retired_[0].images.push_back(std::move(image)); }
void Device::retire(Buffer&& buffer) { retired_[0].buffers.push_back(std::move(buffer)); }

void Device::flushRetired(int frameSlot)
{
    auto& r = retired_[frameSlot];
    r.images.clear();
    r.buffers.clear();
}

// =============================================================================
// Caches (initialized after Device construction, needs forward-declared type)
// =============================================================================

ResourceCaches& Device::caches()
{
    jassert(caches_ != nullptr);
    return *caches_;
}

void Device::initCaches()
{
    if (!caches_)
        caches_ = std::make_unique<ResourceCaches>(*this);
}

} // namespace jvk
