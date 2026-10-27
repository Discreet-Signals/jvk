namespace jvk {

// =============================================================================
// Platform probe — read once, before the first Vulkan call.
// =============================================================================

namespace {

struct PlatformProbe
{
    // macOS: running on 10.15, or a Mac-GPU-family-1 device is present (see
    // MetalCapabilities.h). Everything below that keys on "legacy" exists for
    // the 2012-2014 Intel Macs whose Metal drivers were frozen years before
    // the MoltenVK we ship was written.
    bool legacyMetal = false;
    juce::String summary;
};

const PlatformProbe& platformProbe()
{
    static const PlatformProbe probe = []
    {
        PlatformProbe p;
#if JUCE_MAC
        const auto m = core::macos::probeMetal();
        p.legacyMetal = m.legacy();
        p.summary = "macOS " + juce::String(m.osMajor) + "." + juce::String(m.osMinor) + "." + juce::String(m.osPatch)
                  + (p.legacyMetal ? " (legacy Metal machine)" : "")
                  + " | GPUs: " + (m.devices.isEmpty() ? juce::String("none") : m.devices.joinIntoString("; "));
#else
        p.summary = juce::SystemStats::getOperatingSystemName() + " | " + juce::SystemStats::getCpuModel();
#endif
        return p;
    }();
    return probe;
}

#if JUCE_MAC
// MoltenVK configuration for legacy Metal machines.
//
// MoltenVK reads its configuration from MVK_CONFIG_* environment variables the
// first time any Vulkan entry point is called, so this runs before
// vkEnumerateInstanceExtensionProperties in createInstance(). It only acts on
// legacy machines (10.15, or any Mac-GPU-family-1 GPU on any OS); everything
// else keeps MoltenVK's defaults.
//
// Why: the MoltenVK we ship (1.4.0, the last release that loads on 10.15)
// changed two defaults after 10.15-era GPUs stopped being tested, and its own
// maintainers have since found both paths broken on exactly that hardware:
//
//  * MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS defaulted to 1 in 1.2.10 (2024-07),
//    and MVKDevice.mm switches descriptor-set argument buffers on for "older
//    versions of macOS using an Intel GPU" specifically — with Tier-1 argument
//    encoders. MoltenVK PR #2758 (1.4.2, 2026-06) found that path unstable on
//    Intel Iris / Iris Pro and NVIDIA GT 750M — Metal compiler internal errors,
//    "IOAF code 9 / Invalid Resource" GPU faults — and moved combined image
//    samplers, sampled images, storage buffers and storage images back to the
//    discrete-resource path on Mac-family-1 GPUs. Every texture jvk binds is a
//    combined image sampler. 1.4.2 itself requires macOS 12, so on 10.15 the
//    equivalent is to keep every descriptor on the discrete path: the way
//    MoltenVK 1.2.9 — the release upstream used as its known-good reference in
//    that PR — bound resources by default.
//  * MVK_CONFIG_USE_MTLHEAP defaulted to 1 in 1.3.0. MoltenVK gives placement
//    heaps only to Mac2 GPUs, so on family-1 hardware this is already off; it
//    is pinned so a 10.15 machine with a newer GPU allocates the same way
//    1.2.9 did.
//  * MVK_CONFIG_SWITCH_SYSTEM_GPU: on a dual-GPU 2012-2014 MacBook Pro the
//    discrete chip is NVIDIA Kepler. A plugin UI has no business forcing the
//    graphics mux onto it; selectPhysicalDevice prefers the integrated GPU on
//    these machines and this keeps MoltenVK from switching the display to
//    the discrete one should that be the only choice.
//
// A value already present in the environment is respected, so a MoltenVK
// variable set by hand for diagnosis still wins.
void configureMoltenVK(const PlatformProbe& p)
{
    if (! p.legacyMetal)
        return;

    auto pin = [](const char* key, const char* value)
    {
        if (const char* existing = std::getenv(key))
            diag::log(juce::String(key) + " already set to " + existing + " in the environment; left alone");
        else
        {
            setenv(key, value, 0);
            diag::log(juce::String(key) + "=" + value);
        }
    };
    diag::log("legacy Metal machine: pinning MoltenVK to its 1.2.9-era resource binding");
    pin("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0");
    pin("MVK_CONFIG_USE_MTLHEAP", "0");
    pin("MVK_CONFIG_SWITCH_SYSTEM_GPU", "0");
}
#endif

// Loads Vulkan into volk's function pointers, once per process; empty when it
// loaded, otherwise why not. Nothing links Vulkan (see jvk.h), so the binary
// loads on any machine, and no vk* call is made unless this succeeded.
//   Windows/Linux: the system loader the GPU driver installs (vulkan-1.dll,
//                  libvulkan.so.1). We ship none: Khronos strongly discourages
//                  a private copy, and a machine without it has no driver.
//   macOS:         the MoltenVK the bundle carries in Contents/Frameworks (in
//                  Debug the SDK loader beside it, so validation layers load);
//                  a plain executable (a test runner, a tool: no bundle), the
//                  MoltenVK beside itself, which jvk_runtime() puts there.
const juce::String& loadVulkan()
{
    static const juce::String error = []() -> juce::String
    {
#if JUCE_MAC
        Dl_info info {};
        if (dladdr((void*) &loadVulkan, &info) == 0 || info.dli_fname == nullptr)
            return "Vulkan could not be located";
       #if JUCE_DEBUG
        const char* name = "libvulkan.1.dylib";
       #else
        const char* name = "libMoltenVK.dylib";
       #endif
        const auto binary = juce::File(info.dli_fname).getParentDirectory();
        PFN_vkGetInstanceProcAddr gipa = nullptr;
        for (const auto& lib : { binary.getSiblingFile("Frameworks").getChildFile(name), binary.getChildFile("libMoltenVK.dylib") })
        {
            if (! lib.existsAsFile())
                continue;
            void* module = dlopen(lib.getFullPathName().toRawUTF8(), RTLD_NOW | RTLD_LOCAL);
            gipa = module != nullptr ? (PFN_vkGetInstanceProcAddr) dlsym(module, "vkGetInstanceProcAddr") : nullptr;
            if (gipa != nullptr)
                break;
        }
        if (gipa == nullptr)
            return juce::String("Vulkan failed to load (") + name + ")";
        volkInitializeCustom(gipa);
#else
        if (volkInitialize() != VK_SUCCESS)
            return "no Vulkan driver is installed";
#endif
        return {};
    }();
    return error;
}

} // namespace

// =============================================================================
// ICD discovery (MoltenVK on macOS) — Debug only: Release loads the bundled
// MoltenVK directly and ships no loader or manifest (see loadVulkan).
// =============================================================================

static inline void ensureICDDiscoverable()
{
#if JUCE_MAC && JUCE_DEBUG
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
static juce::CriticalSection g_deviceAcquireLock;

std::shared_ptr<Device> Device::acquire()
{
    // Lock around the weak_ptr.lock() / new Device() / weak_ptr = d window so
    // simultaneous acquires from different threads can't each decide "not
    // alive" and construct two Devices (leaking one and splitting every
    // subsequent editor into independent Vulkan state). Today all callers are
    // message-thread (AudioProcessorEditor::acquireVulkan, ShaderImage ctor),
    // but the cost is negligible and keeps the singleton invariant rigorous.
    const juce::ScopedLock lk(g_deviceAcquireLock);
    auto d = g_device.lock();
    // A LOST device never renders again (the latch is sticky): editors still
    // holding it fall back to the JUCE renderer, and the next one to acquire
    // starts a fresh Device instead of inheriting the dead one.
    if (d && !d->isLost()) return d;
    d = std::shared_ptr<Device>(new Device());
    g_device = d;
    return d;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

Device::Device()
{
    const auto& platform = platformProbe();
    diag::log("---- jvk bring-up: " + platform.summary);

#if JUCE_MAC
    configureMoltenVK(platform);                   // before the first vk* call
#endif

    // Each step that fails leaves the Device unusable with its reason; no vk*
    // call follows, and every editor falls back to the JUCE renderer.
    if (const auto& error = loadVulkan(); error.isNotEmpty()) { fail(error); return; }
    if (!createInstance()) { fail("the graphics driver doesn't support Vulkan"); return; }
    diag::log("instance created");
    setupDebugMessenger();
    if (!selectPhysicalDevice()) { fail("no graphics card with Vulkan support was found"); return; }
    if (!createLogicalDevice()) { fail("the Vulkan device could not be created"); return; }
    diag::log("logical device created");
    if (!createCommandPool()) { fail("the Vulkan command pool could not be created"); return; }

    // VMA allocator — created before the memory tiers that view it. API
    // version matches the instance (created for Vulkan 1.0 above), so VMA
    // sticks to core-1.0 entry points, fetched through the two volk loaded.
    {
        VmaVulkanFunctions functions {};
        functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        functions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
        VmaAllocatorCreateInfo aci {};
        aci.instance         = instance_;
        aci.physicalDevice   = physDevice_;
        aci.device           = device_;
        aci.vulkanApiVersion = VK_API_VERSION_1_0;
        aci.pVulkanFunctions = &functions;
        if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS) {
            allocator_ = nullptr;
            fail("Vulkan memory could not be set up");
            return;
        }
    }

    pool_    = Memory::L1(allocator_);
    bindings_ = Memory::M(device_);

    loadPipelineCache();
    diag::log("device ready");
}

// =============================================================================
// Pipeline cache — disk-backed (write-to-temp + atomic rename; own header
// with vendor/device/driver/pipelineCacheUUID so stale or foreign data falls
// back to an empty cache instead of feeding a strict driver garbage).
// =============================================================================

namespace {
struct PipelineCacheFileHeader {
    uint32_t magic;
    uint32_t dataSize;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t driverVersion;
    uint8_t  uuid[VK_UUID_SIZE];
};
constexpr uint32_t kPipelineCacheMagic = 0x4A564B50u; // "JVKP"
} // namespace

juce::File Device::pipelineCacheFile()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("jvk").getChildFile("pipeline.cache");
}

void Device::loadPipelineCache()
{
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physDevice_, &props);

    std::vector<uint8_t> data;
    juce::MemoryBlock blob;
    auto f = pipelineCacheFile();
    if (f.existsAsFile() && f.loadFileAsData(blob)
        && blob.getSize() > sizeof(PipelineCacheFileHeader))
    {
        PipelineCacheFileHeader h {};
        memcpy(&h, blob.getData(), sizeof(h));
        if (h.magic == kPipelineCacheMagic
            && h.dataSize == blob.getSize() - sizeof(h)
            && h.vendorID == props.vendorID
            && h.deviceID == props.deviceID
            && h.driverVersion == props.driverVersion
            && memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) == 0)
        {
            data.resize(h.dataSize);
            memcpy(data.data(),
                   static_cast<const uint8_t*>(blob.getData()) + sizeof(h),
                   h.dataSize);
        }
    }

    VkPipelineCacheCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = data.size();
    ci.pInitialData    = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_) != VK_SUCCESS)
    {
        // Corrupt initial data on a strict driver — retry empty.
        pipelineCache_ = VK_NULL_HANDLE;
        ci.initialDataSize = 0;
        ci.pInitialData    = nullptr;
        if (vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_) != VK_SUCCESS)
            pipelineCache_ = VK_NULL_HANDLE;   // fine — creates just run uncached
    }
}

void Device::savePipelineCache()
{
    if (pipelineCache_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE) return;

    size_t size = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr) != VK_SUCCESS
        || size == 0)
        return;
    std::vector<uint8_t> data(size);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data()) != VK_SUCCESS)
        return;

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physDevice_, &props);

    PipelineCacheFileHeader h {};
    h.magic         = kPipelineCacheMagic;
    h.dataSize      = static_cast<uint32_t>(size);
    h.vendorID      = props.vendorID;
    h.deviceID      = props.deviceID;
    h.driverVersion = props.driverVersion;
    memcpy(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE);

    auto f = pipelineCacheFile();
    f.getParentDirectory().createDirectory();
    auto tmp = f.getSiblingFile(f.getFileName() + ".tmp");
    {
        juce::FileOutputStream out(tmp);
        if (out.failedToOpen()) return;
        out.setPosition(0);
        out.truncate();
        out.write(&h, sizeof(h));
        out.write(data.data(), size);
        out.flush();
    }
    // Atomic replace: last writer wins if two plugin processes race.
    tmp.moveFileTo(f);
}

Device::~Device()
{
    if (device_) {
        // External-sync: idle the device under the process queue lock so a
        // teardown never races another editor's worker inside vkQueueSubmit.
        const juce::ScopedLock queueSync(Renderer::queueLock());
        const juce::ScopedLock computeSync(computeLock_);
        vkDeviceWaitIdle(device_);
    }

    caches_.reset();

    if (pipelineCache_ != VK_NULL_HANDLE) {
        savePipelineCache();
        vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        pipelineCache_ = VK_NULL_HANDLE;
    }

    // Destroy memory tiers BEFORE the VkDevice — their destructors call vkDestroy*
    bindings_ = Memory::M();
    pool_     = Memory::L1();

    // After every allocation owner has torn down. In debug builds VMA
    // asserts on leaked allocations here — a free leak detector.
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
    }

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

static bool hasInstanceExtension(const char* name)
{
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (auto& e : exts)
        if (strcmp(e.extensionName, name) == 0) return true;
    return false;
}

bool Device::createInstance()
{
    ensureICDDiscoverable();

    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "jvk";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "jvk";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // 1.2 where the loader has 1.1 or later (a 1.1 loader takes any version,
    // and the instance runs at the lower of it and the driver's); a 1.0
    // loader takes only 1.0. Rendering makes 1.0's calls either way; compute
    // reads its subgroup properties from 1.1.
    uint32_t loader = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion == nullptr || vkEnumerateInstanceVersion(&loader) != VK_SUCCESS)
        loader = VK_API_VERSION_1_0;
    instanceVersion_ = loader >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_2 : VK_API_VERSION_1_0;
    appInfo.apiVersion = instanceVersion_;

    std::vector<const char*> extensions = { VK_KHR_SURFACE_EXTENSION_NAME };
    VkInstanceCreateFlags flags = 0;
#if JUCE_MAC
    extensions.push_back("VK_MVK_macos_surface");
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    // Portability enumeration is the LOADER's extension; bound directly to
    // MoltenVK (Release) it is absent and MoltenVK enumerates itself regardless.
    if (hasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
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
    ci.flags = flags;
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

    if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) return false;
    volkLoadInstance(instance_);   // every instance and device function, dispatched per handle
    return true;
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
    // A plugin UI is light work. On a legacy dual-GPU Mac (2012-2014 MacBook
    // Pro: Intel + NVIDIA Kepler) the integrated GPU is the one whose driver
    // survives it — the discrete chip and its 2019 driver are what the
    // system's graphics mux would otherwise be forced onto for our sake.
    const bool preferIntegrated = platformProbe().legacyMetal;
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score += preferIntegrated ? 100  : 1000; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += preferIntegrated ? 1000 : 100;  break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score += 10;   break;
        default: break;
    }
    score += static_cast<int>(props.limits.maxImageDimension2D / 1000);
    return score;
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

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDevice_, &props);
    DBG("jvk::Device: " << props.deviceName);
    apiVersion_ = std::min(VK_MAKE_API_VERSION(0, VK_API_VERSION_MAJOR(instanceVersion_), VK_API_VERSION_MINOR(instanceVersion_), 0),
                           VK_MAKE_API_VERSION(0, VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), 0));

    // The compute queue: a family of compute without graphics where the GPU
    // has one (it runs beside graphics), else another family with compute
    // (MoltenVK offers four, a queue each, every one a Metal command queue of
    // its own), else a second queue of the graphics family, else the graphics
    // queue itself.
    {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &qfCount, qfs.data());
        for (uint32_t i = 0; i < qfCount && computeFamily_ == UINT32_MAX; i++)
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qfs[i].queueCount > 0)
                computeFamily_ = i;
        for (uint32_t i = 0; i < qfCount && computeFamily_ == UINT32_MAX; i++)
            if (i != graphicsFamily_ && (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[i].queueCount > 0)
                computeFamily_ = i;
        if (computeFamily_ == UINT32_MAX && (qfs[graphicsFamily_].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            computeFamily_ = graphicsFamily_;
            computeIndex_  = qfs[graphicsFamily_].queueCount > 1 ? 1 : 0;
        }
        juce::String families;
        for (uint32_t i = 0; i < qfCount; i++)
            families << (i > 0 ? ", " : "") << (int) i << ": " << (int) qfs[i].queueCount << " x "
                     << ((qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "G" : "") << ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "C" : "")
                     << ((qfs[i].queueFlags & VK_QUEUE_TRANSFER_BIT) ? "T" : "");
        diag::log("queue families: " + families);
        diag::log(computeFamily_ == UINT32_MAX ? juce::String("compute: none")
                  : computeFamily_ != graphicsFamily_ ? "compute: family " + juce::String((int) computeFamily_) + ", a queue of its own"
                  : computeIndex_ == 1 ? juce::String("compute: a second queue of the graphics family")
                  : juce::String("compute: shares the graphics queue"));
    }

    const char* type = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? "discrete"
                     : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                     : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? "virtual" : "other";
    auto vkVer = [](uint32_t v) { return juce::String(v >> 22) + "." + juce::String((v >> 12) & 0x3ff) + "." + juce::String(v & 0xfff); };
#if JUCE_MAC
    // MoltenVK puts its own version in driverVersion as decimal major*10000 + minor*100 + patch.
    const juce::String driver = "MoltenVK " + juce::String(props.driverVersion / 10000) + "."
                              + juce::String((props.driverVersion / 100) % 100) + "." + juce::String(props.driverVersion % 100);
#else
    const juce::String driver = "driver " + vkVer(props.driverVersion) + " (raw " + juce::String((juce::int64) props.driverVersion) + ")";
#endif
    diag::log("GPU: " + juce::String(props.deviceName) + " (" + type + ", vendor 0x" + juce::String::toHexString((int) props.vendorID)
              + ", device 0x" + juce::String::toHexString((int) props.deviceID) + ", " + driver
              + ", Vulkan " + vkVer(props.apiVersion) + ") of " + juce::String((int) count) + " enumerated");
    return true;
}

// =============================================================================
// Logical device
// =============================================================================

bool Device::createLogicalDevice()
{
    // Graphics first; a compute queue of its own below it, so editors' frames
    // win where the driver weighs priorities.
    const float priorities[] = { 1.0f, 0.5f };
    std::set<uint32_t> families = { graphicsFamily_, presentFamily_ };
    if (computeFamily_ != UINT32_MAX)
        families.insert(computeFamily_);
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (auto f : families) {
        VkDeviceQueueCreateInfo qci {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = f;
        qci.queueCount = f == computeFamily_ ? computeIndex_ + 1 : 1;
        qci.pQueuePriorities = f == computeFamily_ && f != graphicsFamily_ ? priorities + 1 : priorities;
        qcis.push_back(qci);
    }

    std::vector<const char*> exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(physDevice_, nullptr, &extCount, available.data());
    for (auto& e : available)
        if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) { exts.push_back("VK_KHR_portability_subset"); break; }

    // Opt in to the optional features jvk can use, where the GPU has them.
    // independentBlend: per-attachment blend state, which render targets
    // need (built-in draws leave every target untouched while writing the
    // main colour). Every desktop GPU and MoltenVK report it.
    // textureCompressionBC: the BCn PixelFormats, for sampled textures.
    VkPhysicalDeviceFeatures supported {};
    vkGetPhysicalDeviceFeatures(physDevice_, &supported);
    VkPhysicalDeviceFeatures features {};
    features.independentBlend = supported.independentBlend;
    independentBlend_ = supported.independentBlend == VK_TRUE;
    features.textureCompressionBC = supported.textureCompressionBC;
    textureCompressionBC_ = supported.textureCompressionBC == VK_TRUE;

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
    if (computeFamily_ != UINT32_MAX)
        vkGetDeviceQueue(device_, computeFamily_, computeIndex_, &computeQueue_);
    return true;
}

juce::CriticalSection& Device::computeQueueLock()
{
    return computeQueueShared() ? Renderer::queueLock() : computeLock_;
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
// Immediate submission
// =============================================================================

void Device::submitImmediate(std::function<void(VkCommandBuffer)> fn)
{
    // Serialize the entire body under the process-wide queue lock so:
    //   - vkAllocateCommandBuffers / vkFreeCommandBuffers on the shared
    //     Device commandPool_ don't race another thread using the same pool
    //     (Vulkan requires command pools to be externally synchronized).
    //   - vkQueueSubmit + vkQueueWaitIdle don't race a Renderer worker's
    //     queue submit (external sync on VkQueue per §4.2.1).
    // Today this is only called from ResourceCaches::createBlackPixel during
    // single-threaded setup, but the lock makes it safe for any future
    // caller to invoke from a worker thread too.
    const juce::ScopedLock queueSync(Renderer::queueLock());

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
// Caches (initialized after Device construction, needs forward-declared type)
// =============================================================================

ResourceCaches& Device::caches()
{
    // caches_ is set once (under g_deviceAcquireLock via initCaches) and
    // never mutated again; safe to return the dereference without locking.
    jassert(caches_ != nullptr);
    return *caches_;
}

void Device::initCaches()
{
    // Idempotent + thread-safe: share the same lock as acquire() so the
    // initial Device construction and its caches construction can't race
    // across simultaneous editor acquisitions.
    const juce::ScopedLock lk(g_deviceAcquireLock);
    if (!caches_)
        caches_ = std::make_unique<ResourceCaches>(*this);
}

} // namespace jvk
