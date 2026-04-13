/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC
 
 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝
 
 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 
 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: VulkanInstance.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include <set>
#if JUCE_MAC || JUCE_LINUX
#include <dlfcn.h>
#endif

namespace jvk::core
{

// Ensures the Vulkan loader can find MoltenVK when running as a plugin
// inside a host DAW. The loader searches the main app bundle for ICD
// manifests, but plugins bundle MoltenVK in their own .vst3/.component.
// This resolves the plugin binary's path via dladdr and sets VK_ICD_FILENAMES
// to point to the bundled ICD JSON.
static inline void ensureICDDiscoverable()
{
#if JUCE_MAC
    // If already set (e.g. by setup-env.sh), don't override
    if (getenv("VK_ICD_FILENAMES") != nullptr)
        return;

    Dl_info info;
    if (dladdr((void*)ensureICDDiscoverable, &info) && info.dli_fname)
    {
        // info.dli_fname = .../Contents/MacOS/BinaryName
        juce::File binary(info.dli_fname);
        juce::File contentsDir = binary.getParentDirectory().getParentDirectory();
        juce::File icdFile = contentsDir.getChildFile("Resources/vulkan/icd.d/MoltenVK_icd.json");

        if (icdFile.existsAsFile())
        {
            setenv("VK_ICD_FILENAMES", icdFile.getFullPathName().toRawUTF8(), 1);
            DBG("jvk: Set VK_ICD_FILENAMES=" << icdFile.getFullPathName());
        }
    }
#endif
}

enum class VulkanStatus
{
    Uninitialized,
    Ready,
    NoDriver,
    NoSuitableDevice,
    InitFailed,
    DeviceLost
};

struct VulkanRendererSettings
{
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_4_BIT;
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    // Frame rate is controlled by present mode (FIFO = vsync, IMMEDIATE = uncapped)
    bool srgbPipeline = false; // true = physically correct blending, false = JUCE-identical
    // When set, swapchain is created at this fixed size and never recreated
    // on resize — only viewport/scissor change.  Set to 0 for normal mode.
    int maxWidth = 0;
    int maxHeight = 0;
};

class VulkanInstance
{
public:
    VulkanInstance() { }
    virtual ~VulkanInstance() { }

protected:
    virtual void prepareComponents(VkCommandBuffer& commandBuffer) { }
    virtual void renderComponents(VkCommandBuffer& commandBuffer) { }

    virtual std::vector<const char*> getExtensions() { return {}; }
    virtual void createSurface() { }

    void initializeVulkan();
    bool createInstance();
    void setupDebugMessenger();
    bool isDeviceSuitable(VkPhysicalDevice device, uint32_t& outGraphicsFamily, uint32_t& outPresentFamily);
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandPool();
    bool createSwapChain();
    bool createRenderPass();
    bool createGraphicsPipeline();
    bool createCommandBuffers();
    bool createSyncObjects();
    void rebuildSwapchainAndPipelines();
    void submitCommandBuffer(int index);
    void checkForResize();
    void execute();
    void release();

    // Vulkan core components
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    // Swapchain components
    std::unique_ptr<SwapChain> swapChain;

    // Graphics pipeline components
    VkRenderPass renderPass;
    std::unique_ptr<Pipeline> defaultPipeline;
    VkViewport viewport = {};
    VkRect2D scissor = {};

    // Drawing components
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    // Synchronization components
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    const int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<VkFence> imagesInFlight;

    // Queues
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    uint32_t presentQueueFamilyIndex = UINT32_MAX;

    // Status and settings
    VulkanStatus status { VulkanStatus::Uninitialized };
    VulkanRendererSettings settings;
    VkSampleCountFlagBits maxSupportedMSAA { VK_SAMPLE_COUNT_1_BIT };

    // Other required members
    juce::Colour clearColor { juce::Colours::black };
    size_t totalFrames = 0;
    size_t currentFrame = 0;
    std::atomic<int> width { 800 };
    std::atomic<int> height { 600 };
    std::atomic<bool> windowResized { false };
    bool needsSwapchainRecreation = false;

};

} // jvk::core
