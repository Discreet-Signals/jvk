/*
  ==============================================================================

    VulkanInstance.cpp
    Created: 11 Oct 2023 1:53:22pm
    Author:  Gavin

  ==============================================================================
*/

#include "VulkanInstance.h"

namespace jvk::core
{

// =============================================================================
// Debug messenger callback (validation layers)
// =============================================================================

#if JUCE_DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        DBG("Vulkan Validation: " << pCallbackData->pMessage);
    return VK_FALSE;
}
#endif

// =============================================================================
// Initialization
// =============================================================================

void VulkanInstance::initializeVulkan()
{
    if (!createInstance())         { status = VulkanStatus::NoDriver; return; }
    setupDebugMessenger();
    createSurface();
    if (!selectPhysicalDevice())   { status = VulkanStatus::NoSuitableDevice; return; }
    if (!createLogicalDevice())    { status = VulkanStatus::InitFailed; return; }
    if (!createCommandPool())      { status = VulkanStatus::InitFailed; return; }
    if (!createSwapChain())        { status = VulkanStatus::InitFailed; return; }
    if (!createRenderPass())       { status = VulkanStatus::InitFailed; return; }
    if (!createGraphicsPipeline()) { status = VulkanStatus::InitFailed; return; }
    if (!createCommandBuffers())   { status = VulkanStatus::InitFailed; return; }
    if (!createSyncObjects())      { status = VulkanStatus::InitFailed; return; }
    status = VulkanStatus::Ready;
}

// =============================================================================
// Instance creation
// =============================================================================

bool VulkanInstance::createInstance()
{
    DBG("Creating Vulkan Instance...");
    ensureICDDiscoverable();

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "JUCE Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "jvk";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    std::vector<const char*> extensions = getExtensions();

#if JUCE_DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
#if JUCE_MAC
    createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Enable validation layers in debug builds
#if JUCE_DEBUG
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    bool validationAvailable = false;
    for (const auto& layer : availableLayers)
    {
        if (strcmp(layer.layerName, validationLayer) == 0)
        {
            validationAvailable = true;
            break;
        }
    }

    if (validationAvailable)
    {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &validationLayer;
        DBG("Vulkan validation layers enabled");
    }
    else
    {
        DBG("Vulkan validation layers not available");
    }
#endif

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS)
    {
        DBG("Failed to Create Vulkan Instance: " << result);
        return false;
    }

    DBG("Created Vulkan Instance!");
    return true;
}

// =============================================================================
// Debug messenger
// =============================================================================

void VulkanInstance::setupDebugMessenger()
{
#if JUCE_DEBUG
    auto createFunc = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!createFunc) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = vulkanDebugCallback;

    createFunc(instance, &createInfo, nullptr, &debugMessenger);
#endif
}

// =============================================================================
// Device selection
// =============================================================================

bool VulkanInstance::isDeviceSuitable(VkPhysicalDevice device,
                                      uint32_t& outGraphicsFamily,
                                      uint32_t& outPresentFamily)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    bool graphicsFound = false, presentFound = false;
    for (uint32_t i = 0; i < queueFamilyCount; i++)
    {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            outGraphicsFamily = i;
            graphicsFound = true;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport)
        {
            outPresentFamily = i;
            presentFound = true;
        }
        if (graphicsFound && presentFound)
            break;
    }

    // Check for swapchain support
    if (graphicsFound && presentFound)
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());
        for (const auto& ext : exts)
            if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
                return true;
    }

    return false;
}

static int scoreDevice(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    int score = 0;
    switch (props.deviceType)
    {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score += 1000; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 100;  break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score += 10;   break;
        default: break;
    }
    score += static_cast<int>(props.limits.maxImageDimension2D / 1000);
    return score;
}

static VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
    if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
    if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
    if (counts & VK_SAMPLE_COUNT_8_BIT)  return VK_SAMPLE_COUNT_8_BIT;
    if (counts & VK_SAMPLE_COUNT_4_BIT)  return VK_SAMPLE_COUNT_4_BIT;
    if (counts & VK_SAMPLE_COUNT_2_BIT)  return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

bool VulkanInstance::selectPhysicalDevice()
{
    DBG("Selecting GPU...");
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        DBG("No GPUs with Vulkan support found!");
        return false;
    }

    std::vector<VkPhysicalDevice> availableDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, availableDevices.data());

    int bestScore = -1;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    uint32_t bestGraphicsFamily = UINT32_MAX, bestPresentFamily = UINT32_MAX;

    for (const auto& device : availableDevices)
    {
        uint32_t gf = UINT32_MAX, pf = UINT32_MAX;
        if (isDeviceSuitable(device, gf, pf))
        {
            int score = scoreDevice(device);
            if (score > bestScore)
            {
                bestScore = score;
                bestDevice = device;
                bestGraphicsFamily = gf;
                bestPresentFamily = pf;
            }
        }
    }

    if (bestDevice == VK_NULL_HANDLE)
    {
        DBG("No suitable GPU found!");
        return false;
    }

    physicalDevice = bestDevice;
    graphicsQueueFamilyIndex = bestGraphicsFamily;
    presentQueueFamilyIndex = bestPresentFamily;

    // Query MSAA capabilities and clamp settings
    maxSupportedMSAA = getMaxUsableSampleCount(physicalDevice);
    if (settings.msaaSamples > maxSupportedMSAA)
        settings.msaaSamples = maxSupportedMSAA;

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    DBG("Selected GPU: " << deviceProperties.deviceName
        << " (max MSAA: " << maxSupportedMSAA << "x)");

    return true;
}

// =============================================================================
// Logical device creation
// =============================================================================

bool VulkanInstance::createLogicalDevice()
{
    DBG("Creating Logical Device...");
    float queuePriority = 1.0f;

    // Create queues for all unique queue families
    std::set<uint32_t> uniqueFamilies = { graphicsQueueFamilyIndex, presentQueueFamilyIndex };
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t family : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // Enable VK_KHR_portability_subset if available (required by MoltenVK)
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availableExts.data());
    for (const auto& ext : availableExts)
    {
        if (strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0)
        {
            deviceExtensions.push_back("VK_KHR_portability_subset");
            break;
        }
    }

    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
    {
        DBG("Failed to create logical device!");
        return false;
    }

    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueFamilyIndex, 0, &presentQueue);

    DBG("Created Logical Device!");
    return true;
}

// =============================================================================
// Swapchain
// =============================================================================

bool VulkanInstance::createSwapChain()
{
    SwapChainInfo info;
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.surface = surface;
    info.size = { width.load(), height.load() };
    info.graphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
    info.presentQueueFamilyIndex = presentQueueFamilyIndex;
    info.renderPass = renderPass;
    info.msaaSamples = settings.msaaSamples;
    info.preferredPresentMode = settings.presentMode;
    info.useSRGBFormat = settings.srgbPipeline;

    // Create new swapchain referencing old for seamless transition
    VkSwapchainKHR previous = swapChain ? swapChain->getInternal() : VK_NULL_HANDLE;
    auto next = std::make_unique<SwapChain>(info, previous);
    if (!next->getInternal())
    {
        DBG("Failed to create swap chain!");
        return false;
    }

    // Swap in the new swapchain (old is destroyed, driver handles transition)
    swapChain = std::move(next);

    // Reallocate command buffers if image count changed
    int newImageCount = swapChain->getNumImages();
    if (static_cast<int>(commandBuffers.size()) != newImageCount)
    {
        if (!commandBuffers.empty())
        {
            vkFreeCommandBuffers(device, commandPool,
                static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
            commandBuffers.clear();
        }
        if (renderPass)
            createCommandBuffers();
    }

    // Reset per-image fence tracking
    imagesInFlight.assign(newImageCount, VK_NULL_HANDLE);

    viewport.width = (float) swapChain->getWidth();
    viewport.height = (float) swapChain->getHeight();
    scissor.extent = swapChain->getExtent();
    return true;
}

// =============================================================================
// Render pass
// =============================================================================

bool VulkanInstance::createRenderPass()
{
    DBG("Creating Render Pass...");

    // Attachment 0: MSAA color
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChain->getFormat();
    colorAttachment.samples = settings.msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorAttachmentRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    // Attachment 1: Resolve (1x)
    VkAttachmentDescription resolveAttachment = {};
    resolveAttachment.format = swapChain->getFormat();
    resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference resolveAttachmentRef = { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    // Attachment 2: Depth
    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = settings.msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef = { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    // Input attachment ref — allows fragment shader to read the MSAA color attachment
    VkAttachmentReference inputAttachmentRef = { 0, VK_IMAGE_LAYOUT_GENERAL };

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pResolveAttachments = &resolveAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.inputAttachmentCount = 1;
    subpass.pInputAttachments = &inputAttachmentRef;

    // Self-dependency: allows reading the color attachment we're writing to
    // (for post-processing effects like HSV, blur that need framebuffer read)
    VkSubpassDependency selfDep = {};
    selfDep.srcSubpass = 0;
    selfDep.dstSubpass = 0;
    selfDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    selfDep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    selfDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    selfDep.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    selfDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkAttachmentDescription attachments[3] = { colorAttachment, resolveAttachment, depthAttachment };
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 3;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &selfDep;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
    {
        DBG("Failed to Create Render Pass!");
        return false;
    }

    swapChain->createFrameBuffers(renderPass);
    DBG("Created Render Pass!");
    return true;
}

// =============================================================================
// Graphics pipeline
// =============================================================================

bool VulkanInstance::createGraphicsPipeline()
{
    shaders::ShaderGroup defaultShaderGroup;
    defaultShaderGroup.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                 shaders::defaults::basic_vert_spv,
                                 shaders::defaults::basic_vert_spvSize);
    defaultShaderGroup.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                 shaders::defaults::basic_frag_spv,
                                 shaders::defaults::basic_frag_spvSize);
    defaultPipeline = std::make_unique<Pipeline>(device, renderPass,
        std::move(defaultShaderGroup), settings.msaaSamples);
    return defaultPipeline != nullptr;
}

// =============================================================================
// Command pool and buffers
// =============================================================================

bool VulkanInstance::createCommandPool()
{
    DBG("Creating Command Pool...");
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        DBG("Failed to Create Command Pool!");
        return false;
    }

    DBG("Created Command Pool!");
    return true;
}

bool VulkanInstance::createCommandBuffers()
{
    commandBuffers.resize(swapChain->getNumImages());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
    {
        DBG("Failed to allocate command buffers!");
        return false;
    }
    return true;
}

// =============================================================================
// Synchronization
// =============================================================================

bool VulkanInstance::createSyncObjects()
{
    DBG("Creating Synchronization Objects...");
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChain->getNumImages(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {
            DBG("Failed to Create Synchronization Objects!");
            return false;
        }
    }

    DBG("Created Synchronization Objects!");
    return true;
}

// =============================================================================
// Rebuild (for runtime settings changes)
// =============================================================================

void VulkanInstance::rebuildSwapchainAndPipelines()
{
    vkDeviceWaitIdle(device);

    for (size_t i = 0; i < commandBuffers.size(); i++)
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffers[i]);
    commandBuffers.clear();

    defaultPipeline.reset();

    if (renderPass)
    {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }

    swapChain.reset();

    createSwapChain();
    createRenderPass();
    createGraphicsPipeline();
    createCommandBuffers();
}

// =============================================================================
// Rendering
// =============================================================================

void VulkanInstance::submitCommandBuffer(int i)
{
    vkResetCommandBuffer(commandBuffers[i], 0);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
        DBG("Failed to begin recording command buffer!");

    vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);
    vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChain->getFrameBuffer(i);
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChain->getExtent();

    std::array<VkClearValue, 3> clearValues = {};
    clearValues[0].color = {{clearColor.getFloatRed(), clearColor.getFloatGreen(), clearColor.getFloatBlue(), clearColor.getFloatAlpha()}};
    clearValues[1].color = clearValues[0].color;
    clearValues[2].depthStencil = { 1.0f, 0 };

    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, defaultPipeline->getInternal());

    renderComponents(commandBuffers[i]);

    vkCmdEndRenderPass(commandBuffers[i]);

    if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
        DBG("Failed to Edit Command Buffer " << i << "!");
}

void VulkanInstance::checkForResize()
{
    // Just consume the flag — actual recreation happens in execute()
    // when we detect OUT_OF_DATE or the flag is set
    if (!windowResized.load())
        return;
    if (width.load() <= 0 || height.load() <= 0)
        return;

    windowResized.store(false);
    needsSwapchainRecreation = true;
}

void VulkanInstance::execute()
{
    // Wait only on this frame slot's fence (not the entire device)
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Recreate swapchain if needed — safe here because this frame slot is idle
    if (needsSwapchainRecreation)
    {
        createSwapChain();
        needsSwapchainRecreation = false;
    }

    // Acquire next image
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain->getInternal(),
        UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        createSwapChain();
        return; // Skip this frame, next frame uses new swapchain
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        DBG("failed to acquire swap chain image!");
        return;
    }
    if (imageIndex >= commandBuffers.size() || imageIndex >= imagesInFlight.size())
        return;

    // Wait on the fence for THIS image if it's still in flight
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);

    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // Record command buffer
    submitCommandBuffer(static_cast<int>(imageIndex));

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        DBG("failed to submit draw command buffer!");

    // Present
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapChain->getInternal() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        needsSwapchainRecreation = true; // Recreate next frame, don't stall now

    totalFrames++;
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// =============================================================================
// Cleanup
// =============================================================================

void VulkanInstance::release()
{
    if (device)
        vkDeviceWaitIdle(device);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (device)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
    }
    for (size_t i = 0; i < commandBuffers.size(); i++)
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffers[i]);

    swapChain.reset();

    if (device) vkDestroyCommandPool(device, commandPool, nullptr);

    defaultPipeline.reset();
    if (device && renderPass) vkDestroyRenderPass(device, renderPass, nullptr);

    if (device) vkDestroyDevice(device, nullptr);

#if JUCE_DEBUG
    if (debugMessenger && instance)
    {
        auto destroyFunc = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroyFunc)
            destroyFunc(instance, debugMessenger, nullptr);
    }
#endif

    if (surface && instance) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
}

} // jvk::core
