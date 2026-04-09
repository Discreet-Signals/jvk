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
 File: ShaderImage.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "ShaderImage.h"

namespace jvk
{

// =============================================================================
// Shared Vulkan context (singleton, reference-counted)
// =============================================================================

struct ShaderImage::Context
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    std::atomic<int> refCount { 0 };

    bool init()
    {
        core::ensureICDDiscoverable();

        // --- Instance ---
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "jvk ShaderImage";
        appInfo.apiVersion = VK_API_VERSION_1_0;

        std::vector<const char*> extensions;
#if JUCE_MAC
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
#if JUCE_DEBUG
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

        VkInstanceCreateInfo instInfo = {};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo = &appInfo;
#if JUCE_MAC
        instInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        instInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instInfo.ppEnabledExtensionNames = extensions.data();

#if JUCE_DEBUG
        const char* validationLayer = "VK_LAYER_KHRONOS_validation";
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        bool hasValidation = false;
        for (const auto& l : layers)
            if (strcmp(l.layerName, validationLayer) == 0) { hasValidation = true; break; }
        if (hasValidation)
        {
            instInfo.enabledLayerCount = 1;
            instInfo.ppEnabledLayerNames = &validationLayer;
        }
#endif

        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS)
            return false;

        // --- Physical device ---
        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
        if (devCount == 0) return false;

        std::vector<VkPhysicalDevice> devices(devCount);
        vkEnumeratePhysicalDevices(instance, &devCount, devices.data());

        // Pick first device with a graphics queue
        for (auto dev : devices)
        {
            uint32_t qfCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qfCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfs.data());

            for (uint32_t i = 0; i < qfCount; i++)
            {
                if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    physicalDevice = dev;
                    graphicsQueueFamilyIndex = i;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE) return false;

        // --- Logical device ---
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = graphicsQueueFamilyIndex;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        std::vector<const char*> devExts;
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availExts(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availExts.data());
        for (const auto& ext : availExts)
            if (strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0)
            { devExts.push_back("VK_KHR_portability_subset"); break; }

        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
        dci.ppEnabledExtensionNames = devExts.data();

        if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS)
            return false;

        vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);

        // --- Command pool ---
        VkCommandPoolCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = graphicsQueueFamilyIndex;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device, &cpci, nullptr, &commandPool) != VK_SUCCESS)
            return false;

        DBG("ShaderImage: Vulkan context initialized");
        return true;
    }

    void destroy()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance)
            vkDestroyInstance(instance, nullptr);
    }
};

ShaderImage::Context& ShaderImage::getContext()
{
    static Context ctx;
    return ctx;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

ShaderImage::ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                         int width, int height, int storageBufSize)
    : w(width), h(height), storageSize(storageBufSize),
      fragShaderCode(fragmentSpv, fragmentSpv + fragmentSpvSize),
      startTime(juce::Time::getMillisecondCounterHiRes() / 1000.0)
{
    auto& ctx = getContext();
    if (ctx.refCount.fetch_add(1) == 0)
    {
        if (!ctx.init())
        {
            DBG("ShaderImage: Failed to initialize Vulkan context");
            return;
        }
    }

    if (storageBufSize > 0)
    {
        fifo = std::make_unique<FIFO<float>>(storageBufSize * 3, 0.0f);
        storageData.resize(storageBufSize, 0.0f);
    }

    displayImage = juce::Image(juce::Image::ARGB, w, h, true);

    createRenderTarget();
    createStorageBuffer();
    createPipeline();

    // Allocate double-buffered command buffers
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx.commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    vkAllocateCommandBuffers(ctx.device, &cbai, commandBuffer);

    // Double-buffered fences (start signaled so first wait is a no-op)
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx.device, &fci, nullptr, &fence[0]);
    vkCreateFence(ctx.device, &fci, nullptr, &fence[1]);

    ready = true;
}

ShaderImage::~ShaderImage()
{
    auto& ctx = getContext();

    if (ctx.device)
    {
        vkWaitForFences(ctx.device, 2, fence, VK_TRUE, UINT64_MAX);
        for (int i = 0; i < 2; i++)
            if (fence[i]) vkDestroyFence(ctx.device, fence[i], nullptr);
    }

    if (ctx.device && (commandBuffer[0] || commandBuffer[1]))
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 2, commandBuffer);

    destroyPipeline();
    destroyStorageBuffer();
    destroyRenderTarget();

    if (ctx.refCount.fetch_sub(1) == 1)
        ctx.destroy();
}

// =============================================================================
// Render target (offscreen image + staging buffer)
// =============================================================================

void ShaderImage::createRenderTarget()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

    // --- Color image ---
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(dev, &ici, nullptr, &colorImage);

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(dev, colorImage, &memReq);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(dev, &mai, nullptr, &colorImageMemory);
    vkBindImageMemory(dev, colorImage, colorImageMemory, 0);

    // --- Image view ---
    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = colorImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &colorImageView);

    // --- Render pass (single color attachment, no MSAA) ---
    VkAttachmentDescription att = {};
    att.format = VK_FORMAT_B8G8R8A8_UNORM;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    VkRenderPassCreateInfo rpci = {};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    vkCreateRenderPass(dev, &rpci, nullptr, &renderPass);

    // --- Framebuffer ---
    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &colorImageView;
    fbci.width = w;
    fbci.height = h;
    fbci.layers = 1;
    vkCreateFramebuffer(dev, &fbci, nullptr, &framebuffer);

    // --- Double-buffered staging buffers (persistently mapped) ---
    VkDeviceSize stagingSize = (VkDeviceSize)w * h * 4;
    for (int i = 0; i < 2; i++)
    {
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = stagingSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &stagingBuffer[i]);

        VkMemoryRequirements bufReq;
        vkGetBufferMemoryRequirements(dev, stagingBuffer[i], &bufReq);
        VkMemoryAllocateInfo bmai = {};
        bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bmai.allocationSize = bufReq.size;
        bmai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, bufReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &bmai, nullptr, &stagingMemory[i]);
        vkBindBufferMemory(dev, stagingBuffer[i], stagingMemory[i], 0);
        vkMapMemory(dev, stagingMemory[i], 0, stagingSize, 0, &mappedStagingPtr[i]);
        memset(mappedStagingPtr[i], 0, stagingSize);
    }
}

void ShaderImage::destroyRenderTarget()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;
    if (!dev) return;

    for (int i = 0; i < 2; i++)
    {
        if (mappedStagingPtr[i]) { vkUnmapMemory(dev, stagingMemory[i]); mappedStagingPtr[i] = nullptr; }
        if (stagingBuffer[i])    { vkDestroyBuffer(dev, stagingBuffer[i], nullptr); stagingBuffer[i] = VK_NULL_HANDLE; }
        if (stagingMemory[i])    { vkFreeMemory(dev, stagingMemory[i], nullptr); stagingMemory[i] = VK_NULL_HANDLE; }
    }
    if (framebuffer)      { vkDestroyFramebuffer(dev, framebuffer, nullptr); framebuffer = VK_NULL_HANDLE; }
    if (renderPass)       { vkDestroyRenderPass(dev, renderPass, nullptr); renderPass = VK_NULL_HANDLE; }
    if (colorImageView)   { vkDestroyImageView(dev, colorImageView, nullptr); colorImageView = VK_NULL_HANDLE; }
    if (colorImage)       { vkDestroyImage(dev, colorImage, nullptr); colorImage = VK_NULL_HANDLE; }
    if (colorImageMemory) { vkFreeMemory(dev, colorImageMemory, nullptr); colorImageMemory = VK_NULL_HANDLE; }
}

// =============================================================================
// Storage buffer
// =============================================================================

void ShaderImage::createStorageBuffer()
{
    if (storageSize <= 0) return;
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

    VkDeviceSize bufSize = sizeof(float) * storageSize;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bufSize;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkCreateBuffer(dev, &bci, nullptr, &storageBuffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, storageBuffer, &req);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(dev, &mai, nullptr, &storageBufferMemory);
    vkBindBufferMemory(dev, storageBuffer, storageBufferMemory, 0);
    vkMapMemory(dev, storageBufferMemory, 0, bufSize, 0, &mappedStoragePtr);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1;
    dslci.pBindings = &binding;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &descriptorSetLayout);

    // Descriptor pool
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &descriptorPool);

    // Allocate + write descriptor set
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &descriptorSetLayout;
    vkAllocateDescriptorSets(dev, &dsai, &descriptorSet);

    VkDescriptorBufferInfo dbi = { storageBuffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet wds = {};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = descriptorSet;
    wds.dstBinding = 0;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.descriptorCount = 1;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
}

void ShaderImage::destroyStorageBuffer()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;
    if (!dev) return;

    if (mappedStoragePtr)     { vkUnmapMemory(dev, storageBufferMemory); mappedStoragePtr = nullptr; }
    if (descriptorPool)       { vkDestroyDescriptorPool(dev, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE; }
    if (descriptorSetLayout)  { vkDestroyDescriptorSetLayout(dev, descriptorSetLayout, nullptr); descriptorSetLayout = VK_NULL_HANDLE; }
    if (storageBuffer)        { vkDestroyBuffer(dev, storageBuffer, nullptr); storageBuffer = VK_NULL_HANDLE; }
    if (storageBufferMemory)  { vkFreeMemory(dev, storageBufferMemory, nullptr); storageBufferMemory = VK_NULL_HANDLE; }
    descriptorSet = VK_NULL_HANDLE;
}

// =============================================================================
// Pipeline
// =============================================================================

void ShaderImage::createPipeline()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

    // Shader modules
    auto createModule = [&](const char* code, size_t size) -> VkShaderModule {
        VkShaderModuleCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = size;
        ci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule mod;
        vkCreateShaderModule(dev, &ci, nullptr, &mod);
        return mod;
    };

    VkShaderModule vertMod = createModule(shaders::fullscreen::vert_spv,
                                           shaders::fullscreen::vert_spvSize);
    VkShaderModule fragMod = createModule(fragShaderCode.data(), fragShaderCode.size());

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    // No vertex input (procedural fullscreen triangle)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp = { 0, 0, (float)w, (float)h, 0, 1 };
    VkRect2D sc = { {0, 0}, {(uint32_t)w, (uint32_t)h} };
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &vp;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt = {};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAtt;

    // Push constants
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    // Pipeline layout
    std::vector<VkDescriptorSetLayout> layouts;
    if (descriptorSetLayout) layouts.push_back(descriptorSetLayout);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo gpci = {};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vertexInput;
    gpci.pInputAssemblyState = &inputAssembly;
    gpci.pViewportState = &viewportState;
    gpci.pRasterizationState = &rasterizer;
    gpci.pMultisampleState = &multisampling;
    gpci.pColorBlendState = &colorBlending;
    gpci.layout = pipelineLayout;
    gpci.renderPass = renderPass;

    vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &graphicsPipeline);

    vkDestroyShaderModule(dev, vertMod, nullptr);
    vkDestroyShaderModule(dev, fragMod, nullptr);
}

void ShaderImage::destroyPipeline()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;
    if (!dev) return;

    if (graphicsPipeline) { vkDestroyPipeline(dev, graphicsPipeline, nullptr); graphicsPipeline = VK_NULL_HANDLE; }
    if (pipelineLayout)   { vkDestroyPipelineLayout(dev, pipelineLayout, nullptr); pipelineLayout = VK_NULL_HANDLE; }
}

// =============================================================================
// Public API
// =============================================================================

void ShaderImage::setSize(int width, int height)
{
    if (width == w && height == h) return;
    if (width <= 0 || height <= 0) return;

    auto& ctx = getContext();
    vkDeviceWaitIdle(ctx.device);

    stopTimer();
    w = width;
    h = height;

    destroyPipeline();
    destroyRenderTarget();

    displayImage = juce::Image(juce::Image::ARGB, w, h, true);

    createRenderTarget();
    createPipeline();
}

void ShaderImage::setFrameRate(int fps)
{
    stopTimer();
    if (fps > 0) startTimerHz(fps);
}

void ShaderImage::pushData(const float* data, int count)
{
    if (storageSize > 0 && count > 0 && fifo)
        fifo->push_num(data, count);
}

void ShaderImage::render()
{
    if (!ready) return;
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

    int prev = 1 - frameIndex;
    int curr = frameIndex;

    // Wait for the previous frame's GPU work to finish
    vkWaitForFences(dev, 1, &fence[prev], VK_TRUE, UINT64_MAX);

    // Copy previous frame's staging buffer to display image (one frame of latency)
    // BGRA → JUCE ARGB: identical layout on little-endian
    {
        juce::Image::BitmapData bd(displayImage, juce::Image::BitmapData::writeOnly);
        const auto* src = static_cast<const uint8_t*>(mappedStagingPtr[prev]);
        for (int y = 0; y < h; y++)
            memcpy(bd.getLinePointer(y), src + y * w * 4, w * 4);
    }

    // Update storage buffer from FIFO
    if (storageSize > 0 && mappedStoragePtr && fifo)
    {
        int available = fifo->size();
        for (int i = 0; i < available && i < storageSize; i++)
        {
            if (fifo->size() > 0)
                storageData[i % storageSize] = fifo->pop();
        }
        memcpy(mappedStoragePtr, storageData.data(), sizeof(float) * storageSize);
    }

    // Reset current frame's fence and record command buffer
    vkResetFences(dev, 1, &fence[curr]);
    vkResetCommandBuffer(commandBuffer[curr], 0);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer[curr], &beginInfo);

    // Begin render pass
    VkClearValue clearValue = {};
    clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

    VkRenderPassBeginInfo rpbi = {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea = {{0, 0}, {(uint32_t)w, (uint32_t)h}};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    vkCmdBeginRenderPass(commandBuffer[curr], &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer[curr], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    // Push constants
    PushConstants pc;
    pc.resolutionX = (float)w;
    pc.resolutionY = (float)h;
    pc.time = (float)(juce::Time::getMillisecondCounterHiRes() / 1000.0 - startTime);
    vkCmdPushConstants(commandBuffer[curr], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    // Bind storage buffer descriptor
    if (descriptorSet)
        vkCmdBindDescriptorSets(commandBuffer[curr], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Draw fullscreen triangle
    vkCmdDraw(commandBuffer[curr], 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer[curr]);

    // Copy rendered image to current staging buffer
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
    vkCmdCopyImageToBuffer(commandBuffer[curr], colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffer[curr], 1, &region);

    vkEndCommandBuffer(commandBuffer[curr]);

    // Submit and return immediately — GPU works while CPU is free
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer[curr];
    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fence[curr]);

    frameIndex = prev;
}

} // namespace jvk
