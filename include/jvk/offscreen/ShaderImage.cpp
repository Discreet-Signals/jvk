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
#include "../reflect/spirv_reflect.h"

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
    std::atomic<int> refCount { 0 };

    bool init()
    {
        core::ensureICDDiscoverable();

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "jvk ShaderImage";
        appInfo.apiVersion = VK_API_VERSION_1_0;

        std::vector<const char*> extensions;
#if JUCE_MAC
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

        VkInstanceCreateInfo instInfo = {};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo = &appInfo;
#if JUCE_MAC
        instInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        instInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instInfo.ppEnabledExtensionNames = extensions.data();

        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS)
            return false;

        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
        if (devCount == 0) return false;

        std::vector<VkPhysicalDevice> devices(devCount);
        vkEnumeratePhysicalDevices(instance, &devCount, devices.data());

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
// SPIR-V Reflection
// =============================================================================

void ShaderImage::reflectShader()
{
    SpvReflectShaderModule module;
    SpvReflectResult result = spvReflectCreateShaderModule(
        fragShaderCode.size(), fragShaderCode.data(), &module);

    if (result != SPV_REFLECT_RESULT_SUCCESS)
    {
        DBG("ShaderImage: SPIR-V reflection failed");
        return;
    }

    uint32_t bindingCount = 0;
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, nullptr);
    std::vector<SpvReflectDescriptorBinding*> bindings(bindingCount);
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, bindings.data());

    // Group bindings by descriptor set
    std::map<uint32_t, std::vector<SpvReflectDescriptorBinding*>> setBindings;
    for (auto* b : bindings)
        setBindings[b->set].push_back(b);

    for (auto& [setIndex, setBinds] : setBindings)
    {
        DescriptorSetInfo dsi;
        dsi.set = setIndex;

        for (auto* b : setBinds)
        {
            if (b->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            {
                dsi.hasStorageBuffer = true;
                dsi.storageBinding = b->binding;
            }
            else if (b->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                TextureSlot slot;
                slot.binding = b->binding;
                dsi.textures.push_back(slot);
            }
        }

        descriptorSets.push_back(std::move(dsi));
    }

    // Sort by set index
    std::sort(descriptorSets.begin(), descriptorSets.end(),
              [](const DescriptorSetInfo& a, const DescriptorSetInfo& b) { return a.set < b.set; });

    spvReflectDestroyShaderModule(&module);
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

    reflectShader();
    createRenderTarget();
    createDescriptors();
    createPipeline();

    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx.commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    vkAllocateCommandBuffers(ctx.device, &cbai, commandBuffer);

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
    destroyDescriptors();
    destroyRenderTarget();

    if (renderPass) { vkDestroyRenderPass(ctx.device, renderPass, nullptr); renderPass = VK_NULL_HANDLE; }

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

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = colorImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &colorImageView);

    // Render pass is format-dependent, not size-dependent — create once
    if (renderPass == VK_NULL_HANDLE)
    {
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
    }

    VkFramebufferCreateInfo fbci = {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &colorImageView;
    fbci.width = w;
    fbci.height = h;
    fbci.layers = 1;
    vkCreateFramebuffer(dev, &fbci, nullptr, &framebuffer);

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
    if (colorImageView)   { vkDestroyImageView(dev, colorImageView, nullptr); colorImageView = VK_NULL_HANDLE; }
    if (colorImage)       { vkDestroyImage(dev, colorImage, nullptr); colorImage = VK_NULL_HANDLE; }
    if (colorImageMemory) { vkFreeMemory(dev, colorImageMemory, nullptr); colorImageMemory = VK_NULL_HANDLE; }
}

// =============================================================================
// Descriptors (auto-created from SPIR-V reflection)
// =============================================================================

void ShaderImage::createDescriptors()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

    for (auto& dsi : descriptorSets)
    {
        // Build layout bindings
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        std::vector<VkDescriptorPoolSize> poolSizes;

        if (dsi.hasStorageBuffer)
        {
            VkDescriptorSetLayoutBinding b = {};
            b.binding = dsi.storageBinding;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            layoutBindings.push_back(b);
            poolSizes.push_back({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 });

            // Create storage buffer
            VkDeviceSize bufSize = sizeof(float) * storageSize;
            VkBufferCreateInfo bci = {};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = bufSize;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            vkCreateBuffer(dev, &bci, nullptr, &dsi.storageBuffer);

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(dev, dsi.storageBuffer, &req);
            VkMemoryAllocateInfo mai = {};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(dev, &mai, nullptr, &dsi.storageBufferMemory);
            vkBindBufferMemory(dev, dsi.storageBuffer, dsi.storageBufferMemory, 0);
            vkMapMemory(dev, dsi.storageBufferMemory, 0, bufSize, 0, &dsi.mappedStoragePtr);
        }

        for (auto& tex : dsi.textures)
        {
            VkDescriptorSetLayoutBinding b = {};
            b.binding = tex.binding;
            b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            layoutBindings.push_back(b);
            poolSizes.push_back({ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 });
        }

        // Create layout
        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        dslci.pBindings = layoutBindings.data();
        vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsi.layout);

        // Create pool
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes = poolSizes.data();
        vkCreateDescriptorPool(dev, &dpci, nullptr, &dsi.pool);

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = dsi.pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsi.layout;
        vkAllocateDescriptorSets(dev, &dsai, &dsi.descriptorSet);

        // Write storage buffer descriptor
        if (dsi.hasStorageBuffer && dsi.storageBuffer)
        {
            VkDescriptorBufferInfo dbi = { dsi.storageBuffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet wds = {};
            wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds.dstSet = dsi.descriptorSet;
            wds.dstBinding = dsi.storageBinding;
            wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds.descriptorCount = 1;
            wds.pBufferInfo = &dbi;
            vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
        }

        // Texture descriptors are written when loadTexture() is called
    }
}

void ShaderImage::destroyDescriptors()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;
    if (!dev) return;

    for (auto& dsi : descriptorSets)
    {
        // Destroy storage buffer
        if (dsi.mappedStoragePtr)    { vkUnmapMemory(dev, dsi.storageBufferMemory); dsi.mappedStoragePtr = nullptr; }
        if (dsi.storageBuffer)      { vkDestroyBuffer(dev, dsi.storageBuffer, nullptr); dsi.storageBuffer = VK_NULL_HANDLE; }
        if (dsi.storageBufferMemory){ vkFreeMemory(dev, dsi.storageBufferMemory, nullptr); dsi.storageBufferMemory = VK_NULL_HANDLE; }

        // Destroy textures
        for (auto& tex : dsi.textures)
        {
            if (tex.sampler) { vkDestroySampler(dev, tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
            if (tex.view)    { vkDestroyImageView(dev, tex.view, nullptr); tex.view = VK_NULL_HANDLE; }
            if (tex.image)   { vkDestroyImage(dev, tex.image, nullptr); tex.image = VK_NULL_HANDLE; }
            if (tex.memory)  { vkFreeMemory(dev, tex.memory, nullptr); tex.memory = VK_NULL_HANDLE; }
            tex.loaded = false;
        }

        if (dsi.pool)   { vkDestroyDescriptorPool(dev, dsi.pool, nullptr); dsi.pool = VK_NULL_HANDLE; }
        if (dsi.layout)  { vkDestroyDescriptorSetLayout(dev, dsi.layout, nullptr); dsi.layout = VK_NULL_HANDLE; }
        dsi.descriptorSet = VK_NULL_HANDLE;
    }
}

// =============================================================================
// Texture loading
// =============================================================================

void ShaderImage::loadTexture(int binding, const juce::Image& image)
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;
    if (!dev) return;

    // Find the texture slot matching this binding
    TextureSlot* slot = nullptr;
    DescriptorSetInfo* ownerSet = nullptr;
    for (auto& dsi : descriptorSets)
        for (auto& tex : dsi.textures)
            if (tex.binding == binding)
            { slot = &tex; ownerSet = &dsi; break; }

    if (!slot || !ownerSet)
    {
        DBG("ShaderImage: No sampler binding " << binding << " found in shader");
        return;
    }

    vkDeviceWaitIdle(dev);

    // Clean up previous texture in this slot
    if (slot->sampler) { vkDestroySampler(dev, slot->sampler, nullptr); slot->sampler = VK_NULL_HANDLE; }
    if (slot->view)    { vkDestroyImageView(dev, slot->view, nullptr); slot->view = VK_NULL_HANDLE; }
    if (slot->image)   { vkDestroyImage(dev, slot->image, nullptr); slot->image = VK_NULL_HANDLE; }
    if (slot->memory)  { vkFreeMemory(dev, slot->memory, nullptr); slot->memory = VK_NULL_HANDLE; }

    int tw = image.getWidth();
    int th = image.getHeight();
    VkDeviceSize imageSize = (VkDeviceSize)tw * th * 4;

    // --- Staging buffer ---
    VkBuffer stagBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = imageSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(dev, &bci, nullptr, &stagBuf);

    VkMemoryRequirements bufReq;
    vkGetBufferMemoryRequirements(dev, stagBuf, &bufReq);
    VkMemoryAllocateInfo bmai = {};
    bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bmai.allocationSize = bufReq.size;
    bmai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(dev, &bmai, nullptr, &stagMem);
    vkBindBufferMemory(dev, stagBuf, stagMem, 0);

    void* mapped = nullptr;
    vkMapMemory(dev, stagMem, 0, imageSize, 0, &mapped);
    {
        juce::Image argb = image.convertedToFormat(juce::Image::ARGB);
        juce::Image::BitmapData bd(argb, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < th; y++)
            memcpy(static_cast<uint8_t*>(mapped) + y * tw * 4, bd.getLinePointer(y), tw * 4);
    }
    vkUnmapMemory(dev, stagMem);

    // --- VkImage ---
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = { (uint32_t)tw, (uint32_t)th, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(dev, &ici, nullptr, &slot->image);

    VkMemoryRequirements imgReq;
    vkGetImageMemoryRequirements(dev, slot->image, &imgReq);
    VkMemoryAllocateInfo imai = {};
    imai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imai.allocationSize = imgReq.size;
    imai.memoryTypeIndex = core::findMemoryType(ctx.physicalDevice, imgReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(dev, &imai, nullptr, &slot->memory);
    vkBindImageMemory(dev, slot->image, slot->memory, 0);

    // --- Upload via one-shot command buffer ---
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx.commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = slot->image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (uint32_t)tw, (uint32_t)th, 1 };
    vkCmdCopyBufferToImage(cmd, stagBuf, slot->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue);
    vkFreeCommandBuffers(dev, ctx.commandPool, 1, &cmd);

    vkDestroyBuffer(dev, stagBuf, nullptr);
    vkFreeMemory(dev, stagMem, nullptr);

    // --- Image view ---
    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = slot->image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &slot->view);

    // --- Sampler ---
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 1.0f;
    vkCreateSampler(dev, &sci, nullptr, &slot->sampler);

    // --- Write descriptor ---
    VkDescriptorImageInfo dii = {};
    dii.sampler = slot->sampler;
    dii.imageView = slot->view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wds = {};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = ownerSet->descriptorSet;
    wds.dstBinding = slot->binding;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wds.descriptorCount = 1;
    wds.pImageInfo = &dii;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    slot->loaded = true;
}

// =============================================================================
// Pipeline
// =============================================================================

void ShaderImage::createPipeline()
{
    auto& ctx = getContext();
    VkDevice dev = ctx.device;

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

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

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

    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    // Collect descriptor set layouts in set-index order
    std::vector<VkDescriptorSetLayout> layouts;
    for (auto& dsi : descriptorSets)
        layouts.push_back(dsi.layout);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout);

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

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
    gpci.pDynamicState = &dynamicState;
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
    vkWaitForFences(ctx.device, 2, fence, VK_TRUE, UINT64_MAX);

    int savedFps = isTimerRunning() ? (1000 / getTimerInterval()) : 0;
    stopTimer();
    w = width;
    h = height;

    destroyRenderTarget();

    displayImage = displayImage.rescaled(w, h);

    createRenderTarget();

    // Submit a render at the new size immediately so the next
    // render() call can copy real content. Skip the staging->display
    // copy on that first call since the staging buffers are zeroed.
    skipNextStagingCopy = true;
    render();

    if (savedFps > 0) startTimerHz(savedFps);
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

    vkWaitForFences(dev, 1, &fence[prev], VK_TRUE, UINT64_MAX);

    // Copy previous frame's staging buffer to display image
    // (skip after resize — staging is zeroed, displayImage has rescaled content)
    if (skipNextStagingCopy)
        skipNextStagingCopy = false;
    else
    {
        juce::Image::BitmapData bd(displayImage, juce::Image::BitmapData::writeOnly);
        const auto* src = static_cast<const uint8_t*>(mappedStagingPtr[prev]);
        for (int y = 0; y < h; y++)
            memcpy(bd.getLinePointer(y), src + y * w * 4, w * 4);
    }

    // Update storage buffer from FIFO
    for (auto& dsi : descriptorSets)
    {
        if (dsi.hasStorageBuffer && dsi.mappedStoragePtr && storageSize > 0 && fifo)
        {
            int available = fifo->size();
            for (int i = 0; i < available && i < storageSize; i++)
            {
                if (fifo->size() > 0)
                    storageData[i % storageSize] = fifo->pop();
            }
            memcpy(dsi.mappedStoragePtr, storageData.data(), sizeof(float) * storageSize);
        }
    }

    vkResetFences(dev, 1, &fence[curr]);
    vkResetCommandBuffer(commandBuffer[curr], 0);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer[curr], &beginInfo);

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

    VkViewport vp = { 0, 0, (float)w, (float)h, 0, 1 };
    vkCmdSetViewport(commandBuffer[curr], 0, 1, &vp);
    VkRect2D sc = { {0, 0}, {(uint32_t)w, (uint32_t)h} };
    vkCmdSetScissor(commandBuffer[curr], 0, 1, &sc);

    // Push constants
    PushConstants pc;
    pc.resolutionX = (float)w;
    pc.resolutionY = (float)h;
    pc.time = (float)(juce::Time::getMillisecondCounterHiRes() / 1000.0 - startTime);
    vkCmdPushConstants(commandBuffer[curr], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    // Bind all descriptor sets
    for (auto& dsi : descriptorSets)
    {
        vkCmdBindDescriptorSets(commandBuffer[curr], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, dsi.set, 1, &dsi.descriptorSet, 0, nullptr);
    }

    vkCmdDraw(commandBuffer[curr], 3, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer[curr]);

    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
    vkCmdCopyImageToBuffer(commandBuffer[curr], colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffer[curr], 1, &region);

    vkEndCommandBuffer(commandBuffer[curr]);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer[curr];
    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fence[curr]);

    frameIndex = prev;
}

} // namespace jvk
