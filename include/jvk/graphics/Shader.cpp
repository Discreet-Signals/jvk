/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: Shader.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "Shader.h"
#include "ShaderRegionVert.h"
#include "../reflect/spirv_reflect.h"
#include <map>
#include <algorithm>
#include <cstring>

namespace jvk
{

// =============================================================================
// SPIR-V Reflection
// =============================================================================

void Shader::reflectShader()
{
    SpvReflectShaderModule module;
    SpvReflectResult result = spvReflectCreateShaderModule(
        fragSpv.size(), fragSpv.data(), &module);

    if (result != SPV_REFLECT_RESULT_SUCCESS)
    {
        DBG("jvk::Shader: SPIR-V reflection failed");
        return;
    }

    uint32_t bindingCount = 0;
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, nullptr);
    std::vector<SpvReflectDescriptorBinding*> bindings(bindingCount);
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, bindings.data());

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

    std::sort(descriptorSets.begin(), descriptorSets.end(),
              [](const DescriptorSetInfo& a, const DescriptorSetInfo& b) { return a.set < b.set; });

    spvReflectDestroyShaderModule(&module);
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

Shader::Shader(const char* fragmentSpv, int fragmentSpvSize, int storageBufferFloats)
    : fragSpv(fragmentSpv, fragmentSpv + fragmentSpvSize),
      storageSize(storageBufferFloats),
      startTime(juce::Time::getMillisecondCounterHiRes() / 1000.0)
{
    if (storageBufferFloats > 0)
        storageData.resize(storageBufferFloats, 0.0f);

    reflectShader();
}

Shader::~Shader()
{
    destroy();
}

// =============================================================================
// Lazy resource creation
// =============================================================================

void Shader::ensureCreated(VkPhysicalDevice physDevice, VkDevice device,
                           VkRenderPass renderPass, VkSampleCountFlagBits msaa)
{
    if (created && device == cachedDevice && renderPass == cachedRenderPass && msaa == cachedMsaa)
        return;

    if (created)
        destroy();

    cachedPhysDevice = physDevice;
    cachedDevice = device;
    cachedRenderPass = renderPass;
    cachedMsaa = msaa;

    // Find graphics queue family and get queue for texture uploads
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, qfs.data());

    uint32_t graphicsFamily = 0;
    for (uint32_t i = 0; i < qfCount; i++)
    {
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        { graphicsFamily = i; break; }
    }

    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = graphicsFamily;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &cpci, nullptr, &commandPool);

    createDescriptors(physDevice, device);
    createPipeline(device, renderPass, msaa);
    created = true;

    // Upload any textures that were deferred before Vulkan resources existed
    uploadPendingTextures();
}

void Shader::uploadPendingTextures()
{
    for (auto& pt : pendingTextures)
        loadTexture(pt.binding, pt.image);
    pendingTextures.clear();
}

void Shader::destroy()
{
    if (!cachedDevice) return;

    vkDeviceWaitIdle(cachedDevice);

    if (pipeline)       { vkDestroyPipeline(cachedDevice, pipeline, nullptr); pipeline = VK_NULL_HANDLE; }
    if (pipelineLayout) { vkDestroyPipelineLayout(cachedDevice, pipelineLayout, nullptr); pipelineLayout = VK_NULL_HANDLE; }

    for (auto& dsi : descriptorSets)
    {
        if (dsi.mappedStoragePtr)    { vkUnmapMemory(cachedDevice, dsi.storageBufferMemory); dsi.mappedStoragePtr = nullptr; }
        if (dsi.storageBuffer)       { vkDestroyBuffer(cachedDevice, dsi.storageBuffer, nullptr); dsi.storageBuffer = VK_NULL_HANDLE; }
        if (dsi.storageBufferMemory) { vkFreeMemory(cachedDevice, dsi.storageBufferMemory, nullptr); dsi.storageBufferMemory = VK_NULL_HANDLE; }

        for (auto& tex : dsi.textures)
        {
            if (tex.sampler) { vkDestroySampler(cachedDevice, tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
            if (tex.view)    { vkDestroyImageView(cachedDevice, tex.view, nullptr); tex.view = VK_NULL_HANDLE; }
            if (tex.image)   { vkDestroyImage(cachedDevice, tex.image, nullptr); tex.image = VK_NULL_HANDLE; }
            if (tex.memory)  { vkFreeMemory(cachedDevice, tex.memory, nullptr); tex.memory = VK_NULL_HANDLE; }
            tex.loaded = false;
        }

        if (dsi.pool)   { vkDestroyDescriptorPool(cachedDevice, dsi.pool, nullptr); dsi.pool = VK_NULL_HANDLE; }
        if (dsi.layout) { vkDestroyDescriptorSetLayout(cachedDevice, dsi.layout, nullptr); dsi.layout = VK_NULL_HANDLE; }
        dsi.descriptorSet = VK_NULL_HANDLE;
    }

    if (commandPool) { vkDestroyCommandPool(cachedDevice, commandPool, nullptr); commandPool = VK_NULL_HANDLE; }

    created = false;
}

// =============================================================================
// Descriptors (from reflected bindings)
// =============================================================================

void Shader::createDescriptors(VkPhysicalDevice physDevice, VkDevice device)
{
    for (auto& dsi : descriptorSets)
    {
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

            VkDeviceSize bufSize = sizeof(float) * storageSize;
            VkBufferCreateInfo bci = {};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = bufSize;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            vkCreateBuffer(device, &bci, nullptr, &dsi.storageBuffer);

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, dsi.storageBuffer, &req);
            VkMemoryAllocateInfo mai = {};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = core::findMemoryType(physDevice, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(device, &mai, nullptr, &dsi.storageBufferMemory);
            vkBindBufferMemory(device, dsi.storageBuffer, dsi.storageBufferMemory, 0);
            vkMapMemory(device, dsi.storageBufferMemory, 0, bufSize, 0, &dsi.mappedStoragePtr);
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

        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        dslci.pBindings = layoutBindings.data();
        vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsi.layout);

        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes = poolSizes.data();
        vkCreateDescriptorPool(device, &dpci, nullptr, &dsi.pool);

        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = dsi.pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsi.layout;
        vkAllocateDescriptorSets(device, &dsai, &dsi.descriptorSet);

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
            vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
        }
    }
}

// =============================================================================
// Pipeline
// =============================================================================

void Shader::createPipeline(VkDevice device, VkRenderPass renderPass, VkSampleCountFlagBits msaa)
{
    auto createModule = [&](const char* code, size_t size) -> VkShaderModule {
        VkShaderModuleCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = size;
        ci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule mod;
        vkCreateShaderModule(device, &ci, nullptr, &mod);
        return mod;
    };

    VkShaderModule vertMod = createModule(shaders::shader_region::vert_spv,
                                           shaders::shader_region::vert_spvSize);
    VkShaderModule fragMod = createModule(fragSpv.data(), fragSpv.size());

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    // No vertex input — vertex shader generates geometry from gl_VertexIndex
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport, scissor, and stencil reference (for path clipping)
    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE };
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

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
    multisampling.rasterizationSamples = msaa;

    // Depth/stencil — stencil test enabled to respect JUCE path clipping
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_TRUE;
    depthStencil.front.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.front.passOp = VK_STENCIL_OP_KEEP;
    depthStencil.front.failOp = VK_STENCIL_OP_KEEP;
    depthStencil.front.depthFailOp = VK_STENCIL_OP_KEEP;
    depthStencil.front.compareMask = 0xFF;
    depthStencil.front.writeMask = 0x00;
    depthStencil.front.reference = 0;
    depthStencil.back = depthStencil.front;

    // Opaque blend — shader output replaces framebuffer content
    VkPipelineColorBlendAttachmentState blendAtt = {};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAtt;

    // Push constants: bytes 0-11 shared (resolution + time), bytes 12-27 vertex-only (viewport + region)
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(float) * 7; // resX, resY, time, vpW, vpH, regX, regY

    std::vector<VkDescriptorSetLayout> layouts;
    for (auto& dsi : descriptorSets)
        layouts.push_back(dsi.layout);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = static_cast<uint32_t>(layouts.size());
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo gpci = {};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vertexInput;
    gpci.pInputAssemblyState = &inputAssembly;
    gpci.pViewportState = &viewportState;
    gpci.pRasterizationState = &rasterizer;
    gpci.pMultisampleState = &multisampling;
    gpci.pDepthStencilState = &depthStencil;
    gpci.pColorBlendState = &colorBlending;
    gpci.pDynamicState = &dynamicState;
    gpci.layout = pipelineLayout;
    gpci.renderPass = renderPass;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline);

    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

// =============================================================================
// Public API
// =============================================================================

void Shader::pushData(const float* data, int count)
{
    if (storageSize <= 0 || count <= 0) return;
    int n = std::min(count, storageSize);
    std::memcpy(storageData.data(), data, sizeof(float) * n);
}

void Shader::loadTexture(int binding, const juce::Image& image)
{
    if (!created)
    {
        // Defer until Vulkan resources exist
        pendingTextures.push_back({ binding, image });
        return;
    }

    // Find texture slot
    TextureSlot* slot = nullptr;
    DescriptorSetInfo* ownerSet = nullptr;
    for (auto& dsi : descriptorSets)
        for (auto& tex : dsi.textures)
            if (tex.binding == binding)
            { slot = &tex; ownerSet = &dsi; break; }

    if (!slot || !ownerSet)
    {
        DBG("jvk::Shader: No sampler binding " << binding << " found in shader");
        return;
    }

    VkDevice device = cachedDevice;
    VkPhysicalDevice physDevice = cachedPhysDevice;
    vkDeviceWaitIdle(device);

    // Clean up previous
    if (slot->sampler) { vkDestroySampler(device, slot->sampler, nullptr); slot->sampler = VK_NULL_HANDLE; }
    if (slot->view)    { vkDestroyImageView(device, slot->view, nullptr); slot->view = VK_NULL_HANDLE; }
    if (slot->image)   { vkDestroyImage(device, slot->image, nullptr); slot->image = VK_NULL_HANDLE; }
    if (slot->memory)  { vkFreeMemory(device, slot->memory, nullptr); slot->memory = VK_NULL_HANDLE; }

    int tw = image.getWidth();
    int th = image.getHeight();
    VkDeviceSize imageSize = (VkDeviceSize)tw * th * 4;

    // Staging buffer
    VkBuffer stagBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = imageSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(device, &bci, nullptr, &stagBuf);

    VkMemoryRequirements bufReq;
    vkGetBufferMemoryRequirements(device, stagBuf, &bufReq);
    VkMemoryAllocateInfo bmai = {};
    bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bmai.allocationSize = bufReq.size;
    bmai.memoryTypeIndex = core::findMemoryType(physDevice, bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &bmai, nullptr, &stagMem);
    vkBindBufferMemory(device, stagBuf, stagMem, 0);

    void* mapped = nullptr;
    vkMapMemory(device, stagMem, 0, imageSize, 0, &mapped);
    {
        juce::Image argb = image.convertedToFormat(juce::Image::ARGB);
        juce::Image::BitmapData bd(argb, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < th; y++)
            std::memcpy(static_cast<uint8_t*>(mapped) + y * tw * 4, bd.getLinePointer(y), tw * 4);
    }
    vkUnmapMemory(device, stagMem);

    // VkImage
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
    vkCreateImage(device, &ici, nullptr, &slot->image);

    VkMemoryRequirements imgReq;
    vkGetImageMemoryRequirements(device, slot->image, &imgReq);
    VkMemoryAllocateInfo imai = {};
    imai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imai.allocationSize = imgReq.size;
    imai.memoryTypeIndex = core::findMemoryType(physDevice, imgReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &imai, nullptr, &slot->memory);
    vkBindImageMemory(device, slot->image, slot->memory, 0);

    // Upload via one-shot command buffer
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbai, &cmd);

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
    vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);

    vkDestroyBuffer(device, stagBuf, nullptr);
    vkFreeMemory(device, stagMem, nullptr);

    // Image view
    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = slot->image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(device, &ivci, nullptr, &slot->view);

    // Sampler
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 1.0f;
    vkCreateSampler(device, &sci, nullptr, &slot->sampler);

    // Write descriptor
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
    vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);

    slot->loaded = true;
}

} // namespace jvk
