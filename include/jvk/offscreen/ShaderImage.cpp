/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: ShaderImage.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk
{

// =============================================================================
// SPIR-V reflection — discover descriptor set layout from fragment bytecode
// =============================================================================

void ShaderImage::reflectShader()
{
    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(fragShaderCode.size(), fragShaderCode.data(), &module)
        != SPV_REFLECT_RESULT_SUCCESS)
    {
        DBG("ShaderImage: SPIR-V reflection failed");
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
                dsi.storageBinding = static_cast<int>(b->binding);
            }
            else if (b->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                TextureSlot slot;
                slot.binding = static_cast<int>(b->binding);
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

ShaderImage::ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                         int width, int height, int storageBufSize)
    : device_(Device::acquire()),
      w(width), h(height),
      fragShaderCode(fragmentSpv, fragmentSpv + fragmentSpvSize),
      storageSize(storageBufSize),
      startTime(juce::Time::getMillisecondCounterHiRes() / 1000.0)
{
    if (storageBufSize > 0)
    {
        int cap = storageBufSize * 3;
        fifo       = std::make_unique<juce::AbstractFifo>(cap);
        fifoBuffer.assign(static_cast<size_t>(cap), 0.0f);
        storageData.assign(static_cast<size_t>(storageBufSize), 0.0f);
    }

    displayImage = juce::Image(juce::Image::ARGB, w, h, true);

    reflectShader();
    createRenderTarget();
    createDescriptors();
    createPipeline();

    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = device_->commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    vkAllocateCommandBuffers(device_->device(), &cbai, commandBuffer);

    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device_->device(), &fci, nullptr, &fence[0]);
    vkCreateFence(device_->device(), &fci, nullptr, &fence[1]);

    ready = true;
}

ShaderImage::~ShaderImage()
{
    stopTimer();

    VkDevice dev = device_->device();
    if (!dev) return;

    vkWaitForFences(dev, 2, fence, VK_TRUE, UINT64_MAX);
    for (int i = 0; i < 2; i++)
        if (fence[i]) vkDestroyFence(dev, fence[i], nullptr);

    if (commandBuffer[0] || commandBuffer[1])
        vkFreeCommandBuffers(dev, device_->commandPool(), 2, commandBuffer);

    destroyPipeline();
    destroyDescriptors();
    destroyRenderTarget();

    if (renderPass) { vkDestroyRenderPass(dev, renderPass, nullptr); renderPass = VK_NULL_HANDLE; }
}

// =============================================================================
// Render target
// =============================================================================

void ShaderImage::createRenderTarget()
{
    VkDevice dev = device_->device();

    VkImageCreateInfo ici {};
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

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, colorImage, &req);
    colorImageAlloc = device_->pool().alloc(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev, colorImage, colorImageAlloc.memory, colorImageAlloc.offset);

    VkImageViewCreateInfo ivci {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = colorImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &colorImageView);

    // Render pass is format-dependent, not size-dependent — create once
    if (renderPass == VK_NULL_HANDLE)
    {
        VkAttachmentDescription att {};
        att.format = VK_FORMAT_B8G8R8A8_UNORM;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentReference ref { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sub {};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &ref;

        VkRenderPassCreateInfo rpci {};
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.attachmentCount = 1;
        rpci.pAttachments = &att;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &sub;
        vkCreateRenderPass(dev, &rpci, nullptr, &renderPass);
    }

    VkFramebufferCreateInfo fbci {};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &colorImageView;
    fbci.width = (uint32_t)w;
    fbci.height = (uint32_t)h;
    fbci.layers = 1;
    vkCreateFramebuffer(dev, &fbci, nullptr, &framebuffer);

    VkDeviceSize stagingSize = (VkDeviceSize)w * h * 4;
    for (int i = 0; i < 2; i++)
    {
        VkBufferCreateInfo bci {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = stagingSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &stagingBuffer[i]);

        VkMemoryRequirements bufReq;
        vkGetBufferMemoryRequirements(dev, stagingBuffer[i], &bufReq);
        VkMemoryAllocateInfo bmai {};
        bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bmai.allocationSize = bufReq.size;
        bmai.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(), bufReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &bmai, nullptr, &stagingMemory[i]);
        vkBindBufferMemory(dev, stagingBuffer[i], stagingMemory[i], 0);
        vkMapMemory(dev, stagingMemory[i], 0, stagingSize, 0, &mappedStagingPtr[i]);
        memset(mappedStagingPtr[i], 0, stagingSize);
    }
}

void ShaderImage::destroyRenderTarget()
{
    VkDevice dev = device_->device();
    if (!dev) return;

    for (int i = 0; i < 2; i++)
    {
        if (mappedStagingPtr[i]) { vkUnmapMemory(dev, stagingMemory[i]); mappedStagingPtr[i] = nullptr; }
        if (stagingBuffer[i])    { vkDestroyBuffer(dev, stagingBuffer[i], nullptr); stagingBuffer[i] = VK_NULL_HANDLE; }
        if (stagingMemory[i])    { vkFreeMemory(dev, stagingMemory[i], nullptr); stagingMemory[i] = VK_NULL_HANDLE; }
    }
    if (framebuffer)     { vkDestroyFramebuffer(dev, framebuffer, nullptr); framebuffer = VK_NULL_HANDLE; }
    if (colorImageView)  { vkDestroyImageView(dev, colorImageView, nullptr); colorImageView = VK_NULL_HANDLE; }
    if (colorImage)      { vkDestroyImage(dev, colorImage, nullptr); colorImage = VK_NULL_HANDLE; }
    if (colorImageAlloc.memory != VK_NULL_HANDLE) { device_->pool().free(colorImageAlloc); colorImageAlloc = {}; }
}

// =============================================================================
// Descriptors (auto-created from SPIR-V reflection)
// =============================================================================

void ShaderImage::createDescriptors()
{
    VkDevice dev = device_->device();

    for (auto& dsi : descriptorSets)
    {
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        std::vector<VkDescriptorPoolSize> poolSizes;

        if (dsi.hasStorageBuffer)
        {
            VkDescriptorSetLayoutBinding b {};
            b.binding = (uint32_t)dsi.storageBinding;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            layoutBindings.push_back(b);
            poolSizes.push_back({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 });

            VkDeviceSize bufSize = sizeof(float) * (VkDeviceSize)storageSize;
            VkBufferCreateInfo bci {};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = bufSize;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            vkCreateBuffer(dev, &bci, nullptr, &dsi.storageBuffer);

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(dev, dsi.storageBuffer, &req);
            VkMemoryAllocateInfo mai {};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(), req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(dev, &mai, nullptr, &dsi.storageBufferMemory);
            vkBindBufferMemory(dev, dsi.storageBuffer, dsi.storageBufferMemory, 0);
            vkMapMemory(dev, dsi.storageBufferMemory, 0, bufSize, 0, &dsi.mappedStoragePtr);
        }

        for (auto& tex : dsi.textures)
        {
            VkDescriptorSetLayoutBinding b {};
            b.binding = (uint32_t)tex.binding;
            b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            layoutBindings.push_back(b);
            poolSizes.push_back({ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 });
        }

        VkDescriptorSetLayoutCreateInfo dslci {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = (uint32_t)layoutBindings.size();
        dslci.pBindings = layoutBindings.data();
        vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsi.layout);

        VkDescriptorPoolCreateInfo dpci {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = (uint32_t)poolSizes.size();
        dpci.pPoolSizes = poolSizes.data();
        vkCreateDescriptorPool(dev, &dpci, nullptr, &dsi.pool);

        VkDescriptorSetAllocateInfo dsai {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = dsi.pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsi.layout;
        vkAllocateDescriptorSets(dev, &dsai, &dsi.descriptorSet);

        if (dsi.hasStorageBuffer && dsi.storageBuffer)
        {
            VkDescriptorBufferInfo dbi { dsi.storageBuffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet wds {};
            wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds.dstSet = dsi.descriptorSet;
            wds.dstBinding = (uint32_t)dsi.storageBinding;
            wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds.descriptorCount = 1;
            wds.pBufferInfo = &dbi;
            vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
        }
    }
}

void ShaderImage::destroyDescriptors()
{
    VkDevice dev = device_->device();
    if (!dev) return;

    for (auto& dsi : descriptorSets)
    {
        if (dsi.mappedStoragePtr)    { vkUnmapMemory(dev, dsi.storageBufferMemory); dsi.mappedStoragePtr = nullptr; }
        if (dsi.storageBuffer)       { vkDestroyBuffer(dev, dsi.storageBuffer, nullptr); dsi.storageBuffer = VK_NULL_HANDLE; }
        if (dsi.storageBufferMemory) { vkFreeMemory(dev, dsi.storageBufferMemory, nullptr); dsi.storageBufferMemory = VK_NULL_HANDLE; }

        for (auto& tex : dsi.textures)
        {
            if (tex.sampler) { vkDestroySampler(dev, tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
            if (tex.view)    { vkDestroyImageView(dev, tex.view, nullptr); tex.view = VK_NULL_HANDLE; }
            if (tex.image)   { vkDestroyImage(dev, tex.image, nullptr); tex.image = VK_NULL_HANDLE; }
            if (tex.alloc.memory != VK_NULL_HANDLE) { device_->pool().free(tex.alloc); tex.alloc = {}; }
            tex.loaded = false;
        }

        if (dsi.pool)   { vkDestroyDescriptorPool(dev, dsi.pool, nullptr); dsi.pool = VK_NULL_HANDLE; }
        if (dsi.layout) { vkDestroyDescriptorSetLayout(dev, dsi.layout, nullptr); dsi.layout = VK_NULL_HANDLE; }
        dsi.descriptorSet = VK_NULL_HANDLE;
    }
}

// =============================================================================
// Texture loading
// =============================================================================

void ShaderImage::loadTexture(int binding, const juce::Image& image)
{
    VkDevice dev = device_->device();
    if (!dev) return;

    TextureSlot* slot = nullptr;
    DescriptorSetInfo* ownerSet = nullptr;
    for (auto& dsi : descriptorSets)
        for (auto& tex : dsi.textures)
            if (tex.binding == binding) { slot = &tex; ownerSet = &dsi; break; }

    if (!slot || !ownerSet)
    {
        DBG("ShaderImage: No sampler binding " << binding << " in shader");
        return;
    }

    vkDeviceWaitIdle(dev);

    if (slot->sampler) { vkDestroySampler(dev, slot->sampler, nullptr); slot->sampler = VK_NULL_HANDLE; }
    if (slot->view)    { vkDestroyImageView(dev, slot->view, nullptr); slot->view = VK_NULL_HANDLE; }
    if (slot->image)   { vkDestroyImage(dev, slot->image, nullptr); slot->image = VK_NULL_HANDLE; }
    if (slot->alloc.memory != VK_NULL_HANDLE) { device_->pool().free(slot->alloc); slot->alloc = {}; }

    int tw = image.getWidth();
    int th = image.getHeight();
    VkDeviceSize imageSize = (VkDeviceSize)tw * th * 4;

    VkBuffer stagBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bci {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = imageSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(dev, &bci, nullptr, &stagBuf);

    VkMemoryRequirements bufReq;
    vkGetBufferMemoryRequirements(dev, stagBuf, &bufReq);
    VkMemoryAllocateInfo bmai {};
    bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bmai.allocationSize = bufReq.size;
    bmai.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(), bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(dev, &bmai, nullptr, &stagMem);
    vkBindBufferMemory(dev, stagBuf, stagMem, 0);

    void* mapped = nullptr;
    vkMapMemory(dev, stagMem, 0, imageSize, 0, &mapped);
    {
        juce::Image argb = image.convertedToFormat(juce::Image::ARGB);
        juce::Image::BitmapData bd(argb, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < th; y++)
            memcpy(static_cast<uint8_t*>(mapped) + y * tw * 4, bd.getLinePointer(y), (size_t)tw * 4);
    }
    vkUnmapMemory(dev, stagMem);

    VkImageCreateInfo ici {};
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
    slot->alloc = device_->pool().alloc(imgReq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(dev, slot->image, slot->alloc.memory, slot->alloc.offset);

    device_->submitImmediate([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier barrier {};
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

        VkBufferImageCopy region {};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { (uint32_t)tw, (uint32_t)th, 1 };
        vkCmdCopyBufferToImage(cmd, stagBuf, slot->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    vkDestroyBuffer(dev, stagBuf, nullptr);
    vkFreeMemory(dev, stagMem, nullptr);

    VkImageViewCreateInfo ivci {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = slot->image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &ivci, nullptr, &slot->view);

    VkSamplerCreateInfo sci {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 1.0f;
    vkCreateSampler(dev, &sci, nullptr, &slot->sampler);

    VkDescriptorImageInfo dii {};
    dii.sampler = slot->sampler;
    dii.imageView = slot->view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wds {};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = ownerSet->descriptorSet;
    wds.dstBinding = (uint32_t)slot->binding;
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
    VkDevice dev = device_->device();

    auto createModule = [&](const char* code, size_t size) -> VkShaderModule {
        VkShaderModuleCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = size;
        ci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule mod;
        vkCreateShaderModule(dev, &ci, nullptr, &mod);
        return mod;
    };

    VkShaderModule vertMod = createModule(shaders::shader_region::vert_spv,
                                          (size_t)shaders::shader_region::vert_spvSize);
    VkShaderModule fragMod = createModule(fragShaderCode.data(), fragShaderCode.size());

    VkPipelineShaderStageCreateInfo stages[2] {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt {};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAtt;

    VkPushConstantRange pcRange {};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    std::vector<VkDescriptorSetLayout> layouts;
    for (auto& dsi : descriptorSets) layouts.push_back(dsi.layout);

    VkPipelineLayoutCreateInfo plci {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = (uint32_t)layouts.size();
    plci.pSetLayouts = layouts.data();
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout);

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo gpci {};
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
    VkDevice dev = device_->device();
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

    vkWaitForFences(device_->device(), 2, fence, VK_TRUE, UINT64_MAX);

    int savedFps = isTimerRunning() ? (1000 / getTimerInterval()) : 0;
    stopTimer();
    w = width;
    h = height;

    destroyRenderTarget();
    displayImage = displayImage.rescaled(w, h);
    createRenderTarget();

    // Stage buffers are zeroed — first copy would blank the display.
    // Skip the staging→display copy once; render() will refill with real content.
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
    if (storageSize <= 0 || count <= 0 || !fifo) return;

    int start1, size1, start2, size2;
    fifo->prepareToWrite(count, start1, size1, start2, size2);
    if (size1 > 0) std::memcpy(fifoBuffer.data() + start1, data, (size_t)size1 * sizeof(float));
    if (size2 > 0) std::memcpy(fifoBuffer.data() + start2, data + size1, (size_t)size2 * sizeof(float));
    fifo->finishedWrite(size1 + size2);
}

void ShaderImage::render()
{
    if (!ready) return;
    VkDevice dev = device_->device();

    int prev = 1 - frameIndex;
    int curr = frameIndex;

    vkWaitForFences(dev, 1, &fence[prev], VK_TRUE, UINT64_MAX);

    if (skipNextStagingCopy)
    {
        skipNextStagingCopy = false;
    }
    else
    {
        juce::Image::BitmapData bd(displayImage, juce::Image::BitmapData::writeOnly);
        const auto* src = static_cast<const uint8_t*>(mappedStagingPtr[prev]);
        for (int y = 0; y < h; y++)
            memcpy(bd.getLinePointer(y), src + y * w * 4, (size_t)w * 4);
    }

    // Drain FIFO into storage descriptor
    for (auto& dsi : descriptorSets)
    {
        if (!dsi.hasStorageBuffer || !dsi.mappedStoragePtr || storageSize <= 0 || !fifo)
            continue;

        int available = fifo->getNumReady();
        int toRead = std::min(available, storageSize);
        if (toRead > 0)
        {
            int start1, size1, start2, size2;
            fifo->prepareToRead(toRead, start1, size1, start2, size2);
            if (size1 > 0) std::memcpy(storageData.data(),         fifoBuffer.data() + start1, (size_t)size1 * sizeof(float));
            if (size2 > 0) std::memcpy(storageData.data() + size1, fifoBuffer.data() + start2, (size_t)size2 * sizeof(float));
            fifo->finishedRead(toRead);
        }
        memcpy(dsi.mappedStoragePtr, storageData.data(), sizeof(float) * (size_t)storageSize);
    }

    vkResetFences(dev, 1, &fence[curr]);
    vkResetCommandBuffer(commandBuffer[curr], 0);

    VkCommandBufferBeginInfo beginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer[curr], &beginInfo);

    VkClearValue clearValue {};
    clearValue.color = {{ 0.0f, 0.0f, 0.0f, 0.0f }};

    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea = {{ 0, 0 }, { (uint32_t)w, (uint32_t)h }};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearValue;

    vkCmdBeginRenderPass(commandBuffer[curr], &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer[curr], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport vp { 0, 0, (float)w, (float)h, 0, 1 };
    vkCmdSetViewport(commandBuffer[curr], 0, 1, &vp);
    VkRect2D sc {{ 0, 0 }, { (uint32_t)w, (uint32_t)h }};
    vkCmdSetScissor(commandBuffer[curr], 0, 1, &sc);

    PushConstants pc;
    pc.resolutionX = (float)w;
    pc.resolutionY = (float)h;
    pc.time        = (float)(juce::Time::getMillisecondCounterHiRes() / 1000.0 - startTime);
    pc.viewportW   = (float)w;
    pc.viewportH   = (float)h;
    pc.regionX     = 0.0f;
    pc.regionY     = 0.0f;
    vkCmdPushConstants(commandBuffer[curr], pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    for (auto& dsi : descriptorSets)
        vkCmdBindDescriptorSets(commandBuffer[curr], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, dsi.set, 1, &dsi.descriptorSet, 0, nullptr);

    vkCmdDraw(commandBuffer[curr], 3, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer[curr]);

    VkBufferImageCopy region {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
    vkCmdCopyImageToBuffer(commandBuffer[curr], colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffer[curr], 1, &region);

    vkEndCommandBuffer(commandBuffer[curr]);

    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer[curr];
    vkQueueSubmit(device_->graphicsQueue(), 1, &submitInfo, fence[curr]);

    frameIndex = prev;
}

} // namespace jvk
