#pragma once

namespace jvk {

class Shader {
public:
    Shader() = default;

    void load(std::span<const uint32_t> fragSpirv)
    {
        spirv_.assign(fragSpirv.begin(), fragSpirv.end());
        reflectShader();
    }

    // Image bindings: ensure the image is registered in ResourceCaches, then
    // stash its view+sampler on the binding. If the shader is already live
    // the write is applied immediately; otherwise ensureCreated() picks it up.
    void set(const juce::String& name, const juce::Image& image, ResourceCaches& caches)
    {
        for (auto& b : bindings_) {
            if (b.name == name && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                uint64_t hash = ResourceCaches::hashImage(image);
                caches.getTexture(hash, image);
                auto* tc = caches.textures().find(hash);
                if (tc == nullptr) return;
                b.imageView = tc->image.view();
                b.sampler   = tc->image.sampler();
                b.bound     = true;
                if (created_ && descriptorSet_ != VK_NULL_HANDLE)
                    Memory::M::writeImage(device_->device(), descriptorSet_,
                                          b.binding, b.imageView, b.sampler);
                return;
            }
        }
    }

    // Float bindings: stored locally, written to V during replay
    void set(const juce::String& name, float value)
    {
        for (auto& b : bindings_) {
            if (b.name == name && b.offsetInBuffer < uniformData_.size() * sizeof(float)) {
                uniformData_[b.offsetInBuffer / sizeof(float)] = value;
                return;
            }
        }
    }

    void set(const juce::String& name, std::span<const float> data)
    {
        for (auto& b : bindings_) {
            if (b.name == name && b.offsetInBuffer + data.size_bytes() <= uniformData_.size() * sizeof(float)) {
                memcpy(&uniformData_[b.offsetInBuffer / sizeof(float)], data.data(), data.size_bytes());
                return;
            }
        }
    }

    void ensureCreated(Device& device, VkRenderPass renderPass, VkSampleCountFlagBits msaa)
    {
        if (created_) return;
        device_ = &device;
        VkDevice d = device.device();

        // Create pipeline layout from reflected bindings. Shaders with no
        // reflected bindings (e.g. fragment-only effects driven by push
        // constants) build a layout with zero descriptor sets.
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        for (auto& b : bindings_) {
            layoutBindings.push_back({
                b.binding, b.type, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
            });
        }

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        if (!layoutBindings.empty()) {
            layoutId_ = device.bindings().registerLayout(layoutBindings.data(),
                static_cast<uint32_t>(layoutBindings.size()));
            descriptorSet_ = device.bindings().alloc(layoutId_);
            setLayout = device.bindings().getLayout(layoutId_);

            // Bind defaults (1x1 black pixel) for unset image bindings so the
            // descriptor slot is never sampled uninitialized.
            for (auto& b : bindings_) {
                if (!b.bound && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    b.imageView = device.caches().defaultImageView();
                    b.sampler   = device.caches().defaultSampler();
                }
            }

            // Back the reflected uniform/storage blocks with one host-visible
            // coherent VkBuffer sized to the total reflected block bytes (the
            // same total `uniformData_` is sized for). Each UBO/SSBO binding
            // gets a descriptor write pointing at its slice via offset+range.
            // Per-draw we memcpy uniformData_ → mapped pointer in
            // ShaderPipeline::dispatch so set(name, value) reaches the GPU.
            const VkDeviceSize bufferSize = uniformData_.size() * sizeof(float);
            if (bufferSize > 0) {
                VkBufferCreateInfo bci {};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size  = bufferSize;
                // Mark the buffer with both UBO and SSBO usage so a single
                // backing buffer can serve every reflected block, regardless
                // of whether the user shader declared it as `uniform` or
                // `buffer`. The descriptor type drives how the GPU reads it.
                bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                vkCreateBuffer(d, &bci, nullptr, &uniformBuffer_);

                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(d, uniformBuffer_, &req);
                VkMemoryAllocateInfo ai {};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize  = req.size;
                ai.memoryTypeIndex = Memory::findMemoryType(device.physicalDevice(),
                    req.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                vkAllocateMemory(d, &ai, nullptr, &uniformMemory_);
                vkBindBufferMemory(d, uniformBuffer_, uniformMemory_, 0);
                vkMapMemory(d, uniformMemory_, 0, bufferSize, 0, &uniformMapped_);

                // Initial copy so any set() calls made before ensureCreated
                // are visible on the GPU's first read.
                std::memcpy(uniformMapped_, uniformData_.data(), bufferSize);
            }

            // Wire the descriptor set: one write per binding so the shader
            // sees its UBO/SSBO buffers and image samplers as soon as it's
            // bound. Image bindings either use the user-supplied descriptor
            // (set via set(name, image, caches)) or the default 1x1 fallback.
            std::vector<VkWriteDescriptorSet>   writes;
            std::vector<VkDescriptorBufferInfo> bufferInfos;
            std::vector<VkDescriptorImageInfo>  imageInfos;
            bufferInfos.reserve(bindings_.size());
            imageInfos.reserve(bindings_.size());
            for (auto& b : bindings_) {
                if (b.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    bufferInfos.push_back({ uniformBuffer_, b.offsetInBuffer, b.sizeInBuffer });
                    VkWriteDescriptorSet w {};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = descriptorSet_;
                    w.dstBinding = b.binding;
                    w.descriptorType = b.type;
                    w.descriptorCount = 1;
                    w.pBufferInfo = &bufferInfos.back();
                    writes.push_back(w);
                }
                else if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    imageInfos.push_back({ b.sampler, b.imageView,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
                    VkWriteDescriptorSet w {};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = descriptorSet_;
                    w.dstBinding = b.binding;
                    w.descriptorType = b.type;
                    w.descriptorCount = 1;
                    w.pImageInfo = &imageInfos.back();
                    writes.push_back(w);
                }
            }
            if (!writes.empty())
                vkUpdateDescriptorSets(d, static_cast<uint32_t>(writes.size()),
                                       writes.data(), 0, nullptr);
        }

        // Create pipeline (fullscreen triangle, fragment-only shader)
        // Uses the shader_region vertex shader (generates fullscreen tri from gl_VertexIndex)
        std::vector<uint32_t> fullscreenVert(
            reinterpret_cast<const uint32_t*>(shaders::shader_region::vert_spv),
            reinterpret_cast<const uint32_t*>(shaders::shader_region::vert_spv) + shaders::shader_region::vert_spvSize / 4);

        VkShaderModuleCreateInfo vci {};
        vci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vci.codeSize = fullscreenVert.size() * 4;
        vci.pCode = fullscreenVert.data();
        VkShaderModule vertMod;
        vkCreateShaderModule(d, &vci, nullptr, &vertMod);

        VkShaderModuleCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fci.codeSize = spirv_.size() * 4;
        fci.pCode = spirv_.data();
        VkShaderModule fragMod;
        vkCreateShaderModule(d, &fci, nullptr, &fragMod);

        // Push constant layout mirrors shader_region.vert:
        //   bytes  0..11 — resolution (vec2) + time (float) — vertex + fragment
        //   bytes 12..27 — viewport (vec2) + region origin (vec2) — vertex only
        // One unified range covers both stages; fragment just reads the head.
        VkPushConstantRange pushRange {
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(float) * 7
        };

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = (setLayout != VK_NULL_HANDLE) ? 1u : 0u;
        pli.pSetLayouts = (setLayout != VK_NULL_HANDLE) ? &setLayout : nullptr;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pushRange;
        vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

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

        VkPipelineInputAssemblyStateCreateInfo inputAsm {};
        inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vpState {};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster {};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        raster.cullMode = VK_CULL_MODE_NONE;

        VkPipelineMultisampleStateCreateInfo ms {};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = msaa;

        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo cb {};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;

        // Stencil reference is pushed per-draw (= current clip depth). We
        // no longer mask bits per-level — stencilCompareMask stays at the
        // default 0xFF so the EQUAL test checks the full depth counter.
        VkDynamicState dynStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        };
        VkPipelineDynamicStateCreateInfo dynState {};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3;
        dynState.pDynamicStates = dynStates;

        // Build two variants sharing one layout: the normal variant for
        // stencilDepth==0, and a clip variant that matches on the active
        // stencil bits so shader draws respect path clips just like ColorOps.
        auto buildVariant = [&](bool stencilTest) -> VkPipeline
        {
            VkPipelineDepthStencilStateCreateInfo ds {};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.stencilTestEnable = stencilTest ? VK_TRUE : VK_FALSE;
            if (stencilTest) {
                VkStencilOpState op {};
                op.failOp      = VK_STENCIL_OP_KEEP;
                op.passOp      = VK_STENCIL_OP_KEEP;
                op.depthFailOp = VK_STENCIL_OP_KEEP;
                op.compareOp   = VK_COMPARE_OP_EQUAL;
                op.compareMask = 0xFF; // overridden by dynamic state per draw
                op.writeMask   = 0xFF;
                op.reference   = 0;
                ds.front = op;
                ds.back  = op;
            }

            VkGraphicsPipelineCreateInfo pci {};
            pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pci.stageCount = 2;
            pci.pStages = stages;
            pci.pVertexInputState = &vertexInput;
            pci.pInputAssemblyState = &inputAsm;
            pci.pViewportState = &vpState;
            pci.pRasterizationState = &raster;
            pci.pMultisampleState = &ms;
            pci.pDepthStencilState = &ds;
            pci.pColorBlendState = &cb;
            pci.pDynamicState = &dynState;
            pci.layout = layout_;
            pci.renderPass = renderPass;

            VkPipeline result = VK_NULL_HANDLE;
            vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &pci, nullptr, &result);
            return result;
        };

        pipeline_     = buildVariant(false);
        clipPipeline_ = buildVariant(true);

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);

        created_ = true;
    }

    bool isReady() const { return created_; }

    VkPipeline       pipeline()      const { return pipeline_; }
    VkPipeline       clipPipeline()  const { return clipPipeline_ ? clipPipeline_ : pipeline_; }
    VkPipelineLayout layout()        const { return layout_; }
    VkDescriptorSet  descriptorSet() const { return descriptorSet_; }

    const float* uniformData()   const { return uniformData_.data(); }
    size_t       uniformSize()   const { return uniformData_.size() * sizeof(float); }
    // Persistently-mapped pointer to the GPU-visible uniform/storage buffer
    // backing every reflected block. Null if the shader declared no UBO/SSBO
    // bindings. ShaderPipeline::dispatch memcpys uniformData_ into this each
    // draw so set(name, value) propagates to the GPU.
    void*        uniformMapped() const { return uniformMapped_; }

    ~Shader()
    {
        if (!device_) return;
        VkDevice d = device_->device();
        if (uniformMapped_  != nullptr)        vkUnmapMemory(d, uniformMemory_);
        if (uniformBuffer_  != VK_NULL_HANDLE) vkDestroyBuffer(d, uniformBuffer_, nullptr);
        if (uniformMemory_  != VK_NULL_HANDLE) vkFreeMemory(d, uniformMemory_, nullptr);
        if (pipeline_       != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_,     nullptr);
        if (clipPipeline_   != VK_NULL_HANDLE) vkDestroyPipeline(d, clipPipeline_, nullptr);
        if (layout_         != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (descriptorSet_  != VK_NULL_HANDLE) device_->bindings().free(descriptorSet_);
    }

private:
    struct BindingInfo {
        juce::String     name;
        uint32_t         binding;
        VkDescriptorType type;
        uint32_t         offsetInBuffer = 0;
        uint32_t         sizeInBuffer = 0;
        VkImageView      imageView = VK_NULL_HANDLE;
        VkSampler        sampler   = VK_NULL_HANDLE;
        bool             bound = false;
    };

    void reflectShader()
    {
        // Use SPIRV-Reflect to discover bindings
        SpvReflectShaderModule module;
        SpvReflectResult result = spvReflectCreateShaderModule(
            spirv_.size() * 4, spirv_.data(), &module);
        if (result != SPV_REFLECT_RESULT_SUCCESS) return;

        uint32_t count = 0;
        spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
        std::vector<SpvReflectDescriptorBinding*> reflBindings(count);
        spvReflectEnumerateDescriptorBindings(&module, &count, reflBindings.data());

        uint32_t bufferOffset = 0;
        for (auto* rb : reflBindings) {
            BindingInfo info;
            info.name = rb->name ? rb->name : "";
            info.binding = rb->binding;
            info.type = static_cast<VkDescriptorType>(rb->descriptor_type);

            if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                info.offsetInBuffer = bufferOffset;
                info.sizeInBuffer = rb->block.size;
                bufferOffset += rb->block.size;
            }
            bindings_.push_back(info);
        }

        uniformData_.resize(bufferOffset / sizeof(float), 0.0f);
        spvReflectDestroyShaderModule(&module);
    }

    std::vector<BindingInfo>  bindings_;
    std::vector<uint32_t>     spirv_;
    std::vector<float>        uniformData_;

    Device*          device_        = nullptr;
    VkPipeline       pipeline_      = VK_NULL_HANDLE;
    VkPipeline       clipPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout layout_        = VK_NULL_HANDLE;
    VkDescriptorSet  descriptorSet_ = VK_NULL_HANDLE;
    Memory::M::LayoutID layoutId_   = 0;

    // Single host-visible coherent buffer backing every reflected UBO/SSBO
    // block. NOTE: not double-buffered — fine for shaders drawn once per
    // frame whose uniforms drift slowly (e.g. a `time` value); for shaders
    // that draw multiple times per frame with different per-draw uniforms,
    // a per-frame-slot ring would be needed to avoid GPU-vs-CPU races.
    VkBuffer       uniformBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory_ = VK_NULL_HANDLE;
    void*          uniformMapped_ = nullptr;

    bool   created_   = false;
};

} // namespace jvk
