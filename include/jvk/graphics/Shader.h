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

    // Image bindings: resolved through ResourceCaches immediately
    void set(const juce::String& name, const juce::Image& image, ResourceCaches& caches)
    {
        for (auto& b : bindings_) {
            if (b.name == name && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                uint64_t hash = reinterpret_cast<uint64_t>(image.getPixelData().get());
                auto* tc = caches.textures().find(hash);
                auto desc = tc ? tc->descriptorSet : VK_NULL_HANDLE;
                if (desc != VK_NULL_HANDLE) {
                    b.boundDescriptor = desc;
                    b.bound = true;
                }
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

        // Create pipeline layout from reflected bindings
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        for (auto& b : bindings_) {
            layoutBindings.push_back({
                b.binding, b.type, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
            });
        }

        if (!layoutBindings.empty()) {
            layoutId_ = device.bindings().registerLayout(layoutBindings.data(), static_cast<uint32_t>(layoutBindings.size()));
            descriptorSet_ = device.bindings().alloc(layoutId_);

            // Bind defaults (1x1 black pixel) for unset image bindings
            for (auto& b : bindings_) {
                if (!b.bound && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    b.boundDescriptor = device.caches().defaultDescriptor();
            }
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

        VkDescriptorSetLayout setLayout = device.bindings().getLayout(layoutId_);
        VkPushConstantRange pushRange { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        0, sizeof(float) * 4 };

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout;
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

        VkPipelineDepthStencilStateCreateInfo ds {};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.stencilTestEnable = VK_TRUE;

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

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE };
        VkPipelineDynamicStateCreateInfo dynState {};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3;
        dynState.pDynamicStates = dynStates;

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

        vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_);

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);

        startTime_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        created_ = true;
    }

    bool isReady() const { return created_; }

    VkPipeline       pipeline()      const { return pipeline_; }
    VkPipelineLayout layout()        const { return layout_; }
    VkDescriptorSet  descriptorSet() const { return descriptorSet_; }

    const float* uniformData() const { return uniformData_.data(); }
    size_t       uniformSize() const { return uniformData_.size() * sizeof(float); }

    ~Shader()
    {
        if (!device_) return;
        VkDevice d = device_->device();
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (descriptorSet_ != VK_NULL_HANDLE) device_->bindings().free(descriptorSet_);
    }

private:
    struct BindingInfo {
        juce::String     name;
        uint32_t         binding;
        VkDescriptorType type;
        uint32_t         offsetInBuffer = 0;
        uint32_t         sizeInBuffer = 0;
        VkDescriptorSet  boundDescriptor = VK_NULL_HANDLE;
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
    VkPipelineLayout layout_        = VK_NULL_HANDLE;
    VkDescriptorSet  descriptorSet_ = VK_NULL_HANDLE;
    Memory::M::LayoutID layoutId_   = 0;

    double startTime_ = 0;
    bool   created_   = false;
};

} // namespace jvk
