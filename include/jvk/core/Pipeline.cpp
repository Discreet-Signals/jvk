namespace jvk {

Pipeline::~Pipeline()
{
    VkDevice d = device_.device();
    if (pipeline_ != VK_NULL_HANDLE)     vkDestroyPipeline(d, pipeline_, nullptr);
    if (clipPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(d, clipPipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)       vkDestroyPipelineLayout(d, layout_, nullptr);
}

Pipeline::Pipeline(Pipeline&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_), clipPipeline_(o.clipPipeline_),
      layout_(o.layout_), vertSpirv_(std::move(o.vertSpirv_)),
      fragSpirv_(std::move(o.fragSpirv_)), built_(o.built_)
{
    o.pipeline_ = VK_NULL_HANDLE;
    o.clipPipeline_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.built_ = false;
}

void Pipeline::loadVertexShader(std::span<const uint32_t> spirv)
{
    vertSpirv_.assign(spirv.begin(), spirv.end());
}

void Pipeline::loadFragmentShader(std::span<const uint32_t> spirv)
{
    fragSpirv_.assign(spirv.begin(), spirv.end());
}

void Pipeline::defineLayout(Memory::M& bindings)
{
    // Default: 1 combined image sampler at binding 0 (already registered as IMAGE_SAMPLER)
}

void Pipeline::build(VkRenderPass renderPass, VkSampleCountFlagBits msaa)
{
    if (built_) return;
    VkDevice d = device_.device();

    // Create pipeline layout using the default image sampler layout
    VkDescriptorSetLayout setLayout = device_.bindings().getLayout(Memory::M::IMAGE_SAMPLER);
    auto cfg = config();

    VkPipelineLayoutCreateInfo pli {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout;
    pli.pushConstantRangeCount = static_cast<uint32_t>(cfg.pushConstantRanges.size());
    pli.pPushConstantRanges = cfg.pushConstantRanges.empty() ? nullptr : cfg.pushConstantRanges.data();

    vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

    pipeline_ = buildVariant(cfg, renderPass, msaa, layout_);

    auto clip = clipConfig();
    if (clip)
        clipPipeline_ = buildVariant(*clip, renderPass, msaa, layout_);

    built_ = true;
}

VkPipeline Pipeline::buildVariant(const PipelineConfig& cfg, VkRenderPass renderPass,
                                   VkSampleCountFlagBits msaa, VkPipelineLayout layout)
{
    VkDevice d = device_.device();

    // Shader modules
    VkShaderModuleCreateInfo vci {};
    vci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vci.codeSize = vertSpirv_.size() * 4;
    vci.pCode = vertSpirv_.data();
    VkShaderModule vertModule;
    vkCreateShaderModule(d, &vci, nullptr, &vertModule);

    VkShaderModuleCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fci.codeSize = fragSpirv_.size() * 4;
    fci.pCode = fragSpirv_.data();
    VkShaderModule fragModule;
    vkCreateShaderModule(d, &fci, nullptr, &fragModule);

    VkPipelineShaderStageCreateInfo stages[2] {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Vertex input — UIVertex layout
    VkVertexInputBindingDescription binding {};
    binding.stride = sizeof(UIVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4] {};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) };
    attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) };
    attrs[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) };

    VkPipelineVertexInputStateCreateInfo vertexInput {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAsm {};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = cfg.topology;

    VkPipelineViewportStateCreateInfo viewportState {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster {};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    raster.cullMode = cfg.cullMode;
    raster.frontFace = cfg.frontFace;

    VkPipelineMultisampleStateCreateInfo multisample {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = msaa;

    // Depth/stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = cfg.depthTestEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = cfg.depthWriteEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = cfg.depthCompareOp;
    depthStencil.stencilTestEnable = cfg.stencilTestEnable ? VK_TRUE : VK_FALSE;

    VkStencilOpState stencilOp {};
    stencilOp.failOp = cfg.stencilFailOp;
    stencilOp.passOp = cfg.stencilPassOp;
    stencilOp.depthFailOp = cfg.stencilDepthFailOp;
    stencilOp.compareOp = cfg.stencilCompareOp;
    stencilOp.compareMask = cfg.stencilCompareMask;
    stencilOp.writeMask = cfg.stencilWriteMask;
    stencilOp.reference = cfg.stencilReference;
    depthStencil.front = stencilOp;
    depthStencil.back = cfg.separateBackStencil ?
        VkStencilOpState {
            cfg.stencilBackFailOp, cfg.stencilBackPassOp, cfg.stencilBackDepthFailOp,
            cfg.stencilBackCompareOp, cfg.stencilCompareMask, cfg.stencilWriteMask,
            cfg.stencilReference
        } : stencilOp;

    // Color blend
    VkPipelineColorBlendAttachmentState blend {};
    blend.colorWriteMask = cfg.colorWriteEnable
        ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
        : 0;

    switch (cfg.blendMode) {
        case BlendMode::Opaque:
            blend.blendEnable = VK_FALSE; break;
        case BlendMode::AlphaBlend:
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp = VK_BLEND_OP_ADD; break;
        case BlendMode::Additive:
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp = VK_BLEND_OP_ADD; break;
        case BlendMode::Premultiplied:
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend.alphaBlendOp = VK_BLEND_OP_ADD; break;
        case BlendMode::Multiply:
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp = VK_BLEND_OP_ADD; break;
        case BlendMode::Screen:
            blend.blendEnable = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            blend.colorBlendOp = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.alphaBlendOp = VK_BLEND_OP_ADD; break;
    }

    VkPipelineColorBlendStateCreateInfo colorBlend {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE
    };
    VkPipelineDynamicStateCreateInfo dynamicState {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pci {};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vertexInput;
    pci.pInputAssemblyState = &inputAsm;
    pci.pViewportState = &viewportState;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pDepthStencilState = &depthStencil;
    pci.pColorBlendState = &colorBlend;
    pci.pDynamicState = &dynamicState;
    pci.layout = layout;
    pci.renderPass = renderPass;

    VkPipeline result = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &pci, nullptr, &result);

    vkDestroyShaderModule(d, vertModule, nullptr);
    vkDestroyShaderModule(d, fragModule, nullptr);

    return result;
}

} // namespace jvk
