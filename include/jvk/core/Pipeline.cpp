namespace jvk {

Pipeline::~Pipeline()
{
    VkDevice d = device_.device();
    variants_.destroy(d);
    if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
}

Pipeline::Pipeline(Pipeline&& o) noexcept
    : device_(o.device_), paintClearsTargets_(o.paintClearsTargets_),
      variants_(std::move(o.variants_)),
      layout_(o.layout_), renderPass_(o.renderPass_), targets_(std::move(o.targets_)),
      vertSpirv_(std::move(o.vertSpirv_)), fragSpirv_(std::move(o.fragSpirv_)),
      built_(o.built_), hasClip_(o.hasClip_)
{
    o.layout_ = VK_NULL_HANDLE;
    o.built_  = false;
}

VkPipeline Pipeline::variant(uint8_t attachment, bool clip, uint8_t live)
{
    if (!built_) return VK_NULL_HANDLE;
    clip = clip && hasClip_;   // no clip config: the normal variant serves both
    const uint8_t clears = paintClears(targets_, attachment, paintClearsTargets_, live);
    return variants_.get(attachment, clip, clears, targets_.size(), [&]
    {
        return buildVariant(clip ? *clipConfig() : config(), renderPass_, layout_, attachment, clears);
    });
}

void Pipeline::loadVertexShader(std::span<const uint32_t> spirv)
{
    vertSpirv_.assign(spirv.begin(), spirv.end());
}

void Pipeline::loadFragmentShader(std::span<const uint32_t> spirv)
{
    fragSpirv_.assign(spirv.begin(), spirv.end());
}

void Pipeline::build(VkRenderPass renderPass, const std::vector<Target>& targets)
{
    if (built_) return;
    VkDevice d = device_.device();
    renderPass_ = renderPass;
    targets_    = targets;

    // Two descriptor sets, both IMAGE_SAMPLER layout:
    //   set 0 = color source (solid default or gradient LUT)
    //   set 1 = shape source (1x1 default, MSDF atlas page, image texture)
    VkDescriptorSetLayout imageSampler = device_.bindings().getLayout(Memory::M::IMAGE_SAMPLER);
    VkDescriptorSetLayout setLayouts[2] = { imageSampler, imageSampler };
    auto cfg = config();

    VkPipelineLayoutCreateInfo pli {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = setLayouts;
    pli.pushConstantRangeCount = static_cast<uint32_t>(cfg.pushConstantRanges.size());
    pli.pPushConstantRanges = cfg.pushConstantRanges.empty() ? nullptr : cfg.pushConstantRanges.data();

    if (vkCreatePipelineLayout(d, &pli, nullptr, &layout_) != VK_SUCCESS) {
        layout_ = VK_NULL_HANDLE;
        return;   // built_ stays false — retried next frame, never bound null
    }

    // Single-sample only. Geometric AA comes from SDF/MSDF shaders per-pixel;
    // path-fill edges and shader-rendered content can optionally be smoothed
    // via a final subpixel-offset supersample effect pass.
    // The defaults: into the main colour, every target live (the frame's
    // common case up to its last target reader).
    const uint8_t clears = paintClears(targets_, 0, paintClearsTargets_, 0xFF);
    VkPipeline normal = buildVariant(cfg, renderPass, layout_, 0, clears);
    if (normal == VK_NULL_HANDLE)
        return;   // built_ stays false
    variants_.add(0, false, clears, normal);

    auto clip = clipConfig();
    hasClip_ = clip.has_value();
    if (clip) {
        VkPipeline clipped = buildVariant(*clip, renderPass, layout_, 0, clears);
        if (clipped == VK_NULL_HANDLE)
            return;
        variants_.add(0, true, clears, clipped);
    }

    built_ = true;
}

VkPipeline Pipeline::buildVariant(const PipelineConfig& cfg, VkRenderPass renderPass,
                                   VkPipelineLayout layout, uint8_t attachment, uint8_t clears)
{
    VkDevice d = device_.device();

    // Shader modules. Null-init + checked: an uninitialized handle passed to
    // pipeline creation / destroy on the failure path is UB.
    VkShaderModuleCreateInfo vci {};
    vci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vci.codeSize = vertSpirv_.size() * 4;
    vci.pCode = vertSpirv_.data();
    VkShaderModule vertModule = VK_NULL_HANDLE;
    vkCreateShaderModule(d, &vci, nullptr, &vertModule);

    VkShaderModuleCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fci.codeSize = fragSpirv_.size() * 4;
    fci.pCode = fragSpirv_.data();
    VkShaderModule fragModule = VK_NULL_HANDLE;
    vkCreateShaderModule(d, &fci, nullptr, &fragModule);

    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        if (vertModule != VK_NULL_HANDLE) vkDestroyShaderModule(d, vertModule, nullptr);
        if (fragModule != VK_NULL_HANDLE) vkDestroyShaderModule(d, fragModule, nullptr);
        return VK_NULL_HANDLE;
    }

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

    VkVertexInputAttributeDescription attrs[5] {};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) };
    attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) };
    attrs[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) };
    attrs[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, gradientInfo) };

    VkPipelineVertexInputStateCreateInfo vertexInput {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 5;
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
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

    // This draw's blend on the attachment it lands on, the rest per DrawTargets.
    DrawTargets drawTargets(targets_, attachment, blend, clears);
    stages[1].pSpecializationInfo = &drawTargets.specialization.info;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
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
    pci.pColorBlendState = &drawTargets.colorBlend;
    pci.pDynamicState = &dynamicState;
    pci.layout = layout;
    pci.renderPass = renderPass;

    VkPipeline result = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(d, device_.pipelineCache(), 1, &pci, nullptr, &result);

    vkDestroyShaderModule(d, vertModule, nullptr);
    vkDestroyShaderModule(d, fragModule, nullptr);

    return result;
}

} // namespace jvk
