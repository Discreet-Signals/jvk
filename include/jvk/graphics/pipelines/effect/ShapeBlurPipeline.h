#pragma once

namespace jvk {

// =============================================================================
// ShapeBlurPipeline — variable-radius Gaussian blur whose per-pixel radius is
// driven by the signed distance to an analytic shape. Runs the same separable
// H/V ping-pong as EffectPipeline; the only difference is the push-constant
// payload (shape + falloff params) and the fragment shader.
//
// Supports four shape types: axis-aligned rectangle, rounded rectangle,
// ellipse, and capsule (for lines). Each `blur*` API on Graphics packs a
// BlurShapeParams into the arena; Renderer::execute() does the 2-pass replay.
// =============================================================================

class ShapeBlurPipeline {
public:
    struct PushConstants {
        float viewportW, viewportH;   // 8
        float dirX, dirY;             // 8

        float invCol0X, invCol0Y;     // 8  (affine frag.xy → shape-local)
        float invCol1X, invCol1Y;     // 8
        float invCol2X, invCol2Y;     // 8

        float shapeHalfX, shapeHalfY; // 8
        float lineBX, lineBY;         // 8  (line only)

        float maxRadius;              // 4
        float falloff;                // 4
        float displayScale;           // 4
        float cornerRadius;           // 4
        float lineThickness;          // 4

        int   shapeType;              // 4  0=rect,1=rrect,2=ellipse,3=line
        int   edgePlacement;          // 4  0=centered,1=inside,2=outside
        int   inverted;               // 4
    };                                // total: 88 bytes — fits in min-guaranteed 128 B

    ShapeBlurPipeline() = default;
    ~ShapeBlurPipeline() { destroy(); }

    ShapeBlurPipeline(const ShapeBlurPipeline&) = delete;
    ShapeBlurPipeline& operator=(const ShapeBlurPipeline&) = delete;

    void init(Device& device,
              VkRenderPass compatibleRenderPass,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        VkDescriptorSetLayout setLayout =
            device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);

        VkPushConstantRange pcRange {};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

        VkShaderModuleCreateInfo vci {};
        vci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vci.codeSize = vertSpv.size() * 4;
        vci.pCode = vertSpv.data();
        VkShaderModule vertMod;
        vkCreateShaderModule(d, &vci, nullptr, &vertMod);

        VkShaderModuleCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fci.codeSize = fragSpv.size() * 4;
        fci.pCode = fragSpv.data();
        VkShaderModule fragMod;
        vkCreateShaderModule(d, &fci, nullptr, &fragMod);

        VkPipelineShaderStageCreateInfo stages[2] {};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia {};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp {};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster {};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        raster.cullMode = VK_CULL_MODE_NONE;

        VkPipelineMultisampleStateCreateInfo ms {};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds {};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb {};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState {};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2;
        dynState.pDynamicStates = dyn;

        VkGraphicsPipelineCreateInfo pci {};
        pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pci.stageCount = 2;
        pci.pStages = stages;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp;
        pci.pRasterizationState = &raster;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dynState;
        pci.layout = layout_;
        pci.renderPass = compatibleRenderPass;

        vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_);

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);
    }

    // One separable pass. Caller provides direction (1,0) or (0,1) and the
    // pre-packed shape params (everything the shader needs minus viewport
    // size and direction, which we fill in from the extent+dir here).
    void applyPass(VkCommandBuffer cmd,
                   VkDescriptorSet srcDesc,
                   VkFramebuffer   dstFramebuffer,
                   VkRenderPass    dstRenderPass,
                   VkExtent2D      extent,
                   float           dirX, float dirY,
                   const PushConstants& params)
    {
        VkClearValue clear {};
        clear.color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};

        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = dstRenderPass;
        rpbi.framebuffer = dstFramebuffer;
        rpbi.renderArea.extent = extent;
        rpbi.clearValueCount = 1;
        rpbi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp {};
        vp.width    = static_cast<float>(extent.width);
        vp.height   = static_cast<float>(extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);

        VkRect2D sc { {0, 0}, extent };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 1, &srcDesc, 0, nullptr);

        PushConstants pc = params;
        pc.viewportW = static_cast<float>(extent.width);
        pc.viewportH = static_cast<float>(extent.height);
        pc.dirX      = dirX;
        pc.dirY      = dirY;

        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }

private:
    void destroy()
    {
        if (!device_) return;
        VkDevice d = device_->device();
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
        layout_   = VK_NULL_HANDLE;
    }

    Device*          device_   = nullptr;
    VkPipelineLayout layout_   = VK_NULL_HANDLE;
    VkPipeline       pipeline_ = VK_NULL_HANDLE;
};

} // namespace jvk
