#pragma once

namespace jvk {

// =============================================================================
// PathBlurPipeline — analytical-path variable-radius Gaussian blur. Structurally
// identical to ShapeBlurPipeline (separable H/V ping-pong, same blur.vert
// fullscreen-triangle, LOAD-variant render pass), but the fragment shader walks
// a per-frame line-segment SSBO to compute the path SDF instead of sampling an
// analytic shape.
//
// Set 0 — scene source texture (IMAGE_SAMPLER), bound by the dispatcher from
//         the ping-pong source half, identical to ShapeBlur / Blur.
// Set 1 — segment SSBO, shared with PathPipeline — uses its ssboSetLayout()
//         at init time so the descriptor set PathPipeline::ssboDescriptorSet()
//         returns is binding-compatible with this layout.
//
// All distances in push constants are PHYSICAL pixels; the caller converts
// from logical at record time so the fragment shader lives in a single coord
// space (the segments are already physical too).
// =============================================================================

class PathBlurPipeline {
public:
    struct PushConstants {
        float    viewportW, viewportH;  // 8
        float    dirX, dirY;            // 8  (1,0) H pass | (0,1) V pass

        float    maxRadius;             // 4  physical px
        float    falloff;               // 4  physical px
        float    strokeHalfWidth;       // 4  physical px — 0 for fill

        uint32_t segmentStart;          // 4
        uint32_t segmentCount;          // 4
        uint32_t fillRule;              // 4  0=non-zero, 1=even-odd

        int32_t  edgePlacement;         // 4  0=centered, 1=inside, 2=outside
        int32_t  inverted;              // 4  0 or 1

        int32_t  kernelType;              // 4  0 = 1D separable pass, 1 = 2D kernel

        float    _pad0, _pad1, _pad2;   // 12  pad block to 64 bytes
    };                                  // total: 64 bytes

    PathBlurPipeline() = default;
    ~PathBlurPipeline() { destroy(); }

    PathBlurPipeline(const PathBlurPipeline&) = delete;
    PathBlurPipeline& operator=(const PathBlurPipeline&) = delete;

    void init(Device& device,
              VkRenderPass compatibleRenderPass,
              VkDescriptorSetLayout pathSsboSetLayout,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        VkDescriptorSetLayout sceneSetLayout =
            device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);

        VkDescriptorSetLayout setLayouts[2] = { sceneSetLayout, pathSsboSetLayout };

        VkPushConstantRange pcRange {};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 2;
        pli.pSetLayouts = setLayouts;
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
        ds.stencilTestEnable = VK_TRUE;
        // Inside-clip only (Equal), never writes stencil.
        VkStencilOpState so {};
        so.failOp      = VK_STENCIL_OP_KEEP;
        so.passOp      = VK_STENCIL_OP_KEEP;
        so.depthFailOp = VK_STENCIL_OP_KEEP;
        so.compareOp   = VK_COMPARE_OP_EQUAL;
        so.compareMask = 0xFF;
        so.writeMask   = 0x00;
        so.reference   = 0;
        ds.front = so;
        ds.back  = so;

        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb {};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;

        VkDynamicState dyn[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        };
        VkPipelineDynamicStateCreateInfo dynState {};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3;
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

    // One separable pass. Caller provides the scene-source descriptor (set 0)
    // AND the path-SSBO descriptor (set 1 — PathPipeline's current-frame-slot
    // descriptor set). Params contain everything else.
    void applyPass(VkCommandBuffer cmd,
                   VkDescriptorSet srcDesc,
                   VkDescriptorSet pathSsboDesc,
                   VkFramebuffer   dstFramebuffer,
                   VkRenderPass    dstRenderPass,
                   VkExtent2D      extent,
                   VkRect2D        scissor,
                   float           dirX, float dirY,
                   const PushConstants& params,
                   uint32_t        stencilRef = 0)
    {
        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = dstRenderPass;
        rpbi.framebuffer = dstFramebuffer;
        rpbi.renderArea.extent = extent;
        rpbi.clearValueCount = 0;
        rpbi.pClearValues = nullptr;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkDescriptorSet sets[2] = { srcDesc, pathSsboDesc };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 2, sets, 0, nullptr);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilRef);

        PushConstants pc = params;
        pc.viewportW = static_cast<float>(extent.width);
        pc.viewportH = static_cast<float>(extent.height);
        pc.dirX      = dirX;
        pc.dirY      = dirY;
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        VkViewport vp {};
        vp.width    = static_cast<float>(extent.width);
        vp.height   = static_cast<float>(extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

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
