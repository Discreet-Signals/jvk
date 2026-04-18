/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: HSVPipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// HSVPipeline — fullscreen HSV scale/delta post-process.
//
// Sibling of EffectPipeline. One shader (hsv.frag) handles every HSV-space
// operation via the MULTIPLY-then-ADD formula:
//
//   hsv *= (scaleH, scaleS, scaleV);
//   hsv += (deltaH, deltaS, deltaV);
//
// Identity = scales 1.0, deltas 0.0. Common helpers in jvk::Graphics
// (`saturate`, `shiftHue`, ...) just set a subset of the six fields.
//
// Used with a ping-pong pair of framebuffers — pass 1 applies the transform
// A→B; pass 2 runs the identity transform B→A so `current` keeps its
// identity for subsequent scene draws.
// =============================================================================

class HSVPipeline {
public:
    struct PushConstants {
        float viewportW, viewportH;
        float scaleH, scaleS, scaleV;
        float deltaH, deltaS, deltaV;
    };

    HSVPipeline() = default;
    ~HSVPipeline() { destroy(); }

    HSVPipeline(const HSVPipeline&) = delete;
    HSVPipeline& operator=(const HSVPipeline&) = delete;

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

        auto makeModule = [&](std::span<const uint32_t> code) {
            VkShaderModuleCreateInfo ci {};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = code.size() * 4;
            ci.pCode = code.data();
            VkShaderModule m;
            vkCreateShaderModule(d, &ci, nullptr, &m);
            return m;
        };
        VkShaderModule vertMod = makeModule(vertSpv);
        VkShaderModule fragMod = makeModule(fragSpv);

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

    // One HSV pass. Samples `srcDesc`, writes the transformed result into
    // `dstFramebuffer`. Caller is responsible for the ping-pong pair so
    // the final result lands back in "current".
    void applyPass(VkCommandBuffer cmd,
                   VkDescriptorSet srcDesc,
                   VkFramebuffer   dstFramebuffer,
                   VkRenderPass    dstRenderPass,
                   VkExtent2D      extent,
                   const PushConstants& pcData)
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

        PushConstants pc = pcData;
        pc.viewportW = static_cast<float>(extent.width);
        pc.viewportH = static_cast<float>(extent.height);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    // Returns an identity-transform PushConstants (scales=1, deltas=0).
    // Used by the replay-time "pass 2" that copies B→A without modification.
    static PushConstants identity()
    {
        return PushConstants { 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };
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
