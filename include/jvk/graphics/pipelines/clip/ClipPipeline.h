/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: ClipPipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// ClipPipeline — stencil-based clip push/pop via analytical SDF rasterisation.
//
// For each user clipToPath() (rounded rect or arbitrary path), we record a
// DrawOp::PushClipPath carrying a ClipShapeParams. At replay, this pipeline
// rasterises the clip shape's cover quad: the fragment shader evaluates the
// shape's SDF and `discard`s outside fragments. Surviving fragments trigger
// a fixed-function stencil INCR (push) or DECR (pop), so the stencil buffer
// ends up holding "number of active clips containing this pixel". Subsequent
// draws pass the stencil test only where stencil == current clip depth.
//
// Two VkPipelines share one layout + one pair of shader modules:
//   pushPipeline_   stencilPassOp = INCR_WRAP  (cmp = EQUAL, ref = depth-1)
//   popPipeline_    stencilPassOp = DECR_WRAP  (cmp = EQUAL, ref = depth)
//
// Descriptor layout:
//   set 0, binding 0   storage buffer (path segments, shared with PathPipeline)
//
// The SSBO is bound at dispatch time from the PathPipeline's descriptor set —
// no separate buffer, no duplicate upload. For rrect clips the SSBO isn't
// read (shape is computed analytically from push constants), but the binding
// is still required for pipeline-layout compatibility.
// =============================================================================

class ClipPipeline {
public:
    struct PushConstants {
        float    viewportW, viewportH;   // 0..8
        float    centerX,   centerY;     // 8..16   — rrect center (physical px)
        float    halfW,     halfH;       // 16..24  — rrect half-extent
        float    cornerRadius;           // 24..28
        uint32_t shapeType;              // 28..32  — 1 = rrect, 2 = path
        uint32_t segmentStart;           // 32..36  — path only
        uint32_t segmentCount;           // 36..40  — path only
        uint32_t fillRule;               // 40..44  — 0 = non-zero, 1 = even-odd
        uint32_t _pad;                   // 44..48
    };

    ClipPipeline() = default;
    ~ClipPipeline() { destroy(); }

    ClipPipeline(const ClipPipeline&) = delete;
    ClipPipeline& operator=(const ClipPipeline&) = delete;

    void init(Device& device,
              VkRenderPass sceneRenderPass,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        // Set 0 — storage buffer binding that matches PathPipeline's ssbo
        // descriptor set layout exactly (identically defined = compatible
        // under Vulkan descriptor-set-compatibility rules).
        VkDescriptorSetLayoutBinding ssboBinding {};
        ssboBinding.binding = 0;
        ssboBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ssboBinding.descriptorCount = 1;
        ssboBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dli {};
        dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = 1;
        dli.pBindings = &ssboBinding;
        vkCreateDescriptorSetLayout(d, &dli, nullptr, &ssboSetLayout_);

        // Pipeline layout: 1 set + push constants for vertex+fragment.
        VkPushConstantRange pc {};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset = 0;
        pc.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &ssboSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

        // Build push + pop variants. Both share the same shader modules,
        // layout, blend state; only the stencilPassOp differs.
        pushPipeline_ = buildVariant(sceneRenderPass, vertSpv, fragSpv,
                                     VK_STENCIL_OP_INCREMENT_AND_WRAP);
        popPipeline_  = buildVariant(sceneRenderPass, vertSpv, fragSpv,
                                     VK_STENCIL_OP_DECREMENT_AND_WRAP);
    }

    bool ready() const { return pushPipeline_ != VK_NULL_HANDLE; }

    // PushClip — INCR the stencil where the shape is inside. Called with
    // `parentDepth` as the stencil compare reference (only pixels already
    // at the parent clip level INCR, so inner clips intersect outer clips
    // automatically).
    void pushClip(State& state, VkCommandBuffer cmd, Renderer& r,
                  const DrawCommand& drawCmd,
                  const PushConstants& pcData,
                  const juce::Rectangle<float>& coverRect,
                  VkDescriptorSet ssboDesc,
                  uint32_t parentDepth,
                  float viewportW, float viewportH)
    {
        dispatch(state, cmd, r, drawCmd, pushPipeline_, pcData, coverRect,
                 ssboDesc, parentDepth, viewportW, viewportH);
    }

    // PopClip — DECR the stencil at the same fragments PushClip incremented.
    // Called with `currentDepth` (the depth before CPU-side depth--) as the
    // stencil compare reference.
    void popClip(State& state, VkCommandBuffer cmd, Renderer& r,
                 const DrawCommand& drawCmd,
                 const PushConstants& pcData,
                 const juce::Rectangle<float>& coverRect,
                 VkDescriptorSet ssboDesc,
                 uint32_t currentDepth,
                 float viewportW, float viewportH)
    {
        dispatch(state, cmd, r, drawCmd, popPipeline_, pcData, coverRect,
                 ssboDesc, currentDepth, viewportW, viewportH);
    }

private:
    void dispatch(State& state, VkCommandBuffer cmd, Renderer& /*r*/,
                  const DrawCommand& drawCmd,
                  VkPipeline pipeline,
                  const PushConstants& pcData,
                  const juce::Rectangle<float>& coverRect,
                  VkDescriptorSet ssboDesc,
                  uint32_t stencilRef,
                  float viewportW, float viewportH)
    {
        if (!ready()) return;

        state.setCustomPipeline(pipeline, layout_);

        // Scissor — command's clipBounds (parent clip intersected with the
        // cover rect) was already computed on record side.
        auto clip = drawCmd.clipBounds;
        int x0 = std::max(0, clip.getX());
        int y0 = std::max(0, clip.getY());
        int x1 = std::max(x0, clip.getRight());
        int y1 = std::max(y0, clip.getBottom());
        VkRect2D sc { { x0, y0 },
                      { static_cast<uint32_t>(x1 - x0),
                        static_cast<uint32_t>(y1 - y0) } };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        VkViewport vp {};
        vp.width    = viewportW;
        vp.height   = viewportH;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);

        // Stencil reference — which clip depth we compare against.
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilRef);

        // Bind the shared path-segment SSBO at set 0.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 1, &ssboDesc, 0, nullptr);

        // Push constants — viewport + shape geometry + (for paths) segment range.
        PushConstants pc = pcData;
        pc.viewportW = viewportW;
        pc.viewportH = viewportH;
        vkCmdPushConstants(cmd, layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        // Emit a 6-vertex cover quad in physical pixels — populate only
        // `position`; the clip vertex shader ignores the other UIVertex
        // attribs so we leave them zero.
        float x = coverRect.getX(),     y = coverRect.getY();
        float w = coverRect.getWidth(), h = coverRect.getHeight();
        auto mkv = [](float px, float py) {
            return UIVertex {
                glm::vec2 { px, py },
                glm::vec4 { 0.0f },
                glm::vec2 { 0.0f },
                glm::vec4 { 0.0f },
                glm::vec4 { 0.0f }
            };
        };
        UIVertex verts[6] = {
            mkv(x,     y    ),
            mkv(x + w, y    ),
            mkv(x + w, y + h),
            mkv(x,     y    ),
            mkv(x + w, y + h),
            mkv(x,     y + h),
        };
        state.draw(drawCmd, verts, 6);

        // State doesn't track this pipeline/layout/descriptor combination —
        // force full rebind on the next normal draw.
        state.invalidate();
    }

    VkPipeline buildVariant(VkRenderPass renderPass,
                            std::span<const uint32_t> vertSpv,
                            std::span<const uint32_t> fragSpv,
                            VkStencilOp stencilPassOp)
    {
        VkDevice d = device_->device();

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

        // Vertex input — UIVertex layout, but the clip vert shader only
        // reads `position` (location 0). The other attribs are declared
        // so the pipeline is compatible with the shared vertex ring.
        VkVertexInputBindingDescription binding {};
        binding.stride = sizeof(UIVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[5] {};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) };
        attrs[1] = { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UIVertex, color) };
        attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, uv) };
        attrs[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UIVertex, shapeInfo) };
        attrs[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UIVertex, gradientInfo) };

        VkPipelineVertexInputStateCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 5;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia {};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

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
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Stencil: cmp = EQUAL, passOp = INCR_WRAP (push) or DECR_WRAP (pop).
        // failOp = KEEP (fragments outside the parent clip don't touch the
        // stencil). depthFailOp = KEEP. Reference is set dynamically.
        VkStencilOpState so {};
        so.failOp      = VK_STENCIL_OP_KEEP;
        so.passOp      = stencilPassOp;
        so.depthFailOp = VK_STENCIL_OP_KEEP;
        so.compareOp   = VK_COMPARE_OP_EQUAL;
        so.compareMask = 0xFF;
        so.writeMask   = 0xFF;
        so.reference   = 0; // dynamic state — set per dispatch

        VkPipelineDepthStencilStateCreateInfo ds {};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.stencilTestEnable = VK_TRUE;
        ds.front = so;
        ds.back  = so;

        // Color writes OFF — we only touch the stencil.
        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = 0;
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

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);
        return result;
    }

    void destroy()
    {
        if (!device_) return;
        VkDevice d = device_->device();
        if (pushPipeline_   != VK_NULL_HANDLE) vkDestroyPipeline(d, pushPipeline_, nullptr);
        if (popPipeline_    != VK_NULL_HANDLE) vkDestroyPipeline(d, popPipeline_, nullptr);
        if (layout_         != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (ssboSetLayout_  != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, ssboSetLayout_, nullptr);
        device_ = nullptr;
    }

    Device*               device_        = nullptr;
    VkPipelineLayout      layout_        = VK_NULL_HANDLE;
    VkPipeline            pushPipeline_  = VK_NULL_HANDLE;
    VkPipeline            popPipeline_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssboSetLayout_ = VK_NULL_HANDLE;
};

} // namespace jvk
