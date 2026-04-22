/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: FillPipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// FillPipeline — geometry-abstracted 2D fill/stroke/draw pipeline.
//
// Consolidates the legacy ColorPipeline + PathPipeline dispatch paths behind
// the shared GeometryPrimitive format. Every non-effect 2D op (FillRect,
// FillRoundedRect, FillEllipse, StrokeRoundedRect, StrokeEllipse, DrawImage,
// DrawGlyphs, DrawLine, FillPath) funnels through one VkPipeline + one pair
// of shader modules (geometry.vert + fill.frag); per-instance behaviour
// branches on the primitive's `geometryTag`.
//
// Style-B instanced dispatch: `vkCmdDraw(6, primCount, 0, firstInstance)`.
// No vertex buffer binding; the vertex shader expands bbox → quad corner
// via gl_VertexIndex. Single-primitive dispatches today (primCount = 1);
// Phase C fusion will collapse runs of compatible primitives into larger
// batches with no further shader / pipeline changes.
//
// Descriptor layout (pipeline stays bound once per scene RP; only sets 0
// and 3 rebind per-dispatch, and only when the resource changes):
//
//   set 0, binding 0   sampler2D   color LUT — gradient atlas row for
//                                  gradient fills, 1×1 default for solid.
//                                  Rebound per dispatch via
//                                  ResourceCaches::{gradient,default}Descriptor.
//   set 1, binding 0   SSBO        GeometryPrimitive[] — OWNED, per-frame
//                                  staging + flush mirroring PathPipeline's
//                                  segment SSBO pattern.
//   set 2, binding 0   SSBO        Path segments — SHARED with PathPipeline
//                                  via its ssboSetLayout(); the fragment
//                                  shader's tag=5 branch walks it.
//   set 3, binding 0   sampler2D   Shape sampler — MSDF glyph atlas page for
//                                  tag 4, image texture for tag 6, 1×1
//                                  default otherwise. Rebound per dispatch.
//
// Stencil is always-on with EQUAL compare against a dynamic reference. A
// dispatch outside any clip sets ref=0; inside clip depth D sets ref=D.
// Matches BlurPipeline's convention so both pipelines share one mental model.
// =============================================================================

class FillPipeline {
public:
    struct PushConstants {
        float viewportW, viewportH;  // 8
        float _pad0, _pad1;          // 8  — keep block 16-byte aligned
        int   _pad2, _pad3, _pad4, _pad5;
    };                                // total: 32 bytes

    static constexpr VkDeviceSize INITIAL_BUFFER_BYTES = 256 * 1024;   // ≈ 2048 primitives
    static constexpr uint32_t     PRIMITIVE_STRIDE     = 128;           // sizeof(GeometryPrimitive)
    static constexpr int          MAX_FRAMES           = 2;

    FillPipeline() = default;
    ~FillPipeline() { destroy(); }

    FillPipeline(const FillPipeline&) = delete;
    FillPipeline& operator=(const FillPipeline&) = delete;

    void init(Device& device,
              VkRenderPass sceneRenderPass,
              VkDescriptorSetLayout pathSsboSetLayout,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        // Glyph atlas lives here now — formerly inside ColorPipeline. Record-
        // time `drawGlyphs` uses it to look up MSDF atlas UV rects + page
        // indices; the worker's prepare() pass flushes dirty pages before
        // dispatch reads them. (Safe: paint finishes before worker starts.)
        atlas_.init(device);

        // Set 0 — color LUT sampler (SHARED: global IMAGE_SAMPLER layout,
        // same one ResourceCaches builds its gradient / default descriptors
        // against).
        VkDescriptorSetLayout colorLutLayout =
            device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);

        // Set 1 — primitive SSBO layout (OWNED).
        {
            VkDescriptorSetLayoutBinding b {};
            b.binding = 0;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo dli {};
            dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dli.bindingCount = 1;
            dli.pBindings = &b;
            vkCreateDescriptorSetLayout(d, &dli, nullptr, &primSetLayout_);
        }

        // Set 3 — shape sampler (SHARED: same IMAGE_SAMPLER layout).
        VkDescriptorSetLayout shapeSetLayout = colorLutLayout;

        // Primitive-SSBO descriptor pool + per-slot sets.
        {
            VkDescriptorPoolSize poolSize {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = MAX_FRAMES;

            VkDescriptorPoolCreateInfo dpi {};
            dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpi.maxSets = MAX_FRAMES;
            dpi.poolSizeCount = 1;
            dpi.pPoolSizes = &poolSize;
            vkCreateDescriptorPool(d, &dpi, nullptr, &descPool_);

            VkDescriptorSetLayout perSlot[MAX_FRAMES];
            for (int i = 0; i < MAX_FRAMES; i++) perSlot[i] = primSetLayout_;

            VkDescriptorSetAllocateInfo ai {};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descPool_;
            ai.descriptorSetCount = MAX_FRAMES;
            ai.pSetLayouts = perSlot;
            vkAllocateDescriptorSets(d, &ai, primDescSets_);
        }

        // Per-slot host-visible primitive buffers.
        for (int i = 0; i < MAX_FRAMES; i++) {
            allocateBuffer(i, INITIAL_BUFFER_BYTES);
            writeDescriptor(i);
        }

        // Pipeline layout — 4 descriptor sets + PC.
        VkDescriptorSetLayout setLayouts[4] = {
            colorLutLayout, primSetLayout_, pathSsboSetLayout, shapeSetLayout
        };

        VkPushConstantRange pcRange {};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 4;
        pli.pSetLayouts = setLayouts;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

        pipeline_ = buildPipeline(sceneRenderPass, vertSpv, fragSpv);
    }

    bool ready() const { return device_ != nullptr && pipeline_ != VK_NULL_HANDLE; }
    VkPipelineLayout pipelineLayout() const { return layout_; }

    GlyphAtlas& atlas() { return atlas_; }
    const GlyphAtlas& atlas() const { return atlas_; }

    // Worker-thread prepare step — stages dirty MSDF atlas pages into GPU
    // textures. Mirrors ColorPipeline::prepare().
    void prepareFrame() { atlas_.stageDirtyPages(); }

    // ---- Primitive SSBO staging (paint-thread record) --------------------

    void beginFrame() { cpuPrims_.clear(); }

    uint32_t uploadPrimitives(const GeometryPrimitive* prims, uint32_t count)
    {
        const size_t bytes = static_cast<size_t>(count) * PRIMITIVE_STRIDE;
        const uint32_t firstInstance =
            static_cast<uint32_t>(cpuPrims_.size() / PRIMITIVE_STRIDE);
        auto* src = reinterpret_cast<const uint8_t*>(prims);
        cpuPrims_.insert(cpuPrims_.end(), src, src + bytes);
        return firstInstance;
    }

    void flushToGPU(int frameSlot)
    {
        currentFrameSlot_ = frameSlot;
        if (cpuPrims_.empty()) return;
        if (cpuPrims_.size() > static_cast<size_t>(bufferCapacity_[frameSlot]))
            growStorageBuffer(frameSlot, cpuPrims_.size());
        if (mappedPtrs_[frameSlot])
            memcpy(mappedPtrs_[frameSlot], cpuPrims_.data(), cpuPrims_.size());
    }

    // ---- Dispatch (worker-thread replay) ---------------------------------

    // One instanced draw of `primCount` primitives starting at `firstInstance`
    // in the per-frame SSBO. Runs INSIDE the currently-active scene render
    // pass; no RP transitions. The caller sets scissor + viewport from the
    // command; stencilRef travels dynamically so pipeline rebinds aren't
    // needed per clip-depth change.
    void dispatch(State& state, VkCommandBuffer cmd,
                  VkDescriptorSet colorLUTDesc,
                  VkDescriptorSet pathSsboDesc,
                  VkDescriptorSet shapeDesc,
                  VkExtent2D extent,
                  VkRect2D   scissor,
                  uint32_t   firstInstance,
                  uint32_t   primCount,
                  uint8_t    stencilRef)
    {
        if (!ready() || primCount == 0) return;

        // Force State to rebind pipeline (its tracker doesn't understand our
        // custom layout). State is here only so its tracker gets invalidated;
        // dispatch is otherwise self-contained.
        state.setCustomPipeline(pipeline_, layout_, stencilRef);

        VkDescriptorSet sets[4] = {
            colorLUTDesc,
            primDescSets_[currentFrameSlot_],
            pathSsboDesc,
            shapeDesc
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 4, sets, 0, nullptr);

        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilRef);

        PushConstants pc {};
        pc.viewportW = static_cast<float>(extent.width);
        pc.viewportH = static_cast<float>(extent.height);
        vkCmdPushConstants(cmd, layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        VkViewport vp {};
        vp.width    = static_cast<float>(extent.width);
        vp.height   = static_cast<float>(extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDraw(cmd, 6, primCount, 0, firstInstance);

        // Our custom layout isn't the generic Pipeline base's — tell State
        // its pipeline tracker is stale so the next standard op rebinds.
        state.invalidate();
    }

private:
    void destroy()
    {
        if (!device_) return;
        VkDevice d = device_->device();
        if (pipeline_ != VK_NULL_HANDLE)      vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_   != VK_NULL_HANDLE)      vkDestroyPipelineLayout(d, layout_, nullptr);
        if (descPool_ != VK_NULL_HANDLE)      vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (primSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, primSetLayout_, nullptr);
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (mappedPtrs_[i])       { vkUnmapMemory(d, storageMemory_[i]); mappedPtrs_[i] = nullptr; }
            if (storageBuffer_[i])    vkDestroyBuffer(d, storageBuffer_[i], nullptr);
            if (storageMemory_[i])    vkFreeMemory(d, storageMemory_[i], nullptr);
        }
    }

    VkPipeline buildPipeline(VkRenderPass sceneRenderPass,
                             std::span<const uint32_t> vertSpv,
                             std::span<const uint32_t> fragSpv)
    {
        VkDevice d = device_->device();

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
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod;
        stages[1].pName  = "main";

        // No vertex buffer.
        VkPipelineVertexInputStateCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia {};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp {};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo raster {};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth   = 1.0f;
        raster.cullMode    = VK_CULL_MODE_NONE;

        VkPipelineMultisampleStateCreateInfo ms {};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Always-on stencil test (EQUAL). Outside-clip pixels where
        // stencil != ref are discarded; ref=0 is the no-clip case. Matches
        // the ColorPipeline clipConfig's stencil state so visuals are byte-
        // identical to the pre-refactor behaviour.
        VkPipelineDepthStencilStateCreateInfo ds {};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.stencilTestEnable = VK_TRUE;
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

        // Straight-alpha blend: src * srcA + dst * (1 - srcA). Matches JUCE
        // convention — colour output of fill.frag is premultiplied-neutral
        // (color × shape with straight alpha on both).
        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp        = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp        = VK_BLEND_OP_ADD;

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
        pci.pVertexInputState   = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState      = &vp;
        pci.pRasterizationState = &raster;
        pci.pMultisampleState   = &ms;
        pci.pDepthStencilState  = &ds;
        pci.pColorBlendState    = &cb;
        pci.pDynamicState       = &dynState;
        pci.layout              = layout_;
        pci.renderPass          = sceneRenderPass;

        VkPipeline p = VK_NULL_HANDLE;
        vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &pci, nullptr, &p);

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);
        return p;
    }

    void growStorageBuffer(int frameSlot, size_t neededBytes)
    {
        VkDeviceSize newCap = bufferCapacity_[frameSlot];
        while (static_cast<size_t>(newCap) < neededBytes) newCap *= 2;

        VkDevice d = device_->device();
        if (mappedPtrs_[frameSlot])  { vkUnmapMemory(d, storageMemory_[frameSlot]); mappedPtrs_[frameSlot] = nullptr; }
        if (storageBuffer_[frameSlot] != VK_NULL_HANDLE) vkDestroyBuffer(d, storageBuffer_[frameSlot], nullptr);
        if (storageMemory_[frameSlot] != VK_NULL_HANDLE) vkFreeMemory(d, storageMemory_[frameSlot], nullptr);
        storageBuffer_[frameSlot] = VK_NULL_HANDLE;
        storageMemory_[frameSlot] = VK_NULL_HANDLE;

        allocateBuffer(frameSlot, newCap);
        writeDescriptor(frameSlot);
    }

    void allocateBuffer(int frameSlot, VkDeviceSize capacityBytes)
    {
        VkDevice d = device_->device();

        VkBufferCreateInfo bci {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = capacityBytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(d, &bci, nullptr, &storageBuffer_[frameSlot]);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(d, storageBuffer_[frameSlot], &mr);

        VkMemoryAllocateInfo mi {};
        mi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mi.allocationSize = mr.size;
        mi.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(),
            mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(d, &mi, nullptr, &storageMemory_[frameSlot]);
        vkBindBufferMemory(d, storageBuffer_[frameSlot], storageMemory_[frameSlot], 0);
        vkMapMemory(d, storageMemory_[frameSlot], 0, VK_WHOLE_SIZE, 0, &mappedPtrs_[frameSlot]);
        bufferCapacity_[frameSlot] = capacityBytes;
    }

    void writeDescriptor(int frameSlot)
    {
        VkDevice d = device_->device();

        VkDescriptorBufferInfo dbi {};
        dbi.buffer = storageBuffer_[frameSlot];
        dbi.offset = 0;
        dbi.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = primDescSets_[frameSlot];
        w.dstBinding      = 0;
        w.dstArrayElement = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
    }

    Device*               device_        = nullptr;
    VkPipelineLayout      layout_        = VK_NULL_HANDLE;
    VkPipeline            pipeline_      = VK_NULL_HANDLE;

    VkDescriptorSetLayout primSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_      = VK_NULL_HANDLE;
    VkDescriptorSet       primDescSets_[MAX_FRAMES] {};

    VkBuffer              storageBuffer_[MAX_FRAMES] {};
    VkDeviceMemory        storageMemory_[MAX_FRAMES] {};
    void*                 mappedPtrs_[MAX_FRAMES]    {};
    VkDeviceSize          bufferCapacity_[MAX_FRAMES] { INITIAL_BUFFER_BYTES, INITIAL_BUFFER_BYTES };

    std::vector<uint8_t>  cpuPrims_;
    int                   currentFrameSlot_ = 0;

    GlyphAtlas            atlas_;
};

} // namespace jvk
