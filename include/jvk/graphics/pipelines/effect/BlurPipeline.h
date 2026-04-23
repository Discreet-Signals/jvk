/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: BlurPipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// BlurPipeline — geometry-abstracted variable-radius Gaussian blur.
//
// Consolidates ShapeBlurPipeline + PathBlurPipeline behind the shared
// GeometryPrimitive format. One pipeline / one pair of shader modules handles
// rect / rrect / ellipse / capsule / path blurs; per-instance behaviour is
// driven entirely by the primitive's `geometryTag`. Dispatched as a style-B
// instanced quad draw: `vkCmdDraw(6, primCount, 0, firstInstance)` reads N
// primitives from a per-frame primitive SSBO owned by this pipeline.
//
// Descriptor layout:
//   set 0, binding 0   sampler2D   scene source (ping-pong half)
//   set 1, binding 0   SSBO        GeometryPrimitive[] — OWNED by this class
//   set 2, binding 0   SSBO        path segments — SHARED with PathPipeline
//                                  (its `ssboSetLayout()` drives set-2's
//                                  layout so the descriptor set PathPipeline
//                                  returns is binding-compatible here)
//   set 3, binding 0   SSBO        Vogel tap lookup table (Np-independent) —
//                                  OWNED, immutable, filled once at init.
//                                  Eliminates cos/sin/sqrt from the fragment
//                                  inner loop for kernelType==1.
//
// Primitive SSBO lifecycle mirrors PathPipeline's segment SSBO:
//   - beginFrame()            clears the CPU staging buffer.
//   - uploadPrimitives(...)   appends one or more primitives at record time,
//                             returns a stable firstInstance offset.
//   - flushToGPU(frameSlot)   at the top of Renderer::execute (after the
//                             frame fence wait), memcpys the staging buffer
//                             into the slot's mapped SSBO, sets
//                             `currentFrameSlot_`.
//   - dispatch(firstInstance, primCount, ...) issues the instanced draw;
//                             reads from the CURRENT slot's descriptor.
//
// Per-slot SSBOs avoid races: one slot may still be read by the GPU (N-1
// frames ago) while we flush the other (N-2 fence has been waited on).
// =============================================================================

class BlurPipeline {
public:
    struct PushConstants {
        float    viewportW, viewportH;   // 8
        float    dirX, dirY;             // 8   (1,0)/(0,1) for Low; unused for Medium/High
        int32_t  kernelType;             // 4   0=Low (separable), 1=Medium (Vogel), 2=High (2D LST)
        int32_t  _pad0, _pad1, _pad2;    // 12  pad to 32 bytes (16-byte multiple)
    };                                   // total: 32 bytes

    static constexpr VkDeviceSize INITIAL_BUFFER_BYTES = 64 * 1024;   // 64 KB ≈ 512 primitives
    static constexpr uint32_t     PRIMITIVE_STRIDE     = 128;          // sizeof(GeometryPrimitive)
    static constexpr int          MAX_FRAMES           = 2;
    // Must match MAX_POISSON_TAPS in geom_blur.frag's kernelType==1 branch.
    // Table is Np-independent: entry i = vec4(cos(i·GA)·√u, sin(i·GA)·√u,
    // (i+0.5), 0). Built once on host, uploaded once to a device-visible
    // SSBO at init, bound as set 3 for every blur dispatch.
    static constexpr uint32_t     VOGEL_TAP_COUNT      = 1024;

    BlurPipeline() = default;
    ~BlurPipeline() { destroy(); }

    BlurPipeline(const BlurPipeline&) = delete;
    BlurPipeline& operator=(const BlurPipeline&) = delete;

    void init(Device& device,
              VkRenderPass compatibleRenderPass,
              VkDescriptorSetLayout pathSsboSetLayout,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        // Set 0 — scene-source sampler (SHARED: reuse the global
        // IMAGE_SAMPLER layout so the scene ping-pong's IMAGE_SAMPLER
        // descriptor binds here without extra plumbing).
        VkDescriptorSetLayout sceneSetLayout =
            device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);

        // Set 1 — primitive SSBO layout (OWNED).
        {
            VkDescriptorSetLayoutBinding b {};
            b.binding = 0;
            b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            // Accessed in vertex for primitive expansion + flat output, and
            // fragment for the path-winding branch (via flat inputs only,
            // but a fragment-stage read of primFlags still goes through the
            // SSBO in some drivers' compiled path — keep both stages hot).
            b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo dli {};
            dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dli.bindingCount = 1;
            dli.pBindings = &b;
            vkCreateDescriptorSetLayout(d, &dli, nullptr, &primSetLayout_);
        }

        // Pool + per-slot primitive descriptor sets.
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

        // Set 3 — Vogel tap SSBO (OWNED, immutable). Uses its own single-
        // binding layout, allocated from a dedicated pool so the lifetime
        // is independent of the per-slot primitive pool.
        createVogelTapResources();

        // Pipeline layout — 4 descriptor sets + PC.
        VkDescriptorSetLayout setLayouts[4] = {
            sceneSetLayout, primSetLayout_, pathSsboSetLayout, vogelSetLayout_
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

        // Build the VkPipeline.
        pipeline_ = buildPipeline(compatibleRenderPass, vertSpv, fragSpv);
    }

    bool ready() const { return device_ != nullptr && pipeline_ != VK_NULL_HANDLE; }

    // --- Primitive SSBO staging -------------------------------------------

    // Called once per frame on the paint thread BEFORE record begins.
    void beginFrame() { cpuPrims_.clear(); }

    // Append N primitives to the CPU staging buffer. Returns the first
    // instance offset — stable, since the staging buffer is memcpy'd
    // verbatim into the slot's mapped SSBO at flushToGPU time.
    uint32_t uploadPrimitives(const GeometryPrimitive* prims, uint32_t count)
    {
        const size_t bytes = static_cast<size_t>(count) * PRIMITIVE_STRIDE;
        const uint32_t firstInstance =
            static_cast<uint32_t>(cpuPrims_.size() / PRIMITIVE_STRIDE);
        auto* src = reinterpret_cast<const uint8_t*>(prims);
        cpuPrims_.insert(cpuPrims_.end(), src, src + bytes);
        return firstInstance;
    }

    // Worker-thread flush — called inside Renderer::execute() after the
    // frame fence wait. Grows this slot's SSBO if needed, memcpys, and
    // latches the slot so subsequent dispatches read from it.
    void flushToGPU(int frameSlot)
    {
        currentFrameSlot_ = frameSlot;
        if (cpuPrims_.empty()) return;
        if (cpuPrims_.size() > static_cast<size_t>(bufferCapacity_[frameSlot]))
            growStorageBuffer(frameSlot, cpuPrims_.size());
        if (mappedPtrs_[frameSlot])
            memcpy(mappedPtrs_[frameSlot], cpuPrims_.data(), cpuPrims_.size());
    }

    // --- Dispatch ---------------------------------------------------------

    // One separable (or single) blur pass over `primCount` primitives, all
    // sharing `kernelType`. For Low kernel the caller invokes this twice
    // with direction (1,0) then (0,1); Medium/High need one call.
    //
    // Scissor is passed explicitly so the dispatcher can tighten it to the
    // union of the batch's readsPx. The render pass is the LOAD variant —
    // pixels outside the scissor keep their pre-pass contents.
    void applyPass(VkCommandBuffer cmd,
                   VkDescriptorSet sceneSrc,
                   VkDescriptorSet pathSsbo,
                   VkFramebuffer   dstFramebuffer,
                   VkRenderPass    dstRenderPass,
                   VkExtent2D      extent,
                   VkRect2D        scissor,
                   uint32_t        firstInstance,
                   uint32_t        primCount,
                   int32_t         kernelType,
                   float           dirX, float dirY,
                   uint32_t        stencilRef = 0)
    {
        if (!ready() || primCount == 0) return;

        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass  = dstRenderPass;
        rpbi.framebuffer = dstFramebuffer;
        rpbi.renderArea.extent = extent;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkDescriptorSet sets[4] = {
            sceneSrc,
            primDescSets_[currentFrameSlot_],
            pathSsbo,
            vogelDescSet_
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 4, sets, 0, nullptr);

        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilRef);

        PushConstants pc {};
        pc.viewportW  = static_cast<float>(extent.width);
        pc.viewportH  = static_cast<float>(extent.height);
        pc.dirX       = dirX;
        pc.dirY       = dirY;
        pc.kernelType = kernelType;
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
        vkCmdEndRenderPass(cmd);
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
        if (vogelDescPool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(d, vogelDescPool_, nullptr);
        if (vogelSetLayout_  != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, vogelSetLayout_, nullptr);
        if (vogelBuffer_     != VK_NULL_HANDLE) vkDestroyBuffer(d, vogelBuffer_, nullptr);
        if (vogelMemory_     != VK_NULL_HANDLE) vkFreeMemory(d, vogelMemory_, nullptr);
    }

    void createVogelTapResources()
    {
        VkDevice d = device_->device();

        // Layout + pool + set for the single immutable SSBO at set=3.
        VkDescriptorSetLayoutBinding b {};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dli {};
        dli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = 1;
        dli.pBindings    = &b;
        vkCreateDescriptorSetLayout(d, &dli, nullptr, &vogelSetLayout_);

        VkDescriptorPoolSize ps { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo dpi {};
        dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets       = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes    = &ps;
        vkCreateDescriptorPool(d, &dpi, nullptr, &vogelDescPool_);

        VkDescriptorSetAllocateInfo ai {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = vogelDescPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &vogelSetLayout_;
        vkAllocateDescriptorSets(d, &ai, &vogelDescSet_);

        // Backing buffer — host-visible | coherent for a one-shot memcpy. No
        // need for device-local since the table is read once per fragment and
        // 16 KB sits comfortably in caches. Keeping it host-visible avoids a
        // staging-buffer + transfer-cmdbuf dance.
        constexpr VkDeviceSize bytes = VOGEL_TAP_COUNT * sizeof(float) * 4;

        VkBufferCreateInfo bci {};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = bytes;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(d, &bci, nullptr, &vogelBuffer_);

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(d, vogelBuffer_, &mr);
        VkMemoryAllocateInfo mi {};
        mi.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mi.allocationSize  = mr.size;
        mi.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(),
            mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(d, &mi, nullptr, &vogelMemory_);
        vkBindBufferMemory(d, vogelBuffer_, vogelMemory_, 0);

        // Fill: tap[i] = (cos(i·GA)·√u, sin(i·GA)·√u, u, 0) where u = i + 0.5.
        // GA = 2π · (1 − 1/φ) ≈ 2.39996 rad — the golden angle. Pre-scaling by
        // √u collapses the shader's per-tap `sqrt((i+0.5)/Np)` into one mul
        // by 1/√Np that's constant across the tap loop.
        void* mapped = nullptr;
        vkMapMemory(d, vogelMemory_, 0, bytes, 0, &mapped);
        auto* out = static_cast<float*>(mapped);
        constexpr float goldenAngle = 2.399963229728653f;
        for (uint32_t i = 0; i < VOGEL_TAP_COUNT; ++i) {
            const float u     = static_cast<float>(i) + 0.5f;
            const float sqrtU = std::sqrt(u);
            const float ang   = static_cast<float>(i) * goldenAngle;
            out[i * 4 + 0] = std::cos(ang) * sqrtU;
            out[i * 4 + 1] = std::sin(ang) * sqrtU;
            out[i * 4 + 2] = u;
            out[i * 4 + 3] = 0.0f;
        }
        vkUnmapMemory(d, vogelMemory_);

        // Wire the descriptor.
        VkDescriptorBufferInfo dbi {};
        dbi.buffer = vogelBuffer_;
        dbi.offset = 0;
        dbi.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = vogelDescSet_;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
    }

    VkPipeline buildPipeline(VkRenderPass compatibleRenderPass,
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

        // Inside-clip only (Equal) — never writes stencil. Matches ShapeBlur +
        // PathBlur. Outside-clip / outside-bbox pixels are seeded from
        // source by the ping-pong seed-copy prologue (see
        // Renderer::execute's effectPassAndSwap helper) before this pass
        // runs, so the tight bbox dispatch here can write just the blur
        // output without clobbering uncovered regions.
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
        pci.pVertexInputState   = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState      = &vp;
        pci.pRasterizationState = &raster;
        pci.pMultisampleState   = &ms;
        pci.pDepthStencilState  = &ds;
        pci.pColorBlendState    = &cb;
        pci.pDynamicState       = &dynState;
        pci.layout              = layout_;
        pci.renderPass          = compatibleRenderPass;

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

    // Per-slot host-visible primitive SSBO — grows on demand, persistent-mapped.
    VkBuffer              storageBuffer_[MAX_FRAMES] {};
    VkDeviceMemory        storageMemory_[MAX_FRAMES] {};
    void*                 mappedPtrs_[MAX_FRAMES]    {};
    VkDeviceSize          bufferCapacity_[MAX_FRAMES] { INITIAL_BUFFER_BYTES, INITIAL_BUFFER_BYTES };

    // CPU staging — grows unbounded across a frame, cleared on beginFrame().
    std::vector<uint8_t>  cpuPrims_;
    int                   currentFrameSlot_ = 0;

    // Vogel tap lookup — owned, immutable after init, bound at set 3. The
    // pool owns exactly one descriptor set; the buffer stores
    // VOGEL_TAP_COUNT × vec4 (16 KB at 1024 taps).
    VkDescriptorSetLayout vogelSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      vogelDescPool_  = VK_NULL_HANDLE;
    VkDescriptorSet       vogelDescSet_   = VK_NULL_HANDLE;
    VkBuffer              vogelBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory        vogelMemory_    = VK_NULL_HANDLE;
};

} // namespace jvk
