/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: PathPipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// PathPipeline — analytical SDF path renderer (vger / Slug-style).
//
// For each juce::Path filled via Graphics::fillPath, the CPU flattens all
// bezier segments into a list of line segments in *physical pixel* space
// and parks them in this pipeline's per-frame storage buffer. A single
// quad is drawn covering the path's bounding box (plus a pixel of AA
// padding), and the fragment shader at each pixel walks the path's
// segments, computing min-distance + winding-number analytically.
//
// This completely bypasses MSDF atlas baking for path rendering:
//   - Zero rasterisation preprocessing cost → per-frame paths are free.
//   - Resolution-independent by construction (the SDF is re-evaluated
//     analytically at whatever zoom the quad ends up being rendered at).
//   - Single-channel SDF output, not MSDF, so tiny-text-style sharp
//     corners soften a little — but that's the explicit tradeoff, and
//     paths aren't tiny text. Text still goes through GlyphAtlas.
//
// This pipeline is self-contained: it owns its descriptor set layout,
// descriptor pool, storage buffer, pipeline layout, and VkPipeline.
// Registered with the renderer via setPathPipeline(); dispatched from
// Renderer::execute() when it sees a DrawOp::FillPath command.
// =============================================================================

class PathPipeline {
public:
    // Initial storage-buffer capacity in bytes — the SSBO is grown
    // automatically in flushToGPU() when a frame's segments exceed the
    // current allocation, doubling each time. Each line segment is 16
    // bytes (vec4: p0.xy, p1.xy), so 256 KB starts us at 16,384 segments.
    static constexpr VkDeviceSize INITIAL_BUFFER_BYTES = 256 * 1024;
    static constexpr uint32_t     SEGMENT_STRIDE       = 16; // sizeof(vec4)
    // One SSBO per in-flight frame slot. The fence wait in Renderer's
    // execute only guarantees THIS slot's previous use is done (2 frames
    // ago); the OTHER slot's use (1 frame ago) may still be reading its
    // buffer on the GPU. A shared SSBO would race whenever sub-60 fps
    // lets frame N-1 overlap with frame N's flushToGPU — manifests as
    // tile-shaped path corruption on large windows.
    static constexpr int          MAX_FRAMES           = 2;

    PathPipeline() = default;
    ~PathPipeline() { destroy(); }

    PathPipeline(const PathPipeline&) = delete;
    PathPipeline& operator=(const PathPipeline&) = delete;

    void init(Device& device,
              VkRenderPass sceneRenderPass,
              std::span<const uint32_t> vertSpv,
              std::span<const uint32_t> fragSpv)
    {
        device_ = &device;
        VkDevice d = device.device();

        // --- Set 0 — colorLUT layout (SHARED) ----------------------------
        // Reuse the global IMAGE_SAMPLER layout so the existing
        // `ResourceCaches::gradientDescriptor()` / `defaultDescriptor()`
        // sets can be bound directly at set 0, identical to ColorPipeline.
        VkDescriptorSetLayout colorLutLayout =
            device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);

        // --- Set 1 — segments SSBO layout (OWNED) ------------------------
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

        // --- Descriptor pool + sets (one SSBO descriptor per frame slot)
        VkDescriptorPoolSize poolSize {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = MAX_FRAMES;

        VkDescriptorPoolCreateInfo dpi {};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = MAX_FRAMES;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &poolSize;
        vkCreateDescriptorPool(d, &dpi, nullptr, &descPool_);

        VkDescriptorSetLayout setLayoutsPerSlot[MAX_FRAMES];
        for (int i = 0; i < MAX_FRAMES; i++) setLayoutsPerSlot[i] = ssboSetLayout_;

        VkDescriptorSetAllocateInfo ai {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descPool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = setLayoutsPerSlot;
        vkAllocateDescriptorSets(d, &ai, ssboDescSets_);

        // --- Pipeline layout (2 sets + push constants) -------------------
        VkPushConstantRange pcRange {};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(PushConstants);

        VkDescriptorSetLayout setLayouts[2] = { colorLutLayout, ssboSetLayout_ };

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 2;
        pli.pSetLayouts = setLayouts;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(d, &pli, nullptr, &layout_);

        // --- Persistent-mapped host-visible storage buffers (one per slot).
        for (int i = 0; i < MAX_FRAMES; i++) {
            allocateBuffer(i, INITIAL_BUFFER_BYTES);
            writeDescriptor(i);
        }

        // --- Build the VkPipeline ----------------------------------------
        pipeline_ = buildPipeline(sceneRenderPass, vertSpv, fragSpv);
    }

    bool ready() const { return device_ != nullptr && pipeline_ != VK_NULL_HANDLE; }

    // Called once per frame BEFORE record phase — clears the CPU-side
    // segment staging buffer. Record-time `uploadSegments` appends here,
    // then `flushToGPU(frameSlot)` (called inside Renderer::execute after
    // the frame fence wait) memcpys the staging buffer into that slot's
    // mapped SSBO.
    //
    // Why CPU staging: record runs before the frame fence wait. Writing
    // directly into a persistent-mapped SSBO during record would race with
    // the previous frame's GPU reads. CPU-staging defers the GPU-visible
    // write until after the fence wait.
    //
    // Why per-slot SSBOs: the fence wait only guarantees THIS slot's
    // previous use is done (N-2 frames ago). The OTHER slot's use (N-1)
    // may still be reading its SSBO on the GPU. A single shared SSBO
    // races when sub-60 fps lets frame N-1 overlap with frame N's flush.
    void beginFrame() { cpuSegments_.clear(); }

    // Append `count` segments (16 bytes each: vec4 p0.xy,p1.xy) to the
    // CPU staging buffer. Returns the starting segment index into the
    // SSBO — stable, since the CPU buffer is copied verbatim into the
    // SSBO at flushToGPU time. The CPU buffer grows unbounded; the GPU
    // SSBO is resized to fit during flushToGPU.
    uint32_t uploadSegments(const void* segmentData, uint32_t segmentCount)
    {
        size_t bytes = static_cast<size_t>(segmentCount) * SEGMENT_STRIDE;
        uint32_t firstSegment = static_cast<uint32_t>(cpuSegments_.size() / SEGMENT_STRIDE);
        auto* src = static_cast<const uint8_t*>(segmentData);
        cpuSegments_.insert(cpuSegments_.end(), src, src + bytes);
        return firstSegment;
    }

    // Called at the top of Renderer::execute (after target_.beginFrame's
    // fence wait) to push the CPU-staged segments into the slot's mapped
    // SSBO. Sets `currentFrameSlot_` as a side effect so subsequent
    // ssboDescriptorSet() calls return this slot's descriptor.
    //
    // If the slot's SSBO is too small, it's grown (destroy + recreate +
    // rewire the descriptor) — safe because the fence wait above
    // guarantees THIS slot's previous GPU use is done. The OTHER slot's
    // buffer is untouched.
    void flushToGPU(int frameSlot)
    {
        currentFrameSlot_ = frameSlot;
        if (cpuSegments_.empty()) return;
        if (cpuSegments_.size() > static_cast<size_t>(bufferCapacity_[frameSlot]))
            growStorageBuffer(frameSlot, cpuSegments_.size());
        if (mappedPtrs_[frameSlot])
            memcpy(mappedPtrs_[frameSlot], cpuSegments_.data(), cpuSegments_.size());
    }

    // Dispatch a single FillPath draw inside the active scene render pass.
    // `fanVerts` is a 6-vertex UIVertex quad emitted by Graphics::fillPath
    // with per-vertex color + gradientInfo populated exactly like the 2D
    // color pipeline. `colorDesc` is the shared color-source descriptor
    // (`ResourceCaches::gradientDescriptor()` or `defaultDescriptor()`) —
    // path_sdf.frag samples it iff gradientInfo.z > 0. `segmentStart` /
    // `segmentCount` point into the storage buffer this pipeline owns.
    void dispatch(State& state, VkCommandBuffer cmd, Renderer& r,
                  const DrawCommand& drawCmd,
                  const UIVertex* fanVerts, uint32_t vertexCount,
                  uint32_t segmentStart, uint32_t segmentCount,
                  uint32_t fillRule,
                  VkDescriptorSet colorDesc,
                  float viewportW, float viewportH)
    {
        if (!ready() || vertexCount == 0) return;

        state.setCustomPipeline(pipeline_, layout_);

        // Scissor and viewport — use the command's clipBounds for scissor.
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

        // Set 0 — colorLUT (routes through State so its tracker picks up
        // the descriptor and future color ops don't re-bind redundantly).
        state.setColorResource(colorDesc);

        // Set 1 — segments SSBO for the CURRENT frame slot (set by the
        // most recent flushToGPU). State doesn't track this descriptor,
        // so bind it directly.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 1, 1, &ssboDescSets_[currentFrameSlot_], 0, nullptr);

        // Push constants — viewport + segment range + fill rule.
        PushConstants pc {};
        pc.viewportW    = viewportW;
        pc.viewportH    = viewportH;
        pc.segmentStart = segmentStart;
        pc.segmentCount = segmentCount;
        pc.fillRule     = fillRule;
        pc._pad         = 0.0f;
        vkCmdPushConstants(cmd, layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);

        // Write the quad vertices into the State's vertex ring buffer and
        // issue the draw. State::draw wraps all of this; it also owns the
        // current-bound vertex buffer cache, so we let it handle binding.
        state.draw(drawCmd, fanVerts, vertexCount);

        // State's pipeline/descriptor tracker doesn't know about our
        // custom layout → force a full rebind for the next op.
        state.invalidate();
    }

    // Returns the SSBO descriptor set for the CURRENT frame slot (set by
    // the most recent flushToGPU). Used by ClipPipeline's push/pop dispatch
    // which shares this buffer for path-clip segment reads.
    VkDescriptorSet ssboDescriptorSet() const { return ssboDescSets_[currentFrameSlot_]; }

    // The descriptor-set layout that backs `ssboDescriptorSet()`. PathBlur
    // shares it so its own pipeline layout is guaranteed identical to this
    // one — the descriptor set above can then be bound against PathBlur's
    // pipeline without relying on cross-layout "identically-defined"
    // compatibility rules.
    VkDescriptorSetLayout ssboSetLayout() const { return ssboSetLayout_; }

private:
    struct PushConstants {
        float    viewportW, viewportH;
        uint32_t segmentStart;
        uint32_t segmentCount;
        uint32_t fillRule;       // 0 = non-zero, 1 = even-odd
        float    _pad;
    };

    // Grow one slot's SSBO to fit `neededBytes` (doubling). Only the
    // current slot is resized; the OTHER slot's buffer may still be in
    // flight on the GPU and is left alone (it'll grow when its next
    // flushToGPU happens, by which time its fence has been waited on).
    void growStorageBuffer(int frameSlot, size_t neededBytes)
    {
        VkDeviceSize newCap = bufferCapacity_[frameSlot];
        while (static_cast<size_t>(newCap) < neededBytes) newCap *= 2;

        VkDevice d = device_->device();
        if (mappedPtrs_[frameSlot]) {
            vkUnmapMemory(d, storageMemory_[frameSlot]);
            mappedPtrs_[frameSlot] = nullptr;
        }
        if (storageBuffer_[frameSlot] != VK_NULL_HANDLE)
            vkDestroyBuffer(d, storageBuffer_[frameSlot], nullptr);
        if (storageMemory_[frameSlot] != VK_NULL_HANDLE)
            vkFreeMemory(d, storageMemory_[frameSlot], nullptr);
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

        // Host-visible + coherent so CPU writes are immediately visible to
        // GPU without vkFlushMappedMemoryRanges bookkeeping.
        VkMemoryAllocateInfo mi {};
        mi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mi.allocationSize = mr.size;
        mi.memoryTypeIndex = Memory::findMemoryType(device_->physicalDevice(),
            mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(d, &mi, nullptr, &storageMemory_[frameSlot]);
        vkBindBufferMemory(d, storageBuffer_[frameSlot], storageMemory_[frameSlot], 0);
        vkMapMemory(d, storageMemory_[frameSlot], 0, VK_WHOLE_SIZE, 0,
                    &mappedPtrs_[frameSlot]);
        bufferCapacity_[frameSlot] = capacityBytes;
    }

    void writeDescriptor(int frameSlot)
    {
        VkDevice d = device_->device();

        // (Re)wire the slot's storage buffer into its own descriptor set.
        // Called on initial allocation AND after each grow — descriptors
        // targeting the old buffer become invalid when it's destroyed.
        VkDescriptorBufferInfo dbi {};
        dbi.buffer = storageBuffer_[frameSlot];
        dbi.offset = 0;
        dbi.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = ssboDescSets_[frameSlot];
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(d, 1, &write, 0, nullptr);
    }

    VkPipeline buildPipeline(VkRenderPass renderPass,
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
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod;
        stages[1].pName = "main";

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
        ds.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState blend {};
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
        if (pipeline_       != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_         != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (ssboSetLayout_  != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, ssboSetLayout_, nullptr);
        if (descPool_       != VK_NULL_HANDLE) vkDestroyDescriptorPool(d, descPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES; i++) {
            if (mappedPtrs_[i]) {
                vkUnmapMemory(d, storageMemory_[i]);
                mappedPtrs_[i] = nullptr;
            }
            if (storageBuffer_[i] != VK_NULL_HANDLE)
                vkDestroyBuffer(d, storageBuffer_[i], nullptr);
            if (storageMemory_[i] != VK_NULL_HANDLE)
                vkFreeMemory(d, storageMemory_[i], nullptr);
        }
        device_ = nullptr;
    }

    Device*               device_            = nullptr;
    VkPipelineLayout      layout_            = VK_NULL_HANDLE;
    VkPipeline            pipeline_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssboSetLayout_     = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_          = VK_NULL_HANDLE;
    VkDescriptorSet       ssboDescSets_[MAX_FRAMES]  {};
    VkBuffer              storageBuffer_[MAX_FRAMES] {};
    VkDeviceMemory        storageMemory_[MAX_FRAMES] {};
    void*                 mappedPtrs_[MAX_FRAMES]    {};
    VkDeviceSize          bufferCapacity_[MAX_FRAMES]{};
    int                   currentFrameSlot_  = 0;
    std::vector<uint8_t>  cpuSegments_;   // record-phase staging (shared)
};

} // namespace jvk
