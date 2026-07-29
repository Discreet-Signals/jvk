#pragma once

namespace jvk {

class Shader : public FrameRetained {
public:
    Shader() = default;

    void load(std::span<const uint32_t> fragSpirv)
    {
        spirv_.assign(fragSpirv.begin(), fragSpirv.end());
        reflectShader();
    }

    // Image bindings: ensure the image is registered in the Renderer's
    // texture cache, then stash its view+sampler on the binding. Takes the
    // Renderer (not just ResourceCaches) because the cache insert queues its
    // pixel upload onto that Renderer's upload list — so the copy runs in
    // the same frame as the draw that uses it, and two editors never share
    // the upload queue. If the shader is already live the descriptor write
    // is applied immediately; otherwise ensureCreated() picks it up.
    void set(const juce::String& name, const juce::Image& image, Renderer& r)
    {
        for (auto& b : bindings_) {
            if (b.name == name && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                auto& caches = r.caches();
                uint64_t hash = ResourceCaches::hashImage(image);
                caches.getTexture(hash, image, r);
                auto* tc = caches.textures().find(hash);
                if (tc == nullptr) return;

                // Swap the durable pin: release the previous CachedImage
                // (if any), pin the new one. This keeps the entry alive
                // against the cache's LRU eviction for as long as this
                // Shader binding holds its view+sampler.
                if (b.pinnedTexture) b.pinnedTexture->unpin();
                tc->pin();
                b.pinnedTexture = tc;

                b.imageView = tc->image.view();
                b.sampler   = tc->image.sampler();
                b.bound     = true;

                // If the shader is already live, updating descriptorSet_
                // races against any command buffer that bound it and is
                // still pending on the GPU (Vulkan §14.2.1 UB — the
                // layout was created without UPDATE_AFTER_BIND_BIT, so
                // the binding is "statically used"). Gate the write on
                // GPU idle. Heavy-handed (one stall per rebind) but
                // correct without requiring descriptor-indexing feature
                // setup. If dynamic per-frame rebinding becomes a
                // perf issue, switch to UPDATE_AFTER_BIND_BIT on
                // Memory::M's image-sampler layout + pool, or
                // double-buffer descriptorSet_ per frame slot.
                if (created_ && descriptorSet_ != VK_NULL_HANDLE) {
                    const juce::ScopedLock queueSync(Renderer::queueLock());
                    vkDeviceWaitIdle(device_->device());
                    Memory::M::writeImage(device_->device(), descriptorSet_,
                                          b.binding, b.imageView, b.sampler);
                }
                return;
            }
        }
    }

    // Dynamic image bindings: for sources whose CONTENT changes every frame
    // (video feeds, CPU-composited animations). set() resolves through the
    // shared texture cache, which keys on the pixel-buffer ADDRESS and never
    // re-uploads on a hit — mutated (or freed-and-recycled) buffers keep
    // showing their first upload — and every rebind of a live shader costs a
    // device-wide vkDeviceWaitIdle. update() instead gives the binding its
    // own VkImage, written once into the descriptor set, and re-records a
    // pixel copy into it through the Renderer's per-frame upload queue on
    // every call: no descriptor writes in steady state, no stalls, no cache
    // churn. The copy is ordered after all previously submitted fragment
    // work (uploadDynamic's barrier), so in-flight frames finish sampling
    // the old contents first.
    //
    // Pixels are copied out at call time (BGRA, premultiplied — matching
    // juce ARGB memory layout, so alpha-translucent sources stay
    // premultiplied when sampled). Safe to reuse one juce::Image buffer
    // across calls. Call from the message thread during record.
    void update(const juce::String& name, const juce::Image& image, Renderer& r)
    {
        if (!image.isValid()) return;
        for (auto& b : bindings_) {
            if (b.name != name || b.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                continue;

            auto& device = r.device();
            const auto w = static_cast<uint32_t>(image.getWidth());
            const auto h = static_cast<uint32_t>(image.getHeight());

            // (Re)create the owned image on first use or resize — the only
            // times the descriptor is written after creation.
            if (b.ownedImage == nullptr
                || b.ownedImage->width() != w || b.ownedImage->height() != h)
            {
                if (b.ownedImage != nullptr) {
                    // The old image may have a queued upload (two updates
                    // between executes) and in-flight frames sampling it —
                    // drop the former, let the retire queue's fence proof
                    // handle the latter. No stall.
                    r.cancelUploads(b.ownedImage->image());
                    r.retire(std::move(*b.ownedImage));
                }
                // VK_FORMAT_B8G8R8A8_UNORM matches juce ARGB's little-endian
                // byte order (B,G,R,A), so the staging fill below is a
                // straight row memcpy — no per-pixel conversion.
                b.ownedImage = std::make_unique<Image>(device.pool(), device.device(),
                    w, h, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

                // A binding switching over from set() releases its cache pin.
                if (b.pinnedTexture) { b.pinnedTexture->unpin(); b.pinnedTexture = nullptr; }

                b.imageView = b.ownedImage->view();
                b.sampler   = b.ownedImage->sampler();
                b.bound     = true;

                // Same idle-gated one-off write as set() — see the comment
                // there. Only hit when the shader is already live AND the
                // source was resized; steady-state frames never enter here.
                if (created_ && descriptorSet_ != VK_NULL_HANDLE) {
                    const juce::ScopedLock queueSync(Renderer::queueLock());
                    vkDeviceWaitIdle(device.device());
                    Memory::M::writeImage(device.device(), descriptorSet_,
                                          b.binding, b.imageView, b.sampler);
                }
            }

            // Stage this frame's pixels and queue the copy into the frame's
            // command buffer (flushUploads runs before the render pass).
            const VkDeviceSize byteSize = VkDeviceSize(w) * h * 4;
            auto staging = r.staging().alloc(byteSize);
            if (staging.mappedPtr == nullptr) return;

            juce::Image::BitmapData bd(image, juce::Image::BitmapData::readOnly);
            auto* dst = static_cast<uint8_t*>(staging.mappedPtr);
            for (uint32_t y = 0; y < h; ++y)
                std::memcpy(dst + size_t(y) * w * 4,
                            bd.getLinePointer(static_cast<int>(y)),
                            size_t(w) * 4);

            // Pin ourselves for this frame so ~Shader can't destroy the
            // owned VkImage while the queued copy (or the frame that samples
            // it) is still in flight — mirrors drawShader's retain.
            r.retain(this);
            r.uploadDynamic(staging, b.ownedImage->image(), w, h);
            return;
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
            if (b.name == name) {
                if (b.offsetInBuffer + data.size_bytes() <= uniformData_.size() * sizeof(float))
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

        // Create pipeline layout from reflected bindings. Shaders with no
        // reflected bindings (e.g. fragment-only effects driven by push
        // constants) build a layout with zero descriptor sets.
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        for (auto& b : bindings_) {
            layoutBindings.push_back({
                b.binding, b.type, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
            });
        }

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        if (!layoutBindings.empty()) {
            layoutId_ = device.bindings().registerLayout(layoutBindings.data(),
                static_cast<uint32_t>(layoutBindings.size()));
            if (layoutId_ == Memory::M::kInvalidLayout)
                return; // created_ stays false — dispatch keeps gating on isReady()
            descriptorSet_ = device.bindings().alloc(layoutId_);
            if (descriptorSet_ == VK_NULL_HANDLE)
                return;
            setLayout = device.bindings().getLayout(layoutId_);

            // Bind defaults (1x1 black pixel) for unset image bindings so the
            // descriptor slot is never sampled uninitialized.
            for (auto& b : bindings_) {
                if (!b.bound && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    b.imageView = device.caches().defaultImageView();
                    b.sampler   = device.caches().defaultSampler();
                }
            }

            // Back the reflected uniform/storage blocks with one host-visible
            // coherent VkBuffer sized to the total reflected block bytes (the
            // same total `uniformData_` is sized for). Each UBO/SSBO binding
            // gets a descriptor write pointing at its slice via offset+range.
            // Per-draw we memcpy uniformData_ → mapped pointer in
            // ShaderPipeline::dispatch so set(name, value) reaches the GPU.
            const VkDeviceSize bufferSize = uniformData_.size() * sizeof(float);
            if (bufferSize > 0) {
                VkBufferCreateInfo bci {};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size  = bufferSize;
                // Mark the buffer with both UBO and SSBO usage so a single
                // backing buffer can serve every reflected block, regardless
                // of whether the user shader declared it as `uniform` or
                // `buffer`. The descriptor type drives how the GPU reads it.
                bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                vkCreateBuffer(d, &bci, nullptr, &uniformBuffer_);

                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(d, uniformBuffer_, &req);
                VkMemoryAllocateInfo ai {};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize  = req.size;
                ai.memoryTypeIndex = Memory::findMemoryType(device.physicalDevice(),
                    req.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                vkAllocateMemory(d, &ai, nullptr, &uniformMemory_);
                vkBindBufferMemory(d, uniformBuffer_, uniformMemory_, 0);
                vkMapMemory(d, uniformMemory_, 0, bufferSize, 0, &uniformMapped_);

                // Initial copy so any set() calls made before ensureCreated
                // are visible on the GPU's first read.
                std::memcpy(uniformMapped_, uniformData_.data(), bufferSize);
            }

            // Wire the descriptor set: one write per binding so the shader
            // sees its UBO/SSBO buffers and image samplers as soon as it's
            // bound. Image bindings either use the user-supplied descriptor
            // (set via set(name, image, caches)) or the default 1x1 fallback.
            std::vector<VkWriteDescriptorSet>   writes;
            std::vector<VkDescriptorBufferInfo> bufferInfos;
            std::vector<VkDescriptorImageInfo>  imageInfos;
            bufferInfos.reserve(bindings_.size());
            imageInfos.reserve(bindings_.size());
            for (auto& b : bindings_) {
                if (b.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                    bufferInfos.push_back({ uniformBuffer_, b.offsetInBuffer, b.sizeInBuffer });
                    VkWriteDescriptorSet w {};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = descriptorSet_;
                    w.dstBinding = b.binding;
                    w.descriptorType = b.type;
                    w.descriptorCount = 1;
                    w.pBufferInfo = &bufferInfos.back();
                    writes.push_back(w);
                }
                else if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    imageInfos.push_back({ b.sampler, b.imageView,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
                    VkWriteDescriptorSet w {};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = descriptorSet_;
                    w.dstBinding = b.binding;
                    w.descriptorType = b.type;
                    w.descriptorCount = 1;
                    w.pImageInfo = &imageInfos.back();
                    writes.push_back(w);
                }
            }
            if (!writes.empty())
                vkUpdateDescriptorSets(d, static_cast<uint32_t>(writes.size()),
                                       writes.data(), 0, nullptr);
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
        VkShaderModule vertMod = VK_NULL_HANDLE;
        vkCreateShaderModule(d, &vci, nullptr, &vertMod);

        VkShaderModuleCreateInfo fci {};
        fci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fci.codeSize = spirv_.size() * 4;
        fci.pCode = spirv_.data();
        VkShaderModule fragMod = VK_NULL_HANDLE;
        vkCreateShaderModule(d, &fci, nullptr, &fragMod);

        if (vertMod == VK_NULL_HANDLE || fragMod == VK_NULL_HANDLE) {
            if (vertMod != VK_NULL_HANDLE) vkDestroyShaderModule(d, vertMod, nullptr);
            if (fragMod != VK_NULL_HANDLE) vkDestroyShaderModule(d, fragMod, nullptr);
            return; // created_ stays false; isReady() keeps gating dispatch
        }

        // Push constant layout mirrors shader_region.vert:
        //   bytes  0..11 — resolution (vec2) + time (float) — vertex + fragment
        //   bytes 12..27 — viewport (vec2) + region origin (vec2) — vertex only
        // One unified range covers both stages; fragment just reads the head.
        VkPushConstantRange pushRange {
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(float) * 7
        };

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = (setLayout != VK_NULL_HANDLE) ? 1u : 0u;
        pli.pSetLayouts = (setLayout != VK_NULL_HANDLE) ? &setLayout : nullptr;
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

        // Stencil reference is pushed per-draw (= current clip depth). We
        // no longer mask bits per-level — stencilCompareMask stays at the
        // default 0xFF so the EQUAL test checks the full depth counter.
        VkDynamicState dynStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        };
        VkPipelineDynamicStateCreateInfo dynState {};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3;
        dynState.pDynamicStates = dynStates;

        // Build two variants sharing one layout: the normal variant for
        // stencilDepth==0, and a clip variant that matches on the active
        // stencil bits so shader draws respect path clips just like ColorOps.
        auto buildVariant = [&](bool stencilTest) -> VkPipeline
        {
            VkPipelineDepthStencilStateCreateInfo ds {};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.stencilTestEnable = stencilTest ? VK_TRUE : VK_FALSE;
            if (stencilTest) {
                VkStencilOpState op {};
                op.failOp      = VK_STENCIL_OP_KEEP;
                op.passOp      = VK_STENCIL_OP_KEEP;
                op.depthFailOp = VK_STENCIL_OP_KEEP;
                op.compareOp   = VK_COMPARE_OP_EQUAL;
                op.compareMask = 0xFF; // overridden by dynamic state per draw
                op.writeMask   = 0xFF;
                op.reference   = 0;
                ds.front = op;
                ds.back  = op;
            }

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

            VkPipeline result = VK_NULL_HANDLE;
            vkCreateGraphicsPipelines(d, device.pipelineCache(), 1, &pci, nullptr, &result);
            return result;
        };

        pipeline_     = buildVariant(false);
        clipPipeline_ = buildVariant(true);

        vkDestroyShaderModule(d, vertMod, nullptr);
        vkDestroyShaderModule(d, fragMod, nullptr);

        // Only a fully-built shader is ready. The old code set created_
        // unconditionally, so a failed vkCreateGraphicsPipelines still
        // reported isReady() and dispatch bound VK_NULL_HANDLE.
        created_ = (layout_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE
                    && clipPipeline_ != VK_NULL_HANDLE);
    }

    bool isReady() const { return created_; }

    VkPipeline       pipeline()      const { return pipeline_; }
    VkPipeline       clipPipeline()  const { return clipPipeline_ ? clipPipeline_ : pipeline_; }
    VkPipelineLayout layout()        const { return layout_; }
    VkDescriptorSet  descriptorSet() const { return descriptorSet_; }

    const float* uniformData()   const { return uniformData_.data(); }
    size_t       uniformSize()   const { return uniformData_.size() * sizeof(float); }
    // Persistently-mapped pointer to the GPU-visible uniform/storage buffer
    // backing every reflected block. Null if the shader declared no UBO/SSBO
    // bindings. ShaderPipeline::dispatch memcpys uniformData_ into this each
    // draw so set(name, value) propagates to the GPU.
    void*        uniformMapped() const { return uniformMapped_; }

    ~Shader() override
    {
        // Block until every Renderer that pinned us in a recent frame has
        // rotated past the GPU fence — otherwise the worker could still be
        // mid-dispatch holding our Shader pointer, or the GPU could still
        // be sampling our descriptor set / executing our pipeline. Once
        // this returns, no thread (CPU or GPU) is referencing any of the
        // handles we're about to destroy.
        waitUntilUnretained();

        // Release durable pins on every texture we had bound. Each
        // corresponding CachedImage can now be evicted by the shared
        // cache's LRU. Do this BEFORE destroying our own descriptor +
        // pipeline so the order matches set()'s acquisition order in
        // reverse.
        for (auto& b : bindings_) {
            if (b.pinnedTexture) {
                b.pinnedTexture->unpin();
                b.pinnedTexture = nullptr;
            }
            // A queued dynamic-feed upload must not outlive its destination:
            // pending entries survive skipped frames by design, so without
            // this the next successful flushUploads records
            // vkCmdCopyBufferToImage into the VkImage the unique_ptr below
            // is about to destroy. ~Shader has no Renderer back-pointer —
            // cancel across every live Renderer via the registry.
            if (b.ownedImage != nullptr)
                Renderer::cancelUploadsAllRenderers(b.ownedImage->image());
        }

        if (!device_) return;
        VkDevice d = device_->device();
        if (uniformMapped_  != nullptr)        vkUnmapMemory(d, uniformMemory_);
        if (uniformBuffer_  != VK_NULL_HANDLE) vkDestroyBuffer(d, uniformBuffer_, nullptr);
        if (uniformMemory_  != VK_NULL_HANDLE) vkFreeMemory(d, uniformMemory_, nullptr);
        if (pipeline_       != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline_,     nullptr);
        if (clipPipeline_   != VK_NULL_HANDLE) vkDestroyPipeline(d, clipPipeline_, nullptr);
        if (layout_         != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (descriptorSet_  != VK_NULL_HANDLE) device_->bindings().free(descriptorSet_);
        if (layoutId_ != Memory::M::kInvalidLayout)
            device_->bindings().unregisterLayout(layoutId_);
    }

private:
    struct BindingInfo {
        juce::String     name;
        uint32_t         binding;
        VkDescriptorType type;
        uint32_t         offsetInBuffer = 0;
        uint32_t         sizeInBuffer = 0;
        VkImageView      imageView = VK_NULL_HANDLE;
        VkSampler        sampler   = VK_NULL_HANDLE;
        bool             bound = false;
        // Durable pin on the shared-cache CachedImage whose view+sampler
        // are baked into this Shader's descriptorSet_. Without this, the
        // cache's 120-frame LRU could evict the entry (no one re-hits
        // getTexture for a Shader-bound image — drawShader only binds the
        // Shader's own descriptor set), freeing the VkImage/View/Sampler
        // while our descriptor still references them → UB on next draw.
        // Pinned in set(), swapped on rebind, released in ~Shader.
        CachedImage*     pinnedTexture = nullptr;
        // Shader-owned texture for update()-driven bindings (per-frame
        // dynamic content). Mutually exclusive with pinnedTexture: a binding
        // is either a cached static image (set) or an owned dynamic one
        // (update). Destroyed in ~Shader after waitUntilUnretained, so no
        // in-flight frame can still be sampling it.
        std::unique_ptr<Image> ownedImage;
    };

    void reflectShader()
    {
        // Idempotent: a second load() must not APPEND to the previous
        // reflection — stale entries duplicated binding numbers into the
        // descriptor-set layout (invalid) and their offsets pointed into a
        // resized uniformData_. (Re-load after ensureCreated still keeps the
        // old pipeline — created_ short-circuits — but at least the metadata
        // stays coherent.)
        bindings_.clear();
        uniformData_.clear();

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
            // Use CharPointer_UTF8 rather than the raw-char-ptr String ctor so
            // we sidestep juce_String.cpp:327's ASCII-validity jassert — SPIRV-
            // Reflect can hand back pointers into bytecode where non-ASCII
            // bytes (or empty/unset name fields) trip it in Debug builds.
            info.name = rb->name != nullptr
                         ? juce::String(juce::CharPointer_UTF8(rb->name))
                         : juce::String();
            info.binding = rb->binding;
            info.type = static_cast<VkDescriptorType>(rb->descriptor_type);

            if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                // SPIRV-Reflect zeros block.size for every storage buffer (it
                // assumes they contain a runtime-sized array). Recover the
                // actual size by walking members — the last member's offset +
                // padded_size is the concrete block footprint for fixed-size
                // declarations like `float data[15]`. If padded_size is also 0
                // (true runtime array), fall back to size.
                uint32_t blockSize = rb->block.size;
                if (blockSize == 0 && rb->block.member_count > 0) {
                    const auto& last = rb->block.members[rb->block.member_count - 1];
                    blockSize = last.offset + (last.padded_size != 0 ? last.padded_size : last.size);
                }
                info.offsetInBuffer = bufferOffset;
                info.sizeInBuffer   = blockSize;
                bufferOffset       += blockSize;
            }
            bindings_.push_back(std::move(info));
        }

        uniformData_.resize(bufferOffset / sizeof(float), 0.0f);
        spvReflectDestroyShaderModule(&module);
    }

    std::vector<BindingInfo>  bindings_;
    std::vector<uint32_t>     spirv_;
    std::vector<float>        uniformData_;

    Device*          device_        = nullptr;
    VkPipeline       pipeline_      = VK_NULL_HANDLE;
    VkPipeline       clipPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout layout_        = VK_NULL_HANDLE;
    VkDescriptorSet  descriptorSet_ = VK_NULL_HANDLE;
    // kInvalidLayout = "never registered" — the old default of 0 aliased
    // IMAGE_SAMPLER, so an unregister from a binding-less shader would have
    // decremented the shared built-in layout's refcount.
    Memory::M::LayoutID layoutId_   = Memory::M::kInvalidLayout;

    // Single host-visible coherent buffer backing every reflected UBO/SSBO
    // block. NOTE: not double-buffered — fine for shaders drawn once per
    // frame whose uniforms drift slowly (e.g. a `time` value); for shaders
    // that draw multiple times per frame with different per-draw uniforms,
    // a per-frame-slot ring would be needed to avoid GPU-vs-CPU races.
    VkBuffer       uniformBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory_ = VK_NULL_HANDLE;
    void*          uniformMapped_ = nullptr;

    bool   created_   = false;
};

} // namespace jvk
