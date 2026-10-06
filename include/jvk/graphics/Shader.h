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

                // If the shader is already live, rewriting its descriptor
                // sets races against any command buffer that bound them and
                // is still pending on the GPU (Vulkan §14.2.1 UB — the
                // layout was created without UPDATE_AFTER_BIND_BIT, so the
                // binding is "statically used"). rewriteLiveImage gates the
                // write on GPU idle. Heavy-handed (one stall per rebind) but
                // correct without descriptor-indexing feature setup. If
                // dynamic per-frame rebinding becomes a perf issue, switch
                // to UPDATE_AFTER_BIND_BIT on Memory::M's image-sampler
                // layout + pool.
                rewriteLiveImage(b);
                return;
            }
        }
    }

    // GPU-resident image bindings: an Image the caller created and fills on
    // the GPU (Renderer::initializeImage + uploadRegion), for example a
    // texture array or an atlas page. Nothing is copied and nothing goes
    // through the shared cache. The view type must match the binding: a
    // `sampler2DArray` needs an Image made with Shape::arrayView (or more
    // than one layer).
    //
    // The CALLER owns `image` and keeps it alive while this Shader can still
    // draw with it: destroy it after the Shader, or hand it to
    // Renderer::retire(), which waits out the frames in flight. Binding once
    // before the first draw is free; rebinding a live Shader costs the same
    // device-idle as set(name, juce::Image).
    void set(const juce::String& name, const Image& image)
    {
        if (image.view() == VK_NULL_HANDLE || image.sampler() == VK_NULL_HANDLE) return;
        for (auto& b : bindings_) {
            if (b.name == name && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                jassert (b.arrayed == image.isArray());   // sampler2DArray ↔ array view, sampler2D ↔ 2D view
                if (b.pinnedTexture) { b.pinnedTexture->unpin(); b.pinnedTexture = nullptr; }
                b.imageView = image.view();
                b.sampler   = image.sampler();
                b.bound     = true;
                rewriteLiveImage(b);
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
                rewriteLiveImage(b);
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
            if ((b.name == name || b.blockName == name) && b.offsetInBuffer < uniformData_.size() * sizeof(float)) {
                uniformData_[b.offsetInBuffer / sizeof(float)] = value;
                return;
            }
        }
    }

    void set(const juce::String& name, std::span<const float> data)
    {
        for (auto& b : bindings_) {
            if (b.name == name || b.blockName == name) {
                if (b.offsetInBuffer + data.size_bytes() <= uniformData_.size() * sizeof(float))
                    memcpy(&uniformData_[b.offsetInBuffer / sizeof(float)], data.data(), data.size_bytes());
                return;
            }
        }
    }

    // ===== Per-draw constants ================================================
    //
    // Up to 24 floats that belong to ONE draw: Graphics::drawShader copies
    // what was set last into the draw it records, so one Shader can be drawn
    // many times in a frame with different values (an atlas drawn at several
    // places, each with its own cell). Uniform blocks set with set() are per shader
    // instead: every draw of a frame sees the last value. The shader reads the
    // constants from its push-constant block, after jvk's own eight floats:
    //
    //   layout(push_constant) uniform PC {
    //       float resolutionX, resolutionY, time, viewportW, viewportH,
    //             regionX, regionY, reserved;           // jvk's, bytes 0..31
    //       vec4  frame;                                // the draw's: bytes 32..
    //       ...                                         //   up to 24 floats
    //   } pc;
    static constexpr uint32_t kDrawConstantsOffset = 32;

    void setDrawConstants (std::span<const float> values)
    {
        jassert (values.size() <= static_cast<size_t>(kShaderDrawConstants));
        drawConstantCount_ = static_cast<uint32_t>(std::min(values.size(), static_cast<size_t>(kShaderDrawConstants)));
        std::copy_n(values.begin(), drawConstantCount_, drawConstants_.begin());
    }
    std::span<const float> drawConstants() const { return { drawConstants_.data(), drawConstantCount_ }; }

    // ===== Reading targets ===================================================
    //
    // A shader reads a target by declaring a sampler2D with the target's name
    // (`layout(binding = n) uniform sampler2D normal;`). Drawn inline with
    // Graphics::drawShader, such a shader runs as ITS OWN PASS at that point
    // in paint order: it sees everything drawn into the targets before it,
    // and writes the main colour only (location 0, blended Over; the clip
    // still applies). jvk binds the targets when the draw is recorded and
    // rebinds them when the window resizes; nothing to set by hand.
    // A shader reads THE SCENE, the main colour as painted so far, by declaring
    // `layout(set = 1, binding = 0) uniform sampler2D scene;`. It then runs as
    // its own pass too and REPLACES the main colour inside its region with
    // what it outputs: it has the scene to blend with itself, so a scene
    // reader that changes nothing outputs texture(scene, ...) (a lighting
    // pass, a colour grade, a distortion). The clip still applies.
    bool readsScene() const { return readsScene_; }

    // The targets of `rt` this shader samples, by sampler name (bit i =
    // target i). Worked out once per set of target images (targetsGeneration),
    // not per draw.
    uint8_t targetsRead (const RenderTarget& rt)
    {
        if (rt.targetsGeneration() != readMaskGeneration_) {
            readMask_ = 0;
            const auto& targets = rt.targets();
            for (size_t i = 0; i < targets.size() && i < 8; ++i) {
                const auto name = targets[i].name.toString();
                for (auto& b : bindings_)
                    if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && !b.arrayed && b.name == name)
                        readMask_ |= static_cast<uint8_t>(1u << i);
            }
            readMaskGeneration_ = rt.targetsGeneration();
        }
        return readMask_;
    }

    // The targets this shader writes (bit i = target i: it declares output
    // location i + 1), of the `targetCount` its pass carries.
    uint8_t targetsWritten (size_t targetCount) const
    {
        uint8_t mask = 0;
        for (auto location : outputLocations_)
            if (location >= 1 && location <= targetCount && location <= 8)
                mask |= static_cast<uint8_t>(1u << (location - 1));
        return mask;
    }

    // Whether it writes the main colour: location 0, or no declared output.
    bool writesMain() const { return writesLocation(0) || outputLocations_.empty(); }

    // Message thread, while recording (the render worker is idle): points each
    // sampler named after a target at that target's current image whenever
    // the target images were recreated since the last bind (the first draw, a
    // resize, a new set of targets, another window). Keyed on the target's
    // generation, never on handle values: a recreated image can reuse them.
    void bindTargets (const RenderTarget& rt)
    {
        if (rt.targetsGeneration() == boundTargetsGeneration_) return;
        const auto& targets = rt.targets();
        std::vector<const BindingInfo*> rewritten;
        for (size_t i = 0; i < targets.size(); ++i) {
            const Image* image = rt.targetImage(i);
            if (image == nullptr) continue;
            const auto name = targets[i].name.toString();
            for (auto& b : bindings_) {
                if (b.type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b.arrayed || b.name != name)
                    continue;
                if (b.pinnedTexture) { b.pinnedTexture->unpin(); b.pinnedTexture = nullptr; }
                b.imageView = image->view();
                b.sampler   = image->sampler();
                b.bound     = true;
                rewritten.push_back(&b);
            }
        }
        rewriteLiveImages(rewritten);
        boundTargetsGeneration_ = rt.targetsGeneration();
    }

    // `targets`: the render targets of the pass this shader draws in. Each
    // output location i + 1 the shader declares writes target i with that
    // target's Blend; targets it doesn't declare are left untouched.
    void ensureCreated(Device& device, VkRenderPass renderPass, VkSampleCountFlagBits msaa,
                       const std::vector<Target>& targets = {})
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
            // One descriptor set per frame-in-flight slot: they differ only in
            // which slice of the uniform/storage buffer they point at (below).
            for (auto& set : descriptorSets_) {
                set = device.bindings().alloc(layoutId_);
                if (set == VK_NULL_HANDLE)
                    return;   // ~Shader frees whichever sets were allocated
            }
            setLayout = device.bindings().getLayout(layoutId_);

            // Bind defaults (1x1 black pixel) for unset image bindings so the
            // descriptor slot is never sampled uninitialized. A sampler2DArray
            // binding gets the array view of the same pixel.
            for (auto& b : bindings_) {
                if (!b.bound && b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                    b.imageView = b.arrayed ? device.caches().defaultArrayImageView()
                                            : device.caches().defaultImageView();
                    b.sampler   = device.caches().defaultSampler();
                }
            }

            // Back the reflected uniform/storage blocks with one host-visible
            // coherent VkBuffer holding one SLICE PER FRAME SLOT, each sized
            // to the total reflected block bytes (the same total `uniformData_`
            // is sized for). Each slot's descriptor set points its UBO/SSBO
            // bindings into its own slice. Per draw, ShaderPipeline::dispatch
            // memcpys uniformData_ into the slice of the frame slot being
            // recorded. With one shared slice (the old layout), that copy
            // overwrote data the GPU could still be reading for the previous
            // frame, which is still in flight.
            const VkDeviceSize blockBytes = uniformData_.size() * sizeof(float);
            slotStride_ = (blockBytes + kBlockAlign - 1) / kBlockAlign * kBlockAlign;
            const VkDeviceSize bufferSize = slotStride_ * kSlots;
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
                // are visible on the GPU's first read, whichever slot it is.
                for (int s = 0; s < kSlots; ++s)
                    std::memcpy(static_cast<char*>(uniformMapped_) + slotStride_ * s,
                                uniformData_.data(), blockBytes);
            }

            // Wire the descriptor sets: one write per binding per slot so the
            // shader sees its UBO/SSBO buffers and image samplers as soon as
            // it's bound. Image bindings either use the user-supplied
            // descriptor (set()/update()) or the default 1x1 fallback.
            std::vector<VkWriteDescriptorSet>   writes;
            std::vector<VkDescriptorBufferInfo> bufferInfos;
            std::vector<VkDescriptorImageInfo>  imageInfos;
            bufferInfos.reserve(bindings_.size() * kSlots);   // pointers into these stay valid
            imageInfos.reserve(bindings_.size() * kSlots);
            for (int s = 0; s < kSlots; ++s) {
                for (auto& b : bindings_) {
                    if (b.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                        b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                        bufferInfos.push_back({ uniformBuffer_, slotStride_ * s + b.offsetInBuffer, b.sizeInBuffer });
                        VkWriteDescriptorSet w {};
                        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        w.dstSet = descriptorSets_[s];
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
                        w.dstSet = descriptorSets_[s];
                        w.dstBinding = b.binding;
                        w.descriptorType = b.type;
                        w.descriptorCount = 1;
                        w.pImageInfo = &imageInfos.back();
                        writes.push_back(w);
                    }
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
        //   bytes   0..11  — resolution (vec2) + time (float) — vertex + fragment
        //   bytes  12..27  — viewport (vec2) + region origin (vec2) — vertex only
        //   bytes  32..127 — the draw's constants (setDrawConstants), fragment
        // One unified range covers both stages; each reads what it declares.
        VkPushConstantRange pushRange {
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, kDrawConstantsOffset + sizeof(float) * kShaderDrawConstants
        };

        VkPipelineLayoutCreateInfo pli {};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        // A scene reader's set 1 is the scene's sampler, the IMAGE_SAMPLER layout of the
        // descriptor sets jvk keeps for each half of the scene (bound per draw at replay).
        // Its set 0, when it declares no bindings of its own, borrows that layout too.
        VkDescriptorSetLayout setLayouts[2] = { setLayout, VK_NULL_HANDLE };
        uint32_t setCount = (setLayout != VK_NULL_HANDLE) ? 1u : 0u;
        if (readsScene_) {
            VkDescriptorSetLayout sceneLayout = device.bindings().getLayout(Memory::M::IMAGE_SAMPLER);
            if (setLayouts[0] == VK_NULL_HANDLE) setLayouts[0] = sceneLayout;
            setLayouts[1] = sceneLayout;
            setCount = 2;
        }
        pli.setLayoutCount = setCount;
        pli.pSetLayouts = setCount > 0 ? setLayouts : nullptr;
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

        // Location 0: the main colour, blended Over (replaced by a shader that
        // reads the scene: what it writes is the colour it read, modified). Locations 1..N: the pass's render targets, each
        // with its declared Blend where this shader writes it, untouched where
        // it doesn't. A shader with no location-0 output leaves the main
        // colour untouched too.
        std::vector<VkPipelineColorBlendAttachmentState> blends(1 + targets.size(), blendAttachment(Target::Blend::None));
        if (writesMain())
            blends[0] = blendAttachment(readsScene_ ? Target::Blend::Replace : Target::Blend::Over);
        for (size_t i = 0; i < targets.size(); ++i)
            if (writesLocation(static_cast<uint32_t>(i + 1)))
                blends[1 + i] = blendAttachment(targets[i].blend);

        VkPipelineColorBlendStateCreateInfo cb {};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = static_cast<uint32_t>(blends.size());
        cb.pAttachments = blends.data();

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

        builtFor_.clear();
        for (auto& t : targets)
            builtFor_.push_back(t.format);

        // Only a fully-built shader is ready. The old code set created_
        // unconditionally, so a failed vkCreateGraphicsPipelines still
        // reported isReady() and dispatch bound VK_NULL_HANDLE.
        created_ = (layout_ != VK_NULL_HANDLE && pipeline_ != VK_NULL_HANDLE
                    && clipPipeline_ != VK_NULL_HANDLE);
    }

    bool isReady() const { return created_; }

    // True when the pipeline was built for a pass carrying exactly these
    // targets (by format, which is what makes two passes compatible). A
    // Shader builds once, for the first pass it draws in, and drawing it in
    // a pass of another layout is invalid Vulkan: targets are fixed while
    // Vulkan runs, so a new set means new Shaders (AudioProcessorEditor::
    // setTargets).
    bool isBuiltFor (const std::vector<Target>& targets) const
    {
        return std::equal(builtFor_.begin(), builtFor_.end(), targets.begin(), targets.end(),
                          [](PixelFormat f, const Target& t) { return f == t.format; });
    }

    // Frame-in-flight slots, matching the Renderer's. Each slot has its own
    // descriptor set and its own slice of the uniform/storage buffer.
    static constexpr int kSlots = 2;

    VkPipeline       pipeline()      const { return pipeline_; }
    VkPipeline       clipPipeline()  const { return clipPipeline_ ? clipPipeline_ : pipeline_; }
    VkPipelineLayout layout()        const { return layout_; }
    VkDescriptorSet  descriptorSet (int frameSlot) const { return descriptorSets_[slotIndex (frameSlot)]; }

    const float* uniformData()   const { return uniformData_.data(); }
    size_t       uniformSize()   const { return uniformData_.size() * sizeof(float); }
    // Persistently-mapped pointer to this frame slot's slice of the
    // GPU-visible uniform/storage buffer backing every reflected block. Null
    // if the shader declared no UBO/SSBO bindings. ShaderPipeline::dispatch
    // memcpys uniformData_ into it each draw so set(name, value) reaches the
    // GPU without touching a slice an earlier frame is still reading.
    void* uniformMapped (int frameSlot) const
    {
        return uniformMapped_ == nullptr ? nullptr
             : static_cast<char*>(uniformMapped_) + slotStride_ * slotIndex (frameSlot);
    }

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
        for (auto set : descriptorSets_)
            if (set != VK_NULL_HANDLE) device_->bindings().free(set);
        if (layoutId_ != Memory::M::kInvalidLayout)
            device_->bindings().unregisterLayout(layoutId_);
    }

private:
    struct BindingInfo {
        juce::String     name;
        juce::String     blockName;   // a uniform block's declared name; set() takes either
        uint32_t         binding;
        VkDescriptorType type;
        uint32_t         offsetInBuffer = 0;
        uint32_t         sizeInBuffer = 0;
        VkImageView      imageView = VK_NULL_HANDLE;
        VkSampler        sampler   = VK_NULL_HANDLE;
        bool             bound = false;
        bool             arrayed = false;   // an image binding declared as an array (sampler2DArray)
        // Durable pin on the shared-cache CachedImage whose view+sampler
        // are baked into this Shader's descriptor sets. Without this, the
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
        readsScene_ = false;
        for (auto* rb : reflBindings) {
            if (rb->set == 1) {   // set 1, binding 0: the scene (readsScene); jvk binds it
                readsScene_ = readsScene_
                    || (rb->binding == 0 && rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
                continue;
            }
            BindingInfo info;
            // Use CharPointer_UTF8 rather than the raw-char-ptr String ctor so
            // we sidestep juce_String.cpp:327's ASCII-validity jassert — SPIRV-
            // Reflect can hand back pointers into bytecode where non-ASCII
            // bytes (or empty/unset name fields) trip it in Debug builds.
            info.name = rb->name != nullptr
                         ? juce::String(juce::CharPointer_UTF8(rb->name))
                         : juce::String();
            if (rb->type_description != nullptr && rb->type_description->type_name != nullptr)
                info.blockName = juce::String(juce::CharPointer_UTF8(rb->type_description->type_name));
            info.binding = rb->binding;
            info.type = static_cast<VkDescriptorType>(rb->descriptor_type);
            if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                info.arrayed = rb->image.arrayed != 0;

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
                // Every block starts on a kBlockAlign boundary: a descriptor's
                // buffer offset must be a multiple of the device's
                // min{Uniform,Storage}BufferOffsetAlignment, which the spec
                // caps at 256. Packed back to back, a second block (say a
                // 176-byte block followed by another) landed on an offset
                // most devices reject.
                bufferOffset        = (bufferOffset + kBlockAlign - 1) / kBlockAlign * kBlockAlign;
                info.offsetInBuffer = bufferOffset;
                info.sizeInBuffer   = blockSize;
                bufferOffset       += blockSize;
            }
            bindings_.push_back(std::move(info));
        }

        uniformData_.resize((bufferOffset + sizeof(float) - 1) / sizeof(float), 0.0f);

        // Fragment outputs: which colour locations this shader writes
        // (location 0 = main colour, i + 1 = render target i). An output
        // array covers consecutive locations.
        outputLocations_.clear();
        uint32_t outCount = 0;
        spvReflectEnumerateOutputVariables(&module, &outCount, nullptr);
        std::vector<SpvReflectInterfaceVariable*> outs(outCount);
        spvReflectEnumerateOutputVariables(&module, &outCount, outs.data());
        for (auto* v : outs) {
            if (v == nullptr || (v->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN) != 0) continue;
            const uint32_t n = v->array.dims_count > 0 ? std::max(1u, v->array.dims[0]) : 1u;
            for (uint32_t k = 0; k < n; ++k)
                outputLocations_.push_back(v->location + k);
        }

        spvReflectDestroyShaderModule(&module);
    }

    bool writesLocation (uint32_t location) const
    {
        return std::find(outputLocations_.begin(), outputLocations_.end(), location) != outputLocations_.end();
    }

    // The largest min{Uniform,Storage}BufferOffsetAlignment the spec allows.
    static constexpr VkDeviceSize kBlockAlign = 256;

    static int slotIndex (int frameSlot) { return ((frameSlot % kSlots) + kSlots) % kSlots; }

    // A binding's image changed on a Shader that is already live: write it
    // into every slot's descriptor set. The sets were created without
    // UPDATE_AFTER_BIND, so a command buffer still pending on the GPU may
    // reference them (Vulkan §14.2.1); gate the write on device idle. Heavy
    // handed, one stall per rebind, so bind before the first draw where you
    // can.
    void rewriteLiveImage (const BindingInfo& b) { rewriteLiveImages({ &b }); }

    // Every slot's descriptor set at once, after ONE device idle: a set may
    // still be in use by a frame in flight.
    void rewriteLiveImages (const std::vector<const BindingInfo*>& changed)
    {
        if (!created_ || device_ == nullptr || changed.empty()) return;
        const juce::ScopedLock queueSync(Renderer::queueLock());
        vkDeviceWaitIdle(device_->device());
        for (auto* b : changed)
            for (auto set : descriptorSets_)
                if (set != VK_NULL_HANDLE)
                    Memory::M::writeImage(device_->device(), set, b->binding, b->imageView, b->sampler);
    }

    std::vector<BindingInfo>  bindings_;
    std::vector<uint32_t>     spirv_;
    std::vector<float>        uniformData_;
    std::vector<uint32_t>     outputLocations_;     // reflected fragment output locations
    std::vector<PixelFormat>  builtFor_;            // the targets of the pass the pipeline was built for
    std::array<float, kShaderDrawConstants> drawConstants_ {};   // setDrawConstants
    uint32_t                  drawConstantCount_ = 0;
    uint64_t                  boundTargetsGeneration_ = 0;   // bindTargets: the targets the samplers point at
    uint64_t                  readMaskGeneration_ = ~uint64_t(0);   // targetsRead: the targets readMask_ is for
    uint8_t                   readMask_ = 0;
    bool                      readsScene_ = false;            // set 1, binding 0: the scene (readsScene)

    Device*          device_        = nullptr;
    VkPipeline       pipeline_      = VK_NULL_HANDLE;
    VkPipeline       clipPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout layout_        = VK_NULL_HANDLE;
    VkDescriptorSet  descriptorSets_[kSlots] = {};
    // kInvalidLayout = "never registered" — the old default of 0 aliased
    // IMAGE_SAMPLER, so an unregister from a binding-less shader would have
    // decremented the shared built-in layout's refcount.
    Memory::M::LayoutID layoutId_   = Memory::M::kInvalidLayout;

    // One host-visible coherent buffer backing every reflected UBO/SSBO
    // block, one slice (slotStride_ bytes) per frame slot, so a frame's copy
    // never lands on data the GPU is still reading for the frame before.
    // Within ONE frame the slice is still shared: a Shader drawn several
    // times per frame gives every draw the values from the last set() before
    // the frame executes.
    VkBuffer       uniformBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory_ = VK_NULL_HANDLE;
    void*          uniformMapped_ = nullptr;
    VkDeviceSize   slotStride_    = 0;

    bool   created_   = false;
};

inline void Graphics::prepareShaderDraw(Shader& shader, DrawShaderParams& params)
{
    const auto& rt = renderer_.target();
    params.targetsRead    = shader.targetsRead(rt);
    params.targetsWritten = shader.targetsWritten(rt.targets().size());
    params.writesMain     = shader.writesMain();
    if (params.targetsRead != 0)
        shader.bindTargets(rt);
    const auto constants = shader.drawConstants();
    params.constantCount = static_cast<uint32_t>(constants.size());
    std::copy(constants.begin(), constants.end(), params.constants);
}

} // namespace jvk
