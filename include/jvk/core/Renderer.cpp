namespace jvk {

// =============================================================================
// Pipeline registration
// =============================================================================

void Renderer::registerPipeline(Pipeline& pipeline)
{
    auto ops = pipeline.supportedOps();
    for (auto op : ops)
        pipelineForOp_[static_cast<size_t>(op)] = &pipeline;
}

// =============================================================================
// Execute — sort + beginFrame + flush + render pass + replay + endFrame
// =============================================================================

void Renderer::execute()
{
    auto frame = target_.beginFrame();
    if (frame.cmd == VK_NULL_HANDLE) return;

    vertices_.beginFrame(frame.frameSlot);
    device_.flushRetired(frame.frameSlot);

    // Pipeline prepare (atlas dirty pages, etc.) and gradient/texture uploads
    {
        Pipeline* seen[static_cast<size_t>(DrawOp::COUNT)] = {};
        int n = 0;
        for (auto* p : pipelineForOp_) {
            if (!p) continue;
            bool dup = false;
            for (int i = 0; i < n; i++) if (seen[i] == p) { dup = true; break; }
            if (!dup) { seen[n++] = p; p->prepare(); }
        }
    }
    device_.caches().gradientAtlas().stageUploads();
    device_.flushUploads(frame.cmd);

    auto const& sb = target_.sceneBuffers(frame.frameSlot);

    // Ping-pong state. Scene draws go into `current`; an effect samples
    // `current` and writes into `other`, then the two swap so subsequent
    // scene draws land on top of the effect's output.
    VkFramebuffer   currentSceneFB  = sb.framebufferA;  // scene RP writes here
    VkFramebuffer   currentEffectFB = sb.effectFBtoA;   // effect writes into colorA
    VkFramebuffer   otherSceneFB    = sb.framebufferB;
    VkFramebuffer   otherEffectFB   = sb.effectFBtoB;
    VkDescriptorSet currentSampler  = sb.samplerA;      // samples colorA
    VkDescriptorSet otherSampler    = sb.samplerB;
    VkImage         currentImage    = sb.colorA.image();

    auto beginSceneRP = [&](VkRenderPass rp, VkFramebuffer fb, bool withClears) {
        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = rp;
        rpbi.framebuffer = fb;
        rpbi.renderArea.extent = frame.extent;

        VkClearValue clears[2] {};
        clears[0].color = {{ 0.0f, 0.0f, 0.0f, 0.0f }};
        clears[1].depthStencil = { 1.0f, 0 };
        if (withClears) {
            rpbi.clearValueCount = 2;
            rpbi.pClearValues = clears;
        }
        vkCmdBeginRenderPass(frame.cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp {};
        vp.width    = static_cast<float>(frame.extent.width);
        vp.height   = static_cast<float>(frame.extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &vp);
        VkRect2D sc {}; sc.extent = frame.extent;
        vkCmdSetScissor(frame.cmd, 0, 1, &sc);

        state_.invalidate();
    };

    beginSceneRP(target_.sceneRenderPassClear(), currentSceneFB, /*withClears=*/true);
    state_.begin(frame.cmd, vertices_,
        static_cast<float>(frame.extent.width),
        static_cast<float>(frame.extent.height));

    VkRenderPass scenePassLoad = target_.sceneRenderPassLoad();
    VkRenderPass sceneBuildRP  = target_.sceneRenderPassClear(); // compat for build

    for (auto& cmd : commands_) {
        if (cmd.op == DrawOp::EffectKernel) {
            vkCmdEndRenderPass(frame.cmd);

            if (postProcess_) {
                auto& bp = arena_.read<BlurParams>(cmd.dataOffset);
                // Separable Gaussian: horizontal into `other`, vertical back
                // into `current`. `current` keeps its identity so subsequent
                // scene draws land on top of the blurred result.
                postProcess_->applyPass(frame.cmd,
                    currentSampler, otherEffectFB, target_.effectRenderPass(),
                    frame.extent, 1.0f, 0.0f, bp.radius);
                postProcess_->applyPass(frame.cmd,
                    otherSampler, currentEffectFB, target_.effectRenderPass(),
                    frame.extent, 0.0f, 1.0f, bp.radius);
            }

            beginSceneRP(scenePassLoad, currentSceneFB, /*withClears=*/false);
            continue;
        }
        if (cmd.op == DrawOp::BlurShape) {
            vkCmdEndRenderPass(frame.cmd);

            if (shapeBlur_) {
                auto& sp = arena_.read<BlurShapeParams>(cmd.dataOffset);

                ShapeBlurPipeline::PushConstants pc {};
                pc.invCol0X = sp.invXform[0]; pc.invCol0Y = sp.invXform[1];
                pc.invCol1X = sp.invXform[2]; pc.invCol1Y = sp.invXform[3];
                pc.invCol2X = sp.invXform[4]; pc.invCol2Y = sp.invXform[5];
                pc.shapeHalfX = sp.shapeHalf[0];
                pc.shapeHalfY = sp.shapeHalf[1];
                pc.lineBX     = sp.lineB[0];
                pc.lineBY     = sp.lineB[1];
                pc.maxRadius     = sp.maxRadius;
                pc.falloff       = sp.falloff;
                pc.displayScale  = sp.displayScale;
                pc.cornerRadius  = sp.cornerRadius;
                pc.lineThickness = sp.lineThickness;
                pc.shapeType     = static_cast<int>(sp.shapeType);
                pc.edgePlacement = static_cast<int>(sp.edgePlacement);
                pc.inverted      = static_cast<int>(sp.inverted);

                VkRect2D scissor { {0, 0}, frame.extent };
                VkRenderPass rp = target_.effectRenderPass();
                shapeBlur_->applyPass(frame.cmd,
                    currentSampler, otherEffectFB, rp,
                    frame.extent, scissor, 1.0f, 0.0f, pc);
                shapeBlur_->applyPass(frame.cmd,
                    otherSampler, currentEffectFB, rp,
                    frame.extent, scissor, 0.0f, 1.0f, pc);
            }

            beginSceneRP(scenePassLoad, currentSceneFB, /*withClears=*/false);
            continue;
        }
        auto* pipeline = pipelineForOp_[static_cast<size_t>(cmd.op)];
        if (!pipeline) continue;
        if (!pipeline->isBuilt())
            pipeline->build(sceneBuildRP);
        state_.setPipeline(pipeline);
        pipeline->execute(*this, arena_, cmd);
    }

    vkCmdEndRenderPass(frame.cmd);

    if (frame.swapImage == VK_NULL_HANDLE) {
        // Offscreen target — no swap image to blit to. Scene content remains
        // in currentImage (SHADER_READ_ONLY_OPTIMAL) for the caller to sample.
        target_.endFrame(frame);
        frameCounter_++;
        return;
    }

    // Blit currentTarget → swap image. currentImage is in SHADER_READ_ONLY
    // (finalLayout of the scene RP). Transition both images and copy.
    VkImageMemoryBarrier barriers[2] {};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = currentImage;
    barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = frame.swapImage;
    barriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(frame.cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, barriers);

    VkImageBlit region {};
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[1] = { (int32_t)frame.extent.width, (int32_t)frame.extent.height, 1 };
    region.dstOffsets[1] = { (int32_t)frame.extent.width, (int32_t)frame.extent.height, 1 };
    vkCmdBlitImage(frame.cmd,
        currentImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        frame.swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region, VK_FILTER_NEAREST);

    VkImageMemoryBarrier present {};
    present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    present.dstAccessMask = 0;
    present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    present.image = frame.swapImage;
    present.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(frame.cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &present);

    target_.endFrame(frame);
    frameCounter_++;
}

// =============================================================================
// State implementation
// =============================================================================

void State::begin(VkCommandBuffer cmd, Memory::V& vertices, float vpWidth, float vpHeight)
{
    cmd_ = cmd;
    vertices_ = &vertices;
    vpWidth_ = vpWidth;
    vpHeight_ = vpHeight;
    invalidate();
}

void State::invalidate()
{
    currentPipeline_ = nullptr;
    boundPipeline_ = VK_NULL_HANDLE;
    boundLayout_ = VK_NULL_HANDLE;
    boundColorSet_ = VK_NULL_HANDLE;
    boundShapeSet_ = VK_NULL_HANDLE;
    boundStencilRef_ = 0;
    boundScissor_ = { -1, -1, 0, 0 };
    boundVertexBuffer_ = VK_NULL_HANDLE;
}

void State::setPipeline(Pipeline* pipeline)
{
    if (!pipeline) return;

    VkPipeline handle;
    if (stencilDepth_ > 0)
        handle = pipeline->clipHandle();
    else
        handle = pipeline->handle();

    if (handle != boundPipeline_) {
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, handle);
        boundPipeline_ = handle;

        // Layout change invalidates descriptor set bindings in Vulkan
        if (boundLayout_ != pipeline->layout()) {
            boundColorSet_ = VK_NULL_HANDLE;
            boundShapeSet_ = VK_NULL_HANDLE;
            boundLayout_ = pipeline->layout();
        }

        // Push viewport size for pixel→NDC conversion in vertex shader
        float vpSize[2] = { vpWidth_, vpHeight_ };
        vkCmdPushConstants(cmd_, boundLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vpSize), vpSize);
    }

    // Per-level bit stencil: each clip level owns bit N. The clip variant
    // tests that ALL active bits are set (EQUAL with mask = (1<<depth)-1).
    if (stencilDepth_ > 0) {
        uint32_t mask = (1u << stencilDepth_) - 1;
        vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, mask);
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, mask);
        boundStencilRef_ = mask;
    }

    currentPipeline_ = pipeline;
}

void State::setCustomPipeline(VkPipeline pipeline, VkPipelineLayout layout)
{
    if (pipeline != boundPipeline_) {
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        boundPipeline_ = pipeline;
        boundLayout_ = layout;
    }
    if (stencilDepth_ > 0)
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth_);

    currentPipeline_ = nullptr; // force rebind on next setPipeline
}

void State::setResources(VkDescriptorSet colorSet, VkDescriptorSet shapeSet)
{
    setColorResource(colorSet);
    setShapeResource(shapeSet);
}

void State::setColorResource(VkDescriptorSet set)
{
    if (set != boundColorSet_) {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
            boundLayout_, 0, 1, &set, 0, nullptr);
        boundColorSet_ = set;
    }
}

void State::setShapeResource(VkDescriptorSet set)
{
    if (set != boundShapeSet_) {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
            boundLayout_, 1, 1, &set, 0, nullptr);
        boundShapeSet_ = set;
    }
}

void State::pushConstants(uint32_t offset, uint32_t size, const void* data)
{
    vkCmdPushConstants(cmd_, boundLayout_, VK_SHADER_STAGE_VERTEX_BIT,
        offset, size, data);
}

void State::drawCached(const DrawCommand& cmd, VkBuffer vbuf,
                       uint32_t firstVertex, uint32_t vertexCount)
{
    if (vertexCount == 0) return;

    // Scissor — command clip intersected with the current clip-stack bounds.
    juce::Rectangle<int> clip = cmd.clipBounds;
    if (!currentClipBounds_.isEmpty())
        clip = clip.getIntersection(currentClipBounds_);

    if (clip != boundScissor_) {
        VkRect2D sc;
        sc.offset = { std::max(0, clip.getX()), std::max(0, clip.getY()) };
        sc.extent = { static_cast<uint32_t>(std::max(0, clip.getWidth())),
                      static_cast<uint32_t>(std::max(0, clip.getHeight())) };
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        boundScissor_ = clip;
    }

    // One big mesh pool — bind once, then firstVertex picks the subrange.
    if (vbuf != boundVertexBuffer_) {
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd_, 0, 1, &vbuf, &zero);
        boundVertexBuffer_ = vbuf;
    }
    vkCmdDraw(cmd_, vertexCount, 1, firstVertex, 0);
}

void State::draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count)
{
    if (count == 0) return;

    // Set scissor from command clip bounds
    juce::Rectangle<int> clip = cmd.clipBounds;
    if (!currentClipBounds_.isEmpty())
        clip = clip.getIntersection(currentClipBounds_);

    if (clip != boundScissor_) {
        VkRect2D sc;
        sc.offset = { std::max(0, clip.getX()), std::max(0, clip.getY()) };
        sc.extent = { static_cast<uint32_t>(std::max(0, clip.getWidth())),
                      static_cast<uint32_t>(std::max(0, clip.getHeight())) };
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        boundScissor_ = clip;
    }

    // Ring buffer — bind it once at offset 0 and use firstVertex for the
    // subrange. Each write() returns a vertex-aligned byte offset; we convert
    // to a vertex index so the bind only changes when the slot grows.
    VkDeviceSize byteCount  = count * sizeof(UIVertex);
    VkDeviceSize byteOffset = vertices_->write(verts, byteCount);
    VkBuffer     buf        = vertices_->getBuffer();
    uint32_t     firstVert  = static_cast<uint32_t>(byteOffset / sizeof(UIVertex));

    if (buf != boundVertexBuffer_) {
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd_, 0, 1, &buf, &zero);
        boundVertexBuffer_ = buf;
    }
    vkCmdDraw(cmd_, count, 1, firstVert, 0);
}

void State::pushClipRect(const juce::Rectangle<int>& rect)
{
    ClipEntry entry;
    entry.type = ClipEntry::Rect;
    entry.rect = currentClipBounds_.isEmpty() ? rect : currentClipBounds_.getIntersection(rect);
    clipStack_.push_back(std::move(entry));
    currentClipBounds_ = clipStack_.back().rect;
}

void State::pushClipPath(const juce::Path& path, const juce::AffineTransform& transform)
{
    ClipEntry entry;
    entry.type = ClipEntry::Path;
    entry.rect = currentClipBounds_;
    // Stencil write geometry is handled by the stencil pipeline module
    // during execute() — State just tracks the nesting depth
    clipStack_.push_back(std::move(entry));
    stencilDepth_++;
}

void State::popClip()
{
    if (clipStack_.empty()) return;
    auto& top = clipStack_.back();
    if (top.type == ClipEntry::Path)
        stencilDepth_--;

    clipStack_.pop_back();

    // Restore clip bounds from remaining stack
    currentClipBounds_ = {};
    for (auto& entry : clipStack_) {
        if (entry.type == ClipEntry::Rect) {
            if (currentClipBounds_.isEmpty())
                currentClipBounds_ = entry.rect;
            else
                currentClipBounds_ = currentClipBounds_.getIntersection(entry.rect);
        }
    }
}

void State::setStencilWriteMask(uint32_t mask)
{
    vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, mask);
}

void State::setStencilCompareMask(uint32_t mask)
{
    vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, mask);
}

} // namespace jvk
