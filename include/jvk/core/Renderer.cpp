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
    // Sort: stable sort by z, then by pipeline within z (future optimization)
    // For now, replay in submission order as spec says.

    auto frame = target_.beginFrame();
    if (frame.cmd == VK_NULL_HANDLE) return;

    vertices_.beginFrame(frame.frameSlot);
    device_.flushRetired(frame.frameSlot);

    // Let pipelines stage pending uploads (atlas pages, deferred textures)
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

    // Stage all rows that were registered during recording. One contiguous
    // upload rather than one descriptor set per gradient.
    device_.caches().gradientAtlas().stageUploads();

    device_.flushUploads(frame.cmd);

    // Begin render pass
    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = target_.renderPass();
    rpbi.framebuffer = frame.framebuffer;
    rpbi.renderArea.extent = frame.extent;

    std::array<VkClearValue, 3> clears {};
    clears[0].color = {{ 0.0f, 0.0f, 0.0f, 1.0f }};
    clears[1].color = clears[0].color;
    clears[2].depthStencil = { 1.0f, 0 };
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues = clears.data();

    vkCmdBeginRenderPass(frame.cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    // Set initial viewport and scissor
    VkViewport vp {};
    vp.width = static_cast<float>(frame.extent.width);
    vp.height = static_cast<float>(frame.extent.height);
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(frame.cmd, 0, 1, &vp);

    VkRect2D sc {};
    sc.extent = frame.extent;
    vkCmdSetScissor(frame.cmd, 0, 1, &sc);

    // Initialize state for replay
    state_.begin(frame.cmd, vertices_, vp.width, vp.height);

    // Replay commands — Renderer handles lazy build + pipeline binding
    for (auto& cmd : commands_) {
        auto* pipeline = pipelineForOp_[static_cast<size_t>(cmd.op)];
        if (!pipeline) continue;
        if (!pipeline->isBuilt())
            pipeline->build(target_.renderPass(), target_.msaaSamples());
        state_.setPipeline(pipeline);
        pipeline->execute(*this, arena_, cmd);
    }

    vkCmdEndRenderPass(frame.cmd);

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
}

void State::setPipeline(Pipeline* pipeline)
{
    if (!pipeline) return;

    VkPipeline handle;
    if (stencilDepth_ > 0 && pipeline->clipHandle() != VK_NULL_HANDLE)
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

    // Write vertices and bind
    VkDeviceSize byteCount = count * sizeof(UIVertex);
    VkDeviceSize offset = vertices_->write(verts, byteCount);
    VkBuffer buf = vertices_->getBuffer();
    vkCmdBindVertexBuffers(cmd_, 0, 1, &buf, &offset);
    vkCmdDraw(cmd_, count, 1, 0, 0);
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
