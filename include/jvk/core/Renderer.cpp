namespace jvk {

// =============================================================================
// Construction / destruction — starts and stops the worker thread.
// =============================================================================

Renderer::Renderer(Device& device, RenderTarget& target)
    : device_(device), target_(target),
      vertices_(device.physicalDevice(), device.device()),
      worker_(std::make_unique<Worker>(*this))
{
    worker_->startThread();
}

Renderer::~Renderer() = default;
// Worker is the last-declared member, so it is the first destroyed: its
// dtor calls stopThread(), which signals exit + notify + joins. All other
// Renderer members remain valid while the worker is completing its final
// execute() — guaranteeing no dangling references from the worker thread.

// =============================================================================
// Threaded-execute control
// =============================================================================

void Renderer::submit()
{
    // Release here publishes every CPU-side write made during record (command
    // list, arena, path SSBO, mapped Vulkan buffers) to the worker's acquire
    // load at the top of execute().
    workerBusy_.store(true, std::memory_order_release);
    if (worker_) worker_->notify();
}

void Renderer::waitForIdle()
{
    // Used only for resize and teardown — both rare. Yield rather than burn
    // a CPU core; the wait is bounded by one execute duration (~16 ms on
    // VSync, longer only under DWM drag stalls).
    while (workerBusy_.load(std::memory_order_acquire))
        juce::Thread::yield();
}

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

    // Snapshot the device clock once per frame so every shader dispatched
    // during this execute() sees an identical `time` value — guarantees
    // deterministic per-frame motion and prevents drift between e.g. two
    // DrawShader ops in the same frame reading slightly different clocks.
    const float frameTime = device_.time();

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
    // Flush record-phase path/clip segments into THIS frame slot's SSBO.
    // Safe now because target_.beginFrame() above waited on this slot's
    // fence, so the GPU is done reading this slot's buffer from 2 frames
    // ago. The OTHER slot's buffer may still be in flight (frame N-1)
    // and is not touched.
    if (pathPipeline_) pathPipeline_->flushToGPU(frame.frameSlot);
    device_.flushUploads(frame.cmd);

    auto const& sb = target_.sceneBuffers(frame.frameSlot);

    // =========================================================================
    // Ping-pong model (synchronous, one active buffer at a time).
    //
    //   pp[0] is half A, pp[1] is half B. `cur` is the index of the half
    //   that is currently the active scene target. Normal draws write pp[cur].
    //
    //   When a post-process effect runs:
    //     1. End the active scene RP (finalLayout transitions pp[cur] color
    //        to SHADER_READ_ONLY; stencil transitions to DS_ATT_OPTIMAL).
    //     2. Run the effect pass in its own RP: sample pp[cur] color, write
    //        pp[1-cur] color.
    //     3. Swap: cur ^= 1. The active half is now the one we just wrote.
    //     4. Resume scene RP LOAD on pp[cur].sceneFB. Stencil image is the
    //        SAME physical image (shared between framebufferA and
    //        framebufferB) so LOAD_OP_LOAD restores the clip state we left.
    //
    //   Separable effects (Gaussian blur) do two effect passes and therefore
    //   two swaps — they naturally land back on the half they started on.
    //   Non-separable effects (HSV) do one pass and leave cur flipped.
    //
    //   At end of frame we blit pp[cur] → swapchain. There is never a race
    //   between the effect writing and the scene RP LOAD because they target
    //   DIFFERENT halves; the scene RP LOAD reads the half the effect just
    //   wrote, whose writes are made visible by the effect RP's outgoing
    //   subpass dependency.
    // =========================================================================
    struct Half {
        VkFramebuffer   sceneFB;
        VkFramebuffer   effectFB;
        VkDescriptorSet sampler;
        VkImage         image;
    };
    Half pp[2] = {
        { sb.framebufferA, sb.effectFBtoA, sb.samplerA, sb.colorA.image() },
        { sb.framebufferB, sb.effectFBtoB, sb.samplerB, sb.colorB.image() },
    };
    int cur = 0;

    VkRenderPass scenePassLoad = target_.sceneRenderPassLoad();
    VkRenderPass sceneBuildRP  = target_.sceneRenderPassClear(); // compat for build

    auto beginSceneRP = [&](VkRenderPass rp, bool withClears) {
        VkRenderPassBeginInfo rpbi {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = rp;
        rpbi.framebuffer = pp[cur].sceneFB;
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

    // Runs a single post-process pass (caller provides the dispatch via
    // `applyPass`) and swaps the active half. Must be called between a
    // vkCmdEndRenderPass and the matching resume of the scene RP.
    auto effectPassAndSwap = [&](auto applyPass) {
        int dst = cur ^ 1;
        applyPass(pp[cur].sampler, pp[dst].effectFB);
        cur = dst;
    };

    // When a clip is active, the effect pass will discard fragments outside
    // the clip via stencil test. Those outside-clip destination pixels would
    // then hold undefined (LOAD'd) data. Running a NOT_EQUAL-stencil copy
    // pass first fills exactly those outside-clip pixels with the source,
    // so the two passes together cover the destination without overlap. The
    // copy does NOT swap — the subsequent effect pass is the one that
    // flips `cur`.
    auto preCopyIfClipped = [&](uint8_t stencilDepth) {
        if (stencilDepth == 0 || !copyEffect_) return;
        int dst = cur ^ 1;
        copyEffect_->applyPass(frame.cmd,
            pp[cur].sampler, pp[dst].effectFB, target_.effectRenderPass(),
            frame.extent,
            /*dir unused*/ 0.0f, 0.0f,
            /*radius unused*/ 0.0f,
            /*stencilRef*/ stencilDepth);
    };

    beginSceneRP(target_.sceneRenderPassClear(), /*withClears=*/true);
    state_.begin(frame.cmd, vertices_,
        static_cast<float>(frame.extent.width),
        static_cast<float>(frame.extent.height));

    for (auto& cmd : commands_) {
        if (cmd.op == DrawOp::EffectKernel) {
            // Separable Gaussian blur. Pre-copy if clipped (seeds outside-
            // clip pixels with source), then H pass + V pass. Each pass
            // swaps; the two swaps cancel so `cur` lands back on the same
            // half it started on. Stencil test in the blur pipeline keeps
            // the H/V writes inside the clip; the pre-copy handled outside.
            vkCmdEndRenderPass(frame.cmd);
            if (postProcess_) {
                auto& bp = arena_.read<BlurParams>(cmd.dataOffset);
                preCopyIfClipped(cmd.stencilDepth);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    postProcess_->applyPass(frame.cmd, src, dst,
                        target_.effectRenderPass(), frame.extent,
                        1.0f, 0.0f, bp.radius,
                        static_cast<uint32_t>(cmd.stencilDepth));
                });
                // Second pass runs on the freshly-written half. Its clip
                // state is the same, but the outside-clip pixels of the
                // new `cur` half haven't been seeded yet — pre-copy again.
                preCopyIfClipped(cmd.stencilDepth);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    postProcess_->applyPass(frame.cmd, src, dst,
                        target_.effectRenderPass(), frame.extent,
                        0.0f, 1.0f, bp.radius,
                        static_cast<uint32_t>(cmd.stencilDepth));
                });
            }
            beginSceneRP(scenePassLoad, /*withClears=*/false);
            continue;
        }
        if (cmd.op == DrawOp::PushClipRect) {
            // cmd.clipBounds is already the pixel-space intersection of the
            // new rect and the existing clip stack (computed at record
            // time in Graphics::clipToRectangle). State tracks this as the
            // current scissor bounds.
            state_.pushClipRect(cmd.clipBounds);
            continue;
        }
        if (cmd.op == DrawOp::PopClipRect) {
            state_.popClipRect();
            continue;
        }
        if (cmd.op == DrawOp::PushClipPath) {
            // Analytical-SDF clip push — fragment shader discards outside the
            // clip shape, stencil INCR_WRAP where inside at stencil == parentDepth.
            if (clipPipeline_ && pathPipeline_) {
                auto& p = arena_.read<ClipShapeParams>(cmd.dataOffset);
                ClipPipeline::PushConstants pc {};
                pc.shapeType    = p.shapeType;
                pc.centerX      = p.centerX;
                pc.centerY      = p.centerY;
                pc.halfW        = p.halfW;
                pc.halfH        = p.halfH;
                pc.cornerRadius = p.cornerRadius;
                pc.segmentStart = p.segmentStart;
                pc.segmentCount = p.segmentCount;
                pc.fillRule     = p.fillRule;
                clipPipeline_->pushClip(state_, frame.cmd, *this, cmd,
                    pc, p.coverRect,
                    pathPipeline_->ssboDescriptorSet(),
                    static_cast<uint32_t>(cmd.stencilDepth),
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height));
            }
            state_.pushStencilDepth();
            continue;
        }
        if (cmd.op == DrawOp::PopClipPath) {
            // Analytical-SDF clip pop — DECR_WRAP at stencil == currentDepth
            // (before the CPU decrement). cmd.stencilDepth here is the depth
            // one level DEEPER than parent (i.e. the push's target depth),
            // so stencil == that value is exactly what the INCR produced.
            if (clipPipeline_ && pathPipeline_) {
                auto& p = arena_.read<ClipShapeParams>(cmd.dataOffset);
                ClipPipeline::PushConstants pc {};
                pc.shapeType    = p.shapeType;
                pc.centerX      = p.centerX;
                pc.centerY      = p.centerY;
                pc.halfW        = p.halfW;
                pc.halfH        = p.halfH;
                pc.cornerRadius = p.cornerRadius;
                pc.segmentStart = p.segmentStart;
                pc.segmentCount = p.segmentCount;
                pc.fillRule     = p.fillRule;
                clipPipeline_->popClip(state_, frame.cmd, *this, cmd,
                    pc, p.coverRect,
                    pathPipeline_->ssboDescriptorSet(),
                    static_cast<uint32_t>(cmd.stencilDepth),
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height));
            }
            state_.popStencilDepth();
            continue;
        }
        if (cmd.op == DrawOp::FillPath) {
            // Analytical SDF path fill — dispatched via PathPipeline which
            // owns the segment storage buffer + the SDF fragment shader.
            // Stays inside the scene render pass (no RP transitions).
            if (pathPipeline_) {
                auto& p    = arena_.read<FillPathParams>(cmd.dataOffset);
                auto& fill = getFill(p.fillIndex);
                // Colour source descriptor — matches the ColorPipeline pattern
                // (gradient atlas row for gradient fills, 1x1 default for
                // solids). path_sdf.frag samples it iff gradientInfo.z > 0.
                VkDescriptorSet colorDesc =
                    (fill.isGradient() && fill.gradient)
                        ? caches().gradientDescriptor()
                        : caches().defaultDescriptor();
                pathPipeline_->dispatch(state_, frame.cmd, *this, cmd,
                    p.quadVerts, 6,
                    p.segmentStart, p.segmentCount, p.fillRule,
                    colorDesc,
                    static_cast<float>(frame.extent.width),
                    static_cast<float>(frame.extent.height));
            }
            continue;
        }
        if (cmd.op == DrawOp::DrawShader) {
            // DrawShader is a regular scene draw — no render-pass transition.
            // Dispatch updates State's bound pipeline/layout and issues one
            // fullscreen triangle against the current scene framebuffer.
            if (shaderPipeline_) {
                auto& sp = arena_.read<DrawShaderParams>(cmd.dataOffset);
                auto* shader = static_cast<Shader*>(sp.shader);
                if (shader) {
                    shaderPipeline_->dispatch(state_, frame.cmd, *shader,
                        sp.region,
                        static_cast<float>(frame.extent.width),
                        static_cast<float>(frame.extent.height),
                        cmd.clipBounds,
                        cmd.stencilDepth,
                        frameTime);
                }
            }
            continue;
        }
        if (cmd.op == DrawOp::EffectHSV) {
            // Non-separable HSV — single pass + single swap. Pre-copy if
            // clipped so outside-clip pixels carry source; effect writes
            // inside-clip.
            vkCmdEndRenderPass(frame.cmd);
            if (hsvPipeline_) {
                auto& hp = arena_.read<HSVParams>(cmd.dataOffset);
                HSVPipeline::PushConstants pc {};
                pc.scaleH = hp.scaleH; pc.scaleS = hp.scaleS; pc.scaleV = hp.scaleV;
                pc.deltaH = hp.deltaH; pc.deltaS = hp.deltaS; pc.deltaV = hp.deltaV;
                preCopyIfClipped(cmd.stencilDepth);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    hsvPipeline_->applyPass(frame.cmd, src, dst,
                        target_.effectRenderPass(), frame.extent, pc,
                        static_cast<uint32_t>(cmd.stencilDepth));
                });
            }
            beginSceneRP(scenePassLoad, /*withClears=*/false);
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
                preCopyIfClipped(cmd.stencilDepth);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                        frame.extent, scissor, 1.0f, 0.0f, pc,
                        static_cast<uint32_t>(cmd.stencilDepth));
                });
                preCopyIfClipped(cmd.stencilDepth);
                effectPassAndSwap([&](VkDescriptorSet src, VkFramebuffer dst) {
                    shapeBlur_->applyPass(frame.cmd, src, dst, rp,
                        frame.extent, scissor, 0.0f, 1.0f, pc,
                        static_cast<uint32_t>(cmd.stencilDepth));
                });
            }
            beginSceneRP(scenePassLoad, /*withClears=*/false);
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

    // After the command stream, `cur` is the index of whichever ping-pong
    // half holds the final composited frame. An odd number of single-pass
    // effects leaves us on B; any other pattern (no effects, separable
    // effects only) leaves us on A. Blit whatever is current.
    VkImage currentImage = pp[cur].image;

    if (frame.swapImage == VK_NULL_HANDLE) {
        // Offscreen target — no swap image to blit to. Scene content remains
        // in currentImage (SHADER_READ_ONLY_OPTIMAL) for the caller to sample.
        // Serialize queue submit+present against other editors sharing the Device's
    // VkQueue (Vulkan external sync requirement).
    {
        const juce::ScopedLock queueSync(queueLock());
        target_.endFrame(frame);
    }
        frameCounter_++;
        return;
    }

    // Blit currentImage → swap image. currentImage is in SHADER_READ_ONLY
    // (finalLayout of the last scene RP). Transition both images and copy.
    VkImageMemoryBarrier barriers[2] {};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // Cover both possible last-use types of colorA: it was either last
    // written as a color attachment (final scene RP's writes → finalLayout
    // transition) or last sampled in a shader (if an effect pass ran and
    // then we re-entered scene RP LOAD). Including both bits eliminates
    // a subtle write-after-read hazard on tile renderers at slow frame
    // rates that caused post-effect draws to flicker every other frame.
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT
                              | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
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

    // Serialize queue submit+present against other editors sharing the Device's
    // VkQueue (Vulkan external sync requirement).
    {
        const juce::ScopedLock queueSync(queueLock());
        target_.endFrame(frame);
    }
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

    // Clip variant of the pipeline has stencilTest on (cmp = EQUAL, ops =
    // KEEP), reference pushed dynamically below. Non-clip variant has no
    // stencil test at all — used whenever stencilDepth_ == 0.
    VkPipeline handle = (stencilDepth_ > 0) ? pipeline->clipHandle()
                                            : pipeline->handle();

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

    // Stencil reference = clip depth. Every draw op passes only where
    // stencil == depth (i.e. inside all active clips).
    if (stencilDepth_ > 0 && stencilDepth_ != boundStencilRef_) {
        vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth_);
        boundStencilRef_ = stencilDepth_;
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

void State::draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count)
{
    if (count == 0) return;

    // Set scissor from command clip bounds
    juce::Rectangle<int> clip = cmd.clipBounds;
    if (!currentClipBounds_.isEmpty())
        clip = clip.getIntersection(currentClipBounds_);

    if (clip != boundScissor_) {
        // Clamp each edge to the framebuffer before computing the extent,
        // so a clip starting off-screen (x<0) doesn't produce an offset of
        // 0 with the un-clipped width and leak past the right edge.
        int x0 = std::max(0, clip.getX());
        int y0 = std::max(0, clip.getY());
        int x1 = std::max(x0, clip.getRight());
        int y1 = std::max(y0, clip.getBottom());
        VkRect2D sc;
        sc.offset = { x0, y0 };
        sc.extent = { static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0) };
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
    auto clipped = currentClipBounds_.isEmpty()
        ? rect
        : currentClipBounds_.getIntersection(rect);
    clipRectStack_.push_back(clipped);
    currentClipBounds_ = clipped;
}

void State::popClipRect()
{
    if (clipRectStack_.empty()) return;
    clipRectStack_.pop_back();
    currentClipBounds_ = clipRectStack_.empty()
        ? juce::Rectangle<int>{}
        : clipRectStack_.back();
}

void State::pushStencilDepth()
{
    stencilDepth_++;
}

void State::popStencilDepth()
{
    if (stencilDepth_ > 0) stencilDepth_--;
}

} // namespace jvk
