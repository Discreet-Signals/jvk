/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC
 
 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝
 
 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 
 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: AudioProcessorEditor.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class AudioProcessorEditor : public juce::AudioProcessorEditor,
                              private juce::ComponentListener
{
public:
    explicit AudioProcessorEditor(juce::AudioProcessor& p)
        : juce::AudioProcessorEditor(p), paintBridge(*this)
    {
        addAndMakeVisible(vulkanRenderer);
        vulkanRenderer.setInterceptsMouseClicks(false, false);
        vulkanRenderer.addChildComponent(&paintBridge);
        addComponentListener(this);
        setCachedComponentImage(new NullCachedImage());
        vulkanRenderer.setRendering(true);
    }

    explicit AudioProcessorEditor(juce::AudioProcessor* p)
        : juce::AudioProcessorEditor(p), paintBridge(*this)
    {
        addAndMakeVisible(vulkanRenderer);
        vulkanRenderer.setInterceptsMouseClicks(false, false);
        vulkanRenderer.addChildComponent(&paintBridge);
        addComponentListener(this);
        setCachedComponentImage(new NullCachedImage());
        vulkanRenderer.setRendering(true);
    }

    ~AudioProcessorEditor() override
    {
        removeComponentListener(this);
    }

    // Toggle between Vulkan and JUCE software rendering.
    // When Vulkan is enabled, JUCE's software repaint is disabled via
    // setCachedComponentImage — only the Vulkan render loop calls paint().
    // When disabled, normal JUCE software rendering resumes.
    void setVulkanEnabled(bool enabled)
    {
        vulkanEnabled = enabled;
        vulkanRenderer.setVisible(enabled);
        if (enabled)
        {
            setCachedComponentImage(new NullCachedImage());
            vulkanRenderer.setRendering(true);
        }
        else
        {
            vulkanRenderer.setRendering(false);
            setCachedComponentImage(nullptr);
        }
        repaint();
    }

    bool isVulkanEnabled() const { return vulkanEnabled; }

    bool isVulkanReady() const
    {
        return vulkanRenderer.getStatus() == core::VulkanStatus::Ready;
    }

    bool isInVulkanRender() const { return inVulkanRender; }

    VulkanRenderer& getVulkanRenderer() { return vulkanRenderer; }

    // Physically correct sRGB blending (brighter semi-transparent content)
    // vs JUCE-identical UNORM blending (default). Requires rebuild.
    void setSRGBPipeline(bool enabled)
    {
        // Would need swapchain + pipeline rebuild — for now just store the flag
        srgbPipelineMode = enabled;
    }
    bool isSRGBPipeline() const { return srgbPipelineMode; }

    int getVulkanFps() const { return vulkanFps; }

private:
    // Tells JUCE's repaint manager to skip software rendering for this component.
    // paintWithinParentContext() sees this and calls paint() on it (no-op) instead
    // of paintEntireComponent(). The Vulkan render path calls paintEntireComponent()
    // directly, bypassing this entirely.
    struct NullCachedImage : public juce::CachedComponentImage
    {
        void paint(juce::Graphics&) override {}
        bool invalidateAll() override { return false; }
        bool invalidate(const juce::Rectangle<int>&) override { return false; }
        void releaseResources() override {}
    };

    void componentMovedOrResized(juce::Component&, bool, bool wasResized) override
    {
        if (wasResized)
        {
            if (auto* c = getConstrainer())
            {
                constexpr float pixelScale = 2.0f;
                vulkanRenderer.setMaxSize(
                    static_cast<int>(c->getMaximumWidth()  * pixelScale),
                    static_cast<int>(c->getMaximumHeight() * pixelScale));
            }
            vulkanRenderer.setBounds(getLocalBounds());
            // PaintBridge bounds must match the 2x swapchain resolution
            auto b = getLocalBounds().toFloat();
            paintBridge.setBounds(b.getX() * 2.0f, b.getY() * 2.0f,
                                   b.getWidth() * 2.0f, b.getHeight() * 2.0f);
        }
    }

    // Internal VulkanComponent that creates the 2D pipeline and paints
    // the editor's component tree through VulkanGraphicsContext
    class PaintBridge : public VulkanComponent
    {
    public:
        explicit PaintBridge(AudioProcessorEditor& e) : editor(e) {}

        void addedToRenderer(const VulkanRenderer& renderer) override
        {
            if (renderer.getStatus() != core::VulkanStatus::Ready)
                return;

            physDevice = renderer.getPhysicalDevice();
            device = renderer.getDevice();

            VertexLayout vertLayout;
            vertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
            vertLayout.attributes = {
                { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
            };

            auto pushRanges = std::vector<VkPushConstantRange>{
                { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }
            };

            // --- Descriptor set layout for texture binding ---
            descriptorHelper = std::make_unique<core::DescriptorHelper>(device);
            descriptorHelper->addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                          VK_SHADER_STAGE_FRAGMENT_BIT);
            auto dsLayout = descriptorHelper->buildLayout();
            descriptorHelper->createPool(512); // descriptor sets for textures, atlas pages, gradients

            // --- Main SDF+textured pipeline ---
            {
                shaders::ShaderGroup sg;
                sg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                              shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                sg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                              shaders::ui2d::frag_spv, shaders::ui2d::frag_spvSize);
                sg.addDescriptorSetLayout(dsLayout);

                PipelineConfig config;
                config.cullMode = VK_CULL_MODE_NONE;
                config.depthTestEnable = false;
                config.depthWriteEnable = false;
                config.blendMode = BlendMode::AlphaBlend;
                config.pushConstantRanges = pushRanges;

                mainPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(sg), renderer.getMSAASamples(),
                    std::move(vertLayout), std::move(config));
            }

            // --- Main pipeline with stencil clip test ---
            // Same as main pipeline, but only draws where stencil != 0.
            // Used when a path clip is active (stencilClipDepth > 0).
            {
                VertexLayout clipVertLayout;
                clipVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                clipVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                shaders::ShaderGroup clipSg;
                clipSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                  shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                clipSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                  shaders::ui2d::frag_spv, shaders::ui2d::frag_spvSize);
                clipSg.addDescriptorSetLayout(dsLayout);

                PipelineConfig clipConfig;
                clipConfig.cullMode = VK_CULL_MODE_NONE;
                clipConfig.depthTestEnable = false;
                clipConfig.depthWriteEnable = false;
                clipConfig.blendMode = BlendMode::AlphaBlend;
                clipConfig.pushConstantRanges = pushRanges;
                // Stencil test: only draw where stencil >= ref (ref set dynamically)
                clipConfig.stencilTestEnable = true;
                clipConfig.stencilCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
                clipConfig.stencilPassOp = VK_STENCIL_OP_KEEP;
                clipConfig.stencilFailOp = VK_STENCIL_OP_KEEP;
                clipConfig.stencilWriteMask = 0x00; // don't modify stencil during drawing

                mainClipPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(clipSg), renderer.getMSAASamples(),
                    std::move(clipVertLayout), std::move(clipConfig));
            }

            // --- Stencil write pipeline ---
            {
                VertexLayout stencilVertLayout;
                stencilVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                stencilVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                shaders::ShaderGroup stencilSg;
                stencilSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                     shaders::ui2d::stencil_vert_spv, shaders::ui2d::stencil_vert_spvSize);
                stencilSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                     shaders::ui2d::stencil_frag_spv, shaders::ui2d::stencil_frag_spvSize);

                PipelineConfig stencilConfig;
                stencilConfig.cullMode = VK_CULL_MODE_NONE;
                stencilConfig.depthTestEnable = false;
                stencilConfig.depthWriteEnable = false;
                stencilConfig.blendMode = BlendMode::Opaque;
                stencilConfig.colorWriteEnable = false;
                stencilConfig.stencilTestEnable = true;
                stencilConfig.stencilPassOp = VK_STENCIL_OP_INCREMENT_AND_WRAP;
                stencilConfig.stencilCompareOp = VK_COMPARE_OP_ALWAYS;
                // Back faces decrement for correct non-zero winding rule
                stencilConfig.separateBackStencil = true;
                stencilConfig.stencilBackPassOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
                stencilConfig.stencilBackCompareOp = VK_COMPARE_OP_ALWAYS;
                stencilConfig.pushConstantRanges = pushRanges;

                stencilPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(stencilSg), renderer.getMSAASamples(),
                    std::move(stencilVertLayout), std::move(stencilConfig));
            }

            // --- Stencil cover pipeline (reads stencil, draws color where non-zero) ---
            {
                shaders::ShaderGroup coverSg;
                coverSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                   shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                coverSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                   shaders::ui2d::frag_spv, shaders::ui2d::frag_spvSize);
                coverSg.addDescriptorSetLayout(dsLayout);

                VertexLayout coverVertLayout;
                coverVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                coverVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                PipelineConfig coverConfig;
                coverConfig.cullMode = VK_CULL_MODE_NONE;
                coverConfig.depthTestEnable = false;
                coverConfig.depthWriteEnable = false;
                coverConfig.blendMode = BlendMode::AlphaBlend;
                coverConfig.colorWriteEnable = true;
                coverConfig.stencilTestEnable = true;
                coverConfig.stencilCompareOp = VK_COMPARE_OP_NOT_EQUAL;
                coverConfig.stencilReference = 0;
                coverConfig.stencilPassOp = VK_STENCIL_OP_ZERO;
                coverConfig.stencilFailOp = VK_STENCIL_OP_KEEP;
                coverConfig.pushConstantRanges = pushRanges;

                stencilCoverPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(coverSg), renderer.getMSAASamples(),
                    std::move(coverVertLayout), std::move(coverConfig));
            }



            // --- Multiply blend pipeline (for color grading effects) ---
            {
                shaders::ShaderGroup mulSg;
                mulSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                 shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                mulSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                 shaders::ui2d::frag_spv, shaders::ui2d::frag_spvSize);
                mulSg.addDescriptorSetLayout(descriptorHelper->getLayout());

                VertexLayout mulVertLayout;
                mulVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                mulVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                PipelineConfig mulConfig;
                mulConfig.cullMode = VK_CULL_MODE_NONE;
                mulConfig.depthTestEnable = false;
                mulConfig.depthWriteEnable = false;
                mulConfig.blendMode = BlendMode::Multiply;
                mulConfig.pushConstantRanges = pushRanges;

                multiplyPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(mulSg), renderer.getMSAASamples(),
                    std::move(mulVertLayout), std::move(mulConfig));
            }

            // --- HSV effect pipeline ---
            // TODO: HSV needs framebuffer read-back. Requires either:
            //   1. Copy framebuffer to temp texture (like blur does), or
            //   2. Proper subpassInput with MoltenVK resource type annotations.
            // Disabled for now — drawEffectQuad null-checks the pipeline pointer.
            // hsvPipeline remains nullptr.

            // --- Blur pipeline (reads from temp texture sampler) ---
            {
                shaders::ShaderGroup blurSg;
                blurSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                  shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                blurSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                  shaders::ui2d::blur_frag_spv, shaders::ui2d::blur_frag_spvSize);
                blurSg.addDescriptorSetLayout(descriptorHelper->getLayout());

                VertexLayout blurVertLayout;
                blurVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                blurVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                PipelineConfig blurConfig;
                blurConfig.cullMode = VK_CULL_MODE_NONE;
                blurConfig.depthTestEnable = false;
                blurConfig.depthWriteEnable = false;
                blurConfig.blendMode = BlendMode::Opaque; // overwrite with blurred result
                blurConfig.pushConstantRanges = pushRanges;

                blurPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(blurSg), renderer.getMSAASamples(),
                    std::move(blurVertLayout), std::move(blurConfig));
            }

            // --- 1x1 white default texture ---
            {
                uint32_t whitePixel = 0xFFFFFFFF;
                defaultTexture.create({ physDevice, device, 1, 1,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
                defaultTexture.upload(physDevice, renderer.getCommandPool(),
                    renderer.getGraphicsQueue(), &whitePixel, 1, 1, VK_FORMAT_R8G8B8A8_UNORM);
                defaultTexture.createSampler(VK_FILTER_NEAREST);
            }

            // --- Default descriptor set ---
            defaultDescriptorSet = descriptorHelper->allocateSet();
            core::DescriptorHelper::writeImage(device, defaultDescriptorSet, 0,
                defaultTexture.getView(), defaultTexture.getSampler());

            // --- Glyph atlas for GPU text rendering ---
            glyphAtlas.init(physDevice, device, renderer.getCommandPool(),
                            renderer.getGraphicsQueue(), descriptorHelper.get());

            // --- Gradient LUT cache for GPU gradient evaluation ---
            gradCache.descriptorHelper = descriptorHelper.get();
            gradCache.physDevice = physDevice;
            gradCache.device = device;
            gradCache.commandPool = renderer.getCommandPool();
            gradCache.graphicsQueue = renderer.getGraphicsQueue();

            // --- Texture cache for GPU image rendering ---
            texCache.descriptorHelper = descriptorHelper.get();
            texCache.physDevice = physDevice;
            texCache.device = device;
            texCache.commandPool = renderer.getCommandPool();
            texCache.graphicsQueue = renderer.getGraphicsQueue();

            // --- GPU memory pool (sub-allocator) ---
            memoryPool.init(physDevice, device);

            // --- Vertex ring buffer (zero-alloc dynamic geometry) ---
            ringBuffer.create(physDevice, device);

            // --- Staging belt for batched texture uploads ---
            stagingBelt.init(physDevice, device);

            // Wire staging into caches (enables deferred uploads)
            gradCache.stagingBelt = &stagingBelt;
            gradCache.pendingUploads = &pendingUploads;
            gradCache.defaultDescriptorSet = defaultDescriptorSet;
            texCache.stagingBelt = &stagingBelt;
            texCache.pendingUploads = &pendingUploads;
            texCache.defaultDescriptorSet = defaultDescriptorSet;

            setPipeline(mainPipeline->getInternal());
        }

        void removedFromRenderer(const VulkanRenderer&) override
        {
            if (!device) return;
            // Free any retired descriptor sets before destroying the pool
            for (int i = 0; i < MAX_FRAMES; ++i)
            {
                for (auto ds : retiredDescriptorSets[i])
                    descriptorHelper->freeSet(ds);
                retiredDescriptorSets[i].clear();
            }
            mainPipeline.reset();
            mainClipPipeline.reset();
            stencilPipeline.reset();
            stencilCoverPipeline.reset();
            multiplyPipeline.reset();
            hsvPipeline.reset();
            blurPipeline.reset();
            descriptorHelper.reset();
            hsvDescriptorHelper.reset();
            defaultTexture.destroy();
            blurTempImage.destroy();
            ringBuffer.destroy();
            stagingBelt.destroy();
            texCache.clear();
            gradCache.clear();
            glyphAtlas.clear();
            memoryPool.destroy();
        }

        void prepareFrame(VkCommandBuffer& commandBuffer) override
        {
            int frameIdx = static_cast<int>(getRenderer()->getCurrentFrame() % MAX_FRAMES);

            // Safe: VulkanInstance waited on inFlightFences[currentFrame]
            deletionQueues[frameIdx].flush();

            // Free descriptor sets retired during a previous frame on this slot
            for (auto ds : retiredDescriptorSets[frameIdx])
                descriptorHelper->freeSet(ds);
            retiredDescriptorSets[frameIdx].clear();

            stagingBelt.recycle(usedStagingBlocks[frameIdx]);

            // Begin ring buffer frame — resets write head, passes DeletionQueue for growth
            if (ringBuffer.isValid())
                ringBuffer.beginFrame(frameIdx, &deletionQueues[frameIdx]);

            // Record upload commands for resources staged during previous frames.
            // These must execute before the render pass so images are in
            // SHADER_READ_ONLY_OPTIMAL layout when fragment shaders sample them.
            core::StagingBelt::recordUploads(commandBuffer, pendingUploads);

            // Move staging blocks used for these uploads to the per-frame list
            // so they stay alive until this frame's fence signals.
            stagingBelt.moveActiveTo(usedStagingBlocks[frameIdx]);
        }

        void render(VkCommandBuffer& commandBuffer) override
        {
            if (!device) return;
            auto bounds = getBounds();
            float w = bounds.getWidth();
            float h = bounds.getHeight();
            if (w <= 0 || h <= 0) return;

            // Lazy-allocate blur temp image (persistent, reused every frame)
            uint32_t fw = static_cast<uint32_t>(w), fh = static_cast<uint32_t>(h);
            int frameIdx = static_cast<int>(getRenderer()->getCurrentFrame() % MAX_FRAMES);
            if (blurTempImage.getWidth() != fw || blurTempImage.getHeight() != fh)
            {
                // Retire old descriptor set — freed once the GPU is done with this frame slot
                if (blurDescriptorSet != VK_NULL_HANDLE)
                    retiredDescriptorSets[frameIdx].push_back(blurDescriptorSet);
                blurDescriptorSet = VK_NULL_HANDLE;

                blurTempImage.destroy();
                if (fw > 0 && fh > 0)
                {
                    blurTempImage.create({ physDevice, device, fw, fh,
                        VK_FORMAT_B8G8R8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
                    blurTempImage.createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

                    // Update blur descriptor set
                    blurDescriptorSet = descriptorHelper->allocateSet();
                    if (blurDescriptorSet != VK_NULL_HANDLE)
                        core::DescriptorHelper::writeImage(device, blurDescriptorSet, 0,
                            blurTempImage.getView(), blurTempImage.getSampler());
                }
            }

            // Build render pass info for blur
            RenderPassInfo rpInfo;
            rpInfo.renderPass = getRenderer()->getRenderPass();
            // We need the current framebuffer and MSAA image — get from swapchain via renderer
            // Note: these are set by VulkanInstance before calling render
            rpInfo.extent = { fw, fh };

            // Update cache frame counters and evict stale entries
            texCache.currentFrame++;
            gradCache.currentFrame++;
            if (texCache.currentFrame % 60 == 0)
            {
                texCache.evict();
                gradCache.evict();
            }

            VulkanGraphicsContext ctx(
                commandBuffer, mainPipeline->getInternal(), mainPipeline->getLayout(),
                w, h, physDevice, device, 2.0f,
                &persistentBuffers[frameIdx], &mappedPtrs[frameIdx],
                stencilPipeline.get(), stencilCoverPipeline.get(),
                defaultDescriptorSet,
                editor.srgbPipelineMode, multiplyPipeline.get(),
                hsvPipeline.get(), blurPipeline.get(), rpInfo,
                &blurTempImage, blurDescriptorSet, &texCache, &glyphAtlas, &gradCache,
                &deletionQueues[frameIdx],
                ringBuffer.isValid() ? &ringBuffer : nullptr);
            ctx.mainClipPipeline = mainClipPipeline.get();

            juce::Graphics g(ctx);
            editor.inVulkanRender = true;
            editor.paintEntireComponent(g, false);
            editor.inVulkanRender = false;

            // FPS tracking — always active
            {
                auto now = std::chrono::steady_clock::now();
                fpsTimestamps.push_back(now);
                auto cutoff = now - std::chrono::seconds(1);
                while (!fpsTimestamps.empty() && fpsTimestamps.front() < cutoff)
                    fpsTimestamps.pop_front();
                editor.vulkanFps = static_cast<int>(fpsTimestamps.size());
            }

            // Stage dirty atlas pages for upload in the next frame's prepareFrame().
            // New glyphs render as invisible (white atlas background) for one frame.
            glyphAtlas.stageDirtyPages(stagingBelt, pendingUploads);

            graphics::flush(ctx);

            // End ring buffer frame — record end position
            if (ringBuffer.isValid())
                ringBuffer.endFrame();
        }

        AudioProcessorEditor& editor;
        std::unique_ptr<Pipeline> mainPipeline;
        std::unique_ptr<Pipeline> mainClipPipeline;
        std::unique_ptr<Pipeline> stencilPipeline;
        std::unique_ptr<Pipeline> stencilCoverPipeline;
        std::unique_ptr<Pipeline> multiplyPipeline;
        std::unique_ptr<Pipeline> hsvPipeline;
        std::unique_ptr<Pipeline> blurPipeline;
        std::unique_ptr<core::DescriptorHelper> descriptorHelper;
        std::unique_ptr<core::DescriptorHelper> hsvDescriptorHelper;
        core::Image defaultTexture;
        VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;
        VkPhysicalDevice physDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        static constexpr int MAX_FRAMES = 2;
        core::GPUMemoryPool memoryPool;             // sub-allocator (reduces vkAllocateMemory to ~3 calls)
        core::VertexRingBuffer ringBuffer;           // zero-alloc per-frame vertex uploads
        core::Buffer persistentBuffers[MAX_FRAMES]; // legacy fallback (used if ring buffer unavailable)
        void* mappedPtrs[MAX_FRAMES] = {};
        core::DeletionQueue deletionQueues[MAX_FRAMES];
        core::StagingBelt stagingBelt;
        std::vector<core::StagingBelt::Block> usedStagingBlocks[MAX_FRAMES];
        std::vector<core::PendingUpload> pendingUploads;
        core::Image blurTempImage;      // lazy persistent — allocated on first blur, reused
        VkDescriptorSet blurDescriptorSet = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> retiredDescriptorSets[MAX_FRAMES]; // freed after fence
        std::deque<std::chrono::steady_clock::time_point> fpsTimestamps;
        TextureCache texCache;          // GPU texture cache for drawImage
        GradientCache gradCache;        // GPU gradient LUT cache
        GlyphAtlas glyphAtlas;          // GPU glyph atlas for text rendering
    };

    VulkanRenderer vulkanRenderer;
    PaintBridge paintBridge;
    bool vulkanEnabled = true;
    bool srgbPipelineMode = false;
    bool inVulkanRender = false;
    int vulkanFps = 0;
};

} // jvk
