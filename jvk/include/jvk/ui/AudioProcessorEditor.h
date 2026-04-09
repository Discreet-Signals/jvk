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
        vulkanRenderer.addChildComponent(&paintBridge);
        addComponentListener(this);
    }

    explicit AudioProcessorEditor(juce::AudioProcessor* p)
        : juce::AudioProcessorEditor(p), paintBridge(*this)
    {
        addAndMakeVisible(vulkanRenderer);
        vulkanRenderer.addChildComponent(&paintBridge);
        addComponentListener(this);
    }

    ~AudioProcessorEditor() override
    {
        removeComponentListener(this);
    }

    // Toggle between Vulkan and JUCE software rendering
    void setVulkanEnabled(bool enabled)
    {
        vulkanEnabled = enabled;
        vulkanRenderer.setVisible(enabled);
        repaint();
    }

    bool isVulkanEnabled() const { return vulkanEnabled; }

    bool isVulkanReady() const
    {
        return vulkanRenderer.getStatus() == core::VulkanStatus::Ready;
    }

    // Physically correct sRGB blending (brighter semi-transparent content)
    // vs JUCE-identical UNORM blending (default). Requires rebuild.
    void setSRGBPipeline(bool enabled)
    {
        // Would need swapchain + pipeline rebuild — for now just store the flag
        srgbPipelineMode = enabled;
    }
    bool isSRGBPipeline() const { return srgbPipelineMode; }

private:
    void componentMovedOrResized(juce::Component&, bool, bool wasResized) override
    {
        if (wasResized)
        {
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
            descriptorHelper->createPool(16); // 16 descriptor sets per frame

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
                stencilConfig.pushConstantRanges = pushRanges;

                stencilPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(stencilSg), renderer.getMSAASamples(),
                    std::move(stencilVertLayout), std::move(stencilConfig));
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

            // --- HSV effect pipeline (reads framebuffer via input attachment) ---
            {
                // Separate descriptor set layout for input attachment (set 0, binding 1)
                hsvDescriptorHelper = std::make_unique<core::DescriptorHelper>(device);
                hsvDescriptorHelper->addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT);
                hsvDescriptorHelper->addBinding(1, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT);
                auto hsvDsLayout = hsvDescriptorHelper->buildLayout();
                hsvDescriptorHelper->createPool(4);

                shaders::ShaderGroup hsvSg;
                hsvSg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                                 shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
                hsvSg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                                 shaders::ui2d::hsv_frag_spv, shaders::ui2d::hsv_frag_spvSize);
                hsvSg.addDescriptorSetLayout(hsvDsLayout);

                VertexLayout hsvVertLayout;
                hsvVertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
                hsvVertLayout.attributes = {
                    { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                    { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                    { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) },
                    { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, shapeInfo) }
                };

                PipelineConfig hsvConfig;
                hsvConfig.cullMode = VK_CULL_MODE_NONE;
                hsvConfig.depthTestEnable = false;
                hsvConfig.depthWriteEnable = false;
                hsvConfig.blendMode = BlendMode::Opaque; // overwrite with HSV result
                hsvConfig.pushConstantRanges = pushRanges;

                hsvPipeline = std::make_unique<Pipeline>(
                    device, renderer.getRenderPass(),
                    std::move(hsvSg), renderer.getMSAASamples(),
                    std::move(hsvVertLayout), std::move(hsvConfig));
            }

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

            setPipeline(mainPipeline->getInternal());
        }

        void removedFromRenderer(const VulkanRenderer&) override
        {
            mainPipeline.reset();
            stencilPipeline.reset();
            descriptorHelper.reset();
            defaultTexture.destroy();
        }

        void render(VkCommandBuffer& commandBuffer) override
        {
            auto bounds = getBounds();
            float w = bounds.getWidth();
            float h = bounds.getHeight();
            if (w <= 0 || h <= 0) return;

            // Lazy-allocate blur temp image (persistent, reused every frame)
            uint32_t fw = static_cast<uint32_t>(w), fh = static_cast<uint32_t>(h);
            if (blurTempImage.getWidth() != fw || blurTempImage.getHeight() != fh)
            {
                blurTempImage.destroy();
                if (fw > 0 && fh > 0)
                {
                    blurTempImage.create({ physDevice, device, fw, fh,
                        VK_FORMAT_B8G8R8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
                    blurTempImage.createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

                    // Update blur descriptor set
                    blurDescriptorSet = descriptorHelper->allocateSet();
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

            VulkanGraphicsContext ctx(
                commandBuffer, mainPipeline->getInternal(), mainPipeline->getLayout(),
                w, h, physDevice, device, 2.0f, &persistentBuffer, &mappedPtr,
                stencilPipeline.get(), defaultDescriptorSet,
                editor.srgbPipelineMode, multiplyPipeline.get(),
                hsvPipeline.get(), blurPipeline.get(), rpInfo,
                &blurTempImage, blurDescriptorSet);

            juce::Graphics g(ctx);
            editor.paintEntireComponent(g, false);
            ctx.flush();
        }

        AudioProcessorEditor& editor;
        std::unique_ptr<Pipeline> mainPipeline;
        std::unique_ptr<Pipeline> stencilPipeline;
        std::unique_ptr<Pipeline> multiplyPipeline;
        std::unique_ptr<Pipeline> hsvPipeline;
        std::unique_ptr<Pipeline> blurPipeline;
        std::unique_ptr<core::DescriptorHelper> descriptorHelper;
        std::unique_ptr<core::DescriptorHelper> hsvDescriptorHelper;
        core::Image defaultTexture;
        VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;
        VkPhysicalDevice physDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        core::Buffer persistentBuffer;
        void* mappedPtr = nullptr;
        core::Image blurTempImage;      // lazy persistent — allocated on first blur, reused
        VkDescriptorSet blurDescriptorSet = VK_NULL_HANDLE;
    };

    VulkanRenderer vulkanRenderer;
    PaintBridge paintBridge;
    bool vulkanEnabled = true;
    bool srgbPipelineMode = false;
};

} // jvk
