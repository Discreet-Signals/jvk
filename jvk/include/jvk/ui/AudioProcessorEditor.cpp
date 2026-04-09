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
 File: AudioProcessorEditor.cpp
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

    bool isVulkanReady() const
    {
        return vulkanRenderer.getStatus() == core::VulkanStatus::Ready;
    }

private:
    void componentMovedOrResized(juce::Component&, bool, bool wasResized) override
    {
        if (wasResized)
        {
            vulkanRenderer.setBounds(getLocalBounds());
            paintBridge.setBounds(getLocalBounds().toFloat());
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

            shaders::ShaderGroup sg;
            sg.addShader(VK_SHADER_STAGE_VERTEX_BIT,
                          shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize);
            sg.addShader(VK_SHADER_STAGE_FRAGMENT_BIT,
                          shaders::ui2d::color_frag_spv, shaders::ui2d::color_frag_spvSize);

            VertexLayout vertLayout;
            vertLayout.binding = { 0, sizeof(UIVertex), VK_VERTEX_INPUT_RATE_VERTEX };
            vertLayout.attributes = {
                { 0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, position) },
                { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  offsetof(UIVertex, color) },
                { 2, 0, VK_FORMAT_R32G32_SFLOAT,        offsetof(UIVertex, uv) }
            };

            PipelineConfig config;
            config.cullMode = VK_CULL_MODE_NONE;
            config.depthTestEnable = false;
            config.depthWriteEnable = false;
            config.blendMode = BlendMode::AlphaBlend;
            config.pushConstantRanges = {
                { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }
            };

            pipeline = std::make_unique<Pipeline>(
                device, renderer.getRenderPass(),
                std::move(sg), renderer.getMSAASamples(),
                std::move(vertLayout), std::move(config));

            setPipeline(pipeline->getInternal());
        }

        void removedFromRenderer(const VulkanRenderer&) override
        {
            frameBuffers.clear();
            pipeline.reset();
        }

        void render(VkCommandBuffer& commandBuffer) override
        {
            // Release previous frame's GPU buffers (safe — fence was waited)
            frameBuffers.clear();

            auto bounds = getBounds();
            float w = bounds.getWidth();
            float h = bounds.getHeight();
            if (w <= 0 || h <= 0) return;

            VulkanGraphicsContext ctx(
                commandBuffer, pipeline->getInternal(), pipeline->getLayout(),
                w, h, physDevice, device);

            // Create a juce::Graphics backed by our Vulkan context
            juce::Graphics g(ctx);

            // Paint the editor itself
            editor.paint(g);

            // Recursively paint all child components (except the VulkanRenderer)
            paintChildren(g, editor);

            ctx.flush();

            // Keep vertex buffers alive until GPU finishes
            frameBuffers = std::move(ctx.getFrameBuffers());
        }

    private:
        void paintChildren(juce::Graphics& g, juce::Component& parent)
        {
            for (int i = 0; i < parent.getNumChildComponents(); i++)
            {
                auto* child = parent.getChildComponent(i);

                // Skip the VulkanRenderer itself (it's our own surface)
                if (child == &editor.vulkanRenderer)
                    continue;

                if (!child->isVisible() || child->getWidth() <= 0 || child->getHeight() <= 0)
                    continue;

                auto childBounds = child->getBounds();

                g.saveState();
                g.setOrigin(childBounds.getPosition());
                g.reduceClipRegion(child->getLocalBounds());
                child->paint(g);
                child->paintOverChildren(g);
                paintChildren(g, *child);
                g.restoreState();
            }
        }

        AudioProcessorEditor& editor;
        std::unique_ptr<Pipeline> pipeline;
        VkPhysicalDevice physDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        std::vector<core::Buffer> frameBuffers;
    };

    VulkanRenderer vulkanRenderer;
    PaintBridge paintBridge;
};

} // jvk
