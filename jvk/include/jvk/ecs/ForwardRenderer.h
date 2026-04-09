/*
  ==============================================================================

    ForwardRenderer.h
    Default forward renderer. Iterates objects, binds materials, draws meshes.
    Sorts opaque objects front-to-back, transparent back-to-front.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class ForwardRenderer : public Renderer
{
public:
    ForwardRenderer() = default;

    void initialize(VkPhysicalDevice phys, VkDevice dev,
                    VkRenderPass rp, VkSampleCountFlagBits msaa,
                    VkCommandPool cmdPool, VkQueue queue) override
    {
        physicalDevice = phys;
        device = dev;
        renderPass = rp;
        msaaSamples = msaa;
        commandPool = cmdPool;
        graphicsQueue = queue;
    }

    void shutdown() override
    {
    }

    void onRenderPassChanged(VkRenderPass rp, VkSampleCountFlagBits msaa) override
    {
        renderPass = rp;
        msaaSamples = msaa;
    }

    void render(VkCommandBuffer cmd, const Scene& scene) override
    {
        // Render all enabled objects
        const_cast<Scene&>(scene).forEachObject([&](const juce::String&, Object& obj)
        {
            if (obj.enabled)
                obj.render(cmd, scene.camera);
        });
    }

private:
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
};

} // jvk
