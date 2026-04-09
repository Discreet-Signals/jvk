/*
  ==============================================================================

    Renderer.h
    Abstract renderer interface. Subclass to implement custom rendering
    approaches (forward, deferred, PBR, ray traced, etc).

  ==============================================================================
*/

#pragma once

namespace jvk
{

class Scene; // forward declaration

class Renderer
{
public:
    virtual ~Renderer() = default;

    // Called once when Vulkan context is available
    virtual void initialize(VkPhysicalDevice physicalDevice, VkDevice device,
                            VkRenderPass renderPass, VkSampleCountFlagBits msaa,
                            VkCommandPool commandPool, VkQueue graphicsQueue) = 0;

    virtual void shutdown() = 0;

    // Called when render pass or MSAA settings change (rebuild pipelines)
    virtual void onRenderPassChanged(VkRenderPass renderPass, VkSampleCountFlagBits msaa) = 0;

    // Render the scene for this frame
    virtual void render(VkCommandBuffer cmd, const Scene& scene) = 0;
};

} // jvk
