/*
  ==============================================================================

    Material.h
    Abstract material interface. Subclass to implement custom materials
    (unlit, PBR, ray traced, etc).

  ==============================================================================
*/

#pragma once

namespace jvk
{

class Material : public ecs::ComponentBase<Material>
{
public:
    virtual ~Material() = default;

    // Called by renderer to bind this material for drawing
    virtual void bind(VkCommandBuffer cmd) = 0;

    // Called when the renderer needs to (re)create pipelines
    virtual void createPipeline(VkDevice device, VkRenderPass renderPass,
                                 VkSampleCountFlagBits msaa) = 0;
    virtual void destroyPipeline() = 0;

    // Pipeline access for the renderer
    virtual VkPipeline getPipeline() const = 0;
    virtual VkPipelineLayout getPipelineLayout() const = 0;
};

} // jvk
