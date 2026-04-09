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
 File: Material.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
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
