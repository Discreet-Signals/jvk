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
 File: Renderer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
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
