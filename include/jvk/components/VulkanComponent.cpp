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
 File: VulkanComponent.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "VulkanComponent.h"

namespace jvk
{

void VulkanComponent::sortChildren()
{
    std::sort(children.begin(), children.end(), [](VulkanComponent* a, VulkanComponent* b) { return a->getOrder() < b->getOrder(); });
}

void VulkanComponent::addChildComponent(VulkanComponent* child, bool sort_children)
{
    if (dynamic_cast<VulkanRenderer*>(child))
    {
        DBG("jvk::VulkanComponent - Cannot add VulkanRenderer as child component!");
        return;
    }

    for (VulkanComponent* c : children)
        if (child == c)
            return;
    children.push_back(child);
    child->parent = this;
    child->setRenderer(renderer);
    if (sort_children)
        sortChildren();
}

void VulkanComponent::addChildComponentWithPipeline(VulkanComponent* child, VkPipeline child_pipeline, bool sort_children)
{
    child->setPipeline(child_pipeline);
    addChildComponent(child);
}
void VulkanComponent::addChildComponentWithDefaultPipeline(VulkanComponent* child, bool sort_children)
{
    child->setPipeline(pipeline);
    addChildComponent(child);
}

void VulkanComponent::setRenderer(VulkanRenderer* new_renderer)
{
    if (new_renderer == nullptr)
        return;

    if (renderer != nullptr)
        removedFromRenderer(*renderer);

    renderer = new_renderer;
    addedToRenderer(*renderer);

    if (getPipeline() == VK_NULL_HANDLE)
        DBG("jvk::VulkanComponent - Child added with invalid pipeline!");

    for (VulkanComponent* child : children)
        child->setRenderer(new_renderer);
}

void VulkanComponent::renderInternal(VkCommandBuffer& commandBuffer)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipeline());

    VkViewport viewport = {};
    viewport.x = bounds.getX();
    viewport.y = bounds.getY();
    viewport.width = bounds.getWidth();
    viewport.height = bounds.getHeight();
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {static_cast<int32_t>(viewport.x), static_cast<int32_t>(viewport.y)};
    scissor.extent = {static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height)};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    render(commandBuffer);
}

void VulkanComponent::renderChildren(VkCommandBuffer& commandBuffer)
{
    for (VulkanComponent* child : children)
    {
        child->renderInternal(commandBuffer);
        child->renderChildren(commandBuffer);
    }
}

// =============================================================================
// VulkanRenderer
// =============================================================================

VulkanRenderer::VulkanRenderer()
{
    renderer = this;
    initializeVulkan();
    if (status == core::VulkanStatus::Ready)
        setPipeline(getDefaultPipeline());
}

VulkanRenderer::VulkanRenderer(const core::VulkanRendererSettings& s)
{
    settings = s;
    renderer = this;
    initializeVulkan();
    if (status == core::VulkanStatus::Ready)
        setPipeline(getDefaultPipeline());
}

// =============================================================================
// Runtime settings changes
// =============================================================================

void VulkanRenderer::setMSAA(VkSampleCountFlagBits samples)
{
    if (samples > maxSupportedMSAA) samples = maxSupportedMSAA;
    if (samples == settings.msaaSamples) return;
    settings.msaaSamples = samples;
    rebuildAll();
}

void VulkanRenderer::setPresentMode(VkPresentModeKHR mode)
{
    if (mode == settings.presentMode) return;
    settings.presentMode = mode;
    rebuildAll();
}

void VulkanRenderer::setRendering(bool enabled)
{
    rendering = enabled;
    if (enabled)
        startTimerHz(1000);
    else
        stopTimer();
}

void VulkanRenderer::rebuildAll()
{
    if (status != core::VulkanStatus::Ready) return;

    bool wasRendering = rendering;
    stopTimer();

    if (device) vkDeviceWaitIdle(device);

    notifyChildrenRemoved();
    rebuildSwapchainAndPipelines();
    notifyChildrenAdded();

    setPipeline(getDefaultPipeline());

    if (wasRendering)
        startTimerHz(1000);
}

void VulkanRenderer::notifyChildrenRemoved()
{
    for (VulkanComponent* child : VulkanComponent::getChildren())
    {
        child->removedFromRenderer(*this);
        for (VulkanComponent* grandchild : child->getChildren())
        {
            grandchild->removedFromRenderer(*this);
        }
    }
}

void VulkanRenderer::notifyChildrenAdded()
{
    for (VulkanComponent* child : VulkanComponent::getChildren())
    {
        child->addedToRenderer(*this);
        for (VulkanComponent* grandchild : child->getChildren())
        {
            grandchild->addedToRenderer(*this);
        }
    }
}

} // jvk
