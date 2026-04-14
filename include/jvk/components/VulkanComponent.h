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
 File: VulkanComponent.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class VulkanRenderer;

class VulkanComponent
{
public:
    VulkanComponent() { }
    virtual ~VulkanComponent() { if (renderer != nullptr) removedFromRenderer(*renderer); }

    virtual void addedToRenderer(const jvk::VulkanRenderer& renderer) { }
    virtual void removedFromRenderer(const jvk::VulkanRenderer& renderer) { }
    virtual void prepareFrame(VkCommandBuffer& commandBuffer) { }
    virtual void render(VkCommandBuffer& commandBuffer) { }
    virtual void resized() { }

    void setPipeline(VkPipeline new_pipeline) { pipeline = new_pipeline; }
    VkPipeline getPipeline() const { return pipeline; }

    void setBounds(juce::Rectangle<float> new_bounds) { bounds = new_bounds; resized(); }
    void setBounds(float x, float y, float width, float height) { setBounds({x,y,width,height}); }
    juce::Rectangle<float> getBounds() const { return bounds; }

    void setOrder(int new_order) { order = new_order; }
    int getOrder() const { return order; }

    void sortChildren();
    void addChildComponent(VulkanComponent* child, bool sort_children = true);
    void addChildComponentWithPipeline(VulkanComponent* child, VkPipeline pipeline, bool sort_children = true);
    void addChildComponentWithDefaultPipeline(VulkanComponent* child, bool sort_children = true);
    const std::vector<VulkanComponent*> getChildren() const { return children; }
    VulkanRenderer* getRenderer() const { return renderer; }
    VulkanComponent* getParent() { return parent; }

private:
    friend class VulkanRenderer;
    void setRenderer(VulkanRenderer* renderer);
    void renderInternal(VkCommandBuffer& commandBuffer);
    void renderChildren(VkCommandBuffer& commandBuffer);
    VulkanRenderer* renderer { nullptr };
    VulkanComponent* parent { nullptr };
    std::vector<VulkanComponent*> children;
    juce::Rectangle<float> bounds { 0,0,0,0 };
    VkPipeline pipeline { VK_NULL_HANDLE };
    int order { 0 };

};

#if JUCE_MAC
using OSWindowComponent = core::macos::VulkanNSViewComponent;
#elif JUCE_WINDOWS
using OSWindowComponent = core::windows::VulkanHWNDComponent;
#endif

class VulkanRenderer :
    private VulkanComponent,
    public OSWindowComponent,
    private juce::Timer
{
public:
    VulkanRenderer();
    explicit VulkanRenderer(const core::VulkanRendererSettings& s);
    virtual ~VulkanRenderer()
    {
        stopTimer();
        rendering = false;
        if (device)
            vkDeviceWaitIdle(device);
    }

    // --- Status ---
    core::VulkanStatus getStatus() const { return status; }
    std::function<void(core::VulkanStatus)> onStatusChanged;

    // --- Settings accessors ---
    VkPipeline getDefaultPipeline() const { return defaultPipeline->getInternal(); }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkDevice getDevice() const { return device; }
    VkRenderPass getRenderPass() const { return renderPass; }
    VkSampleCountFlagBits getMSAASamples() const { return settings.msaaSamples; }
    VkSampleCountFlagBits getMaxSupportedMSAA() const { return maxSupportedMSAA; }
    VkCommandPool getCommandPool() const { return commandPool; }
    VkQueue getGraphicsQueue() const { return graphicsQueue; }
    size_t getCurrentFrame() const { return currentFrame; }
    const core::VulkanRendererSettings& getSettings() const { return settings; }

    // --- Runtime settings changes ---
    void setMSAA(VkSampleCountFlagBits samples);
    void setPresentMode(VkPresentModeKHR mode);
    void setMaxSize(int maxPixelWidth, int maxPixelHeight);
    void setRendering(bool enabled);
    bool isRendering() const { return rendering; }

    // --- Bounds ---
    void setBounds(juce::Rectangle<int> new_bounds) { OSWindowComponent::setBounds(new_bounds); }
    void setBounds(int x, int y, int width, int height) { OSWindowComponent::setBounds({x,y,width,height}); }
    juce::Rectangle<int> getBounds() const { return OSWindowComponent::getBounds(); }

    // --- Component hierarchy ---
    void sortChildren() { VulkanComponent::sortChildren(); }
    void addChildComponent(VulkanComponent* child, bool sort_children = true) { VulkanComponent::addChildComponent(child, sort_children); }
    void addChildComponentWithPipeline(VulkanComponent* child, VkPipeline pipeline, bool sort_children = true) { VulkanComponent::addChildComponentWithPipeline(child, pipeline, sort_children); }
    void addChildComponentWithDefaultPipeline(VulkanComponent* child, bool sort_children = true) { VulkanComponent::addChildComponentWithDefaultPipeline(child, sort_children); }
    const std::vector<VulkanComponent*> getChildren() const { return children; }

protected:
    virtual void render(VkCommandBuffer& commandBuffer) override { }

private:
    void prepareComponents(VkCommandBuffer& commandBuffer) override
    {
        prepareFrame(commandBuffer);
        const std::vector<VulkanComponent*> components = VulkanComponent::getChildren();
        for (VulkanComponent* comp : components)
            comp->prepareFrame(commandBuffer);
    }

    void renderComponents(VkCommandBuffer& commandBuffer) override
    {
        renderInternal(commandBuffer);
        const std::vector<VulkanComponent*> components = VulkanComponent::getChildren();
        for (VulkanComponent* comp : components)
        {
            comp->renderInternal(commandBuffer);
            comp->renderChildren(commandBuffer);
        }
    };

    void timerCallback() override
    {
        if (!rendering || status != core::VulkanStatus::Ready) return;

        // Non-blocking fence check — only render when a frame slot is free.
        // Avoids blocking the message thread on vkWaitForFences inside execute().
        if (vkGetFenceStatus(device, inFlightFences[currentFrame]) != VK_SUCCESS)
            return; // GPU still working on this slot, try again next tick

        checkForResize();
        execute();
    }

    void rebuildAll();
    void notifyChildrenRemoved();
    void notifyChildrenAdded();

    bool rendering = false;
};

} // jvk
