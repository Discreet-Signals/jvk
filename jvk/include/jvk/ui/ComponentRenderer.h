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
 File: ComponentRenderer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class ComponentRenderer : public Renderer
{
public:
    ComponentRenderer() = default;

    void setRoot(Component* r) { root = r; }
    Component* getRoot() const { return root; }

    // --- Mouse input routing ---
    void mouseDown(const juce::MouseEvent& e)  { routeMouseEvent(e, &Component::mouseDown); }
    void mouseUp(const juce::MouseEvent& e)    { routeMouseEvent(e, &Component::mouseUp); }
    void mouseDrag(const juce::MouseEvent& e)  { routeMouseEvent(e, &Component::mouseDrag); }
    void mouseMove(const juce::MouseEvent& e)
    {
        float x = static_cast<float>(e.getPosition().x);
        float y = static_cast<float>(e.getPosition().y);
        Component* hit = findComponentAt(root, x, y);

        if (hit != hoveredComponent)
        {
            if (hoveredComponent) hoveredComponent->mouseExit(e);
            hoveredComponent = hit;
            if (hoveredComponent) hoveredComponent->mouseEnter(e);
        }
        if (hoveredComponent)
            hoveredComponent->mouseMove(e);
    }

    // --- Renderer interface ---
    void initialize(VkPhysicalDevice phys, VkDevice dev,
                    VkRenderPass rp, VkSampleCountFlagBits msaa,
                    VkCommandPool cmdPool, VkQueue queue) override
    {
        physicalDevice = phys;
        device = dev;
        renderPass = rp;
        msaaSamples = msaa;
    }

    void shutdown() override {}

    void onRenderPassChanged(VkRenderPass rp, VkSampleCountFlagBits msaa) override
    {
        renderPass = rp;
        msaaSamples = msaa;
    }

    void render(VkCommandBuffer cmd, const Scene& scene) override
    {
        if (!root) return;
        renderComponent(cmd, root, scene);
    }

    // Set viewport dimensions for orthographic camera setup
    void setViewportSize(float w, float h)
    {
        viewportWidth = w;
        viewportHeight = h;
    }

private:
    void renderComponent(VkCommandBuffer cmd, Component* comp, const Scene& scene)
    {
        if (!comp->isVisible()) return;

        // Collect paint commands
        auto absoluteBounds = getAbsoluteBounds(comp);
        Graphics g(absoluteBounds);
        comp->paint(g);
        comp->clearDirty();

        // Set scissor to component bounds
        VkRect2D scissor;
        scissor.offset = {
            static_cast<int32_t>(absoluteBounds.getX()),
            static_cast<int32_t>(absoluteBounds.getY())
        };
        scissor.extent = {
            static_cast<uint32_t>(absoluteBounds.getWidth()),
            static_cast<uint32_t>(absoluteBounds.getHeight())
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Render draw commands
        for (const auto& drawCmd : g.getCommands())
            renderDrawCommand(cmd, drawCmd, absoluteBounds);

        // Render children (sorted by z-order)
        auto sortedChildren = comp->getChildren();
        std::sort(sortedChildren.begin(), sortedChildren.end(),
                   [](const Component* a, const Component* b) { return a->getZOrder() < b->getZOrder(); });

        for (auto* child : sortedChildren)
            renderComponent(cmd, child, scene);
    }

    void renderDrawCommand(VkCommandBuffer cmd, const DrawCommand& drawCmd,
                            const juce::Rectangle<float>& parentBounds)
    {
        // Apply clip if present
        if (drawCmd.hasClip)
        {
            VkRect2D clip;
            clip.offset = {
                static_cast<int32_t>(drawCmd.clipRect.getX()),
                static_cast<int32_t>(drawCmd.clipRect.getY())
            };
            clip.extent = {
                static_cast<uint32_t>(drawCmd.clipRect.getWidth()),
                static_cast<uint32_t>(drawCmd.clipRect.getHeight())
            };
            vkCmdSetScissor(cmd, 0, 1, &clip);
        }

        // TODO: Render draw commands using SDF pipelines
        // For now, the infrastructure is in place. Each DrawCommandType
        // will be rendered using the appropriate pipeline:
        //   - FillRect/DrawRect/FillRoundedRect/etc -> SDF shape pipeline
        //   - DrawText -> SDF text pipeline with font atlas
        //   - DrawImage -> Textured quad pipeline
        //
        // Implementation requires SPIR-V shaders for:
        //   1. SDF shape fragment shader (rect, rounded rect, ellipse, line)
        //   2. SDF text fragment shader (multi-channel signed distance field)
        //   3. Textured quad fragment shader
    }

    juce::Rectangle<float> getAbsoluteBounds(Component* comp)
    {
        auto b = comp->getBounds();
        auto* p = comp->getParent();
        while (p)
        {
            b = b.translated(p->getBounds().getX(), p->getBounds().getY());
            p = p->getParent();
        }
        return b;
    }

    Component* findComponentAt(Component* comp, float x, float y)
    {
        if (!comp || !comp->isVisible()) return nullptr;

        auto bounds = getAbsoluteBounds(comp);
        if (!bounds.contains(x, y)) return nullptr;

        // Check children in reverse z-order (front to back)
        auto sortedChildren = comp->getChildren();
        std::sort(sortedChildren.begin(), sortedChildren.end(),
                   [](const Component* a, const Component* b) { return a->getZOrder() > b->getZOrder(); });

        for (auto* child : sortedChildren)
        {
            auto* hit = findComponentAt(child, x, y);
            if (hit) return hit;
        }

        float localX = x - bounds.getX();
        float localY = y - bounds.getY();
        if (comp->hitTest(localX, localY))
            return comp;

        return nullptr;
    }

    template <typename MouseFn>
    void routeMouseEvent(const juce::MouseEvent& e, MouseFn fn)
    {
        float x = static_cast<float>(e.getPosition().x);
        float y = static_cast<float>(e.getPosition().y);
        Component* hit = findComponentAt(root, x, y);
        if (hit)
            (hit->*fn)(e);
    }

    Component* root = nullptr;
    Component* hoveredComponent = nullptr;
    float viewportWidth = 0, viewportHeight = 0;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
};

} // jvk
