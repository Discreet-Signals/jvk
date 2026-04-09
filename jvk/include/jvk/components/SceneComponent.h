/*
  ==============================================================================

    SceneComponent.h
    VulkanComponent that hosts a Scene + pluggable Renderer.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class SceneComponent : public VulkanComponent
{
public:
    SceneComponent() = default;

    Scene& getScene() { return scene; }
    const Scene& getScene() const { return scene; }
    Camera& getCamera() { return scene.camera; }

    // Set the renderer implementation
    void setRenderer(std::unique_ptr<Renderer> r) { sceneRenderer = std::move(r); }
    Renderer* getRendererImpl() const { return sceneRenderer.get(); }

    // Deferred initialization (runs after Vulkan context is ready)
    void initializeScene(std::function<void(Scene&)>&& fn)
    {
        if (initialized && getRenderer())
            fn(scene);
        else
            initFunctions.push_back(std::move(fn));
    }

protected:
    void addedToRenderer(const VulkanRenderer& renderer) override
    {
        if (sceneRenderer)
        {
            sceneRenderer->initialize(renderer.getPhysicalDevice(),
                                       renderer.getDevice(),
                                       renderer.getRenderPass(),
                                       renderer.getMSAASamples(),
                                       VK_NULL_HANDLE, // commandPool not exposed yet
                                       VK_NULL_HANDLE); // queue not exposed yet
        }

        for (auto& fn : initFunctions)
            fn(scene);
        initFunctions.clear();
        initialized = true;
    }

    void removedFromRenderer(const VulkanRenderer& renderer) override
    {
        if (sceneRenderer)
            sceneRenderer->shutdown();
        initialized = false;
    }

    void render(VkCommandBuffer& commandBuffer) override
    {
        scene.update(1.0 / 60.0); // TODO: compute actual dt from timer
        if (sceneRenderer)
            sceneRenderer->render(commandBuffer, scene);
    }

    void resized() override
    {
        auto bounds = getBounds();
        if (bounds.getWidth() > 0 && bounds.getHeight() > 0)
        {
            float aspect = bounds.getWidth() / bounds.getHeight();
            if (!scene.camera.isOrthographic())
                scene.camera.setPerspective(glm::radians(45.0f), aspect);
        }
    }

private:
    Scene scene;
    std::unique_ptr<Renderer> sceneRenderer;
    std::vector<std::function<void(Scene&)>> initFunctions;
    bool initialized = false;
};

} // jvk
