/*
  ==============================================================================

    ObjectImpl.h
    Object::render implementation — included after Camera.h.

  ==============================================================================
*/

#pragma once

namespace jvk
{

inline void Object::render(VkCommandBuffer cmd, const Camera& camera)
{
    if (!enabled)
        return;

    for (auto& comp : components)
        comp->prepare();
    for (auto& comp : components)
        comp->apply();

    if (auto* mat = getMaterial())
    {
        mat->bind(cmd);

        glm::mat4 mvp = camera.getViewProjection() * transform->getMatrix();
        vkCmdPushConstants(cmd, mat->getPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
    }

    if (auto* mr = getMeshRenderer())
        mr->render(cmd);

    for (auto& comp : components)
        comp->cease();
}

} // jvk
