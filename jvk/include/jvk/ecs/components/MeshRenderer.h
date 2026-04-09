/*
  ==============================================================================

    MeshRenderer.h
    ECS component that holds a GPU mesh reference and issues draw commands.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class MeshRenderer : public ecs::ComponentBase<MeshRenderer>
{
public:
    MeshRenderer() = default;

    void setMesh(std::shared_ptr<objects::GPUMesh> m) { mesh = std::move(m); }

    objects::GPUMesh* getMesh() const { return mesh.get(); }

    void render(VkCommandBuffer cmd)
    {
        if (mesh && mesh->isValid())
        {
            mesh->bind(cmd);
            mesh->draw(cmd);
        }
    }

private:
    std::shared_ptr<objects::GPUMesh> mesh;
};

} // jvk
