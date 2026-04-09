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
 File: MeshRenderer.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
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
