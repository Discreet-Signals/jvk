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
 File: Object.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Camera; // forward declaration

class Object : public Entity
{
public:
    Object()
    {
        auto t = std::make_unique<Transform>();
        transform = t.get();
        addComponent(std::move(t));
    }

    // Declared here, defined after Camera (see bottom of file or inline after Camera.h)
    inline void render(VkCommandBuffer cmd, const Camera& camera);

    // Ray intersection (AABB from transform position + scale)
    bool rayIntersects(glm::vec3 rayOrigin, glm::vec3 rayDir, float& distance) const
    {
        glm::vec3 pos = transform->getPosition();
        glm::vec3 halfSize = transform->getScale() * 0.5f;
        glm::vec3 bmin = pos - halfSize;
        glm::vec3 bmax = pos + halfSize;

        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; i++)
        {
            if (std::abs(rayDir[i]) < 1e-8f)
            {
                if (rayOrigin[i] < bmin[i] || rayOrigin[i] > bmax[i])
                    return false;
            }
            else
            {
                float t1 = (bmin[i] - rayOrigin[i]) / rayDir[i];
                float t2 = (bmax[i] - rayOrigin[i]) / rayDir[i];
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax)
                    return false;
            }
        }

        distance = tmin >= 0.0f ? tmin : tmax;
        return distance >= 0.0f;
    }

    // Convenience accessors
    Transform* getTransform() { return transform; }
    const Transform* getTransform() const { return transform; }
    MeshRenderer* getMeshRenderer() { return getComponent<MeshRenderer>(); }
    Material* getMaterial() { return getComponent<Material>(); }
    Texture* getTexture(int index = 0) { return getComponent<Texture>(index); }

    Transform* transform = nullptr;
};

} // jvk
