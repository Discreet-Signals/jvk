/*
  ==============================================================================

    Scene.h
    Container for objects, camera, raycasting, and UpdaterVectorRegistry.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class Scene
{
public:
    Scene() = default;

    Object& createObject(const juce::String& name)
    {
        auto obj = std::make_unique<Object>();
        obj->setScene(this);
        Object& ref = *obj;
        objects.insert_or_assign(name, std::move(obj));
        return ref;
    }

    Object* getObject(const juce::String& name)
    {
        auto it = objects.find(name);
        if (it == objects.end())
            return nullptr;
        return it->second.get();
    }

    void removeObject(const juce::String& name)
    {
        objects.erase(name);
    }

    Object* raycast(float screenX, float screenY,
                     float viewportWidth, float viewportHeight) const
    {
        glm::vec3 rayDir = camera.screenToWorldRay(screenX, screenY,
                                                     viewportWidth, viewportHeight);
        glm::vec3 rayOrigin = camera.transform->getPosition();
        return raycast(rayOrigin, rayDir);
    }

    Object* raycast(glm::vec3 origin, glm::vec3 direction) const
    {
        Object* closest = nullptr;
        float closestDist = std::numeric_limits<float>::max();

        for (auto& [name, obj] : objects)
        {
            if (!obj->enabled) continue;
            float dist;
            if (obj->rayIntersects(origin, direction, dist) && dist < closestDist)
            {
                closestDist = dist;
                closest = obj.get();
            }
        }
        return closest;
    }

    void update(double dt)
    {
        pools.update(dt);
    }

    template <typename T>
    UpdaterVector<T>& getUpdaterVector() { return pools.get<T>(); }

    template <typename Fn>
    void forEachObject(Fn&& fn)
    {
        for (auto& [name, obj] : objects)
            fn(name, *obj);
    }

    template <typename Fn>
    void forEachObject(Fn&& fn) const
    {
        for (const auto& [name, obj] : objects)
            fn(name, *obj);
    }

    Camera camera;
    UpdaterVectorRegistry pools;

private:
    std::map<juce::String, std::unique_ptr<Object>> objects;
};

// Entity implementation that needs Scene definition
inline void Entity::setScene(Scene* s)
{
    if (scene)
        for (auto& c : components)
            c->unregisterSelf(scene->pools);
    scene = s;
    if (scene)
        for (auto& c : components)
            c->registerSelf(scene->pools);
}

inline UpdaterVectorRegistry& Entity::getRegistry()
{
    return scene->pools;
}

} // jvk
