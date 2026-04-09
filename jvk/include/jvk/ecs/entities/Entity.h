/*
  ==============================================================================

    Entity.h
    Pure component container with scene back-pointer for auto-registration.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class Scene; // forward

class Entity
{
public:
    Entity() = default;

    virtual ~Entity()
    {
        if (scene)
            for (auto& c : components)
                c->unregisterSelf(getRegistry());
    }

    void addComponent(std::unique_ptr<ecs::Component>&& component)
    {
        if (scene)
            component->registerSelf(getRegistry());
        components.push_back(std::move(component));
    }

    template <class T>
    T* getComponent(int index = 0)
    {
        int i = 0;
        for (auto& component : components)
            if (component->getType() == T::type)
                if (i++ == index)
                    return static_cast<T*>(component.get());
        return nullptr;
    }

    template <class T>
    const T* getComponent(int index = 0) const
    {
        int i = 0;
        for (auto& component : components)
            if (component->getType() == T::type)
                if (i++ == index)
                    return static_cast<const T*>(component.get());
        return nullptr;
    }

    bool enabled = true;

    void setScene(Scene* s);
    Scene* getScene() const { return scene; }

protected:
    Scene* scene = nullptr;
    std::vector<std::unique_ptr<ecs::Component>> components;

private:
    inline UpdaterVectorRegistry& getRegistry();
};

} // jvk
