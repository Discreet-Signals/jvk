/*
  ==============================================================================

    ECS.h
    Entity-Component System foundation.

    Component     — base class with CRTP type IDs and render lifecycle
    Updater<T>    — component that self-registers into an UpdaterVector
                    for vectorized per-frame batch processing

  ==============================================================================
*/

#pragma once

namespace jvk
{

class UpdaterVectorRegistry; // forward

namespace ecs
{

class Component
{
public:
    virtual ~Component() { }
    virtual int getType() const = 0;

    // Render lifecycle (per-object, called during draw)
    virtual void prepare() { }
    virtual void apply() { }
    virtual void cease() { }

    // Scene registration hooks (overridden by Updater<T>)
    virtual void registerSelf(UpdaterVectorRegistry&) { }
    virtual void unregisterSelf(UpdaterVectorRegistry&) { }
};

template <class T>
class ComponentBase : public Component
{
public:
    static int type;
    int getType() const override { return T::type; }
};

static int nextType = 0;
template <typename T> int ComponentBase<T>::type(nextType++);

} // ecs
} // jvk
