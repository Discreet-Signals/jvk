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
 File: ComponentPool.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// Base for type-erased storage in the registry
class UpdaterVectorBase
{
public:
    virtual ~UpdaterVectorBase() = default;
    virtual void update(double dt) = 0;
    virtual size_t size() const = 0;
};

// Typed vector: contiguous pointers to updaters of type T
template <typename T>
class UpdaterVector : public UpdaterVectorBase
{
public:
    void add(T* item) { items.push_back(item); }

    void remove(T* item)
    {
        items.erase(std::remove(items.begin(), items.end(), item), items.end());
    }

    void update(double dt) override
    {
        for (auto* item : items)
            item->update(dt);
    }

    size_t size() const override { return items.size(); }

    const std::vector<T*>& getItems() const { return items; }

    template <typename Fn>
    void forEach(Fn&& fn)
    {
        for (auto* item : items)
            fn(*item);
    }

private:
    std::vector<T*> items;
};

// Registry: one UpdaterVector per type, auto-created on first access
class UpdaterVectorRegistry
{
public:
    template <typename T>
    UpdaterVector<T>& get()
    {
        int typeId = ecs::ComponentBase<T>::type;

        if (typeId >= static_cast<int>(vectors.size()))
            vectors.resize(typeId + 1);

        if (!vectors[typeId])
            vectors[typeId] = std::make_unique<UpdaterVector<T>>();

        return *static_cast<UpdaterVector<T>*>(vectors[typeId].get());
    }

    void update(double dt)
    {
        for (auto& v : vectors)
            if (v)
                v->update(dt);
    }

private:
    std::vector<std::unique_ptr<UpdaterVectorBase>> vectors;
};

// Updater<T>: component that needs per-frame batch updates.
// Inheriting from this automatically registers into the scene's
// UpdaterVector<T> when the component is added to an entity.
template <class T>
class Updater : public ecs::ComponentBase<T>
{
public:
    virtual void update(double dt) = 0;

    void registerSelf(UpdaterVectorRegistry& registry) override
    {
        registry.get<T>().add(static_cast<T*>(this));
        registered = &registry;
    }

    void unregisterSelf(UpdaterVectorRegistry& registry) override
    {
        registry.get<T>().remove(static_cast<T*>(this));
        registered = nullptr;
    }

    ~Updater() override
    {
        if (registered)
            registered->get<T>().remove(static_cast<T*>(this));
    }

private:
    UpdaterVectorRegistry* registered = nullptr;
};

} // jvk
