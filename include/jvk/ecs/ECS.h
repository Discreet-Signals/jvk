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
 File: ECS.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
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
