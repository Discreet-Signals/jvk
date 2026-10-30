/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: WindowHost.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// WindowHost — a backend's way into a JUCE window
//
// What jvk::AudioProcessorEditor needs from a backend, and nothing it can do
// itself: the process-wide gpu::Backend, and a native child view inside the
// editor's peer with a Surface presenting into it. Everything else (the frame
// loop, paint, the software fallback, resize policy) is the editor's.
//
// One per editor. The hosts are backends/metal/MetalHost.mm and
// backends/vulkan/VulkanHost.h; createWindowHost() picks the build's.
// =============================================================================

class WindowHost
{
public:
    virtual ~WindowHost() = default;

    // The backend every window and offscreen renderer of this kind shares,
    // made on first ask. Null when this machine can't run it (reason()).
    virtual std::shared_ptr<gpu::Backend> backend() = 0;
    virtual juce::String reason() const = 0;

    // Makes the native view inside `editor` (its bounds the editor's local
    // bounds) and the surface presenting into it, `sizeHint` physical pixels
    // until the platform says otherwise. False with `failure` set when it
    // can't; false with `failure` empty when it can't YET (Windows: the
    // editor has no peer), so the editor asks again once it does.
    virtual bool attach(juce::Component& editor, gpu::Extent sizeHint, juce::String& failure) = 0;
    // Destroys the surface and the native view. The editor's renderer must be
    // gone, or idle, first.
    virtual void detach() = 0;

    virtual gpu::Surface* surface() = 0;
    virtual void setBounds(juce::Rectangle<int> localBounds) = 0;

    // True when the native view was made in a context the editor's window no
    // longer has (Windows: a host rebuilt the window in another DPI awareness)
    // and the whole stack must be detached and attached again.
    virtual bool needsReattach(juce::Component& editor) const { (void) editor; return false; }
};

std::unique_ptr<WindowHost> createWindowHost();

// The process's shared backend for this build: what the window hosts draw
// with, and what offscreen hosts (ShaderImage) and compute (jvk::compute) use.
// Made or shared without a window, so any thread may ask. Null when it can't
// run, and `whyNot` says why.
std::shared_ptr<gpu::Backend> defaultBackend(juce::String* whyNot = nullptr);

} // namespace jvk
