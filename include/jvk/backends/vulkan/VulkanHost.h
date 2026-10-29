/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanHost.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::vulkan
{

// =============================================================================
// WindowHost — a Vulkan swapchain in a native child view of a JUCE editor
//
// macOS: a layer-backed NSView with a CAMetalLayer (NSViewGenerator), hosted
// by a juce::NSViewComponent, surfaced through MoltenVK. Windows: a child
// HWND made in the peer's DPI-awareness context (HWNDGenerator), hosted by a
// juce::HWNDComponent. Neither takes mouse input: the editor's own components
// sit on top of the pixels and get every event.
// =============================================================================

class WindowHost final : public jvk::WindowHost
{
public:
    WindowHost() = default;
    ~WindowHost() override { detach(); }

    std::shared_ptr<gpu::Backend> backend() override;
    juce::String reason() const override { return reason_; }

    bool attach(juce::Component& editor, gpu::Extent sizeHint, juce::String& failure) override;
    void detach() override;

    gpu::Surface* surface() override { return surface_.get(); }
    void setBounds(juce::Rectangle<int> localBounds) override { view_.setBounds(localBounds); }
    bool needsReattach(juce::Component& editor) const override;

private:
    std::shared_ptr<Backend>          backend_;
    std::unique_ptr<SwapchainSurface> surface_;
    juce::Component*                  editor_ = nullptr;
    juce::String                      reason_;

#if JUCE_MAC
    std::unique_ptr<jvk::core::macos::NSViewGenerator> nsView_;
    juce::NSViewComponent view_;
#elif JUCE_WINDOWS
    std::unique_ptr<jvk::core::windows::HWNDGenerator> hwnd_;
    juce::HWNDComponent view_;
#else
    juce::Component view_;
#endif
};

} // namespace jvk::vulkan
