/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanHost.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk
{

std::unique_ptr<WindowHost> createWindowHost()
{
    return std::make_unique<vulkan::WindowHost>();
}

} // namespace jvk

namespace jvk::vulkan
{

std::shared_ptr<gpu::Backend> WindowHost::backend()
{
    if (backend_ == nullptr) backend_ = Backend::acquire();
    if (!backend_->usable())
    {
        reason_ = backend_->unavailableReason();
        return nullptr;
    }
    if (backend_->isLost())
    {
        reason_ = "the graphics device was lost";
        return nullptr;
    }
    return backend_;
}

bool WindowHost::attach(juce::Component& editor, gpu::Extent sizeHint, juce::String& failure)
{
    failure = {};
    if (surface_ != nullptr) return true;
    if (backend() == nullptr) { failure = reason_; return false; }

    VkInstance instance = backend_->device().instance();
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    void* nativeWindow = nullptr;

#if JUCE_MAC
    // Born at the editor's size: the first swapchain is built from the
    // layer's currentExtent, and a wrong-aspect birth frame presents the
    // opening frames stretched until the next extent check.
    editor.addAndMakeVisible(view_);
    view_.setInterceptsMouseClicks(false, false);
    view_.setBounds(editor.getLocalBounds());
    nsView_ = std::make_unique<jvk::core::macos::NSViewGenerator>();
    void* nsView = nsView_->create(editor.getWidth(), editor.getHeight());
    if (nsView == nullptr) { detach(); failure = "the Vulkan view could not be created"; return false; }
    view_.setView(nsView);

    VkMacOSSurfaceCreateInfoMVK ci {};
    ci.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
    ci.pView = nsView;
    if (vkCreateMacOSSurfaceMVK == nullptr
        || vkCreateMacOSSurfaceMVK(instance, &ci, nullptr, &surface) != VK_SUCCESS)
        surface = VK_NULL_HANDLE;
#elif JUCE_WINDOWS
    // The child HWND must be made in the DPI-awareness context of the window
    // tree it joins (Windows can't parent across contexts), which is only
    // knowable from the peer: until there is one, try again later.
    auto* peer = editor.getPeer();
    if (peer == nullptr) return false;
    editor.addAndMakeVisible(view_);
    view_.setInterceptsMouseClicks(false, false);
    view_.setBounds(editor.getLocalBounds());
    hwnd_ = std::make_unique<jvk::core::windows::HWNDGenerator>();
    HWND child = hwnd_->create(peer->getNativeHandle());
    if (child == nullptr) { detach(); failure = "the Vulkan window could not be created"; return false; }
    view_.setHWND(static_cast<void*>(child));
    nativeWindow = static_cast<void*>(child);

    VkWin32SurfaceCreateInfoKHR ci {};
    ci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hwnd      = child;
    // The HINSTANCE that registered the window class: the plugin's, not the host's.
    ci.hinstance = jvk::core::windows::HWNDGenerator::moduleHInstance();
    if (vkCreateWin32SurfaceKHR == nullptr
        || vkCreateWin32SurfaceKHR(instance, &ci, nullptr, &surface) != VK_SUCCESS)
        surface = VK_NULL_HANDLE;
#else
    juce::ignoreUnused(instance, sizeHint);
    failure = "Vulkan windows aren't supported on this platform yet";
    return false;
#endif

#if JUCE_MAC || JUCE_WINDOWS
    editor_ = &editor;
    if (surface == VK_NULL_HANDLE)
    {
        detach();
        failure = "the Vulkan surface could not be created";
        return false;
    }
    surface_ = std::make_unique<SwapchainSurface>(backend_, surface, sizeHint, nativeWindow);
    view_.setVisible(true);
    diag::log("vulkan: swapchain " + juce::String((int) surface_->extent().width) + "x"
              + juce::String((int) surface_->extent().height));
    return true;
#endif
}

void WindowHost::detach()
{
    surface_.reset();   // waits for its frames on the GPU
#if JUCE_MAC
    view_.setView(nullptr);
    nsView_.reset();
#elif JUCE_WINDOWS
    view_.setHWND(nullptr);
    hwnd_.reset();
#endif
    view_.setVisible(false);
    if (auto* parent = view_.getParentComponent()) parent->removeChildComponent(&view_);
    editor_ = nullptr;
}

bool WindowHost::needsReattach(juce::Component& editor) const
{
#if JUCE_WINDOWS
    // A live window's DPI-awareness context can't change, and a host can
    // rebuild the editor's window tree in the other one (Ableton's
    // "Auto-Scale Plugin Window"): every cross-window coordinate is then
    // virtualized, so the stack is rebuilt in the new context.
    if (auto* peer = editor.getPeer())
        if (hwnd_ != nullptr && hwnd_->get() != nullptr)
            return !jvk::core::windows::dpi::sameContext(hwnd_->get(), static_cast<HWND>(peer->getNativeHandle()));
#else
    juce::ignoreUnused(editor);
#endif
    return false;
}

} // namespace jvk::vulkan
