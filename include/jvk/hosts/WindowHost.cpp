/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: WindowHost.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk
{

#if JVK_METAL
namespace metal   // MetalHost.mm (Objective-C++: its types aren't seen here)
{
std::unique_ptr<jvk::WindowHost> createWindowHost();
std::shared_ptr<gpu::Backend>    defaultBackend(juce::String* whyNot);
}
#endif

// The build's backend: Metal on macOS, Vulkan everywhere else (JVK_BACKENDS).
// A build with both runs Metal unless JVK_BACKEND=vulkan, to compare them.
std::unique_ptr<WindowHost> createWindowHost()
{
#if JVK_METAL && JVK_VULKAN
    if (juce::SystemStats::getEnvironmentVariable("JVK_BACKEND", {}).equalsIgnoreCase("vulkan"))
        return std::make_unique<vulkan::WindowHost>();
#endif
#if JVK_METAL
    return metal::createWindowHost();
#else
    return std::make_unique<vulkan::WindowHost>();
#endif
}

std::shared_ptr<gpu::Backend> defaultBackend(juce::String* whyNot)
{
#if JVK_VULKAN
    auto vulkanBackend = [whyNot]() -> std::shared_ptr<gpu::Backend>
    {
        auto b = vulkan::Backend::acquire();
        const juce::String why = !b->usable() ? b->unavailableReason()
                               : b->isLost()  ? juce::String("the graphics device was lost") : juce::String();
        if (why.isEmpty()) return b;
        if (whyNot != nullptr) *whyNot = why;
        return nullptr;
    };
#endif
#if JVK_METAL && JVK_VULKAN
    if (juce::SystemStats::getEnvironmentVariable("JVK_BACKEND", {}).equalsIgnoreCase("vulkan"))
        return vulkanBackend();
#endif
#if JVK_METAL
    return metal::defaultBackend(whyNot);
#else
    return vulkanBackend();
#endif
}

} // namespace jvk
