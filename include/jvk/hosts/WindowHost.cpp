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
namespace metal { std::unique_ptr<jvk::WindowHost> createWindowHost(); }   // MetalHost.mm
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

} // namespace jvk
