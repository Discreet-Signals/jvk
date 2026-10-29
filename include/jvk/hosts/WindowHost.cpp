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

#if JUCE_MAC
namespace metal { std::unique_ptr<jvk::WindowHost> createWindowHost(); }   // jvk_metal.mm
#endif

// Metal on macOS (JVK_BACKEND=vulkan runs MoltenVK instead, to compare them);
// Vulkan everywhere else.
std::unique_ptr<WindowHost> createWindowHost()
{
#if JUCE_MAC
    if (! juce::SystemStats::getEnvironmentVariable("JVK_BACKEND", {}).equalsIgnoreCase("vulkan"))
        return metal::createWindowHost();
#endif
    return std::make_unique<vulkan::WindowHost>();
}

} // namespace jvk
