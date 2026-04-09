#include "jvk.h"

// Core
#include "include/jvk/core/wrapper/SwapChain.cpp"
#include "include/jvk/core/wrapper/Pipeline.cpp"
#include "include/jvk/core/VulkanInstance.cpp"
#if JUCE_MAC
#include "include/jvk/core/macos/VulkanNSViewComponent.cpp"
#elif JUCE_WINDOWS
#include "include/jvk/core/windows/VulkanHWNDComponent.cpp"
#endif

// Components
#include "include/jvk/components/VulkanComponent.cpp"

// Offscreen
#include "include/jvk/offscreen/ShaderImage.cpp"
