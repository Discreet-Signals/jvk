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
 File: jvk.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

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
