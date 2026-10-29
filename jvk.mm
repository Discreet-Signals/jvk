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
 File: jvk.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include "jvk.cpp"

#if JUCE_MAC
#include "include/jvk/backends/vulkan/macos/NSViewGenerator.mm"
#include "include/jvk/backends/vulkan/macos/MetalCapabilities.mm"

// Metal backend (the macOS default: hosts/WindowHost.cpp). Its SPIR-V → MSL
// translator is the separate jvk_spirv_cross.cpp.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "include/jvk/backends/metal/SpirvToMsl.h"
#include "include/jvk/backends/metal/MetalBackend.h"
#include "include/jvk/backends/metal/MetalBackend.mm"
#include "include/jvk/backends/metal/MetalSurface.mm"
#include "include/jvk/backends/metal/MetalHost.mm"
#endif
