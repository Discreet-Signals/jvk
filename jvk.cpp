/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: jvk.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

// Compile the vendored Vulkan Memory Allocator and volk implementations into
// this (the module's single) TU. Must precede jvk.h, which includes both headers.
#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION
#include "jvk.h"

// Core
#include "include/jvk/core/Context.cpp"
#include "include/jvk/core/Renderer.cpp"
#include "include/jvk/core/Shader.cpp"

// Hosts
#include "include/jvk/hosts/WindowHost.cpp"
#include "include/jvk/hosts/ShaderImage.cpp"

// Vulkan backend
#include "include/jvk/backends/vulkan/Device.cpp"
#include "include/jvk/backends/vulkan/VulkanBackend.cpp"
#include "include/jvk/backends/vulkan/VulkanSurface.cpp"
#include "include/jvk/backends/vulkan/VulkanHost.cpp"

// SPIRV-Reflect (C source)
extern "C" {
#include "include/jvk/reflect/spirv_reflect.c"
}
