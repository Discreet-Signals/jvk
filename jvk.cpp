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

#include "jvk.h"

// Core implementations
#include "include/jvk/core/Device.cpp"
#include "include/jvk/core/Renderer.cpp"
#include "include/jvk/core/CommandScheduler.cpp"
#include "include/jvk/core/Pipeline.cpp"
#include "include/jvk/core/RenderTarget.cpp"
#include "include/jvk/offscreen/ShaderImage.cpp"

// SPIRV-Reflect (C source)
extern "C" {
#include "include/jvk/reflect/spirv_reflect.c"
}
