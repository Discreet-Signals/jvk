/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: jvk.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

 ID:            jvk
 vendor:        Discreet Signals
 version:       2.0
 name:          Discreet Signals Vulkan Module
 description:   Vulkan 2D rendering backend for JUCE
 website:       https://github.com/Discreet-Signals/jvk
 searchpaths:   include
 dependencies:  juce_core, juce_events, juce_data_structures, juce_graphics, juce_gui_basics, juce_gui_extra, juce_audio_processors

 END_JUCE_MODULE_DECLARATION

 *******************************************************************************/

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <span>
#include <optional>
#include <functional>
#include <type_traits>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Core — no JUCE dependency
#include "include/jvk/core/PipelineConfig.h"
#include "include/jvk/core/Memory.h"
#include "include/jvk/core/Resource.h"
#include "include/jvk/core/Device.h"
#include "include/jvk/core/FrameRetained.h"
#include "include/jvk/core/Renderer.h"
// CommandScheduler.h is included AFTER Params.h below because its fusion
// scratch buffer holds concatenated GeometryPrimitive runs. Renderer.h
// forward-declares CommandScheduler so the std::unique_ptr member there is
// fine at this point.
#include "include/jvk/core/Pipeline.h"
#include "include/jvk/core/RenderTarget.h"
#include "include/jvk/core/Cache.h"

// Shaders (embedded SPIR-V)
#include "include/jvk/graphics/UI2DShaders.h"
#include "include/jvk/graphics/ShaderRegionVert.h"

// SPIRV-Reflect (for Shader.h)
#include "include/jvk/reflect/spirv_reflect.h"

// Graphics — JUCE 2D rendering
#include "include/jvk/graphics/MSDF.h"
#include "include/jvk/graphics/GlyphAtlas.h"
#include "include/jvk/graphics/pipelines/Params.h"
// Needs GeometryPrimitive from Params.h for its fusion scratch vector.
#include "include/jvk/core/CommandScheduler.h"
#include "include/jvk/graphics/pipelines/blend/BlendPipeline.h"
#include "include/jvk/graphics/pipelines/effect/EffectPipeline.h"
#include "include/jvk/graphics/pipelines/effect/HSVPipeline.h"
#include "include/jvk/graphics/pipelines/path/PathPipeline.h"
#include "include/jvk/graphics/pipelines/fill/FillPipeline.h"
#include "include/jvk/graphics/pipelines/effect/BlurPipeline.h"
#include "include/jvk/graphics/pipelines/clip/ClipPipeline.h"
#include "include/jvk/graphics/Graphics.h"
#include "include/jvk/graphics/Shader.h"
#include "include/jvk/graphics/pipelines/shader/ShaderPipeline.h"

// Platform surface creation (must come before AudioProcessorEditor)
#if JUCE_MAC
#include "include/jvk/core/macos/NSViewGenerator.h"
#elif JUCE_WINDOWS
#include "include/jvk/core/windows/HWNDGenerator.h"
#endif

// User-facing
#include "include/jvk/graphics/AudioProcessorEditor.h"
#include "include/jvk/offscreen/ShaderImage.h"
