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
 File: jvk.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

/*******************************************************************************
 BEGIN_JUCE_MODULE_DECLARATION

 ID:            jvk
 vendor:        Discreet Signals
 version:       1.0
 name:          Discreet Signals Vulkan Module
 description:   Vulkan integration for JUCE
 website:       https://github.com/Discreet-Signals/jvk
 searchpaths:   include
 dependencies:  juce_core, juce_events, juce_data_structures, juce_graphics, juce_gui_basics, juce_gui_extra, juce_audio_processors

 END_JUCE_MODULE_DECLARATION

 *******************************************************************************/

#pragma once

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
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

// Core
#include "include/jvk/core/PipelineConfig.h"
#include "include/jvk/core/wrapper/SwapChain.h"
#include "include/jvk/core/GPUMemoryPool.h"
#include "include/jvk/core/Buffer.h"
#include "include/jvk/core/DescriptorHelper.h"
#include "include/jvk/core/Image.h"
#include "include/jvk/core/StagingBelt.h"
#include "include/jvk/core/DeletionQueue.h"
#include "include/jvk/core/VertexRingBuffer.h"
#include "include/jvk/core/wrapper/Shader.h"
#include "include/jvk/core/wrapper/Pipeline.h"
#include "include/jvk/core/wrapper/Wrapper.h"
#include "include/jvk/core/DefaultShaders.h"
#include "include/jvk/core/VulkanInstance.h"
#if JUCE_MAC
#include "include/jvk/core/macos/NSViewGenerator.h"
#include "include/jvk/core/macos/VulkanNSViewComponent.h"
#elif JUCE_WINDOWS
#include "include/jvk/core/windows/HWNDGenerator.h"
#include "include/jvk/core/windows/VulkanHWNDComponent.h"
#else
#error "jvk: Unsupported Platform!"
#endif

// ECS
#include "include/jvk/ecs/ECS.h"
#include "include/jvk/ecs/ComponentPool.h"
#include "include/jvk/ecs/components/Transform.h"
#include "include/jvk/ecs/entities/Entity.h"
#include "include/jvk/ecs/components/Mesh.h"
#include "include/jvk/ecs/components/Material.h"
#include "include/jvk/ecs/components/Texture.h"
#include "include/jvk/ecs/components/MeshRenderer.h"
#include "include/jvk/ecs/components/Animator.h"
#include "include/jvk/ecs/components/Smoother.h"
#include "include/jvk/ecs/entities/Object.h"
#include "include/jvk/ecs/entities/Camera.h"
#include "include/jvk/ecs/entities/ObjectImpl.h"
#include "include/jvk/ecs/Renderer.h"
#include "include/jvk/ecs/Scene.h"

// Components
#include "include/jvk/FIFO.h"
#include "include/jvk/components/VulkanComponent.h"
#include "include/jvk/components/ShaderComponent.h"

// Scene integration (after VulkanComponent and Scene)
#include "include/jvk/ecs/ForwardRenderer.h"
#include "include/jvk/components/SceneComponent.h"

// Graphics (Vulkan-backed juce::Graphics)
#include "include/jvk/graphics/UI2DShaders.h"
#include "include/jvk/graphics/GlyphAtlas.h"
#include "include/jvk/graphics/VulkanGraphicsContext.h"
#include "include/jvk/graphics/Graphics.h"
#include "include/jvk/graphics/AudioProcessorEditor.h"

// Offscreen rendering
#include "include/jvk/offscreen/FullscreenShaders.h"
#include "include/jvk/offscreen/ShaderImage.h"
