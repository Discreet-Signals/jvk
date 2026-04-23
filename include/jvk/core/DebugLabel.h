/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: DebugLabel.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

// =============================================================================
// jvk::debug — thin wrapper around VK_EXT_debug_utils command labels.
//
// GPU profilers (Xcode Metal capture, RenderDoc, Nsight) surface the begin/
// end/insert labels as scope names, so each render pass and draw shows up
// with a human-readable tag instead of an anonymous encoder.
//
// ---- Where to actually SEE the labels ---------------------------------------
//
//   Xcode → Debug → Capture GPU Workload. Open the capture, expand the
//   "Dependency Viewer" / "Counters" tree — every `ScopedLabel` appears as
//   a named scope wrapping the render passes it contains.
//
//   Instruments' Metal System Trace does NOT show pushDebugGroup labels.
//   That view only shows MTLCommandEncoder.label, which MoltenVK hard-codes
//   as "vkCmdBeginRenderPass RenderEncoder" regardless of what we push.
//
// ---- Build-time gating ------------------------------------------------------
//
// JVK_DEBUG_LABELS controls whether labels are emitted at all:
//   1  (default for JUCE_DEBUG)  : real calls into vkCmdBeginDebugUtilsLabelEXT
//   0  (default for non-debug)   : every entry point is an empty inline stub
//                                   that the optimizer deletes — zero runtime
//                                   overhead, no MoltenVK indirection, no
//                                   Metal debug-group bookkeeping.
//
// Override at the compile line with -DJVK_DEBUG_LABELS=1 to keep labels in
// a Release build (e.g. for profiling a shipping-config binary).
// =============================================================================

#ifndef JVK_DEBUG_LABELS
 #if JUCE_DEBUG
  #define JVK_DEBUG_LABELS 1
 #else
  #define JVK_DEBUG_LABELS 0
 #endif
#endif

namespace jvk::debug {

#if JVK_DEBUG_LABELS

// ----------------------------------------------------------------------------
// Enabled path — real vkCmd*DebugUtilsLabel calls. Function pointers live as
// C++17 inline variables (single shared instance across all TUs).
// ----------------------------------------------------------------------------

inline PFN_vkCmdBeginDebugUtilsLabelEXT  g_beginFn  = nullptr;
inline PFN_vkCmdEndDebugUtilsLabelEXT    g_endFn    = nullptr;
inline PFN_vkCmdInsertDebugUtilsLabelEXT g_insertFn = nullptr;

// Load function pointers from `instance`. Called from Device::createInstance
// right after vkCreateInstance succeeds. Idempotent; safe to re-init.
inline void init(VkInstance instance) noexcept
{
    if (instance == VK_NULL_HANDLE) {
        g_beginFn  = nullptr;
        g_endFn    = nullptr;
        g_insertFn = nullptr;
        return;
    }
    g_beginFn  = (PFN_vkCmdBeginDebugUtilsLabelEXT)
        vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT");
    g_endFn    = (PFN_vkCmdEndDebugUtilsLabelEXT)
        vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT");
    g_insertFn = (PFN_vkCmdInsertDebugUtilsLabelEXT)
        vkGetInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT");
}

inline void beginLabel(VkCommandBuffer cmd, const char* name,
                       float r = 0.7f, float g = 0.7f,
                       float b = 0.7f, float a = 1.0f) noexcept
{
    if (!g_beginFn) return;
    VkDebugUtilsLabelEXT label {};
    label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0]   = r;
    label.color[1]   = g;
    label.color[2]   = b;
    label.color[3]   = a;
    g_beginFn(cmd, &label);
}

inline void endLabel(VkCommandBuffer cmd) noexcept
{
    if (!g_endFn) return;
    g_endFn(cmd);
}

inline void insertLabel(VkCommandBuffer cmd, const char* name,
                        float r = 0.7f, float g = 0.7f,
                        float b = 0.7f, float a = 1.0f) noexcept
{
    if (!g_insertFn) return;
    VkDebugUtilsLabelEXT label {};
    label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0]   = r;
    label.color[1]   = g;
    label.color[2]   = b;
    label.color[3]   = a;
    g_insertFn(cmd, &label);
}

struct ScopedLabel
{
    VkCommandBuffer cmd;

    ScopedLabel(VkCommandBuffer c, const char* name,
                float r = 0.7f, float g = 0.7f,
                float b = 0.7f, float a = 1.0f) noexcept
        : cmd(c)
    {
        beginLabel(cmd, name, r, g, b, a);
    }

    ~ScopedLabel() noexcept { endLabel(cmd); }

    ScopedLabel(const ScopedLabel&)            = delete;
    ScopedLabel& operator=(const ScopedLabel&) = delete;
    ScopedLabel(ScopedLabel&&)                 = delete;
    ScopedLabel& operator=(ScopedLabel&&)      = delete;
};

#else

// ----------------------------------------------------------------------------
// Disabled path — every entry point is an empty `inline` body. The optimizer
// deletes the call completely, so there's no function-pointer load, no
// branch, no string argument materialisation, and no MoltenVK indirection in
// Release builds. `ScopedLabel` also leaves no destructor body, so the RAII
// sites in Renderer.cpp compile to nothing.
// ----------------------------------------------------------------------------

inline void init(VkInstance) noexcept {}

inline void beginLabel(VkCommandBuffer, const char*,
                       float = 0.0f, float = 0.0f,
                       float = 0.0f, float = 1.0f) noexcept {}

inline void endLabel(VkCommandBuffer) noexcept {}

inline void insertLabel(VkCommandBuffer, const char*,
                        float = 0.0f, float = 0.0f,
                        float = 0.0f, float = 1.0f) noexcept {}

struct ScopedLabel
{
    ScopedLabel(VkCommandBuffer, const char*,
                float = 0.0f, float = 0.0f,
                float = 0.0f, float = 1.0f) noexcept {}

    ScopedLabel(const ScopedLabel&)            = delete;
    ScopedLabel& operator=(const ScopedLabel&) = delete;
    ScopedLabel(ScopedLabel&&)                 = delete;
    ScopedLabel& operator=(ScopedLabel&&)      = delete;
};

#endif // JVK_DEBUG_LABELS

} // namespace jvk::debug
