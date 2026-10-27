cmake_minimum_required(VERSION 3.20)
project(jvk_project VERSION 1.0)

# ==============================================================================
# JUCE
# ==============================================================================

if (NOT COMMAND juce_add_module)
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.12
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

# ==============================================================================
# msdfgen — MSDF text rendering (core only, no FreeType dependency)
# ==============================================================================

FetchContent_Declare(
    msdfgen
    GIT_REPOSITORY https://github.com/Chlumsky/msdfgen.git
    GIT_TAG        v1.12
    GIT_SHALLOW    TRUE
)
set(MSDFGEN_CORE_ONLY ON CACHE BOOL "" FORCE)
set(MSDFGEN_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(MSDFGEN_USE_SKIA OFF CACHE BOOL "" FORCE)
set(MSDFGEN_USE_VCPKG OFF CACHE BOOL "" FORCE)
set(MSDFGEN_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(msdfgen)

# ==============================================================================
# jvk module — manual INTERFACE target
#
# We cannot use juce_add_module() because it does a GLOB_RECURSE from the
# module directory. Since the module lives at the repo root alongside
# examples/, shaders/, and build/, the glob picks up non-module files and
# triggers infinite reconfigure loops.
#
# This creates the same INTERFACE target that juce_add_module would.
# Consumers link with:  target_link_libraries(MyPlugin PRIVATE jvk)
# ==============================================================================

if (NOT TARGET jvk)
    add_library(jvk INTERFACE)

    # Parent dir as include root so #include <jvk/jvk.h> resolves
    target_include_directories(jvk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/..")

    # Module sources — compiled by each consuming target (standard JUCE module pattern).
    # On macOS the .mm includes the .cpp; on other platforms compile .cpp directly.
    if (APPLE)
        target_sources(jvk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/jvk.mm")
    else()
        target_sources(jvk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/jvk.cpp")
    endif()

    # C++20 required for std::span, std::optional
    target_compile_features(jvk INTERFACE cxx_std_20)

    # Tell JUCE this module is available
    target_compile_definitions(jvk INTERFACE JUCE_MODULE_AVAILABLE_jvk=1)

    # MSDF text rendering
    target_link_libraries(jvk INTERFACE msdfgen::msdfgen-core)
endif()

# ==============================================================================
# Vulkan SDK discovery
# ==============================================================================

if (APPLE AND NOT DEFINED ENV{VULKAN_SDK})
    file(GLOB _DS_VK_SDK_CANDIDATES "$ENV{HOME}/VulkanSDK/*/macOS")
    if (_DS_VK_SDK_CANDIDATES)
        list(SORT _DS_VK_SDK_CANDIDATES ORDER DESCENDING)
        list(GET _DS_VK_SDK_CANDIDATES 0 _DS_VK_SDK_BEST)
        set(ENV{VULKAN_SDK} "${_DS_VK_SDK_BEST}")
        message(STATUS "jvk: Auto-detected VULKAN_SDK=$ENV{VULKAN_SDK}")
    endif()
endif()

find_package(Vulkan REQUIRED)
# Headers only: jvk LOADS Vulkan at run time and never links it (volk, see jvk.h
# and loadVulkan in Device.cpp), so a binary always loads, Vulkan or not.
target_link_libraries(jvk INTERFACE Vulkan::Headers)

# ==============================================================================
# Shader compilation — auto-regenerate UI2DShaders.h whenever any .vert/.frag
# changes. The custom command is wired into jvk's interface sources so every
# consumer implicitly depends on the header being up-to-date before their
# compile of jvk.mm / jvk.cpp (which #include UI2DShaders.h). No manual
# `cd shaders && ./compile.sh` required.
# ==============================================================================

set(JVK_SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ui2d.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ui2d.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/clip.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/clip.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shape_blur.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hsv.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_sdf.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_sdf.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_region.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile.sh"
)

set(JVK_SHADER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/jvk/graphics/UI2DShaders.h")

# Pass the CMake-discovered glslc to compile.sh so it doesn't depend on PATH
if (Vulkan_GLSLC_EXECUTABLE)
    set(_JVK_GLSLC_ENV "GLSLC=${Vulkan_GLSLC_EXECUTABLE}")
else()
    set(_JVK_GLSLC_ENV)
endif()

add_custom_command(
    OUTPUT  "${JVK_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E env ${_JVK_GLSLC_ENV}
            bash "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile.sh"
    DEPENDS ${JVK_SHADER_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    COMMENT "jvk: Compiling shaders → UI2DShaders.h"
    VERBATIM
)

# Custom target so the generated file gets built even on clean first builds,
# and an explicit `cmake --build build --target jvk_shaders` target exists.
add_custom_target(jvk_shaders DEPENDS "${JVK_SHADER_HEADER}")

# Attach the generated header as an interface source of jvk. Any consumer
# that compiles jvk.mm / jvk.cpp (which #include UI2DShaders.h) inherits this
# as a build-graph dependency, so CMake runs the custom command first.
target_sources(jvk INTERFACE "${JVK_SHADER_HEADER}")

# ==============================================================================
# jvk_compute_shaders — a consumer's compute kernels (GLSL), compiled to SPIR-V
# and embedded in one header, rebuilt whenever a kernel or anything it
# includes changes. jvk's GLSL library is on the include path:
# #include "jvk/scan.glsl", "jvk/compute.glsl".
#
#   jvk_compute_shaders(<target>
#       HEADER     <MyKernels.h>          # the generated header's name
#       NAMESPACE  <my::kernels>          # each kernel a std::span<const uint32_t> named after its file
#       SOURCES    <a.comp> ...           # a.comp -> my::kernels::a
#       [INCLUDE_DIRS <dir> ...]          # for the kernels' own #includes
#       [TARGET_ENV vulkan1.0|vulkan1.1|...])   # vulkan1.0 by default: loads on any Vulkan
#
# The header goes in the build tree, on <target>'s include path, and is one of
# its sources (INTERFACE for an interface library), so it is made first.
# ==============================================================================
function(jvk_compute_shaders TARGET)
    cmake_parse_arguments(ARG "" "HEADER;NAMESPACE;TARGET_ENV" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if (NOT ARG_HEADER OR NOT ARG_NAMESPACE OR NOT ARG_SOURCES)
        message(FATAL_ERROR "jvk_compute_shaders(${TARGET}): HEADER, NAMESPACE and SOURCES are required")
    endif()
    if (NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "jvk_compute_shaders(${TARGET}): glslc not found (it comes with the Vulkan SDK)")
    endif()
    if (NOT ARG_TARGET_ENV)
        set(ARG_TARGET_ENV vulkan1.0)
    endif()

    set(_out "${CMAKE_CURRENT_BINARY_DIR}/jvk_compute/${TARGET}")
    file(MAKE_DIRECTORY "${_out}")
    set(_includes "-I${CMAKE_CURRENT_FUNCTION_LIST_DIR}/shaders/include")
    foreach(_dir ${ARG_INCLUDE_DIRS})
        list(APPEND _includes "-I${_dir}")
    endforeach()

    # glslc's dependency files rebuild a kernel when something it includes changes. Ninja and
    # Makefiles take them from CMake 3.20, Xcode and Visual Studio from 3.21; before that, only
    # the kernel's own source is watched.
    set(_depfiles OFF)
    if (CMAKE_GENERATOR MATCHES "Ninja|Makefiles" OR CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        set(_depfiles ON)
    endif()

    set(_names)
    set(_spvs)
    foreach(_source ${ARG_SOURCES})
        get_filename_component(_source "${_source}" ABSOLUTE)
        get_filename_component(_name "${_source}" NAME_WE)
        string(MAKE_C_IDENTIFIER "${_name}" _name)
        set(_spv "${_out}/${_name}.spv")
        set(_depfile)
        if (_depfiles)
            set(_depfile DEPFILE "${_spv}.d")
        endif()
        add_custom_command(
            OUTPUT  "${_spv}"
            COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=${ARG_TARGET_ENV} -O
                    ${_includes} -MD -MF "${_spv}.d" -o "${_spv}" "${_source}"
            DEPENDS "${_source}"
            ${_depfile}
            COMMENT "jvk: compiling ${_name} (compute)"
            VERBATIM
        )
        list(APPEND _names "${_name}")
        list(APPEND _spvs "${_spv}")
    endforeach()

    string(REPLACE ";" "|" _namesArg "${_names}")
    string(REPLACE ";" "|" _spvsArg "${_spvs}")
    set(_header "${_out}/${ARG_HEADER}")
    add_custom_command(
        OUTPUT  "${_header}"
        COMMAND ${CMAKE_COMMAND} -DOUTPUT=${_header} -DNAMESPACE=${ARG_NAMESPACE}
                -DNAMES=${_namesArg} -DFILES=${_spvsArg}
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/jvk_embed_spirv.cmake"
        DEPENDS ${_spvs} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cmake/jvk_embed_spirv.cmake"
        COMMENT "jvk: embedding ${ARG_HEADER}"
        VERBATIM
    )
    add_custom_target(${TARGET}_jvk_compute DEPENDS "${_header}")

    get_target_property(_type ${TARGET} TYPE)
    if (_type STREQUAL "INTERFACE_LIBRARY")
        target_sources(${TARGET} INTERFACE "${_header}")
        target_include_directories(${TARGET} INTERFACE "${_out}")
    else()
        target_sources(${TARGET} PRIVATE "${_header}")
        target_include_directories(${TARGET} PRIVATE "${_out}")
    endif()
endfunction()

# ==============================================================================
# Platform configuration
# ==============================================================================

if (APPLE)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_MACOS_MVK)
    target_link_libraries(jvk INTERFACE "-framework QuartzCore")

    # The SHIPPED driver is pinned here, never read from whichever SDK is installed:
    # MoltenVK 1.4.0 is the last build that runs on macOS 10.15 (1.4.1 raised the
    # floor to 11.0 and deleted the runtime guards, so it cannot be rebuilt lower).
    # Release loads it straight from the bundle (Device.cpp loadVulkan) — MoltenVK's
    # documented no-loader mode: one dylib, no ICD manifest, no system ICD/layer can
    # intervene. Debug loads the SDK loader so the validation layers can dispatch;
    # jvk_bundle adds the loader and manifest for that config only.
    FetchContent_Declare(
        moltenvk
        URL      https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.0/MoltenVK-macos.tar
        URL_HASH SHA256=f4feaf6a4988352de8e6d49874ccc1cd6a45021e3cb476e8531bef7ecc73e93a
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(moltenvk)
    set(DS_VULKAN_MVK_LIB "${moltenvk_SOURCE_DIR}/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib"
        CACHE INTERNAL "Shipped MoltenVK dylib")
    get_filename_component(_DS_VK_LIB_DIR "${Vulkan_LIBRARY}" DIRECTORY)
    set(DS_VULKAN_LOADER "${_DS_VK_LIB_DIR}/libvulkan.1.dylib" CACHE INTERNAL "SDK loader (Debug only)")

    set(DS_VULKAN_ICD_JSON "${CMAKE_CURRENT_BINARY_DIR}/MoltenVK_icd.json" CACHE INTERNAL "ICD manifest (Debug only)")
    file(WRITE "${DS_VULKAN_ICD_JSON}" [=[{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "../../../Frameworks/libMoltenVK.dylib",
        "api_version": "1.2.0",
        "is_portability_driver": true
    }
}]=])

elseif (WIN32)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_WIN32_KHR)

elseif (UNIX)
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_XLIB_KHR)
    endif()
endif()

# ==============================================================================
# jvk_bundle — per plugin-format target runtime setup. Call from the consuming
# project after juce_add_plugin (or any target that produces a bundle/exe).
#   macOS:   copy libMoltenVK into the bundle; Debug adds the loader + ICD manifest.
#   Windows, Linux: no-op. jvk loads the system loader the GPU driver installs;
#            we ship none (Khronos strongly discourages a private copy).
# ==============================================================================
function(jvk_bundle TARGET)
    if (APPLE AND DS_VULKAN_MVK_LIB)
        set(_fw  "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks")
        set(_icd "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Resources/vulkan/icd.d")
        # Outside Debug, `-E true` swallows the loader/manifest steps; the rm first
        # keeps an incremental Release bundle from carrying a stale loader.
        set(_dbg_mkdir "$<IF:$<CONFIG:Debug>,make_directory,true>")
        set(_dbg_copy  "$<IF:$<CONFIG:Debug>,copy_if_different,true>")
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_fw}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DS_VULKAN_MVK_LIB}" "${_fw}/libMoltenVK.dylib"
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${_fw}/libvulkan.1.dylib"
                "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Resources/vulkan"
            COMMAND ${CMAKE_COMMAND} -E ${_dbg_copy} "${DS_VULKAN_LOADER}" "${_fw}/libvulkan.1.dylib"
            COMMAND ${CMAKE_COMMAND} -E ${_dbg_mkdir} "${_icd}"
            COMMAND ${CMAKE_COMMAND} -E ${_dbg_copy} "${DS_VULKAN_ICD_JSON}" "${_icd}/MoltenVK_icd.json"
            COMMENT "jvk: Bundling MoltenVK into ${TARGET}"
        )
    endif()
endfunction()

# ==============================================================================
# jvk_runtime — Vulkan for a plain executable (a test runner, a command-line
# tool: no bundle). macOS: the shipped MoltenVK copied beside the executable,
# where Device.cpp's loadVulkan looks after a bundle's Frameworks. Windows,
# Linux: no-op (the system loader, as for plugins).
# ==============================================================================
function(jvk_runtime TARGET)
    if (APPLE AND DS_VULKAN_MVK_LIB)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DS_VULKAN_MVK_LIB}" "$<TARGET_FILE_DIR:${TARGET}>/libMoltenVK.dylib"
            COMMENT "jvk: MoltenVK beside ${TARGET}"
        )
    endif()
endfunction()

# ==============================================================================
# Examples — EXCLUDE_FROM_ALL: show up as IDE targets, don't build by default.
# Build a specific one with: cmake --build build --target VulkanCompressor_Standalone
# ==============================================================================

add_subdirectory(examples/benchmark          EXCLUDE_FROM_ALL)
add_subdirectory(examples/compliance        EXCLUDE_FROM_ALL)
add_subdirectory(examples/compressor        EXCLUDE_FROM_ALL)
add_subdirectory(examples/shader_background EXCLUDE_FROM_ALL)
