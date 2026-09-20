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

# Windows cross-compile: an ARM64-host Vulkan SDK ships x64 binaries in
# Lib-x64/Bin-x64 alongside the native ARM64 Lib/Bin. FindVulkan picks Lib/
# for any 64-bit target, which would hand the x64 linker an ARM64 .lib and
# produce unresolved-external errors. Steer to the x64 paths explicitly.
if (WIN32 AND DEFINED ENV{VULKAN_SDK}
    AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64"
    AND EXISTS "$ENV{VULKAN_SDK}/Lib-x64/vulkan-1.lib")

    set(Vulkan_LIBRARY "$ENV{VULKAN_SDK}/Lib-x64/vulkan-1.lib"
        CACHE FILEPATH "x64 Vulkan loader for cross-compile" FORCE)

    if (EXISTS "$ENV{VULKAN_SDK}/Bin-x64/glslc.exe")
        set(Vulkan_GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/Bin-x64/glslc.exe"
            CACHE FILEPATH "x64 glslc for cross-compile" FORCE)
    endif()
    if (EXISTS "$ENV{VULKAN_SDK}/Bin-x64/glslangValidator.exe")
        set(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE
            "$ENV{VULKAN_SDK}/Bin-x64/glslangValidator.exe"
            CACHE FILEPATH "x64 glslangValidator for cross-compile" FORCE)
    endif()

    message(STATUS "jvk: Cross-compile x64 on ARM64 host → using Lib-x64/Bin-x64")
endif()

find_package(Vulkan REQUIRED)
if (NOT APPLE)
    target_link_libraries(jvk INTERFACE Vulkan::Vulkan)
endif()

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
# Platform configuration
# ==============================================================================

if (APPLE)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_MACOS_MVK)
    target_link_libraries(jvk INTERFACE "-framework QuartzCore")
    target_link_options(jvk INTERFACE "LINKER:-rpath,@loader_path/../Frameworks")

    # The SHIPPED driver is pinned here, never read from whichever SDK is installed:
    # MoltenVK 1.4.0 is the last build that runs on macOS 10.15 (1.4.1 raised the
    # floor to 11.0 and deleted the runtime guards, so it cannot be rebuilt lower).
    # Release binds straight to it — MoltenVK's documented no-loader mode: one dylib,
    # no ICD manifest, no system ICD/layer can intervene. Debug binds to the SDK
    # loader so the validation layers can dispatch; jvk_bundle adds the loader and
    # manifest for that config only.
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

    target_link_libraries(jvk INTERFACE Vulkan::Headers
        "$<IF:$<CONFIG:Debug>,Vulkan::Vulkan,${DS_VULKAN_MVK_LIB}>")

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

    # The Vulkan LOADER (vulkan-1.dll) is not part of Windows: GPU drivers put
    # it in System32. A machine with no Vulkan-capable driver (Microsoft Basic
    # Display Adapter, many VMs, remote-desktop hosts) has no copy, and a
    # plugin that imports it the ordinary way fails to load at all — the DAW
    # reports a broken plugin before any of our code runs. So the plugin ships
    # LunarG's redistributable runtime (the same loader the SDK installer puts
    # in System32; Apache-2.0 / MIT, licence copied beside it) and the import
    # is delay-loaded: jvk::Device resolves the system copy first and this one
    # second, and if neither exists no vk* call is ever made and the editor
    # falls back to the JUCE renderer. With a loader but no driver ICD,
    # vkCreateInstance reports VK_ERROR_INCOMPATIBLE_DRIVER and the same
    # fallback applies — a driver cannot be shipped, only the loader can.
    # Pinned like MoltenVK below; the hash is the one LunarG publishes for the
    # file (vulkan.lunarg.com/sdk/files.json).
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        set(_DS_VK_RT_ARCH "x64")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(X86|x86|i[3-6]86)$")
        set(_DS_VK_RT_ARCH "x86")
    endif()
    if (_DS_VK_RT_ARCH)
        FetchContent_Declare(
            vulkanrt
            URL      https://sdk.lunarg.com/sdk/download/1.4.357.0/windows/VulkanRT-X64-1.4.357.0-Components.zip
            URL_HASH SHA256=a14672efed15aafc7f5a16572d35cd3a3416eadf670aeee3cdf50ee32d5fbf83
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(vulkanrt)
        set(DS_VULKAN_WIN_LOADER  "${vulkanrt_SOURCE_DIR}/${_DS_VK_RT_ARCH}/vulkan-1.dll"
            CACHE INTERNAL "Shipped Vulkan loader")
        set(DS_VULKAN_WIN_LICENSE "${vulkanrt_SOURCE_DIR}/VulkanRT-License.txt"
            CACHE INTERNAL "Licence of the shipped Vulkan loader")
    else()
        message(STATUS "jvk: no redistributable Vulkan loader for ${CMAKE_SYSTEM_PROCESSOR}; the system runtime alone will be used")
        set(DS_VULKAN_WIN_LOADER "" CACHE INTERNAL "Shipped Vulkan loader")
    endif()

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
#   Windows: explicit PRIVATE link to Vulkan::Vulkan (JUCE's per-format
#            _SharedCode static libs don't reliably propagate the INTERFACE
#            Vulkan dep through to the final .exe / .vst3 under MSVC), the
#            import DELAY-LOADED so the binary loads on a machine with no
#            vulkan-1.dll, and LunarG's loader copied beside the binary (see
#            the Windows block above and ensureVulkanLoader in Device.cpp).
#   Linux:   no-op (system loader handles Vulkan at runtime).
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
    elseif (WIN32 AND TARGET Vulkan::Vulkan)
        target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
        if (MSVC)
            # Resolved on the first vk* call, which jvk::Device makes only
            # after it has loaded a loader itself.
            target_link_libraries(${TARGET} PRIVATE delayimp)
            target_link_options(${TARGET} PRIVATE "/DELAYLOAD:vulkan-1.dll")
        endif()
        if (DS_VULKAN_WIN_LOADER)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DS_VULKAN_WIN_LOADER}"  "$<TARGET_FILE_DIR:${TARGET}>/vulkan-1.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DS_VULKAN_WIN_LICENSE}" "$<TARGET_FILE_DIR:${TARGET}>/vulkan-1-LICENSE.txt"
                COMMENT "jvk: Bundling the Vulkan loader beside ${TARGET}"
            )
        endif()
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
