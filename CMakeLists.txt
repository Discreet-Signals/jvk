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
target_link_libraries(jvk INTERFACE Vulkan::Vulkan)

# ==============================================================================
# Platform configuration
# ==============================================================================

if (APPLE)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_MACOS_MVK)
    target_link_libraries(jvk INTERFACE "-framework QuartzCore")
    target_link_options(jvk INTERFACE "LINKER:-rpath,@loader_path/../Frameworks")

    if (DEFINED ENV{VULKAN_SDK})
        set(_DS_VK_MVK_LIB "$ENV{VULKAN_SDK}/lib/libMoltenVK.dylib")
        set(_DS_VK_LOADER   "$ENV{VULKAN_SDK}/lib/libvulkan.1.dylib")
    endif()

    if (_DS_VK_MVK_LIB AND EXISTS "${_DS_VK_MVK_LIB}")
        target_link_libraries(jvk INTERFACE "${_DS_VK_MVK_LIB}")
    endif()

    if (_DS_VK_MVK_LIB AND EXISTS "${_DS_VK_MVK_LIB}")
        set(DS_VULKAN_MVK_LIB "${_DS_VK_MVK_LIB}" CACHE INTERNAL "Path to libMoltenVK.dylib")
        set(DS_VULKAN_LOADER  "${_DS_VK_LOADER}"   CACHE INTERNAL "Path to libvulkan.1.dylib")

        set(_DS_VK_ICD_JSON "${CMAKE_CURRENT_BINARY_DIR}/MoltenVK_icd.json")
        file(WRITE "${_DS_VK_ICD_JSON}" [=[{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "../../../Frameworks/libMoltenVK.dylib",
        "api_version": "1.2.0",
        "is_portability_driver": true
    }
}]=])
        set(DS_VULKAN_ICD_JSON "${_DS_VK_ICD_JSON}" CACHE INTERNAL "Path to bundled ICD manifest")

        function(jvk_bundle TARGET)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DS_VULKAN_MVK_LIB}"
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks/libMoltenVK.dylib"
                COMMENT "jvk: Bundling MoltenVK into ${TARGET}"
            )
            if (EXISTS "${DS_VULKAN_LOADER}")
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${DS_VULKAN_LOADER}"
                        "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks/libvulkan.1.dylib"
                )
            endif()
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Resources/vulkan/icd.d"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DS_VULKAN_ICD_JSON}"
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json"
            )
        endfunction()
    endif()

elseif (WIN32)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_WIN32_KHR)

elseif (UNIX)
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_XLIB_KHR)
    endif()
endif()

# ==============================================================================
# Examples — EXCLUDE_FROM_ALL: show up as IDE targets, don't build by default.
# Build a specific one with: cmake --build build --target VulkanCompressor_Standalone
# ==============================================================================

add_subdirectory(examples/benchmark          EXCLUDE_FROM_ALL)
add_subdirectory(examples/compressor        EXCLUDE_FROM_ALL)
add_subdirectory(examples/triangles         EXCLUDE_FROM_ALL)
add_subdirectory(examples/waveform          EXCLUDE_FROM_ALL)
add_subdirectory(examples/shader_background EXCLUDE_FROM_ALL)
