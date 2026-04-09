if (NOT COMMAND juce_add_module)
    message(FATAL_ERROR "JUCE must be added to your project before jvk!")
endif()

juce_add_module("${CMAKE_CURRENT_SOURCE_DIR}")

# ==============================================================================
# Vulkan SDK discovery
# ==============================================================================
# On macOS, setup-env.sh sets VULKAN_SDK to ~/VulkanSDK/<version>/macOS.
# IDEs often don't inherit shell env vars, so auto-detect when missing.
# On Windows/Linux the SDK installer registers itself system-wide and
# find_package(Vulkan) works without VULKAN_SDK being set.

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
    # --- macOS / MoltenVK ---
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_MACOS_MVK)
    target_link_libraries(jvk INTERFACE "-framework QuartzCore")
    target_link_options(jvk INTERFACE "LINKER:-rpath,@loader_path/../Frameworks")

    # Locate MoltenVK from the SDK
    if (DEFINED ENV{VULKAN_SDK})
        set(_DS_VK_MVK_LIB "$ENV{VULKAN_SDK}/lib/libMoltenVK.dylib")
        set(_DS_VK_LOADER   "$ENV{VULKAN_SDK}/lib/libvulkan.1.dylib")
    endif()

    # Link MoltenVK
    if (_DS_VK_MVK_LIB AND EXISTS "${_DS_VK_MVK_LIB}")
        target_link_libraries(jvk INTERFACE "${_DS_VK_MVK_LIB}")
    endif()

    # ------------------------------------------------------------------
    # jvk_bundle(TARGET)
    #
    # Call this on each JUCE format target (e.g. MyPlugin_Standalone) to
    # copy the Vulkan loader + MoltenVK + ICD manifest into the .app/.vst3
    # bundle so the plugin works without VULKAN_SDK at runtime.
    #
    # Example:
    #   foreach(fmt Standalone VST3 AU)
    #       if (TARGET MyPlugin_${fmt})
    #           jvk_bundle(MyPlugin_${fmt})
    #       endif()
    #   endforeach()
    # ------------------------------------------------------------------
    if (_DS_VK_MVK_LIB AND EXISTS "${_DS_VK_MVK_LIB}")
        set(DS_VULKAN_MVK_LIB "${_DS_VK_MVK_LIB}" CACHE INTERNAL "Path to libMoltenVK.dylib")
        set(DS_VULKAN_LOADER  "${_DS_VK_LOADER}"   CACHE INTERNAL "Path to libvulkan.1.dylib")

        # Write the ICD manifest once at configure time
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
            # Copy MoltenVK into Frameworks/
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DS_VULKAN_MVK_LIB}"
                    "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks/libMoltenVK.dylib"
                COMMENT "jvk: Bundling MoltenVK into ${TARGET}"
            )
            # Copy Vulkan loader into Frameworks/
            if (EXISTS "${DS_VULKAN_LOADER}")
                add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${DS_VULKAN_LOADER}"
                        "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents/Frameworks/libvulkan.1.dylib"
                )
            endif()
            # Install ICD manifest so the loader finds MoltenVK without env vars
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
    # --- Windows ---
    # The Vulkan SDK installs a system-wide loader (vulkan-1.dll) and GPU
    # vendors ship ICD drivers, so no bundling is needed.
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_WIN32_KHR)

elseif (UNIX)
    # --- Linux (X11 / Wayland) ---
    # Vulkan loader and Mesa/proprietary ICDs are system packages.
    # Surface extension is selected at compile time.
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_XLIB_KHR)
    endif()
endif()
