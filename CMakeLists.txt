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
# Backends — what this build compiles, and so what it ships. A list:
#   Metal    macOS only, and its default: nothing of Vulkan is compiled or
#            bundled (no MoltenVK, volk, VMA, nor the built-in shaders' SPIR-V).
#   Vulkan   everywhere else; on macOS through MoltenVK, which jvk_bundle copies
#            into each bundle.
# "Metal;Vulkan" compiles both: Metal runs unless JVK_BACKEND=vulkan is set at
# run time, to compare them.
# ==============================================================================

if (APPLE)
    set(JVK_BACKENDS "Metal" CACHE STRING "jvk backends to compile: Metal, Vulkan, or both")
else()
    set(JVK_BACKENDS "Vulkan" CACHE STRING "jvk backends to compile: Vulkan")
endif()
foreach(_jvk_backend ${JVK_BACKENDS})
    if (NOT _jvk_backend MATCHES "^(Metal|Vulkan)$")
        message(FATAL_ERROR "jvk: unknown backend \"${_jvk_backend}\" in JVK_BACKENDS (Metal, Vulkan)")
    endif()
endforeach()
set(JVK_METAL  FALSE)
set(JVK_VULKAN FALSE)
if ("Metal" IN_LIST JVK_BACKENDS)
    set(JVK_METAL TRUE)
endif()
if ("Vulkan" IN_LIST JVK_BACKENDS)
    set(JVK_VULKAN TRUE)
endif()
if (JVK_METAL AND NOT APPLE)
    message(FATAL_ERROR "jvk: the Metal backend is macOS only")
endif()
if (NOT JVK_METAL AND NOT JVK_VULKAN)
    message(FATAL_ERROR "jvk: JVK_BACKENDS names no backend")
endif()
message(STATUS "jvk: backends ${JVK_BACKENDS}")

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

    # Tell JUCE this module is available, and jvk.h which backends to compile
    target_compile_definitions(jvk INTERFACE JUCE_MODULE_AVAILABLE_jvk=1
        JVK_METAL=$<BOOL:${JVK_METAL}> JVK_VULKAN=$<BOOL:${JVK_VULKAN}>)

    # MSDF text rendering
    target_link_libraries(jvk INTERFACE msdfgen::msdfgen-core)
endif()

# ==============================================================================
# Core invariant: nothing under include/jvk/core names a GPU API. core/ is the
# translation layer every backend shares (TRANSLATION_LAYER.md); one Vulkan,
# Metal, D3D12 or WebGPU name there and it belongs to that backend again.
# Checked at configure time, so re-run CMake after editing core/.
# ==============================================================================

file(GLOB_RECURSE _JVK_CORE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jvk/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/jvk/core/*.cpp")
foreach(_jvk_core_file ${_JVK_CORE_FILES})
    file(STRINGS "${_jvk_core_file}" _jvk_api_hits
         REGEX "(^|[^A-Za-z0-9_])(Vk[A-Z]|vk[A-Z]|VK_|MTL[A-Z]|ID3D12|WGPU|wgpu[A-Z])")
    if (_jvk_api_hits)
        message(FATAL_ERROR "jvk: ${_jvk_core_file} names a GPU API; core/ must stay backend-neutral:\n${_jvk_api_hits}")
    endif()
endforeach()

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
# and loadVulkan in Device.cpp), so a binary always loads, Vulkan or not. Every
# build takes the SDK, Metal alone too: its glslc compiles the shaders, and its
# include folder holds the glm core uses.
target_link_libraries(jvk INTERFACE Vulkan::Headers)

# ==============================================================================
# Shader compilation — regenerate the shader bundle (core/shaders/Bundle.h:
# Vulkan SPIR-V, MSL translated by SPIRV-Cross, reflected layouts) whenever a
# shader changes. The custom command is wired into jvk's interface sources so
# every consumer depends on the header being up to date before it compiles
# jvk.mm / jvk.cpp. By hand: `cd shaders && ./compile.sh`.
# ==============================================================================

set(JVK_SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/jvk_portable.glsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ui2d.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ui2d.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/noise.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_sdf.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_sdf.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/clip.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/clip.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/clip_mask.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hsv.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shape_blur.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_blur.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_region.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile.sh"
)

set(JVK_SHADER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/jvk/core/shaders/Bundle.h")

# The tools, from the Vulkan SDK when CMake found one; compile.sh searches the
# SDK and PATH itself for any left unset.
if (Vulkan_GLSLC_EXECUTABLE)
    set(JVK_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
else()
    find_program(JVK_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
endif()
find_program(JVK_SPIRV_CROSS spirv-cross HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
find_program(JVK_SPIRV_VAL   spirv-val   HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")

set(_JVK_SHADER_ENV)
if (JVK_GLSLC)
    list(APPEND _JVK_SHADER_ENV "GLSLC=${JVK_GLSLC}")
endif()
if (JVK_SPIRV_CROSS)
    list(APPEND _JVK_SHADER_ENV "SPIRV_CROSS=${JVK_SPIRV_CROSS}")
endif()
if (JVK_SPIRV_VAL)
    list(APPEND _JVK_SHADER_ENV "SPIRV_VAL=${JVK_SPIRV_VAL}")
endif()

add_custom_command(
    OUTPUT  "${JVK_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E env ${_JVK_SHADER_ENV}
            bash "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile.sh"
    DEPENDS ${JVK_SHADER_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    COMMENT "jvk: Compiling shaders → Bundle.h"
    VERBATIM
)

# Custom target so the generated file gets built even on clean first builds,
# and an explicit `cmake --build build --target jvk_shaders` target exists.
add_custom_target(jvk_shaders DEPENDS "${JVK_SHADER_HEADER}")

# Attach the generated header as an interface source of jvk. Any consumer
# that compiles jvk.mm / jvk.cpp (which #include Bundle.h) inherits this
# as a build-graph dependency, so CMake runs the custom command first.
target_sources(jvk INTERFACE "${JVK_SHADER_HEADER}")

# ==============================================================================
# jvk_msl(<fragment.spv> <out.metal>) — a jvk::Shader's MSL, what the Metal
# backend draws it with, translated at build time from its fragment stage's
# SPIR-V by tools/jvk_msl.cpp (built for the build machine, with the vendored
# SPIRV-Cross; nothing of it ships). Pass the file's text to Shader::load /
# Graphics::shader beside the SPIR-V. A build without Metal writes it empty.
# ==============================================================================

# The vendored SPIRV-Cross, compiled once for the build-machine tools (jvk_msl,
# jvk_kernel); built only when one of them is.
if (NOT TARGET jvk_spirv_cross)
    add_library(jvk_spirv_cross STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/tools/spirv_cross.cpp")
    target_compile_features(jvk_spirv_cross PUBLIC cxx_std_17)
    target_include_directories(jvk_spirv_cross PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tools")
endif()

if (JVK_METAL AND NOT TARGET jvk_msl)
    add_executable(jvk_msl EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/tools/jvk_msl.cpp")
    target_link_libraries(jvk_msl PRIVATE jvk_spirv_cross)
endif()

function(jvk_msl SPIRV OUTPUT)
    if (TARGET jvk_msl)
        add_custom_command(OUTPUT "${OUTPUT}"
            COMMAND jvk_msl "${SPIRV}" "${OUTPUT}"
            DEPENDS "${SPIRV}" jvk_msl
            COMMENT "jvk: ${SPIRV} → MSL"
            VERBATIM)
    else()
        add_custom_command(OUTPUT "${OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
            DEPENDS "${SPIRV}"
            VERBATIM)
    endif()
endfunction()

# ==============================================================================
# jvk_compute_shaders — a project's compute kernels (GLSL, COMPUTE.md), built
# for the backends the build has, in one header: each kernel's SPIR-V, its MSL
# and its interface, reflected at build time by tools/jvk_kernel.cpp. It
# rebuilds whenever a kernel or anything it includes changes. jvk's GLSL
# library is on the include path: #include "jvk/compute.glsl", "jvk/scan.glsl".
#
#   jvk_compute_shaders(<target>
#       HEADER     <MyKernels.h>          # the generated header's name
#       NAMESPACE  <my::kernels>          # each kernel a jvk::compute::KernelSource named after its file
#       SOURCES    <a.comp> ...           # a.comp -> my::kernels::a
#       [INCLUDE_DIRS <dir> ...]          # for the kernels' own #includes
#       [TARGET_ENV vulkan1.0|vulkan1.1|...])   # vulkan1.0 by default: loads on any Vulkan
#
# The header goes in the build tree, on <target>'s include path, and is one of
# its sources (INTERFACE for an interface library), so it is made first.
# ==============================================================================

if (NOT TARGET jvk_kernel)
    add_executable(jvk_kernel EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/tools/jvk_kernel.cpp")
    target_link_libraries(jvk_kernel PRIVATE jvk_spirv_cross)
endif()

function(jvk_compute_shaders TARGET)
    cmake_parse_arguments(ARG "" "HEADER;NAMESPACE;TARGET_ENV" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if (NOT ARG_HEADER OR NOT ARG_NAMESPACE OR NOT ARG_SOURCES)
        message(FATAL_ERROR "jvk_compute_shaders(${TARGET}): HEADER, NAMESPACE and SOURCES are required")
    endif()
    # glslc from the Vulkan SDK (find_package(Vulkan)'s, a cache variable, so
    # seen from any directory), else wherever it is.
    set(_glslc "${Vulkan_GLSLC_EXECUTABLE}")
    if (NOT _glslc)
        find_program(JVK_KERNEL_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
        set(_glslc "${JVK_KERNEL_GLSLC}")
    endif()
    if (NOT _glslc)
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

    set(_kernels)
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
            COMMAND "${_glslc}" -fshader-stage=compute --target-env=${ARG_TARGET_ENV} -O
                    ${_includes} -MD -MF "${_spv}.d" -o "${_spv}" "${_source}"
            DEPENDS "${_source}"
            ${_depfile}
            COMMENT "jvk: compiling ${_name} (compute)"
            VERBATIM
        )
        list(APPEND _kernels "${_name}=${_spv}")
        list(APPEND _spvs "${_spv}")
    endforeach()

    set(_header "${_out}/${ARG_HEADER}")
    add_custom_command(
        OUTPUT  "${_header}"
        COMMAND jvk_kernel "${_header}" "${ARG_NAMESPACE}" ${_kernels}
        DEPENDS ${_spvs} jvk_kernel
        COMMENT "jvk: reflecting and translating ${ARG_HEADER}"
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
    target_link_libraries(jvk INTERFACE "-framework QuartzCore" "-framework Metal" "-framework CoreGraphics")
endif()

if (APPLE AND JVK_VULKAN)
    target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_MACOS_MVK)

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

elseif (UNIX AND NOT APPLE)
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(jvk INTERFACE VK_USE_PLATFORM_XLIB_KHR)
    endif()
endif()

# ==============================================================================
# jvk_bundle — per plugin-format target runtime setup. Call from the consuming
# project after juce_add_plugin (or any target that produces a bundle/exe).
#   macOS, Vulkan: copy libMoltenVK into the bundle; Debug adds the loader + ICD
#            manifest.
#   macOS, Metal alone: nothing to copy; takes out what a Vulkan build left.
#   Windows, Linux: no-op. jvk loads the system loader the GPU driver installs;
#            we ship none (Khronos strongly discourages a private copy).
# ==============================================================================
function(jvk_bundle TARGET)
    if (APPLE AND NOT "Vulkan" IN_LIST JVK_BACKENDS)
        # A bundle built before with Vulkan keeps its libraries until removed,
        # and Frameworks with them when nothing else is in it.
        set(_contents "$<TARGET_BUNDLE_DIR:${TARGET}>/Contents")
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${_contents}/Frameworks/libMoltenVK.dylib"
                "${_contents}/Frameworks/libvulkan.1.dylib" "${_contents}/Resources/vulkan"
            COMMAND find "${_contents}" -maxdepth 1 -name Frameworks -type d -empty -delete
            COMMENT "jvk: ${TARGET} draws with Metal, no Vulkan libraries"
            VERBATIM
        )
    elseif (APPLE)
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
# tool: no bundle). macOS with the Vulkan backend: the shipped MoltenVK copied
# beside the executable, where Device.cpp's loadVulkan looks after a bundle's
# Frameworks. Metal alone, Windows, Linux: no-op (nothing to ship; the system
# loader, as for plugins).
# ==============================================================================
function(jvk_runtime TARGET)
    if (APPLE AND DS_VULKAN_MVK_LIB AND "Vulkan" IN_LIST JVK_BACKENDS)
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
add_subdirectory(examples/compute           EXCLUDE_FROM_ALL)
add_subdirectory(examples/shader_background EXCLUDE_FROM_ALL)
