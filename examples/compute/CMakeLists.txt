# jvk compute (COMPUTE.md): a self-test on a stream, and a waveform a kernel
# writes in paint and a jvk::Shader draws, on whichever backend the build has.
juce_add_plugin(JvkCompute
    PRODUCT_NAME "JvkCompute"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.JvkCompute"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE JvCo
    FORMATS Standalone VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
)

# The kernels, built for this build's backends into ComputeKernels.h.
jvk_compute_shaders(JvkCompute HEADER ComputeKernels.h NAMESPACE kernels
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/Source/kernels/saxpy.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source/kernels/histogram.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source/kernels/prefix.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source/kernels/waveform.comp")

# The waveform's fragment shader: SPIR-V, and the MSL jvk's Metal backend draws
# (empty in a build without Metal).
find_program(JVK_COMPUTE_GLSLC glslc REQUIRED HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
set(_frag_src "${CMAKE_CURRENT_SOURCE_DIR}/Source/waveform.frag")
set(_frag_spv "${CMAKE_CURRENT_BINARY_DIR}/waveform.frag.spv")
set(_frag_msl "${CMAKE_CURRENT_BINARY_DIR}/waveform.frag.metal")
add_custom_command(
    OUTPUT  "${_frag_spv}"
    COMMAND "${JVK_COMPUTE_GLSLC}" -fshader-stage=fragment "${_frag_src}" -o "${_frag_spv}"
    DEPENDS "${_frag_src}"
    COMMENT "Compiling waveform.frag → SPIR-V"
    VERBATIM
)
jvk_msl("${_frag_spv}" "${_frag_msl}")
juce_add_binary_data(JvkCompute_BinaryData SOURCES "${_frag_spv}" "${_frag_msl}")

target_sources(JvkCompute PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(JvkCompute PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(JvkCompute
    PRIVATE
        JvkCompute_BinaryData
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        jvk
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

if (COMMAND jvk_bundle)
    foreach(_fmt Standalone VST3 AU)
        if (TARGET JvkCompute_${_fmt})
            jvk_bundle(JvkCompute_${_fmt})
        endif()
    endforeach()
endif()
