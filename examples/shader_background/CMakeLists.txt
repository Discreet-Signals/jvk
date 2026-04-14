juce_add_plugin(ShaderBackground
    PRODUCT_NAME "ShaderBackground"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.ShaderBackground"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE ShBg
    FORMATS Standalone VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "ShaderBackground needs microphone access for audio input."
    HARDENED_RUNTIME_ENABLED TRUE
    HARDENED_RUNTIME_OPTIONS "com.apple.security.device.audio-input"
)

# --- Compile GLSL → SPIR-V automatically ---
find_program(GLSLC glslc REQUIRED)

set(_frag_src "${CMAKE_CURRENT_SOURCE_DIR}/Source/background.frag")
set(_frag_spv "${CMAKE_CURRENT_SOURCE_DIR}/Source/background.frag.spv")

add_custom_command(
    OUTPUT  "${_frag_spv}"
    COMMAND "${GLSLC}" -fshader-stage=fragment "${_frag_src}" -o "${_frag_spv}"
    DEPENDS "${_frag_src}"
    COMMENT "Compiling background.frag → SPIR-V"
)

juce_add_binary_data(ShaderBackground_BinaryData SOURCES
    "${_frag_spv}"
)

target_sources(ShaderBackground PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(ShaderBackground PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(ShaderBackground
    PRIVATE
        ShaderBackground_BinaryData
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
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

# --- Post-build: fix moduleinfo.json + hardened runtime signing ---
# Must run after ALL JUCE post-build steps (which re-sign the bundle).
if (APPLE AND TARGET ShaderBackground_VST3)
    set(_vst3_bundle "${CMAKE_CURRENT_BINARY_DIR}/ShaderBackground_artefacts/$<CONFIG>/VST3/ShaderBackground.vst3")
    set(_vst3_ent "${CMAKE_CURRENT_BINARY_DIR}/ShaderBackground_artefacts/JuceLibraryCode/ShaderBackground_VST3.entitlements")

    add_custom_target(ShaderBackground_finalize ALL
        # Fix trailing commas in moduleinfo.json
        COMMAND ${CMAKE_COMMAND}
            -DMODULEINFO=${_vst3_bundle}/Contents/Resources/moduleinfo.json
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../../fix_moduleinfo.cmake
        # Re-sign with hardened runtime (overwrites JUCE's adhoc signature)
        COMMAND codesign --force --sign - --options runtime
            --entitlements "${_vst3_ent}"
            "${_vst3_bundle}"
        COMMENT "Finalizing VST3: fix moduleinfo + hardened runtime signing"
    )
    add_dependencies(ShaderBackground_finalize ShaderBackground_VST3)
endif()

# Sign other formats with hardened runtime too
if (APPLE AND NOT CMAKE_GENERATOR MATCHES "Xcode")
    foreach(_fmt Standalone AU)
        if (TARGET ShaderBackground_${_fmt})
            set(_ent "${CMAKE_BINARY_DIR}/ShaderBackground_artefacts/JuceLibraryCode/ShaderBackground_${_fmt}.entitlements")
            add_custom_command(TARGET ShaderBackground_${_fmt} POST_BUILD
                COMMAND codesign --force --sign - --options runtime
                    --entitlements "${_ent}"
                    "$<TARGET_BUNDLE_DIR:ShaderBackground_${_fmt}>"
                COMMENT "Signing ${_fmt} with hardened runtime"
            )
        endif()
    endforeach()
endif()

# --- Bundle Vulkan runtime into plugin targets (macOS only, no-op elsewhere) ---
if (COMMAND jvk_bundle)
    foreach(_fmt Standalone VST3 AU)
        if (TARGET ShaderBackground_${_fmt})
            jvk_bundle(ShaderBackground_${_fmt})
        endif()
    endforeach()
endif()
