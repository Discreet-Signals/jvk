juce_add_plugin(jvkWaveform
    PRODUCT_NAME "jvkWaveform"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.jvkWaveform"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE JvWf
    FORMATS Standalone VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "jvkWaveform needs microphone access to visualize audio input."
)

juce_add_binary_data(jvkWaveform_BinaryData SOURCES
    Source/Waveform.frag.spv
)

target_sources(jvkWaveform PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(jvkWaveform PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(jvkWaveform
    PRIVATE
        jvkWaveform_BinaryData
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

# --- Bundle Vulkan runtime into plugin targets (macOS only, no-op elsewhere) ---
if (COMMAND jvk_bundle)
    foreach(_fmt Standalone VST3 AU)
        if (TARGET jvkWaveform_${_fmt})
            jvk_bundle(jvkWaveform_${_fmt})
        endif()
    endforeach()
endif()
