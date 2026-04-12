juce_add_plugin(JVKBenchmark
    PRODUCT_NAME "JVKBenchmark"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.JVKBenchmark"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE JvBm
    FORMATS Standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "JVKBenchmark needs microphone access for audio input."
    HARDENED_RUNTIME_ENABLED TRUE
    HARDENED_RUNTIME_OPTIONS "com.apple.security.device.audio-input"
)

target_sources(JVKBenchmark PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(JVKBenchmark PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(JVKBenchmark
    PRIVATE
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

if (APPLE AND NOT CMAKE_GENERATOR MATCHES "Xcode")
    foreach(_fmt Standalone)
        if (TARGET JVKBenchmark_${_fmt})
            set(_ent "${CMAKE_CURRENT_BINARY_DIR}/JVKBenchmark_artefacts/JuceLibraryCode/JVKBenchmark_${_fmt}.entitlements")
            add_custom_command(TARGET JVKBenchmark_${_fmt} POST_BUILD
                COMMAND codesign --force --sign - --options runtime
                    --entitlements "${_ent}"
                    "$<TARGET_BUNDLE_DIR:JVKBenchmark_${_fmt}>"
                COMMENT "Signing ${_fmt} with hardened runtime"
            )
        endif()
    endforeach()
endif()

if (COMMAND jvk_bundle)
    foreach(_fmt Standalone)
        if (TARGET JVKBenchmark_${_fmt})
            jvk_bundle(JVKBenchmark_${_fmt})
        endif()
    endforeach()
endif()
