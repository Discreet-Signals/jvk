juce_add_plugin(JVKCompliance
    PRODUCT_NAME "JVKCompliance"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.JVKCompliance"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE JvCp
    FORMATS Standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
)

target_sources(JVKCompliance PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(JVKCompliance PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(JVKCompliance
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

if (COMMAND jvk_bundle)
    foreach(_fmt Standalone)
        if (TARGET JVKCompliance_${_fmt})
            jvk_bundle(JVKCompliance_${_fmt})
        endif()
    endforeach()
endif()
