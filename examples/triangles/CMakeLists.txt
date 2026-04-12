juce_add_plugin(jvkTriangles
    PRODUCT_NAME "jvkTriangles"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.jvkTriangles"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE JvTr
    FORMATS Standalone VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
)

juce_add_binary_data(jvkTriangles_BinaryData SOURCES
    Source/Shaders/basic.vert.spv
    Source/Shaders/basic.frag.spv
    Source/Shaders/invert.frag.spv
)

target_sources(jvkTriangles PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(jvkTriangles PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(jvkTriangles
    PRIVATE
        jvkTriangles_BinaryData
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

# --- Bundle Vulkan runtime into plugin targets (macOS only, no-op elsewhere) ---
if (COMMAND jvk_bundle)
    foreach(_fmt Standalone VST3 AU)
        if (TARGET jvkTriangles_${_fmt})
            jvk_bundle(jvkTriangles_${_fmt})
        endif()
    endforeach()
endif()
