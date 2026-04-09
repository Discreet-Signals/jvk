cmake_minimum_required(VERSION 3.20)
project(VulkanCompressor VERSION 1.0)

if (NOT COMMAND juce_add_plugin)
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

if (NOT TARGET jvk)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/jvk)
endif()

juce_add_plugin(VulkanCompressor
    PRODUCT_NAME "VulkanCompressor"
    COMPANY_NAME "DiscreetSignals"
    BUNDLE_ID "com.discreetsignals.VulkanCompressor"
    PLUGIN_MANUFACTURER_CODE DsSg
    PLUGIN_CODE VkCp
    FORMATS Standalone VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "VulkanCompressor needs microphone access for audio input."
    HARDENED_RUNTIME_ENABLED TRUE
    HARDENED_RUNTIME_OPTIONS "com.apple.security.device.audio-input"
)

target_sources(VulkanCompressor PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(VulkanCompressor PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(VulkanCompressor
    PRIVATE
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

if (APPLE AND TARGET VulkanCompressor_VST3)
    set(_vst3_bundle "${CMAKE_CURRENT_BINARY_DIR}/VulkanCompressor_artefacts/$<CONFIG>/VST3/VulkanCompressor.vst3")
    set(_vst3_ent "${CMAKE_CURRENT_BINARY_DIR}/VulkanCompressor_artefacts/JuceLibraryCode/VulkanCompressor_VST3.entitlements")

    add_custom_target(VulkanCompressor_finalize ALL
        COMMAND ${CMAKE_COMMAND}
            -DMODULEINFO=${_vst3_bundle}/Contents/Resources/moduleinfo.json
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../../fix_moduleinfo.cmake
        COMMAND codesign --force --sign - --options runtime
            --entitlements "${_vst3_ent}"
            "${_vst3_bundle}"
        COMMENT "Finalizing VST3: fix moduleinfo + hardened runtime signing"
    )
    add_dependencies(VulkanCompressor_finalize VulkanCompressor_VST3)
endif()

if (APPLE AND NOT CMAKE_GENERATOR MATCHES "Xcode")
    foreach(_fmt Standalone AU)
        if (TARGET VulkanCompressor_${_fmt})
            set(_ent "${CMAKE_BINARY_DIR}/VulkanCompressor_artefacts/JuceLibraryCode/VulkanCompressor_${_fmt}.entitlements")
            add_custom_command(TARGET VulkanCompressor_${_fmt} POST_BUILD
                COMMAND codesign --force --sign - --options runtime
                    --entitlements "${_ent}"
                    "$<TARGET_BUNDLE_DIR:VulkanCompressor_${_fmt}>"
                COMMENT "Signing ${_fmt} with hardened runtime"
            )
        endif()
    endforeach()
endif()

if (COMMAND jvk_bundle)
    foreach(_fmt Standalone VST3 AU)
        if (TARGET VulkanCompressor_${_fmt})
            jvk_bundle(VulkanCompressor_${_fmt})
        endif()
    endforeach()
endif()
