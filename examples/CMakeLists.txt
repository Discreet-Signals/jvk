cmake_minimum_required(VERSION 3.20)
project(jvk_examples VERSION 1.0)

# --- Fetch JUCE ---
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# --- Add the jvk module ---
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/jvk)

# --- Examples ---
add_subdirectory(triangles)
add_subdirectory(waveform)
