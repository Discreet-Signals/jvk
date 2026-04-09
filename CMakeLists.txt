cmake_minimum_required(VERSION 3.20)
project(jvk_project VERSION 1.0)

option(DS_VULKAN_BUILD_EXAMPLES "Build example projects" OFF)

add_subdirectory(jvk)

if (DS_VULKAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
