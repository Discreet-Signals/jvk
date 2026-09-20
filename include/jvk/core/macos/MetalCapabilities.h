/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: MetalCapabilities.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
namespace jvk::core::macos
{

/** What this Mac's Metal stack looks like, read straight from Metal before
    any Vulkan call is made — MoltenVK reads its configuration from the
    environment on its first call, so anything the answer changes has to be
    decided ahead of it (see Device.cpp, configureMoltenVK). */
struct MetalReport
{
    int osMajor = 0, osMinor = 0, osPatch = 0;

    /** True when any Metal device on the machine does not reach
        MTLGPUFamilyMac2 — "Mac GPU family 1" hardware: Intel HD 4000 / 5000 /
        Iris / Iris Pro and NVIDIA Kepler, the GPUs of the 2012-2014 Macs that
        top out at macOS 10.15 / 11. MoltenVK 1.4.2 uses exactly this test
        (MVKPhysicalDevice::isMacGPUFamily1) to steer those GPUs away from the
        Metal features their drivers cannot handle. */
    bool anyMacFamily1GPU = false;

    /** One line per Metal device, for the bring-up log. */
    juce::StringArray devices;

    /** The whole 10.15 population plus every Mac-family-1 GPU on any OS: the
        machines whose Metal drivers predate the MoltenVK defaults we ship. */
    bool legacy() const { return osMajor < 11 || anyMacFamily1GPU; }
};

MetalReport probeMetal();

} // namespace jvk::core::macos
