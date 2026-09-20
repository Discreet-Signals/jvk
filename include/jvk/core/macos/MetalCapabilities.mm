/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: MetalCapabilities.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "MetalCapabilities.h"

namespace jvk::core::macos
{

MetalReport probeMetal()
{
    MetalReport r;
    @autoreleasepool
    {
        const NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
        r.osMajor = (int) v.majorVersion;
        r.osMinor = (int) v.minorVersion;
        r.osPatch = (int) v.patchVersion;

        // Copy rule: MTLCopyAllDevices returns +1 and this module is MRC
        // (NSViewGenerator releases by hand too), so balance it below.
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> d in devices)
        {
            // supportsFamily: is macOS 10.15+. Anything older cannot answer
            // and is legacy by definition.
            bool mac2 = false;
            if ([d respondsToSelector: @selector(supportsFamily:)])
                mac2 = [d supportsFamily: MTLGPUFamilyMac2];

            r.anyMacFamily1GPU |= ! mac2;

            juce::String line = juce::String::fromUTF8([d.name UTF8String]);
            line += mac2 ? " [Mac2]" : " [Mac1: legacy]";
            if (d.isLowPower)  line += " low-power";
            if (d.isHeadless)  line += " headless";
            if (d.isRemovable) line += " external";
            r.devices.add(line);
        }
        [devices release];
    }
    return r;
}

} // namespace jvk::core::macos
