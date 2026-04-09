/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC
 
 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝
 
 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 
 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: NSViewGenerator.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#include <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include "NSViewGenerator.h"

namespace jvk::core::macos
{

NSViewGenerator::NSViewGenerator()
{
}

NSViewGenerator::~NSViewGenerator()
{
    release();
}

bool NSViewGenerator::isValid()
{
    NSView* view = (NSView*)ptr;
    return view;
}

void* NSViewGenerator::create()
{
    NSView* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 800, 600)];
    if (!view.layer || ![view.layer isKindOfClass:[CAMetalLayer class]])
    {
        view.wantsLayer = YES;
        view.layer = [CAMetalLayer layer];
    }
    view.layer.opaque = NO;
    view.layer.contentsScale = 2.0;  // 2x supersampling for sharp edges
    view.layer.backgroundColor = CGColorGetConstantColor(kCGColorClear);
    [view retain];
    ptr = (void*)view;
    return (void*)view;
}

void NSViewGenerator::release()
{
    NSView* view = (NSView*)ptr;
    if (view)
    {
        [view removeFromSuperview];
        [view release];
        ptr = nullptr;
    }
}

} // jvk::core::macos
