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

// NSView subclass that is transparent to mouse events.
// Returns nil from hitTest: so macOS skips this view and delivers
// mouse events to the JUCE component view behind it.
@interface JVKMetalView : NSView
@end

@implementation JVKMetalView
- (NSView*)hitTest:(NSPoint)point { return nil; }
- (BOOL)acceptsFirstMouse:(NSEvent*)event { return NO; }
@end

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

void* NSViewGenerator::create(int width, int height)
{
    NSView* view = [[JVKMetalView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
    if (!view.layer || ![view.layer isKindOfClass:[CAMetalLayer class]])
    {
        view.wantsLayer = YES;
        view.layer = [CAMetalLayer layer];
    }
    view.layer.opaque = NO;
    view.layer.contentsScale = 2.0;  // Match Retina backing scale; MSAA handles anti-aliasing
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
