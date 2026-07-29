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
 File: NSViewGenerator.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
namespace jvk::core::macos
{

class NSViewGenerator
{
public:
    NSViewGenerator();
    ~NSViewGenerator();

    bool isValid();
    /** Width/height in logical points — the layer's frame at birth. MoltenVK's
        currentExtent reads it, and createSwapchain treats currentExtent as
        authoritative, so a wrong birth frame becomes a wrong-aspect first
        swapchain, presented stretched until the next extent check. */
    void* create(int width, int height);
    void release();
private:
    // Null-init is load-bearing: ~NSViewGenerator → release() reads this,
    // and create() may never run (zero-size window, Windows branch bail) —
    // uninitialized, the dtor sent removeFromSuperview to a garbage pointer.
    void* ptr = nullptr;
};

} // jvk::core::macos
