/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: HWNDGenerator.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include <windows.h>

namespace jvk::core::windows
{

// Creates a child HWND that Vulkan renders into. The HWND is handed to
// juce::HWNDComponent::setHWND — JUCE re-parents it to the plugin window
// and flips the style to WS_CHILD|WS_VISIBLE. Mouse events pass through
// (HTTRANSPARENT) so JUCE's component tree owns input.
class HWNDGenerator
{
public:
    HWNDGenerator() = default;
    ~HWNDGenerator() { release(); }

    HWNDGenerator(const HWNDGenerator&) = delete;
    HWNDGenerator& operator=(const HWNDGenerator&) = delete;

    HWND create()
    {
        if (hwnd) return hwnd;
        HINSTANCE hInst = moduleHInstance();
        registerClassOnce(hInst);
        hwnd = CreateWindowEx(
            WS_EX_NOACTIVATE | WS_EX_TRANSPARENT,
            className(),
            TEXT("jvk"),
            WS_POPUP,
            0, 0, 1, 1,
            nullptr, nullptr,
            hInst,
            nullptr);
        return hwnd;
    }

    HWND get() const { return hwnd; }

    void release()
    {
        if (hwnd)
        {
            DestroyWindow(hwnd);
            hwnd = nullptr;
        }
    }

private:
    HWND hwnd = nullptr;

    static LPCTSTR className() { return TEXT("jvk_vulkan_surface"); }

    // Returns the HINSTANCE of the module containing this function — the
    // plugin DLL, not the host process. RegisterClassEx ties the class to
    // this hInstance, so the WndProc stays valid for the plugin's lifetime.
    static HINSTANCE moduleHInstance()
    {
        HMODULE hm = nullptr;
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&moduleHInstance),
            &hm);
        return reinterpret_cast<HINSTANCE>(hm);
    }

    static void registerClassOnce(HINSTANCE hInst)
    {
        WNDCLASSEX wc {};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wc.lpfnWndProc = &WindowProc;
        wc.hInstance = hInst;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = nullptr;
        wc.lpszClassName = className();
        if (!RegisterClassEx(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
            DBG("jvk: RegisterClassEx failed, err=" << (int)GetLastError());
    }

    static LRESULT CALLBACK WindowProc(HWND h, UINT m, WPARAM w, LPARAM l)
    {
        if (m == WM_NCHITTEST) return HTTRANSPARENT;
        return DefWindowProc(h, m, w, l);
    }
};

} // jvk::core::windows

// windows.h defines legacy macros that collide with common identifiers.
#ifdef near
 #undef near
#endif
#ifdef far
 #undef far
#endif
#ifdef small
 #undef small
#endif
