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

        // Force Per-Monitor-Aware-V2 on this HWND. Some hosts run plugin-GUI
        // threads in DPI_UNAWARE context for legacy compatibility; creating
        // the HWND there triggers DWM DPI virtualization — the backing buffer
        // is allocated at logical size and stretch-blitted on present, giving
        // blurry output and expensive stretch-path presents that stall the
        // message thread. The scope restores the prior context on exit; a
        // no-op on pre-Win10-1703 where SetThreadDpiAwarenessContext is absent.
        // No WS_EX_TRANSPARENT — that flag tells DWM the pixels should blend
        // with what's underneath, forcing a composition-blend path for the
        // child's DXGI swapchain over the JUCE peer's surface. On Windows
        // that path serializes DWM composition with window drag + input,
        // freezing the OS-level move/mouse thread even while the app's
        // render thread itself is healthy. Mouse pass-through is handled
        // by HTTRANSPARENT in WindowProc; paint order is irrelevant because
        // the swapchain fills the whole HWND opaquely.
        ScopedThreadDpiAwareness dpiScope;
        hwnd = CreateWindowEx(
            WS_EX_NOACTIVATE,
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

    // Returns the HINSTANCE of the module containing this function — the
    // plugin DLL, not the host process. RegisterClassEx ties the class to
    // this hInstance, and VkWin32SurfaceCreateInfoKHR::hinstance must match,
    // so callers creating the surface use this too.
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

private:
    HWND hwnd = nullptr;

    static LPCTSTR className() { return TEXT("jvk_vulkan_surface"); }

    class ScopedThreadDpiAwareness
    {
    public:
        ScopedThreadDpiAwareness()
        {
            if (auto fn = getSetFn())
            {
               #ifdef DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                old = fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
               #else
                old = fn(reinterpret_cast<Ctx>(static_cast<intptr_t>(-4)));
               #endif
            }
        }
        ~ScopedThreadDpiAwareness()
        {
            if (old != nullptr)
                if (auto fn = getSetFn())
                    fn(old);
        }
        ScopedThreadDpiAwareness(const ScopedThreadDpiAwareness&) = delete;
        ScopedThreadDpiAwareness& operator=(const ScopedThreadDpiAwareness&) = delete;

    private:
        using Ctx = DPI_AWARENESS_CONTEXT;
        using SetFn = Ctx (WINAPI*)(Ctx);
        static SetFn getSetFn()
        {
            static SetFn fn = []() -> SetFn {
                if (HMODULE user32 = GetModuleHandleA("user32.dll"))
                    return reinterpret_cast<SetFn>(
                        GetProcAddress(user32, "SetThreadDpiAwarenessContext"));
                return nullptr;
            }();
            return fn;
        }
        Ctx old = nullptr;
    };

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
