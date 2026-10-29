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

// ---------------------------------------------------------------------------
// Per-window DPI-awareness-context utilities (Win10 1607+, loaded dynamically).
// Every helper degrades to a no-op where the APIs are absent — those systems
// have a single process-wide awareness, so there is nothing to mismatch.
// ---------------------------------------------------------------------------
namespace dpi
{
    using Ctx = DPI_AWARENESS_CONTEXT;

    struct Api
    {
        Ctx  (WINAPI* setThread)(Ctx)     = nullptr;
        Ctx  (WINAPI* getWindow)(HWND)    = nullptr;
        BOOL (WINAPI* areEqual)(Ctx, Ctx) = nullptr;

        static const Api& get()
        {
            static const Api api = []
            {
                Api a;
                if (HMODULE user32 = GetModuleHandleA("user32.dll"))
                {
                    a.setThread = reinterpret_cast<Ctx(WINAPI*)(Ctx)>(
                        GetProcAddress(user32, "SetThreadDpiAwarenessContext"));
                    a.getWindow = reinterpret_cast<Ctx(WINAPI*)(HWND)>(
                        GetProcAddress(user32, "GetWindowDpiAwarenessContext"));
                    a.areEqual  = reinterpret_cast<BOOL(WINAPI*)(Ctx, Ctx)>(
                        GetProcAddress(user32, "AreDpiAwarenessContextsEqual"));
                }
                return a;
            }();
            return api;
        }
    };

    // The DPI-awareness context `window` was created in (null if unknowable).
    inline Ctx windowContext(HWND window)
    {
        auto& api = Api::get();
        if (window && api.getWindow) return api.getWindow(window);
        return nullptr;
    }

    // True when both windows live in the same DPI-awareness context — the
    // precondition Windows puts on parent/child window relationships, and
    // the one juce::HWNDComponent silently assumes when it positions an
    // embedded HWND with coordinates computed in the peer's space.
    // Unknowable (old OS, null handle) counts as "same": nothing to fix.
    inline bool sameContext(HWND a, HWND b)
    {
        auto& api = Api::get();
        if (!api.areEqual) return true;
        Ctx ca = windowContext(a), cb = windowContext(b);
        if (ca == nullptr || cb == nullptr) return true;
        return api.areEqual(ca, cb) != FALSE;
    }

    // RAII: switch the calling thread to `ctx`; restores on exit. No-op when
    // ctx is null or the API is unavailable.
    class ScopedThreadContext
    {
    public:
        explicit ScopedThreadContext(Ctx ctx)
        {
            auto& api = Api::get();
            if (ctx && api.setThread) old_ = api.setThread(ctx);
        }
        ~ScopedThreadContext()
        {
            if (old_) Api::get().setThread(old_);
        }
        ScopedThreadContext(const ScopedThreadContext&) = delete;
        ScopedThreadContext& operator=(const ScopedThreadContext&) = delete;

    private:
        Ctx old_ = nullptr;
    };

    // RAII: switch the calling thread into `window`'s own DPI-awareness
    // context. Win32 coordinate APIs (GetClientRect & friends — including
    // the ones a Vulkan ICD runs internally during surface-capability
    // queries, acquire and present) DPI-virtualize their results whenever
    // the calling thread's context differs from the window's. Pinning the
    // thread to the window's context makes every query return coordinates
    // in the window's real, unvirtualized space — the same answer on every
    // thread, which is what lets the swapchain size converge.
    class ScopedWindowContext : public ScopedThreadContext
    {
    public:
        explicit ScopedWindowContext(HWND window)
            : ScopedThreadContext(windowContext(window)) {}
        explicit ScopedWindowContext(void* window)
            : ScopedWindowContext(reinterpret_cast<HWND>(window)) {}
    };
} // namespace dpi

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

    // `parentWindow` is the window this HWND is about to be parented into
    // (the plugin editor's peer). The child is created in the SAME
    // DPI-awareness context as that tree. Windows does not support
    // parenting windows whose DPI-awareness contexts differ — SetParent
    // across contexts is documented-unsupported, DWM composition of the
    // mixed tree misbehaves (black plugin windows under Ableton's
    // "Auto-Scale Plugin Window", which hosts the editor in a DPI-UNAWARE
    // tree and bitmap-stretches it), and every cross-context coordinate
    // exchange gets DPI-virtualized: juce::HWNDComponent computes our
    // bounds in the peer's coordinate space but applies them via
    // SetWindowPos in OUR context, so a context mismatch mis-scales the
    // child (the "UI fills only ~2/3 of the window" bug). Matching the
    // parent keeps the whole tree in one coordinate space — which is
    // exactly what JUCE itself does when it creates peers for embedded
    // plugin windows (ScopedThreadDPIAwarenessSetter).
    //
    // With no parent supplied, fall back to Per-Monitor-Aware-V2 (the old
    // behaviour) so standalone/experimental windows escape DPI
    // virtualization when the surrounding thread context is stale.
    HWND create(void* parentWindow = nullptr)
    {
        if (hwnd) return hwnd;
        HINSTANCE hInst = moduleHInstance();
        registerClassOnce(hInst);

        dpi::Ctx ctx = dpi::windowContext(reinterpret_cast<HWND>(parentWindow));
        if (ctx == nullptr)
        {
           #ifdef DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            ctx = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2;
           #else
            ctx = reinterpret_cast<dpi::Ctx>(static_cast<intptr_t>(-4));
           #endif
        }

        // No WS_EX_TRANSPARENT — that flag tells DWM the pixels should blend
        // with what's underneath, forcing a composition-blend path for the
        // child's DXGI swapchain over the JUCE peer's surface. On Windows
        // that path serializes DWM composition with window drag + input,
        // freezing the OS-level move/mouse thread even while the app's
        // render thread itself is healthy. Mouse pass-through is handled
        // by HTTRANSPARENT in WindowProc; paint order is irrelevant because
        // the swapchain fills the whole HWND opaquely.
        dpi::ScopedThreadContext dpiScope(ctx);
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
