/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: X11Window.h
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core::x11
{

// =============================================================================
// Window — the child X11 window Vulkan renders into on Linux
//
// No X11 header is included, here or anywhere in jvk: <X11/Xlib.h> defines
// None, Bool, Status, Complex and KeyPress as macros, and jvk.h is included
// by every consumer. The handful of Xlib calls needed are declared below and
// looked up in libX11 at run time, which is also how JUCE reaches it, so a
// plugin still loads on a machine without X. Display is an opaque pointer,
// a window is its XID.
//
// The window is made on a connection of its own, not JUCE's (which JUCE does
// not expose without those headers). XIDs are the server's, so a child of
// the peer's window can be made from any connection, and the Vulkan driver
// only needs a connection and the XID.
//
// It selects no events and blocks none from propagating, so the server
// delivers every pointer event over it to the nearest ancestor that wants
// them: the JUCE peer. That is the pass-through HTTRANSPARENT gives the
// Windows child.
// =============================================================================
class Window
{
public:
    using Id = unsigned long;

    Window() = default;
    ~Window() { release(); }

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // `parentWindow`: the peer's native handle, an XID carried in a pointer.
    bool create(void* parentWindow, int width, int height)
    {
        if (window_ != 0) return true;
        auto& x = Api::get();
        if (!x.usable() || parentWindow == nullptr) return false;

        display_ = x.openDisplay(nullptr);
        if (display_ == nullptr) return false;

        parent_ = (Id) reinterpret_cast<std::uintptr_t>(parentWindow);
        const IgnoreErrors ignore;
        window_ = x.createSimpleWindow(display_, parent_, 0, 0,
                                       (unsigned) std::max(1, width), (unsigned) std::max(1, height),
                                       0, 0, 0);
        if (window_ != 0) x.mapWindow(display_, window_);
        x.sync(display_, 0);
        if (window_ == 0) { release(); return false; }
        return true;
    }

    // In the parent's pixels. Synchronous: the swapchain is sized from the
    // window, so the server must have it before the next extent query.
    void setBounds(int x, int y, int width, int height)
    {
        if (window_ == 0) return;
        auto& api = Api::get();
        const IgnoreErrors ignore;
        api.moveResizeWindow(display_, window_, x, y,
                             (unsigned) std::max(1, width), (unsigned) std::max(1, height));
        api.sync(display_, 0);
    }

    void* display() const { return display_; }
    Id    id() const      { return window_; }
    Id    parent() const  { return parent_; }

    void release()
    {
        auto& x = Api::get();
        if (display_ != nullptr)
        {
            {
                // The server has already destroyed it if the peer's window
                // went first, and the BadWindow that answers is not an error.
                const IgnoreErrors ignore;
                if (window_ != 0) x.destroyWindow(display_, window_);
                x.sync(display_, 0);
            }
            x.closeDisplay(display_);
        }
        display_ = nullptr;
        window_ = parent_ = 0;
    }

private:
    using ErrorHandler = int (*)(void*, void*);

    struct Api
    {
        int   (*initThreads)() = nullptr;
        void* (*openDisplay)(const char*) = nullptr;
        int   (*closeDisplay)(void*) = nullptr;
        Id    (*createSimpleWindow)(void*, Id, int, int, unsigned, unsigned, unsigned,
                                    unsigned long, unsigned long) = nullptr;
        int   (*mapWindow)(void*, Id) = nullptr;
        int   (*moveResizeWindow)(void*, Id, int, int, unsigned, unsigned) = nullptr;
        int   (*destroyWindow)(void*, Id) = nullptr;
        int   (*sync)(void*, int) = nullptr;
        ErrorHandler (*setErrorHandler)(ErrorHandler) = nullptr;

        bool usable() const
        {
            return openDisplay && closeDisplay && createSimpleWindow && mapWindow
                && moveResizeWindow && destroyWindow && sync && setErrorHandler;
        }

        static const Api& get()
        {
            static const Api api = []
            {
                Api a;
                // Never closed, not even at exit (hence the leak): the Vulkan
                // driver holds on to libX11 through the connection it was
                // given, and outlives this object.
                static auto& lib = *new juce::DynamicLibrary();
                if (!lib.open("libX11.so.6")) return a;
                auto load = [](auto& fn, const char* name)
                {
                    fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(lib.getFunction(name));
                };
                load(a.initThreads, "XInitThreads");
                load(a.openDisplay, "XOpenDisplay");
                load(a.closeDisplay, "XCloseDisplay");
                load(a.createSimpleWindow, "XCreateSimpleWindow");
                load(a.mapWindow, "XMapWindow");
                load(a.moveResizeWindow, "XMoveResizeWindow");
                load(a.destroyWindow, "XDestroyWindow");
                load(a.sync, "XSync");
                load(a.setErrorHandler, "XSetErrorHandler");
                // The driver presents from its own threads on this connection.
                if (a.initThreads) a.initThreads();
                return a;
            }();
            return api;
        }
    };

    // Xlib's error handler is one per process, and its default exits the
    // process. Around our own requests it is swapped for one that does
    // nothing, then put back: the host's (or JUCE's) is never left replaced.
    struct IgnoreErrors
    {
        IgnoreErrors()  { old = Api::get().setErrorHandler([](void*, void*) -> int { return 0; }); }
        ~IgnoreErrors() { Api::get().setErrorHandler(old); }
        ErrorHandler old = nullptr;
    };

    void* display_ = nullptr;
    Id    window_ = 0;
    Id    parent_ = 0;
};

// The juce::Component that keeps a Window over its own bounds, as
// juce::HWNDComponent and juce::NSViewComponent do for the other platforms.
class View final : public juce::Component
{
public:
    void setWindow(Window* window)
    {
        window_ = window;
        watcher_ = window != nullptr ? std::make_unique<Watcher>(*this) : nullptr;
        updateWindowBounds();
    }

    void updateWindowBounds()
    {
        if (window_ == nullptr) return;
        if (auto* peer = getPeer())
        {
            // The same conversion juce::XEmbedComponent makes.
            const auto r = peer->getComponent().getLocalArea(this, getLocalBounds())
                             * peer->getPlatformScaleFactor()
                             * peer->getComponent().getDesktopScaleFactor();
            window_->setBounds(r.getX(), r.getY(), r.getWidth(), r.getHeight());
        }
    }

private:
    struct Watcher final : juce::ComponentMovementWatcher
    {
        explicit Watcher(View& v) : juce::ComponentMovementWatcher(&v), view(v) {}
        using juce::ComponentMovementWatcher::componentMovedOrResized;
        using juce::ComponentMovementWatcher::componentVisibilityChanged;
        void componentMovedOrResized(bool, bool) override { view.updateWindowBounds(); }
        void componentPeerChanged() override              { view.updateWindowBounds(); }
        void componentVisibilityChanged() override        {}
        View& view;
    };

    Window* window_ = nullptr;
    std::unique_ptr<Watcher> watcher_;
};

} // namespace jvk::core::x11
