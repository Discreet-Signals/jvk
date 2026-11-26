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
// Made through xcb, the X protocol library Xlib itself sits on (libxcb), not
// Xlib: xcb answers each request with its own error, where Xlib's error
// handler is one per process (swapping it around our requests raced any
// other thread's Xlib calls), and an xcb connection is thread-safe as it is,
// where Xlib needs XInitThreads before any other call in the process (too
// late for a plugin). The Vulkan driver presents from its own threads on
// this connection (VK_KHR_xcb_surface).
//
// No X11 header is included, here or anywhere in jvk: <X11/Xlib.h> defines
// None, Bool, Status, Complex and KeyPress as macros, and jvk.h is included
// by every consumer. The handful of xcb calls needed are declared below and
// looked up in libxcb at run time, which is how JUCE reaches X too, so a
// plugin still loads on a machine without X. A connection is an opaque
// pointer, a window its XID.
//
// The window is made on a connection of its own, not JUCE's (which JUCE does
// not expose). XIDs are the server's, so a child of the peer's window can be
// made from any connection, and the Vulkan driver only needs a connection
// and the XID.
//
// It selects no events and blocks none from propagating, so the server
// delivers every pointer event over it to the nearest ancestor that wants
// them: the JUCE peer. That is the pass-through HTTRANSPARENT gives the
// Windows child.
// =============================================================================
class Window
{
public:
    using Id = std::uint32_t;   // xcb_window_t

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

        connection_ = x.connect(nullptr, nullptr);   // $DISPLAY, as XOpenDisplay(nullptr)
        if (connection_ == nullptr || x.hasError(connection_) != 0) { release(); return false; }

        parent_ = (Id) reinterpret_cast<std::uintptr_t>(parentWindow);
        const Id window = x.generateId(connection_);
        // The parent's depth and visual, a black background: what XCreateSimpleWindow made.
        const std::uint32_t black = 0;
        if (failed(x.createWindow(connection_, copyFromParent, window, parent_, 0, 0,
                                  size(width), size(height), 0, inputOutput, copyFromParent,
                                  backPixel, &black)))
        {
            release();
            return false;
        }
        window_ = window;
        // Checked, thus waited for: the window is mapped before the surface is made on it.
        if (failed(x.mapWindow(connection_, window_))) { release(); return false; }
        return true;
    }

    // In the parent's pixels. Synchronous: the swapchain is sized from the
    // window, so the server must have it before the next extent query.
    void setBounds(int x, int y, int width, int height)
    {
        if (window_ == 0) return;
        const std::uint32_t values[] = { (std::uint32_t) x, (std::uint32_t) y, size(width), size(height) };
        constexpr auto all = (std::uint16_t) (configX | configY | configWidth | configHeight);
        failed(Api::get().configureWindow(connection_, window_, all, values));
    }

    void* connection() const { return connection_; }
    Id    id() const         { return window_; }
    Id    parent() const     { return parent_; }

    void release()
    {
        auto& x = Api::get();
        if (connection_ != nullptr)
        {
            // The server has already destroyed it if the peer's window went
            // first, and the BadWindow that answers is not an error.
            if (window_ != 0) failed(x.destroyWindow(connection_, window_));
            x.disconnect(connection_);
        }
        connection_ = nullptr;
        window_ = parent_ = 0;
    }

private:
    struct Cookie { unsigned int sequence; };   // xcb_void_cookie_t

    // xcb's values for what is used here (xproto.h).
    static constexpr std::uint8_t  copyFromParent = 0;
    static constexpr std::uint16_t inputOutput = 1;
    static constexpr std::uint32_t backPixel = 2;
    static constexpr std::uint16_t configX = 1, configY = 2, configWidth = 4, configHeight = 8;

    static std::uint16_t size(int pixels) { return (std::uint16_t) std::clamp(pixels, 1, 32767); }

    struct Api
    {
        void*         (*connect)(const char*, int*) = nullptr;
        int           (*hasError)(void*) = nullptr;
        void          (*disconnect)(void*) = nullptr;
        std::uint32_t (*generateId)(void*) = nullptr;
        Cookie        (*createWindow)(void*, std::uint8_t, Id, Id, std::int16_t, std::int16_t,
                                      std::uint16_t, std::uint16_t, std::uint16_t, std::uint16_t,
                                      std::uint32_t, std::uint32_t, const void*) = nullptr;
        Cookie        (*mapWindow)(void*, Id) = nullptr;
        Cookie        (*configureWindow)(void*, Id, std::uint16_t, const void*) = nullptr;
        Cookie        (*destroyWindow)(void*, Id) = nullptr;
        void*         (*requestCheck)(void*, Cookie) = nullptr;

        bool usable() const
        {
            return connect && hasError && disconnect && generateId && createWindow
                && mapWindow && configureWindow && destroyWindow && requestCheck;
        }

        static const Api& get()
        {
            static const Api api = []
            {
                Api a;
                // Never closed, not even at exit (hence the leak): the Vulkan
                // driver holds on to libxcb through the connection it was
                // given, and outlives this object.
                static auto& lib = *new juce::DynamicLibrary();
                if (!lib.open("libxcb.so.1")) return a;
                auto load = [](auto& fn, const char* name)
                {
                    fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(lib.getFunction(name));
                };
                load(a.connect, "xcb_connect");
                load(a.hasError, "xcb_connection_has_error");
                load(a.disconnect, "xcb_disconnect");
                load(a.generateId, "xcb_generate_id");
                load(a.createWindow, "xcb_create_window_checked");
                load(a.mapWindow, "xcb_map_window_checked");
                load(a.configureWindow, "xcb_configure_window_checked");
                load(a.destroyWindow, "xcb_destroy_window_checked");
                load(a.requestCheck, "xcb_request_check");
                return a;
            }();
            return api;
        }
    };

    // Waits for a checked request's answer: true when it was an error, which
    // is ours to free and reaches no handler anywhere else.
    bool failed(Cookie cookie) const
    {
        void* error = Api::get().requestCheck(connection_, cookie);
        const bool any = error != nullptr;
        std::free(error);   // xcb hands the error over: freed with free()
        return any;
    }

    void* connection_ = nullptr;
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
