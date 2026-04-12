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
 File: HWNDGenerator.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include <windows.h>

namespace jvk::core::windows
{

class HWNDGenerator
{
public:
    HWNDGenerator() : hwnd(nullptr) { }
    ~HWNDGenerator() { release(); }

    HWND& get() { return hwnd; }
    HWND& create()
    {
        DBG("Creating HWND...");
        HINSTANCE hInstance = GetModuleHandle(nullptr);
        const char* className = "jvk";
        WNDCLASSEX wc = {};
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &WindowProc;
        wc.hInstance = hInstance;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszClassName = className;

        if (!RegisterClassEx(&wc))
            DBG("Failed to Create Window!");
        hwnd = CreateWindowEx(
            WS_EX_TRANSPARENT | WS_EX_LAYERED,
            className,
            className,
            WS_POPUP,
            0, 0, 800, 600,
            nullptr, nullptr,
            GetModuleHandle(nullptr),
            nullptr
        );
        if (hwnd)
            DBG("Created HWND!");
        else
            DBG("Failed To Create Window");
        return hwnd;
    }
    void setSize(int width, int height)
    {
        SetWindowPos(hwnd, NULL, 0, 0, width, height, SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOMOVE);
    }
    void release()
    {
        DestroyWindow(hwnd);
        MSG msg;
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        UnregisterClass("jvk", GetModuleHandle(nullptr));
    }
private:
    HWND hwnd;

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        switch (uMsg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        // Forward all mouse messages to the parent (JUCE's HWND)
        case WM_NCHITTEST:
            return HTTRANSPARENT;
        case WM_MOUSEMOVE:
        case WM_LBUTTONDOWN: case WM_LBUTTONUP:
        case WM_RBUTTONDOWN: case WM_RBUTTONUP:
        case WM_MBUTTONDOWN: case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:  case WM_MOUSEHWHEEL:
        case WM_MOUSELEAVE:
        {
            HWND parent = GetParent(hwnd);
            if (parent)
                return SendMessage(parent, uMsg, wParam, lParam);
            break;
        }
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
};

} // jvk::core::windows
