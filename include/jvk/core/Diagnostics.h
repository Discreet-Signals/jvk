#pragma once
// =============================================================================
// GPU bring-up log.
//
// One short line per stage of Vulkan bring-up (platform, loader, instance,
// GPU, device, swapchain, first frame, 120th frame, device loss), appended to
//   macOS:   ~/Library/jvk/gpu.log          (beside jvk's pipeline.cache)
//   Windows: %APPDATA%\jvk\gpu.log
// and forced to disk with fsync after every line. That last part is the point:
// when a machine goes down mid bring-up (a GPU driver panic takes the whole
// system with it), the page cache is lost with it, and a log that was merely
// flushed says nothing. A synced log's last line names the stage that killed
// the machine. A dozen lines per editor open; the cost is invisible.
//
// Rotates at 512 KB so it never grows without bound.
// =============================================================================
#include <cstdio>
#if JUCE_WINDOWS
 #include <io.h>
#else
 #include <unistd.h>
#endif

namespace jvk::diag {

inline juce::File logFile()
{
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("jvk").getChildFile("gpu.log");
}

inline void log(const juce::String& line)
{
    static juce::CriticalSection lock;
    const juce::ScopedLock l(lock);

    // The host names the process: "Live", "REAPER", "Punk Sans" (standalone).
    static const juce::String host =
        juce::File::getSpecialLocation(juce::File::hostApplicationPath).getFileNameWithoutExtension();
#if JUCE_WINDOWS
    static const auto pid = (long long) GetCurrentProcessId();
#else
    static const auto pid = (long long) getpid();
#endif

    auto f = logFile();
    f.getParentDirectory().createDirectory();
    if (f.getSize() > 512 * 1024)
    {
        auto old = f.getSiblingFile("gpu.log.1");
        old.deleteFile();
        f.moveFileTo(old);
    }

#if JUCE_WINDOWS
    FILE* fp = _wfopen(f.getFullPathName().toWideCharPointer(), L"ab");
#else
    FILE* fp = std::fopen(f.getFullPathName().toRawUTF8(), "ab");
#endif
    if (fp == nullptr)
        return;

    const auto stamp = juce::Time::getCurrentTime().formatted("%Y-%m-%d %H:%M:%S");
    std::fprintf(fp, "%s %s[%lld] %s\n", stamp.toRawUTF8(), host.toRawUTF8(), pid, line.toRawUTF8());
    std::fflush(fp);
#if JUCE_WINDOWS
    _commit(_fileno(fp));
#else
    fsync(fileno(fp));
#endif
    std::fclose(fp);
}

} // namespace jvk::diag
