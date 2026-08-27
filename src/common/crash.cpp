#include "common/crash.h"

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <exception>

#if defined(_WIN32)
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace stream_moe {

namespace {

FILE* g_fatal = nullptr;

void fatal_open(const char* path) {
    g_fatal = std::fopen(path ? path : "temp/stream_moe_fatal.log", "ab");
}

void fatal_write(const char* what, uintptr_t code) {
    if (!g_fatal) return;
    char ts[32];
    const std::time_t now = std::time(nullptr);
    std::tm tmv;
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);
    std::fprintf(g_fatal, "[%s] %s code=0x%llX pid=%d\n",
                 ts, what, (unsigned long long) code,
#if defined(_WIN32)
                 (int) GetCurrentProcessId());
#else
                 (int) getpid());
#endif
    std::fflush(g_fatal);
}

void on_fatal(const char* what, uintptr_t code) {
    fatal_write(what, code);
#if defined(_WIN32)
    if (g_fatal) {
        // best-effort call stack so a repeat crash can be located
        void* frames[32];
        USHORT n = RtlCaptureStackBackTrace(1, 32, frames, nullptr);
        std::fprintf(g_fatal, "  stack(%u):", (unsigned) n);
        for (USHORT i = 0; i < n; ++i) std::fprintf(g_fatal, " %p", frames[i]);
        std::fprintf(g_fatal, "\n");
        std::fflush(g_fatal);
    }
#endif
    if (g_fatal) {
        std::fclose(g_fatal);
        g_fatal = nullptr;
    }
    _Exit(1);
}

#if defined(_WIN32)
LONG WINAPI seh_handler(EXCEPTION_POINTERS* ep) {
    on_fatal("SEH EXCEPTION", (uintptr_t) ep->ExceptionRecord->ExceptionCode);
    return EXCEPTION_EXECUTE_HANDLER; // not reached (on_fatal exits)
}
#endif

void signal_handler(int sig) {
    on_fatal("SIGNAL", (uintptr_t) sig);
}

} // namespace

void install_crash_handlers(const char* log_path) {
    const char* env_path = std::getenv("STREAM_MOE_FATAL_LOG");
    fatal_open(log_path ? log_path : (env_path && env_path[0] ? env_path : nullptr));

#if defined(_WIN32)
    SetUnhandledExceptionFilter(seh_handler);
    _set_abort_behavior(0, _WRITE_ABORT_MSG); // route abort() through the signal handler
#endif
    std::set_terminate([] { on_fatal("TERMINATE (uncaught C++ exception)", 0); });
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGFPE,  signal_handler);
    std::signal(SIGILL,  signal_handler);
}

} // namespace stream_moe
