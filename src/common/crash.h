#pragma once

// Global crash/SEH/signal guard: on any fatal error, append a timestamped
// diagnostic line to the log file (default "temp/stream_moe_fatal.log",
// overridable via env STREAM_MOE_FATAL_LOG or an explicit path), then exit(1).
namespace stream_moe {

void install_crash_handlers(const char* log_path = nullptr);

} // namespace stream_moe
