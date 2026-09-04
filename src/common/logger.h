#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdlib>

namespace stream_moe {

enum class log_level {
    debug,
    info,
    warn,
    error
};

// StreamMoE logs print unconditionally unless gated. Default threshold = info:
// debug (per-expert move events, demote storms, ...) is off unless
// STREAM_MOE_LOG=debug. llama's -lv does not control this logger.
inline log_level& log_threshold() {
    static log_level th = [] {
        const char* e = std::getenv("STREAM_MOE_LOG");
        if (e && std::string(e) == "debug") return log_level::debug;
        if (e && std::string(e) == "warn")  return log_level::warn;
        if (e && std::string(e) == "error") return log_level::error;
        return log_level::info;
    } ();
    return th;
}

inline void log_msg(log_level level, const std::string& msg) {
    if (level < log_threshold()) return;
    const char* tag = "[INFO]";
    switch (level) {
        case log_level::debug: tag = "[DEBUG]"; break;
        case log_level::info:  tag = "[INFO]";  break;
        case log_level::warn:  tag = "[WARN]";  break;
        case log_level::error: tag = "[ERROR]"; break;
    }
    std::cout << tag << " " << msg << std::endl;
}

#define LOG_DEBUG(msg) do { std::ostringstream _oss; _oss << msg; ::stream_moe::log_msg(::stream_moe::log_level::debug, _oss.str()); } while(0)
#define LOG_INFO(msg)  do { std::ostringstream _oss; _oss << msg; ::stream_moe::log_msg(::stream_moe::log_level::info,  _oss.str()); } while(0)
#define LOG_WARN(msg)  do { std::ostringstream _oss; _oss << msg; ::stream_moe::log_msg(::stream_moe::log_level::warn,  _oss.str()); } while(0)
#define LOG_ERROR(msg) do { std::ostringstream _oss; _oss << msg; ::stream_moe::log_msg(::stream_moe::log_level::error, _oss.str()); } while(0)

} // namespace stream_moe