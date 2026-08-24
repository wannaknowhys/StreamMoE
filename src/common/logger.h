#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

namespace stream_moe {

enum class log_level {
    debug,
    info,
    warn,
    error
};

inline void log_msg(log_level level, const std::string& msg) {
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