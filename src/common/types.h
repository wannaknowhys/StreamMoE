#include <thread>
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

namespace stream_moe {

constexpr size_t DIO_SECTOR_SIZE = 4096;
constexpr size_t DIO_ALIGN_MASK  = DIO_SECTOR_SIZE - 1;
constexpr size_t MAX_SUB_TENSORS_PER_EXPERT = 8;

// Round down value to alignment multiple
template <typename T>
inline constexpr T align_floor(T val, size_t align = DIO_SECTOR_SIZE) {
    return static_cast<T>(val & ~static_cast<T>(align - 1));
}

// Round up value to alignment multiple
template <typename T>
inline constexpr T align_ceil(T val, size_t align = DIO_SECTOR_SIZE) {
    return static_cast<T>((val + static_cast<T>(align - 1)) & ~static_cast<T>(align - 1));
}

// Check if integer value is aligned
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline constexpr bool is_aligned(T val, size_t align = DIO_SECTOR_SIZE) {
    return (val & static_cast<T>(align - 1)) == 0;
}

// Check if pointer is aligned
template <typename T, typename std::enable_if<std::is_pointer<T>::value, int>::type = 0>
inline constexpr bool is_aligned(T ptr, size_t align = DIO_SECTOR_SIZE) {
    return (reinterpret_cast<uintptr_t>(ptr) & (align - 1)) == 0;
}

} // namespace stream_moe
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace stream_moe {

inline size_t get_available_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return static_cast<size_t>(mem_status.ullAvailPhys);
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback 16GB
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>(info.freeram) * info.mem_unit;
    }
    return 16ULL * 1024 * 1024 * 1024;
#endif
}

inline size_t get_total_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return static_cast<size_t>(mem_status.ullTotalPhys);
    }
    return 32ULL * 1024 * 1024 * 1024;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>(info.totalram) * info.mem_unit;
    }
    return 32ULL * 1024 * 1024 * 1024;
#endif
}

inline uint32_t get_default_threads() {
    uint32_t count = std::thread::hardware_concurrency();
    if (count >= 32) return 16; // 16 physical cores on 32-thread CPU
    if (count >= 16) return 16;
    if (count >= 8)  return 8;
    return count > 0 ? count : 4;
}

} // namespace stream_moe