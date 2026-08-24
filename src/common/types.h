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