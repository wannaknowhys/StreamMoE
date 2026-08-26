#pragma once

// Cross-platform aligned allocation helpers (Windows/Linux/other).
// std::aligned_alloc is unavailable under MSVC/clang-win, so use the
// platform-native paths with a portable interface.

#include <cstddef>
#include <cstdlib>

namespace stream_moe {

inline void* aligned_alloc_ptr(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0) return nullptr;
    return p;
#endif
}

inline void aligned_free_ptr(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace stream_moe
