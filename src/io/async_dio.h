#pragma once

#include "common/types.h"
#include <string>
#include <vector>
#include <memory>

namespace stream_moe {

struct dio_file_t {
    void*        raw_handle = nullptr;
    uint64_t     file_size  = 0;
    std::wstring path_w;
    std::string  path_u8;
};

struct aio_req_t;

// Platform-specific internal state hook
struct aio_platform_data_t {
    void* internal_ptr = nullptr;
    uint64_t internal_u64[4] = {0};
};

struct aio_req_t {
    dio_file_t*         file         = nullptr;
    uint64_t            file_offset  = 0;       // Must be 4KB sector aligned
    void*               aligned_buf  = nullptr; // Must be 4KB sector aligned
    uint32_t            aligned_len  = 0;       // Must be 4KB sector aligned
    void*               user_data    = nullptr; // User context / slot pointer
    uint32_t            bytes_read   = 0;       // Output: bytes read
    int32_t             error_code   = 0;       // Output: 0 = success, non-zero = error
    bool                is_completed = false;   // Output: true when completed
    aio_platform_data_t platform_data;          // Internal (e.g. OVERLAPPED on Win32)
};

// Abstract Async Direct I/O Engine
class async_dio_engine {
public:
    virtual ~async_dio_engine() = default;

    // Factory
    static std::unique_ptr<async_dio_engine> create(uint32_t max_in_flight = 1024);

    // File Management (Direct I/O, unbuffered, asynchronous)
    virtual dio_file_t* open_file(const std::string& path_u8) = 0;
    virtual dio_file_t* open_file_w(const std::wstring& path_w) = 0;
    virtual void        close_file(dio_file_t* file) = 0;

    // Batch I/O Operations
    // Submits 'count' requests asynchronously. Returns number of successfully submitted requests, or negative error code.
    virtual int32_t  submit_batch(aio_req_t* reqs, uint32_t count) = 0;

    // Waits for completed events.
    // min_complete: minimum number of events to wait for before returning (0 = non-blocking poll).
    // timeout_ms: maximum time to wait in milliseconds (0xFFFFFFFF = INFINITE).
    // Returns number of completed events placed into out_completed array (up to max_events).
    virtual uint32_t wait_events(aio_req_t** out_completed, uint32_t max_events, uint32_t min_complete, uint32_t timeout_ms) = 0;

    // Direct I/O Aligned Memory Management (VirtualAlloc on Windows, posix_memalign / mmap on Linux)
    static void* alloc_aligned(size_t size, size_t alignment = DIO_SECTOR_SIZE);
    static void  free_aligned(void* ptr);
};

// RAII helper for aligned buffers
struct aligned_buffer_deleter {
    void operator()(void* ptr) const {
        async_dio_engine::free_aligned(ptr);
    }
};

using aligned_buffer_ptr = std::unique_ptr<uint8_t, aligned_buffer_deleter>;

inline aligned_buffer_ptr make_aligned_buffer(size_t size, size_t alignment = DIO_SECTOR_SIZE) {
    void* ptr = async_dio_engine::alloc_aligned(size, alignment);
    if (!ptr && size > 0) {
        throw std::bad_alloc();
    }
    return aligned_buffer_ptr(static_cast<uint8_t*>(ptr));
}

} // namespace stream_moe