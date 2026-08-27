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

#include "io/async_dio.h"
#include "common/logger.h"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cassert>

namespace stream_moe {

namespace {

std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring result(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &result[0], len);
    return result;
}

std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string result(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), &result[0], len, nullptr, nullptr);
    return result;
}

struct win_overlapped_ctx {
    OVERLAPPED  ov;
    aio_req_t*  req;
};

static_assert(sizeof(win_overlapped_ctx) <= sizeof(aio_platform_data_t), "aio_platform_data_t size is too small for win_overlapped_ctx");

} // namespace

class async_dio_engine_win : public async_dio_engine {
public:
    explicit async_dio_engine_win(uint32_t max_in_flight)
        : max_in_flight_(max_in_flight) {
        iocp_handle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
        if (!iocp_handle_) {
            DWORD err = GetLastError();
            throw std::runtime_error("Failed to create IOCP handle, Win32 error: " + std::to_string(err));
        }
    }

    ~async_dio_engine_win() override {
        if (iocp_handle_) {
            CloseHandle(iocp_handle_);
            iocp_handle_ = nullptr;
        }

        std::lock_guard<std::mutex> lock(files_mutex_);
        for (auto* file : open_files_) {
            if (file && file->raw_handle) {
                CloseHandle(static_cast<HANDLE>(file->raw_handle));
                delete file;
            }
        }
        open_files_.clear();
    }

    dio_file_t* open_file(const std::string& path_u8) override {
        std::wstring path_w = utf8_to_wstring(path_u8);
        return open_file_w(path_w);
    }

    dio_file_t* open_file_w(const std::wstring& path_w) override {
        // Open file with NO_BUFFERING (Direct I/O) and OVERLAPPED (Async I/O)
        HANDLE hFile = CreateFileW(
            path_w.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
            NULL
        );

        if (hFile == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            LOG_ERROR("CreateFileW failed for " << wstring_to_utf8(path_w) << ", err: " << err);
            return nullptr;
        }

        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) {
            DWORD err = GetLastError();
            LOG_ERROR("GetFileSizeEx failed for " << wstring_to_utf8(path_w) << ", err: " << err);
            CloseHandle(hFile);
            return nullptr;
        }

        // Associate with IOCP
        HANDLE hIocp = CreateIoCompletionPort(hFile, iocp_handle_, reinterpret_cast<ULONG_PTR>(hFile), 0);
        if (!hIocp) {
            DWORD err = GetLastError();
            LOG_ERROR("CreateIoCompletionPort association failed, err: " << err);
            CloseHandle(hFile);
            return nullptr;
        }

        auto* file = new dio_file_t();
        file->raw_handle = hFile;
        file->file_size  = static_cast<uint64_t>(size.QuadPart);
        file->path_w     = path_w;
        file->path_u8    = wstring_to_utf8(path_w);

        {
            std::lock_guard<std::mutex> lock(files_mutex_);
            open_files_.push_back(file);
        }

        return file;
    }

    void close_file(dio_file_t* file) override {
        if (!file) return;

        {
            std::lock_guard<std::mutex> lock(files_mutex_);
            auto it = std::find(open_files_.begin(), open_files_.end(), file);
            if (it != open_files_.end()) {
                open_files_.erase(it);
            }
        }

        if (file->raw_handle) {
            CloseHandle(static_cast<HANDLE>(file->raw_handle));
            file->raw_handle = nullptr;
        }
        delete file;
    }

    int32_t submit_batch(aio_req_t* reqs, uint32_t count) override {
        if (!reqs || count == 0) return 0;

        int32_t submitted = 0;
        for (uint32_t i = 0; i < count; ++i) {
            aio_req_t& req = reqs[i];
            req.is_completed = false;
            req.bytes_read   = 0;
            req.error_code   = 0;

            if (!req.file || !req.file->raw_handle) {
                req.error_code = ERROR_INVALID_HANDLE;
                req.is_completed = true;
                continue;
            }

            if (!is_aligned(req.file_offset, DIO_SECTOR_SIZE) ||
                !is_aligned(req.aligned_buf, DIO_SECTOR_SIZE) ||
                !is_aligned(req.aligned_len, DIO_SECTOR_SIZE)) {
                LOG_ERROR("submit_batch: unaligned request! offset=" << req.file_offset 
                          << ", buf=" << req.aligned_buf << ", len=" << req.aligned_len);
                req.error_code = ERROR_INVALID_PARAMETER;
                req.is_completed = true;
                continue;
            }

            if (req.aligned_len == 0) {
                req.is_completed = true;
                req.error_code   = 0;
                submitted++;
                continue;
            }

            // Setup OVERLAPPED context in platform_data
            auto* ctx = reinterpret_cast<win_overlapped_ctx*>(&req.platform_data);
            std::memset(&ctx->ov, 0, sizeof(OVERLAPPED));
            ctx->ov.Offset     = static_cast<DWORD>(req.file_offset & 0xFFFFFFFF);
            ctx->ov.OffsetHigh = static_cast<DWORD>((req.file_offset >> 32) & 0xFFFFFFFF);
            ctx->req           = &req;

            HANDLE hFile = static_cast<HANDLE>(req.file->raw_handle);
            BOOL ok = ReadFile(hFile, req.aligned_buf, req.aligned_len, NULL, &ctx->ov);
            if (!ok) {
                DWORD err = GetLastError();
                if (err == ERROR_IO_PENDING) {
                    submitted++;
                } else if (err == ERROR_HANDLE_EOF) {
                    req.bytes_read   = 0;
                    req.error_code   = 0;
                    req.is_completed = true;
                    submitted++;
                } else {
                    LOG_ERROR("ReadFile failed immediately, err: " << err
                              << " buf=" << req.aligned_buf << " len=" << req.aligned_len
                              << " off=" << req.file_offset);
                    req.error_code   = static_cast<int32_t>(err);
                    req.is_completed = true;
                }
            } else {
                // Succeeded synchronously, IOCP will still receive the completion entry
                submitted++;
            }
        }

        return submitted;
    }

    uint32_t wait_events(aio_req_t** out_completed, uint32_t max_events, uint32_t min_complete, uint32_t timeout_ms) override {
        if (!out_completed || max_events == 0) return 0;

        uint32_t total_completed = 0;
        const uint32_t batch_cap = std::min<uint32_t>(max_events, 256);
        std::vector<OVERLAPPED_ENTRY> entries(batch_cap);

        auto start_time = std::chrono::steady_clock::now();

        while (total_completed < max_events) {
            ULONG removed = 0;
            DWORD cur_timeout = timeout_ms;

            if (timeout_ms != INFINITE && timeout_ms > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                if (elapsed >= timeout_ms) {
                    break;
                }
                cur_timeout = static_cast<DWORD>(timeout_ms - elapsed);
            }

            ULONG to_fetch = std::min<ULONG>(batch_cap, max_events - total_completed);
            BOOL ok = GetQueuedCompletionStatusEx(
                iocp_handle_,
                entries.data(),
                to_fetch,
                &removed,
                cur_timeout,
                FALSE
            );

            if (ok && removed > 0) {
                for (ULONG i = 0; i < removed; ++i) {
                    const auto& entry = entries[i];
                    auto* ctx = reinterpret_cast<win_overlapped_ctx*>(entry.lpOverlapped);
                    if (ctx && ctx->req) {
                        aio_req_t* req = ctx->req;
                        req->bytes_read = entry.dwNumberOfBytesTransferred;
                        req->error_code = (entry.Internal == 0) ? 0 : static_cast<int32_t>(entry.Internal);
                        req->is_completed = true;
                        out_completed[total_completed++] = req;
                    }
                }
            } else {
                DWORD err = GetLastError();
                if (err == WAIT_TIMEOUT) {
                    break;
                }
                // Other IOCP errors
                break;
            }

            if (total_completed >= min_complete) {
                break;
            }
        }

        return total_completed;
    }

private:
    [[maybe_unused]] uint32_t max_in_flight_ = 1024;
    HANDLE                    iocp_handle_   = nullptr;
    std::mutex                files_mutex_;
    std::vector<dio_file_t*>  open_files_;
};

// Aligned Memory Allocation via VirtualAlloc
void* async_dio_engine::alloc_aligned(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    // VirtualAlloc naturally allocates on 64KB granularity and 4KB page boundaries
    size_t alloc_size = align_ceil(size, alignment);
    void* ptr = VirtualAlloc(NULL, alloc_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return ptr;
}

void async_dio_engine::free_aligned(void* ptr) {
    if (ptr) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}

std::unique_ptr<async_dio_engine> async_dio_engine::create(uint32_t max_in_flight) {
    return std::make_unique<async_dio_engine_win>(max_in_flight);
}

} // namespace stream_moe

#endif // _WIN32