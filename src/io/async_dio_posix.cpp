#if !defined(_WIN32)

#include "io/async_dio.h"
#include "common/logger.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <algorithm>

namespace stream_moe {

class async_dio_engine_posix : public async_dio_engine {
public:
    explicit async_dio_engine_posix(uint32_t max_in_flight)
        : max_in_flight_(max_in_flight) {}

    ~async_dio_engine_posix() override {
        std::lock_guard<std::mutex> lock(files_mutex_);
        for (auto* file : open_files_) {
            if (file && file->raw_handle) {
                int fd = reinterpret_cast<intptr_t>(file->raw_handle);
                close(fd);
                delete file;
            }
        }
        open_files_.clear();
    }

    dio_file_t* open_file(const std::string& path_u8) override {
        // Direct I/O on Linux via O_DIRECT
        int flags = O_RDONLY;
#if defined(__linux__) && defined(O_DIRECT)
        flags |= O_DIRECT;
#endif
        int fd = open(path_u8.c_str(), flags);
        if (fd < 0) {
            LOG_ERROR("open failed for " << path_u8 << ", errno: " << errno);
            return nullptr;
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            return nullptr;
        }

        auto* file = new dio_file_t();
        file->raw_handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
        file->file_size  = static_cast<uint64_t>(st.st_size);
        file->path_u8    = path_u8;

        {
            std::lock_guard<std::mutex> lock(files_mutex_);
            open_files_.push_back(file);
        }
        return file;
    }

    dio_file_t* open_file_w(const std::wstring& path_w) override {
        // Not used on POSIX
        return nullptr;
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
            int fd = reinterpret_cast<intptr_t>(file->raw_handle);
            close(fd);
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
                req.error_code   = EBADF;
                req.is_completed = true;
                continue;
            }

            int fd = reinterpret_cast<intptr_t>(req.file->raw_handle);
            ssize_t n = pread(fd, req.aligned_buf, req.aligned_len, req.file_offset);
            if (n >= 0) {
                req.bytes_read   = static_cast<uint32_t>(n);
                req.error_code   = 0;
                req.is_completed = true;
                submitted++;
                completed_queue_.push_back(&req);
            } else {
                req.error_code   = errno;
                req.is_completed = true;
            }
        }
        return submitted;
    }

    uint32_t wait_events(aio_req_t** out_completed, uint32_t max_events, uint32_t min_complete, uint32_t timeout_ms) override {
        uint32_t n = 0;
        while (n < max_events && !completed_queue_.empty()) {
            out_completed[n++] = completed_queue_.front();
            completed_queue_.erase(completed_queue_.begin());
        }
        return n;
    }

private:
    uint32_t                 max_in_flight_ = 1024;
    std::mutex               files_mutex_;
    std::vector<dio_file_t*> open_files_;
    std::vector<aio_req_t*>  completed_queue_;
};

void* async_dio_engine::alloc_aligned(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, align_ceil(size, alignment)) != 0) {
        return nullptr;
    }
    return ptr;
}

void async_dio_engine::free_aligned(void* ptr) {
    free(ptr);
}

std::unique_ptr<async_dio_engine> async_dio_engine::create(uint32_t max_in_flight) {
    return std::make_unique<async_dio_engine_posix>(max_in_flight);
}

} // namespace stream_moe

#endif // !_WIN32