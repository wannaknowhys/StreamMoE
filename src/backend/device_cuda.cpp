// CUDA backend ops for the device interface (Phase 2).
// Compiled only when GGML_CUDA=ON (see src/cmake/stmoe_routeb_sources.cmake).
// Plain .cpp linked against cudart; no .cu file is needed because this file
// only calls the CUDA runtime API (no kernels).
//
// Async DMA: cudaMemcpyAsync on one lazily-created stream per device +
// cudaEventRecord; the token is a heap record holding the event (plus the
// host range this transfer pinned, when it pinned one). The enqueue functions
// never call a synchronizing CUDA API: completion is observed via
// cudaEventQuery (poll) or cudaEventSynchronize (wait, which also destroys
// the event and unpins the range). On an enqueue error the copy falls back
// to a synchronous cudaMemcpy and returns a null token (already complete);
// the error is logged, never swallowed silently.
//
// Host memory must be pinned for cudaMemcpyAsync to be async: with a pageable
// host endpoint the driver completes the copy inside the call, so the
// "async" enqueue would block for the full transfer time (measured: a 512 MiB
// DtoH enqueue took 131 ms pageable vs 11 us pinned). Each transfer therefore
// pins its host range with cudaHostRegister before enqueueing and unpins it in
// the waiter. A range that is already registered (for example the scheduler
// staging ringbuffer, or a caller-pinned probe buffer) is used as-is and is
// NOT unpinned by the waiter. Pinning is cheap for the real per-slice sizes
// (a 4 MiB range registers in ~0.3 ms) but costs ~18 ms for a 512 MiB range,
// so callers that time the enqueue must pin up front.
//
// Device selection: the DMA signatures carry no device index, so the device is
// derived per call from the device pointer via cudaPointerGetAttributes, then
// cudaSetDevice (per-thread) before the copy.

#include "backend/device_iface.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace stream_moe {

namespace {

// One DMA stream per CUDA device, created lazily under a mutex: both the move
// worker thread and the scheduler thread enqueue through here.
cudaStream_t cuda_dma_stream(int device) {
    static std::mutex mtx;
    static std::unordered_map<int, cudaStream_t> streams;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = streams.find(device);
    if (it != streams.end()) return it->second;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] cudaStreamCreate failed for device %d: %s\n",
                     device, cudaGetErrorString(cudaGetLastError()));
        return nullptr;
    }
    streams.emplace(device, stream);
    return stream;
}

// Device owning `ptr` (-1 when the pointer is not a CUDA device pointer or
// the query fails). Clears any pending CUDA error on the failure path.
int cuda_device_of_ptr(const void* ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return attr.device;
}

// Completion token: the event plus the host range this transfer pinned (if it
// pinned one). The waiter unpins the range after the event completes; the
// poller only queries the event and never touches the range. A null host_ptr
// means the range was already registered by someone else (used as-is, never
// unpinned here) or the copy ran synchronously (no token is returned then).
struct cuda_dma_token {
    cudaEvent_t event;
    void* host_ptr;
    size_t bytes;
};

// Enqueue one copy between a device buffer (base + off) and host memory.
// dtoh = true: device -> host (reader). false: host -> device (writer).
// Returns a completion token, or nullptr when the copy is already complete
// (zero bytes, or a synchronous fallback after an enqueue error).
void* cuda_dma_enqueue(void* buffer, size_t off, void* host_ptr, size_t bytes, bool dtoh) {
    if (!buffer || !host_ptr || bytes == 0) return nullptr;
    void* base = ggml_backend_buffer_get_base(static_cast<ggml_backend_buffer_t>(buffer));
    if (!base) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA: null device base\n");
        return nullptr;
    }
    void* dev_ptr = static_cast<uint8_t*>(base) + off;
    const int device = cuda_device_of_ptr(dev_ptr);
    if (device < 0) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA: cannot resolve device for %p\n", dev_ptr);
        return nullptr;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA: cudaSetDevice(%d) failed: %s\n",
                     device, cudaGetErrorString(cudaGetLastError()));
        return nullptr;
    }
    // Synchronous fallback (already complete on return). Used when no stream
    // exists or an enqueue step fails.
    auto sync_copy = [&]() {
        const cudaError_t err = dtoh ? cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)
                                     : cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[stream_moe/cuda] DMA sync fallback failed: %s\n",
                         cudaGetErrorString(err));
        }
    };
    cudaStream_t stream = cuda_dma_stream(device);
    if (!stream) {
        sync_copy();
        return nullptr;
    }
    // Pin the host range so the async copy is genuinely async. A range that
    // is already registered is used as-is (registered_by_us stays false, so
    // the waiter will not unpin memory it did not pin). Any other
    // registration error falls back to a synchronous copy.
    bool registered_by_us = false;
    const cudaError_t reg_err = cudaHostRegister(host_ptr, bytes, cudaHostRegisterDefault);
    if (reg_err == cudaSuccess) {
        registered_by_us = true;
    } else if (reg_err != cudaErrorHostMemoryAlreadyRegistered) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA cudaHostRegister failed: %s\n",
                     cudaGetErrorString(reg_err));
        sync_copy();
        return nullptr;
    }
    cudaError_t err = dtoh ? cudaMemcpyAsync(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost, stream)
                           : cudaMemcpyAsync(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA cudaMemcpyAsync failed: %s\n",
                     cudaGetErrorString(err));
        sync_copy();
        if (registered_by_us) cudaHostUnregister(host_ptr);
        return nullptr;
    }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA cudaEventCreate failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        // The copy is already queued; wait for it here so the caller does not
        // observe a completed token while the copy is still in flight. This
        // is an error path only, not the normal enqueue path.
        cudaStreamSynchronize(stream);
        if (registered_by_us) cudaHostUnregister(host_ptr);
        return nullptr;
    }
    if (cudaEventRecord(ev, stream) != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA cudaEventRecord failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        cudaEventDestroy(ev);
        cudaStreamSynchronize(stream);
        if (registered_by_us) cudaHostUnregister(host_ptr);
        return nullptr;
    }
    cuda_dma_token* tok = new cuda_dma_token{ev, registered_by_us ? host_ptr : nullptr,
                                             registered_by_us ? bytes : 0};
    return static_cast<void*>(tok);
}

} // namespace

void* stmoe_dev_cuda_buffer_host_ptr(ggml_backend_buffer_t /* buffer */) {
    // CUDA device memory has no host map.
    return nullptr;
}

void* stmoe_dev_cuda_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    // Device address (kernel use only; host must never dereference it).
    // ggml_backend_buffer_get_base on a CUDA buffer is the cudaMalloc device
    // pointer (cudaMallocManaged only when GGML_CUDA_ENABLE_UNIFIED_MEMORY is
    // set, which it is not on this host).
    if (!buffer) return nullptr;
    void* base = ggml_backend_buffer_get_base(buffer);
    return base ? static_cast<uint8_t*>(base) + off : nullptr;
}

void* stmoe_dev_cuda_dma_read(void* buffer, size_t off, void* dst, size_t bytes) {
    return cuda_dma_enqueue(buffer, off, dst, bytes, /* dtoh = */ true);
}

void* stmoe_dev_cuda_dma_write(void* buffer, size_t off, const void* src, size_t bytes) {
    return cuda_dma_enqueue(buffer, off, const_cast<void*>(src), bytes, /* dtoh = */ false);
}

bool stmoe_dev_cuda_dma_wait(stmoe_dev_dma_token_t token) {
    if (!token) return true;
    cuda_dma_token* tok = static_cast<cuda_dma_token*>(token);
    const cudaError_t err = cudaEventSynchronize(tok->event);
    cudaEventDestroy(tok->event);
    if (tok->host_ptr) {
        const cudaError_t uerr = cudaHostUnregister(tok->host_ptr);
        if (uerr != cudaSuccess) {
            std::fprintf(stderr, "[stream_moe/cuda] DMA cudaHostUnregister failed: %s\n",
                         cudaGetErrorString(uerr));
        }
    }
    delete tok;
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[stream_moe/cuda] DMA wait failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool stmoe_dev_cuda_dma_poll(stmoe_dev_dma_token_t token) {
    if (!token) return true;
    cudaEvent_t ev = static_cast<cuda_dma_token*>(token)->event;
    const cudaError_t err = cudaEventQuery(ev);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    // Any other query error: the event is unusable. Log loudly and report
    // complete so a load never stalls forever on a dead token (the underlying
    // CUDA error is already visible in the log).
    std::fprintf(stderr, "[stream_moe/cuda] DMA poll error: %s\n", cudaGetErrorString(err));
    return true;
}

} // namespace stream_moe
