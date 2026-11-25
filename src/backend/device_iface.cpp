// Backend-agnostic device interface: kind dispatch + CPU ops (Phase 1 + 2).
// The Vulkan ops live in device_vulkan.cpp (GGML_VULKAN=ON only) and the CUDA
// ops in device_cuda.cpp (GGML_CUDA=ON only); this file must never reference
// a stmoe_vk_* symbol or a CUDA runtime symbol, so CPU-only builds link (B33).

#include "backend/device_iface.h"

#include <cstdint>
#include <cstring>

namespace stream_moe {

const char * stmoe_dev_kind_name(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN: return "Vulkan";
        case STMOE_DEV_CUDA:   return "CUDA";
        case STMOE_DEV_CPU:
        default:               return "CPU";
    }
}

stmoe_dev_kind stmoe_dev_kind_from_name(const char * name) {
    if (!name) return STMOE_DEV_CPU;
    if (std::strncmp(name, "Vulkan", 6) == 0) return STMOE_DEV_VULKAN;
    if (std::strncmp(name, "CUDA", 4) == 0)   return STMOE_DEV_CUDA;
    return STMOE_DEV_CPU;
}

stmoe_dev_kind stmoe_dev_kind_for_buft(ggml_backend_buffer_type_t buft) {
    if (!buft) return STMOE_DEV_CPU;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) return STMOE_DEV_CPU;
    return stmoe_dev_kind_from_name(ggml_backend_dev_name(dev));
}

stmoe_dev_kind stmoe_dev_kind_for_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer) return STMOE_DEV_CPU;
    return stmoe_dev_kind_for_buft(ggml_backend_buffer_get_type(buffer));
}

namespace {

// CPU ops: device memory IS host memory, so DMA is a synchronous memcpy and
// every token is null (already complete).
void * cpu_buffer_host_ptr(ggml_backend_buffer_t buffer) {
    return buffer ? ggml_backend_buffer_get_base(buffer) : nullptr;
}

void * cpu_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    void * base = cpu_buffer_host_ptr(buffer);
    return base ? static_cast<uint8_t *>(base) + off : nullptr;
}

void * cpu_dma_read(void * buffer, size_t off, void * dst, size_t bytes) {
    if (!buffer || !dst || bytes == 0) return nullptr;
    const void * src = cpu_buffer_host_offset(static_cast<ggml_backend_buffer_t>(buffer), off);
    if (src) std::memcpy(dst, src, bytes);
    return nullptr;   // complete
}

void * cpu_dma_write(void * buffer, size_t off, const void * src, size_t bytes) {
    if (!buffer || !src || bytes == 0) return nullptr;
    void * dst = cpu_buffer_host_offset(static_cast<ggml_backend_buffer_t>(buffer), off);
    if (dst) std::memcpy(dst, src, bytes);
    return nullptr;   // complete
}

bool cpu_dma_wait(stmoe_dev_dma_token_t /* token */) { return true; }
bool cpu_dma_poll(stmoe_dev_dma_token_t /* token */) { return true; }

} // namespace

#ifdef STREAMMOE_HAVE_VULKAN
// Implemented in device_vulkan.cpp (compiled only when GGML_VULKAN=ON).
void * stmoe_dev_vulkan_buffer_host_ptr(ggml_backend_buffer_t buffer);
void * stmoe_dev_vulkan_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);
void * stmoe_dev_vulkan_dma_read(void * buffer, size_t off, void * dst, size_t bytes);
void * stmoe_dev_vulkan_dma_write(void * buffer, size_t off, const void * src, size_t bytes);
bool   stmoe_dev_vulkan_dma_wait(stmoe_dev_dma_token_t token);
bool   stmoe_dev_vulkan_dma_poll(stmoe_dev_dma_token_t token);
#endif

#ifdef STREAMMOE_HAVE_CUDA
// Implemented in device_cuda.cpp (compiled only when GGML_CUDA=ON).
void * stmoe_dev_cuda_buffer_host_ptr(ggml_backend_buffer_t buffer);
void * stmoe_dev_cuda_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);
void * stmoe_dev_cuda_dma_read(void * buffer, size_t off, void * dst, size_t bytes);
void * stmoe_dev_cuda_dma_write(void * buffer, size_t off, const void * src, size_t bytes);
bool   stmoe_dev_cuda_dma_wait(stmoe_dev_dma_token_t token);
bool   stmoe_dev_cuda_dma_poll(stmoe_dev_dma_token_t token);
#endif

void * stmoe_dev_buffer_host_ptr(ggml_backend_buffer_t buffer) {
    switch (stmoe_dev_kind_for_buffer(buffer)) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_buffer_host_ptr(buffer);
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_buffer_host_ptr(buffer);
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_buffer_host_ptr(buffer);
    }
}

void * stmoe_dev_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    switch (stmoe_dev_kind_for_buffer(buffer)) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_buffer_host_offset(buffer, off);
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_buffer_host_offset(buffer, off);
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_buffer_host_offset(buffer, off);
    }
}

stmoe_dev_dma_read_fn_t stmoe_dev_dma_reader(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_dma_read;
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_dma_read;
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_dma_read;
    }
}

stmoe_dev_dma_write_fn_t stmoe_dev_dma_writer(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_dma_write;
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_dma_write;
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_dma_write;
    }
}

stmoe_dev_dma_wait_fn_t stmoe_dev_dma_waiter(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_dma_wait;
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_dma_wait;
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_dma_wait;
    }
}

stmoe_dev_dma_poll_fn_t stmoe_dev_dma_poller(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return stmoe_dev_vulkan_dma_poll;
#else
            return nullptr; // No Vulkan in this build (CPU-only tag).
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return stmoe_dev_cuda_dma_poll;
#else
            return nullptr; // No CUDA in this build.
#endif
        case STMOE_DEV_CPU:
        default:
            return cpu_dma_poll;
    }
}

bool stmoe_dev_dma_available(stmoe_dev_kind kind) {
    switch (kind) {
        case STMOE_DEV_VULKAN:
#ifdef STREAMMOE_HAVE_VULKAN
            return true;
#else
            return false;
#endif
        case STMOE_DEV_CUDA:
#ifdef STREAMMOE_HAVE_CUDA
            return true;
#else
            return false;
#endif
        case STMOE_DEV_CPU:
        default:
            return true;
    }
}

} // namespace stream_moe
