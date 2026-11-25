// Backend-agnostic device interface: kind dispatch + CPU ops (Phase 1).
// The Vulkan ops live in device_vulkan.cpp (GGML_VULKAN=ON only); this file
// must never reference a stmoe_vk_* symbol, so CPU-only builds link (B33).

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

// CPU ops: device memory IS host memory.
void * cpu_buffer_host_ptr(ggml_backend_buffer_t buffer) {
    return buffer ? ggml_backend_buffer_get_base(buffer) : nullptr;
}

void * cpu_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    void * base = cpu_buffer_host_ptr(buffer);
    return base ? static_cast<uint8_t *>(base) + off : nullptr;
}

void cpu_dma_read(void * buffer, size_t off, void * dst, size_t bytes) {
    if (!buffer || !dst || bytes == 0) return;
    const void * src = cpu_buffer_host_offset(static_cast<ggml_backend_buffer_t>(buffer), off);
    if (src) std::memcpy(dst, src, bytes);
}

} // namespace

#ifdef STREAMMOE_HAVE_VULKAN
// Implemented in device_vulkan.cpp (compiled only when GGML_VULKAN=ON).
void * stmoe_dev_vulkan_buffer_host_ptr(ggml_backend_buffer_t buffer);
void * stmoe_dev_vulkan_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);
void  stmoe_dev_vulkan_dma_read(void * buffer, size_t off, void * dst, size_t bytes);
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
            return nullptr; // Phase 2: CUDA device memory has no host map.
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
            return nullptr; // Phase 2: device address is filled by the CUDA path.
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
            return nullptr; // Phase 2.
        case STMOE_DEV_CPU:
        default:
            return cpu_dma_read;
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
            return false; // Phase 2.
        case STMOE_DEV_CPU:
        default:
            return true;
    }
}

} // namespace stream_moe
