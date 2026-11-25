// Vulkan backend ops for the device interface (Phase 1 + 2).
// Compiled only when GGML_VULKAN=ON (see src/cmake/stmoe_routeb_sources.cmake):
// the stmoe_vk_* symbols only exist when the vendored ggml-vulkan build
// includes the route-B frags (STREAM_MOE_ROUTE_B anchor).
//
// DMA is synchronous here: the reader is the existing transfer-queue read and
// the writer is a memcpy into the host map (a Vulkan device buffer is
// host-visible), so every token is null (already complete).

#include "backend/device_iface.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

// Route-B extensions defined by patches/route-b/common/stmoe_routeb_vk_hostmap.frag
// and stmoe_routeb_vk_dma.frag, included at the anchor in vendored ggml-vulkan.cpp.
// Global scope: the frags define them outside any namespace.
void* stmoe_vk_buffer_host_ptr(ggml_backend_buffer_t buffer);
void* stmoe_vk_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);
void  stmoe_vk_dma_read(void* vram_buffer, size_t off, void* dst, size_t bytes);

namespace stream_moe {

void* stmoe_dev_vulkan_buffer_host_ptr(ggml_backend_buffer_t buffer) {
    return stmoe_vk_buffer_host_ptr(buffer);
}

void* stmoe_dev_vulkan_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    return stmoe_vk_buffer_host_offset(buffer, off);
}

void* stmoe_dev_vulkan_dma_read(void* buffer, size_t off, void* dst, size_t bytes) {
    stmoe_vk_dma_read(buffer, off, dst, bytes);
    return nullptr;   // sync read: complete on return
}

void* stmoe_dev_vulkan_dma_write(void* buffer, size_t off, const void* src, size_t bytes) {
    if (!buffer || !src || bytes == 0) return nullptr;
    void* base = stmoe_vk_buffer_host_ptr(static_cast<ggml_backend_buffer_t>(buffer));
    if (!base) {
        // No host map: there is nowhere to copy to. Never silently report
        // success here - a null token means "already complete" to the
        // caller, and the slot would be marked READY with no data in it.
        std::fprintf(stderr, "stmoe_vulkan_dma_write: buffer has no host map, %zu bytes at off %zu NOT written\n", bytes, off);
        return nullptr;
    }
    std::memcpy(static_cast<uint8_t*>(base) + off, src, bytes);
    return nullptr;   // host-map write: complete on return
}

bool stmoe_dev_vulkan_dma_wait(stmoe_dev_dma_token_t /* token */) { return true; }
bool stmoe_dev_vulkan_dma_poll(stmoe_dev_dma_token_t /* token */) { return true; }

} // namespace stream_moe
