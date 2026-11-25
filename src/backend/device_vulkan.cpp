// Vulkan backend ops for the device interface (Phase 1 + 2).
// Compiled only when GGML_VULKAN=ON (see src/cmake/stmoe_routeb_sources.cmake):
// the stmoe_vk_* symbols only exist when the vendored ggml-vulkan build
// includes the route-B frags (STREAM_MOE_ROUTE_B anchor).
//
// DMA is synchronous here: the reader is the existing transfer-queue read and
// the writer is the transfer-queue write (stmoe_vk_dma_write, which memcpys
// into the host map for host-visible buffers and stages + copies for
// device-local ones), so every token is null (already complete) - except the
// failure sentinel below, which dma_wait reports as failed so a load/move
// that could not be written fails instead of marking its slot READY empty.

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
bool  stmoe_vk_dma_write(void* vram_buffer, size_t off, const void* src, size_t bytes);

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

// Sentinel token meaning "the synchronous write failed". dma_wait reports
// it as failed (false); dma_poll reports it as complete (true) so the load
// or move settles and fails instead of parking forever.
static uint8_t vk_dma_failed_token_storage;

void* stmoe_dev_vulkan_dma_write(void* buffer, size_t off, const void* src, size_t bytes) {
    if (bytes == 0) return nullptr;   // nothing to write
    if (!buffer || !src) {
        std::fprintf(stderr, "stmoe_vulkan_dma_write: null buffer/src, %zu bytes at off %zu NOT written\n", bytes, off);
        return &vk_dma_failed_token_storage;
    }
    if (!stmoe_vk_dma_write(buffer, off, src, bytes)) {
        // Invalid buffer handle (null context / no device buffer): there is
        // nowhere to copy to. Never silently report success here - a null
        // token means "already complete" to the caller, and the slot would
        // be marked READY with no data in it (the B40 bug).
        std::fprintf(stderr, "stmoe_vulkan_dma_write: invalid buffer, %zu bytes at off %zu NOT written\n", bytes, off);
        return &vk_dma_failed_token_storage;
    }
    return nullptr;   // sync write: complete on return
}

bool stmoe_dev_vulkan_dma_wait(stmoe_dev_dma_token_t token) {
    if (token == &vk_dma_failed_token_storage) return false;
    return true;
}
bool stmoe_dev_vulkan_dma_poll(stmoe_dev_dma_token_t /* token */) {
    // Sync writer: everything is complete (a failed token is done too;
    // wait reports the failure).
    return true;
}

} // namespace stream_moe
