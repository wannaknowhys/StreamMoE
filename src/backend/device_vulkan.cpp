// Vulkan backend ops for the device interface (Phase 1).
// Compiled only when GGML_VULKAN=ON (see src/cmake/stmoe_routeb_sources.cmake):
// the stmoe_vk_* symbols only exist when the vendored ggml-vulkan build
// includes the route-B frags (STREAM_MOE_ROUTE_B anchor).

#include "backend/device_iface.h"

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

void stmoe_dev_vulkan_dma_read(void* buffer, size_t off, void* dst, size_t bytes) {
    stmoe_vk_dma_read(buffer, off, dst, bytes);
}

} // namespace stream_moe
