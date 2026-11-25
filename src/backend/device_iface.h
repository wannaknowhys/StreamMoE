#pragma once

// Backend-agnostic device interface for route-B device pools (Phase 1).
//
// Before Phase 1 the route-B engine called the ggml-vulkan route-B extensions
// (stmoe_vk_buffer_host_ptr / stmoe_vk_buffer_host_offset / stmoe_vk_dma_read)
// directly, so a CPU-only build (GGML_VULKAN=OFF) failed to link with undefined
// symbols (BUG_TRACKER B33). All backend-specific access now goes through this
// interface: callers name a backend kind, never a `stmoe_vk_*` symbol, and the
// Vulkan implementation lives in device_vulkan.cpp, which is only compiled
// when GGML_VULKAN=ON (see src/cmake/stmoe_routeb_sources.cmake).
//
// Kind dispatch by buffer: the kind is derived from the backend device that
// owns the buffer's buffer type, so call sites only swap the function name.
// CUDA is an enum value only in Phase 1; its ops land in Phase 2.

#include "ggml-backend.h"

#include <cstddef>

namespace stream_moe {

enum stmoe_dev_kind {
    STMOE_DEV_CPU = 0,
    STMOE_DEV_VULKAN = 1,
    STMOE_DEV_CUDA = 2,
};

// Printable kind name (for logs). Never null.
const char * stmoe_dev_kind_name(stmoe_dev_kind kind);

// "Vulkan0" -> VULKAN, "CUDA0" -> CUDA, "CPU" / "RAM" / null / unknown -> CPU.
stmoe_dev_kind stmoe_dev_kind_from_name(const char * name);

// Kind of the backend device owning a buffer type / buffer (CPU when unknown).
stmoe_dev_kind stmoe_dev_kind_for_buft(ggml_backend_buffer_type_t buft);
stmoe_dev_kind stmoe_dev_kind_for_buffer(ggml_backend_buffer_t buffer);

// Host-visible base pointer of a buffer (nullptr when it has no host map).
// CPU: ggml_backend_buffer_get_base. Vulkan: stmoe_vk_buffer_host_ptr.
void * stmoe_dev_buffer_host_ptr(ggml_backend_buffer_t buffer);

// Address of byte `off` inside `buffer`, for binding a tensor's data pointer.
// CPU: base + off (a real host pointer). Vulkan: fake-base vk_ptr_base + off
// (kernels derive the offset as data - vk_ptr_base; never dereference on host).
void * stmoe_dev_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);

// DMA reader: copy `bytes` at byte `off` in a device buffer to host `dst`.
// Same signature as expert_scheduler::pool_dma_read_fn.
typedef void (*stmoe_dev_dma_read_fn_t)(void * buffer, size_t off, void * dst, size_t bytes);

// DMA reader for a kind (nullptr when the kind has no reader in this build).
// CPU: memcpy. Vulkan: transfer-queue read. CUDA: nullptr until Phase 2.
stmoe_dev_dma_read_fn_t stmoe_dev_dma_reader(stmoe_dev_kind kind);

// True when dma_read() for the kind is backed by a real device copy path.
bool stmoe_dev_dma_available(stmoe_dev_kind kind);

} // namespace stream_moe
