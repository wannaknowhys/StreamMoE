#pragma once

// Backend-agnostic device interface for route-B device pools (Phase 1 + 2).
//
// Before Phase 1 the route-B engine called the ggml-vulkan route-B extensions
// (stmoe_vk_buffer_host_ptr / stmoe_vk_buffer_host_offset / stmoe_vk_dma_read)
// directly, so a CPU-only build (GGML_VULKAN=OFF) failed to link with undefined
// symbols (BUG_TRACKER B33). All backend-specific access now goes through this
// interface: callers name a backend kind, never a `stmoe_vk_*` symbol, and the
// Vulkan implementation lives in device_vulkan.cpp, which is only compiled
// when GGML_VULKAN=ON (see src/cmake/stmoe_routeb_sources.cmake). The CUDA
// implementation lives in device_cuda.cpp, compiled only when GGML_CUDA=ON
// (Phase 2).
//
// Kind dispatch by buffer: the kind is derived from the backend device that
// owns the buffer's buffer type, so call sites only swap the function name.
//
// Async DMA (Phase 2): the DMA reader/writer enqueue a copy and return an
// opaque completion token (nullptr = already complete, e.g. CPU memcpy).
// The enqueue functions NEVER block: no cudaStreamSynchronize /
// cudaDeviceSynchronize / ggml_vk_buffer_read inside them. The caller observes
// completion through the wait function (blocking reap: waits, then destroys
// the token) or the poll function (non-blocking check; the token stays valid
// until wait reaps it). The destination of a read / the source of a write must
// not be touched until the token reports complete.

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
// CUDA: nullptr (device memory has no host map).
void * stmoe_dev_buffer_host_ptr(ggml_backend_buffer_t buffer);

// Address of byte `off` inside `buffer`, for binding a tensor's data pointer.
// CPU: base + off (a real host pointer). Vulkan: fake-base vk_ptr_base + off
// (kernels derive the offset as data - vk_ptr_base; never dereference on host).
// CUDA: device base + off (a device address; host must never dereference it).
void * stmoe_dev_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);

// Async DMA completion token. nullptr = the copy is already complete.
typedef void * stmoe_dev_dma_token_t;

// DMA reader: enqueue a copy of `bytes` at byte `off` in a device buffer to
// host `dst`, and return its completion token (nullptr = already complete).
// Same shape as expert_scheduler::pool_dma_read_fn.
typedef stmoe_dev_dma_token_t (*stmoe_dev_dma_read_fn_t)(void * buffer, size_t off, void * dst, size_t bytes);

// DMA writer: enqueue a copy of `bytes` from host `src` to byte `off` in a
// device buffer, and return its completion token (nullptr = already complete).
typedef stmoe_dev_dma_token_t (*stmoe_dev_dma_write_fn_t)(void * buffer, size_t off, const void * src, size_t bytes);

// DMA completion: block until `token` completes, then destroy it. A null
// token is already complete (returns true). Returns false on a copy error.
typedef bool (*stmoe_dev_dma_wait_fn_t)(stmoe_dev_dma_token_t token);

// DMA completion poll: true when `token` is complete (or null); the token
// stays valid until the wait function reaps it. Never blocks.
typedef bool (*stmoe_dev_dma_poll_fn_t)(stmoe_dev_dma_token_t token);

// DMA reader / writer / waiter / poller for a kind (nullptr when the kind has
// no implementation in this build).
// CPU: memcpy (tokens always null). Vulkan: transfer-queue read / host-map
// write (tokens always null; the sync read is the existing stmoe_vk_dma_read).
// CUDA: cudaMemcpyAsync + cudaEventRecord (Phase 2).
stmoe_dev_dma_read_fn_t  stmoe_dev_dma_reader(stmoe_dev_kind kind);
stmoe_dev_dma_write_fn_t stmoe_dev_dma_writer(stmoe_dev_kind kind);
stmoe_dev_dma_wait_fn_t  stmoe_dev_dma_waiter(stmoe_dev_kind kind);
stmoe_dev_dma_poll_fn_t  stmoe_dev_dma_poller(stmoe_dev_kind kind);

// True when the kind has a real device copy path in this build.
bool stmoe_dev_dma_available(stmoe_dev_kind kind);

} // namespace stream_moe
