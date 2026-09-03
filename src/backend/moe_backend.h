#pragma once

// Route B custom ggml backend (docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4, docs/Backend.md §1).
// Model-agnostic: any MoE whose expert weights are named `ffn_*_exps` / `ffn_*_shexp`
// (standard llama.cpp MoE tensor schema) can be routed here via tensor_buft_overrides.
// Dense tensors are never touched - they keep llama.cpp defaults.
//
// Skeleton status: device/buft registration + support queries are real;
// graph_compute currently rejects execution (GGML_STATUS_FAILED) until the
// per-expert mini-graph delegation lands (next milestone). Enable only behind
// an explicit opt-in flag - mainline behavior is unchanged when disabled.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

namespace stream_moe {

// The weight buft that `tensor_buft_overrides` points MoE expert tensors to.
// alloc_buffer returns a lightweight accounting handle (size = declared bytes,
// dummy non-null base, no-op set/get/memset) - no physical allocation.
//
// Multi-model (docs/MULTI_MODEL_POOL.md): each model pool gets its OWN expert
// buft via stream_moe_create_expert_buft(), so the backend can tell pools apart
// by buft identity (main model buft vs draft model buft). No single shared
// expert buft anymore.
ggml_backend_buffer_type_t stream_moe_create_expert_buft();

// Dense weight buft for v2-chunk strip models: a real malloc-backed host buft
// with per-pool identity (like the expert buft) so llama.cpp's override skips
// its tensor_info read and route_b fills the tensor from dense.srcs segments.
ggml_backend_buffer_type_t stream_moe_create_dense_buft();

// Default (compute) buffer type for this backend: plain host memory for the
// node outputs (cur/ids/dst) that other backends consume across splits.
ggml_backend_buffer_type_t stream_moe_compute_buft();

// Register the "stream_moe" backend/device with ggml's registry. Idempotent.
// Must be called before llama_model_load_from_file so the device lands in
// model.devices and the scheduler picks it up for MoE ops.
void stream_moe_register_backend();

// Buffer type accessor (valid after stream_moe_register_backend).
ggml_backend_buffer_type_t stream_moe_register_backend_helper_compute_buft();

// Bind one model's expert pool scheduler to its expert buft. graph_compute
// looks up the scheduler by the weight buft of the graph's MoE nodes.
void stream_moe_backend_bind_scheduler(ggml_backend_buffer_type_t buft, class expert_scheduler* sched);
void stream_moe_backend_set_threads(int threads);

// ---- M2 device-exec resources (per device pool) --------------------------
// One exec context per device pool (pool index 1 = first device region /
// Vulkan0). route_b registers the backend + buffer types once its vram pool is
// allocated; the executor lazily ensures the arena (chain intermediates +
// per-expert contribution) and staging (cur/ids upload) buffers before a build.
struct device_exec_ctx_t {
    ggml_backend_t             be         = nullptr;
    ggml_backend_buffer_type_t arena_buft = nullptr;  // device compute buft
    ggml_backend_buffer_type_t stage_buft = nullptr;  // host-visible staging buft
    ggml_backend_buffer_t      arena      = nullptr;
    size_t                     arena_cap  = 0;
    uint8_t*                   arena_map  = nullptr;
    ggml_backend_buffer_t      stage      = nullptr;
    size_t                     stage_cap  = 0;
    uint8_t*                   stage_map  = nullptr;
};
// Register pool `pool`'s device exec backend. Idempotent per pool.
void stream_moe_backend_bind_device_exec(uint32_t pool, ggml_backend_t be,
                                         ggml_backend_buffer_type_t arena_buft,
                                         ggml_backend_buffer_type_t stage_buft);
// Exec context of device pool `pool`, or nullptr when unregistered.
device_exec_ctx_t* stream_moe_backend_device_exec(uint32_t pool);
// Lazily (re)allocate the arena / staging buffers to cover the requested bytes.
// Returns false on allocation failure. No-op sizes keep existing buffers.
bool stream_moe_backend_device_ensure(uint32_t pool, size_t arena_bytes, size_t stage_bytes);

} // namespace stream_moe
