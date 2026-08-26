#pragma once

// Route B MUL_MAT_ID / MUL_MAT executor (docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4.3).
// Builds a per-expert mini-graph in the scratch arena and runs it with the CPU
// backend's stock kernels (dequant + GEMM), reading expert weights from the
// scheduler's compact slots. Model-agnostic.

#include "ggml.h"
#include "ggml-backend.h"
#include "backend/minigraph.h"
#include "backend/scheduler.h"

namespace stream_moe {

// Executes the MoE weight ops in `nodes` (all from our buft) for one
// graph_compute call. `nodes` must contain GGML_OP_MUL_MAT_ID (and eventually
// GGML_OP_MUL_MAT for shared experts). Applies the route B pin lifecycle:
//   - split containing only non-down nodes -> pin experts (no release)
//   - split containing a down node -> wait_ready, then release after compute
// Returns ggml status.
enum ggml_status moe_exec_mul_mat_id(
    ggml_context* arena_ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    const ggml_tensor* const* nodes,
    int n_nodes,
    int n_threads);

} // namespace stream_moe
