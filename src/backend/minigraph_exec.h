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

// Executes the privatised layer nodes of `cgraph` (all from our buft) for one
// graph_compute call. The dense head/tail runs as a range of the ORIGINAL graph
// nodes (ggml_graph_view) - no clone graph. Applies the route B pin lifecycle.
// Returns ggml status.
enum ggml_status moe_exec_mul_mat_id(
    ggml_cgraph* cgraph,
    ggml_context* arena_ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    int n_threads);

} // namespace stream_moe
