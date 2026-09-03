#pragma once

#include "ggml-backend.h"

// Route B MoE-chain privatisation guard (M1/G3, docs/ROUTE_B_GPU_PHASE.md §3.5).
// Before the executor hides MoE-chain intermediates (results in our private
// arena, main-graph dst left uncomputed) we must prove no node OUTSIDE the
// layer's MoE chain consumes them. That relation is topology-only (arch/layers),
// shape-independent - an architectural conclusion, so it is verified once and
// the verdict cached by model-topology signature.
//
// Call after each model.build_graph() site (llama-context.cpp). Fail-fast: any
// external consumer logs full context and the process exits (no silent
// fallback). Chain/gating/verdict logic lives here, llama-context only calls in.

struct ggml_cgraph;
struct ggml_tensor;

namespace stream_moe {

// Single chain-node predicate, shared by supports_op (collection), verify and
// the executor traversal (docs/ROUTE_B_GPU_PHASE.md §3.5). True for nodes of the
// MoE expert chain that we take over: the routed expert MUL_MAT_IDs, the hidden
// (privatised) chain intermediates (named ffn_moe_* minus gating/output), and
// the output end (ffn_moe_out - written to the main dst, not hidden).
bool moe_chain_node_is_privatizable(const ggml_tensor * node);

// Explicit collection (docs/ROUTE_B_GPU_PHASE.md §3.5): right after build_graph
// and BEFORE the scheduler splits, pin every privatizable chain COMPUTE node to
// our backend via ggml_backend_sched_set_tensor_backend (sched pass1 respects
// user assignments) so the whole chain lands as one split in our graph_compute.
// View/layout nodes are skipped - pass4 follows view_src automatically.
bool moe_chain_assign_backend(struct ggml_cgraph * gf, ggml_backend_sched_t sched,
                              ggml_backend_t our_backend);

// Ping-pong private intermediate buffer (M1 privatisation). Two persistent
// aligned buffers (parity 0/1) shared across graph_compute calls - the whole
// MoE chain writes its hidden intermediates here instead of the main-graph dst
// (odd/even compute nodes alternate buffers, so a write never clobbers an
// input still being read one step away). Grows on demand; process-lifetime.
void * moe_chain_pingpong_buffer(int parity, size_t need_bytes);

// Full-allocation fallback mode (used when verify finds a long-range dependency
// that ping-pong would clobber): ONE fixed buffer sized to a single layer's
// hidden-intermediate sum, allocated once (never re-grown - a grow would
// invalidate already-pointed nd->data). Each hidden compute node gets its own
// byte range inside it; the executor tracks per-layer offsets and resets them
// at the layer end (moe_out). ping_pong_ok() = false in this mode.
bool   moe_chain_pingpong_ok();
void   moe_chain_set_full_alloc(size_t layer_sum_bytes);
void * moe_chain_fullalloc_buffer(size_t need_bytes);

// Verify the graph: collect hidden MoE-chain intermediates and scan the whole
// graph for external consumers. Returns true on pass; on violation logs and
// exits the process (fail-fast, no escape hatch).
bool moe_chain_verify_graph(struct ggml_cgraph * gf);

} // namespace stream_moe
