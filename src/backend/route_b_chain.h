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

// Verify the graph: collect hidden MoE-chain intermediates and scan the whole
// graph for external consumers. Returns true on pass; on violation logs and
// exits the process (fail-fast, no escape hatch).
bool moe_chain_verify_graph(struct ggml_cgraph * gf);

} // namespace stream_moe
