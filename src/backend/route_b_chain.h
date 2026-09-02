#pragma once

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

namespace stream_moe {

// Verify the graph: collect hidden MoE-chain intermediates and scan the whole
// graph for external consumers. Returns true on pass; on violation logs and
// exits the process (fail-fast, no escape hatch).
bool moe_chain_verify_graph(struct ggml_cgraph * gf);

} // namespace stream_moe
