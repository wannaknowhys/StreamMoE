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

#include <vector>

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
                              ggml_backend_t our_backend,
                              const std::vector<ggml_backend_dev_t> & physical_layers, bool probe = false);

// Full-allocation mode for the layer hidden intermediates (the compact
// bucket engine allocates every twin output inside this): ONE fixed buffer
// sized to a single layer's hidden-intermediate sum, allocated once (never
// re-grown - a grow would invalidate already-pointed nd->data). Each hidden
// compute node gets its own byte range inside it; the executor tracks
// per-layer offsets and resets them at the layer end (moe_out).
void   moe_chain_set_full_alloc(size_t layer_sum_bytes);
void * moe_chain_fullalloc_buffer(size_t need_bytes);

void route_b_begin_graph();
void route_b_on_node(const ggml_tensor * node, int il, ggml_backend_dev_t physical);
int route_b_official_layer(const ggml_tensor * node);
ggml_backend_dev_t route_b_op_physical(const ggml_tensor * node);

// Debug: true when `p` lies inside the whole-layer arena (R3). Used to check
// that the scheduler did not overwrite pre-allocated node data.
bool route_b_in_arena(const void * p);

// Expert pool devices recorded by route_b_setup (per-device arena plan).
void route_b_add_expert_pool_device(const char * dev);
// The device that owns this model's experts ("" = host / several devices).
const char * route_b_closure_device();

// Device name -> backend registry (docs/PER_DEVICE_ARENA.md SS8.5, phase 3).
// route_b_setup records every device pool's backend here; the executor resolves
// a layer's dense head/tail backend from the node's placement device. The host
// plan resolves to the CPU backend the executor already holds (no registry hit).
void route_b_add_device_backend(const char * dev, ggml_backend_t be);
ggml_backend_t route_b_device_backend(const char * dev);   // nullptr when unknown

// Device that owns a captured node ("" = host). Valid after the graph build;
// refreshed by layout_arena on every build. "" for an unknown node.
const char * route_b_node_device(const struct ggml_tensor * node);

// Generic cross-device transfer (docs/PER_DEVICE_ARENA.md 3/SS8.3). Every captured
// edge whose producer and consumer devices differ gets a consumer-side copy (a
// shell) in the consumer device's `xfer` region, and the consumer's src is
// rewired to it. The executor runs ggml_backend_tensor_copy(src, dst) once at
// `stage` (below) - so a producer written many times inside a layer copies once,
// not per write. One shell per (producer, consumer device, stage). A tensor is
// the producer node's output, so it stays on the producer's device.
enum route_b_xfer_stage {
    ROUTE_B_XFER_LAYER_FRONT = 0,   // cross-layer carry: before the consumer layer's head
    ROUTE_B_XFER_CLOSURE     = 1,   // within-layer: before the MoE closure (e.g. cur)
    ROUTE_B_XFER_TAIL        = 2,   // within-layer: before the dense tail (e.g. moe_out)
};
struct route_b_relay_t {
    const ggml_tensor * src = nullptr;   // producer tensor on the producer's device
    ggml_tensor *       dst = nullptr;   // local copy on the consumer's device
    int32_t             layer = -1;      // layer whose stage runs the copy
    int                 stage = ROUTE_B_XFER_LAYER_FRONT;
};
// All transfers of the current graph build (empty when single-device / CPU-only).
const std::vector<route_b_relay_t> & route_b_relays();

// True for llama's fused ops (FLASH_ATTN_EXT / LIGHTNING_INDEXER / DSV4_HC_*)
// whose fusion llama_context::resolve probes against the layer's device.
bool route_b_is_fused_op(enum ggml_op op);

// Debug: write a node's full bytes to <STREAM_MOE_TMP_BIN_DIR>/ub<N>/<name>.bin
// (+ .meta). Used for offline cos comparison of whole-layer vs baseline.
void route_b_dump_node_bin(int layer, const char * name, const char * op,
                           int type, int64_t ne0, int64_t ne1, const void * data, size_t nb);
// Debug: start a new ubatch (bumps the bin-dump subdirectory index).
void route_b_begin_ubatch();

// Debug: monotonic build counter (incremented by layout_arena per graph build),
// so per-build dumps can be told apart.
int route_b_build_id();

#if defined(STREAM_MOE_ROUTE_B) && defined(STREAM_MOE_PREFILL_EXPORT)
// Prefill export: tensors the export reads after graph compute (via the sched
// eval callback). Whole-layer ownership reuses arena slots inside a layer, so
// that post-split read can land on an overwritten slot; layout_arena keeps these
// in the retained (never-reused) region instead. Replaced on every graph build;
// pass an empty list when not exporting. Only the route_b + prefill intersection
// needs it (route_b-only has no observer, prefill-only has no slot reuse).
void route_b_set_export_retained(const std::vector<const struct ggml_tensor *> & ts);
#endif

// Verify the graph: collect hidden MoE-chain intermediates and scan the whole
// graph for external consumers. Returns true on pass; on violation logs and
// exits the process (fail-fast, no escape hatch).
bool moe_chain_verify_graph(struct ggml_cgraph * gf);

// ---- whole-layer burst capture (M2 executor) ----------------------------
// Per-layer privatised compute-node sequence, captured at build time (the
// graph is delivered one node per graph_compute, so the executor needs the
// whole-layer order up front). Each entry is the main-graph compute node
// (stable while llama reuses the built graph across decodes; capture is
// refreshed on every rebuild). Includes the anonymous per-topk convergence
// adds and the moe_out end.
struct moe_view_alias_t {
    ggml_tensor * view = nullptr;   // chain view of a hidden producer
    ggml_tensor * prod = nullptr;   // underlying producer compute node
    int64_t        off = 0;         // byte offset inside the producer output
};
struct moe_layer_exec_t {
    int32_t layer = -1;
    std::vector<ggml_tensor*>   compute;        // compute nodes, exec order
    std::vector<moe_view_alias_t> view_aliases; // views over hidden producers
    std::vector<ggml_tensor*>   input_layouts;  // layout tensors feeding compute
    // Result-buffer layout (node-level interval reuse, computed by verify):
    // out_off[i] = byte offset inside the layer's result buffer where
    // compute[i]'s output lands; the buffer itself is one block of
    // result_bytes, reused across layers. -1 = full-alloc fallback needed
    // (layout conflict) - the executor then falls back to per-node offsets.
    std::vector<int64_t> out_off;     // aligned with compute
    size_t               result_bytes = 0;   // layer result buffer size (max over compute)
    bool                 layout_ok    = false;
    // External leaves consumed by the closure (verify input-side analysis,
    // SS7.6.5): srcs of closure compute nodes whose producer (after unwrapping
    // views) is NOT inside this layer's compute sequence. These are the llama-
    // side tensors a per-device graph must upload / reference as leaf inputs
    // (cur, ids/routing, scale/weight sources). role tags what the consuming
    // mm (or weightless) node used it for, for the offline dump.
    struct ext_leaf_t {
        ggml_tensor * tensor = nullptr;
        const char  * role   = nullptr;   // "cur" / "ids" / "w" / "scale" / "other"
        const char  * user   = nullptr;   // consuming node's name / op
    };
    std::vector<ext_leaf_t> external_leaves;
};
// Layer of the privatised exec sequence containing `node` (pointer match), or
// -1 when `node` is not a captured privatised compute node.
int32_t moe_chain_layer_of_node(const ggml_tensor * node);
// Whole exec sequence of `layer`, or nullptr.
const moe_layer_exec_t * moe_chain_layer_exec(int32_t layer);
// Index of `node` inside its layer's exec sequence, or -1.
int32_t moe_chain_layer_index(int32_t layer, const ggml_tensor * node);

// Whole-layer capture (docs/ROUTE_B_LAYER_OWNERSHIP.md L2): all compute nodes of
// `layer` (dense + MoE), graph order. nullptr when the layer was not captured.
// Layer attribution: llama name suffix "-<il>", anonymous nodes inherit from
// their producers.
const std::vector<ggml_tensor*> * moe_chain_layer_nodes(int32_t layer);

// Debug-only: whole-layer capture populated on every build (both the whole-layer
// and the MoE-only path). Used to dump per-node contents for A/B comparison.
const std::vector<ggml_tensor*> * moe_chain_layer_nodes_all(int32_t layer);

// Build-time layer plan (docs/LAYER_EXECUTOR_DESIGN.md 4.1): the layer's node
// sets and boundaries, computed once at graph build. Replaces the runtime
// head/tail split heuristic in exec_layer_burst.
struct moe_layer_plan_t {
    int32_t layer = -1;
    std::vector<ggml_tensor*> all;   // all layer compute nodes (graph order)
    std::vector<ggml_tensor*> head;  // dense before the MoE input
    std::vector<ggml_tensor*> tail;  // dense after the MoE output
    const moe_layer_exec_t * moe = nullptr;   // MoE closure (g_layer_exec)
    ggml_tensor * moe_out = nullptr; // ffn_moe_out anchor
    ggml_tensor * input   = nullptr; // layer input tensor (best-effort boundary)
    ggml_tensor * output  = nullptr; // layer output tensor (best-effort boundary)
};
// Plan of an assigned layer, or nullptr when the layer was not captured.
const moe_layer_plan_t * moe_chain_layer_plan(int32_t layer);

} // namespace stream_moe
