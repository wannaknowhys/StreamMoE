#include "backend/minigraph_exec.h"
#include "backend/route_b_chain.h"
#include "common/logger.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace stream_moe {

namespace {

// "blk.N.ffn_gate_exps.weight" -> N and branch ("ffn_gate_exps.weight" etc.)
struct parsed_node_t {
    uint32_t layer = 0;
    bool     down  = false;
    bool     ok    = false;
};

// Read an ids element honoring the tensor's real row stride. The routing ids
// tensor is NOT guaranteed contiguous: hash layers (L0-2) are compact, but
// argsort layers (L3+) use a large nb[1] (e.g. 1024 bytes) with sparse rows.
#define MOE_ID_AT(ids, t, k) \
    (*(const int32_t*)((const char*)(ids)->data + (size_t)(t) * (ids)->nb[1] + (size_t)(k) * (ids)->nb[0]))

parsed_node_t parse_weight_name(const char* name) {
    parsed_node_t r;
    if (!name) return r;
    std::string s(name);
    const std::string prefix = "blk.";
    if (s.rfind(prefix, 0) != 0) return r;
    size_t p = s.find('.', prefix.size());
    if (p == std::string::npos) return r;
    try { r.layer = static_cast<uint32_t>(std::stoul(s.substr(prefix.size(), p - prefix.size()))); }
    catch (...) { return r; }
    std::string branch = s.substr(p + 1);
    r.down = branch.rfind("ffn_down_exps", 0) == 0;
    r.ok = true;
    return r;
}

struct keyed_expert_t {
    uint32_t layer, expert;
};

// Cross-call pin state for the route B pin lifecycle (docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4.8,
// role-based): the scheduler delivers a layer's gate / up / down MUL_MAT_ID nodes as SEPARATE
// graph_compute calls. Pin on the first non-down call, reuse on later non-down calls, and
// release ALL of the layer's pins on the down call. Single decode thread per context.
struct pin_state_t {
    int32_t layer = -1;
    std::vector<expert_handle_t> pins;
};
pin_state_t& pin_state() {
    static pin_state_t s;
    return s;
}
bool pin_state_has(const pin_state_t& st, uint32_t layer, uint32_t expert) {
    for (const auto& h : st.pins)
        if (h.layer == layer && h.expert == expert) return true;
    return false;
}
expert_handle_t pin_state_find(const pin_state_t& st, uint32_t layer, uint32_t expert) {
    for (const auto& h : st.pins)
        if (h.layer == layer && h.expert == expert) return h;
    return {-1, 0, layer, expert, 0, false};
}

} // namespace

// Privatisation modes:
//  - ping-pong (default): hidden compute-node outputs alternate between two
//    shared buffers (odd/even); safe while a node's inputs live at most one
//    compute step back (verify checks this and falls back otherwise).
//  - full-alloc (fallback): each hidden node gets its own byte range in one
//    growable buffer; per-layer offset/counter, reset at the layer end (moe_out).
// moe_out itself is never hidden - it writes the main-graph dst for the dense
// side.
static int s_hide_flip = 0;
static std::vector<int>    s_layer_cnt;
static std::vector<size_t> s_layer_off;

static bool is_view_op(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_CONT;
}


static void ensure_layer_state(int layer) {
    if ((int) s_layer_cnt.size() <= layer) {
        s_layer_cnt.resize(layer + 1);
        s_layer_off.resize(layer + 1);
    }
}

static void reset_layer(int layer) {
    ensure_layer_state(layer);
    s_layer_cnt[layer] = 0;
    s_layer_off[layer] = 0;
}

// Layer from a named MoE-chain node ("ffn_moe_geglu-0" -> 0). mm nodes carry
// their layer from the weight name (pn.layer) instead.
static int layer_of_name(const char * name) {
    if (!name) return -1;
    const char * dash = strrchr(name, '-');
    return dash ? atoi(dash + 1) : -1;
}

static bool hide_output(ggml_tensor * nd, int layer) {
    if (!moe_chain_pingpong_ok() && layer >= 0) {
        ensure_layer_state(layer);
        const size_t need = ggml_nbytes(nd);
        void * pb = moe_chain_fullalloc_buffer(s_layer_off[layer] + need);
        if (!pb) return false;
        nd->data = static_cast<char*>(pb) + s_layer_off[layer];
        s_layer_off[layer] += need;
        s_layer_cnt[layer]++;
        return true;
    }
    // unnamed nodes (e.g. moe_out summation adds) carry no layer - fall back to
    // ping-pong flip for them (their safety is regression-verified) even in
    // full-alloc mode.
    void * pb = moe_chain_pingpong_buffer(s_hide_flip & 1, ggml_nbytes(nd));
    if (!pb) return false;
    nd->data = pb;
    s_hide_flip++;
    return true;
}

// Route B delegation: the compact slot pool is one contiguous block with a uniform stride
// (slot e at pool_base + e*slot_size). So each original MUL_MAT_ID node can be executed by
// the OFFICIAL ggml_mul_mat_id kernel using a weight leaf [ne00, ne01, num_slots] whose
// nb[2] = slot_size, plus an ids tensor translated (in OUR private mini-graph) from expert
// ids to slot indices.
//
// All mini-graph tensors are leaf wrappers (op == NONE, data copied from the main graph /
// slot memory) - we NEVER view/reshape main-graph op nodes, which would drag the whole
// upstream chain into the mini-graph.
//
// Pin lifecycle (docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4.8, role-based):
//   - split with only non-down nodes: pin experts here (no release)
//   - split with a down node: wait_ready, compute, then release
static enum ggml_status exec_split_legacy_impl(
    ggml_context* ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    const ggml_tensor* const* nodes,
    int n_nodes,
    int n_threads)
{
    if (n_nodes == 0) return GGML_STATUS_SUCCESS;

    const moe_model_topology_t& topo = sched.topology();

    // ---- phase 1: parse + pin (non-down) / wait (down) ----
    std::vector<keyed_expert_t> pin_keys, down_keys;
    std::vector<expert_handle_t> pins;
    bool has_down = false;

    auto add_key = [&](std::vector<keyed_expert_t>& v, uint32_t l, uint32_t e) {
        for (const auto& x : v) if (x.layer == l && x.expert == e) return;
        v.push_back({l, e});
    };

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* nd = nodes[i];
        if (nd->op != GGML_OP_MUL_MAT_ID) continue; // view/layout ops handled in phase 2
        parsed_node_t pn = parse_weight_name(nd->src[0]->name);
        if (!pn.ok) {
            LOG_ERROR("stream_moe: cannot parse weight name " << nd->src[0]->name);
            return GGML_STATUS_FAILED;
        }
        const ggml_tensor* ids = nd->src[2];
        if (!ids->data) return GGML_STATUS_FAILED;
        for (int t = 0; t < ids->ne[1]; ++t)
            for (int k = 0; k < ids->ne[0]; ++k) {
                int32_t e = MOE_ID_AT(ids, t, k);
                if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) return GGML_STATUS_FAILED;
                if (pn.down) add_key(down_keys, pn.layer, static_cast<uint32_t>(e));
                else         add_key(pin_keys,  pn.layer, static_cast<uint32_t>(e));
            }
        if (pn.down) has_down = true;
    }

    if (has_down) {
        for (const auto& k : down_keys) sched.wait_ready(k.layer, k.expert);
    } else {
        uint32_t layer = pin_keys.empty() ? 0 : pin_keys[0].layer;
        if (pin_state().layer != static_cast<int32_t>(layer)) {
            for (const auto& h : pin_state().pins) sched.unpin(h); // safety: stray from prior layer
            pin_state().layer = static_cast<int32_t>(layer);
            pin_state().pins.clear();
        }
        for (const auto& k : pin_keys) {
            if (!pin_state_has(pin_state(), k.layer, k.expert)) {
                pin_state().pins.push_back(sched.pin_expert(k.layer, k.expert));
            }
        }
    }
    // ---- phase 2: one official mul_mat_id per node, in a leaf-only mini-graph ----
    bool ok = true;
    for (int i = 0; i < n_nodes && ok; ++i) {
        const ggml_tensor* nd = nodes[i];

        // View / layout ops on our host compute buffer: pure views, no weight
        // read. Point the dst at the source slice (VIEW) or keep the data
        // pointer (RESHAPE/TRANSPOSE/PERMUTE/CONT - nb already fixed by graph).
        if (nd->op != GGML_OP_MUL_MAT_ID) {
            const bool is_layout = nd->op == GGML_OP_VIEW || nd->op == GGML_OP_RESHAPE ||
                                   nd->op == GGML_OP_TRANSPOSE || nd->op == GGML_OP_PERMUTE ||
                                   nd->op == GGML_OP_CONT;
            if (is_layout) {
                if (!nd->src[0] || !nd->src[0]->data) {
                    LOG_ERROR("stream_moe: view op without source data " << ggml_op_name(nd->op));
                    ok = false;
                    break;
                }
                ggml_tensor* mut = const_cast<ggml_tensor*>(nd);
                if (nd->op == GGML_OP_VIEW) {
                    mut->data = (char*)nd->src[0]->data + nd->view_offs;
                } else {
                    mut->data = nd->src[0]->data;
                }
                continue;
            }
            // Weightless MoE-chain compute node: srcs are materialised on the
            // main graph (MUL_MAT_ID dsts written by our mini-graphs; views
            // pointed into them). Run the ORIGINAL node via the CPU backend in a
            // manual single-node graph.
            for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
                if (nd->src[s] && !nd->src[s]->data) {
                    LOG_ERROR("stream_moe: chain compute src without data for " << ggml_op_name(nd->op));
                    ok = false;
                }
            }
            if (!ok) break;
            const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
            const int layer = layer_of_name(nd->name);
            if (!is_out) {
                if (!hide_output(const_cast<ggml_tensor*>(nd), layer)) { ok = false; break; }
            }
            ggml_cgraph* gf = ggml_new_graph(ctx);
            gf->nodes[0] = const_cast<ggml_tensor*>(nd);
            gf->n_nodes  = 1;
            if (ggml_backend_graph_compute(cpu_backend, gf) != GGML_STATUS_SUCCESS) {
                LOG_ERROR("stream_moe: chain compute failed for " << (nd->name ? nd->name : ggml_op_name(nd->op)));
                ok = false;
            }
            if (is_out && layer >= 0) reset_layer(layer);   // layer end: free full-alloc ranges
            continue;
        }

        const ggml_tensor* w   = nd->src[0];
        const ggml_tensor* cur = nd->src[1];
        const ggml_tensor* ids = nd->src[2];
        const ggml_tensor* dst = nd;
        parsed_node_t pn = parse_weight_name(w->name);

        // hide: reroute this MUL_MAT_ID's output into the private buffer
        // (dst == nd; mini-graph writes nd->data, which now points at the buffer)
        if (!hide_output(const_cast<ggml_tensor*>(nd), pn.layer)) { ok = false; break; }

        // branch layout (compact slot offset, uniform across experts)
        size_t off = 0, bytes = 0;
        if (!sched.branch_layout(pn.layer, 0, w->name, off, bytes)) {
            LOG_ERROR("stream_moe: no slot layout for " << w->name);
            ok = false; break;
        }

        // Resolve the resident region(s) of this MUL_MAT_ID's active experts.
        // All active experts must share ONE region (single-region execution);
        // a mixed RAM/VRAM active set needs per-region sub-computes, which only
        // appear once device moves land (phase B).
        uint32_t gidx = sched.group_of(pn.layer);
        if (gidx == static_cast<uint32_t>(-1)) {
            LOG_ERROR("stream_moe: no subpool group for layer " << pn.layer);
            ok = false; break;
        }
        const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
        std::vector<int32_t> slotbuf(n_ids, -1);
        const expert_scheduler::subpool_t* use_sp = nullptr;
        for (int t = 0; t < ids->ne[1] && ok; ++t)
            for (int k = 0; k < ids->ne[0] && ok; ++k) {
                const int32_t e = MOE_ID_AT(ids, t, k);
                int32_t slot = -1;
                if (pn.down) {
                    slot = sched.slot_of(pn.layer, static_cast<uint32_t>(e));
                } else {
                    expert_handle_t h = pin_state_find(pin_state(), pn.layer, static_cast<uint32_t>(e));
                    if (h.pinned) slot = h.slot;
                }
                if (slot < 0) { ok = false; break; }
                const expert_scheduler::subpool_t* osp = sched.subpool_of_slot(slot);
                if (!osp) { ok = false; break; }
                if (use_sp && use_sp != osp) {
                    LOG_ERROR("stream_moe: mixed-region active set for " << w->name
                              << " (pool " << use_sp->pool << " + pool " << osp->pool << ") not yet supported");
                    ok = false; break;
                }
                if (!use_sp) use_sp = osp;
                slotbuf[static_cast<size_t>(t) * ids->ne[0] + k] = slot;
            }
        if (!ok) break;
        if (!use_sp) {
            // No active experts (empty routing): nothing to compute for this node.
            break;
        }
        const expert_scheduler::subpool_t& sp = *use_sp;
        static bool logged_region = false;
        if (!logged_region && sp.pool != 0) {
            logged_region = true;
            LOG_INFO("stream_moe: expert execution reads pool " << sp.pool
                     << " (device region, host-mapped VRAM)");
        }
        const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);
        ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
        w3d->ne[2] = g_n_slots;
        w3d->nb[2] = sp.expert_size;
        w3d->nb[3] = sp.expert_size * g_n_slots;
        w3d->data  = sp.base + off;

        // ids_slot: translate expert ids -> region-local slot indices
        ggml_tensor* ids_slot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
        auto* ids_slot_data = static_cast<int32_t*>(ids_slot->data);
        for (size_t i = 0; i < n_ids; ++i) {
            if (slotbuf[i] < 0) { ok = false; break; }
            ids_slot_data[i] = slotbuf[i] - static_cast<int32_t>(sp.slot_begin);
        }
        if (!ok) break;

        // b_leaf: wrap the main graph's activation (contiguous CPU buffer)
        ggml_tensor* b_leaf = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
        b_leaf->data = cur->data;
        std::memcpy(b_leaf->nb, cur->nb, sizeof(b_leaf->nb));

        // leaf-only graph; result written straight into the main node's output
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* mm = ggml_mul_mat_id(ctx, w3d, b_leaf, ids_slot);
        mm->data = dst->data; // nb of mm == nb of dst (same shape, both contiguous)

        ggml_build_forward_expand(gf, mm);
        if (ggml_backend_graph_compute(cpu_backend, gf) != GGML_STATUS_SUCCESS) {
            LOG_ERROR("stream_moe: mini-graph compute failed for " << w->name);
            ok = false;
        }
        }

    // ---- phase 3: release all of the layer's pins (down role) ----
    if (has_down) {
        for (const auto& h : pin_state().pins) sched.unpin(h);
        pin_state().pins.clear();
        pin_state().layer = -1;
    }

    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

// ========================= whole-layer burst executor ======================
// M2 executor (docs/M2_DEVICE_EXECUTOR.md 2.2): llama delivers the privatised
// chain one node per graph_compute; on the layer's first node we run the WHOLE
// captured sequence at once. Hidden outputs go to a per-layer full-alloc buffer
// (each node its own range - the anonymous convergence adds read weighted views
// several steps back, so ping-pong would clobber). Views over hidden producers
// get their data pointer refreshed when the producer output lands, so later
// (even dense-side) readers see the right memory.

// ---- per-layer full-alloc hide (unconditional; burst knows the layer) -----
static bool hide_burst(ggml_tensor * nd, int32_t layer) {
    ensure_layer_state(layer);
    const size_t need = ggml_nbytes(nd);
    void * pb = moe_chain_fullalloc_buffer(s_layer_off[layer] + need);
    if (!pb) return false;
    nd->data = static_cast<char*>(pb) + s_layer_off[layer];
    s_layer_off[layer] += need;
    return true;
}

// Refresh every view alias of `prod` inside `layer`'s capture so the views
// point at the producer's current (possibly hidden) output.
static void refresh_aliases(int32_t layer, const ggml_tensor * prod) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex) return;
    for (const auto & va : ex->view_aliases) {
        if (va.prod == prod) {
            va.view->data = static_cast<char*>(prod->data) + va.off;
        }
    }
}

// Slot of a pinned (layer, expert), or -1.
static int32_t pin_slot(const std::vector<expert_handle_t>& pins, uint32_t layer, uint32_t expert) {
    for (const auto & h : pins) {
        if (h.pinned && h.layer == layer && h.expert == expert) return h.slot;
    }
    return -1;
}


// Execute one captured compute node (view aliases already refreshed by the
// producers). mm / weightless compute both run as single-node CPU graphs.
static bool exec_one_burst(ggml_context * ctx, ggml_backend_t cpu, expert_scheduler& sched,
                           const moe_model_topology_t& topo, ggml_tensor * nd,
                           int32_t layer, const std::vector<expert_handle_t>& pins) {
    if (is_view_op(nd)) {   // capture excludes views; defensive only
        ggml_tensor * mut = const_cast<ggml_tensor*>(nd);
        mut->data = nd->op == GGML_OP_VIEW
                    ? static_cast<char*>(nd->src[0]->data) + nd->view_offs
                    : nd->src[0]->data;
        return true;
    }

    if (nd->op == GGML_OP_MUL_MAT_ID) {
        ggml_tensor * w   = const_cast<ggml_tensor*>(nd->src[0]);
        ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
        ggml_tensor * ids = const_cast<ggml_tensor*>(nd->src[2]);
        ggml_tensor * dst = nd;
        if (!ids->data) return false;
        if (!hide_burst(dst, layer)) return false;

        parsed_node_t pn = parse_weight_name(w->name);
        if (!pn.ok) { LOG_ERROR("stream_moe: cannot parse weight name " << w->name); return false; }

        size_t off = 0, bytes = 0;
        if (!sched.branch_layout(pn.layer, 0, w->name, off, bytes)) {
            LOG_ERROR("stream_moe: no slot layout for " << w->name);
            return false;
        }

        uint32_t gidx = sched.group_of(pn.layer);
        if (gidx == static_cast<uint32_t>(-1)) return false;
        const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
        std::vector<int32_t> slotbuf(n_ids, -1);
        const expert_scheduler::subpool_t* use_sp = nullptr;
        for (int t = 0; t < ids->ne[1]; ++t) {
            for (int k = 0; k < ids->ne[0]; ++k) {
                const int32_t e = MOE_ID_AT(ids, t, k);
                if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) return false;
                const int32_t slot = pin_slot(pins, pn.layer, static_cast<uint32_t>(e));
                if (slot < 0) return false;
                const expert_scheduler::subpool_t* osp = sched.subpool_of_slot(slot);
                if (!osp) return false;
                if (use_sp && use_sp != osp) {
                    LOG_ERROR("stream_moe: mixed-region active set for " << w->name
                              << " (pool " << use_sp->pool << " + pool " << osp->pool << ") - device split pending");
                    return false;
                }
                if (!use_sp) use_sp = osp;
                slotbuf[static_cast<size_t>(t) * ids->ne[0] + k] = slot;
            }
        }
        if (!use_sp) return true;   // no active experts
        const expert_scheduler::subpool_t& sp = *use_sp;

        const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);
        ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
        w3d->ne[2] = g_n_slots;
        w3d->nb[2] = sp.expert_size;
        w3d->nb[3] = sp.expert_size * g_n_slots;
        w3d->data  = sp.base + off;

        ggml_tensor* ids_slot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
        auto* ids_slot_data = static_cast<int32_t*>(ids_slot->data);
        for (size_t i = 0; i < n_ids; ++i) {
            ids_slot_data[i] = slotbuf[i] - static_cast<int32_t>(sp.slot_begin);
        }

        ggml_tensor* b_leaf = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
        b_leaf->data = cur->data;
        std::memcpy(b_leaf->nb, cur->nb, sizeof(b_leaf->nb));

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_tensor* mm = ggml_mul_mat_id(ctx, w3d, b_leaf, ids_slot);
        mm->data = dst->data;
        ggml_build_forward_expand(gf, mm);
        if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
            LOG_ERROR("stream_moe: burst mm compute failed for " << w->name);
            return false;
        }
        refresh_aliases(layer, nd);
        return true;
    }

    // weightless chain compute (or the moe_out end)
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (nd->src[s] && !nd->src[s]->data) {
            LOG_ERROR("stream_moe: burst chain compute src without data for " << ggml_op_name(nd->op));
            return false;
        }
    }
    const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
    if (!is_out) {
        if (!hide_burst(nd, layer)) return false;
    }
    ggml_cgraph* gf = ggml_new_graph(ctx);
    gf->nodes[0] = nd;
    gf->n_nodes  = 1;
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: burst chain compute failed for " << (nd->name ? nd->name : ggml_op_name(nd->op)));
        return false;
    }
    if (!is_out) {
        refresh_aliases(layer, nd);
    } else {
        reset_layer(layer);   // layer end: release the full-alloc ranges
    }
    return true;
}

// Burst one whole layer from its captured sequence.
static enum ggml_status exec_layer_burst(int32_t layer, ggml_context * ctx,
                                         ggml_backend_t cpu, expert_scheduler& sched,
                                         int /*n_threads*/) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex || ex->compute.empty()) return GGML_STATUS_SUCCESS;
    const moe_model_topology_t& topo = sched.topology();

    // Per-layer full-alloc capacity for the hidden intermediates.
    size_t lsum = 0;
    for (const auto * cn : ex->compute) {
        if (!(cn->name && strstr(cn->name, "ffn_moe_out"))) lsum += ggml_nbytes(cn);
    }
    moe_chain_set_full_alloc(lsum);
    reset_layer(layer);

    // Fix input-side layout tensors (views/reshapes feeding the compute, whose
    // producer is a weight/leaf or a llama-side tensor).
    for (const auto * lt : ex->input_layouts) {
        if (!lt || !lt->src[0] || !lt->src[0]->data) continue;
        ggml_tensor * mut = const_cast<ggml_tensor*>(lt);
        mut->data = lt->op == GGML_OP_VIEW
                    ? static_cast<char*>(lt->src[0]->data) + lt->view_offs
                    : lt->src[0]->data;
    }

    // Pin the layer's whole active expert set (all mm nodes share the ids).
    std::vector<keyed_expert_t> keys;
    auto add_key = [&](uint32_t l, uint32_t e) {
        for (const auto & x : keys) if (x.layer == l && x.expert == e) return;
        keys.push_back({ l, e });
    };
    for (const auto * cn : ex->compute) {
        if (!cn || cn->op != GGML_OP_MUL_MAT_ID || !cn->src[0] || !cn->src[2]) continue;
        parsed_node_t pn = parse_weight_name(cn->src[0]->name);
        if (!pn.ok) continue;
        const ggml_tensor * ids = cn->src[2];
        if (!ids->data) continue;
        for (int t = 0; t < ids->ne[1]; ++t)
            for (int k = 0; k < ids->ne[0]; ++k) {
                const int32_t e = MOE_ID_AT(ids, t, k);
                if (e >= 0 && e < static_cast<int32_t>(topo.n_expert)) add_key(pn.layer, static_cast<uint32_t>(e));
            }
    }
    std::vector<expert_handle_t> pins;
    pins.reserve(keys.size());
    for (const auto & k : keys) {
        expert_handle_t h = sched.pin_expert(k.layer, k.expert);
        if (!h.pinned) {
            LOG_ERROR("stream_moe: burst pin failed L" << k.layer << " E" << k.expert);
            for (const auto & hh : pins) sched.unpin(hh);
            return GGML_STATUS_FAILED;
        }
        pins.push_back(h);
    }

    bool ok = true;
    for (const auto * cn : ex->compute) {
        if (!ok) break;
        ok = exec_one_burst(ctx, cpu, sched, topo, const_cast<ggml_tensor*>(cn), layer, pins);
    }
    for (const auto & h : pins) sched.unpin(h);
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

// graph_compute entry: dispatch to the whole-layer burst on the layer's first
// privatised node; later same-layer splits are no-ops (already produced by the
// burst). Nodes outside the capture fall back to the per-split legacy path.
enum ggml_status moe_exec_mul_mat_id(
    ggml_context* ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    const ggml_tensor* const* nodes,
    int n_nodes,
    int n_threads)
{
    if (n_nodes == 0) return GGML_STATUS_SUCCESS;

    const ggml_tensor * first = nodes[0];
    const int32_t layer = moe_chain_layer_of_node(first);
    if (layer < 0) {
        // Not a captured privatised node: legacy per-split execution.
        return exec_split_legacy_impl(ctx, cpu_backend, sched, nodes, n_nodes, n_threads);
    }
    const int32_t idx = moe_chain_layer_index(layer, first);
    static thread_local int32_t g_bl = -1;
    if (idx > 0 && g_bl == layer) {
        return GGML_STATUS_SUCCESS;   // already produced by this pass's burst
    }
    const enum ggml_status st = exec_layer_burst(layer, ctx, cpu_backend, sched, n_threads);
    g_bl = layer;
    return st;
}

} // namespace stream_moe
