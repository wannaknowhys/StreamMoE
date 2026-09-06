#include "backend/minigraph_exec.h"
#include "backend/mix_split.h"
#include "backend/route_b_chain.h"
#include "backend/moe_backend.h"
#include "common/logger.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef STREAM_MOE_TEMP
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#endif

// ggml-vulkan route-B extensions.
void* stmoe_vk_buffer_host_ptr(ggml_backend_buffer_t buffer);
void* stmoe_vk_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off);

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
        // Down role: gate/up splits of this layer already pinned the experts
        // (held in pin_state across graph_compute calls). They are READY by
        // construction (a pin only succeeds on a READY slot and refcount>0
        // blocks eviction). Just confirm every down key is pinned; a missing one
        // (edge: down split before any gate/up pin) is batch-pinned here.
        std::vector<keyed_expert_t> missing;
        for (const auto & k : down_keys) {
            if (!pin_state_has(pin_state(), k.layer, k.expert)) missing.push_back(k);
        }
        if (!missing.empty()) {
            uint64_t need[BITMAP_WORDS] = { 0 };
            batch_await_t aw;
            for (const auto & k : missing) expert_scheduler::bit_set(need, k.expert);
            std::vector<expert_handle_t> hp(missing.size());
            const int32_t np = sched.pin_layer(missing[0].layer, need, aw,
                                               hp.data(), static_cast<uint32_t>(hp.size()));
            if (np < 0) return GGML_STATUS_FAILED;
            for (int i = 0; i < np; ++i) pin_state().pins.push_back(hp[static_cast<size_t>(i)]);
        }
    } else {
        uint32_t layer = pin_keys.empty() ? 0 : pin_keys[0].layer;
        if (pin_state().layer != static_cast<int32_t>(layer)) {
            for (const auto& h : pin_state().pins) sched.unpin(h); // safety: stray from prior layer
            pin_state().layer = static_cast<int32_t>(layer);
            pin_state().pins.clear();
        }
        // pin the keys not yet held (one batch request for this split's set)
        std::vector<keyed_expert_t> missing;
        for (const auto & k : pin_keys) {
            if (!pin_state_has(pin_state(), k.layer, k.expert)) missing.push_back(k);
        }
        if (!missing.empty()) {
            uint64_t need[BITMAP_WORDS] = { 0 };
            batch_await_t aw;
            for (const auto & k : missing) expert_scheduler::bit_set(need, k.expert);
            std::vector<expert_handle_t> hp(missing.size());
            const int32_t np = sched.pin_layer(layer, need, aw, hp.data(),
                                               static_cast<uint32_t>(hp.size()));
            if (np < 0) {
                for (const auto& h : pin_state().pins) sched.unpin(h);
                return GGML_STATUS_FAILED;
            }
            for (int i = 0; i < np; ++i) pin_state().pins.push_back(hp[static_cast<size_t>(i)]);
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
        // SoA column: the branch's tensor slices live in one column at
        // `col_off` with stride `col_stride` (= compact per-expert bytes =
        // the stride vulkan hardcodes for MUL_MAT_ID).
        size_t col_off = 0, col_stride = 0;
        uint32_t col_index = 0;
        if (!sched.column_layout(sp, w->name, col_off, col_stride, col_index)) {
            LOG_ERROR("stream_moe: no column for " << w->name);
            ok = false; break;
        }
        const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);
        ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
        w3d->ne[2] = g_n_slots;
        w3d->nb[2] = col_stride;
        w3d->nb[3] = col_stride * g_n_slots;
        w3d->data  = sp.base + col_off;

        // ids_slot: translate expert ids -> region-local slot indices (data on
        // a local heap buffer - the exec ctx is no_alloc)
        std::vector<int32_t> ids_local(n_ids, -1);
        for (size_t i = 0; i < n_ids; ++i) {
            if (slotbuf[i] < 0) { ok = false; break; }
            ids_local[i] = slotbuf[i] - static_cast<int32_t>(sp.slot_begin);
        }
        if (!ok) break;
        ggml_tensor* ids_slot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
        ids_slot->data = ids_local.data();

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

// ---- per-layer layout-aware hide (burst knows the layer) ------------------
// hide_burst routes compute[i]'s output to the layer result block at the
// verify-computed offset (node-level interval reuse, layout_ok). Fallback when
// the layer has no usable layout: per-node bump (s_layer_off[layer]).
static bool hide_burst(ggml_tensor * nd, int32_t layer) {
    ensure_layer_state(layer);
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    int64_t off = -1;
    if (ex && ex->layout_ok) {
        const int32_t idx = moe_chain_layer_index(layer, nd);
        if (idx >= 0 && idx < (int32_t) ex->out_off.size() && ex->out_off[idx] >= 0) {
            off = ex->out_off[idx];
        }
    }
    if (off < 0) {
        // fallback: per-node bump (no verify layout for this layer)
        const size_t need = ggml_nbytes(nd);
        void * pb = moe_chain_fullalloc_buffer(s_layer_off[layer] + need);
        if (!pb) return false;
        nd->data = static_cast<char*>(pb) + s_layer_off[layer];
        s_layer_off[layer] += need;
        return true;
    }
    const size_t need = ggml_nbytes(nd);
    void * pb = moe_chain_fullalloc_buffer(static_cast<size_t>(off) + need);
    if (!pb) return false;
    nd->data = static_cast<char*>(pb) + off;
    return true;
}

// Refresh every view alias of `prod` inside `layer`'s capture so the views
// point at the producer's current (possibly hidden) output. Buffer follows the
// producer (device execution reads views through the producer's vk buffer).
static void refresh_aliases(int32_t layer, const ggml_tensor * prod) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex) return;
    for (const auto & va : ex->view_aliases) {
        if (va.prod == prod) {
            ggml_tensor * mut = const_cast<ggml_tensor*>(va.view);
            mut->buffer = const_cast<ggml_tensor*>(prod)->buffer;
            // Offset folded into data; view_offs cleared so device kernels
            // (which add view_offs themselves) do not double-count.
            mut->data = static_cast<char*>(prod->data) + va.off;
            mut->view_offs = 0;
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

// ---------------------------------------------------------------------------
// Mixed-region MUL_MAT_ID execution (J6).
//
// Kernel facts (ggml_cpu mul_mat_id, ggml-cpu.c):
//   ids      [n_k, n_t]; entry (k,t) = expert id
//   src0     weight [d_in, d_out, n_slot]; expert slice at ids[k][t]
//   src1     cur [d_in, ne11, ne1(ids) tokens]; column chosen as
//            (id % ne11, token): gate_up cur ne11=1 => one column shared by all
//            experts; down cur ne11=n_k => one column per (topk slot, token).
//   dst      [d_out, n_k, n_t]; column (k,t) at dst[k*nb1 + t*nb2]
//
// So a mul_mat_id is a per-column (topk-slot, token) op: expert weights from
// ids, activation from cur, result to the matching dst column. Splitting by
// pool = running one sub-mm whose ids/cur/dst are the pool's OWN slice of
// (topk-slot, token) columns, then scattering the compact sub-dst columns back
// to the real main-dst columns.
//
// mix_split (backend/mix_split.cpp) produces, per pool, peel rounds. Each
// round is a full rectangle [width, n_active]: ids_sub stores pool-local slot
// numbers (row-major token-major, ggml ids layout), scatter[idx] records the
// original (t,k) of round idx = a*width+s. We rebuild cur_sub so sub-mm
// column (s,a) reads source cur column (scatter_k % cur_ne11, scatter_t).
// ---------------------------------------------------------------------------

// Build the sub-mm's rebuilt activation from the source cur columns the round
// references. Shared cur (ne11==1): width identical columns per token, so we
// store [d_in, width, n_active] with all width columns equal (kernel then reads
// s%width==s). Per-expert cur (ne11>1): column s carries source column
// (scatter_k % ne11, t). Returns false on a mis-sized source.
static bool build_cur_sub(const ggml_tensor * cur,
                          const std::vector<mix_scatter_t>& scatter,
                          uint32_t width, uint32_t n_active,
                          std::vector<float>& cur_sub, size_t& d_in_out) {
    const size_t d_in = static_cast<size_t>(cur->ne[0]);
    if (cur->type != GGML_TYPE_F32) { LOG_ERROR("stream_moe: mixed cur not f32"); return false; }
    d_in_out = d_in;
    const int64_t cur_ne11 = cur->ne[1];
    cur_sub.assign(static_cast<size_t>(width) * n_active * d_in, 0.0f);
    for (uint32_t a = 0; a < n_active; ++a) {
        for (uint32_t s = 0; s < width; ++s) {
            const mix_scatter_t & sc = scatter[static_cast<size_t>(a) * width + s];
            const int64_t kk = sc.k % cur_ne11;
            const int64_t tt = sc.t;
            const float * src = (const float *) ((const char *) cur->data + kk*cur->nb[1] + tt*cur->nb[2]);
            float * dst = &cur_sub[(static_cast<size_t>(a) * width + s) * d_in];
            std::memcpy(dst, src, d_in * sizeof(float));
        }
    }
    return true;
}

// Scatter a compact sub-dst [d_out, width, n_active] back into the main dst
// columns. Main dst is contiguous [d_out, n_k, n_t] (hide_burst'd host buffer);
// sub column idx=(a*width+s) -> main column (scatter_k + n_k*scatter_t)*d_out.
static void scatter_sub_dst(uint8_t * main_dst, size_t main_dst_nk,
                            const std::vector<mix_scatter_t>& scatter,
                            uint32_t width, uint32_t n_active,
                            const void * sub_dst, size_t d_out, size_t esize) {
    for (uint32_t a = 0; a < n_active; ++a) {
        for (uint32_t s = 0; s < width; ++s) {
            const mix_scatter_t & sc = scatter[static_cast<size_t>(a) * width + s];
            const size_t idx = static_cast<size_t>(a) * width + s;
            const size_t sub_off = idx * d_out * esize;
            const size_t main_off = (static_cast<size_t>(sc.k) + main_dst_nk * sc.t) * d_out * esize;
            std::memcpy(main_dst + main_off, (const char*)sub_dst + sub_off, d_out * esize);
        }
    }
}

// One pool's sub-mm round, CPU backend (pool 0). Writes the compact sub-dst
// [d_out, width, n_active] into `out` (f32, sized by the caller).
static bool exec_round_cpu(ggml_context * ctx, ggml_backend_t cpu,
                           const ggml_tensor * w,
                           const expert_scheduler::subpool_t & sp,
                           size_t col_off, size_t col_stride,
                           const std::vector<int32_t>& ids_sub, uint32_t width,
                           uint32_t n_active,
                           const std::vector<float>& cur_sub, size_t d_in,
                           const ggml_tensor * nd,
                           std::vector<float>& out) {
    const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);
    // Leaves' DATA point at our heap buffers (not the ctx pool) - the exec ctx
    // is a fixed-size arena sized from the graph nodes; a 129-token mixed round
    // would otherwise exhaust it allocating b_leaf/ids_leaf per sub-mm.
    ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = g_n_slots;
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * g_n_slots;
    w3d->data  = sp.base + col_off;

    ggml_tensor* ids_leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, n_active);
    ids_leaf->data = const_cast<int32_t*>(ids_sub.data());

    ggml_tensor* b_leaf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t)d_in, width, n_active);
    b_leaf->data = const_cast<float*>(cur_sub.data());

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* mm = ggml_mul_mat_id(ctx, w3d, b_leaf, ids_leaf);
    out.resize(ggml_nbytes(mm) / sizeof(float), 0.0f);
    mm->data = out.data();
    ggml_build_forward_expand(gf, mm);
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: mixed cpu round compute failed for " << (w->name ? w->name : "?"));
        return false;
    }
    (void)nd;
    return true;
}

// One pool's sub-mm round, device backend (pool != 0): upload ids/cur to the
// staging buffer, compute in the arena, read the compact sub-dst back to `out`.
static bool exec_round_vk(ggml_context * ctx, const ggml_tensor * w,
                          const expert_scheduler::subpool_t & sp,
                          size_t col_off, size_t col_stride,
                          const std::vector<int32_t>& ids_sub, uint32_t width,
                          uint32_t n_active,
                          const std::vector<float>& cur_sub, size_t d_in,
                          std::vector<float>& out) {
    device_exec_ctx_t* dv = stream_moe_backend_device_exec(sp.pool);
    if (!dv || !dv->be) {
        LOG_ERROR("stream_moe: mixed no device exec context for pool " << sp.pool);
        return false;
    }
    const size_t ids_bytes = ids_sub.size() * sizeof(int32_t);
    const size_t cur_bytes = cur_sub.size() * sizeof(float);
    const size_t d_out = static_cast<size_t>(w->ne[1]);
    const size_t out_bytes = d_out * static_cast<size_t>(width) * n_active * sizeof(float);
    // arena for the round's dst; staging holds ids + cur (ids first)
    if (!stream_moe_backend_device_ensure(sp.pool, out_bytes + 4u * 1024 * 1024,
                                           ids_bytes + cur_bytes + 4u * 1024 * 1024)) {
        LOG_ERROR("stream_moe: mixed round arena/stage ensure failed (pool " << sp.pool << ")");
        return false;
    }
    if (!dv->stage_map || !dv->arena_map) {
        LOG_ERROR("stream_moe: mixed round arena/stage host maps unavailable (pool " << sp.pool << ")");
        return false;
    }
    const size_t cur_off = (ids_bytes + 63u) & ~size_t(63u);
    std::memcpy(dv->stage_map, ids_sub.data(), ids_bytes);
    std::memcpy(dv->stage_map + cur_off, cur_sub.data(), cur_bytes);

    ggml_backend_buffer_t wbuf = reinterpret_cast<ggml_backend_buffer_t>(sp.dev_buf);
    const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);

    ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = g_n_slots;
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * g_n_slots;
    w3d->buffer = wbuf;
    w3d->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(wbuf, col_off));

    ggml_tensor* ids_leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, n_active);
    ids_leaf->buffer = dv->stage;
    ids_leaf->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv->stage, 0));

    ggml_tensor* b_leaf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t)d_in, width, n_active);
    b_leaf->buffer = dv->stage;
    b_leaf->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv->stage, cur_off));

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* mm = ggml_mul_mat_id(ctx, w3d, b_leaf, ids_leaf);
    out.resize(out_bytes / sizeof(float), 0.0f);
    mm->buffer = dv->arena;
    mm->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv->arena, 0));
    ggml_build_forward_expand(gf, mm);
    if (ggml_backend_graph_compute(dv->be, gf) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: mixed vk round compute failed for " << (w->name ? w->name : "?"));
        return false;
    }
    std::memcpy(out.data(), dv->arena_map, out_bytes);
    return true;
}

#ifdef STREAM_MOE_TEMP
// ---- column-split self-test (temporary diagnostic) ------------------------
// Validates that an arbitrary rectangular partition of the (k,t) column domain
// recomputed as independent compact mm rounds + scatter reproduces the single
// full-width mm result byte-for-byte. Bucket geometry and the scatter map are
// the same machinery a future per-device/bucket executor relies on, so this is
// the correctness base for the whole bucket design (docs/M2_DEVICE_EXECUTOR.md
// SS7.3). Controlled by env STREAM_MOE_TMP_SPLIT (values: vertical, vertical_
// asym, horizontal, horizontal_asym, mixed2x2 - or "all"). Runs only on a
// single-pool full-width plan (plan.rounds has one round covering all columns).
// Layout note: llama dst is [d_out, n_k, n_t]; a round's compact mm computes a
// width-column slice of every active token; scatter writes each (a*width+s)
// output column back to main-dst column (k + t*n_k). A block is a rectangle
// over k-slot ranges and token ranges. Within a single pool every token routes
// all n_k slots to that pool, so any rectangle (k0..k1, t0..t1) is a uniform
// full rectangle with zero waste.

struct tmp_blk_t {
    uint32_t k0 = 0, k1 = 0;   // k-slot half-open range
    uint32_t t0 = 0, t1 = 0;   // token half-open range
};

static std::vector<tmp_blk_t> tmp_split_blocks(const char * which,
                                               uint32_t n_k, uint32_t n_t) {
    std::vector<tmp_blk_t> out;
    const bool all = std::string(which) == "all";
    const auto add = [&](uint32_t k0, uint32_t k1, uint32_t t0, uint32_t t1) {
        k1 = std::min(k1, n_k); t1 = std::min(t1, n_t);
        if (k0 < k1 && t0 < t1) out.push_back({ k0, k1, t0, t1 });
    };
    if (std::string(which) == "full") {
        // Full domain as the reference sanity check (single block == full round).
        add(0, n_k, 0, n_t);
        return out;
    }
    if (all || std::string(which) == "vertical") {
        // vertical cut: split k slots (equal halves)
        const uint32_t h = n_k / 2;
        add(0, h, 0, n_t);
        add(h, n_k, 0, n_t);
    }
    if (all || std::string(which) == "vertical_asym") {
        // asymmetric expert counts: 2 + 3 + rest
        const uint32_t a = std::min(2u, n_k), b = std::min(5u, n_k);
        add(0, a, 0, n_t);
        add(a, b, 0, n_t);
        add(b, n_k, 0, n_t);
    }
    if (all || std::string(which) == "horizontal") {
        // horizontal cut: split token range (equal halves)
        const uint32_t h = n_t / 2;
        add(0, n_k, 0, h);
        add(0, n_k, h, n_t);
    }
    if (all || std::string(which) == "horizontal_asym") {
        // asymmetric token cut: 1/3 + 1/3 + rest
        const uint32_t a = n_t / 3, b = 2 * n_t / 3;
        add(0, n_k, 0, a);
        add(0, n_k, a, b);
        add(0, n_k, b, n_t);
    }
    if (all || std::string(which) == "mixed2x2") {
        // one vertical x one horizontal cut = four quadrants
        const uint32_t hk = n_k / 2, ht = n_t / 2;
        add(0, hk, 0, ht);
        add(0, hk, ht, n_t);
        add(hk, n_k, 0, ht);
        add(hk, n_k, ht, n_t);
    }
    return out;
}

// Turn a list of rectangular blocks (k0..k1 x t0..t1) into independent mix
// rounds replacing a single full-width round. Every token in the t range of a
// block contributes the k-slice [k0,k1) -> a legal compact mm round
// [width=(k1-k0), n_active=(t1-t0)] with zero waste. ids (not consumed by the
// exec loop, which rebuilds from scatter + ids_compact) mirror the scatter
// order so the round is never seen as empty.
static mix_plan_t tmp_plan_from_blocks(const mix_plan_t & base,
                                       const std::vector<tmp_blk_t> & blocks,
                                       uint32_t n_k,
                                       const std::vector<int32_t>& ids_compact) {
    mix_plan_t out;
    out.n_expert_used = base.n_expert_used;
    out.n_tokens      = base.n_tokens;
    out.n_pools       = base.n_pools;
    out.buckets       = base.buckets;
    const uint32_t pool = base.rounds.empty() ? 0 : base.rounds[0].pool;
    for (const auto & b : blocks) {
        mix_round_t r;
        r.pool     = pool;
        r.width    = b.k1 - b.k0;
        r.n_active = b.t1 - b.t0;
        r.ids.reserve(static_cast<size_t>(r.width) * r.n_active);
        r.scatter.reserve(static_cast<size_t>(r.width) * r.n_active);
        for (uint32_t t = b.t0; t < b.t1; ++t) {
            for (uint32_t k = b.k0; k < b.k1; ++k) {
                r.scatter.push_back({ t, k });
                r.ids.push_back(ids_compact[static_cast<size_t>(t) * n_k + k]);
            }
        }
        if (!r.ids.empty()) out.rounds.push_back(std::move(r));
    }
    return out;
}

// Run one rectangular block as an independent compact mm round and scatter its
// output into `dst` (caller zeroed). Mirrors the exec_mixed_mm round handling:
// build pool-local slot ids, rebuild activation columns, compact mm on the CPU
// backend, scatter back at each (t,k). Returns false on any setup failure.
static bool tmp_run_block(ggml_context * ctx, ggml_backend_t cpu,
                          expert_scheduler & sched, ggml_tensor * nd,
                          uint32_t layer, const std::vector<expert_handle_t>& pins,
                          const std::vector<int32_t>& ids_compact,
                          const std::vector<int32_t>& expert_pool,
                          const moe_model_topology_t & topo,
                          uint32_t n_k, const tmp_blk_t & blk,
                          uint8_t * dst, size_t dst_nk, size_t d_out) {
    ggml_tensor * w   = const_cast<ggml_tensor*>(nd->src[0]);
    ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
    const uint32_t width   = blk.k1 - blk.k0;
    const uint32_t n_tok   = blk.t1 - blk.t0;
    // Block columns in (t asc, k asc) = llama layout rows (t in block), k fastest.
    std::vector<mix_scatter_t> scatter;
    scatter.reserve(static_cast<size_t>(width) * n_tok);
    for (uint32_t t = blk.t0; t < blk.t1; ++t)
        for (uint32_t k = blk.k0; k < blk.k1; ++k)
            scatter.push_back({ t, k });
    // Resolve the pool/subpool + column layout from the block's first expert.
    const mix_scatter_t & sc0 = scatter[0];
    const int32_t e0 = ids_compact[static_cast<size_t>(sc0.t) * n_k + sc0.k];
    if (e0 < 0) return false;
    const int32_t slot0 = pin_slot(pins, layer, static_cast<uint32_t>(e0));
    if (slot0 < 0) return false;
    const expert_scheduler::subpool_t * sp = sched.subpool_of_slot(slot0);
    if (!sp) return false;
    size_t col_off = 0, col_stride = 0;
    uint32_t col_index = 0;
    if (!sched.column_layout(*sp, w->name, col_off, col_stride, col_index)) return false;
    // Pool-local slot ids for the block columns.
    std::vector<int32_t> ids_sub(scatter.size());
    for (size_t idx = 0; idx < scatter.size(); ++idx) {
        const int32_t e = ids_compact[static_cast<size_t>(scatter[idx].t) * n_k + scatter[idx].k];
        const int32_t slot = pin_slot(pins, layer, static_cast<uint32_t>(e));
        if (slot < 0 || sched.subpool_of_slot(slot) != sp) return false;  // mixed subpool: abort this block
        ids_sub[idx] = slot - static_cast<int32_t>(sp->slot_begin);
    }
    std::vector<float> cur_sub;
    size_t d_in = 0;
    if (!build_cur_sub(cur, scatter, width, n_tok, cur_sub, d_in)) return false;
    std::vector<float> out;
    if (!exec_round_cpu(ctx, cpu, w, *sp, col_off, col_stride, ids_sub, width,
                        n_tok, cur_sub, d_in, nd, out)) {
        return false;
    }
    scatter_sub_dst(dst, dst_nk, scatter, width, n_tok, out.data(), d_out, sizeof(float));
    return true;
}

// Compare the full-width result (`ref`, main dst) against a block re-run that
// scatters every block into `dst` (zeroed first). Byte-exact required; the
// compact mm is the same CPU kernel as the full round, just split by columns,
// so the per-column result must be bit-identical.
static bool tmp_split_verify(ggml_context * ctx, ggml_backend_t cpu,
                             expert_scheduler & sched, ggml_tensor * nd,
                             uint32_t layer, const std::vector<expert_handle_t>& pins,
                             const std::vector<int32_t>& ids_compact,
                             const std::vector<int32_t>& expert_pool,
                             const moe_model_topology_t & topo,
                             uint32_t n_k, uint32_t n_t,
                             const uint8_t * ref, const char * which) {
    const size_t d_out = static_cast<size_t>(nd->ne[0]);
    const size_t dst_bytes = ggml_nbytes(nd);
    std::vector<uint8_t> dst(dst_bytes, 0);
    const auto blocks = tmp_split_blocks(which, n_k, n_t);
    size_t cols = 0;
    for (const auto & b : blocks) {
        if (!tmp_run_block(ctx, cpu, sched, nd, layer, pins, ids_compact, expert_pool,
                           topo, n_k, b, dst.data(), n_k, d_out)) {
            fprintf(stderr, "[split] %-14s block [k %u..%u) x [t %u..%u) SKIPPED (setup)\n",
                    which, b.k0, b.k1, b.t0, b.t1);
            return false;
        }
        cols += (size_t)(b.k1 - b.k0) * (b.t1 - b.t0);
    }
    // A valid partition covers every (k,t) exactly once.
    if (cols != (size_t) n_k * n_t) {
        fprintf(stderr, "[split] %-14s block coverage %zu != %u x %u\n", which, cols, n_k, n_t);
        return false;
    }
    const size_t nf = dst_bytes / sizeof(float);
    const float * a = (const float *) ref;
    const float * b = (const float *) dst.data();
    size_t first = nf;
    for (size_t i = 0; i < nf; ++i) {
        if (a[i] != b[i]) { first = i; break; }
    }
    const bool ok = first == nf;
    fprintf(stderr, "[split] %-14s blocks=%zu cols=%zu bytes=%zu %s",
            which, blocks.size(), cols, dst_bytes, ok ? "PASS (byte-identical)" : "FAIL");
    if (!ok) fprintf(stderr, " (first diff @ elem %zu)", first);
    fprintf(stderr, "\n");
    return ok;
}

// Entry gate for the split self-test; runs the requested cut families against
// the just-computed full-width main dst.
static void tmp_split_test(const char * which, ggml_context * ctx, ggml_backend_t cpu,
                           expert_scheduler & sched, ggml_tensor * nd, uint32_t layer,
                           const std::vector<expert_handle_t>& pins,
                           const std::vector<int32_t>& ids_compact,
                           const std::vector<int32_t>& expert_pool,
                           const moe_model_topology_t & topo,
                           uint32_t n_k, uint32_t n_t, uint8_t * main_dst) {
    std::vector<std::string> families;
    if (std::string(which) == "all") {
        families = { "full", "vertical", "vertical_asym", "horizontal",
                     "horizontal_asym", "mixed2x2" };
    } else {
        families = { which };
    }
    for (const auto & f : families) {
        tmp_split_verify(ctx, cpu, sched, nd, layer, pins, ids_compact, expert_pool,
                         topo, n_k, n_t, main_dst, f.c_str());
    }
}

// ---- bucket-granularity dump (TEMP) ---------------------------------------
// Dumps every compact mm ROUND (one bucket) of a mixed mm node: the round's
// pool-local slot ids, the rebuilt activation, the compact mm output and its
// (t,k) scatter map. Offline tooling can then place each bucket's output back
// into the full-width main dst and compare column-by-column against a
// full-width baseline dump, WITHOUT waiting for the whole layer burst to
// finish. Controlled by env (independent from STREAM_MOE_TMP_DUMP):
//   STREAM_MOE_TMP_BUCKET_DUMP       =1 enable
//   STREAM_MOE_TMP_BUCKET_DUMP_DIR    dump directory (default temp/tmp_bucket_dump)
//   STREAM_MOE_TMP_BUCKET_DUMP_LAYER  layer to dump (default 0)
// Self-contained (own mkdir/write) so it can live before the tmp_dump_*
// harness further down the file.

static void tmp_bucket_mkdirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || path[i] == '\\') {
#ifdef _WIN32
            _mkdir(cur.c_str());
#else
            ::mkdir(cur.c_str(), 0755);
#endif
        }
    }
    if (!cur.empty()) {
#ifdef _WIN32
        _mkdir(cur.c_str());
#else
        ::mkdir(cur.c_str(), 0755);
#endif
    }
}
static void tmp_bucket_write(const std::string & path, const void * data, size_t bytes) {
    tmp_bucket_mkdirs(path.substr(0, path.find_last_of("/\\")));
    if (!data || bytes == 0) return;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { LOG_ERROR("stream_moe: [tmp] bucket dump open failed " << path); return; }
    const size_t wr = fwrite(data, 1, bytes, f);
    fclose(f);
    if (wr != bytes) LOG_ERROR("stream_moe: [tmp] bucket dump short write " << path);
}
static void tmp_bucket_meta(const std::string & path, int64_t ne0, int64_t ne1, int64_t ne2,
                            const char * type, size_t nbytes, const char * tag) {
    char j[512];
    snprintf(j, sizeof(j),
             "{\"ne\":[%lld,%lld,%lld],\"type\":\"%s\",\"nbytes\":%zu,\"tag\":\"%s\"}\n",
             (long long) ne0, (long long) ne1, (long long) ne2, type, nbytes, tag ? tag : "");
    tmp_bucket_write(path + ".json", j, strlen(j));
}
struct tmp_bucket_cfg_t {
    bool on = false;
    std::string dir;
    int32_t layer = 0;
};
const tmp_bucket_cfg_t & tmp_bucket_cfg() {
    static tmp_bucket_cfg_t c = [] {
        tmp_bucket_cfg_t r;
        const char * en = std::getenv("STREAM_MOE_TMP_BUCKET_DUMP");
        if (en && std::string(en) == "1") {
            r.on = true;
            const char * d = std::getenv("STREAM_MOE_TMP_BUCKET_DUMP_DIR");
            r.dir = d && *d ? d : "temp/tmp_bucket_dump";
            const char * l = std::getenv("STREAM_MOE_TMP_BUCKET_DUMP_LAYER");
            if (l && *l) r.layer = atoi(l);
        }
        return r;
    } ();
    return c;
}
// Dump one compact round (bucket) of a mixed mm node.
//  - ids_sub: pool-local slot ids [width * n_active]
//  - cur_sub: rebuilt activation [d_in, width, n_active]
//  - out:     compact mm output [d_out, width, n_active]
//  - scatter: (t,k) sidecar, one "t k" per (a*width+s), text
static void tmp_dump_bucket_round(uint32_t layer, size_t round_idx, ggml_tensor * nd,
                                  uint32_t pool, uint32_t width, uint32_t n_active,
                                  const std::vector<int32_t>& ids_sub,
                                  const std::vector<float>& cur_sub, size_t d_in,
                                  const std::vector<float>& out,
                                  const std::vector<mix_scatter_t>& scatter) {
    const tmp_bucket_cfg_t & cfg = tmp_bucket_cfg();
    if (!cfg.on || static_cast<int32_t>(layer) != cfg.layer) return;
    if (out.empty() || scatter.empty()) return;
    char sub[64];
    snprintf(sub, sizeof(sub), "L%u", layer);
    const std::string dir = cfg.dir + "/" + sub;
    char b[64];
    snprintf(b, sizeof(b), "r%03zu_p%u_w%u_a%u", round_idx, pool, width, n_active);
    const size_t d_out = static_cast<size_t>(nd->ne[0]);
    const size_t ncols = static_cast<size_t>(width) * n_active;
    const size_t esize = sizeof(float);
    // weight name, sanitized, for the offline alignment key
    const char * wn = (nd->src[0] && nd->src[0]->name) ? nd->src[0]->name : "anon";
    std::string s(wn);
    for (auto & ch : s) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '*' || ch == '?' ||
            ch == '"' || ch == '<' || ch == '>' || ch == '|') ch = '_';
    }
    const std::string pre = dir + "/" + s + "_" + b;
    tmp_bucket_write(pre + ".ids_sub.bin", ids_sub.data(), ids_sub.size() * sizeof(int32_t));
    tmp_bucket_meta(pre + ".ids_sub.bin", ncols, 1, 1, "i32", ids_sub.size() * sizeof(int32_t), "ids_sub");
    if (!cur_sub.empty()) {
        tmp_bucket_write(pre + ".cur_sub.bin", cur_sub.data(), cur_sub.size() * sizeof(float));
        tmp_bucket_meta(pre + ".cur_sub.bin", static_cast<int64_t>(d_in), width, n_active,
                        "f32", cur_sub.size() * esize, "cur_sub");
    }
    tmp_bucket_write(pre + ".out.bin", out.data(), out.size() * sizeof(float));
    tmp_bucket_meta(pre + ".out.bin", static_cast<int64_t>(d_out), width, n_active,
                    "f32", out.size() * esize, "out");
    // scatter as text: one "t k" per (a*width+s) row
    std::string sc;
    sc.reserve(scatter.size() * 12);
    for (const auto & e : scatter) {
        char line[32];
        snprintf(line, sizeof(line), "%u %u\n", e.t, e.k);
        sc += line;
    }
    tmp_bucket_write(pre + ".scatter.txt", sc.data(), sc.size());
}
#endif // STREAM_MOE_TEMP

// Execute one mixed MUL_MAT_ID node: run each pool's peel rounds on that pool's
// backend, scatter each round's compact output into the (hide_burst'd) main dst.
// `ids_compact` = llama-layout expert ids [n_k, n_t] (k fastest) compacted from
// the sparse routing tensor; `expert_pool` maps expert -> owning pool. Main dst
// is nd->data (already hide_burst'd by the caller).
static bool exec_mixed_mm(ggml_context * ctx, ggml_backend_t cpu,
                          expert_scheduler & sched, const moe_model_topology_t & topo,
                          ggml_tensor * nd, int32_t layer,
                          const std::vector<expert_handle_t>& pins,
                          const std::vector<int32_t>& ids_compact,
                          const std::vector<int32_t>& expert_pool) {
    ggml_tensor * w   = const_cast<ggml_tensor*>(nd->src[0]);
    ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
    ggml_tensor * ids = const_cast<ggml_tensor*>(nd->src[2]);
    const uint32_t n_k = static_cast<uint32_t>(ids->ne[0]);
    const uint32_t n_t = static_cast<uint32_t>(ids->ne[1]);
    if (cur->type != GGML_TYPE_F32) {
        LOG_ERROR("stream_moe: mixed mm cur not f32 for " << (w->name ? w->name : "?"));
        return false;
    }
    const size_t d_out = static_cast<size_t>(nd->ne[0]);
    if (nd->type != GGML_TYPE_F32) {
        LOG_ERROR("stream_moe: mixed mm dst not f32 for " << (w->name ? w->name : "?"));
        return false;
    }

    mix_plan_t plan = build_mix_plan(ids_compact.data(), n_k, n_t,
                                     expert_pool.data(), topo.n_expert, sched.n_pools());
#ifdef STREAM_MOE_TEMP
    // "Artificial multi-bucket" switch: when the natural plan is a SINGLE
    // full-width round (one pool owns every column), optionally re-split the
    // (k,t) column domain into multiple rectangular buckets and execute them
    // as independent rounds - same exec loop, same scatter, same tail fold.
    // This is the exec-layer stand-in for real multi-pool/multi-device buckets
    // while debugging the bucket-chain executor (docs/M2_DEVICE_EXECUTOR.md
    // SS7.3). Env STREAM_MOE_TMP_FORCE_BUCKETS selects the cut (a tmp_split_*
    // family). Each bucket round still dumps via STREAM_MOE_TMP_BUCKET_DUMP, so
    // per-bucket placement can be verified offline.
    {
        const bool single_full = plan.rounds.size() == 1 &&
            plan.rounds[0].width == n_k && plan.rounds[0].n_active == n_t;
        const char * fb = std::getenv("STREAM_MOE_TMP_FORCE_BUCKETS");
        if (single_full && fb && fb[0] && std::string(fb) != "full") {
            const auto blocks = tmp_split_blocks(fb, n_k, n_t);
            if (blocks.size() > 1) {
                mix_plan_t fp = tmp_plan_from_blocks(plan, blocks, n_k, ids_compact);
                fprintf(stderr, "[force_buckets] L%d %s: 1 full round -> %zu buckets\n",
                        layer, w->name ? w->name : "?", (size_t) fp.rounds.size());
                plan = std::move(fp);
            }
        }
    }
#endif
    if (plan.rounds.empty()) return true;

    uint8_t * main_dst = static_cast<uint8_t*>(nd->data);
    const size_t esize = sizeof(float);

    size_t round_idx = 0;
    for (const auto & r : plan.rounds) {
        if (r.ids.empty()) continue;
        // subpool owning this round's experts (first slot decides)
        const mix_scatter_t & sc0 = r.scatter[0];
        const int32_t e0 = ids_compact[static_cast<size_t>(sc0.t) * n_k + sc0.k];
        const int32_t slot0 = pin_slot(pins, static_cast<uint32_t>(layer), static_cast<uint32_t>(e0));
        const expert_scheduler::subpool_t * sp = slot0 < 0 ? nullptr : sched.subpool_of_slot(slot0);
        if (!sp) {
            LOG_ERROR("stream_moe: mixed round has no subpool (pool " << r.pool << ")");
            return false;
        }
#ifdef STREAM_MOE_TEMP
        if (std::getenv("STREAM_MOE_TMP_DUMP")) {
            LOG_INFO("stream_moe: [tmp] mixed round pool=" << r.pool << " width=" << r.width
                     << " active=" << r.n_active << " first_e=" << e0 << " sp_pool=" << sp->pool);
        }
#endif
        size_t col_off = 0, col_stride = 0;
        uint32_t col_index = 0;
        if (!sched.column_layout(*sp, w->name, col_off, col_stride, col_index)) {
            LOG_ERROR("stream_moe: mixed no column for " << w->name << " in pool " << sp->pool);
            return false;
        }
        // pool-local slot ids for this round
        std::vector<int32_t> ids_sub(r.ids.size());
        for (size_t idx = 0; idx < r.ids.size(); ++idx) {
            const mix_scatter_t & sc = r.scatter[idx];
            const int32_t e = ids_compact[static_cast<size_t>(sc.t) * n_k + sc.k];
            const int32_t slot = pin_slot(pins, static_cast<uint32_t>(layer), static_cast<uint32_t>(e));
            if (slot < 0) { LOG_ERROR("stream_moe: mixed round expert not pinned"); return false; }
            ids_sub[idx] = slot - static_cast<int32_t>(sp->slot_begin);
        }
        // rebuilt activation columns
        std::vector<float> cur_sub;
        size_t d_in = 0;
        if (!build_cur_sub(cur, r.scatter, r.width, r.n_active, cur_sub, d_in)) return false;

        std::vector<float> out;
        const bool ok = sp->pool == 0
            ? exec_round_cpu(ctx, cpu, w, *sp, col_off, col_stride, ids_sub, r.width, r.n_active, cur_sub, d_in, nd, out)
            : exec_round_vk(ctx, w, *sp, col_off, col_stride, ids_sub, r.width, r.n_active, cur_sub, d_in, out);
        if (!ok) return false;
#ifdef STREAM_MOE_TEMP
        if (std::getenv("STREAM_MOE_TMP_BUCKET_DUMP")) {
            tmp_dump_bucket_round(static_cast<uint32_t>(layer), round_idx, nd,
                                  r.pool, r.width, r.n_active, ids_sub, cur_sub, d_in,
                                  out, r.scatter);
        }
#endif
        scatter_sub_dst(main_dst, n_k, r.scatter, r.width, r.n_active, out.data(), d_out, esize);
        ++round_idx;
    }
#ifdef STREAM_MOE_TEMP
    // Column-split self-test: single full-width round (all columns to one pool)
    // is the baseline - recompute the same mm as arbitrary rectangular column
    // buckets and require a byte-identical main dst. Env STREAM_MOE_TMP_SPLIT
    // selects the cut families ("all" runs every one).
    if (plan.rounds.size() == 1) {
        const char * sp_name = std::getenv("STREAM_MOE_TMP_SPLIT");
        if (sp_name && sp_name[0]) {
            tmp_split_test(sp_name, ctx, cpu, sched, nd, static_cast<uint32_t>(layer),
                           pins, ids_compact, expert_pool, topo, n_k, n_t, main_dst);
        }
    }
#endif
    return true;
}


#ifdef STREAM_MOE_TEMP
// ---- L0 binary dump harness (temporary diagnostics, STREAM_MOE_TEMP only) --
// Dumps full raw bytes of per-layer entry/exit data so a pure-CPU run and a
// mixed-RAM/VRAM run can be compared node-by-node and expert-by-expert.
// Controlled by env:
//   STREAM_MOE_TMP_DUMP      =1 enable
//   STREAM_MOE_TMP_DUMP_DIR   dump directory (created on demand)
//   STREAM_MOE_TMP_DUMP_LAYER layer to dump ("all" = every layer; default 0)
//   STREAM_MOE_TMP_DUMP_OUT_ONLY =1 only ffn_moe_out (exit) nodes (mixed run)
// Layout of one node's files under <dir>/L<layer>/i<seq>_<op>_<tag>:
//   .bin                 full output bytes (after execution)
//   .cur.bin             mm activation input src[1] (mm only)
//   .ids.bin             routing ids src[2] (mm only)
//   .w_e<N>.bin          resident weight column slice of expert N (mm only)
struct tmp_dump_cfg_t {
    bool   on = false, out_only = false;
    bool   mm_only = false;   // keep only mm outputs + moe_out (small offline gate)
    std::string dir;
    int32_t layer = 0;   // -1 = all layers
};
const tmp_dump_cfg_t & tmp_dump_cfg() {
    static tmp_dump_cfg_t c = [] {
        tmp_dump_cfg_t r;
        const char * en = std::getenv("STREAM_MOE_TMP_DUMP");
        if (en && std::string(en) == "1") {
            r.on = true;
            const char * d  = std::getenv("STREAM_MOE_TMP_DUMP_DIR");
            r.dir = d && *d ? d : "temp/tmp_l0_dump";
            const char * l  = std::getenv("STREAM_MOE_TMP_DUMP_LAYER");
            if (l && *l) r.layer = std::string(l) == "all" ? -1 : atoi(l);
            const char * o  = std::getenv("STREAM_MOE_TMP_DUMP_OUT_ONLY");
            if (o && std::string(o) == "1") r.out_only = true;
            const char * m  = std::getenv("STREAM_MOE_TMP_DUMP_MM_ONLY");
            if (m && std::string(m) == "1") r.mm_only = true;
        }
        return r;
    } ();
    return c;
}
static void tmp_mkdirs(const std::string & path) {
    // create every missing level (Windows fopen fails on a missing dir); the
    // cumulative prefix is kept so later segments nest under the earlier ones.
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || path[i] == '\\') {
#ifdef _WIN32
            _mkdir(cur.c_str());
#else
            ::mkdir(cur.c_str(), 0755);
#endif
        }
    }
    if (!cur.empty()) {
#ifdef _WIN32
        _mkdir(cur.c_str());
#else
        ::mkdir(cur.c_str(), 0755);
#endif
    }
}
static void tmp_dump_write(const std::string & dir, const std::string & fname,
                           const void * data, size_t bytes) {
    tmp_mkdirs(dir);
    if (!data || bytes == 0) return;
    const std::string path = dir + "/" + fname;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { LOG_ERROR("stream_moe: [tmp] dump open failed " << path); return; }
    const size_t wr = fwrite(data, 1, bytes, f);
    fclose(f);
    if (wr != bytes) LOG_ERROR("stream_moe: [tmp] dump short write " << path << " (" << wr << "/" << bytes << ")");
}
// Sidecar metadata JSON for an accompanying raw .bin: the tensor's own
// ne/nb/type/op/name so an OFFLINE tool can re-slice the raw bytes into
// per-expert columns without any contiguity assumption. .bin stays pure raw.
static void tmp_dump_meta(const std::string & dir, const std::string & fname,
                          const ggml_tensor * t) {
    if (!t) return;
    char j[512];
    const char * ty = ggml_type_name(t->type);
    snprintf(j, sizeof(j),
             "{\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu],"
             "\"type\":\"%s\",\"nbytes\":%zu,\"op\":\"%s\",\"name\":\"%s\"}\n",
             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
             t->nb[0], t->nb[1], t->nb[2], t->nb[3],
             ty ? ty : "?", ggml_nbytes(t), ggml_op_name(t->op),
             t->name ? t->name : "");
    tmp_dump_write(dir, fname + ".json", j, strlen(j));
}
// Write a tensor's raw bytes + its metadata sidecar under one base name.
static void tmp_dump_tensor(const std::string & dir, const std::string & fname,
                            const ggml_tensor * t, const void * data, size_t bytes) {
    tmp_dump_write(dir, fname, data, bytes);
    tmp_dump_meta(dir, fname, t);
}
static std::string tmp_sanitize(const char * s) {
    if (!s || !*s) return "anon";
    std::string r;
    for (const char * p = s; *p; ++p) {
        const char ch = *p;
        r += (ch == '/' || ch == '\\' || ch == ':' || ch == '*' || ch == '?' ||
              ch == '"' || ch == '<' || ch == '>' || ch == '|') ? '_' : ch;
    }
    return r;
}
// Dump a captured node after execution. Entry side (cur/ids/weight slices) is
// dumped for mm nodes in full mode; exit data = the node's own output bytes.
static void tmp_dump_node(expert_scheduler & sched, const moe_model_topology_t & topo,
                          const std::vector<expert_handle_t> & pins, int32_t layer,
                          size_t seq, ggml_tensor * nd) {
    const tmp_dump_cfg_t & cfg = tmp_dump_cfg();
    if (!cfg.on || (cfg.layer >= 0 && layer != cfg.layer)) return;
    const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
    if (cfg.out_only && !is_out) return;
    const bool is_mm = nd->op == GGML_OP_MUL_MAT_ID;
    if (cfg.mm_only && !is_mm && !is_out) return;   // mm_only: skip intermediate nodes
    char sub[64];
    snprintf(sub, sizeof(sub), "L%d", layer);
    const std::string dir = cfg.dir + "/" + sub;
    // Owner pool prefix (offline alignment key): the pool this data lives on /
    // was produced by. `cpu_` (pool 0) vs `gpu_` (any device pool). For an mm
    // the prefix follows the region its active set executed on (rescanned here
    // from the pinned slots - exec already ran, so use_sp is unambiguous);
    // weight slices use their own slot's pool (a mixed layer then produces
    // per-expert files under both prefixes, which is the whole point).
    uint32_t mm_pool = 0;
    if (is_mm) {
        const ggml_tensor * ids = nd->src[2];
        if (ids && ids->data) {
            for (int t = 0; t < ids->ne[1]; ++t) {
                for (int k = 0; k < ids->ne[0]; ++k) {
                    const int32_t e = MOE_ID_AT(ids, t, k);
                    if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) continue;
                    const int32_t slot = pin_slot(pins, (uint32_t)layer, (uint32_t)e);
                    if (slot < 0) continue;
                    const expert_scheduler::subpool_t * osp = sched.subpool_of_slot(slot);
                    if (osp && osp->pool != 0) { mm_pool = osp->pool; break; }
                }
                if (mm_pool != 0) break;
            }
        }
    }
    const char * mm_pre = mm_pool != 0 ? "gpu" : "cpu";
    const char * node_pre = (is_mm && mm_pool != 0) ? "gpu" : "cpu";
    char base[256];
    snprintf(base, sizeof(base), "i%03zu_%s_%s", seq, ggml_op_name(nd->op),
             tmp_sanitize(nd->name ? nd->name : "").c_str());
    if (is_mm && !cfg.out_only) {
        // routing ids always (column->expert map; the offline gate reads it);
        // activation + per-expert weight slices only in full mode.
        const ggml_tensor * ids = nd->src[2];
        if (ids && ids->data) {
            tmp_dump_tensor(dir, std::string(node_pre) + "_" + base + ".ids.bin", ids, ids->data, ggml_nbytes(ids));
        }
        if (!cfg.mm_only) {
        // entry: activation input + routing ids + resident weight column slices
        const ggml_tensor * cur = nd->src[1];
        if (cur && cur->data) {
            tmp_dump_tensor(dir, std::string(node_pre) + "_" + base + ".cur.bin", cur, cur->data, ggml_nbytes(cur));
        }
        // weight slice per active expert (SoA column slice at its slot); the
        // filename carries the expert's OWN pool prefix (mm_pre is the layer
        // burst owner; individual experts may still live elsewhere).
        const ggml_tensor * w = nd->src[0];
        if (w && w->name && ids && ids->data && topo.n_expert > 0) {
            std::vector<bool> done(topo.n_expert, false);
            for (int t = 0; t < ids->ne[1]; ++t) {
                for (int k = 0; k < ids->ne[0]; ++k) {
                    const int32_t e = MOE_ID_AT(ids, t, k);
                    if (e < 0 || e >= static_cast<int32_t>(topo.n_expert) || done[e]) continue;
                    done[e] = true;
                    const int32_t slot = pin_slot(pins, (uint32_t)layer, (uint32_t)e);
                    if (slot < 0) continue;
                    const expert_scheduler::subpool_t * sp = sched.subpool_of_slot(slot);
                    if (!sp) continue;
                    size_t col_off = 0, col_stride = 0; uint32_t ci = 0;
                    if (!sched.column_layout(*sp, w->name, col_off, col_stride, ci)) continue;
                    const uint8_t * p = sp->base + col_off +
                        static_cast<size_t>(slot - static_cast<int32_t>(sp->slot_begin)) * col_stride;
                    const char * pre = sp->pool != 0 ? "gpu" : "cpu";
                    char wf[256];
                    snprintf(wf, sizeof(wf), "%s_%s.w_e%d.bin", pre, base, e);
                    // A weight slice is one expert's compact column (stride
                    // bytes) of the full tensor - not the whole tensor. Emit
                    // the slice's own minimal meta (no ne/nb: offline use is
                    // per-expert byte comparison, not column re-slicing).
                    tmp_dump_write(dir, wf, p, col_stride);
                    char j[256];
                    snprintf(j, sizeof(j),
                             "{\"type\":\"%s\",\"nbytes\":%zu,\"expert\":%d,\"slice_of\":\"%s\"}\n",
                             ggml_type_name(w->type), col_stride, e, w->name ? w->name : "");
                    tmp_dump_write(dir, std::string(wf) + ".json", j, strlen(j));
                }
            }
        }
        }   // !cfg.mm_only
    }
    // exit: the node's own output bytes (full run only; out_only already filtered)
    if (nd->data) {
        tmp_dump_tensor(dir, std::string(node_pre) + "_" + base + ".bin", nd, nd->data, ggml_nbytes(nd));
    }
}
#endif // STREAM_MOE_TEMP

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

        uint32_t gidx = sched.group_of(pn.layer);
        if (gidx == static_cast<uint32_t>(-1)) return false;
        const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
        // Resolve the pool each (k,t) expert lives on, then run the mm through
        // exec_mixed_mm: per-pool peel rounds, each on its own backend. Single
        // pool = one round; multi pool = one round per pool. (The old single-
        // pool fast paths - exec_mm_vk / the whole-layer CPU mini-graph - were
        // removed: they duplicated the executor and the device one misbehaved
        // on large prefill batches.)
        std::vector<int32_t> expert_pool(topo.n_expert, -1);
        std::vector<int32_t> ids_compact(n_ids, -1);
        bool any_active = false;
        for (int t = 0; t < ids->ne[1]; ++t) {
            for (int k = 0; k < ids->ne[0]; ++k) {
                const int32_t e = MOE_ID_AT(ids, t, k);
                if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) return false;
                const int32_t slot = pin_slot(pins, pn.layer, static_cast<uint32_t>(e));
                if (slot < 0) return false;
                const expert_scheduler::subpool_t* osp = sched.subpool_of_slot(slot);
                if (!osp) return false;
                ids_compact[static_cast<size_t>(t) * ids->ne[0] + k] = e;
                expert_pool[e] = static_cast<int32_t>(osp->pool);
                any_active = true;
            }
        }
        if (!any_active) return true;   // no active experts
        const bool ok_m = exec_mixed_mm(ctx, cpu, sched, topo, nd, layer, pins,
                                        ids_compact, expert_pool);
        if (ok_m) refresh_aliases(layer, nd);
        return ok_m;
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

// Whole-layer clone-graph executor, refactored into append_* units (GPU
// mini-graph shape on CPU, env STREAM_MOE_TMP_CHAIN_GRAPH). Rebuild the layer
// as ONE cgraph / one graph_compute: for each bucket append the bucket's chain
// (mm shell + weightless clones), fold the bucket's experts (sum over w) into
// per-token columns, scatter into the full-width accumulator, then the exit
// writes moe_out's dst. This is the CPU twin of the per-device mini-graph
// builder (docs/M2_DEVICE_EXECUTOR.md SS2/SS7.7). Currently ONE full-width
// bucket per layer (single device / all columns); the append_scatter step is a
// WIDTH-CHECK mock until multi-bucket token subsets land.
//
// Clone srcs are wrapped as LEAVES carrying the main-graph data pointer (already
// positioned by the hide/refresh flow) so the graph has NO external op nodes;
// nodes are added with ggml_build_forward_expand, executed in layer order, and
// inter-node data flows through the shared arena regions (producer clone writes
// its hide region, consumer clone reads it via its src leaf).

struct chain_ctx_t {
    ggml_context *             ctx  = nullptr;
    ggml_backend_t             cpu  = nullptr;
    expert_scheduler *         sched = nullptr;
    const moe_model_topology_t * topo = nullptr;
    int32_t                    layer = -1;
    const std::vector<expert_handle_t> * pins = nullptr;
    ggml_cgraph *              gf = nullptr;
    const moe_layer_exec_t *   ex = nullptr;
    // in-flight bucket geometry (full-width today: w == ids->ne[0])
    int64_t                    w_b = 0;   // experts per token in this bucket
    int64_t                    n_t = 0;   // tokens (full-width == ids->ne[1])
    // slot-local ids buffers per mm, kept alive until graph_compute (their
    // data pointers feed ids leaves; vector moves keep the heap data address).
    std::vector<std::vector<int32_t>> mm_ids_pool;
    // float buffers for no_alloc fold intermediates (kept alive until compute)
    std::vector<std::vector<float>>    fold_buf;
    // fold_repl: skip the anonymous per-topk fold + moe_out on the graph and
    // produce them via append_expert_fold + exit memcpy instead. Off = the old
    // IDENTICAL whole-clone behaviour (bring-up gate).
    bool fold_repl = false;
    // SS7.8 multi-device structure (single-device CPU phase: device_used == 1):
    //   per-device acc_d[d_out, n_t] = the device's expert-folded output (the
    //   fold result lands here), and RAM add_in[device_used, d_out, n_t] = the
    //   anonymous-add input, one slot per device (index k = device ordinal).
    int64_t                 d_out = 0;
    int64_t                 device_used = 0;
    std::vector<float>      add_in;     // host mirror of the RAM [device_used, d_out, n_t]
};

// Append a routed mm as a shell mul_mat_id writing straight into the hidden dst
// (full-width round assumption: every (k,t) expert in ONE subpool). Returns the
// mm tensor.
static ggml_tensor * append_mm_shell(chain_ctx_t & c, ggml_tensor * nd) {
    ggml_tensor * w   = const_cast<ggml_tensor*>(nd->src[0]);
    ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
    ggml_tensor * ids = const_cast<ggml_tensor*>(nd->src[2]);
    if (!ids->data || !cur->data) return nullptr;
    parsed_node_t pn = parse_weight_name(w->name);
    if (!pn.ok) return nullptr;
    uint32_t gidx = c.sched->group_of(pn.layer);
    if (gidx == static_cast<uint32_t>(-1)) return nullptr;

    const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
    int32_t slot_begin = -1;
    c.mm_ids_pool.emplace_back(n_ids, 0);           // keep alive until graph_compute
    std::vector<int32_t> & mm_ids = c.mm_ids_pool.back();
    for (size_t i = 0; i < n_ids; ++i) {
        const int32_t e = MOE_ID_AT(ids, (int)(i / (size_t)ids->ne[0]), (int)(i % (size_t)ids->ne[0]));
        if (e < 0 || e >= static_cast<int32_t>(c.topo->n_expert)) return nullptr;
        const int32_t slot = pin_slot(*c.pins, pn.layer, static_cast<uint32_t>(e));
        if (slot < 0) return nullptr;
        const expert_scheduler::subpool_t * osp = c.sched->subpool_of_slot(slot);
        if (!osp) return nullptr;
        if (slot_begin < 0) slot_begin = static_cast<int32_t>(osp->slot_begin);
        if (static_cast<int32_t>(osp->slot_begin) != slot_begin) return nullptr;
        mm_ids[i] = slot - static_cast<int32_t>(osp->slot_begin);
    }
    const int32_t se0 = MOE_ID_AT(ids, 0, 0);
    const expert_scheduler::subpool_t * sp = se0 < 0 ? nullptr
        : c.sched->subpool_of_slot(pin_slot(*c.pins, pn.layer, static_cast<uint32_t>(se0)));
    if (!sp) return nullptr;
    size_t col_off = 0, col_stride = 0; uint32_t col_index = 0;
    if (!c.sched->column_layout(*sp, w->name, col_off, col_stride, col_index)) return nullptr;

    ggml_tensor * w3d = ggml_new_tensor_3d(c.ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = static_cast<int32_t>(sp->n_slots);
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * sp->n_slots;
    w3d->data  = sp->base + col_off;

    ggml_tensor * ids_leaf = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
    ids_leaf->data = mm_ids.data();   // pool keeps it alive until compute

    ggml_tensor * cur_leaf = ggml_new_tensor_4d(c.ctx, cur->type,
                                                cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    cur_leaf->nb[0] = cur->nb[0]; cur_leaf->nb[1] = cur->nb[1];
    cur_leaf->nb[2] = cur->nb[2]; cur_leaf->nb[3] = cur->nb[3];
    cur_leaf->data = cur->data;

    ggml_tensor * mm = ggml_mul_mat_id(c.ctx, w3d, cur_leaf, ids_leaf);
    mm->data = nd->data;   // shell mm writes the hidden dst directly
    ggml_build_forward_expand(c.gf, mm);
    return mm;
}

// Clone a weightless / anonymous / moe_out op tensor: fresh node with identical
// op/params/ne/nb, every src wrapped as a leaf carrying the main-graph data
// pointer. Returns the clone.
static ggml_tensor * append_op_clone(chain_ctx_t & c, ggml_tensor * nd) {
    ggml_tensor * cl = ggml_new_tensor_4d(c.ctx, nd->type,
                                          nd->ne[0], nd->ne[1], nd->ne[2], nd->ne[3]);
    cl->nb[0] = nd->nb[0]; cl->nb[1] = nd->nb[1];
    cl->nb[2] = nd->nb[2]; cl->nb[3] = nd->nb[3];
    cl->op = nd->op;
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS; ++i) cl->op_params[i] = nd->op_params[i];
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        ggml_tensor * src = nd->src[s];
        if (!src) continue;
        ggml_tensor * sv = ggml_new_tensor_4d(c.ctx, src->type,
                                              src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        sv->nb[0] = src->nb[0]; sv->nb[1] = src->nb[1];
        sv->nb[2] = src->nb[2]; sv->nb[3] = src->nb[3];
        sv->data = src->data;
        cl->src[s] = sv;
        if (!sv->data) return nullptr;
    }
    cl->data = nd->data;   // out = llama dst (not hidden); else hide position
    ggml_build_forward_expand(c.gf, cl);
    return cl;
}

// Append one bucket's chain: walk the layer's compute sequence and clone the
// mm shells + weightless ops that belong to this bucket. When c.fold_repl is
// false the WHOLE sequence (including the anonymous per-topk fold and moe_out)
// is cloned - the old IDENTICAL behaviour; when true the anonymous fold is
// skipped (replaced by append_expert_fold). Returns the last clone.
static ggml_tensor * append_bucket_chain(chain_ctx_t & c) {
    ggml_tensor * last = nullptr;
    for (const auto * cn : c.ex->compute) {
        ggml_tensor * nd = const_cast<ggml_tensor*>(cn);
        if (is_view_op(nd)) continue;
        const bool is_mm = nd->op == GGML_OP_MUL_MAT_ID;
        const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
        const bool is_fold = !is_mm && !is_out &&
            nd->op == GGML_OP_ADD && !(nd->name && strstr(nd->name, "ffn_moe_"));
        if (is_mm) {
            last = append_mm_shell(c, nd);
            if (!last) return nullptr;
        } else if (is_out) {
            if (!c.fold_repl) {
                last = append_op_clone(c, nd);   // old behaviour: moe_out on graph
                if (!last) return nullptr;
            }   // fold_repl: exit writes it from the accumulator
        } else if (is_fold) {
            if (!c.fold_repl) {   // old behaviour: clone the anonymous adds
                last = append_op_clone(c, nd);
                if (!last) return nullptr;
            }   // fold_repl: skipped, replaced by append_expert_fold
        } else {
            last = append_op_clone(c, nd);
            if (!last) return nullptr;
        }
    }
    return last;
}

// Fold the bucket's experts: sum over the per-token expert axis (w_b, currently
// the full ids->ne[0]) so the weighted output [d_out, w_b, n_t] becomes a
// per-token column [d_out, n_t]. Returns the folded tensor (width == n_t).
// Intermediate/output buffers are manually kept alive (no_alloc ctx).
static ggml_tensor * append_expert_fold(chain_ctx_t & c, ggml_tensor * weighted) {
    if (!weighted) return nullptr;
    const int64_t ne0 = weighted->ne[0];   // d_out
    const int64_t nw  = weighted->ne[1];   // w_b (experts per token)
    const int64_t nt  = weighted->ne[2];   // n_t tokens
    // permute (view) so the expert axis (ne1) becomes ne0, materialise it
    // contiguous (sum_rows needs nb0 == esize), then sum over the expert axis:
    // sum_rows -> [1, d_out, n_t]; cont to [d_out, n_t].
    ggml_tensor * p   = ggml_permute(c.ctx, weighted, 1, 0, 2, 3);   // view: [nw, d_out, n_t]
    ggml_tensor * pc  = ggml_cont(c.ctx, p);                         // contiguous [nw, d_out, n_t]
    c.fold_buf.emplace_back((size_t)(nw * ne0 * nt), 0.0f);
    pc->data = c.fold_buf.back().data();
    ggml_tensor * s   = ggml_sum_rows(c.ctx, pc);                    // [1, d_out, n_t]
    c.fold_buf.emplace_back((size_t)(ne0 * nt), 0.0f);               // s data
    s->data = c.fold_buf.back().data();
    ggml_tensor * acc = ggml_cont_2d(c.ctx, s, ne0, nt);             // [d_out, n_t]
    if (!acc) return nullptr;
    c.fold_buf.emplace_back((size_t)(ne0 * nt), 0.0f);               // acc data
    acc->data = c.fold_buf.back().data();
    ggml_build_forward_expand(c.gf, acc);
    c.w_b = 1;
    c.n_t = nt;
    return acc;
}

// Scatter the bucket's per-token columns into the full-width accumulator
// [d_out, n_t]. MOCK: for the current single full-width bucket the per-token
// fold already has width == n_t, so nothing to scatter; assert width parity and
// abort loudly if a future bucket needs a real (sub-token) scatter.
static bool append_scatter_to_fullwidth(chain_ctx_t & c, ggml_tensor * per_token) {
    if (!per_token) return false;
    const int64_t width = per_token->ne[1];   // n_tokens the bucket produced
    if (width == c.n_t) {
        return true;   // full-width bucket: already in accumulator geometry
    }
    fprintf(stderr,
        "[stream_moe] append_scatter_to_fullwidth: MOCK ABORT - bucket width %lld != n_t %lld. "
        "Multi-bucket sub-token scatter not implemented yet.\n",
        (long long) width, (long long) c.n_t);
    return false;
}

// Exit: run the graph; when fold_repl, follow the SS7.8 three-stage shape:
//   1) graph_compute (the whole bucket chain + expert folds run here);
//   2) copy each participating device's acc_d into its RAM add_in slot
//      (CPU phase: memcpy; GPU phase: one ggml_backend_tensor_copy per device,
//      device->host via tensor_get -> vulkan get_tensor -> vk_buffer_read);
//   3) host fold: sum add_in's device_used slots -> [d_out, n_t] -> moe_out.
static bool chain_exit(chain_ctx_t & c, ggml_tensor * acc) {
    if (ggml_backend_graph_compute(c.cpu, c.gf) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: chain graph compute failed for layer " << c.layer);
        return false;
    }
    if (!c.fold_repl) return true;   // moe_out computed on the graph

    ggml_tensor * moe_out = nullptr;
    for (const auto * cn : c.ex->compute) {
        if (cn->name && strstr(cn->name, "ffn_moe_out") != nullptr) { moe_out = const_cast<ggml_tensor*>(cn); break; }
    }
    if (!moe_out || !acc || !acc->data) return true;
    const size_t slot = (size_t)(c.d_out * c.n_t);

    // stage 2: acc_d (device-folded output, expert width 1) -> add_in[k]
    // Single device today (device_used == 1): k = 0. GPU phase: this is the
    // ggml_backend_tensor_copy(acc_d, add_in_slot) boundary.
    if (c.add_in.size() < slot * (size_t)c.device_used) c.add_in.resize(slot * (size_t)c.device_used);
    std::memcpy(c.add_in.data(), acc->data, slot * sizeof(float));

    // stage 3: host fold over device_used slots -> moe_out (dense is on CPU).
    // First slot overwrites moe_out, later slots accumulate.
    float * out = (float *) moe_out->data;
    for (size_t d = 0; d < (size_t)c.device_used; ++d) {
        const float * src = c.add_in.data() + d * slot;
        for (size_t i = 0; i < slot; ++i) out[i] = d == 0 ? src[i] : out[i] + src[i];
    }
    return true;
}

// Whole-layer single-graph path: for each bucket append its chain, fold its
// experts, scatter to full width, then the exit writes moe_out.
static enum ggml_status exec_layer_burst_chain(int32_t layer, ggml_context * ctx,
                                               ggml_backend_t cpu,
                                               expert_scheduler & sched,
                                               const moe_layer_exec_t * ex,
                                               const std::vector<expert_handle_t>& pins) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    chain_ctx_t c;
    c.ctx   = ctx;
    c.cpu   = cpu;
    c.sched = &sched;
    c.topo  = &sched.topology();
    c.layer = layer;
    c.pins  = &pins;
    c.gf    = gf;
    c.ex    = ex;
#ifdef STREAM_MOE_TEMP
    c.fold_repl = std::getenv("STREAM_MOE_TMP_CHAIN_FOLD") != nullptr;
#endif

    // Set full-width geometry from the first routed mm's ids (single bucket).
    for (const auto * cn : ex->compute) {
        if (!cn || cn->op != GGML_OP_MUL_MAT_ID || !cn->src[2]) continue;
        c.w_b = cn->src[2]->ne[0];
        c.n_t = cn->src[2]->ne[1];
        break;
    }

    // Single full-width bucket today (SS7.8: device_used == 1 in this CPU
    // phase). The per-device accumulator acc_d is the expert-folded output; RAM
    // add_in lives in chain_ctx and is filled by chain_exit.
    ggml_tensor * last = append_bucket_chain(c);
    if (!last) { LOG_ERROR("stream_moe: chain append_bucket_chain failed L" << layer); return GGML_STATUS_FAILED; }
    ggml_tensor * acc = nullptr;
    if (c.fold_repl) {
        // weighted (=last before fold when fold_repl) -> expert fold -> the
        // per-device acc_d columns [d_out, n_t] (expert width 1)
        acc = append_expert_fold(c, last);
        if (!acc) { LOG_ERROR("stream_moe: chain append_expert_fold failed L" << layer); return GGML_STATUS_FAILED; }
        if (!append_scatter_to_fullwidth(c, acc)) { return GGML_STATUS_FAILED; }
        c.d_out       = acc->ne[0];
        c.device_used = 1;
    }
    if (!chain_exit(c, acc)) { return GGML_STATUS_FAILED; }

#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DUMP")) {
        size_t seq = 0;
        for (const auto * cn : ex->compute) {
            tmp_dump_node(*c.sched, *c.topo, *c.pins, layer, seq, const_cast<ggml_tensor*>(cn));
            ++seq;
        }
    }
#endif
    reset_layer(layer);
    return GGML_STATUS_SUCCESS;
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
    // Batch semantics (2026-09): ONE request carrying the whole layer's expert
    // bitmap; missing experts load concurrently (IOCP n-way), exec wakes once.
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
    // All keys belong to `layer` (burst is per-layer); build the needed bitmap.
    if (!keys.empty() && keys[0].layer != static_cast<uint32_t>(layer)) {
        LOG_ERROR("stream_moe: burst keys layer mismatch (" << keys[0].layer << " vs " << layer << ")");
        return GGML_STATUS_FAILED;
    }
    uint64_t needed[BITMAP_WORDS] = { 0 };
    for (const auto & k : keys) expert_scheduler::bit_set(needed, k.expert);
    batch_await_t await;
    std::vector<expert_handle_t> pins(keys.size());
    const int32_t np = sched.pin_layer(static_cast<uint32_t>(layer), needed, await,
                                       pins.data(), static_cast<uint32_t>(pins.size()));
    if (np < 0 || static_cast<size_t>(np) != keys.size()) {
        LOG_ERROR("stream_moe: burst pin_layer failed (wanted " << keys.size() << ", got " << np << ")");
        return GGML_STATUS_FAILED;
    }
    // pin_layer fills handles in ascending-expert order; exec_one_burst's
    // pin_slot(pins, layer, expert) scans by (layer,expert) so any order works.
    (void)0;

    bool ok = true;
    // Hidden outputs use the per-layer result block. When verify produced a
    // node-level interval layout (layout_ok) the block is sized from it
    // (result_bytes, ~2-3 slots/layer); otherwise fall back to the full-alloc
    // sum (every node its own range). mm nodes run through exec_mixed_mm
    // (per-pool rounds, CPU or vulkan) writing back to the hidden host dst,
    // then the weightless chain runs on the host.
    {
        const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
        const size_t need = (ex && ex->layout_ok) ? ex->result_bytes : lsum;
        moe_chain_set_full_alloc(need);
    }
    reset_layer(layer);
#ifdef STREAM_MOE_TEMP
    // Whole-layer single-graph path (GPU mini-graph shape on CPU). The shared
    // interval layout is NOT used: every node gets its own arena region (per-
    // node bump via hide_output), because in ONE cgraph all outputs coexist to
    // the end. mm dsts and weightless dsts land in distinct regions; views are
    // refreshed to those hidden producers before the graph runs.
    if (std::getenv("STREAM_MOE_TMP_CHAIN_GRAPH")) {
        // EXPERIMENT (2026-09-06): does the verify INTERVAL layout (hide_burst
        // -> out_off, shared byte regions by last-use) hold for the one-cgraph
        // path too? The CPU backend executes gf->nodes in order (no cross-node
        // concurrency), so the per-node last-use reasoning should transfer: a
        // later node reusing an earlier result's region only runs after every
        // reader of that earlier result (last-use < the reuse index). If this
        // stays IDENTICAL the verify out_off layout is directly reusable as the
        // per-device arena offset - no separate per-device layout needed. The
        // prior hide_output full-alloc bump was a conservative choice, never
        // proven necessary against the interval layout in this path.
        const moe_layer_exec_t * exl = moe_chain_layer_exec(layer);
        moe_chain_set_full_alloc((exl && exl->layout_ok) ? exl->result_bytes : lsum);
        reset_layer(layer);
        for (const auto * cn : ex->compute) {
            ggml_tensor * m = const_cast<ggml_tensor*>(cn);
            if (is_view_op(m)) continue;
            if (m->name && strstr(m->name, "ffn_moe_out")) continue;  // llama dst
            if (!hide_burst(m, layer)) {
                LOG_ERROR("stream_moe: chain hide failed L" << layer);
                for (const auto & h : pins) sched.unpin(h);
                return GGML_STATUS_FAILED;
            }
        }
        for (const auto & va : ex->view_aliases) refresh_aliases(layer, va.prod);
        const enum ggml_status st = exec_layer_burst_chain(layer, ctx, cpu, sched, ex, pins);
        for (const auto & h : pins) sched.unpin(h);
        return st;
    }
#endif
#ifdef STREAM_MOE_TEMP
    {
        // TEMP M-diagnostic: where does this layer's pinned active set live?
        uint32_t n_pool0 = 0, n_pool1 = 0;
        for (const auto & h : pins) {
            const auto * osp = sched.subpool_of_slot(h.slot);
            if (osp && osp->pool == 0) ++n_pool0; else if (osp) ++n_pool1;
        }
        LOG_INFO("stream_moe: [tmp] burst L" << layer << " pins=" << pins.size()
                 << " pool0=" << n_pool0 << " pool1=" << n_pool1);
    }
#endif
    size_t seq = 0;
    for (const auto * cn : ex->compute) {
        if (!ok) break;
        ok = exec_one_burst(ctx, cpu, sched, topo, const_cast<ggml_tensor*>(cn), layer, pins);
#ifdef STREAM_MOE_TEMP
        if (ok) {
            tmp_dump_node(sched, topo, pins, layer, seq, const_cast<ggml_tensor*>(cn));
        }
#endif
        ++seq;
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
