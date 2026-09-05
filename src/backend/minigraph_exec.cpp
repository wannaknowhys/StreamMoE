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
    if (plan.rounds.empty()) return true;

    uint8_t * main_dst = static_cast<uint8_t*>(nd->data);
    const size_t esize = sizeof(float);

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
        scatter_sub_dst(main_dst, n_k, r.scatter, r.width, r.n_active, out.data(), d_out, esize);
    }
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
    // Hidden outputs use the per-layer full-alloc buffer (each compute node its
    // own range); mm nodes run through exec_mixed_mm (per-pool rounds, CPU or
    // vulkan) writing back to the hidden host dst, then the weightless chain
    // runs on the host. The retired whole-layer device-resident executor
    // (b4-2/C2b4) was removed - it repurposed main-graph tensor buffers.
    moe_chain_set_full_alloc(lsum);
    reset_layer(layer);
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
