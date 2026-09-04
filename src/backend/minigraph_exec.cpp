#include "backend/minigraph_exec.h"
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

// Vulkan mm execution (C2b4): the MUL_MAT_ID runs on the device backend inside
// VRAM. Weight shell points at the pool buffer slot; cur and the translated
// ids are uploaded into the staging buffer (bumped by stage_off); the result is
// computed at arena offset dst_off. When readback is true the result is copied
// to the hidden host dst; in the whole-layer device-resident mode (readback
// false) it stays in the arena for the host-free chain. `slotbuf` carries the
// global slot per (k,t).
static bool exec_mm_vk(ggml_context * ctx, expert_scheduler & sched,
                       device_exec_ctx_t & dv, ggml_tensor * nd,
                       const parsed_node_t & pn,
                       const std::vector<int32_t> & slotbuf,
                       const expert_scheduler::subpool_t & sp, size_t col_off,
                       size_t col_stride, size_t dst_off, size_t & stage_off, bool readback) {
    ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
    ggml_tensor * ids = const_cast<ggml_tensor*>(nd->src[2]);
    const ggml_tensor * w  = nd->src[0];
    const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
    const size_t mm_bytes = ggml_nbytes(nd);
    const size_t cur_bytes = ggml_nbytes(cur);
    const size_t ids_bytes = n_ids * sizeof(int32_t);

    if (!stream_moe_backend_device_ensure(sp.pool, dst_off + mm_bytes + 4u * 1024 * 1024,
                                           stage_off + ids_bytes + cur_bytes + 4u * 1024 * 1024)) {
        LOG_ERROR("stream_moe: device arena/stage ensure failed (pool " << sp.pool << ")");
        return false;
    }
    if (!dv.stage_map || !dv.arena_map) {
        LOG_ERROR("stream_moe: device arena/stage host maps unavailable (pool " << sp.pool << ")");
        return false;
    }

    // cur already on the device (previous private arena node) or a host tensor
    // that must be uploaded to the staging buffer.
    const bool cur_on_dev = cur->buffer && cur->buffer->buft == dv.arena_buft;
    if (!cur_on_dev) {
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (cur->nb[d] != cur->nb[d - 1] * cur->ne[d - 1]) {
                LOG_ERROR("stream_moe: vk mm activation is not contiguous (" << ggml_op_name(nd->op) << ")");
                return false;
            }
        }
        if (!stream_moe_backend_device_ensure(sp.pool, 1, stage_off + ids_bytes + cur_bytes + 4u * 1024 * 1024)) {
            LOG_ERROR("stream_moe: device stage ensure failed (pool " << sp.pool << ")");
            return false;
        }
    }
    const size_t ids_off = (stage_off + 63u) & ~size_t(63u);
    const size_t cur_off = (ids_off + ids_bytes + 63u) & ~size_t(63u);
    if (!cur_on_dev) {
        auto * slot32 = reinterpret_cast<int32_t*>(dv.stage_map + ids_off);
        for (size_t i = 0; i < n_ids; ++i) {
            slot32[i] = slotbuf[i] - static_cast<int32_t>(sp.slot_begin);
        }
        std::memcpy(dv.stage_map + cur_off, cur->data, cur_bytes);
    }
    stage_off = cur_off + cur_bytes;

    ggml_backend_buffer_t wbuf = reinterpret_cast<ggml_backend_buffer_t>(sp.dev_buf);
    const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);

    ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = g_n_slots;
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * g_n_slots;
    w3d->buffer = wbuf;
    w3d->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(wbuf, col_off));

    ggml_tensor* ids_leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
    ids_leaf->buffer = dv.stage;
    ids_leaf->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv.stage, ids_off));

    ggml_tensor* b_leaf;
    if (cur_on_dev) {
        b_leaf = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
        b_leaf->buffer = cur->buffer;
        b_leaf->data   = cur->data;
        std::memcpy(b_leaf->nb, cur->nb, sizeof(b_leaf->nb));
    } else {
        b_leaf = ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]);
        b_leaf->buffer = dv.stage;
        b_leaf->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv.stage, cur_off));
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* mm = ggml_mul_mat_id(ctx, w3d, b_leaf, ids_leaf);
    mm->buffer = dv.arena;
    mm->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv.arena, dst_off));
    ggml_build_forward_expand(gf, mm);

    if (ggml_backend_graph_compute(dv.be, gf) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: vulkan mm compute failed for " << (w->name ? w->name : "?"));
        return false;
    }
    if (readback) {
        std::memcpy(nd->data, dv.arena_map + dst_off, mm_bytes);
    }
    return true;
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

        // SoA column geometry for this branch's tensor (col_off region-relative,
        // stride = compact perExpert = vulkan's hardcoded MUL_MAT_ID stride).
        size_t col_off = 0, col_stride = 0;
        uint32_t col_index = 0;
        if (!sched.column_layout(sp, w->name, col_off, col_stride, col_index)) {
            LOG_ERROR("stream_moe: no column for " << w->name);
            return false;
        }

        // Device-resident active set: compute the mm on the device backend in
        // VRAM (weight shell + uploaded cur/ids), copy the result back to the
        // hidden host dst for the host-side weightless chain.
        if (sp.pool != 0) {
            device_exec_ctx_t* dv = stream_moe_backend_device_exec(sp.pool);
            if (!dv || !dv->be) {
                LOG_ERROR("stream_moe: no device exec context for pool " << sp.pool);
                return false;
            }
            size_t stage_off = 0;
            return exec_mm_vk(ctx, sched, *dv, nd, pn, slotbuf, sp, col_off, col_stride, 0, stage_off, true);
        }

        const int32_t g_n_slots = static_cast<int32_t>(sp.n_slots);
        ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
        w3d->ne[2] = g_n_slots;
        w3d->nb[2] = col_stride;
        w3d->nb[3] = col_stride * g_n_slots;
        w3d->data  = sp.base + col_off;

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
// ---- whole-layer device-resident execution (C2b4 v2) ----------------------
// A layer whose active experts all live on one device pool runs entirely on
// that device: hidden outputs are allocated in the device arena (buffer/data on
// the main tensors), dense-side inputs are uploaded to the staging buffer as
// device leaves, weightless nodes run as single-node graphs on the device
// backend, and only moe_out copies its result back to the host main dst.
static bool dev_arena_hide(ggml_tensor * nd, uint32_t pool, size_t & bump, size_t & off) {
    device_exec_ctx_t * dv = stream_moe_backend_device_exec(pool);
    if (!dv || !dv->be) return false;
    const size_t need = ggml_nbytes(nd);
    const size_t a = (bump + 63u) & ~size_t(63u);
    if (!stream_moe_backend_device_ensure(pool, a + need + 4096, 1)) return false;
    dv = stream_moe_backend_device_exec(pool);
    if (!dv->arena_map) return false;
    nd->buffer = dv->arena;
    nd->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv->arena, a));
    bump = a + need;
    off  = a;
    return true;
}

// Return a device-side tensor for `src`: itself when it already lives on the
// device, otherwise an uploaded copy leaf in the staging buffer.
static ggml_tensor * dev_stage_leaf(ggml_context * ctx, ggml_tensor * src,
                                    uint32_t pool, size_t & bump) {
    device_exec_ctx_t * dv = stream_moe_backend_device_exec(pool);
    if (!dv) return nullptr;
    if (src->buffer && src->buffer->buft == dv->arena_buft) return src;
    const size_t need = ggml_nbytes(src);
    const size_t a = (bump + 63u) & ~size_t(63u);
    if (!stream_moe_backend_device_ensure(pool, 1, a + need + 4096)) return nullptr;
    dv = stream_moe_backend_device_exec(pool);
    if (!dv->stage_map) return nullptr;
    std::memcpy(dv->stage_map + a, src->data, need);
    ggml_tensor * leaf = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], 1);
    for (int d = 0; d < 4; ++d) { leaf->ne[d] = src->ne[d]; leaf->nb[d] = src->nb[d]; }
    leaf->buffer = dv->stage;
    leaf->data   = static_cast<char*>(stmoe_vk_buffer_host_offset(dv->stage, a));
    bump = a + need;
    return leaf;
}

[[maybe_unused]] static bool exec_device_chain(int32_t layer, ggml_context * ctx, expert_scheduler & sched,
                              const moe_model_topology_t & topo, uint32_t pool,
                              const std::vector<expert_handle_t> & pins) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex || ex->compute.empty()) return true;
    device_exec_ctx_t * dv = stream_moe_backend_device_exec(pool);
    if (!dv || !dv->be) return false;
    size_t arena_bump = 0, stage_bump = 0;
    bool ok = true;
    // Reserve the WHOLE layer's arena/stage up front: growing the arena mid-
    // layer would invalidate already-hidden outputs (their buffer gets freed).
    {
        size_t need_total = 0;
        for (const auto * cn : ex->compute) need_total += ggml_nbytes(cn);
        if (!stream_moe_backend_device_ensure(pool, need_total + 64u * 1024 * 1024,
                                               need_total + 64u * 1024 * 1024)) {
            LOG_ERROR("stream_moe: device layer arena/stage reserve failed (pool " << pool << ")");
            return false;
        }
    }
    for (const auto * cn_ : ex->compute) {
        if (!ok) break;
        ggml_tensor * nd = const_cast<ggml_tensor*>(cn_);

        if (nd->op == GGML_OP_MUL_MAT_ID) {
            parsed_node_t pn = parse_weight_name(nd->src[0]->name);
            if (!pn.ok) { ok = false; break; }
            if (sched.group_of(pn.layer) == static_cast<uint32_t>(-1)) { ok = false; break; }
            const ggml_tensor * ids = nd->src[2];
            if (!ids->data) { ok = false; break; }
            const size_t n_ids = static_cast<size_t>(ids->ne[0]) * static_cast<size_t>(ids->ne[1]);
            std::vector<int32_t> slotbuf(n_ids, -1);
            const expert_scheduler::subpool_t * use_sp = nullptr;
            for (int t = 0; t < ids->ne[1] && ok; ++t) {
                for (int k = 0; k < ids->ne[0] && ok; ++k) {
                    const int32_t e = MOE_ID_AT(ids, t, k);
                    if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) { ok = false; break; }
                    const int32_t slot = pin_slot(pins, pn.layer, static_cast<uint32_t>(e));
                    if (slot < 0) { ok = false; break; }
                    const auto * osp = sched.subpool_of_slot(slot);
                    if (!osp) { ok = false; break; }
                    if (use_sp && use_sp != osp) {
                        LOG_ERROR("stream_moe: device-chain mixed-region active set for " << nd->src[0]->name);
                        ok = false; break;
                    }
                    if (!use_sp) use_sp = osp;
                    slotbuf[static_cast<size_t>(t) * ids->ne[0] + k] = slot;
                }
            }
            if (!ok) break;
            if (!use_sp || use_sp->pool != pool) { ok = false; break; }
            size_t col_off = 0, col_stride = 0;
            uint32_t col_index = 0;
            if (!sched.column_layout(*use_sp, nd->src[0]->name, col_off, col_stride, col_index)) { ok = false; break; }
            size_t dst_off = 0;
            if (!dev_arena_hide(nd, pool, arena_bump, dst_off)) { ok = false; break; }
            if (!exec_mm_vk(ctx, sched, *dv, nd, pn, slotbuf, *use_sp, col_off, col_stride, dst_off, stage_bump, false)) {
                ok = false; break;
            }
            refresh_aliases(layer, nd);
            continue;
        }

        if (is_view_op(nd)) {   // capture excludes views; defensive only
            ggml_tensor * mut = nd;
            mut->data = nd->op == GGML_OP_VIEW
                        ? static_cast<char*>(nd->src[0]->data) + nd->view_offs
                        : nd->src[0]->data;
            continue;
        }

        // weightless compute (or the moe_out end)
        const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
        for (int s = 0; s < GGML_MAX_SRC && ok; ++s) {
            ggml_tensor * src = nd->src[s];
            if (!src) continue;
            ggml_tensor * leaf = dev_stage_leaf(ctx, src, pool, stage_bump);
            if (!leaf) { ok = false; break; }
            if (leaf != src) nd->src[s] = leaf;   // external (host) src -> device leaf
        }
        if (!ok) break;

        if (is_out) {
            void * host_dst = nd->data;
            size_t off = 0;
            if (!dev_arena_hide(nd, pool, arena_bump, off)) { ok = false; break; }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            gf->nodes[0] = nd; gf->n_nodes = 1;
            if (ggml_backend_graph_compute(dv->be, gf) != GGML_STATUS_SUCCESS) {
                LOG_ERROR("stream_moe: device moe_out compute failed");
                ok = false; break;
            }
            std::memcpy(host_dst, dv->arena_map + off, ggml_nbytes(nd));
            nd->data = host_dst;   // restore the host dst for the dense side
        } else {
            size_t off = 0;
            if (!dev_arena_hide(nd, pool, arena_bump, off)) { ok = false; break; }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            gf->nodes[0] = nd; gf->n_nodes = 1;
            if (ggml_backend_graph_compute(dv->be, gf) != GGML_STATUS_SUCCESS) {
                LOG_ERROR("stream_moe: device chain compute failed for "
                          << (nd->name ? nd->name : ggml_op_name(nd->op)));
                ok = false; break;
            }
            refresh_aliases(layer, nd);
        }
    }
    return ok;
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
    // Device-resident whole-layer execution (b4-2) is disabled for now: it
    // repurposes main-graph tensor buffers and fights llama's scheduler buffer
    // bookkeeping after the burst. The CPU path below keeps v1 semantics for
    // device pools (per-mm vulkan compute + host readback, host weightless
    // chain). Re-enable once the arena-clone executor (no main-graph mutation)
    // replaces exec_device_chain.
    moe_chain_set_full_alloc(lsum);
    reset_layer(layer);
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
