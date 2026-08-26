#include "backend/minigraph_exec.h"
#include "common/logger.h"
#include "ggml-impl.h"

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
    return {-1, 0, layer, expert, false};
}

} // namespace

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
enum ggml_status moe_exec_mul_mat_id(
    ggml_context* ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    const ggml_tensor* const* nodes,
    int n_nodes,
    int n_threads)
{
    if (n_nodes == 0) return GGML_STATUS_SUCCESS;

    const moe_model_topology_t& topo = sched.topology();
    const int32_t n_slots = static_cast<int32_t>(sched.num_slots());

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
        if (nd->op != GGML_OP_MUL_MAT_ID) {
            LOG_ERROR("stream_moe: unsupported op " << ggml_op_name(nd->op)
                     << " node=" << (nd->name[0] ? nd->name : "?")
                     << " w=" << (nd->src[0] && nd->src[0]->name[0] ? nd->src[0]->name : "?"));
            return GGML_STATUS_FAILED;
        }
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
        const ggml_tensor* w   = nd->src[0];
        const ggml_tensor* cur = nd->src[1];
        const ggml_tensor* ids = nd->src[2];
        const ggml_tensor* dst = nd;
        parsed_node_t pn = parse_weight_name(w->name);

        // branch layout (compact slot offset, uniform across experts)
        size_t off = 0, bytes = 0;
        if (!sched.branch_layout(pn.layer, 0, w->name, off, bytes)) {
            LOG_ERROR("stream_moe: no slot layout for " << w->name);
            ok = false; break;
        }

        // w3d leaf: [ne00, ne01, num_slots], data = pool + branch offset, nb[2] = slot stride
        ggml_tensor* w3d = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], 1);
        w3d->ne[2] = n_slots;
        w3d->nb[2] = sched.slot_size();
        w3d->nb[3] = w3d->nb[2] * n_slots;
        w3d->data  = sched.pool_base() + off;

        // ids_slot: translate expert ids -> slot indices (private, main graph untouched)
        ggml_tensor* ids_slot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]);
        auto* ids_slot_data = static_cast<int32_t*>(ids_slot->data);
        for (int t = 0; t < ids->ne[1] && ok; ++t)
            for (int k = 0; k < ids->ne[0] && ok; ++k) {
                int32_t e = MOE_ID_AT(ids, t, k);
                int32_t slot = -1;
                if (pn.down) {
                    slot = sched.slot_of(pn.layer, static_cast<uint32_t>(e));
                } else {
                    expert_handle_t h = pin_state_find(pin_state(), pn.layer, static_cast<uint32_t>(e));
                    if (h.pinned) slot = h.slot;
                }
                if (slot < 0 || slot >= n_slots) { ok = false; break; }
                ids_slot_data[static_cast<size_t>(t) * ids->ne[0] + k] = slot;
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

} // namespace stream_moe
