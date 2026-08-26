#include "backend/minigraph_exec.h"
#include "common/logger.h"

#include <cstring>
#include <map>
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

} // namespace

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

    // ---- phase 1: parse nodes, group experts per role ----
    // key = layer * n_expert + expert
    std::vector<uint32_t> pin_keys;    // non-down nodes
    std::vector<uint32_t> down_keys;   // down node
    bool has_non_down = false, has_down = false;
    int32_t ids_ne0 = -1, ids_ne1 = -1;

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
        ids_ne0 = static_cast<int32_t>(ids->ne[0]);
        ids_ne1 = static_cast<int32_t>(ids->ne[1]);
        const int32_t* ids_data = static_cast<const int32_t*>(ids->data);
        if (!ids_data) return GGML_STATUS_FAILED;

        auto add_key = [&](std::vector<uint32_t>& v, uint32_t e) {
            uint32_t key = pn.layer * topo.n_expert + e;
            for (uint32_t k : v) if (k == key) return;
            v.push_back(key);
        };
        for (int t = 0; t < ids_ne1; ++t)
            for (int k = 0; k < ids_ne0; ++k) {
                int32_t e = ids_data[static_cast<size_t>(t) * ids_ne0 + k];
                if (e < 0 || e >= static_cast<int32_t>(topo.n_expert)) return GGML_STATUS_FAILED;
                if (pn.down) add_key(down_keys, static_cast<uint32_t>(e));
                else         add_key(pin_keys, static_cast<uint32_t>(e));
            }
        if (pn.down) has_down = true; else has_non_down = true;
    }

    // ---- phase 2: pin (non-down role) / wait (down role) ----
    std::vector<expert_handle_t> pins;
    pins.reserve(pin_keys.size());
    for (uint32_t key : pin_keys) {
        uint32_t l = key / topo.n_expert, e = key % topo.n_expert;
        pins.push_back(sched.pin_expert(l, e));
    }
    for (uint32_t key : down_keys) {
        uint32_t l = key / topo.n_expert, e = key % topo.n_expert;
        sched.wait_ready(l, e);
    }

    // ---- phase 3: build + run mini-graph per node ----
    bool ok = true;
    for (int i = 0; i < n_nodes && ok; ++i) {
        const ggml_tensor* nd = nodes[i];
        const ggml_tensor* w = nd->src[0];
        const ggml_tensor* cur = nd->src[1];
        const ggml_tensor* ids = nd->src[2];
        const ggml_tensor* dst = nd;
        ggml_tensor* w_nc = const_cast<ggml_tensor*>(w);

        parsed_node_t pn = parse_weight_name(w->name);
        const int32_t* ids_data = static_cast<const int32_t*>(ids->data);
        const int n_ids = ids_ne0, n_tok = ids_ne1;

        // group (k, t) by expert, preserving order
        std::map<uint32_t, std::vector<std::pair<int32_t, int32_t>>> groups;
        for (int t = 0; t < n_tok; ++t)
            for (int k = 0; k < n_ids; ++k) {
                uint32_t e = static_cast<uint32_t>(ids_data[static_cast<size_t>(t) * n_ids + k]);
                groups[e].push_back({k, t});
            }

        // flat dst column = k*n_tok + t
        ggml_tensor* dst_flat = ggml_view_2d(ctx, const_cast<ggml_tensor*>(dst),
                                             dst->ne[0], dst->ne[1] * dst->ne[2],
                                             dst->nb[1], 0);
        // cur may be 3D [ne00, 1, n_tok]; flatten then gather the token columns
        // ggml_get_rows(a, idx) -> [a->ne[0], idx->ne[0]] (column gather)
        ggml_tensor* cur2 = ggml_reshape_2d(ctx, const_cast<ggml_tensor*>(cur), cur->ne[0], cur->ne[1] * cur->ne[2]);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        for (const auto& g : groups) {
            uint32_t e = g.first;
            const auto& pairs = g.second;
            size_t off = 0, bytes = 0;
            if (!sched.branch_layout(pn.layer, e, w->name, off, bytes)) {
                LOG_ERROR("stream_moe: no layout for " << w->name << " L" << pn.layer << " E" << e);
                ok = false; break;
            }
            int32_t slot = -1;
            if (pn.down) {
                slot = sched.slot_of(pn.layer, e);
            } else {
                // find matching handle
                for (const auto& h : pins)
                    if (h.layer == pn.layer && h.expert == e) { slot = h.slot; break; }
            }
            uint8_t* wmem = sched.slot_mem(slot);
            if (!wmem) { ok = false; break; }

            // weight view over the slot slice
            ggml_tensor* w2d = ggml_new_tensor_2d(ctx, w->type, w->ne[0], w->ne[1]);
            w2d->data = wmem + off;
            {
                const uint8_t* probe = static_cast<const uint8_t*>(w2d->data);
                LOG_INFO("[moe] wdata@" << (void*)probe << " first8="
                         << std::hex << (int)probe[0] << " " << (int)probe[1] << " " << (int)probe[2] << " " << (int)probe[3]
                         << " " << (int)probe[4] << " " << (int)probe[5] << " " << (int)probe[6] << " " << (int)probe[7] << std::dec);
            }

            // input: cur columns for this expert's (k,t) pairs
            size_t count = pairs.size();
            ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(count));
            auto* idx_t_data = static_cast<int32_t*>(idx_t->data);
            ggml_tensor* idx_c = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(count));
            auto* idx_c_data = static_cast<int32_t*>(idx_c->data);
            for (size_t j = 0; j < count; ++j) {
                idx_t_data[j] = pairs[j].second;                       // token index
                idx_c_data[j] = pairs[j].first * n_tok + pairs[j].second; // dst column
            }
            ggml_tensor* gathered = ggml_get_rows(ctx, cur2, idx_t);   // [ne00, count]
            LOG_INFO("[moe] L" << pn.layer << " " << w->name << " w=[" << w->ne[0] << "," << w->ne[1]
                     << "] gathered=[" << gathered->ne[0] << "," << gathered->ne[1] << "] count=" << count);
            ggml_tensor* out = ggml_mul_mat(ctx, w2d, gathered);       // [ne01, count]
            ggml_tensor* scat = ggml_set_rows(ctx, dst_flat, out, idx_c);

            ggml_build_forward_expand(gf, scat);
        }

        if (ok) {
            enum ggml_status st = ggml_backend_graph_compute(cpu_backend, gf);
            if (st != GGML_STATUS_SUCCESS) { ok = false; }
        }
    }

    // ---- phase 4: release (down role only) ----
    if (has_down) {
        for (uint32_t key : down_keys) {
            uint32_t l = key / topo.n_expert, e = key % topo.n_expert;
            int32_t slot = sched.slot_of(l, e);
            if (slot >= 0) sched.unpin_slot(slot);
        }
    }

    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

} // namespace stream_moe
