#include "backend/route_b_chain.h"

#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace stream_moe {

bool moe_chain_assign_backend(ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t our_backend) {
    if (!gf || !sched || !our_backend) return false;
    int n = 0;
    size_t tot_bytes = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * nd = gf->nodes[i];
        if (!moe_chain_node_is_privatizable(nd)) continue;
        const enum ggml_op op = nd->op;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_TRANSPOSE ||
            op == GGML_OP_PERMUTE || op == GGML_OP_CONT) continue;
        ggml_backend_sched_set_tensor_backend(sched, nd, our_backend);
        tot_bytes += ggml_nbytes(nd);
        n++;
    }
    fprintf(stderr, "[route_b_verify] chain intermediates: %d nodes, total %zu MB (%.1f MB/layer)\n",
            n, tot_bytes / (1024 * 1024), (double) tot_bytes / (1024.0 * 1024.0 * 30.0));
    fprintf(stderr, "[route_b_verify] chain backend assigned: %d compute nodes -> stream_moe\n", n);
    return true;
}

namespace {

bool has(const char * hay, const char * needle) {
    return hay && needle && strstr(hay, needle) != nullptr;
}

// Gating-segment nodes stay on the dense side (never hidden, not chain).
bool is_gating_name(const char * n) {
    return has(n, "ffn_moe_logits")
        || has(n, "ffn_moe_probs")
        || has(n, "ffn_moe_argsort")
        || has(n, "ffn_moe_topk")
        || has(n, "ffn_moe_group")
        || has(n, "ffn_moe_weights");   // get_rows/softmax/sum/clamp/norm/scaled
}

// The chain output end: written to the main dst (residual reads it), NOT hidden.
bool is_output_name(const char * n) {
    return has(n, "ffn_moe_out");
}

// Layer-output node where the dense MLP branch and the MoE branch are summed
// (gemma4: "ffn_moe_combined"); executed on the dense side, consumed by the next
// layer's norm - NOT a hidden chain intermediate.
bool is_combined_name(const char * n) {
    return has(n, "ffn_moe_combined");
}

// Any named MoE-domain node (gate_up / gate / up / swiglu / geglu / down /
// weighted / ... ) - the candidate intermediate set before gating/output filters.
bool is_moe_name(const char * n) {
    return has(n, "ffn_moe_");
}

// A privatisable (hidden) MoE-chain intermediate.
bool is_hidden_name(const char * n) {
    return is_moe_name(n) && !is_gating_name(n) && !is_output_name(n) && !is_combined_name(n);
}

// Routed expert MUL_MAT_ID anchor: weight carries "_exps", not shared "_shexp".
bool is_routed_mm(const ggml_tensor * n) {
    if (!n || n->op != GGML_OP_MUL_MAT_ID || !n->src[0] || !n->src[0]->name) return false;
    return has(n->src[0]->name, "_exps") && !has(n->src[0]->name, "_shexp");
}

// Layer index from the anchor's weight name "blk.<N>....".
int mm_layer(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0];
    if (!w || !w->name || strncmp(w->name, "blk.", 4) != 0) return -1;
    const char * d = strchr(w->name + 4, '.');
    if (!d) return -1;
    const size_t k = (size_t)(d - (w->name + 4));
    char buf[16];
    if (k >= sizeof(buf)) return -1;
    memcpy(buf, w->name + 4, k);
    buf[k] = 0;
    return atoi(buf);
}

bool is_view_op(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_CONT;
}

} // namespace

// ---- single chain-node predicate -----------------------------------------

bool moe_chain_node_is_privatizable(const ggml_tensor * node) {
    if (!node) return false;
    // routed expert MUL_MAT_ID: weight name carries "_exps" but not shared "_shexp"
    if (node->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * w = node->src[0];
        if (w && w->name && has(w->name, "_exps") && !has(w->name, "_shexp")) return true;
    }
    const char * nm = node->name ? node->name : "";
    if (!nm[0]) return false;
    // hidden chain intermediate or the output end (both named in the ffn_moe_ domain)
    return is_hidden_name(nm) || is_output_name(nm);
}

bool moe_chain_verify_graph(ggml_cgraph * gf) {
    if (!gf) return true;
    const int N = gf->n_nodes;

    // ---- capture: topological closure. Anchors = routed expert MUL_MAT_ID
    // (layer from weight name). Forward BFS along consumers, stopping expansion
    // at ffn_moe_out (the chain end - included, its own consumers excluded).
    // Layer id propagates along edges.
    std::vector<int> layer(N, -1);
    std::vector<char> inq(N, 0);
    std::vector<char> chain(N, 0);
    std::vector<int> q;
    int n_anchors = 0;
    for (int i = 0; i < N; ++i) {
        if (is_routed_mm(gf->nodes[i])) {
            layer[i] = mm_layer(gf->nodes[i]);
            q.push_back(i); inq[i] = 1;
            n_anchors++;
        }
    }
    for (size_t h = 0; h < q.size(); ++h) {
        const int i = q[h];
        const ggml_tensor * ni = gf->nodes[i];
        const bool is_out = ni->name && strstr(ni->name, "ffn_moe_out") != nullptr;
        for (int j = 0; j < N; ++j) {
            if (chain[j] || inq[j]) continue;
            if (layer[j] != -1) continue;
            const ggml_tensor * nj = gf->nodes[j];
            bool uses = false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) if (nj->src[s] == ni) { uses = true; break; }
            if (!uses) continue;
            if (is_out) continue;              // consumer of moe_out: outside the chain
            layer[j] = layer[i];
            inq[j] = 1;
            q.push_back(j);
        }
        chain[i] = 1;
        inq[i] = 0;
    }

    // ---- debug print: per-layer tree (graph order = execution order), capture
    // span, and odd/even ping-pong buffer budgets (compute nodes only, views
    // skipped; even = first compute node, odd = second).
    std::map<int, std::vector<int>> by_layer;
    int chain_total = 0;
    for (int i = 0; i < N; ++i) {
        if (chain[i]) { by_layer[layer[i]].push_back(i); chain_total++; }
    }
    fprintf(stderr, "[route_b_cap] anchors=%d chain_nodes=%d/%d\n", n_anchors, chain_total, N);
    size_t g_even_max = 0, g_odd_max = 0;
    for (auto & kv : by_layer) {
        const int L = kv.first;
        const auto & v = kv.second;
        size_t even_max = 0, odd_max = 0;
        const ggml_tensor * even_node = nullptr, * odd_node = nullptr;
        int ncomp = 0;
#ifdef STREAM_MOE_CHAIN_DEBUG
        fprintf(stderr, "  L%d chain (%zu):\n", L, v.size());
#endif
        for (size_t k = 0; k < v.size(); ++k) {
            const ggml_tensor * nd = gf->nodes[v[k]];
#ifdef STREAM_MOE_CHAIN_DEBUG
            fprintf(stderr, "    [%2zu] %-14s %-40s nb=%zu\n", k, ggml_op_name(nd->op),
                    nd->name ? nd->name : "?", ggml_nbytes(nd));
#endif
            if (is_view_op(nd)) continue;
            const size_t nb = ggml_nbytes(nd);
            if ((ncomp & 1) == 0) { if (nb > even_max) { even_max = nb; even_node = nd; } }
            else                  { if (nb > odd_max)  { odd_max  = nb; odd_node  = nd; } }
            ncomp++;
        }
#ifdef STREAM_MOE_CHAIN_DEBUG
        fprintf(stderr, "  L%d compute=%d -> even_buf=%zuMB (nb %s)  odd_buf=%zuMB (nb %s)\n",
                L, ncomp,
                even_max / (1024 * 1024), even_node && even_node->name ? even_node->name : "?",
                odd_max / (1024 * 1024), odd_node && odd_node->name ? odd_node->name : "?");
#endif
        if (even_max > g_even_max) g_even_max = even_max;
        if (odd_max > g_odd_max) g_odd_max = odd_max;
    }
    fprintf(stderr, "[route_b_cap] ping-pong budget (per layer): even_buf=%zuMB  odd_buf=%zuMB  total=%zuMB\n",
            g_even_max / (1024 * 1024), g_odd_max / (1024 * 1024),
            (g_even_max + g_odd_max) / (1024 * 1024));

    // ---- external-consumer check: no node OUTSIDE the chain may reference a
    // chain node except the moe_out end (which post-norm/dense legitimately
    // consumes). Topology-only - no name heuristics, no unnamed exemptions.
    int violations = 0;
    for (int j = 0; j < N && violations < 8; ++j) {
        if (chain[j]) continue;
        const ggml_tensor * cj = gf->nodes[j];
        if (cj->op == GGML_OP_NONE) continue;   // leaves (weights/inputs) do not consume
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = cj->src[s];
            if (!src) continue;
            int si = -1;
            for (int k = 0; k < N; ++k) if (gf->nodes[k] == src) { si = k; break; }
            if (si < 0 || !chain[si]) continue;
            if (src->name && strstr(src->name, "ffn_moe_out")) continue;   // end: allowed
            fprintf(stderr,
                "[route_b_cap] VIOLATION: chain node '%s' (%s) referenced by external '%s' (%s)\n",
                src->name ? src->name : "?", ggml_op_name(src->op),
                cj->name ? cj->name : "?", ggml_op_name(cj->op));
            violations++;
            break;
        }
    }
    fprintf(stderr, "[route_b_cap] external_violations=%d\n", violations);

    if (violations > 0) {
        fprintf(stderr,
            "[route_b_cap] FAIL: a model structure consumes a privatised MoE intermediate.\n"
            "           This breaks the privatisation premise (docs/ROUTE_B_GPU_PHASE.md §3.1).\n"
            "           Refusing to run. No fallback.\n");
        exit(1);
    }
    return true;
}

} // namespace stream_moe
