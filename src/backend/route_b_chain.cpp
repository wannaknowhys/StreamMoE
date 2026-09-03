#include "backend/route_b_chain.h"

#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace stream_moe {

bool moe_chain_assign_backend(ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t our_backend) {
    if (!gf || !sched || !our_backend) return false;
    int n = 0, n_mm = 0, n_skipped_layout = 0, n_not_chain = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * nd = gf->nodes[i];
        if (!moe_chain_node_is_privatizable(nd)) { n_not_chain++; continue; }
        const enum ggml_op op = nd->op;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_TRANSPOSE ||
            op == GGML_OP_PERMUTE || op == GGML_OP_CONT) { n_skipped_layout++; continue; }
        ggml_backend_sched_set_tensor_backend(sched, nd, our_backend);
        if (op == GGML_OP_MUL_MAT_ID) n_mm++;
        n++;
        if (n <= 6) {
            fprintf(stderr, "[route_b_verify] assign op=%s name=%s\n", ggml_op_name(op), nd->name ? nd->name : "?");
        }
    }
    fprintf(stderr, "[route_b_verify] assign: total=%d (mul_mat_id=%d other=%d) skipped_layout=%d not_chain=%d\n",
            n, n_mm, n - n_mm, n_skipped_layout, n_not_chain);
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

    // Count hidden (privatized) intermediates once for the log. Unnamed
    // view/layout glue is chain-internal by construction (it aliases a hidden
    // result) and is exempt from the external-consumer scan below.
    int n_hidden = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        const char * nm = nd->name ? nd->name : "";
        if (is_hidden_name(nm)) n_hidden++;
    }

    // External-consumer scan: for every hidden intermediate, every OTHER node
    // that references it as a src and is not chain-internal is a violation.
    int violations = 0;
    for (int i = 0; i < gf->n_nodes && violations < 8; ++i) {
        const ggml_tensor * p = gf->nodes[i];
        const char * pn = p->name ? p->name : "";
        if (!is_hidden_name(pn)) continue; // not hidden

        for (int j = 0; j < gf->n_nodes; ++j) {
            if (j == i) continue;
            const ggml_tensor * c = gf->nodes[j];
            const char * cn = c->name ? c->name : "";
            if (is_moe_name(cn) && !is_gating_name(cn)) continue;   // chain-internal consumer: ok
            if (is_output_name(cn)) continue;                        // moe_out end: ok (it consumes hidden)
            if (!cn[0]) continue;                                    // unnamed view glue: chain-internal
            if (strncmp(cn, "node_", 5) == 0) continue;              // auto-named unnamed node (ggml default): chain-internal
            if (c->op == GGML_OP_NONE) continue;                     // leaves (weights/inputs) do not consume

            bool uses = false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (c->src[s] == p) { uses = true; break; }
            }
            if (uses) {
                fprintf(stderr,
                    "[route_b_verify] VIOLATION: hidden MoE intermediate '%s' has external consumer '%s'\n",
                    pn, cn);
                violations++;
                break; // this hidden node already flagged; keep scanning others
            }
        }
    }

    // Architecture verdict is cached by model-topology signature; for now log it
    // once per process (a build can call this several times - reserve/encode).
    static bool reported = false;
    if (!reported) {
        fprintf(stderr, "[route_b_verify] architecture scan: hidden_chain_nodes=%d external_consumers=%d\n",
                n_hidden, violations);
        reported = true;
    }

    if (violations > 0) {
        fprintf(stderr,
            "[route_b_verify] FAIL: a model structure consumes a privatised MoE intermediate.\n"
            "           This breaks the privatisation premise (docs/ROUTE_B_GPU_PHASE.md §3.1).\n"
            "           Refusing to run. No fallback.\n");
        exit(1);
    }
    return true;
}

} // namespace stream_moe
