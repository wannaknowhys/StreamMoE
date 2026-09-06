#include "backend/route_b_chain.h"

#include "backend/alloc.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace stream_moe {

namespace {
bool g_pingpong_ok = true;
void * g_fullalloc_buf = nullptr;
size_t g_fullalloc_cap = 0;

// Whole-layer burst capture: per-layer privatised compute sequence from the
// last graph build (see moe_chain_assign_backend).
std::map<int, moe_layer_exec_t> g_layer_exec;
}

bool moe_chain_pingpong_ok() { return g_pingpong_ok; }

void moe_chain_set_full_alloc(size_t layer_sum_bytes) {
    g_pingpong_ok = false;
    if (layer_sum_bytes > g_fullalloc_cap) {
        aligned_free_ptr(g_fullalloc_buf);
        g_fullalloc_buf = aligned_alloc_ptr(layer_sum_bytes, 64);
        g_fullalloc_cap = g_fullalloc_buf ? layer_sum_bytes : 0;
    }
}

void * moe_chain_fullalloc_buffer(size_t need_bytes) {
    if (!g_fullalloc_buf) return nullptr;   // set_full_alloc(layer_sum) must run first
    if (need_bytes > g_fullalloc_cap) {
        fprintf(stderr, "[route_b_cap] ERROR: full-alloc overflow need=%zu cap=%zu\n", need_bytes, g_fullalloc_cap);
        return nullptr;
    }
    return g_fullalloc_buf;
}

void * moe_chain_pingpong_buffer(int parity, size_t need_bytes) {
    static void * buf[2] = {nullptr, nullptr};
    static size_t sz[2] = {0, 0};
    const int p = parity & 1;
    if (need_bytes > sz[p]) {
        aligned_free_ptr(buf[p]);
        buf[p] = aligned_alloc_ptr(need_bytes, 64);
        sz[p] = buf[p] ? need_bytes : 0;
    }
    return buf[p];
}

namespace {
// Forward declarations (definitions live further down the file).
bool is_routed_mm(const ggml_tensor * n);
int  mm_layer(const ggml_tensor * n);
bool is_view_op(const ggml_tensor * n);
void collect_chain(const ggml_cgraph * gf, std::vector<char>& chain,
                   std::vector<int>& layer, int& n_anchors);
} // namespace

bool moe_chain_assign_backend(ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_t our_backend) {
    if (!gf || !sched || !our_backend) return false;
    // Privatise the WHOLE chain closure (BFS from routed mm anchors, shared
    // with verify): routed mm + named hidden intermediates + output end + the
    // anonymous per-topk convergence adds. View/layout nodes are left to the
    // scheduler (pass4 follows view_src). The same traversal fills the
    // whole-layer burst capture (exec order per layer).
    std::vector<char> chain;
    std::vector<int> layer;
    int n_anchors = 0;
    collect_chain(gf, chain, layer, n_anchors);
    g_layer_exec.clear();
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (!chain[i]) continue;
        ggml_tensor * nd = gf->nodes[i];
        if (is_view_op(nd)) continue;
        const int L = layer[i];
        if (L >= 0) {
            moe_layer_exec_t & ex = g_layer_exec[L];
            if (ex.compute.empty()) ex.layer = L;
            ex.compute.push_back(nd);
        }
    }

    // ---- input-producer closure -------------------------------------------
    // Privatise weightless compute that feeds the chain from llama's side
    // (REPEAT / GET_ROWS of the expert scales, ...) so the whole-layer burst
    // runs without waiting on llama to interleave those steps. Gating-segment
    // outputs (logits/probs/argsort/topk/weights) stay dense - they are ready
    // before the layer's first privatised split anyway.
    auto is_gating_nm = [](const char * nm) -> bool {
        if (!nm) return false;
        return strstr(nm, "ffn_moe_logits") != nullptr || strstr(nm, "ffn_moe_probs") != nullptr ||
               strstr(nm, "ffn_moe_argsort") != nullptr || strstr(nm, "ffn_moe_topk") != nullptr ||
               strstr(nm, "ffn_moe_group") != nullptr || strstr(nm, "ffn_moe_weights") != nullptr;
    };
    std::unordered_map<const ggml_tensor*, int> pos;
    for (int i = 0; i < gf->n_nodes; ++i) pos[gf->nodes[i]] = i;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto & kv : g_layer_exec) {
            moe_layer_exec_t & ex = kv.second;
            auto has = [&](const ggml_tensor * nd) -> bool {
                for (const auto * x : ex.compute) if (x == nd) return true;
                return false;
            };
            std::vector<ggml_tensor*> add;
            for (const auto * cn : ex.compute) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * t = cn->src[s];
                    if (!t) continue;
                    while (t && is_view_op(t)) t = t->src[0];
                    if (!t || t->op == GGML_OP_NONE) continue;      // weight/input leaf
                    if (t->op == GGML_OP_MUL_MAT_ID) continue;      // already in the closure
                    // Only chain-side helpers join: anonymous tensors (llama
                    // node_N / our STREAMMOE# buft names) or ffn_moe_* names
                    // outside the gating segment. Dense-named producers (norm,
                    // attn, embd, ...) stay on llama's dense side - they are
                    // ready before the layer's first privatised split.
                    const char * nm = t->name;
                    const bool anonymous = !nm || !nm[0] ||
                        strncmp(nm, "node_", 5) == 0 || strncmp(nm, "STREAMMOE#", 10) == 0;
                    const bool moe_named = nm && strncmp(nm, "ffn_moe_", 8) == 0 && !is_gating_nm(nm);
                    if (!anonymous && !moe_named) continue;
                    if (has(t)) continue;
                    add.push_back(const_cast<ggml_tensor*>(t));
                }
            }
            if (add.empty()) continue;
            changed = true;
            for (auto * nd : add) ex.compute.push_back(nd);
            // keep topological order (producer precedes consumer in the graph)
            std::stable_sort(ex.compute.begin(), ex.compute.end(),
                             [&](const ggml_tensor * a, const ggml_tensor * b) {
                auto ia = pos.find(a), ib = pos.find(b);
                return ia != pos.end() && ib != pos.end() ? ia->second < ib->second
                     : ia != pos.end();
            });
        }
    }

    // Privatise the whole per-layer compute closure (named + anonymous
    // convergence adds + input producers). Their later llama splits are served
    // as no-ops by the burst executor.
    int n = 0;
    size_t tot_bytes = 0;
    for (auto & kv : g_layer_exec) {
        for (auto * nd : kv.second.compute) {
            ggml_backend_sched_set_tensor_backend(sched, nd, our_backend);
            tot_bytes += ggml_nbytes(nd);
            n++;
        }
    }
#ifdef STREAM_MOE_CHAIN_DEBUG
    fprintf(stderr, "[route_b_verify] chain closure: anchors=%d, %d compute nodes assigned (%zu MB)\n",
            n_anchors, n, tot_bytes / (1024 * 1024));
    for (const auto & kv : g_layer_exec) {
        fprintf(stderr, "  L%d: %zu compute nodes, %zu view aliases\n",
                kv.first, kv.second.compute.size(), kv.second.view_aliases.size());
    }
#endif

    // Chain views over hidden producers (the burst executor refreshes their
    // data pointer whenever the producer output is hidden/recomputed). Resolve
    // each view through nested layout ops to its producer + cumulative offset.
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (!chain[i]) continue;
        ggml_tensor * nd = gf->nodes[i];
        if (!is_view_op(nd)) continue;
        const int L = layer[i];
        if (L < 0) continue;
        ggml_tensor * t = nd;
        int64_t off = 0;
        while (t && is_view_op(t)) {
            if (t->op == GGML_OP_VIEW) off += t->view_offs;
            t = t->src[0];
        }
        if (!t) continue;
        moe_layer_exec_t & ex = g_layer_exec[L];
        bool prod_in_exec = false;
        for (const auto * cn : ex.compute) {
            if (cn == t) { prod_in_exec = true; break; }
        }
        if (prod_in_exec) ex.view_aliases.push_back({ nd, t, off });
    }

    // Input-side layout tensors that feed the privatised compute (e.g. the
    // reshaped expert-scale views of the down path). Their producer is a
    // weight/leaf or llama-side tensor, so the burst executor must fix their
    // data pointer (view -> src data + offset) before running the layer.
    for (auto & kv : g_layer_exec) {
        moe_layer_exec_t & ex = kv.second;
        auto push_layout = [&](ggml_tensor * lt) {
            for (const auto * x : ex.input_layouts) if (x == lt) return;
            ex.input_layouts.push_back(lt);
        };
        for (const auto * cn : ex.compute) {
            if (!cn) continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * t = cn->src[s];
                while (t && is_view_op(t)) {
                    push_layout(t);
                    t = t->src[0];
                }
            }
        }
    }

    // ---- result-buffer layout (node-level interval reuse) ----------------
    // Runs after the closure is complete (g_layer_exec holds every layer's
    // compute sequence, views excluded). For each layer: resolve every
    // chain-internal edge (consumer reads producer; views alias to the
    // producer) and take each result's LAST reader index (last_use). A result
    // stays live until last_use executes. Greedy interval packing in exec
    // order: a slot whose occupant's last_use < current index is free (its
    // reader already ran), so the node reuses it. Needs ~2 slots (gemma) /
    // 3 (deepseek) per layer instead of one slot per node. Output:
    //   ex.out_off[i]   = byte offset of compute[i]'s output in the layer block
    //   ex.result_bytes = the layer block size (reused across layers by exec)
    //   ex.layout_ok    = false -> exec falls back to per-node bump (defensive)
    for (auto & kv : g_layer_exec) {
        moe_layer_exec_t & ex = kv.second;
        const auto & comp = ex.compute;
        const size_t C = comp.size();
        ex.out_off.assign(C, -1);
        ex.result_bytes = 0;
        ex.layout_ok = false;
        if (C == 0) { ex.layout_ok = true; continue; }
        // producer identity: tensor (possibly a view) -> compute index
        auto prod_of = [&](const ggml_tensor * t) -> int {
            const ggml_tensor * p = t;
            while (p && is_view_op(p)) p = p->src[0];
            if (!p) return -1;
            for (size_t i = 0; i < C; ++i) if (comp[i] == p) return (int) i;
            return -1;
        };
        // last reader index per producer (result live until that reader runs)
        std::vector<int> last_use(C, -1);
        for (int c = (int) C - 1; c >= 0; --c) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const int p = prod_of(comp[c] ? comp[c]->src[s] : nullptr);
                if (p >= 0 && p < c && last_use[p] < 0) last_use[p] = c;
            }
        }
        struct slot_t { size_t size = 0; int end = 0; bool used = false; };
        std::vector<slot_t> slots;
        std::vector<int> node_slot(C, -1);
        for (size_t i = 0; i < C; ++i) {
            const size_t nb = comp[i] ? ggml_nbytes(comp[i]) : 0;
            const int end = last_use[i] < 0 ? (int) C : last_use[i] + 1;
            int chosen = -1;
            for (size_t s = 0; s < slots.size(); ++s) {
                if (!slots[s].used || slots[s].end <= (int) i) { chosen = (int) s; break; }
            }
            if (chosen < 0) { chosen = (int) slots.size(); slots.push_back({}); }
            slot_t & st = slots[chosen];
            st.used = true;
            if (nb > st.size) st.size = nb;
            st.end = std::max(st.end, end);
            node_slot[i] = chosen;
        }
        // assign ascending slot bases (interleaved nodes sharing a slot reuse
        // the SAME base - the earlier occupant is dead by then, so overwriting
        // is safe). result_bytes = total across slots (one layer block).
        std::vector<size_t> slot_base(slots.size(), 0);
        {
            size_t acc = 0;
            for (size_t s = 0; s < slots.size(); ++s) { slot_base[s] = acc; acc += slots[s].size; }
            ex.result_bytes = acc;
        }
        for (size_t i = 0; i < C; ++i) {
            const int s = node_slot[i];
            ex.out_off[i] = (int64_t) slot_base[s];
        }
        ex.layout_ok = true;
    }
#ifdef STREAM_MOE_TEMP
    if (getenv("STREAM_MOE_CAP_DUMP")) {
        fprintf(stderr, "\n=== [cap-dump] result-buffer layout (interval, node-level) ===\n");
        size_t gmax = 0;
        for (const auto & kv : g_layer_exec) {
            const moe_layer_exec_t & ex = kv.second;
            fprintf(stderr, "  L%d: %zu compute, result_bytes=%zu layout_ok=%d\n",
                    kv.first, ex.compute.size(), ex.result_bytes, (int) ex.layout_ok);
            for (size_t i = 0; i < ex.compute.size() && i < ex.out_off.size(); ++i) {
                fprintf(stderr, "    compute#%2zu off=%8lld %-12s %s\n", i,
                        (long long) ex.out_off[i],
                        ggml_op_name(ex.compute[i] ? ex.compute[i]->op : GGML_OP_NONE),
                        ex.compute[i] && ex.compute[i]->name ? ex.compute[i]->name : "(anon)");
            }
            if (ex.result_bytes > gmax) gmax = ex.result_bytes;
        }
        fprintf(stderr, "[cap-dump] max layer result block = %zu bytes\n", gmax);
        fflush(stderr);
    }
#endif

    return true;
}

int32_t moe_chain_layer_of_node(const ggml_tensor * node) {
    if (!node) return -1;
    for (const auto & kv : g_layer_exec) {
        for (const auto * cn : kv.second.compute) {
            if (cn == node) return kv.first;
        }
    }
    return -1;
}

const moe_layer_exec_t * moe_chain_layer_exec(int32_t layer) {
    auto it = g_layer_exec.find(layer);
    return it == g_layer_exec.end() ? nullptr : &it->second;
}

int32_t moe_chain_layer_index(int32_t layer, const ggml_tensor * node) {
    const auto * ex = moe_chain_layer_exec(layer);
    if (!ex || !node) return -1;
    for (size_t k = 0; k < ex->compute.size(); ++k) {
        if (ex->compute[k] == node) return static_cast<int32_t>(k);
    }
    return -1;
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

// Privatised chain closure over one built graph: forward BFS from every routed
// expert MUL_MAT_ID anchor along consumers, stopping expansion at ffn_moe_out
// (the chain end - included, its own consumers excluded). Used by both verify
// (external-consumer / ping-pong checks) and assign_backend (which privatises
// the WHOLE closure - including the anonymous per-topk convergence adds that
// carry no ffn_moe_ name).
static void collect_chain(const ggml_cgraph * gf, std::vector<char>& chain,
                          std::vector<int>& layer, int& n_anchors) {
    const int N = gf->n_nodes;
    chain.assign(N, 0);
    layer.assign(N, -1);
    n_anchors = 0;
    std::vector<int> q;
    std::vector<char> inq(N, 0);
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

    // ---- capture: topological closure (BFS from routed mm anchors), shared
    // with assign_backend.
    std::vector<int> layer;
    std::vector<char> chain;
    int n_anchors = 0;
    collect_chain(gf, chain, layer, n_anchors);

    // ---- debug print: per-layer tree (graph order = execution order), capture
    // span, and odd/even ping-pong buffer budgets (compute nodes only, views
    // skipped; even = first compute node, odd = second).
    std::map<int, std::vector<int>> by_layer;
    int chain_total = 0;
    for (int i = 0; i < N; ++i) {
        if (chain[i]) { by_layer[layer[i]].push_back(i); chain_total++; }
    }
#ifdef STREAM_MOE_CHAIN_DEBUG
    fprintf(stderr, "[route_b_cap] anchors=%d chain_nodes=%d/%d\n", n_anchors, chain_total, N);
#endif
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
#ifdef STREAM_MOE_CHAIN_DEBUG
    fprintf(stderr, "[route_b_cap] ping-pong budget (per layer): even_buf=%zuMB  odd_buf=%zuMB  total=%zuMB\n",
            g_even_max / (1024 * 1024), g_odd_max / (1024 * 1024),
            (g_even_max + g_odd_max) / (1024 * 1024));
#endif

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
#ifdef STREAM_MOE_CHAIN_DEBUG
    fprintf(stderr, "[route_b_cap] external_violations=%d\n", violations);
#endif

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
