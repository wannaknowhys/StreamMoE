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

    // ---- result-buffer layout (best-fit decreasing, node-level) ----------
    // Runs after the closure is complete (g_layer_exec holds every layer's
    // compute sequence, views excluded). For each layer: resolve every
    // chain-internal edge (consumer reads producer; views alias to the
    // producer) and take each result's LAST reader index (last_use); a result
    // stays live until last_use executes. Pack compute outputs into one
    // per-layer result block by BEST-FIT DECREASING: place nodes by size
    // descending, each at the lowest offset whose bytes do not collide with an
    // already-placed result whose live interval overlaps. Large blocks land
    // first so small ones fill their gaps (reaches the peak-simultaneous-live
    // lower bound; gemma ~180KB/layer, deepseek ~197KB/layer at 1 token, vs
    // full-alloc 417/524KB). Validated in diagnostics/layout_sim. Output:
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
        // Result-buffer layout by best-fit decreasing (offline interval packing,
        // docs layout_sim sim.js): place nodes in size-descending order; each
        // node gets the lowest offset whose byte range does not collide with any
        // already-placed result whose live interval [i, last_use] overlaps it.
        // Larger blocks land first so small blocks fill the gaps left behind -
        // reaches the lower bound (peak simultaneous-live bytes) on gemma and
        // deepseek (~6-20% below the old exec-order first-fit slot allocator).
        // Outputs absolute per-node offsets into one per-layer block.
        struct blk_t { int64_t off = 0; size_t size = 0; int start = 0, end = 0; };
        std::vector<int> order(C);
        for (size_t i = 0; i < C; ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int x, int y) {
            const size_t sx = comp[x] ? ggml_nbytes(comp[x]) : 0;
            const size_t sy = comp[y] ? ggml_nbytes(comp[y]) : 0;
            if (sx != sy) return sx > sy;             // larger first
            return x < y;
        });
        std::vector<blk_t> placed;
        int64_t arena = 0;
        for (const int ni : order) {
            const size_t nb = comp[ni] ? ggml_nbytes(comp[ni]) : 0;
            const int start = ni;
            const int end = last_use[ni] < 0 ? (int) C : last_use[ni];
            int64_t off = 0;
            for (;;) {
                // blocks whose interval overlaps [start,end] AND whose bytes
                // overlap [off, off+nb) block this offset
                int64_t furthest = -1;
                for (const auto & p : placed) {
                    const bool t_ov = !(end < p.start || p.end < start);
                    if (!t_ov) continue;
                    const bool s_ov = off < (int64_t)(p.off + (int64_t) p.size) &&
                                      (int64_t) p.off < off + (int64_t) nb;
                    if (s_ov) furthest = std::max(furthest, p.off + (int64_t) p.size);
                }
                if (furthest < 0) break;   // no overlapping block: this off fits
                off = furthest;            // jump past the blocker
            }
            placed.push_back({ off, nb, start, end });
            arena = std::max(arena, off + (int64_t) nb);
        }
        for (size_t i = 0; i < C; ++i) {
            // recover per-node offset from placed (order maps to node id)
            for (const auto & p : placed) if (p.start == (int) i) { ex.out_off[i] = p.off; break; }
        }
        // ---- layout self-check (always): results whose live intervals overlap
        // in exec time must not share byte ranges. Violation -> per-node bump.
        ex.result_bytes = (size_t) arena;
        ex.layout_ok = true;
        for (size_t a = 0; a < C && ex.layout_ok; ++a) {
            for (size_t b = a + 1; b < C; ++b) {
                const int la = last_use[a] < 0 ? (int) C : last_use[a];
                const int lb = last_use[b] < 0 ? (int) C : last_use[b];
                if (!(!((int) a > lb || (int) b > la))) continue;   // time-disjoint: ok
                const int64_t oa = ex.out_off[a], ob = ex.out_off[b];
                const size_t sa = comp[a] ? ggml_nbytes(comp[a]) : 0;
                const size_t sb = comp[b] ? ggml_nbytes(comp[b]) : 0;
                const bool space_overlap = !((int64_t)(oa + (int64_t) sa) <= ob ||
                                             (int64_t)(ob + (int64_t) sb) <= oa);
                if (space_overlap) {
                    fprintf(stderr,
                        "[route_b_cap] LAYOUT CLASH: L%d node a=%zu(last %d,off %lld,sz %zu) "
                        "overlaps b=%zu(last %d,off %lld,sz %zu)\n",
                        kv.first, a, la, (long long) oa, sa, b, lb, (long long) ob, sb);
                    ex.layout_ok = false;
                }
            }
        }

        // ---- external-leaf (input-side) analysis (SS7.6.5) -------------------
        // Every src of a closure compute node whose producer (views unwrapped)
        // is NOT a node of this layer's compute sequence is an external leaf:
        // a llama-side tensor the closure consumes. The per-device graph must
        // upload or reference these as leaf inputs. Classify by the consuming
        // op / src slot for the offline dump (no executor change).
        ex.external_leaves.clear();
        auto in_compute = [&](const ggml_tensor * p) -> bool {
            for (const auto * cn : ex.compute) if (cn == p) return true;
            return false;
        };
        for (const auto * cn : ex.compute) {
            if (!cn) continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * t = cn->src[s];
                if (!t) continue;
                // unwrap view/layout chain to the producer root
                const ggml_tensor * p = t;
                while (p && is_view_op(p)) p = p->src[0];
                if (!p) continue;
                if (in_compute(p)) continue;   // chain-internal (or leaf of chain)
                // external leaf - dedupe by tensor pointer
                bool seen = false;
                for (const auto & el : ex.external_leaves)
                    if (el.tensor == const_cast<ggml_tensor*>(p)) { seen = true; break; }
                if (seen) continue;
                const char * role = "other";
                if (cn->op == GGML_OP_MUL_MAT_ID) {
                    if (s == 0) role = "w";       // expert weight (pool-shell base)
                    else if (s == 1) role = "cur"; // activation (norm output)
                    else if (s == 2) role = "ids"; // routing ids
                } else if (cn->op == GGML_OP_MUL || cn->op == GGML_OP_GET_ROWS ||
                           cn->op == GGML_OP_REPEAT) {
                    role = "scale";               // per-expert scale sources
                }
                moe_layer_exec_t::ext_leaf_t el;
                el.tensor = const_cast<ggml_tensor*>(p);
                el.role   = role;
                el.user   = cn->name ? cn->name : ggml_op_name(cn->op);
                ex.external_leaves.push_back(el);
            }
        }
    }
#ifdef STREAM_MOE_TEMP
    if (getenv("STREAM_MOE_CAP_DUMP")) {
        fprintf(stderr, "\n=== [cap-dump] result-buffer layout (interval, node-level) ===\n");
        size_t gmax = 0;
        for (const auto & kv : g_layer_exec) {
            const moe_layer_exec_t & ex = kv.second;
            size_t sum_all = 0;   // full-alloc reference (every node its own block)
            for (const auto * nd : ex.compute) sum_all += nd ? ggml_nbytes(nd) : 0;
            fprintf(stderr, "  L%d: %zu compute, layout=%zuB  full-alloc(ref)=%zuB  (interval saves %.0f%%)\n",
                    kv.first, ex.compute.size(), ex.result_bytes, sum_all,
                    sum_all ? 100.0 * (1.0 - (double) ex.result_bytes / (double) sum_all) : 0.0);
            // group nodes by out_off (== same reuse slot), print each slot's
            // occupants and their individual bytes to expose per-slot waste
            // (slot is sized to its largest occupant; smaller reusees overlap).
            std::map<int64_t, std::vector<size_t>> by_off;
            for (size_t i = 0; i < ex.compute.size(); ++i) {
                if (i < ex.out_off.size()) by_off[ex.out_off[i]].push_back(i);
            }
            size_t slot_n = 0;
            for (const auto & b : by_off) {
                fprintf(stderr, "    slot#%zu @ off=%-8lld size=%zu occupants=",
                        slot_n++, (long long) b.first,
                        [&] { size_t m = 0; for (auto i : b.second) m = std::max(m, ggml_nbytes(ex.compute[i])); return m; }());
                for (size_t k = 0; k < b.second.size(); ++k) {
                    const size_t i = b.second[k];
                    const ggml_tensor * nd = ex.compute[i];
                    fprintf(stderr, "%s%s(%zuB)", k ? "," : "", nd && nd->name ? nd->name : "?", ggml_nbytes(nd));
                }
                fprintf(stderr, "\n");
            }
            if (ex.result_bytes > gmax) gmax = ex.result_bytes;
            // external leaves consumed by this layer's closure (SS7.6.5)
            if (!ex.external_leaves.empty()) {
                fprintf(stderr, "    external leaves (%zu):\n", ex.external_leaves.size());
                for (const auto & el : ex.external_leaves) {
                    fprintf(stderr, "      [%s] %-5s used by %-30s ne=[%lld,%lld,%lld] nb=%zu\n",
                            el.tensor && el.tensor->name ? el.tensor->name : "?",
                            el.role ? el.role : "?",
                            el.user ? el.user : "?",
                            (long long)(el.tensor ? el.tensor->ne[0] : 0),
                            (long long)(el.tensor ? el.tensor->ne[1] : 0),
                            (long long)(el.tensor ? el.tensor->ne[2] : 0),
                            el.tensor ? ggml_nbytes(el.tensor) : 0);
                }
            }
        }
        fprintf(stderr, "[cap-dump] max layer result block = %zu bytes\n", gmax);
        fflush(stderr);
    }
    // Machine-readable export for the offline layout simulator
    // (diagnostics/layout_sim): one row per compute node.
    //   layer,exec_idx,name,op,bytes,last_use
    // last_use = index of the LAST node that reads this result (-1 if never read
    // inside the layer = live to the end). Interval is [exec_idx, last_use].
    if (getenv("STREAM_MOE_CAP_CSV")) {
        fprintf(stdout, "#layer,exec_idx,name,op,bytes,last_use\n");
        for (const auto & kv : g_layer_exec) {
            const moe_layer_exec_t & ex = kv.second;
            const auto & comp = ex.compute;
            const size_t C = comp.size();
            if (C == 0) continue;
            auto prod_of = [&](const ggml_tensor * t) -> int {
                const ggml_tensor * p = t;
                while (p && is_view_op(p)) p = p->src[0];
                if (!p) return -1;
                for (size_t i = 0; i < C; ++i) if (comp[i] == p) return (int) i;
                return -1;
            };
            std::vector<int> last_use(C, -1);
            for (int c = (int) C - 1; c >= 0; --c)
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const int p = prod_of(comp[c] ? comp[c]->src[s] : nullptr);
                    if (p >= 0 && p < c && last_use[p] < 0) last_use[p] = c;
                }
            for (size_t i = 0; i < C; ++i) {
                const ggml_tensor * nd = comp[i];
                fprintf(stdout, "%d,%zu,%s,%s,%zu,%d\n",
                        kv.first, i,
                        nd && nd->name ? nd->name : "(anon)",
                        nd ? ggml_op_name(nd->op) : "?",
                        nd ? ggml_nbytes(nd) : 0,
                        last_use[i]);
            }
        }
        fflush(stdout);
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
