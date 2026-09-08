#include "backend/minigraph_exec.h"
#include "backend/route_b_chain.h"
#include "backend/moe_backend.h"
#include "backend/mix_split.h"
#include "backend/scatter_plan.h"
#include "common/logger.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
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

} // namespace

static bool is_view_op(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE ||
           n->op == GGML_OP_PERMUTE || n->op == GGML_OP_CONT;
}


// Slot of a pinned (layer, expert), or -1.
static int32_t pin_slot(const std::vector<expert_handle_t>& pins, uint32_t layer, uint32_t expert) {
    for (const auto & h : pins) {
        if (h.pinned && h.layer == layer && h.expert == expert) return h.slot;
    }
    return -1;
}

#ifdef STREAM_MOE_TEMP
// Test-only forced split (docs/BUCKET_EXEC_TOKEN_SUBSET.md SS3): turn the real
// plan's single full round into SCATTERED token-subset rounds so the subset path
// (cur gather + index-gather weights + scatter_plan reorder) is exercised on the
// CPU engine. Splits BOTH axes by parity so (t,k) is non-contiguous. Throwaway
// validation code - delete once the subset path is verified.
static std::vector<mix_round_t> test_split_rounds(
        const int32_t * ids, uint32_t n_k, uint32_t n_t,
        const int32_t * expert_pool, uint32_t n_expert, uint32_t n_pools) {
    mix_plan_t base = build_mix_plan(ids, n_k, n_t, expert_pool, n_expert, n_pools);
    if (base.rounds.size() != 1 || base.rounds[0].width != n_k ||
        base.rounds[0].n_active != n_t) {
        return base.rounds;
    }
    const uint32_t ksplit = (n_k % 2u == 0u) ? 2u : 1u;
    const uint32_t kw     = (ksplit == 2u) ? n_k / 2u : n_k;
    std::vector<mix_round_t> out;
    for (uint32_t kp = 0; kp < ksplit; ++kp) {
        for (uint32_t tp = 0; tp < 2u; ++tp) {
            mix_round_t r;
            r.pool  = base.rounds[0].pool;
            r.width = kw;
            for (uint32_t t = tp; t < n_t; t += 2u) ++r.n_active;
            if (r.n_active == 0 || kw == 0) continue;
            r.ids.reserve((size_t) r.n_active * kw);
            r.scatter.reserve((size_t) r.n_active * kw);
            for (uint32_t t = tp; t < n_t; t += 2u) {
                for (uint32_t s = 0; s < kw; ++s) {
                    const uint32_t k = (ksplit == 2u) ? (2u * s + kp) : s;
                    r.ids.push_back(ids[(size_t) t * n_k + k]);
                    r.scatter.push_back({ t, k });
                }
            }
            out.push_back(std::move(r));
        }
    }
    return out;
}
#endif

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

#ifdef STREAM_MOE_TEMP
// ---- closure structure dump (TEMP diagnostics) ----------------------------
// Prints the whole-layer closure as the chain path sees it (ex->compute: op /
// name / ne / nbytes / src role) plus the external leaves and view aliases that
// feed it, so the compact twin engine is written against the REAL captured
// graph rather than assumed topology. Env STREAM_MOE_TMP_CHAIN_DUMP_STRUCT=1,
// optional STREAM_MOE_TMP_CHAIN_DUMP_LAYER=<N> (default 0).
static void tmp_dump_chain_struct(const moe_layer_exec_t * ex) {
    if (!ex) return;
    fprintf(stderr, "[chain_struct] L%d closure: %zu compute nodes\n",
            ex->layer, ex->compute.size());
    for (size_t i = 0; i < ex->compute.size(); ++i) {
        const ggml_tensor * nd = ex->compute[i];
        if (!nd) { fprintf(stderr, "  [%zu] (null)\n", i); continue; }
        fprintf(stderr, "  [%zu] op=%-12s name=%-28s ne=[%lld,%lld,%lld,%lld] nb=%zu data=%p\n",
                i, ggml_op_name(nd->op),
                nd->name ? nd->name : "(anon)",
                (long long) nd->ne[0], (long long) nd->ne[1],
                (long long) nd->ne[2], (long long) nd->ne[3],
                ggml_nbytes(nd), (void*) nd->data);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (!src) continue;
            // role: is the src produced inside this closure (chain-internal) or
            // an external leaf (weight/cur/ids/scale from llama's side)?
            const char * role = "EXTERN";
            for (const auto * cn : ex->compute) if (cn == src) { role = "CHAIN"; break; }
            fprintf(stderr, "        s%d[%s] op=%-12s name=%-30s ne=[%lld,%lld,%lld,%lld] nb=%zu data=%p\n",
                    s, role, ggml_op_name(src->op),
                    src->name ? src->name : "(anon)",
                    (long long) src->ne[0], (long long) src->ne[1],
                    (long long) src->ne[2], (long long) src->ne[3],
                    ggml_nbytes(src), (void*) src->data);
        }
    }
    if (!ex->external_leaves.empty()) {
        fprintf(stderr, "  external leaves (%zu):\n", ex->external_leaves.size());
        for (const auto & el : ex->external_leaves) {
            fprintf(stderr, "    [%s] role=%-6s used by %-30s ne=[%lld,%lld,%lld] nb=%zu\n",
                    el.tensor && el.tensor->name ? el.tensor->name : "?",
                    el.role ? el.role : "?",
                    el.user ? el.user : "?",
                    (long long)(el.tensor ? el.tensor->ne[0] : 0),
                    (long long)(el.tensor ? el.tensor->ne[1] : 0),
                    (long long)(el.tensor ? el.tensor->ne[2] : 0),
                    el.tensor ? ggml_nbytes(el.tensor) : 0);
        }
    }
    if (!ex->view_aliases.empty()) {
        fprintf(stderr, "  view aliases (%zu):\n", ex->view_aliases.size());
        for (const auto & va : ex->view_aliases) {
            fprintf(stderr, "    view %-30s <- prod %-30s off=%lld\n",
                    va.view && va.view->name ? va.view->name : "(anon)",
                    va.prod && va.prod->name ? va.prod->name : "(anon)",
                    (long long) va.off);
        }
    }
    fflush(stderr);
}
#endif
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
    int64_t                 n_slots = 0;          // full routed slot count (ids ne0)
    // acc_d: the device's expert-folded accumulator [d_out, n_t] (bucketized
    // path only; single full-width bucket fold goes through fold_buf + exit).
    // Process-lifetime grow-only, zeroed per layer before the bucket loop.
    std::vector<float>      acc_d;
};

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
    return acc;
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

// =============== compact-chain bucket engine (the only executor) ==========
// The ONLY whole-layer executor: the round list comes from build_mix_plan (see
// exec_layer_burst_chain_buckets). Each round is rebuilt COMPACT ([d, w_b,
// n_active]) - mm over the round's ids subset writing a compact dst, weightless
// twins narrowed to w_b - then folded over w_b and scatter-added into acc_d; the
// exit writes moe_out.
//
// Closure facts this relies on (gemma L0, verified by STREAM_MOE_TMP_CHAIN_DUMP_STRUCT):
//   chain tensors [d, n_k, n_t]; slot axis = ne1 (per-token routed slot); fold
//   ADD tree sums ne1 -> [d_out, n_t]; scale REPEAT node_56 [1,128,T] is full
//   (per-expert), GET_ROWS node_57 [1,w_b,T] gathers per (slot,token) by the
//   round expert ids; down mm cur = the round's own compact GLU twin (kernel
//   reads src1 col = slot % ne11 with ne11 = w_b).

// --- per-bucket compact state -----------------------------------------------
struct bucket_build_t {
    chain_ctx_t *          c = nullptr;
    const moe_layer_exec_t* ex = nullptr;
    // current round geometry: w_b = slots per active token, n_active = tokens
    int64_t w_b = 0, n_active = 0;
    // index of the closure node currently being cloned (== its out_off slot).
    // Twins write their compact output at the SAME out_off region as the main
    // full-width node (compact [d, w_b, n_t] <= full [d, n_k, n_t], so it stays
    // inside the verify-allocated byte range); the one-cgraph serial execution
    // makes bucket N+1's chain reuse bucket N's dead regions (M2 §7.2.1).
    int64_t seq = -1;
    // routing ids (expert ids) for the bucket rows [w_b, n_active], t-major,
    // and slot-local translation [w_b, n_active]
    std::vector<int32_t>   ids_exp;     // kept alive (leaf data ptr)
    std::vector<int32_t>   ids_slot;    // kept alive (mm leaf data ptr)
    const ggml_tensor *    ids = nullptr;     // main routing ids (read-only, full)
    const ggml_tensor *    ids_data = nullptr;   // full main ids data (read-only)
    int64_t ids_ne0 = 0, ids_ne1 = 0;
    // current round (mix_plan): r->ids / r->scatter are the round's column set.
    // t_round[a] = original token of round column a (a order); order[i] = the
    // round column of tight column i (scatter_plan). gather_cache dedups the
    // per-round external gather nodes (cleared per round; nodes stay in c.gf).
    const mix_round_t *        r = nullptr;
    std::vector<uint32_t>      t_round;
    std::vector<uint32_t>      order;
    std::unordered_map<const ggml_tensor*, ggml_tensor*> gather_cache;
    // main node -> compact twin (op built at w_b width, data pinned into the
    // layer result arena when the closure has a layout, else c->fold_buf)
    std::unordered_map<const ggml_tensor*, ggml_tensor*> twin;
    // diagnostics: how many twin outputs landed in the arena vs heap fallback
    int64_t n_arena = 0, n_heap = 0;
    // small helper: fresh float buffer kept alive until graph_compute
    float * buf(size_t n) { c->fold_buf.emplace_back(n, 0.0f); return c->fold_buf.back().data(); }
    // Output region for the twin of the current closure node (b.seq). When the
    // layer has a verify layout the twin lands at arena + out_off[seq]; returns
    // nullptr if the layout is unavailable or the node has no slot -> caller
    // falls back to its own heap buffer.
    void * twin_out(size_t nbytes) {
        if (!ex || !ex->layout_ok || seq < 0 || seq >= (int64_t) ex->out_off.size() ||
            ex->out_off[(size_t) seq] < 0) return nullptr;
        const size_t need = (size_t) ex->out_off[(size_t) seq] + nbytes;
        void * base = moe_chain_fullalloc_buffer(need);
        return base ? static_cast<char*>(base) + ex->out_off[(size_t) seq] : nullptr;
    }
};

// Make a plain NONE leaf with the given geometry/data (helper to avoid repeated
// ggml_new_tensor_4d + nb-copy boilerplate).
static ggml_tensor * bucket_mk_leaf(chain_ctx_t & c, enum ggml_type type,
                                    const int64_t ne[4], const size_t nb[4], void * data) {
    ggml_tensor * l = ggml_new_tensor_4d(c.ctx, type, ne[0], ne[1], ne[2], ne[3]);
    for (int i = 0; i < 4; ++i) l->nb[i] = nb[i];
    l->data = data;
    return l;
}

// ---- general index-gather (docs/BUCKET_EXEC_TOKEN_SUBSET.md SS2.1/2.1a) ------
// get_rows gathers the ROW axis (ne1) of a 2D source. Build a flat / 2D source
// leaf over `m->data`, an i32 index leaf in tight order, get_rows, reshape to
// the chain geometry. The reshape view carries the data dependency, so it is the
// consumer's src (a detached leaf would lose the edge).
static ggml_tensor * bucket_gather_per_slot(bucket_build_t & b, const ggml_tensor * m) {
    // m is a per-(slot, token) external leaf [1, n_k, n_t] (routing weights). A
    // mix_plan round selects an ARBITRARY (t,k) subset, so the affine k-slice no
    // longer applies: gather flat element offsets.
    chain_ctx_t & c = *b.c;
    const size_t esz = sizeof(float);
    // flat source [1, span] over m->data; index uses the tensor's real strides
    // so a strided view still works. span = max flat element index + 1.
    const size_t stride_k = m->nb[1], stride_t = m->nb[2];
    const size_t span = (stride_t * (size_t)(m->ne[2] ? m->ne[2] - 1 : 0) +
                         stride_k * (size_t)(m->ne[1] ? m->ne[1] - 1 : 0)) / esz + 1;
    int64_t sne[4] = { 1, (int64_t) span, 1, 1 };
    size_t  snb[4] = { esz, esz, span * esz, span * esz };
    ggml_tensor * flat = bucket_mk_leaf(c, m->type, sne, snb, m->data);
    std::vector<int32_t> idx((size_t) b.w_b * b.n_active);
    for (int64_t i = 0; i < b.n_active; ++i) {
        const uint32_t a = b.order[(size_t) i];
        for (int64_t s = 0; s < b.w_b; ++s) {
            const mix_scatter_t & sc = b.r->scatter[(size_t) a * b.w_b + (size_t) s];
            idx[(size_t) i * b.w_b + (size_t) s] =
                (int32_t)((sc.k * stride_k + sc.t * stride_t) / esz);
        }
    }
    c.mm_ids_pool.push_back(std::move(idx));
    const int64_t n = (int64_t) b.w_b * b.n_active;
    int64_t ine[4] = { n, 1, 1, 1 };
    size_t  inb[4] = { 4, (size_t) n * 4, (size_t) n * 4, (size_t) n * 4 };
    ggml_tensor * idx_leaf = bucket_mk_leaf(c, GGML_TYPE_I32, ine, inb,
                                            c.mm_ids_pool.back().data());
    ggml_tensor * gr = ggml_get_rows(c.ctx, flat, idx_leaf);
    gr->data = b.buf((size_t) b.w_b * b.n_active);
    return ggml_reshape_3d(c.ctx, gr, 1, b.w_b, b.n_active);
}

static ggml_tensor * bucket_gather_cur(bucket_build_t & b, const ggml_tensor * m) {
    // m is the shared activation [d, 1, n_t] (one column per token). Gather the
    // token axis in tight order -> [d, 1, n_active]. Token stride is m->nb[2]
    // (ne1 == 1); the 2D source leaf [d, n_t] uses it as the row stride.
    chain_ctx_t & c = *b.c;
    const int64_t d = m->ne[0], n_t = m->ne[2];
    const size_t  esz = sizeof(float);
    int64_t sne[4] = { d, n_t, 1, 1 };
    size_t  snb[4] = { m->nb[0], m->nb[2], m->nb[2] * (size_t) n_t,
                       m->nb[2] * (size_t) n_t };
    ggml_tensor * src2d = bucket_mk_leaf(c, m->type, sne, snb, m->data);
    std::vector<int32_t> idx((size_t) b.n_active);
    for (int64_t i = 0; i < b.n_active; ++i) {
        idx[(size_t) i] = (int32_t) b.t_round[b.order[(size_t) i]];
    }
    c.mm_ids_pool.push_back(std::move(idx));
    int64_t ine[4] = { b.n_active, 1, 1, 1 };
    size_t  inb[4] = { 4, (size_t) b.n_active * 4, (size_t) b.n_active * 4,
                       (size_t) b.n_active * 4 };
    ggml_tensor * idx_leaf = bucket_mk_leaf(c, GGML_TYPE_I32, ine, inb,
                                            c.mm_ids_pool.back().data());
    ggml_tensor * gr = ggml_get_rows(c.ctx, src2d, idx_leaf);
    gr->data = b.buf((size_t) d * b.n_active);
    return ggml_reshape_3d(c.ctx, gr, d, 1, b.n_active);
}

// External leaf: per-slot [1,n_k,n_t] -> tight index-gather; otherwise a plain
// full leaf (expert weight tables / scale broadcast, untouched by bucketing).
static ggml_tensor * bucket_ext_leaf(bucket_build_t & b, const ggml_tensor * m) {
    chain_ctx_t & c = *b.c;
    const int64_t n_k = b.ids_ne0;
    if (m->ne[0] == 1 && m->ne[1] == n_k) {
        auto it = b.gather_cache.find(m);
        if (it == b.gather_cache.end()) {
            it = b.gather_cache.emplace(m, bucket_gather_per_slot(b, m)).first;
        }
        return it->second;
    }
    int64_t ne[4]; size_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = m->ne[i]; nb[i] = m->nb[i]; }
    return bucket_mk_leaf(c, m->type, ne, nb, m->data);
}

// Leaf twin of a chain producer `p` (already built) narrowed to the bucket.
static ggml_tensor * bucket_twin_leaf(bucket_build_t & b, const ggml_tensor * p) {
    auto it = b.twin.find(p);
    if (it == b.twin.end()) return nullptr;
    const ggml_tensor * t = it->second;
    int64_t ne[4]; size_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = t->ne[i]; nb[i] = t->nb[i]; }
    return bucket_mk_leaf(*b.c, t->type, ne, nb, t->data);
}

// Resolve a weightless clone's src[s]: a chain producer twin leaf, a view of a
// chain producer (data = producer twin + view byte offset along d), or an
// external data leaf (cur/ids/scale table). Views over hidden producers only
// slice along d (ne0) in this gemma closure (gate/up halves at view_offs 0 /
// n_ff*esize), so the twin view keeps the twin's ne[1..3]/nb and only slices d.
static ggml_tensor * bucket_src_leaf(bucket_build_t & b, const ggml_tensor * src) {
    // unwrap view/layout chain to the producer root, tracking byte offset
    const ggml_tensor * root = src;
    int64_t off = 0;
    while (root && (root->op == GGML_OP_VIEW || root->op == GGML_OP_RESHAPE ||
                    root->op == GGML_OP_TRANSPOSE || root->op == GGML_OP_PERMUTE ||
                    root->op == GGML_OP_CONT)) {
        if (root->op == GGML_OP_VIEW) off += root->view_offs;
        root = root->src[0];
    }
    if (root) {
        auto it = b.twin.find(root);
        if (it != b.twin.end()) {
            const ggml_tensor * t = it->second;
            int64_t ne[4]; size_t nb[4];
            for (int i = 0; i < 4; ++i) { ne[i] = t->ne[i]; nb[i] = t->nb[i]; }
            // apply the outermost view's d-slice: ne0 (d rows) from the src view
            ne[0] = src->ne[0];
            return bucket_mk_leaf(*b.c, t->type, ne, nb,
                                  static_cast<char*>(t->data) + off);
        }
    }
    // external leaf: cur (shared per token) / ids / scale table / weights
    return bucket_ext_leaf(b, src);
}

// Compact mm shell for the current bucket. `nd` is the main MUL_MAT_ID node.
// ids = the bucket's slot-local subset [w_b, n_active]; cur = chain twin leaf
// (down: the bucket's own compact GLU) or external shared cur (gate_up). dst =
// compact [d_out, w_b, n_active] region (NOT nd->data).
static ggml_tensor * append_mm_bucket(bucket_build_t & b, ggml_tensor * nd) {
    chain_ctx_t & c = *b.c;
    ggml_tensor * w   = const_cast<ggml_tensor*>(nd->src[0]);
    ggml_tensor * cur = const_cast<ggml_tensor*>(nd->src[1]);
    if (!b.ids_data) return nullptr;
    parsed_node_t pn = parse_weight_name(w->name);
    if (!pn.ok) return nullptr;
    uint32_t gidx = c.sched->group_of(pn.layer);
    if (gidx == static_cast<uint32_t>(-1)) return nullptr;

    // slot-local ids for the bucket rows (all tokens): reuse pool-region index.
    const int32_t se0 = (int32_t) b.ids_exp[0];
    const int32_t slot0 = pin_slot(*c.pins, pn.layer, (uint32_t) se0);
    if (slot0 < 0) return nullptr;
    const expert_scheduler::subpool_t * sp = c.sched->subpool_of_slot(slot0);
    if (!sp) return nullptr;
    size_t col_off = 0, col_stride = 0; uint32_t col_index = 0;
    if (!c.sched->column_layout(*sp, w->name, col_off, col_stride, col_index)) return nullptr;

    // ids leaf data already staged in b.ids_slot (slot - slot_begin), t-major
    // [w_b, n_active]; feed with the leaf's shape.
    c.mm_ids_pool.push_back(b.ids_slot);
    int64_t ne_ids[4] = { b.w_b, b.n_active, 1, 1 };
    size_t nb_ids[4] = { 4, (size_t)(b.w_b) * 4, (size_t)(b.w_b * b.n_active) * 4, 0 };
    nb_ids[3] = nb_ids[2];
    ggml_tensor * ids_leaf = bucket_mk_leaf(c, GGML_TYPE_I32, ne_ids, nb_ids,
                                            c.mm_ids_pool.back().data());

    ggml_tensor * w3d = ggml_new_tensor_3d(c.ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = static_cast<int32_t>(sp->n_slots);
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * sp->n_slots;
    w3d->data  = sp->base + col_off;

    // cur: chain twin (down mm reads the bucket's own compact GLU) or external
    // shared cur leaf (gate_up, ne11 == 1 -> every bucket slot reads col 0).
    ggml_tensor * cur_leaf = nullptr;
    if (cur->op != GGML_OP_NONE) {
        cur_leaf = bucket_twin_leaf(b, cur);
    }
    if (!cur_leaf) {
        // External shared activation: gather the token axis (GPU shape: each
        // bucket owns its cur copy). Cached per round (gate + up mm share it).
        auto it = b.gather_cache.find(cur);
        if (it == b.gather_cache.end()) {
            it = b.gather_cache.emplace(cur, bucket_gather_cur(b, cur)).first;
        }
        cur_leaf = it->second;
        if (!cur_leaf) return nullptr;
    }
    ggml_tensor * mm = ggml_mul_mat_id(c.ctx, w3d, cur_leaf, ids_leaf);
    // compact dst [d_out, w_b, n_active]: pinned into the layer result arena at
    // this closure node's out_off (full-width region; compact is smaller so it
    // stays inside). Falls back to a per-bucket heap buffer when the layer has
    // no verify layout (then every bucket keeps its own copies - safe, just no
    // serial reuse).
    const size_t nf = (size_t)(w->ne[1] * b.w_b * b.n_active);
    void * dst_arena = b.twin_out(nf * sizeof(float));
    if (dst_arena) ++b.n_arena; else ++b.n_heap;
    mm->data = dst_arena ? dst_arena : (void*) b.buf(nf);
    // nb: contiguous compact layout (ggml_new_tensor_4d would not apply to an op)
    mm->nb[0] = 4;
    mm->nb[1] = (size_t)(w->ne[1]) * 4;
    mm->nb[2] = (size_t)(w->ne[1] * b.w_b) * 4;
    mm->nb[3] = mm->nb[2] * (size_t) b.n_active;
    ggml_build_forward_expand(c.gf, mm);
    b.twin[nd] = mm;
    return mm;
}

// Compact weightless clone for the current bucket: same op/op_params as `nd`,
// slot axis narrowed to w_b, srcs resolved to bucket twins / compact leaves.
// fold_repl skips the anonymous fold + moe_out. REPEAT (per-expert scale) is
// full-width; GET_ROWS takes the bucket ids_exp subset (expert ids, t-major).
static ggml_tensor * append_op_bucket(bucket_build_t & b, ggml_tensor * nd) {
    chain_ctx_t & c = *b.c;
    const bool is_out = nd->name && strstr(nd->name, "ffn_moe_out") != nullptr;
    if (is_out || (nd->op == GGML_OP_ADD && !(nd->name && strstr(nd->name, "ffn_moe_"))))
        return nullptr;   // fold_repl: replaced by append_expert_fold / acc
    if (nd->op == GGML_OP_MUL_MAT_ID) return append_mm_bucket(b, nd);

    // GET_ROWS / REPEAT / MUL / GLU / ... : op clone with narrowed slot axis.
    int64_t ne[4]; size_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = nd->ne[i]; nb[i] = nd->nb[i]; }
    // narrow the slot axis to w_b. In this closure every chain weightless output
    // carries the per-token routed slot on ne1 ([d, n_k, n_t] or [1, n_k, n_t]);
    // fold-ADD outputs (token axis on ne1) and moe_out never reach this branch
    // (returned nullptr above). REPEAT node_56 is [1,128,n_t] (128 experts) and
    // is bucket-independent - its ne1 != n_k so it stays full width.
    const int64_t n_k = b.ids_ne0;
    if (nd->ne[1] == n_k && nd->ne[0] != n_k) ne[1] = b.w_b;
    // Token axis is ne2 for every chain tensor; narrow it to the round's active
    // token subset. (Fold/moe_out, which carry tokens on ne1, never reach here.)
    if (nd->ne[2] == b.ids_ne1) ne[2] = b.n_active;
    // contiguous compact dst (op outputs in this closure are f32)
    size_t nf = 1;
    for (int i = 0; i < 4; ++i) nf *= (size_t) ne[i];
    // rebuild contiguous nb for the compact dst
    nb[0] = 4;
    for (int i = 1; i < 4; ++i) nb[i] = nb[i-1] * (size_t) ne[i-1];

    ggml_tensor * cl = ggml_new_tensor_4d(c.ctx, nd->type, ne[0], ne[1], ne[2], ne[3]);
    for (int i = 0; i < 4; ++i) cl->nb[i] = nb[i];
    cl->op = nd->op;
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS; ++i) cl->op_params[i] = nd->op_params[i];
    // per-src resolution. GET_ROWS src1 = ids: replace with the bucket ids_exp
    // subset (expert ids, i32, t-major [w_b, n_active]) - get_rows gathers rows
    // of the REPEAT scale table by expert id.
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        ggml_tensor * src = nd->src[s];
        if (!src) continue;
        ggml_tensor * lf = nullptr;
        if (nd->op == GGML_OP_GET_ROWS && s == 1 && src->type == GGML_TYPE_I32) {
            // ids leaf must point at a buffer kept alive until graph_compute: the
            // per-bucket member is overwritten by the next bucket's assign().
            c.mm_ids_pool.push_back(b.ids_exp);
            int64_t sne[4] = { b.w_b, b.n_active, 1, 1 };
            size_t snb[4] = { 4, (size_t)(b.w_b) * 4, (size_t)(b.w_b * b.n_active) * 4, (size_t)(b.w_b * b.n_active) * 4 };
            lf = bucket_mk_leaf(c, GGML_TYPE_I32, sne, snb, c.mm_ids_pool.back().data());
#ifdef STREAM_MOE_TEMP
            if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG")) {
                int32_t mn = INT32_MAX, mx = INT32_MIN;
                for (size_t q = 0; q < b.ids_exp.size(); ++q) { mn = std::min(mn, b.ids_exp[q]); mx = std::max(mx, b.ids_exp[q]); }
                fprintf(stderr, "[chain_buckets] GET_ROWS ids_exp[%zu] range [%d,%d] src0=",
                        b.ids_exp.size(), mn, mx);
                fprintf(stderr, "%s", (nd->src[0] && nd->src[0]->name) ? nd->src[0]->name : "?");
                fprintf(stderr, " s0ne=[%lld,%lld,%lld,%lld] out_ne=[%lld,%lld,%lld]\n",
                        (long long) (nd->src[0] ? nd->src[0]->ne[0] : 0),
                        (long long) (nd->src[0] ? nd->src[0]->ne[1] : 0),
                        (long long) (nd->src[0] ? nd->src[0]->ne[2] : 0),
                        (long long) (nd->src[0] ? nd->src[0]->ne[3] : 0),
                        (long long) cl->ne[0], (long long) cl->ne[1], (long long) cl->ne[2]);
            }
#endif
        } else {
            lf = bucket_src_leaf(b, src);
        }
        if (!lf) return nullptr;
        cl->src[s] = lf;
    }
    void * dst_arena = b.twin_out(nf * sizeof(float));
    if (dst_arena) ++b.n_arena; else ++b.n_heap;
    cl->data = dst_arena ? dst_arena : (void*) b.buf(nf);
    ggml_build_forward_expand(c.gf, cl);
    b.twin[nd] = cl;
    return cl;
}

// Build one bucket: walk the closure and produce the compact chain; returns the
// final (weighted) twin (the last before the anonymous fold when fold_repl).
// Each closure node's twin writes its compact output at arena + out_off[i]
// (same byte region the main full-width node occupies; compact is smaller, so
// it stays inside). In one serial cgraph bucket N+1's chain then reuses bucket
// N's dead regions (M2 §7.2.1) - no per-bucket heap intermediates.
static ggml_tensor * append_bucket_chain_compact(bucket_build_t & b) {
    ggml_tensor * last = nullptr;
    for (size_t i = 0; i < b.ex->compute.size(); ++i) {
        ggml_tensor * nd = const_cast<ggml_tensor*>(b.ex->compute[i]);
        if (is_view_op(nd)) continue;
        b.seq = (int64_t) i;   // twin output slot == closure node's out_off
        ggml_tensor * t = nd->op == GGML_OP_MUL_MAT_ID ? append_mm_bucket(b, nd)
                                                       : append_op_bucket(b, nd);
        if (!t) continue;   // fold_repl: fold/moe_out return nullptr -> keep last
        last = t;
    }
    return last;
}

// Whole-layer compact-chain bucket engine (the only executor). The round list
// comes from build_mix_plan (real per-pool token-subset rounds; a single RAM
// pool degenerates to one full round). Each round builds a compact chain on
// [w_b, n_active], folds w_b -> a tight per-token partial, and scatter-adds it
// into acc_d at the original token positions via scatter_plan; the exit writes
// moe_out. See docs/BUCKET_EXEC_TOKEN_SUBSET.md.
static enum ggml_status exec_layer_burst_chain_buckets(int32_t layer, ggml_context * ctx,
                                                       ggml_backend_t cpu,
                                                       expert_scheduler & sched,
                                                       const moe_layer_exec_t * ex,
                                                       const std::vector<expert_handle_t>& pins) {
    const ggml_tensor * ids = nullptr;
    for (const auto * cn : ex->compute) {
        if (cn && cn->op == GGML_OP_MUL_MAT_ID && cn->src[2]) { ids = cn->src[2]; break; }
    }
    if (!ids || !ids->data) return GGML_STATUS_FAILED;
    const uint32_t n_k = static_cast<uint32_t>(ids->ne[0]);
    const uint32_t n_t = static_cast<uint32_t>(ids->ne[1]);
    const uint32_t n_expert = sched.topology().n_expert;
    const uint32_t n_pools  = sched.n_pools();

    // Compact routing ids (MOE_ID_AT honors the real row stride: hash layers are
    // compact, argsort layers are sparse). build_mix_plan wants [t*n_k + k].
    std::vector<int32_t> ids_compact((size_t) n_k * n_t, 0);
    for (uint32_t t = 0; t < n_t; ++t) {
        for (uint32_t k = 0; k < n_k; ++k) {
            ids_compact[(size_t) t * n_k + k] = MOE_ID_AT(ids, (int) t, (int) k);
        }
    }

    // Per-expert pool from the pinned handles (pin_layer returns one per active
    // expert with its pool). -1 = not active / unknown.
    std::vector<int32_t> expert_pool(n_expert, -1);
    for (const auto & h : pins) {
        if (h.expert < n_expert) expert_pool[h.expert] = (int32_t) h.pool;
    }

    // Round list: real build_mix_plan rounds (single RAM pool degenerates to one
    // full round); STREAM_MOE_TEMP may force the test-only scattered split.
    std::vector<mix_round_t> rounds;
#ifdef STREAM_MOE_TEMP
    const char * split_env = std::getenv("STREAM_MOE_TMP_BUCKET_ROUNDS");
    if (split_env && *split_env) {
        rounds = test_split_rounds(ids_compact.data(), n_k, n_t, expert_pool.data(),
                                   n_expert, n_pools);
    } else
#endif
    {
        mix_plan_t plan = build_mix_plan(ids_compact.data(), n_k, n_t,
                                         expert_pool.data(), n_expert, n_pools);
        rounds = std::move(plan.rounds);
    }
    if (rounds.empty()) {
        LOG_ERROR("stream_moe: chain_buckets empty round list L" << layer);
        return GGML_STATUS_FAILED;
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG")) {
        fprintf(stderr, "[chain_buckets] L%d: n_k=%u n_t=%u -> %zu rounds\n",
                layer, n_k, n_t, rounds.size());
    }
#endif

    ggml_cgraph * gf = ggml_new_graph(ctx);
    chain_ctx_t c;
    c.ctx   = ctx;   c.cpu   = cpu;   c.sched = &sched;
    c.topo  = &sched.topology();
    c.layer = layer; c.pins  = &pins; c.gf    = gf; c.ex    = ex;
    c.fold_repl = true;
    c.w_b = n_k; c.n_t = n_t; c.d_out = 0; c.device_used = 1;

    bucket_build_t b;
    b.c = &c; b.ex = ex; b.ids = ids; b.ids_data = ids;
    b.ids_ne0 = n_k; b.ids_ne1 = n_t;

    // acc_d: process-lifetime, sized to the fold output once known
    int64_t d_out = 0;
    for (const auto * cn : ex->compute) {
        if (!cn) continue;
        d_out = std::max(d_out, (int64_t)(cn->ne[0]));
    }
    if (d_out <= 0) return GGML_STATUS_FAILED;
    c.d_out = d_out;
    const size_t acc_sz = (size_t)(c.d_out * c.n_t);
    if (c.acc_d.size() < acc_sz) c.acc_d.assign(acc_sz, 0.0f);
    std::fill(c.acc_d.begin(), c.acc_d.end(), 0.0f);

    for (size_t ri = 0; ri < rounds.size(); ++ri) {
        const mix_round_t & r = rounds[ri];
        if (r.width == 0 || r.n_active == 0) continue;
        b.r = &r;
        b.w_b = r.width; b.n_active = r.n_active;

        // scatter_plan: t_round[a] = original token of round column a; order[i]
        // = round column of tight column i; segs = acc arithmetic runs.
        b.t_round.resize(r.n_active);
        for (uint32_t a = 0; a < r.n_active; ++a) {
            b.t_round[a] = r.scatter[(size_t) a * r.width].t;
        }
        scatter_plan_t sp = build_scatter_plan(b.t_round.data(), r.n_active, n_t);
        if (sp.order.size() != r.n_active) {
            LOG_ERROR("stream_moe: chain_buckets scatter_plan failed L" << layer
                      << " round " << ri << " (n_active " << r.n_active << ")");
            return GGML_STATUS_FAILED;
        }
        b.order = std::move(sp.order);

        // tight-order ids (expert id + pool-local slot id).
        b.ids_exp.assign((size_t) r.width * r.n_active, 0);
        b.ids_slot.assign((size_t) r.width * r.n_active, 0);
        bool ok = true;
        for (uint32_t i = 0; i < r.n_active && ok; ++i) {
            const uint32_t a = b.order[i];
            for (uint32_t s = 0; s < r.width; ++s) {
                const int32_t e = r.ids[(size_t) a * r.width + s];
                if (e < 0 || e >= static_cast<int32_t>(n_expert)) { ok = false; break; }
                const int32_t slot = pin_slot(pins, (uint32_t) layer, (uint32_t) e);
                if (slot < 0) { ok = false; break; }
                const expert_scheduler::subpool_t * osp = c.sched->subpool_of_slot(slot);
                if (!osp) { ok = false; break; }
                const size_t idx = (size_t) i * r.width + s;
                b.ids_exp[idx]  = e;
                b.ids_slot[idx] = slot - static_cast<int32_t>(osp->slot_begin);
            }
        }
        if (!ok) { LOG_ERROR("stream_moe: chain_buckets ids staging failed L" << layer); return GGML_STATUS_FAILED; }
#ifdef STREAM_MOE_TEMP
        if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG")) {
            fprintf(stderr, "[chain_buckets]   r%zu: pool=%u w_b=%u n_active=%u segs=%zu\n",
                    ri, r.pool, r.width, r.n_active, sp.segs.size());
        }
#endif
        // Keep the round's ids copies alive until graph_compute (append-only pool).
        c.mm_ids_pool.push_back(b.ids_exp);
        c.mm_ids_pool.push_back(b.ids_slot);
        b.ids_exp  = c.mm_ids_pool[c.mm_ids_pool.size() - 2];
        b.ids_slot = c.mm_ids_pool[c.mm_ids_pool.size() - 1];
        b.twin.clear();
        b.gather_cache.clear();
        b.n_arena = 0; b.n_heap = 0;
        ggml_tensor * last = append_bucket_chain_compact(b);
        if (!last) { LOG_ERROR("stream_moe: chain_buckets append failed L" << layer); return GGML_STATUS_FAILED; }
        ggml_tensor * per_token = append_expert_fold(c, last);
        if (!per_token) return GGML_STATUS_FAILED;
        if (per_token->ne[1] != (int64_t) r.n_active) {
            LOG_ERROR("stream_moe: chain_buckets fold width " << per_token->ne[1]
                      << " != n_active " << r.n_active << " L" << layer);
            return GGML_STATUS_FAILED;
        }
        // acc_d[d_out, n_t] += per_token tight columns, one ggml_acc per run.
        ggml_tensor * acc = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, c.d_out, c.n_t);
        acc->data = c.acc_d.data();
        acc->nb[0] = 4; acc->nb[1] = (size_t)(c.d_out) * 4;
        const size_t col = (size_t)(c.d_out) * 4;
        for (const auto & seg : sp.segs) {
            if (seg.len == 0) continue;
            ggml_tensor * pv = ggml_view_2d(c.ctx, per_token, c.d_out, seg.len,
                                            per_token->nb[1], (size_t) seg.src * per_token->nb[1]);
            ggml_tensor * a = ggml_acc_inplace(c.ctx, acc, pv,
                                               (size_t) seg.delta * col, 0, 0,
                                               (size_t) seg.dst * col);
            ggml_build_forward_expand(c.gf, a);
        }
    }
    // exit: acc_d -> moe_out (device_used == 1 -> the anonymous cross-device fold
    // is a plain copy). Reuse chain_exit's stage 2/3 with an acc tensor view of
    // the persistent acc_d buffer.
    {
        ggml_tensor * acc = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, c.d_out, c.n_t);
        acc->data = c.acc_d.data();
        acc->nb[0] = 4; acc->nb[1] = (size_t)(c.d_out) * 4;
        if (!chain_exit(c, acc)) return GGML_STATUS_FAILED;
    }
#ifdef STREAM_MOE_TEMP
    // Numeric gate dump: moe_out per layer (same tmp_dump_node harness as the
    // single-bucket path) so an offline diff gates acc_d results.
    if (std::getenv("STREAM_MOE_TMP_DUMP")) {
        size_t seq = 0;
        for (const auto * cn : ex->compute) {
            const ggml_tensor * m = cn;
            if (m && m->name && strstr(m->name, "ffn_moe_out") != nullptr) {
                tmp_dump_node(sched, *c.topo, pins, layer, seq, const_cast<ggml_tensor*>(cn));
            }
            ++seq;
        }
    }
#endif
    return GGML_STATUS_SUCCESS;
}

// Burst one whole layer from its captured sequence.
static enum ggml_status exec_layer_burst(int32_t layer, ggml_context * ctx,
                                         ggml_backend_t cpu, expert_scheduler& sched,
                                         int /*n_threads*/) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex || ex->compute.empty()) return GGML_STATUS_SUCCESS;
    const moe_model_topology_t& topo = sched.topology();
#ifdef STREAM_MOE_TEMP
    // Closure-structure dump (env STREAM_MOE_TMP_CHAIN_DUMP_STRUCT=1). Fire
    // once per process on the layer selected by STREAM_MOE_TMP_CHAIN_DUMP_LAYER
    // (default 0), before any exec, so the printed topology is the real
    // captured graph.
    {
        static bool dumped = false;
        const char * en = std::getenv("STREAM_MOE_TMP_CHAIN_DUMP_STRUCT");
        int want = 0;
        const char * wl = std::getenv("STREAM_MOE_TMP_CHAIN_DUMP_LAYER");
        if (wl && *wl) want = atoi(wl);
        if (en && std::string(en) == "1" && !dumped && layer == want) {
            dumped = true;
            tmp_dump_chain_struct(ex);
        }
    }
#endif

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
    // Batch semantics: ONE request carrying the whole layer's expert bitmap;
    // missing experts load concurrently (IOCP n-way), exec wakes once.
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
    // pin_layer fills handles in ascending-expert order; pin_slot scans by
    // (layer, expert), so the bucket engine accepts any handle order.

    // Hidden outputs live in the per-layer full-alloc block (result-buffer
    // layout when verify produced one, else the full-alloc sum).
    {
        const size_t need = (ex && ex->layout_ok) ? ex->result_bytes : lsum;
        moe_chain_set_full_alloc(need);
    }

    // The compact-chain bucket engine is the only executor: a full-width single
    // bucket by default (no env), or an env-selected multi-bucket cut.
    const enum ggml_status st = exec_layer_burst_chain_buckets(layer, ctx, cpu, sched, ex, pins);
    for (const auto & h : pins) sched.unpin(h);
    return st;
}

// graph_compute entry: dispatch to the whole-layer burst on the layer's first
// privatised node; later same-layer splits are no-ops (already produced by the
// burst). An un-captured MUL_MAT_ID (layer < 0) is a hard error - the legacy
// per-split path is gone.
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
    // View/layout splits of captured producers are no-ops: the layer burst
    // already fixed their data pointers (input_layouts) and computed the whole
    // chain. They reach us because the scheduler follows view_src of a
    // privatised mm output; there is nothing to execute here.
    if (is_view_op(first)) return GGML_STATUS_SUCCESS;

    const int32_t layer = moe_chain_layer_of_node(first);
    if (layer < 0) {
        fprintf(stderr,
                "[stream_moe] un-captured MUL_MAT_ID: node='%s' w='%s' op=%s ne=[%lld,%lld,%lld] n_nodes=%d\n",
                first->name ? first->name : "(anon)",
                first->src[0] && first->src[0]->name ? first->src[0]->name : "?",
                ggml_op_name(first->op),
                (long long) first->ne[0], (long long) first->ne[1], (long long) first->ne[2],
                n_nodes);
        LOG_ERROR("stream_moe: un-captured MUL_MAT_ID split reached the executor (no legacy path)");
        return GGML_STATUS_FAILED;
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
