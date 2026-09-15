#include "backend/route_b_chain.h"

#include "backend/alloc.h"
#include "backend/moe_backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace stream_moe {

// Best-fit-decreasing interval packing (defined below). Declared here so
// layout_arena (in the anonymous namespace) can use it.
static size_t pack_interval(const std::vector<ggml_tensor*> & nodes,
                            const std::vector<int> & start,
                            const std::vector<int> & end,
                            std::vector<int64_t> & out_off);

namespace {
void * g_fullalloc_buf = nullptr;
size_t g_fullalloc_cap = 0;

// Whole-layer arena (R3): one ggml_backend_buffer holding
// [carry][compact][closure]. Null on the production (MoE-only) path.
ggml_backend_buffer_t g_arena = nullptr;
size_t g_arena_cap = 0;
size_t g_arena_closure_off = 0;   // byte offset of the closure block inside g_arena
size_t g_arena_closure_size = 0;
int    g_dump_ubatch = -1;        // bin-dump ubatch subdirectory index
int    g_build_id = 0;            // layout_arena call counter (per-build seed)

// Per-device arena plan (docs/PER_DEVICE_ARENA.md 3). One plan per device, each
// with its own grow-only buffer. Two regions by liveness:
//   carry  - cross-layer pipeline (logically one, physically per device; moved
//            explicitly at a device boundary)
//   scratch- within-layer temporaries (compact merged with closure), reused
//            across layers
// Key = device name ("" = host). Filled by layout_arena (per-device grouping
// lands incrementally; today the host plan carries everything).
struct region_plan_t {
    std::string dev;
    ggml_backend_buffer_t buf = nullptr;
    size_t cap = 0;
    size_t off_carryN = 0, off_scratch = 0, off_xfer = 0;
    size_t carry1_size = 0, carryN_size = 0, scratch_size = 0, xfer_size = 0;
    size_t need = 0;
    std::unordered_map<const ggml_tensor*, size_t> carry1_off, carryN_off, scratch_off, xfer_off;
    std::set<const ggml_tensor*> cross_device;   // carry a remote consumer reads
};
static std::map<std::string, region_plan_t> g_plans;

// Device name -> backend (docs/PER_DEVICE_ARENA.md SS8.5). Recorded by
// route_b_setup; the executor uses it to run a layer's dense head/tail on the
// placement device. Host plan nodes resolve to the CPU backend (no entry).
static std::map<std::string, ggml_backend_t> g_dev_backends;
// Node -> device ("" = host), refreshed by layout_arena each build.
static std::unordered_map<const ggml_tensor*, std::string> g_node_dev;

// Cross-device carry relays of the current build (docs/PER_DEVICE_ARENA.md 3).
// A carry tensor is the producer node's output, so it stays on the producer's
// device; a consumer on another device gets a local copy (a shell allocated in
// the consumer's carry region) and its src is rewired to it. The executor runs
// ggml_backend_tensor_copy(src, dst) at the consumer layer's front.
static std::vector<route_b_relay_t> g_relays;
// Shell context for the relay copies (no_alloc; buffer/data set by layout_arena).
// Recreated every layout_arena - the graph is rebuilt, so shells never outlive
// the build that references them.
static ggml_context * g_relay_ctx = nullptr;

#if defined(STREAM_MOE_ROUTE_B) && defined(STREAM_MOE_PREFILL_EXPORT)
// Tensors the prefill export reads after compute (route_b_set_export_retained).
// layout_arena keeps them out of the reuse pool so the export's post-split read
// sees the computed value. Refreshed on every graph build.
static std::vector<const ggml_tensor*> g_export_retained;
#endif

bool is_alias_op(const ggml_tensor * n);   // defined later in this file
bool is_view_op(const ggml_tensor * n);    // defined later in this file

// C4 replication cap: closure-used non-per-expert leaves at or below this size
// are copied once into every device pool (gemma per-expert scale = 512 B/layer).
constexpr size_t kResidentLeafMaxBytes = 1u << 20;   // 1 MiB

// Whole-layer burst capture: per-layer privatised compute sequence from the
// last graph build (see moe_chain_assign_backend).
std::map<int, moe_layer_exec_t> g_layer_exec;
// Whole-layer node capture (dense + MoE, graph order) from the last build.
std::map<int, std::vector<ggml_tensor*>> g_layer_nodes;
// Debug-only: whole-layer capture populated on EVERY build (both paths), so the
// executor can dump per-node contents even on the MoE-only baseline path.
std::map<int, std::vector<ggml_tensor*>> g_layer_nodes_all;
// Official layer index per node, recorded from llama's graph-build callback
// (route_b_on_node). Populated during build, cleared by moe_chain_assign_backend.
std::unordered_map<const ggml_tensor*, int> g_official_layer;
// Build-time layer plans (docs/LAYER_EXECUTOR_DESIGN.md 4.1).
std::map<int, moe_layer_plan_t> g_layer_plan;

// Build the per-layer plan (head / MoE / tail / moe_out) from the captured layer
// list + the MoE closure. Same split rule as the old runtime code in
// exec_layer_burst: tail = forward closure of ffn_moe_out inside the layer;
// head = the rest (minus the MoE closure itself).
void build_layer_plans() {
    g_layer_plan.clear();
    for (auto & kv : g_layer_nodes) {
        const int L = kv.first;
        const std::vector<ggml_tensor*> & lns = kv.second;
        moe_layer_plan_t plan;
        plan.layer = L;
        plan.all = lns;
        auto ex_it = g_layer_exec.find(L);
        const moe_layer_exec_t * ex = (ex_it != g_layer_exec.end()) ? &ex_it->second : nullptr;
        plan.moe = ex;

        std::unordered_map<const ggml_tensor*, size_t> lidx;
        lidx.reserve(lns.size() * 2);
        for (size_t i = 0; i < lns.size(); ++i) lidx[lns[i]] = i;
        auto in_closure = [&](const ggml_tensor * t) {
            if (!ex) return false;
            for (const auto * cn : ex->compute) if (cn == t) return true;
            return false;
        };

        const ggml_tensor * out = nullptr;
        if (ex) for (const auto * cn : ex->compute)
            if (cn && cn->name && strstr(cn->name, "ffn_moe_out")) { out = cn; break; }
        plan.moe_out = const_cast<ggml_tensor*>(out);

        std::vector<char> down(lns.size(), 0);
        auto seed = out ? lidx.find(out) : lidx.end();
        if (seed != lidx.end()) down[seed->second] = 1;
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < lns.size(); ++i) {
                if (down[i]) continue;
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    auto pit = lidx.find(lns[i]->src[s]);
                    if (pit != lidx.end() && down[pit->second]) { down[i] = 1; changed = true; break; }
                }
            }
        }
        for (size_t i = 0; i < lns.size(); ++i) {
            if (in_closure(lns[i])) continue;
            if (down[i]) plan.tail.push_back(lns[i]);
            else         plan.head.push_back(lns[i]);
        }
#ifdef STREAM_MOE_TEMP
        if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG")) {
            size_t outside = 0;
            for (const auto * cn : (ex ? ex->compute : std::vector<ggml_tensor*>{}))
                if (!lidx.count(cn)) ++outside;
            if (outside)
                fprintf(stderr, "[route_b_verify] L%d closure has %zu node(s) outside layer list (lns=%zu)\n",
                        L, outside, lns.size());
        }
#endif
        g_layer_plan[L] = std::move(plan);
    }
}

// Consumer check (docs/ROUTE_B_LAYER_OWNERSHIP.md 3.1 Check 1): a layer's
// intermediate must not be consumed outside the layer, except the layer output
// consumed by the next layer. Reports layers with more than one forward
// cross-layer source and any backward edge. Diagnostics only (no hard fail).
void verify_layer_consumers(const ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor*, int> nlayer;
    nlayer.reserve((size_t) gf->n_nodes * 2);
    for (auto & kv : g_layer_nodes_all)
        for (auto * nd : kv.second) nlayer[nd] = kv.first;

    std::map<int, std::set<const ggml_tensor*>> fwd;   // layer -> nodes consumed by a later layer
    int backward = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        auto it = nlayer.find(nd);
        if (it == nlayer.end()) continue;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (!src) continue;
            auto sit = nlayer.find(src);
            if (sit == nlayer.end()) continue;       // external / model I/O
            if (sit->second == it->second) continue;
            if (sit->second < it->second) fwd[sit->second].insert(src);
            else {                        // consumer in an earlier layer
                ++backward;
                if (backward <= 12)
                    fprintf(stderr, "[route_b_verify] backward: consumer L%d '%s' (%s) <- src L%d '%s' (%s)\n",
                            it->second, nd->name ? nd->name : "?", ggml_op_name(nd->op),
                            sit->second, src->name ? src->name : "?", ggml_op_name(src->op));
            }
        }
    }
    for (auto & kv : fwd) {
        if (kv.second.size() > 1) {
            fprintf(stderr, "[route_b_verify] L%d has %zu nodes consumed by later layers:",
                    kv.first, kv.second.size());
            for (auto * t : kv.second) fprintf(stderr, " %s", t->name ? t->name : "?");
            fprintf(stderr, "\n");
        }
    }
    if (backward)
        fprintf(stderr, "[route_b_verify] %d backward cross-layer edge(s)\n", backward);
}

// R3: lay out the whole-layer arena as [carry][compact][closure] and point every
// captured node's buffer/data at it, so the scheduler skips them (no whole-graph
// compute buffer). carry = captured tensor consumed in another layer (fixed,
// first); compact = within-layer temporaries (per-layer bump, reused across
// layers); closure = the MoE closure block (moe_chain_fullalloc_buffer returns
// its base). docs/LAYER_EXECUTOR_DESIGN.md 4.2/4.3.
void layout_arena(ggml_backend_t our_backend, const ggml_cgraph * gf) {
    if (g_layer_nodes.empty()) return;
    ++g_build_id;

    // Token count of this build (ids ne[1]) - for the arena-size debug log.
    int64_t n_tok = -1;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        if (nd->op == GGML_OP_MUL_MAT_ID && nd->src[2]) { n_tok = nd->src[2]->ne[1]; break; }
    }

    std::unordered_map<const ggml_tensor*, int> nlayer;
    nlayer.reserve((size_t) gf->n_nodes * 2);
    for (auto & kv : g_layer_nodes_all)
        for (auto * nd : kv.second) nlayer[nd] = kv.first;

    // Per-device plan (docs/PER_DEVICE_ARENA.md 3): node -> device. The device
    // owns the node's weight (dense layers -> their --dense-placement device,
    // experts -> their pool device, via the weight buffer's device); weightless
    // nodes inherit from a producer. The CPU device collapses to the host plan
    // (route B's arena lives on our_backend). Computed here; the per-device
    // buffers/regions land with the device executor.
    ggml_backend_dev_t our_dev = ggml_backend_get_device(our_backend);
    auto dev_key = [our_dev](ggml_backend_dev_t d) -> std::string {
        if (!d) return "";
        // The host plan is our own backend's device plus the CPU device; both are
        // host memory. Only a real (non-host) device is a separate plan.
        if (d == our_dev || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) return "";
        const char * n = ggml_backend_dev_name(d);
        return (n && *n) ? std::string(n) : std::string();
    };
    std::unordered_map<const ggml_tensor*, std::string> dev_of;
    dev_of.reserve((size_t) gf->n_nodes * 2);
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        if (nlayer.find(nd) == nlayer.end()) continue;
        std::string d; bool found = false;
        for (int s = 0; s < GGML_MAX_SRC && !found; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (!src || !src->buffer || !src->buffer->buft) continue;
            ggml_backend_dev_t wd = ggml_backend_buft_get_device(src->buffer->buft);
            if (!wd) continue;
            d = dev_key(wd); found = true;
        }
        if (!found) {
            for (int s = 0; s < GGML_MAX_SRC && !found; ++s) {
                auto it = dev_of.find(nd->src[s]);
                if (it != dev_of.end()) { d = it->second; found = true; }
            }
        }
        dev_of[nd] = d;   // "" = host
    }
    // The MoE closure runs where its experts live, not where the gating came
    // from: the expert weights are not in a normal buffer, so dev_of could not
    // resolve them from a weight operand. Place every closure node on the pool's
    // device (host when there is no single pool device). Done before the carry
    // analysis so a closure consumer reports the pool device.
    const std::string closure_dev = route_b_closure_device();
    for (auto & kv : g_layer_exec)
        for (const ggml_tensor * cn : kv.second.compute)
            dev_of[const_cast<ggml_tensor*>(cn)] = closure_dev;

    // carry: captured node consumed in a different layer
    std::set<const ggml_tensor*> carry;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        auto it = nlayer.find(nd);
        if (it == nlayer.end()) continue;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (!src) continue;
            auto sit = nlayer.find(src);
            if (sit != nlayer.end() && sit->second != it->second) carry.insert(src);
        }
    }

    // Carry distance analysis: for each cross-layer tensor, the max consumer
    // layer minus its own layer. distance 1 = pipeline (reusable across layers);
    // distance > 1 = must be retained (cross-N). Also record the max consumer's
    // DEVICE: a carry tensor read by a consumer on another device must be relayed
    // (docs/PER_DEVICE_ARENA.md 3). DEBUG: report the split.
    std::unordered_map<const ggml_tensor*, int> max_consumer;
    std::unordered_map<const ggml_tensor*, std::string> consumer_dev;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * nd = gf->nodes[i];
        auto it = nlayer.find(nd);
        if (it == nlayer.end()) continue;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nd->src[s];
            if (!src) continue;
            auto sit = nlayer.find(src);
            if (sit == nlayer.end() || sit->second == it->second) continue;
            auto & m = max_consumer[src];
            if (m < it->second) {
                m = it->second;
                auto dit = dev_of.find(nd);
                consumer_dev[src] = dit != dev_of.end() ? dit->second : std::string();
            }
        }
    }
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG")) {
        size_t c1_b = 0, cN_b = 0, c1_max = 0;
        int c1_n = 0, cN_n = 0;
        for (const ggml_tensor * nd : carry) {
            if (nd->view_src) continue;
            auto cit = nlayer.find(nd);
            auto mit = max_consumer.find(nd);
            if (cit == nlayer.end() || mit == max_consumer.end()) continue;
            const size_t nb = ggml_nbytes(nd);
            if (mit->second - cit->second <= 1) { ++c1_n; c1_b += nb; if (nb > c1_max) c1_max = nb; }
            else                                  { ++cN_n; cN_b += nb; }
        }
        fprintf(stderr, "[route_b_verify] carry: cross-1 n=%d bytes=%zu (max=%zu) | cross-N n=%d bytes=%zu\n",
                c1_n, c1_b, c1_max, cN_n, cN_b);
        for (const ggml_tensor * nd : carry) {
            if (nd->view_src) continue;
            auto cit = nlayer.find(nd);
            auto mit = max_consumer.find(nd);
            if (cit == nlayer.end() || mit == max_consumer.end()) continue;
            if (mit->second - cit->second > 1)
                fprintf(stderr, "[route_b_verify]   cross-N '%s' L%d -> L%d sz=%zu\n",
                        nd->name ? nd->name : "?", cit->second, mit->second, ggml_nbytes(nd));
        }
    }

    // ---- carry split: cross-1 (double-buffered pipeline) + cross-N (retained)
    // cross-1 = layer L's output consumed by L+1. Buffers indexed by L%2 so O_L
    // and O_{L+1} coexist at L+1's tail (residual read + next output write).
    // Only nodes that OWN a fresh output buffer may be pre-allocated; a view
    // (view_src != NULL) aliases another tensor, so its data must follow it.
    std::vector<ggml_tensor*> c1, cN;
    for (const ggml_tensor * nd : carry) {
        if (nd->view_src) continue;
        auto cit = nlayer.find(nd);
        auto mit = max_consumer.find(nd);
        if (cit == nlayer.end() || mit == max_consumer.end()) continue;
        if (mit->second - cit->second <= 1) c1.push_back(const_cast<ggml_tensor*>(nd));
        else                                cN.push_back(const_cast<ggml_tensor*>(nd));
    }
#if defined(STREAM_MOE_ROUTE_B) && defined(STREAM_MOE_PREFILL_EXPORT)
    // Prefill-export observed tensors must stay live until the export reads them
    // (after the split): pull them out of carry1 (double-buffered) and the compact
    // reuse pool into the retained carryN region. Only captured layer nodes get a
    // slot; a tensor the sched owns directly is already read at the right time.
    for (const ggml_tensor * t : g_export_retained) {
        if (nlayer.find(t) == nlayer.end()) continue;
        carry.insert(t);
        c1.erase(std::remove(c1.begin(), c1.end(), t), c1.end());
        if (std::find(cN.begin(), cN.end(), t) == cN.end()) cN.push_back(const_cast<ggml_tensor*>(t));
    }
#endif
    // ---- generic cross-device transfers (docs/PER_DEVICE_ARENA.md 3/SS8.3) ----
    // Every captured edge whose producer and consumer devices differ gets a
    // consumer-side copy (shell) in the consumer device's `xfer` region, and the
    // consumer's src is rewired to it. The executor copies src -> dst once at the
    // tagged stage, so a producer written many times inside a layer copies once,
    // not per write. CPU-only has no cross-device edge -> nothing is rewired
    // (numerics unchanged). Covers cross-layer carry, `cur` (before the closure)
    // and `moe_out` (before the tail) uniformly.
    g_relays.clear();
    if (g_relay_ctx) { ggml_free(g_relay_ctx); g_relay_ctx = nullptr; }
    {
        ggml_init_params ip = { 2u * 1024u * 1024u, nullptr, /*no_alloc=*/true };
        g_relay_ctx = ggml_init(ip);
    }
    struct shell_key_t { const ggml_tensor * p; std::string dev; int stage; };
    std::vector<shell_key_t> shell_keys;
    std::vector<ggml_tensor*> shell_dsts;
    std::vector<int>          shell_layers;
    auto shell_for = [&](const ggml_tensor * p, const std::string & dev, int stage,
                         int layer) -> ggml_tensor * {
        for (size_t i = 0; i < shell_keys.size(); ++i)
            if (shell_keys[i].p == p && shell_keys[i].dev == dev && shell_keys[i].stage == stage)
                return shell_dsts[i];
        ggml_tensor * dst = g_relay_ctx
            ? ggml_new_tensor_4d(g_relay_ctx, p->type, p->ne[0], p->ne[1], p->ne[2], p->ne[3])
            : nullptr;
        if (!dst) return nullptr;
        shell_keys.push_back({ p, dev, stage });
        shell_dsts.push_back(dst);
        shell_layers.push_back(layer);
        nlayer[dst] = layer;
        dev_of[dst] = dev;
        return dst;
    };
    auto consumer_stage = [&](int L, const ggml_tensor * c) -> int {
        if (const moe_layer_exec_t * ex = moe_chain_layer_exec(L))
            for (const auto * cn : ex->compute) if (cn == c) return ROUTE_B_XFER_CLOSURE;
        if (const moe_layer_plan_t * lp = moe_chain_layer_plan(L))
            for (const auto * t : lp->tail) if (t == c) return ROUTE_B_XFER_TAIL;
        return ROUTE_B_XFER_LAYER_FRONT;
    };
    for (auto & kv : g_layer_nodes) {
        const int L = kv.first;
        for (ggml_tensor * c : kv.second) {
            if (!c) continue;
            auto cit = dev_of.find(c);
            const std::string cdev = cit != dev_of.end() ? cit->second : std::string();
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                ggml_tensor * p = c->src[s];
                if (!p || p->view_src) continue;          // views follow their root
                auto pit = dev_of.find(p);
                if (pit == dev_of.end()) continue;        // external leaf / weight
                if (pit->second == cdev) continue;        // same device: no copy
                int stage = ROUTE_B_XFER_LAYER_FRONT;
                auto pl = nlayer.find(p);
                if (pl != nlayer.end() && pl->second == L) stage = consumer_stage(L, c);
                ggml_tensor * dst = shell_for(p, cdev, stage, L);
                if (!dst) continue;
                c->src[s] = dst;
                g_relays.push_back({ p, dst, L, stage });
            }
        }
    }

    // ---- per-device plans (docs/PER_DEVICE_ARENA.md 3) --------------------
    // One region_plan_t per device, each with its own grow-only buffer. Regions
    // by liveness: carry (cross-layer pipeline; logically one, physically per
    // device, relayed at a device boundary) + scratch (within-layer temporaries:
    // dense head/tail MERGED with the MoE closure, one pool per device).
    std::map<std::string, region_plan_t> plans;
    for (auto & kv : dev_of) plans[kv.second];   // one plan per device with a node

    for (auto & pkv : plans) {
        const std::string & D = pkv.first;
        region_plan_t & plan = pkv.second;
        plan.dev = D;

        // --- carry1: cross-1 boundary tensors produced on D, parity
        // double-buffered (O_L and O_{L+1} coexist at L+1's tail).
        {
            std::map<int, size_t> layer_bytes;
            for (ggml_tensor * nd : c1)
                if (dev_of[nd] == D) layer_bytes[nlayer[nd]] += ggml_nbytes(nd);
            size_t buf = 0;
            for (auto & kv : layer_bytes) buf = std::max(buf, kv.second);
            buf = (buf + 63) & ~size_t(63);
            plan.carry1_size = 2 * buf;
            std::map<int, size_t> cur;
            for (ggml_tensor * nd : c1) {
                if (dev_of[nd] != D) continue;
                const int L = nlayer[nd];
                size_t & off = cur[L];
                off = (off + 63) & ~size_t(63);
                plan.carry1_off[nd] = (L % 2 ? buf : 0) + off;
                off += ggml_nbytes(nd);
            }
        }

        // --- carryN: retained (live across the span; relayed at boundaries)
        {
            std::vector<ggml_tensor*> cn;
            for (ggml_tensor * nd : cN) if (dev_of[nd] == D) cn.push_back(nd);
            if (!cn.empty()) {
                std::vector<int> s(cn.size()), e(cn.size());
                for (size_t i = 0; i < cn.size(); ++i) { s[i] = (int) i; e[i] = (int) cn.size(); }
                std::vector<int64_t> o;
                plan.carryN_size = pack_interval(cn, s, e, o);
                for (size_t i = 0; i < cn.size(); ++i) plan.carryN_off[cn[i]] = (size_t) o[i];
            }
        }

        // --- cross_device: a carry tensor produced on D but read by a consumer
        // on another device. The executor relays it (ggml_backend_tensor_copy,
        // M2_DEVICE_EXECUTOR SS7.8 transport) before the consumer's first read;
        // carry1 and carryN move together at every boundary (docs/PER_DEVICE_ARENA.md 3).
        for (const ggml_tensor * nd : carry) {
            if (dev_of[nd] != D) continue;
            auto cd = consumer_dev.find(nd);
            if (cd != consumer_dev.end() && cd->second != D) plan.cross_device.insert(nd);
        }

        // --- scratch: dense head/tail + MoE closure, one pool per device.
        // Time axis = global exec order (head, closure, tail); last_use = the
        // latest consumer (possibly on another device). Size = max over layers.
        for (auto & kv : g_layer_nodes) {
            const moe_layer_plan_t * lp = moe_chain_layer_plan(kv.first);
            const moe_layer_exec_t * ex = moe_chain_layer_exec(kv.first);
            std::unordered_map<const ggml_tensor*, int> etime;
            int H = 0;
            if (lp) for (ggml_tensor * h : lp->head) etime[h] = H++;
            const int CB = H;   // closure base
            if (ex) for (size_t i = 0; i < ex->compute.size(); ++i) etime[ex->compute[i]] = CB + (int) i;
            int T = CB + (ex ? (int) ex->compute.size() : 0);
            if (lp) for (ggml_tensor * t : lp->tail) { auto e = etime.find(t); if (e == etime.end()) etime[t] = T++; }
            std::unordered_map<const ggml_tensor*, int> last_etime;
            for (auto & ep : etime) {
                const ggml_tensor * node = ep.first;
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * src = node->src[s];
                    if (!src) continue;
                    // unwrap view/layout chains: a consumer may read a VIEW of a
                    // producer, and the producer must stay live until then.
                    while (src && is_view_op(src)) src = src->src[0];
                    if (!src) continue;
                    auto it = last_etime.find(src);
                    if (it == last_etime.end()) last_etime[src] = ep.second;
                    else if (it->second < ep.second) it->second = ep.second;
                }
            }
            std::vector<ggml_tensor*> cns;
            std::vector<int> cstart;
            for (ggml_tensor * nd : kv.second) {
                if (nd->view_src || carry.count(nd)) continue;
                if (dev_of[nd] != D) continue;
                auto et = etime.find(nd);
                if (et == etime.end()) continue;
                cns.push_back(nd);
                cstart.push_back(et->second);
            }
            if (cns.empty()) continue;
            std::vector<int> cend(cns.size());
            for (size_t k = 0; k < cns.size(); ++k) {
                auto it = last_etime.find(cns[k]);
                cend[k] = it != last_etime.end() ? std::max(it->second, cstart[k]) : T;
            }
            std::vector<int64_t> o;
            size_t lb;
            // Scratch packing is the production layout now (the earlier gemma
            // "divergence" was the prefill-export capture reading a reused slot,
            // fixed). STREAM_MOE_TMP_COMPACT_PACK=0 forces the byte-sum layout
            // for A/B debugging only.
            const char * cp = std::getenv("STREAM_MOE_TMP_COMPACT_PACK");
            if (!(cp && cp[0] == '0')) {
                lb = pack_interval(cns, cstart, cend, o);
            } else {
                size_t off = 0;
                o.resize(cns.size());
                for (size_t k = 0; k < cns.size(); ++k) { off = (off + 63) & ~size_t(63); o[k] = (int64_t) off; off += ggml_nbytes(cns[k]); }
                lb = off;
            }
            plan.scratch_size = std::max(plan.scratch_size, lb);
            for (size_t k = 0; k < cns.size(); ++k) plan.scratch_off[cns[k]] = (size_t) o[k];
            // The closure's own layout now indexes the merged scratch, so the
            // executor (moe_chain_fullalloc_buffer base + out_off) lands in the
            // same pool as the dense head/tail.
            if (ex && !ex->compute.empty()) {
                moe_layer_exec_t * mut = const_cast<moe_layer_exec_t*>(ex);
                for (size_t i = 0; i < ex->compute.size(); ++i) {
                    auto oit = plan.scratch_off.find(ex->compute[i]);
                    if (oit != plan.scratch_off.end()) mut->out_off[i] = (int64_t) oit->second;
                }
                mut->result_bytes = lb;
                mut->layout_ok = true;
            }
#ifdef STREAM_MOE_TEMP
            if (std::getenv("STREAM_MOE_TMP_COMPACT_DEBUG") && kv.first == 0) {
                for (size_t k = 0; k < cns.size(); ++k)
                    fprintf(stderr, "[cpack] b%d L0 %-24s start=%d end=%d off=%lld sz=%zu ptr=%p\n",
                            g_build_id, cns[k]->name ? cns[k]->name : "?", cstart[k], cend[k], (long long) o[k], ggml_nbytes(cns[k]), (const void *) cns[k]);
            }
#endif
        }

        // --- xfer: consumer-side copies of cross-device producers (one shell per
        // producer/device/stage). A shell is live only during the layer whose
        // stage consumes it, so the region is reused across layers -> size = max
        // over layers of that layer's shells on D.
        {
            std::map<int, size_t> layer_bytes;
            for (size_t i = 0; i < shell_keys.size(); ++i) {
                if (shell_keys[i].dev != D) continue;
                layer_bytes[shell_layers[i]] += ggml_nbytes(shell_dsts[i]);
            }
            size_t m = 0;
            for (auto & kv : layer_bytes) m = std::max(m, kv.second);
            m = (m + 63) & ~size_t(63);
            plan.xfer_size = m;
            std::map<int, size_t> cur;
            for (size_t i = 0; i < shell_keys.size(); ++i) {
                if (shell_keys[i].dev != D) continue;
                size_t & off = cur[shell_layers[i]];
                off = (off + 63) & ~size_t(63);
                plan.xfer_off[shell_dsts[i]] = off;
                off += ggml_nbytes(shell_dsts[i]);
            }
        }

        // --- grow-only device buffer (host buft when D == "")
        plan.off_carryN = plan.carry1_size;
        plan.off_scratch = plan.carry1_size + plan.carryN_size;
        plan.off_xfer = plan.carry1_size + plan.carryN_size + plan.scratch_size;
        plan.need = plan.off_xfer + plan.xfer_size + 4096;
        if (plan.need > plan.cap) {
            if (plan.buf) ggml_backend_buffer_free(plan.buf);
            ggml_backend_buffer_type_t buft = nullptr;
            if (D.empty()) {
                buft = ggml_backend_get_default_buffer_type(our_backend);
            } else if (ggml_backend_dev_t d = ggml_backend_dev_by_name(D.c_str())) {
                buft = ggml_backend_dev_buffer_type(d);
            }
            plan.buf = buft ? ggml_backend_buft_alloc_buffer(buft, plan.need) : nullptr;
            plan.cap = plan.buf ? plan.need : 0;
        }
    }

    // Assign every captured node to its device's buffer/region.
    for (auto & kv : g_layer_nodes) {
        for (auto * nd : kv.second) {
            if (nd->view_src) continue;   // view/in-place: data follows view_src
            auto pit = plans.find(dev_of[nd]);
            if (pit == plans.end() || !pit->second.buf) continue;
            region_plan_t & plan = pit->second;
            char * base = static_cast<char*>(ggml_backend_buffer_get_base(plan.buf));
            const size_t off_carryN  = plan.carry1_size;
            const size_t off_scratch = plan.carry1_size + plan.carryN_size;
            nd->buffer = plan.buf;
            auto c1i = plan.carry1_off.find(nd);
            if (c1i != plan.carry1_off.end()) { nd->data = base + c1i->second; continue; }
            auto cNi = plan.carryN_off.find(nd);
            if (cNi != plan.carryN_off.end()) { nd->data = base + off_carryN + cNi->second; continue; }
            auto mit = plan.scratch_off.find(nd);
            if (mit != plan.scratch_off.end()) { nd->data = base + off_scratch + mit->second; continue; }
        }
    }
    // Transfer copies are shells (not captured nodes): bind them to the
    // consumer device's xfer slot so the executor can ggml_backend_tensor_copy
    // into them.
    for (const route_b_relay_t & r : g_relays) {
        auto pit = plans.find(dev_of[r.dst]);
        if (pit == plans.end() || !pit->second.buf) continue;
        region_plan_t & plan = pit->second;
        char * rbase = static_cast<char*>(ggml_backend_buffer_get_base(plan.buf));
        r.dst->buffer = plan.buf;
        auto xi = plan.xfer_off.find(r.dst);
        if (xi != plan.xfer_off.end()) r.dst->data = rbase + plan.off_xfer + xi->second;
    }
    g_plans = plans;   // expose to moe_chain_fullalloc_buffer / route_b_in_arena
    g_node_dev = dev_of;   // expose node -> device to the executor (phase 3)
    // Host plan drives the debug/report paths and the CPU executor.
    region_plan_t & hostp = g_plans[""];
    g_arena = hostp.buf;
    g_arena_cap = hostp.cap;
    g_arena_closure_off  = hostp.carry1_size + hostp.carryN_size;
    g_arena_closure_size = hostp.scratch_size;
    if (!g_arena) return;
    char * base = static_cast<char*>(ggml_backend_buffer_get_base(g_arena));
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG")) {
        for (auto & kv : g_layer_nodes)
            for (auto * nd : kv.second) {
                if (nd->view_src) continue;
                if (!hostp.carry1_off.count(nd) && !hostp.carryN_off.count(nd) && !hostp.scratch_off.count(nd))
                    fprintf(stderr, "[route_b_cap] UNASSIGNED '%s' op=%s dev=%s\n",
                            nd->name ? nd->name : "?", ggml_op_name(nd->op), dev_of[nd].c_str());
                if (nd->buffer != g_arena || !nd->data) continue;
                const char * p = (const char*) nd->data;
                if (p < base || p + ggml_nbytes(nd) > base + hostp.cap)
                    fprintf(stderr, "[route_b_cap] OOB '%s' off=%zu sz=%zu need=%zu\n",
                            nd->name ? nd->name : "?", (size_t)(p - base), ggml_nbytes(nd), hostp.cap);
            }
    }
#endif
    // second pass: nodes whose output aliases another tensor (view/in-place)
    // follow the view_src chain to the root's data + accumulated offsets.
    // NOTE: resolve via view_src, not src[0] (SET_ROWS stores its sources in a
    // legacy order, src[0] is not the aliased tensor).
    for (auto & kv : g_layer_nodes) {
        for (auto * nd : kv.second) {
            if (!nd->view_src) continue;
            const ggml_tensor * t = nd;
            int64_t off = 0;
            while (t && t->view_src) {
                if (t->op == GGML_OP_VIEW) off += t->view_offs;
                t = t->view_src;
            }
            if (t && t->data) nd->data = static_cast<char*>(t->data) + off;
        }
    }
    if (std::getenv("STREAM_MOE_TMP_DUMP_PLAN")) {
        fprintf(stderr, "==== arena plan (host): carry1=%zu carryN=%zu scratch=%zu total=%zu ====\n",
                hostp.carry1_size, hostp.carryN_size, hostp.scratch_size, g_arena_cap);
        fprintf(stderr, "---- cross-1 (carry1) %zu nodes ----\n", hostp.carry1_off.size());
        for (auto & kv : hostp.carry1_off)
            fprintf(stderr, "  %-30s L%-3d off=%zu sz=%zu\n", kv.first->name ? kv.first->name : "(anon)",
                    route_b_official_layer(kv.first), kv.second, ggml_nbytes(kv.first));
        fprintf(stderr, "---- cross-N (carryN) %zu nodes ----\n", hostp.carryN_off.size());
        for (auto & kv : hostp.carryN_off)
            fprintf(stderr, "  %-30s L%-3d off=%zu sz=%zu\n", kv.first->name ? kv.first->name : "(anon)",
                    route_b_official_layer(kv.first), kv.second, ggml_nbytes(kv.first));
        fprintf(stderr, "---- compute graph (%d nodes) ----\n", gf->n_nodes);
        for (int i = 0; i < gf->n_nodes; ++i) {
            const ggml_tensor * nd = gf->nodes[i];
            const char * region = "-"; size_t off = 0;
            auto a = hostp.carry1_off.find(nd);
            if (a != hostp.carry1_off.end()) { region = "carry1"; off = a->second; }
            else {
                auto b = hostp.carryN_off.find(nd);
                if (b != hostp.carryN_off.end()) { region = "carryN"; off = hostp.carry1_size + b->second; }
                else {
                    auto m = hostp.scratch_off.find(nd);
                    if (m != hostp.scratch_off.end()) { region = "scratch"; off = hostp.carry1_size + hostp.carryN_size + m->second; }
                }
            }
            fprintf(stderr, "  [%4d] %-30s %-14s L%-3d %-8s off=%zu\n", i,
                    nd->name ? nd->name : "(anon)", ggml_op_name(nd->op),
                    route_b_official_layer(nd), region, off);
        }
        fprintf(stderr, "---- MoE plan ----\n");
        for (auto & kv : g_layer_exec) {
            const moe_layer_exec_t & ex = kv.second;
            fprintf(stderr, "  L%d: %zu nodes result_bytes=%zu layout_ok=%d\n",
                    kv.first, ex.compute.size(), ex.result_bytes, (int) ex.layout_ok);
            for (size_t i = 0; i < ex.compute.size(); ++i)
                fprintf(stderr, "    c[%zu] %-26s %-12s out_off=%lld\n", i,
                        ex.compute[i]->name ? ex.compute[i]->name : "(anon)",
                        ggml_op_name(ex.compute[i]->op),
                        (long long) (i < ex.out_off.size() ? ex.out_off[i] : -1));
        }
        fflush(stderr);
    }
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG"))
        fprintf(stderr, "[route_b_verify] arena: n_tok=%lld carry1=%zu carryN=%zu scratch=%zu need=%zu cap=%zu bytes (carry=%zu nodes)\n",
                (long long) n_tok, hostp.carry1_size, hostp.carryN_size, hostp.scratch_size,
                hostp.need, g_arena_cap, carry.size());

    // Per-device plan: one line per device with its own buffer + region sizes.
    if (std::getenv("STREAM_MOE_TMP_DUMP_PLAN")) {
        fprintf(stderr, "==== per-device plan (%zu device(s)) ====\n", g_plans.size());
        for (auto & kv : g_plans) {
            const region_plan_t & p = kv.second;
            fprintf(stderr, "  dev=%-12s nodes=%-5zu carry1=%-9zu carryN=%-9zu scratch=%-9zu cross_dev=%-4zu cap=%zu\n",
                    kv.first.empty() ? "CPU(host)" : kv.first.c_str(),
                    p.carry1_off.size() + p.carryN_off.size() + p.scratch_off.size(),
                    p.carry1_size, p.carryN_size, p.scratch_size, p.cross_device.size(), p.cap);
        }
    }
}
}

void moe_chain_set_full_alloc(size_t layer_sum_bytes) {
    if (layer_sum_bytes > g_fullalloc_cap) {
        aligned_free_ptr(g_fullalloc_buf);
        g_fullalloc_buf = aligned_alloc_ptr(layer_sum_bytes, 64);
        g_fullalloc_cap = g_fullalloc_buf ? layer_sum_bytes : 0;
    }
}

void * moe_chain_fullalloc_buffer(size_t need_bytes) {
    // Whole-layer: the closure block is the scratch region of the device that
    // owns the experts (layout_arena merged the dense head/tail and the closure
    // into that one pool), so the closure twins are pre-allocated there too.
    {
        auto pit = g_plans.find(route_b_closure_device());
        if (pit != g_plans.end() && pit->second.buf) {
            const region_plan_t & p = pit->second;
            if (need_bytes > p.scratch_size) {
                fprintf(stderr, "[route_b_cap] ERROR: closure block overflow need=%zu cap=%zu\n",
                        need_bytes, p.scratch_size);
                return nullptr;
            }
            return static_cast<char*>(ggml_backend_buffer_get_base(p.buf)) + p.carry1_size + p.carryN_size;
        }
    }
    if (!g_fullalloc_buf) return nullptr;   // set_full_alloc(layer_sum) must run first
    if (need_bytes > g_fullalloc_cap) {
        fprintf(stderr, "[route_b_cap] ERROR: full-alloc overflow need=%zu cap=%zu\n", need_bytes, g_fullalloc_cap);
        return nullptr;
    }
    return g_fullalloc_buf;
}

// Official layer attribution, fed by llama's graph-build callback (see the
// phase-1 anchor in llama-context.cpp::graph_get_cb).
void route_b_on_node(const ggml_tensor * node, int il) {
    if (node) g_official_layer[node] = il;
}

int route_b_official_layer(const ggml_tensor * node) {
    if (!node) return -1;
    auto it = g_official_layer.find(node);
    return it == g_official_layer.end() ? -1 : it->second;
}

// Expert pool devices, in creation order (empty = host/RAM). Recorded by
// route_b_setup. The MoE closure runs where the experts live, so these place it
// (docs/PER_DEVICE_ARENA.md 3): a single non-host device -> the closure goes
// there; none or several -> the host plan.
static std::vector<std::string> g_expert_pool_devs;

void route_b_add_expert_pool_device(const char * dev) {
    g_expert_pool_devs.push_back(dev ? dev : "");
}

const char * route_b_closure_device() {
    const std::string * one = nullptr;
    for (const std::string & d : g_expert_pool_devs) {
        if (d.empty()) continue;
        if (one && *one != d) return "";   // pools span devices -> host plan
        one = &d;
    }
    return one ? one->c_str() : "";
}

void route_b_add_device_backend(const char * dev, ggml_backend_t be) {
    if (!dev || !*dev || !be) return;
    g_dev_backends[dev] = be;
}

ggml_backend_t route_b_device_backend(const char * dev) {
    if (!dev || !*dev) return nullptr;
    auto it = g_dev_backends.find(dev);
    return it == g_dev_backends.end() ? nullptr : it->second;
}

const char * route_b_node_device(const ggml_tensor * node) {
    if (!node) return "";
    auto it = g_node_dev.find(node);
    return it == g_node_dev.end() ? "" : it->second.c_str();
}

const std::vector<route_b_relay_t> & route_b_relays() { return g_relays; }

bool route_b_in_arena(const void * p) {
    if (!p) return false;
    const char * q = static_cast<const char*>(p);
    for (auto & kv : g_plans) {
        const region_plan_t & plan = kv.second;
        if (!plan.buf) continue;
        const char * base = static_cast<const char*>(ggml_backend_buffer_get_base(plan.buf));
        if (q >= base && q < base + plan.cap) return true;
    }
    return false;
}

bool route_b_whole_layer_active() {
    // Whole-layer ownership is the production path: route B owns every compute
    // node, the scheduler sees one split, and all activations live in our arena.
    return true;
}

int route_b_build_id() { return g_build_id; }

#if defined(STREAM_MOE_ROUTE_B) && defined(STREAM_MOE_PREFILL_EXPORT)
void route_b_set_export_retained(const std::vector<const ggml_tensor*> & ts) {
    g_export_retained.clear();
    for (const ggml_tensor * t : ts) {
        // A view aliases its root's slot; retain the root (that is what would be
        // reused). layout_arena's view pass then gives the view the right data.
        while (t && t->view_src) t = t->view_src;
        if (t) g_export_retained.push_back(t);
    }
}
#endif

void route_b_begin_ubatch() {
#ifdef STREAM_MOE_TEMP
    ++g_dump_ubatch;
#endif
}

void route_b_dump_node_bin(int layer, const char * name, const char * op,
                           int type, int64_t ne0, int64_t ne1, const void * data, size_t nb) {
    const char * dir = std::getenv("STREAM_MOE_TMP_BIN_DIR");
    if (!dir || !*dir || !name || !data || nb == 0) return;
    std::string nm = name;
    for (char & c : nm) if (c == '/' || c == '\\' || c == ':' || c == ' ' || c == '*') c = '_';
    char sub[64];
    std::snprintf(sub, sizeof(sub), "ub%d", g_dump_ubatch);
    std::string udir = std::string(dir) + "/" + sub;
#ifdef _WIN32
    _mkdir(udir.c_str());
#else
    mkdir(udir.c_str(), 0755);
#endif
    std::string base = udir + "/" + nm;
    FILE * f = std::fopen((base + ".bin").c_str(), "wb");
    if (f) { std::fwrite(data, 1, nb, f); std::fclose(f); }
    FILE * m = std::fopen((base + ".meta").c_str(), "w");
    if (m) {
        std::fprintf(m, "layer=%d name=%s op=%s type=%d ne0=%lld ne1=%lld nb=%zu\n",
                     layer, name, op ? op : "?", type, (long long) ne0, (long long) ne1, nb);
        std::fclose(m);
    }
}

namespace {
// Forward declarations (definitions live further down the file).
bool is_routed_mm(const ggml_tensor * n);
int  mm_layer(const ggml_tensor * n);
bool is_view_op(const ggml_tensor * n);
bool is_alias_op(const ggml_tensor * n);
bool is_fused_op(enum ggml_op op);
void collect_chain(const ggml_cgraph * gf, std::vector<char>& chain,
                   std::vector<int>& layer, int& n_anchors);
void collect_layer_nodes(const ggml_cgraph * gf,
                         std::map<int, std::vector<ggml_tensor*>> & out);
} // namespace

// Best-fit-decreasing interval packing. nodes[i]'s output is live over the
// integer interval [start[i], end[i]] (end = last reader, or the list size when
// never read). Blocks are placed by size descending, each at the lowest offset
// whose bytes do not collide with an already-placed block whose live interval
// overlaps. Fills out_off[i] and returns the total block size. Used by the MoE
// closure result layout, the compact region and the carry region.
static size_t pack_interval(const std::vector<ggml_tensor*> & nodes,
                            const std::vector<int> & start,
                            const std::vector<int> & end,
                            std::vector<int64_t> & out_off) {
    const size_t C = nodes.size();
    out_off.assign(C, -1);
    if (C == 0) return 0;
    struct blk_t { int64_t off; size_t size; int start, end, ni; };
    std::vector<int> order(C);
    for (size_t i = 0; i < C; ++i) order[i] = (int) i;
    std::sort(order.begin(), order.end(), [&](int x, int y) {
        const size_t sx = nodes[x] ? ggml_nbytes(nodes[x]) : 0;
        const size_t sy = nodes[y] ? ggml_nbytes(nodes[y]) : 0;
        if (sx != sy) return sx > sy;
        return x < y;
    });
    std::vector<blk_t> placed;
    int64_t arena = 0;
    for (const int ni : order) {
        const size_t nb = nodes[ni] ? ggml_nbytes(nodes[ni]) : 0;
        const int s = start[ni], e = end[ni];
        int64_t off = 0;
        for (;;) {
            int64_t furthest = -1;
            for (const auto & p : placed) {
                if (e < p.start || p.end < s) continue;
                if (off < (int64_t)(p.off + (int64_t) p.size) && (int64_t) p.off < off + (int64_t) nb)
                    furthest = std::max(furthest, p.off + (int64_t) p.size);
            }
            if (furthest < 0) break;
            off = furthest;
        }
        placed.push_back({ off, nb, s, e, ni });
        arena = std::max(arena, off + (int64_t) nb);
    }
    for (const auto & p : placed) out_off[p.ni] = p.off;
    return (size_t) arena;
}

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

    // Whole-layer ownership (docs/ROUTE_B_LAYER_OWNERSHIP.md): capture every
    // compute node of each layer (dense + MoE + anonymous) and assign it to our
    // backend, so the whole graph is ONE split and all activations stay under our
    // control (the executor runs the dense head/tail around the MoE burst).
    // Only layers whose dense weights are host-resident are taken over: with a
    // device compute buft the executor would need the device path (later
    // milestone); device-dense layers keep the MoE-only split.
    collect_layer_nodes(gf, g_layer_nodes_all);
    verify_layer_consumers(gf);
    {
        std::map<int, std::vector<ggml_tensor*>> & all = g_layer_nodes_all;
        g_layer_nodes.clear();
        for (auto & kv : all) {
            bool dense_host = true;
            for (auto * nd : kv.second) {
                // Check the node's own output buffer too, not just its sources.
                ggml_backend_buffer_t nbuf = nd->view_src ? nd->view_src->buffer : nd->buffer;
                if (nbuf && !ggml_backend_buft_is_host(ggml_backend_buffer_get_type(nbuf)))
                    dense_host = false;
                for (int s = 0; s < GGML_MAX_SRC && dense_host; ++s) {
                    const ggml_tensor * src = nd->src[s];
                    if (!src) continue;
                    ggml_backend_buffer_t buf = src->view_src ? src->view_src->buffer : src->buffer;
                    if (buf && !ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buf)))
                        dense_host = false;
                }
                if (!dense_host) break;
            }
            if (!dense_host) continue;
            g_layer_nodes[kv.first] = kv.second;
            for (auto * nd : kv.second) {
                if (is_alias_op(nd)) continue;
                // Fused ops (FLASH_ATTN_EXT / LIGHTNING_INDEXER / DSV4_HC_*) are
                // claimed too: together with llama_model::dev_layer reporting the
                // StreamMoE device, resolve() sees device_fused == dev_layer and
                // keeps the fused path (moe_dev_supports_op also accepts them).
                ggml_backend_sched_set_tensor_backend(sched, nd, our_backend);
            }
        }
    }
    // Build the per-layer plans (head/tail/moe) from the capture.
    build_layer_plans();
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG"))
        fprintf(stderr, "[route_b_verify] official layer recorded for %zu/%d nodes\n",
                g_official_layer.size(), gf->n_nodes);
#endif
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
        // Result-buffer layout by best-fit decreasing interval packing
        // (pack_interval; docs layout_sim sim.js): reaches the peak
        // simultaneous-live lower bound on gemma/deepseek.
        std::vector<int> pstart(C), pend(C);
        for (size_t i = 0; i < C; ++i) {
            pstart[i] = (int) i;
            pend[i] = last_use[i] < 0 ? (int) C : last_use[i];
        }
        ex.result_bytes = pack_interval(comp, pstart, pend, ex.out_off);
        // ---- layout self-check (always): results whose live intervals overlap
        // in exec time must not share byte ranges. Violation -> per-node bump.
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
        // C4 replication (docs/STREAMMOE_GGUF_FORMAT.md SS3.1): a closure-used
        // leaf that is not per-expert and is small (per-expert scale table) is
        // copied once into every device pool. Idempotent per tensor.
        for (const auto & el : ex.external_leaves) {
            const ggml_tensor * t = el.tensor;
            if (!t || t->op != GGML_OP_NONE || !t->name) continue;
            if (strstr(t->name, "_exps") == nullptr || strstr(t->name, ".weight") != nullptr) continue;
            if (ggml_nbytes(t) > kResidentLeafMaxBytes) continue;
            stream_moe_backend_replicate_leaf(t);
        }
    }

    // [carry][compact][closure] arena: assign every captured node its buffer/
    // data. Runs after the closure result layout so the packed closure size is
    // used. docs/PER_DEVICE_ARENA.md.
    layout_arena(our_backend, gf);
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

    // Consumed; clear so the next graph build starts from a clean side channel.
    g_official_layer.clear();
    return true;
}

int32_t moe_chain_layer_of_node(const ggml_tensor * node) {
    if (!node) return -1;
    for (const auto & kv : g_layer_exec) {
        for (const auto * cn : kv.second.compute) {
            if (cn == node) return kv.first;
        }
    }
    for (const auto & kv : g_layer_nodes) {
        for (const auto * cn : kv.second) {
            if (cn == node) return kv.first;
        }
    }
    return -1;
}

const moe_layer_exec_t * moe_chain_layer_exec(int32_t layer) {
    auto it = g_layer_exec.find(layer);
    return it == g_layer_exec.end() ? nullptr : &it->second;
}

const std::vector<ggml_tensor*> * moe_chain_layer_nodes(int32_t layer) {
    auto it = g_layer_nodes.find(layer);
    return it == g_layer_nodes.end() ? nullptr : &it->second;
}

const std::vector<ggml_tensor*> * moe_chain_layer_nodes_all(int32_t layer) {
    auto it = g_layer_nodes_all.find(layer);
    return it == g_layer_nodes_all.end() ? nullptr : &it->second;
}

const moe_layer_plan_t * moe_chain_layer_plan(int32_t layer) {
    auto it = g_layer_plan.find(layer);
    return it == g_layer_plan.end() ? nullptr : &it->second;
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

// Pure aliases only (see minigraph_exec.cpp): CONT is a real copy, not a view,
// so whole-layer ownership must assign it like any other compute node.
[[maybe_unused]] bool is_alias_op(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
           n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_PERMUTE;
}

// Fused ops that llama_context::resolve probes against the layer's device: if
// such a node is force-claimed to a backend different from the layer's device,
// resolve disables the fused path and the model silently decomposes to primitive
// ops (e.g. deepseek4 DSV4_HC_*). Whole-layer must leave these to llama.
bool is_fused_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_DSV4_HC_COMB:
            return true;
        default:
            return false;
    }
}

// Layer suffix of a llama node name ("ffn_norm-3", "Qcur-3 (reshaped)" -> 3).
// -1 when the name has no trailing "-<digits>" (anonymous node_*, leaves).
static int name_layer_suffix(const char * name) {
    if (!name || !name[0]) return -1;
    std::string s(name);
    const size_t sp = s.rfind(" (");
    if (sp != std::string::npos && s.back() == ')') s.resize(sp);
    const size_t dash = s.rfind('-');
    if (dash == std::string::npos || dash + 1 >= s.size()) return -1;
    for (size_t i = dash + 1; i < s.size(); ++i)
        if (s[i] < '0' || s[i] > '9') return -1;
    return atoi(s.c_str() + dash + 1);
}

// Attribute every compute node of `gf` to its layer (ROUTE_B_LAYER_OWNERSHIP.md
// L1). Named nodes carry llama's "-<il>" suffix; anonymous nodes (node_*, the
// per-topk convergence adds) inherit from an attributed producer. Unattributed
// nodes are model inputs / weights / the output head (external to any layer).
// Graph order is preserved inside each layer's list.
[[maybe_unused]] static void collect_layer_nodes(const ggml_cgraph * gf,
                                std::map<int, std::vector<ggml_tensor*>> & out) {
    out.clear();
    const int N = gf->n_nodes;
    std::vector<int> lay(N, -1);
    std::unordered_map<const ggml_tensor*, int> idx;
    idx.reserve((size_t) N * 2);
    for (int i = 0; i < N; ++i) {
        ggml_tensor * nd = gf->nodes[i];
        const int off = route_b_official_layer(nd);
        const int nms = name_layer_suffix(nd->name);
        if (off >= 0 && nms >= 0 && off != nms) {
            fprintf(stderr, "[route_b_verify] layer mismatch node='%s': official=%d name=%d\n",
                    nd->name ? nd->name : "?", off, nms);
        }
        lay[i] = off >= 0 ? off : nms;   // official primary, name suffix fallback
        idx[nd] = i;
    }
    // Whole-layer ownership: every COMPUTE node must belong to a layer, so the
    // scheduler leaves us the whole graph (one split) and no node is left in the
    // scheduler's buffer pool (that gap caused a mask FILL / k_idxs address
    // collision). docs/LAYER_EXECUTOR_DESIGN.md 4.1.
    std::vector<char> named(N, 0);
    for (int i = 0; i < N; ++i) if (lay[i] >= 0) named[i] = 1;
    // 1. Producer propagation: anonymous nodes inherit their producers' layer.
    //    Use the MAX producer layer: a node is built during the iteration of its
    //    LATEST producer, so it reads the current layer's nodes plus the previous
    //    layer's output (a carry). The old "first producer" rule mis-assigned
    //    such nodes to the previous layer (e.g. deepseek4 hc nodes).
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < N; ++i) {
            if (named[i]) continue;
            int best = lay[i];
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                auto it = idx.find(gf->nodes[i]->src[s]);
                if (it != idx.end() && lay[it->second] >= 0) best = std::max(best, lay[it->second]);
            }
            if (best != lay[i]) { lay[i] = best; changed = true; }
        }
    }
    // 2. Consumer propagation: an unowned compute node inherits its consumers'
    //    layer. Captures in-layer anonymous compute whose producer is a leaf or
    //    graph input (e.g. the -INF attention-mask FILL), which producer
    //    propagation cannot attribute.
    {
        std::unordered_map<const ggml_tensor*, std::vector<int>> consumers;
        consumers.reserve((size_t) N * 2);
        for (int i = 0; i < N; ++i)
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = gf->nodes[i]->src[s];
                if (src) consumers[src].push_back(i);
            }
        bool ch = true;
        while (ch) {
            ch = false;
            for (int i = 0; i < N; ++i) {
                if (lay[i] >= 0 || gf->nodes[i]->op == GGML_OP_NONE) continue;
                int best = -1;
                auto it = consumers.find(gf->nodes[i]);
                if (it != consumers.end())
                    for (int c : it->second) if (lay[c] >= 0) best = std::max(best, lay[c]);
                if (best >= 0) { lay[i] = best; ch = true; }
            }
        }
    }
    // 3. Fallback: any still-unowned compute node takes the nearest owned layer
    //    (prefer the preceding one, else the following one). No compute node is
    //    left unowned.
    {
        int cur = -1;
        for (int i = 0; i < N; ++i) {
            if (lay[i] >= 0) cur = lay[i];
            else if (gf->nodes[i]->op != GGML_OP_NONE && cur >= 0) lay[i] = cur;
        }
        cur = -1;
        for (int i = N - 1; i >= 0; --i) {
            if (lay[i] >= 0) cur = lay[i];
            else if (gf->nodes[i]->op != GGML_OP_NONE && cur >= 0) lay[i] = cur;
        }
    }
    // Out-ids narrowing (docs/LAYER_EXECUTOR_DESIGN.md 4.7): the last layer's
    // `get_rows(x, inp_out_ids)` reads x (often the previous layer's output) and
    // is consumed by the last layer. Producer propagation attributes it to the
    // producer's layer; re-attribute to the consumer's layer.
    {
        std::unordered_map<const ggml_tensor*, int> consumer_layer;
        consumer_layer.reserve((size_t) N);
        for (int i = 0; i < N; ++i) {
            if (lay[i] < 0) continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = gf->nodes[i]->src[s];
                if (!src) continue;
                if (consumer_layer.find(src) == consumer_layer.end()) consumer_layer[src] = lay[i];
            }
        }
        for (int i = 0; i < N; ++i) {
            ggml_tensor * nd = gf->nodes[i];
            if (nd->op != GGML_OP_GET_ROWS || !nd->src[0] || !nd->src[1]) continue;
            if (nd->src[0]->op == GGML_OP_NONE) continue;                  // weight lookup (token embd), not a narrowing
            if (!(nd->src[1]->flags & GGML_TENSOR_FLAG_INPUT)) continue;   // out-ids narrowing
            auto it = consumer_layer.find(nd);
            if (it != consumer_layer.end() && lay[i] != it->second) {
                if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG"))
                    fprintf(stderr, "[route_b_verify] out-ids narrowing '%s' L%d -> consumer L%d\n",
                            nd->name ? nd->name : "?", lay[i], it->second);
                lay[i] = it->second;
            }
        }
    }
    for (int i = 0; i < N; ++i)
        if (lay[i] >= 0) out[lay[i]].push_back(gf->nodes[i]);
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

#ifdef STREAM_MOE_TEMP
    // Whole-graph structural dump (STREAM_MOE_TMP_GRAPH_DUMP=1). One line per
    // compute node and per leaf (weight/input), with the assigned buft when the
    // tensor already has a buffer (weights do; compute nodes get theirs later at
    // sched split). chain=L when the node is in the privatised MoE closure of
    // layer L, else -1. Used to map C1/C2/expert regions per model.
    if (getenv("STREAM_MOE_TMP_GRAPH_DUMP")) {
        auto bname = [](const ggml_tensor * t) -> const char * {
            return t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "-";
        };
        fprintf(stderr, "[gdump] graph n_nodes=%d n_leafs=%d\n", gf->n_nodes, gf->n_leafs);
        for (int i = 0; i < gf->n_nodes; ++i) {
            const ggml_tensor * nd = gf->nodes[i];
            fprintf(stderr, "[gdump] N %4d %-16s %-34s ne=[%lld,%lld,%lld,%lld] bytes=%zu chain=%d buft=%s\n",
                    i, ggml_op_name(nd->op), nd->name ? nd->name : "?",
                    (long long) nd->ne[0], (long long) nd->ne[1], (long long) nd->ne[2], (long long) nd->ne[3],
                    ggml_nbytes(nd), chain[i] ? layer[i] : -1, bname(nd));
        }
        for (int i = 0; i < gf->n_leafs; ++i) {
            const ggml_tensor * t = gf->leafs[i];
            fprintf(stderr, "[gdump] L %4d %-16s %-44s ne=[%lld,%lld,%lld,%lld] bytes=%zu buft=%s\n",
                    i, ggml_op_name(t->op), t->name ? t->name : "?",
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    ggml_nbytes(t), bname(t));
        }
        fflush(stderr);
    }
#endif

#ifdef STREAM_MOE_TEMP
    // Whole-layer capture dump (ROUTE_B_LAYER_OWNERSHIP.md L1,
    // STREAM_MOE_TMP_LAYER_DUMP=1): per-layer node counts and every cross-layer
    // consumer edge. A layer's output (e.g. l_out-L) is expected to have exactly
    // one cross-layer edge L -> L+1; anything else is a partition violation.
    if (getenv("STREAM_MOE_TMP_LAYER_DUMP")) {
        std::map<int, std::vector<ggml_tensor*>> layers;
        collect_layer_nodes(gf, layers);
        std::unordered_map<const ggml_tensor*, int> nl;
        for (auto & kv : layers) for (auto * nd : kv.second) nl[nd] = kv.first;
        int n_unattr = 0;
        for (int i = 0; i < N; ++i) if (!nl.count(gf->nodes[i])) ++n_unattr;
        fprintf(stderr, "[layerdump] graph n_nodes=%d layers=%zu unattributed=%d\n",
                N, layers.size(), n_unattr);
        for (auto & kv : layers) {
            const int L = kv.first;
            size_t bytes = 0;
            for (auto * nd : kv.second) bytes += ggml_nbytes(nd);
            std::map<std::pair<int, std::string>, int> edges;
            for (auto * nd : kv.second) {
                for (int j = 0; j < N; ++j) {
                    const ggml_tensor * cj = gf->nodes[j];
                    if (cj->op == GGML_OP_NONE) continue;
                    bool uses = false;
                    for (int s = 0; s < GGML_MAX_SRC; ++s) if (cj->src[s] == nd) { uses = true; break; }
                    if (!uses) continue;
                    auto it = nl.find(cj);
                    const int cl = (it == nl.end()) ? -2 : it->second;
                    if (cl == L) continue;
                    edges[{cl, std::string(nd->name ? nd->name : "?") + " -> " +
                                (cj->name ? cj->name : "?")}]++;
                }
            }
            fprintf(stderr, "[layerdump] L%d: nodes=%zu bytes=%zu cross_edges=%zu\n",
                    L, kv.second.size(), bytes, edges.size());
            for (auto & e : edges)
                fprintf(stderr, "[layerdump]   L%d -> %d : %s (x%d)\n",
                        L, e.first.first, e.first.second.c_str(), e.second);
        }
        fflush(stderr);
    }
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

bool route_b_is_fused_op(enum ggml_op op) { return is_fused_op(op); }

} // namespace stream_moe
