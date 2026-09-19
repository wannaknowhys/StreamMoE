#include "backend/minigraph_exec.h"
#include "backend/route_b_chain.h"
#include "backend/moe_backend.h"
#include "backend/mix_split.h"
#include "backend/scatter_plan.h"
#include "backend/tensor_io.h"
#include "common/logger.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef STREAM_MOE_TEMP
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#endif

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

#ifdef STREAM_MOE_TEMP
// Accumulating wall-clock timer (STREAM_MOE_TMR=1), one line per bucket at exit.
struct tmr_acc_t {
    const char * name; double ms = 0; long n = 0;
    ~tmr_acc_t() { if (std::getenv("STREAM_MOE_TMR")) fprintf(stderr, "[TMR] %s total=%.1f ms n=%ld avg=%.3f ms\n", name, ms, n, n ? ms / n : 0.0); }
};
#endif

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

// Fast host-pinned staging buffer for device->host control-plane inspection (e.g. routing ids).
// Bypasses the unpinned fallback in ggml-vulkan that serializes every get_async call.
struct pinned_control_staging_t {
    ggml_backend_buffer_t buf = nullptr;
    size_t                size = 0;
    ggml_backend_dev_t    dev = nullptr;

    ~pinned_control_staging_t() {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
    }

    void * ensure(ggml_backend_dev_t d, size_t needed) {
        if (!d) return nullptr;
        if (dev != d && buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
            size = 0;
        }
        dev = d;
        if (needed <= size && buf) {
            return ggml_backend_buffer_get_base(buf);
        }
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(d);
        if (!host_buft) return nullptr;
        const size_t alloc_sz = ((needed + 65535) / 65536) * 65536;
        buf = ggml_backend_buft_alloc_buffer(host_buft, alloc_sz);
        if (!buf) {
            size = 0;
            return nullptr;
        }
        size = alloc_sz;
        return ggml_backend_buffer_get_base(buf);
    }
};

static pinned_control_staging_t g_pinned_control;

// Host image of a tensor for host-side inspection (iron rule): returns t->data
// when host-resident, else a backend-agnostic copy in `buf`. Never dereference
// t->data directly - the tensor may live on a device backend.
static const uint8_t * host_image(const ggml_tensor * t, std::vector<uint8_t> & buf) {
    if (!t->buffer || ggml_backend_buft_is_host(ggml_backend_buffer_get_type(t->buffer))) {
        return static_cast<const uint8_t *>(t->data);
    }
    const size_t nbytes = ggml_nbytes(t);
    buf.resize(nbytes);

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    void * p_stage = dev ? g_pinned_control.ensure(dev, nbytes) : nullptr;
    const char * dev_name = dev ? ggml_backend_dev_name(dev) : nullptr;
    ggml_backend_t be = dev_name ? route_b_device_backend(dev_name) : nullptr;

    if (p_stage && be) {
        ggml_backend_tensor_get_async(be, t, p_stage, 0, nbytes);
        ggml_backend_synchronize(be);
        std::memcpy(buf.data(), p_stage, nbytes);
    } else {
        tensor_read_host(t, buf.data(), 0, nbytes);
    }
    return buf.data();
}

// Read an ids element from a host byte image honoring the tensor's row stride.
static inline int32_t moe_id_at(const uint8_t * base, const ggml_tensor * ids, int t, int k) {
    return *(const int32_t *)(base + (size_t) t * ids->nb[1] + (size_t) k * ids->nb[0]);
}

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

// Pure aliases (no data movement): they share the producer's buffer, so a clone
// graph may skip them and let consumers read the aliased producer directly.
// CONT is NOT an alias - it is a real contiguous copy with its own buffer
// (ggml_cont -> ggml_dup_tensor) and must be executed like any compute node.
static bool is_alias_op(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
           n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_PERMUTE;
}

// Whole-layer ownership (docs/LAYER_EXECUTOR_DESIGN.md 4.4): run the dense
// head/tail as a subgraph of the ORIGINAL main-graph nodes (no clone). The node
// list is in graph order (capture order) and need not be contiguous in the main
// graph - the MoE closure interleaves with the dense head, so a plain
// ggml_graph_view range is not usable. The CPU backend only reads
// nodes/n_nodes when planning (ggml_graph_plan), so a hand-assembled cgraph over
// the original tensors is valid; their data/buffer/view are the arena ones set
// by layout_arena. Views are skipped (their data follows the producer).
#ifdef STREAM_MOE_TEMP
static int32_t live_dump_layer() {
    const char * value = std::getenv("STREAM_MOE_TMP_LIVE_DUMP_LAYER");
    if (!value) return -1;
    if (!*value) throw std::runtime_error("STREAM_MOE_TMP_LIVE_DUMP_LAYER must be a nonnegative int32");
    int32_t layer = 0;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || layer > (std::numeric_limits<int32_t>::max() - (*p - '0')) / 10) {
            throw std::runtime_error("STREAM_MOE_TMP_LIVE_DUMP_LAYER must be a nonnegative int32");
        }
        layer = layer * 10 + (*p - '0');
    }
    return layer;
}

static bool live_dump_once_take(int32_t layer, const char * stage, ggml_backend_t backend) {
    const char * once = std::getenv("STREAM_MOE_TMP_LIVE_DUMP_ONCE");
    if (!once || std::strcmp(once, "1") != 0) return true;
    static std::mutex mutex;
    static std::set<std::tuple<int32_t, std::string, uintptr_t>> seen;
    const std::lock_guard<std::mutex> lock(mutex);
    return seen.emplace(layer, stage, reinterpret_cast<uintptr_t>(backend)).second;
}

static std::filesystem::path live_dump_call_dir(const std::filesystem::path & root, uint64_t & call) {
    static std::mutex mutex;
    static std::filesystem::path session_root, session;
    static uint64_t next_call = 0;
    const std::lock_guard<std::mutex> lock(mutex);
    if (session.empty()) {
        std::filesystem::create_directories(root);
        const std::string prefix = "live_session_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        for (uint64_t attempt = 0;; ++attempt) {
            const auto candidate = root / (prefix + "_" + std::to_string(attempt));
            if (std::filesystem::create_directory(candidate)) {
                session_root = root;
                session = candidate;
                break;
            }
            if (attempt == std::numeric_limits<uint64_t>::max()) throw std::runtime_error("LIVE_DUMP session id exhausted");
        }
    } else if (session_root != root) {
        throw std::runtime_error("STREAM_MOE_TMP_BIN_DIR changed during LIVE_DUMP session");
    }
    if (next_call == std::numeric_limits<uint64_t>::max()) throw std::runtime_error("LIVE_DUMP call id exhausted");
    call = ++next_call;
    const auto dir = session / ("call_" + std::to_string(call));
    if (!std::filesystem::create_directory(dir)) throw std::runtime_error("LIVE_DUMP call directory already exists");
    return dir;
}

static void live_dump_one(const std::filesystem::path & dir, uint64_t call, size_t index, int slot,
                          int32_t layer, const char * stage, int build, ggml_backend_t backend,
                          const ggml_tensor * node, const ggml_tensor * t, std::vector<char> & chunk,
                          std::string & active_path) {
    const std::string stem = "node_" + std::to_string(index) + (slot < 0 ? "_out" : "_src_" + std::to_string(slot));
    const auto bin_path = dir / (stem + ".bin");
    active_path = bin_path.u8string();
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] < 0) throw std::runtime_error("LIVE_DUMP negative tensor dimension");
    }
    const size_t bytes = ggml_nbytes(t);
    const auto buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (bytes && (!t->data || !buffer || !buffer->iface.get_tensor)) {
        throw std::runtime_error("LIVE_DUMP nonempty tensor has no readable data/buffer");
    }
    std::ofstream bin;
    bin.exceptions(std::ios::failbit | std::ios::badbit);
    bin.open(bin_path, std::ios::binary | std::ios::out | std::ios::trunc);
    for (size_t offset = 0; offset < bytes;) {
        const size_t count = std::min(chunk.size(), bytes - offset);
        tensor_read_host(t, chunk.data(), offset, count);
        bin.write(chunk.data(), static_cast<std::streamsize>(count));
        offset += count;
    }
    bin.close();
    const auto meta_path = dir / (stem + ".meta");
    active_path = meta_path.u8string();
    std::ofstream meta;
    meta.exceptions(std::ios::failbit | std::ios::badbit);
    meta.imbue(std::locale::classic());
    meta.open(meta_path, std::ios::binary | std::ios::out | std::ios::trunc);
    meta << "call " << call << "\nindex " << index << "\nslot " << slot
         << "\nphase " << std::quoted(slot < 0 ? "after" : "before")
         << "\nlayer " << layer << "\nstage " << std::quoted(stage) << "\nbuild " << build
         << "\nbackend " << std::quoted(ggml_backend_name(backend))
         << "\nnode_address " << static_cast<const void *>(node) << "\nnode_name " << std::quoted(node->name)
         << "\nname " << std::quoted(t->name) << "\ntype " << static_cast<int>(t->type)
         << ' ' << std::quoted(ggml_type_name(t->type)) << "\nop " << static_cast<int>(t->op)
         << ' ' << std::quoted(ggml_op_name(t->op)) << "\nne";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) meta << ' ' << t->ne[d];
    meta << "\nnb";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) meta << ' ' << t->nb[d];
    meta << "\nop_params";
    for (int32_t param : t->op_params) meta << ' ' << param;
    meta << "\nnbytes " << bytes << "\ntensor_address " << static_cast<const void *>(t)
         << "\nbuffer_address " << static_cast<const void *>(t->buffer)
         << "\nread_buffer_address " << static_cast<const void *>(buffer)
         << "\nbuffer_name " << std::quoted(buffer ? ggml_backend_buffer_name(buffer) : "none")
         << "\ndata " << t->data << "\nview_src " << static_cast<const void *>(t->view_src)
         << "\nview_offs " << t->view_offs << '\n';
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const auto * src = t->src[s];
        meta << "src " << s << ' ' << static_cast<const void *>(src) << ' '
             << std::quoted(src ? src->name : "") << '\n';
    }
    meta.close();
}
#endif

static enum ggml_status run_dense_subgraph(ggml_context * ctx, ggml_backend_t backend,
                                           const std::vector<ggml_tensor*> & list,
                                           [[maybe_unused]] int32_t layer, [[maybe_unused]] const char * stage) {
    if (!backend) return GGML_STATUS_FAILED;
    for (const auto * nd : list) {
        if (!nd || is_alias_op(nd) || nd->op == GGML_OP_NONE) continue;
        if (!ggml_backend_supports_op(backend, nd)) {
            LOG_ERROR("stream_moe: device " << ggml_backend_name(backend) << " does not support "
                      << nd->name << " (" << ggml_op_name(nd->op) << "); no CPU fallback");
            return GGML_STATUS_FAILED;
        }
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_LIVE_DUMP") || std::getenv("STREAM_MOE_TMP_SUBGRAPH_ONE")) {
        uint64_t call = 0;
        size_t index = 0;
        int slot = -1;
        std::string active_path;
        try {
            const char * env_dir = std::getenv("STREAM_MOE_TMP_BIN_DIR");
            if (!env_dir || !*env_dir) throw std::runtime_error("LIVE_DUMP/SUBGRAPH_ONE requires STREAM_MOE_TMP_BIN_DIR");
            const int32_t selected_layer = live_dump_layer();
            const auto root = std::filesystem::absolute(std::filesystem::u8path(env_dir)).lexically_normal();
            active_path = root.u8string();
            if ((selected_layer < 0 || layer == selected_layer) && live_dump_once_take(layer, stage, backend)) {
                const auto dir = live_dump_call_dir(root, call);
                active_path = dir.u8string();
                std::vector<char> chunk(1024 * 1024);
                ggml_cgraph * one = ggml_new_graph_custom(ctx, 4, false);
                if (!one) throw std::runtime_error("LIVE_DUMP single-node graph allocation failed");
                const int build = route_b_build_id();
                for (; index < list.size(); ++index) {
                    ggml_tensor * nd = list[index];
                    if (!nd || is_alias_op(nd)) continue;
                    slot = -1;
                    ggml_backend_synchronize(backend);
                    for (slot = 0; slot < GGML_MAX_SRC; ++slot) {
                        if (nd->src[slot]) live_dump_one(dir, call, index, slot, layer, stage, build, backend,
                                                        nd, nd->src[slot], chunk, active_path);
                    }
                    slot = -1;
                    active_path = (dir / ("node_" + std::to_string(index) + "_out.bin")).u8string();
                    const auto output_buffer = nd->view_src ? nd->view_src->buffer : nd->buffer;
                    if (ggml_nbytes(nd) && (!nd->data || !output_buffer || !output_buffer->iface.get_tensor)) {
                        throw std::runtime_error("LIVE_DUMP nonempty output has no readable data/buffer");
                    }
                    ggml_graph_clear(one);
                    one->nodes[one->n_nodes++] = nd;
                    const auto status = ggml_backend_graph_compute(backend, one);
                    if (status != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("LIVE_DUMP node compute failed, status=" + std::to_string(static_cast<int>(status)));
                    }
                    ggml_backend_synchronize(backend);
                    live_dump_one(dir, call, index, slot, layer, stage, build, backend, nd, nd, chunk, active_path);
                }
                return GGML_STATUS_SUCCESS;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "[live-dump] FAILED layer=%d stage=%s call=%llu index=%zu slot=%d path=%s: %s\n",
                    layer, stage, (unsigned long long) call, index, slot, active_path.c_str(), e.what());
            return GGML_STATUS_FAILED;
        } catch (...) {
            fprintf(stderr, "[live-dump] FAILED layer=%d stage=%s call=%llu index=%zu slot=%d path=%s: unknown exception\n",
                    layer, stage, (unsigned long long) call, index, slot, active_path.c_str());
            return GGML_STATUS_FAILED;
        }
    }
#endif
    if (list.empty()) return GGML_STATUS_SUCCESS;
    std::vector<ggml_tensor*> nodes;
    nodes.reserve(list.size());
    for (ggml_tensor * nd : list) {
        if (!nd || is_alias_op(nd) || !nd->data) continue;
        nodes.push_back(nd);
    }
    if (nodes.empty()) return GGML_STATUS_SUCCESS;
#ifdef STREAM_MOE_TEMP
    const char * tr = std::getenv("STREAM_MOE_TMP_TRACE_NODE");
    if (tr && *tr) {
        for (ggml_tensor * nd : nodes) {
            if (!nd->name || std::strcmp(nd->name, tr) != 0) continue;
            fprintf(stderr, "[trace] '%s' op=%s data=%p\n", nd->name, ggml_op_name(nd->op), nd->data);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nd->src[s];
                if (!src) continue;
                double v0 = 0.0;
                if (src->data) {
                    if (src->type == GGML_TYPE_F32) v0 = *(const float *) src->data;
                    else if (src->type == GGML_TYPE_I32) v0 = (double) *(const int32_t *) src->data;
                }
                fprintf(stderr, "[trace]   src[%d] '%s' op=%s type=%s ne=[%lld,%lld] data=%p v0=%.6g\n",
                        s, src->name ? src->name : "?", ggml_op_name(src->op),
                        ggml_type_name(src->type), (long long) src->ne[0], (long long) src->ne[1],
                        src->data, v0);
            }
        }
    }
#endif
    ggml_cgraph * g = ggml_new_graph_custom(ctx, nodes.size() + 8, false);
    if (!g) {
        LOG_ERROR("stream_moe: dense subgraph build failed");
        return GGML_STATUS_ALLOC_FAILED;
    }
    for (ggml_tensor * nd : nodes) g->nodes[g->n_nodes++] = nd;
    return ggml_backend_graph_compute(backend, g);
}

// Whole-layer dense head/tail per placement device (docs/PER_DEVICE_ARENA.md
// SS8.5, phase 3): group the nodes by their captured device and run each group
// on that device's backend. A whole C1 (--dense-placement) is one device, so the
// common case is a single group; grouping keeps the structure correct if C1 is
static enum ggml_status run_dense_by_device(ggml_context * ctx, ggml_backend_t cpu,
                                            const std::vector<ggml_tensor*> & list,
                                            int32_t layer, const char * stage) {
    if (list.empty()) return GGML_STATUS_SUCCESS;
    std::vector<std::string> keys;
    std::vector<std::vector<ggml_tensor*>> groups;
    for (ggml_tensor * nd : list) {
        const char * d = route_b_node_device(nd);
        if (!d) {
            LOG_ERROR("stream_moe: unknown physical device for '" << nd->name << "'");
            return GGML_STATUS_FAILED;
        }
        const std::string key = d;
        size_t gi = 0;
        for (; gi < keys.size(); ++gi) if (keys[gi] == key) break;
        if (gi == keys.size()) { keys.push_back(key); groups.push_back({}); }
        groups[gi].push_back(nd);
    }
    for (size_t gi = 0; gi < keys.size(); ++gi) {
        ggml_backend_t be = keys[gi].empty() ? cpu : route_b_device_backend(keys[gi].c_str());
        if (!be || std::strcmp(ggml_backend_name(be), "STREAMMOE") == 0) {
            LOG_ERROR("stream_moe: no physical backend for device '" << keys[gi] << "'");
            return GGML_STATUS_FAILED;
        }
        const enum ggml_status st = run_dense_subgraph(ctx, be, groups[gi], layer, stage);
        if (st != GGML_STATUS_SUCCESS) {
            LOG_ERROR("stream_moe: dense subgraph failed on device '" << keys[gi] << "'");
            return st;
        }
    }
    return GGML_STATUS_SUCCESS;
}

#ifdef STREAM_MOE_TEMP
// Mid-layer node dump (STREAM_MOE_TMP_STAGE_DUMP=1): FNV hash of each node's
// current bytes, tagged with the execution stage, so a live node that gets
// overwritten can be spotted before the layer ends.
static void dump_node_hash(int layer, const char * stage, const std::vector<ggml_tensor*> & list) {
    if (!std::getenv("STREAM_MOE_TMP_STAGE_DUMP")) return;
    for (ggml_tensor * nd : list) {
        if (!nd || is_alias_op(nd) || !nd->data) continue;
        const size_t nb = ggml_nbytes(nd);
        const uint8_t * bp = (const uint8_t *) nd->data;
        uint64_t h = 1469598103934665603ull;
        for (size_t bi = 0; bi < nb; ++bi) { h ^= bp[bi]; h *= 1099511628211ull; }
        char vbuf[80] = "";
        if (nd->type == GGML_TYPE_F32 && nb >= 4) {
            const float * fp = (const float *) bp;
            snprintf(vbuf, sizeof(vbuf), " v=[%.5g %.5g %.5g %.5g]", fp[0], fp[1], fp[2], fp[3]);
        }
        fprintf(stderr, "[stage] b%d %-5s L%d %-26s %-12s %016llx data=%p%s\n", route_b_build_id(), stage, layer,
                nd->name ? nd->name : "?", ggml_op_name(nd->op), (unsigned long long) h, nd->data, vbuf);
    }
}
#endif


// Slot of a pinned (layer, expert), or -1.
static int32_t pin_slot(const std::vector<expert_handle_t>& pins, uint32_t layer, uint32_t expert) {
    for (const auto & h : pins) {
        if (h.pinned && h.layer == layer && h.expert == expert) return h.slot;
    }
    return -1;
}

// Per-thread grow-only scratch for one layer's executor temporaries
// (docs/BUCKET_FAST_PATH 搂9.2): integer ids/indices, CPU fold scratch, and the
// flat mix/scatter plan storage. Reset per layer; the buckets engine reserves
// the layer's need up front so a leaf's data pointer never moves mid-build.
struct exec_scratch_t {
    std::vector<int32_t>       i32;          // ids / indices / t_round (bump)
    std::vector<float>         f32;          // CPU fold scratch (bump)
    std::vector<int32_t>       plan_ids;     // mix_plan flat ids
    std::vector<mix_scatter_t> plan_scatter; // mix_plan flat scatter
    std::vector<mix_round_t>   plan_rounds;  // mix_plan rounds (spans)
    std::vector<uint32_t>      sp_order;     // scatter_plan tight order
    std::vector<scatter_seg_t> sp_segs;      // scatter_plan acc runs
    // Grow-only host copies of device-source leaves read on a CPU round
    // (bucket_source_leaf). Independent of f32/i32 so the fragile shared bump
    // sizing cannot be overrun by a leaf read (docs/DEVICE_DENSE_CLOSURE.md 3.8).
    std::vector<uint8_t>       leaf_host;
    size_t i32_used = 0, f32_used = 0, leaf_used = 0;
    void reset() {
        i32_used = 0; f32_used = 0; leaf_used = 0;
        plan_ids.clear(); plan_scatter.clear(); plan_rounds.clear();
        sp_order.clear(); sp_segs.clear();
    }
    int32_t * a32(size_t n) { int32_t * p = i32.data() + i32_used; i32_used += n; return p; }
    float   * af32(size_t n) { float * p = f32.data() + f32_used; f32_used += n; return p; }
    uint8_t * abytes(size_t n) { uint8_t * p = leaf_host.data() + leaf_used; leaf_used += n; return p; }
};
static thread_local exec_scratch_t g_scratch;
#ifdef STREAM_MOE_TEMP
// Test-only forced split (docs/BUCKET_EXEC_TOKEN_SUBSET.md SS3): turn the real
// plan's single full round into SCATTERED token-subset rounds so the subset path
// (cur gather + index-gather weights + scatter_plan reorder) is exercised on the
// CPU engine. Splits BOTH axes by parity so (t,k) is non-contiguous. Throwaway
// validation code - delete once the subset path is verified.
static mix_plan_t test_split_rounds(
        const int32_t * ids, uint32_t n_k, uint32_t n_t,
        const int32_t * expert_pool, uint32_t n_expert, uint32_t n_pools,
        exec_scratch_t & sc) {
    mix_plan_t base = build_mix_plan(ids, n_k, n_t, expert_pool, n_expert, n_pools,
                                     sc.plan_ids, sc.plan_scatter, sc.plan_rounds);
    if (base.n_rounds != 1 || base.rounds[0].width != n_k ||
        base.rounds[0].n_active != n_t) {
        return base;
    }
    const uint32_t ksplit = (n_k % 2u == 0u) ? 2u : 1u;
    const uint32_t kw     = (ksplit == 2u) ? n_k / 2u : n_k;
    const uint32_t base_pool = base.rounds[0].pool;
    sc.plan_ids.clear(); sc.plan_scatter.clear(); sc.plan_rounds.clear();
    for (uint32_t kp = 0; kp < ksplit; ++kp) {
        for (uint32_t tp = 0; tp < 2u; ++tp) {
            uint32_t n_active = 0;
            for (uint32_t t = tp; t < n_t; t += 2u) ++n_active;
            if (n_active == 0 || kw == 0) continue;
            const uint32_t off = static_cast<uint32_t>(sc.plan_ids.size());
            for (uint32_t t = tp; t < n_t; t += 2u) {
                for (uint32_t s = 0; s < kw; ++s) {
                    const uint32_t k = (ksplit == 2u) ? (2u * s + kp) : s;
                    sc.plan_ids.push_back(ids[(size_t) t * n_k + k]);
                    sc.plan_scatter.push_back({ t, k });
                }
            }
            sc.plan_rounds.push_back({ base_pool, kw, n_active, off });
        }
    }
    mix_plan_t plan;
    plan.ids      = sc.plan_ids.data();
    plan.scatter  = sc.plan_scatter.data();
    plan.rounds   = sc.plan_rounds.data();
    plan.n_rounds = static_cast<uint32_t>(sc.plan_rounds.size());
    plan.n_expert_used = n_k; plan.n_tokens = n_t; plan.n_pools = n_pools;
    return plan;
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
// Per-device execution target (M2-2, docs/M2_DEVICE_EXECUTOR.md SS7.9): one
// cgraph + arena/stage buffers on a device pool. The CPU target is dev == null.
struct device_target_t {
    uint32_t               pool = 0;
    ggml_backend_t         be = nullptr;
    ggml_backend_buffer_t  arena = nullptr;   // chain intermediates + acc_d
    ggml_backend_buffer_t  stage = nullptr;   // uploaded cur/ids/weights/scale
    uint8_t *              arena_map = nullptr;
    uint8_t *              stage_map = nullptr;
    size_t                 arena_used = 0;    // bump region (above result_bytes)
    size_t                 stage_used = 0;
    size_t                 acc_off = 0;       // acc_d[d_out, n_t] inside the arena
    ggml_tensor *          acc = nullptr;     // device accumulator shell
    ggml_tensor *          result = nullptr;  // single-target direct output shell
    ggml_cgraph *          gf = nullptr;
};

struct chain_ctx_t {
    ggml_context *             ctx  = nullptr;
    ggml_backend_t             cpu  = nullptr;
    expert_scheduler *         sched = nullptr;
    const moe_model_topology_t * topo = nullptr;
    int32_t                    layer = -1;
    const std::vector<expert_handle_t> * pins = nullptr;
    ggml_cgraph *              gf = nullptr;
    const moe_layer_exec_t *   ex = nullptr;
    // Execution target (2026-09-08): dev == null -> CPU host arena; else the
    // device pool. Pool 0 is always CPU.
    device_target_t *          dev = nullptr;
    bool is_device() const { return dev != nullptr; }
    // in-flight bucket geometry (full-width today: w == ids->ne[0])
    int64_t                    w_b = 0;   // experts per token in this bucket
    int64_t                    n_t = 0;   // tokens (full-width == ids->ne[1])
    // grow-only per-layer scratch (ids/index/fold buffers + plan storage)
    exec_scratch_t *           scratch = nullptr;
    // fold_repl: skip the anonymous per-topk fold + moe_out on the graph and
    // produce them via append_expert_fold + exit memcpy instead. Off = the old
    // IDENTICAL whole-clone behaviour (bring-up gate).
    bool fold_repl = false;
    int64_t                 d_out = 0;
    int64_t                 n_slots = 0;          // full routed slot count (ids ne0)
    // CPU accumulator [d_out, n_t]: process-lifetime grow-only, zeroed per layer.
    // Device targets keep their own acc_d in the device arena.
    std::vector<float>      acc_d;
    // (b) Per-round temporaries (fold pc/s/acc, gathers, cur gather) each get
    // their OWN backend buffer instead of a shared bump with a guessed size
    // (docs/DEVICE_DENSE_CLOSURE.md 3.8). Freed at the layer end.
    std::vector<ggml_backend_buffer_t> owned;
    ggml_backend_buffer_type_t         cpu_buft = nullptr;
};

// Fold / exit helpers live after bucket_build_t (they need bind_fresh).

// Exit fold (SS7.8): sum the per-target accumulators (host mirrors, already
// read back from devices) into moe_out. `accs` holds one [d_out, n_t] pointer
// per participating target (CPU acc first, then devices).
static bool layer_fold(const moe_layer_exec_t * ex, int64_t d_out, int64_t n_t,
                       const std::vector<const float *> & accs) {
    ggml_tensor * moe_out = nullptr;
    for (const auto * cn : ex->compute) {
        if (cn->name && strstr(cn->name, "ffn_moe_out") != nullptr) { moe_out = const_cast<ggml_tensor*>(cn); break; }
    }
    if (!moe_out || !moe_out->data) return true;
    const size_t slot = (size_t)(d_out * n_t);
    // moe_out may live on a device (C1 placement / per-device plan): fold on the
    // host, then write back backend-agnostically (iron rule, docs/GRAPH_PARTITION.md).
    // Sum order is unchanged, so a host moe_out is byte-identical to before.
    std::vector<float> out(slot, 0.0f);
    bool first = true;
    for (const float * a : accs) {
        if (!a) continue;
        for (size_t i = 0; i < slot; ++i) out[i] = first ? a[i] : out[i] + a[i];
        first = false;
    }
    tensor_write_host(moe_out, out.data(), 0, slot * sizeof(float));
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
    // Identity-selection flags for the current round (docs/BUCKET_FAST_PATH):
    // tok_full  = n_active == n_t  -> the cur gather is identity;
    // cell_full = tok_full && w_b == n_k -> the weights gather is identity.
    bool tok_full = false, cell_full = false;
    // index of the closure node currently being cloned (== its out_off slot).
    // Twins write their compact output at the SAME out_off region as the main
    // full-width node (compact [d, w_b, n_t] <= full [d, n_k, n_t], so it stays
    // inside the verify-allocated byte range); the one-cgraph serial execution
    // makes bucket N+1's chain reuse bucket N's dead regions (M2 搂7.2.1).
    int64_t seq = -1;
    // routing ids (expert ids) for the bucket rows [w_b, n_active], t-major,
    // and slot-local translation [w_b, n_active] - scratch slices (stable until
    // the layer's graph_compute).
    int32_t *              ids_exp  = nullptr;
    int32_t *              ids_slot = nullptr;
    const ggml_tensor *    ids = nullptr;     // main routing ids (read-only, full)
    const ggml_tensor *    ids_data = nullptr;   // full main ids data (read-only)
    int64_t ids_ne0 = 0, ids_ne1 = 0;
    // current round (mix_plan): r->off + a*width + s indexes the flat ids/scatter.
    // t_round[a] = original token of round column a (a order); order[i] = the
    // round column of tight column i (scatter_plan).
    const mix_round_t *        r = nullptr;
    uint32_t *                 t_round = nullptr;   // scratch slice (n_active)
    const uint32_t *           order   = nullptr;   // scatter_plan tight order
    // fixed-size per-round dedup tables (closure <= ~20 nodes, gathers <= ~4);
    // replaces the former unordered_maps (no per-round allocation).
    static constexpr int MAX_TWIN = 64, MAX_GATHER = 16;
    struct kv_t { const ggml_tensor * k; ggml_tensor * v; };
    kv_t twin_arr[MAX_TWIN];   int n_twin   = 0;
    kv_t gather_arr[MAX_GATHER]; int n_gather = 0;
    ggml_tensor * twin_get(const ggml_tensor * k) const {
        for (int i = 0; i < n_twin; ++i) if (twin_arr[i].k == k) return twin_arr[i].v;
        return nullptr;
    }
    void twin_put(const ggml_tensor * k, ggml_tensor * v) { twin_arr[n_twin++] = { k, v }; }
    ggml_tensor * gather_get(const ggml_tensor * k) const {
        for (int i = 0; i < n_gather; ++i) if (gather_arr[i].k == k) return gather_arr[i].v;
        return nullptr;
    }
    void gather_put(const ggml_tensor * k, ggml_tensor * v) { gather_arr[n_gather++] = { k, v }; }
    void clear_round() { n_twin = 0; n_gather = 0; }
    // flat plan accessors (round span r->off into the scratch plan storage)
    const mix_scatter_t & scat(uint32_t a, int64_t s) const {
        return c->scratch->plan_scatter[(size_t) r->off + (size_t) a * r->width + (size_t) s];
    }
    const int32_t & rid(uint32_t a, int64_t s) const {
        return c->scratch->plan_ids[(size_t) r->off + (size_t) a * r->width + (size_t) s];
    }
    // diagnostics: how many twin outputs landed in the arena vs heap fallback
    int64_t n_arena = 0, n_heap = 0;
    // small helper: fresh float buffer kept alive until graph_compute
    float * buf(size_t n) { return c->scratch->af32(n); }
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
    // (b) Each per-round temporary gets its own backend buffer on the current
    // target's buft - no shared bump, nothing to size (the class of bug that
    // overran the f32 scratch). Freed at the layer end.
    void bind_owned(ggml_tensor * t, size_t nbytes) {
        chain_ctx_t & cc = *c;
        ggml_backend_buffer_type_t bt = cc.dev
            ? ggml_backend_buffer_get_type(cc.dev->arena) : cc.cpu_buft;
        ggml_backend_buffer_t b = bt ? ggml_backend_buft_alloc_buffer(bt, nbytes) : nullptr;
        if (!b) { t->data = nullptr; return; }
        cc.owned.push_back(b);
        t->buffer = b;
        t->data = cc.dev ? stmoe_vk_buffer_host_offset(b, 0)
                         : ggml_backend_buffer_get_base(b);
    }
    // Bind a fresh output tensor to the current target. Layout tensors (twins)
    // use the verify plan slot (device arena at out_off / host full-alloc);
    // everything else (per-round temporaries) gets its own buffer.
    void bind_fresh(ggml_tensor * t, size_t nbytes, bool use_layout) {
        chain_ctx_t & cc = *c;
        if (use_layout) {
            if (cc.dev) {
                if (ex && ex->layout_ok && seq >= 0 &&
                    seq < (int64_t) ex->out_off.size() && ex->out_off[(size_t) seq] >= 0) {
                    t->buffer = cc.dev->arena;
                    t->data   = stmoe_vk_buffer_host_offset(cc.dev->arena, (size_t) ex->out_off[(size_t) seq]);
                    return;
                }
            } else {
                void * p = twin_out(nbytes);
                if (p) { t->data = p; return; }
            }
        }
        bind_owned(t, nbytes);
    }
};

// Reference an existing buffer region (chain twin / weight shell). `data` is a
// host pointer on CPU, or a fake device pointer (host_offset) on device; in
// device mode `buffer` is the owning backend buffer (no upload).
static ggml_tensor * bucket_ref_leaf(chain_ctx_t & c, enum ggml_type type,
                                     const int64_t ne[4], const size_t nb[4],
                                     void * data, ggml_backend_buffer_t buffer = nullptr) {
    ggml_tensor * l = ggml_new_tensor_4d(c.ctx, type, ne[0], ne[1], ne[2], ne[3]);
    for (int i = 0; i < 4; ++i) l->nb[i] = nb[i];
    l->data = data;
    if (c.dev) l->buffer = buffer;
    return l;
}

// Leaf over host source data: CPU references it in place; device uploads it to
// the target staging buffer and binds the device tensor.
static ggml_tensor * bucket_upload_leaf(chain_ctx_t & c, enum ggml_type type,
                                        const int64_t ne[4], const size_t nb[4],
                                        const void * host_data) {
    ggml_tensor * l = ggml_new_tensor_4d(c.ctx, type, ne[0], ne[1], ne[2], ne[3]);
    for (int i = 0; i < 4; ++i) l->nb[i] = nb[i];
    if (c.dev && host_data) {
        const size_t bytes = ggml_nbytes(l);
        const size_t off = (c.dev->stage_used + 63u) & ~size_t(63u);
        c.dev->stage_used = off + bytes;
        l->buffer = c.dev->stage;
        l->data   = stmoe_vk_buffer_host_offset(c.dev->stage, off);
        // Backend-agnostic upload into the stage buffer (iron rule). The fake
        // base pointer makes the stage buffer's set_tensor resolve `off`.
        tensor_write_host(l, host_data, 0, bytes);
    } else {
        l->data = const_cast<void *>(host_data);
    }
    return l;
}

// Leaf over an EXISTING tensor source that may live on ANY backend (C1
// activations / routing weights after a `--dense-placement` on a device):
//   - device target: backend-agnostic read into the staging buffer;
//   - CPU target, host source: reference the source in place;
//   - CPU target, device source: backend-agnostic read into host scratch.
// Never dereference src->data on the host (iron rule, docs/GRAPH_PARTITION.md).
// The source geometry (ne) is a contiguous view of `src`, so a flat nbytes read
// is valid. This is the single entry point for "a leaf whose source device is
// not known statically" - do not reintroduce host `m->data` uploads.
static ggml_tensor * bucket_source_leaf(chain_ctx_t & c, const ggml_tensor * src,
                                        const int64_t ne[4], const size_t nb[4]) {
    ggml_tensor * l = ggml_new_tensor_4d(c.ctx, src->type, ne[0], ne[1], ne[2], ne[3]);
    for (int i = 0; i < 4; ++i) l->nb[i] = nb[i];
    const bool src_host = !src->buffer ||
        ggml_backend_buft_is_host(ggml_backend_buffer_get_type(src->buffer));
    const size_t bytes = ggml_nbytes(l);
    if (c.dev) {
        bool same_device = false;
        if (c.dev->be && src->buffer) {
            ggml_backend_dev_t s_dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(src->buffer));
            ggml_backend_dev_t d_dev = ggml_backend_get_device(c.dev->be);
            if (s_dev && d_dev && s_dev == d_dev) {
                same_device = true;
            }
        }
        if (same_device) {
            l->buffer    = src->buffer;
            l->data      = src->data;
            l->view_src  = src->view_src;
            l->view_offs = src->view_offs;
            return l;
        }
        const size_t off = (c.dev->stage_used + 63u) & ~size_t(63u);
        c.dev->stage_used = off + bytes;
        l->buffer = c.dev->stage;
        l->data   = stmoe_vk_buffer_host_offset(c.dev->stage, off);
        if (src_host && src->data) {
            tensor_write_host(l, src->data, 0, bytes);
        } else if (src->buffer) {
            ggml_backend_tensor_copy(src, l);
        } else {
            std::vector<uint8_t> tmp(bytes);
            tensor_read_host(src, tmp.data(), 0, bytes);
            tensor_write_host(l, tmp.data(), 0, bytes);
        }
    } else if (src_host) {
        l->data = const_cast<void *>(src->data);
    } else {
        uint8_t * buf = c.scratch->abytes(bytes);
        l->data = buf;
        tensor_read_host(src, buf, 0, bytes);
    }
    return l;
}

// Fold the round's experts: sum over the per-token expert axis (w_b) so the
// weighted output [d_out, w_b, n_active] becomes a per-token column
// [d_out, n_active]. Every intermediate/output is bound to the current target
// (CPU heap/fullalloc or the device arena).
static ggml_tensor * append_expert_fold(bucket_build_t & b, ggml_tensor * weighted) {
    if (!weighted) return nullptr;
    chain_ctx_t & c = *b.c;
    const int64_t ne0 = weighted->ne[0];   // d_out
    const int64_t nw  = weighted->ne[1];   // w_b (experts per token)
    const int64_t nt  = weighted->ne[2];   // n_active tokens
    // Fold path by size (docs/WORK_IN_PROGRESS P1b, measured 2026-09-09): small
    // folds keep the SUM_ROWS reduction (3 ops, low fixed cost - decode); large
    // folds use the per-k view+add chain (upstream fold shape, no CONT/SUM_ROWS;
    // ~1.5x faster at 2048 tokens, crossover ~16MB of weighted data). Both are
    // numerically equivalent within the relaxed gate (different sum order).
    const size_t fold_bytes = (size_t) nw * (size_t) ne0 * (size_t) nt * sizeof(float);
    bool use_add = nw >= 2 && fold_bytes >= (16u * 1024 * 1024);
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_FOLD_ADD")) use_add = nw >= 2;
    if (std::getenv("STREAM_MOE_TMP_FOLD_SUM")) use_add = false;
#endif
    if (use_add) {
        // Ping-pong two [d_out, n_active] buffers across the per-expert adds.
        ggml_tensor * accA = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, ne0, nt);
        b.bind_fresh(accA, (size_t)(ne0 * nt) * sizeof(float), false);
        ggml_tensor * accB = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, ne0, nt);
        b.bind_fresh(accB, (size_t)(ne0 * nt) * sizeof(float), false);
        ggml_tensor * cur = ggml_view_2d(c.ctx, weighted, ne0, nt, weighted->nb[2], 0);
        for (int64_t s = 1; s < nw; ++s) {
            ggml_tensor * v = ggml_view_2d(c.ctx, weighted, ne0, nt,
                                           weighted->nb[2], (size_t) s * weighted->nb[1]);
            ggml_tensor * add = ggml_add(c.ctx, cur, v);
            ggml_tensor * dst = (s & 1) ? accA : accB;
            add->data   = dst->data;
            add->buffer = dst->buffer;
            ggml_build_forward_expand(c.gf, add);
            cur = add;
        }
        return cur;
    }
    ggml_tensor * p   = ggml_permute(c.ctx, weighted, 1, 0, 2, 3);   // view: [nw, d_out, n_active]
    ggml_tensor * pc  = ggml_cont(c.ctx, p);                         // contiguous [nw, d_out, n_active]
    b.bind_fresh(pc, (size_t)(nw * ne0 * nt) * sizeof(float), false);
    ggml_tensor * s   = ggml_sum_rows(c.ctx, pc);                    // [1, d_out, n_active]
    b.bind_fresh(s, (size_t)(ne0 * nt) * sizeof(float), false);
    ggml_tensor * acc = ggml_cont_2d(c.ctx, s, ne0, nt);             // [d_out, n_active]
    if (!acc) return nullptr;
    b.bind_fresh(acc, (size_t)(ne0 * nt) * sizeof(float), false);
    ggml_build_forward_expand(c.gf, acc);
    return acc;
}

// ---- general index-gather (docs/BUCKET_EXEC_TOKEN_SUBSET.md SS2.1/2.1a) ------
// get_rows gathers the ROW axis (ne1) of a 2D source. Build a flat / 2D source
// leaf over `m->data`, an i32 index leaf in tight order, get_rows, reshape to
// the chain geometry. The reshape view carries the data dependency, so it is the
// consumer's src (a detached leaf would lose the edge).
static ggml_tensor * bucket_direct_leaf(bucket_build_t & b, const ggml_tensor * m,
                                        const int64_t ne[4], const size_t nb[4]) {
    // Identity selection (docs/BUCKET_FAST_PATH): the bucket's rectangle IS the
    // whole source grid, so the source is consumed directly instead of gathered.
    // CPU references it in place; a device needs a compact (contiguous) source
    // for the one-shot upload - callers fall back to the gather otherwise.
    chain_ctx_t & c = *b.c;
    return bucket_source_leaf(c, m, ne, nb);
}

static ggml_tensor * bucket_gather_per_slot(bucket_build_t & b, const ggml_tensor * m) {
    // m is a per-(slot, token) external leaf [1, n_k, n_t] (routing weights). A
    // mix_plan round selects an ARBITRARY (t,k) subset, so the affine k-slice no
    // longer applies: gather flat element offsets.
    chain_ctx_t & c = *b.c;
    const size_t esz = sizeof(float);
    // Identity: the round covers the whole (t,k) grid -> reference the source.
    if (b.cell_full &&
        (!c.dev || (m->nb[0] == esz && m->nb[1] == esz &&
                    m->nb[2] == (size_t) m->ne[1] * esz))) {
        int64_t ne[4]; size_t nb[4];
        for (int i = 0; i < 4; ++i) { ne[i] = m->ne[i]; nb[i] = m->nb[i]; }
        return bucket_direct_leaf(b, m, ne, nb);
    }
    // flat source [1, span] over m->data; index uses the tensor's real strides
    // so a strided view still works. span = max flat element index + 1.
    const size_t stride_k = m->nb[1], stride_t = m->nb[2];
    const size_t span = (stride_t * (size_t)(m->ne[2] ? m->ne[2] - 1 : 0) +
                         stride_k * (size_t)(m->ne[1] ? m->ne[1] - 1 : 0)) / esz + 1;
    int64_t sne[4] = { 1, (int64_t) span, 1, 1 };
    size_t  snb[4] = { esz, esz, span * esz, span * esz };
    ggml_tensor * flat = bucket_source_leaf(c, m, sne, snb);
    const int64_t n = (int64_t) b.w_b * b.n_active;
    int32_t * idx = c.scratch->a32((size_t) n);
    for (int64_t i = 0; i < b.n_active; ++i) {
        const uint32_t a = b.order[(size_t) i];
        for (int64_t s = 0; s < b.w_b; ++s) {
            const mix_scatter_t & sc = b.scat(a, s);
            idx[(size_t) i * b.w_b + (size_t) s] =
                (int32_t)((sc.k * stride_k + sc.t * stride_t) / esz);
        }
    }
    int64_t ine[4] = { n, 1, 1, 1 };
    size_t  inb[4] = { 4, (size_t) n * 4, (size_t) n * 4, (size_t) n * 4 };
    ggml_tensor * idx_leaf = bucket_upload_leaf(c, GGML_TYPE_I32, ine, inb, idx);
    ggml_tensor * gr = ggml_get_rows(c.ctx, flat, idx_leaf);
    b.bind_fresh(gr, (size_t) b.w_b * b.n_active * sizeof(float), false);
    return ggml_reshape_3d(c.ctx, gr, 1, b.w_b, b.n_active);
}

static ggml_tensor * bucket_gather_cur(bucket_build_t & b, const ggml_tensor * m) {
    // m is the shared activation [d, 1, n_t] (one column per token). Gather the
    // token axis in tight order -> [d, 1, n_active]. Token stride is m->nb[2]
    // (ne1 == 1); the 2D source leaf [d, n_t] uses it as the row stride.
    chain_ctx_t & c = *b.c;
    const int64_t d = m->ne[0], n_t = m->ne[2];
    const size_t  esz = sizeof(float);
    // Identity: the round covers every token -> reference the source.
    if (b.tok_full &&
        (!c.dev || (m->nb[0] == esz && m->nb[2] == (size_t) d * esz))) {
        int64_t ne[4] = { d, 1, n_t, 1 };
        size_t  nb[4] = { m->nb[0], m->nb[1], m->nb[2], m->nb[3] };
        return bucket_direct_leaf(b, m, ne, nb);
    }
    int64_t sne[4] = { d, n_t, 1, 1 };
    size_t  snb[4] = { m->nb[0], m->nb[2], m->nb[2] * (size_t) n_t,
                       m->nb[2] * (size_t) n_t };
    ggml_tensor * src2d = bucket_source_leaf(c, m, sne, snb);
    int32_t * idx = c.scratch->a32((size_t) b.n_active);
    for (int64_t i = 0; i < b.n_active; ++i) {
        idx[(size_t) i] = (int32_t) b.t_round[b.order[(size_t) i]];
    }
    int64_t ine[4] = { b.n_active, 1, 1, 1 };
    size_t  inb[4] = { 4, (size_t) b.n_active * 4, (size_t) b.n_active * 4,
                       (size_t) b.n_active * 4 };
    ggml_tensor * idx_leaf = bucket_upload_leaf(c, GGML_TYPE_I32, ine, inb, idx);
    ggml_tensor * gr = ggml_get_rows(c.ctx, src2d, idx_leaf);
    b.bind_fresh(gr, (size_t) d * b.n_active * sizeof(float), false);
    return ggml_reshape_3d(c.ctx, gr, d, 1, b.n_active);
}

// External leaf: per-slot [1,n_k,n_t] -> tight index-gather; otherwise a plain
// full leaf (expert weight tables / scale broadcast, untouched by bucketing).
// A C4 leaf replicated at closure analysis (small closure-used non-per-expert,
// e.g. gemma per-expert scale) binds the resident device copy - no re-upload.
static ggml_tensor * bucket_ext_leaf(bucket_build_t & b, const ggml_tensor * m) {
    chain_ctx_t & c = *b.c;
    const int64_t n_k = b.ids_ne0;
    if (m->ne[0] == 1 && m->ne[1] == n_k) {
        ggml_tensor * g = b.gather_get(m);
        if (!g) { g = bucket_gather_per_slot(b, m); b.gather_put(m, g); }
        return g;
    }
    if (c.dev) {
        // resident C4 lookup keys on the unwrapped producer root (verify
        // registered the weight leaf, not the reshape/view that consumes it)
        const ggml_tensor * root = m;
        int64_t off = 0;
        while (root && (root->op == GGML_OP_VIEW || root->op == GGML_OP_RESHAPE ||
                        root->op == GGML_OP_TRANSPOSE || root->op == GGML_OP_PERMUTE ||
                        root->op == GGML_OP_CONT)) {
            if (root->op == GGML_OP_VIEW) off += root->view_offs;
            root = root->src[0];
        }
        ggml_backend_buffer_t rbuf = nullptr;
        void* rdev = root ? stream_moe_backend_resident_leaf(c.dev->pool, root, &rbuf) : nullptr;
        if (rdev) {
            if (std::getenv("STREAM_MOE_DBG"))
                std::fprintf(stderr, "[leaf] bind resident %s\n", m->name ? m->name : "?");
            ggml_tensor * l = ggml_new_tensor_4d(c.ctx, m->type, m->ne[0], m->ne[1], m->ne[2], m->ne[3]);
            for (int i = 0; i < 4; ++i) l->nb[i] = m->nb[i];
            l->buffer = rbuf;
            l->data   = stmoe_vk_buffer_host_offset(rbuf, static_cast<size_t>(off));
            return l;
        }
    }
    int64_t ne[4]; size_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = m->ne[i]; nb[i] = m->nb[i]; }
    return bucket_source_leaf(c, m, ne, nb);
}

// Leaf twin of a chain producer `p` (already built) narrowed to the bucket.
static ggml_tensor * bucket_twin_leaf(bucket_build_t & b, const ggml_tensor * p) {
    const ggml_tensor * t = b.twin_get(p);
    if (!t) return nullptr;
    int64_t ne[4]; size_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = t->ne[i]; nb[i] = t->nb[i]; }
    return bucket_ref_leaf(*b.c, t->type, ne, nb, t->data, t->buffer);
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
        const ggml_tensor * t = b.twin_get(root);
        if (t) {
            int64_t ne[4]; size_t nb[4];
            for (int i = 0; i < 4; ++i) { ne[i] = t->ne[i]; nb[i] = t->nb[i]; }
            // apply the outermost view's d-slice: ne0 (d rows) from the src view
            ne[0] = src->ne[0];
            return bucket_ref_leaf(*b.c, t->type, ne, nb,
                                   static_cast<char*>(t->data) + off, t->buffer);
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
    int64_t ne_ids[4] = { b.w_b, b.n_active, 1, 1 };
    size_t nb_ids[4] = { 4, (size_t)(b.w_b) * 4, (size_t)(b.w_b * b.n_active) * 4, 0 };
    nb_ids[3] = nb_ids[2];
    ggml_tensor * ids_leaf = bucket_upload_leaf(c, GGML_TYPE_I32, ne_ids, nb_ids,
                                                b.ids_slot);

    ggml_tensor * w3d = ggml_new_tensor_3d(c.ctx, w->type, w->ne[0], w->ne[1], 1);
    w3d->ne[2] = static_cast<int32_t>(sp->n_slots);
    w3d->nb[2] = col_stride;
    w3d->nb[3] = col_stride * sp->n_slots;
    if (c.dev) {
        ggml_backend_buffer_t wbuf = reinterpret_cast<ggml_backend_buffer_t>(sp->dev_buf);
        w3d->buffer = wbuf;
        w3d->data   = stmoe_vk_buffer_host_offset(wbuf, col_off);
    } else {
        w3d->data   = sp->base + col_off;
    }

    // cur: chain twin (down mm reads the bucket's own compact GLU) or external
    // shared cur leaf (gate_up, ne11 == 1 -> every bucket slot reads col 0).
    ggml_tensor * cur_leaf = nullptr;
    if (cur->op != GGML_OP_NONE) {
        cur_leaf = bucket_twin_leaf(b, cur);
    }
    if (!cur_leaf) {
        // External shared activation: gather the token axis (GPU shape: each
        // bucket owns its cur copy). Cached per round (gate + up mm share it).
        cur_leaf = b.gather_get(cur);
        if (!cur_leaf) { cur_leaf = bucket_gather_cur(b, cur); b.gather_put(cur, cur_leaf); }
        if (!cur_leaf) return nullptr;
    }
    ggml_tensor * mm = ggml_mul_mat_id(c.ctx, w3d, cur_leaf, ids_leaf);
    // compact dst [d_out, w_b, n_active]: pinned into the layer result arena at
    // this closure node's out_off (full-width region; compact is smaller so it
    // stays inside). Falls back to a per-bucket heap buffer when the layer has
    // no verify layout (then every bucket keeps its own copies - safe, just no
    // serial reuse).
    const size_t nf = (size_t)(w->ne[1] * b.w_b * b.n_active);
    b.bind_fresh(mm, nf * sizeof(float), true);
    // nb: contiguous compact layout (ggml_new_tensor_4d would not apply to an op)
    mm->nb[0] = 4;
    mm->nb[1] = (size_t)(w->ne[1]) * 4;
    mm->nb[2] = (size_t)(w->ne[1] * b.w_b) * 4;
    mm->nb[3] = mm->nb[2] * (size_t) b.n_active;
    ggml_build_forward_expand(c.gf, mm);
    b.twin_put(nd, mm);
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
    std::memcpy(cl->op_params, nd->op_params, GGML_MAX_OP_PARAMS);
    // per-src resolution. GET_ROWS src1 = ids: replace with the bucket ids_exp
    // subset (expert ids, i32, t-major [w_b, n_active]) - get_rows gathers rows
    // of the REPEAT scale table by expert id.
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        ggml_tensor * src = nd->src[s];
        if (!src) continue;
        ggml_tensor * lf = nullptr;
        if (nd->op == GGML_OP_GET_ROWS && s == 1 && src->type == GGML_TYPE_I32) {
            // ids leaf points at the scratch ids_exp (stable until graph_compute)
            int64_t sne[4] = { b.w_b, b.n_active, 1, 1 };
            size_t snb[4] = { 4, (size_t)(b.w_b) * 4, (size_t)(b.w_b * b.n_active) * 4, (size_t)(b.w_b * b.n_active) * 4 };
            lf = bucket_upload_leaf(c, GGML_TYPE_I32, sne, snb, b.ids_exp);
#ifdef STREAM_MOE_TEMP
            if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG")) {
                const size_t n_exp = (size_t) b.w_b * b.n_active;
                int32_t mn = INT32_MAX, mx = INT32_MIN;
                for (size_t q = 0; q < n_exp; ++q) { mn = std::min(mn, b.ids_exp[q]); mx = std::max(mx, b.ids_exp[q]); }
                fprintf(stderr, "[chain_buckets] GET_ROWS ids_exp[%zu] range [%d,%d] src0=",
                        n_exp, mn, mx);
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
    b.bind_fresh(cl, nf * sizeof(float), true);
    ggml_build_forward_expand(c.gf, cl);
    b.twin_put(nd, cl);
    return cl;
}

// Build one bucket: walk the closure and produce the compact chain; returns the
// final (weighted) twin (the last before the anonymous fold when fold_repl).
// Each closure node's twin writes its compact output at arena + out_off[i]
// (same byte region the main full-width node occupies; compact is smaller, so
// it stays inside). In one serial cgraph bucket N+1's chain then reuses bucket
// N's dead regions (M2 搂7.2.1) - no per-bucket heap intermediates.
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

// Device graphs need every tensor's `buffer` set: vulkan resolves the buffer
// from tensor->buffer directly (ggml_vk_tensor_subbuffer), unlike the generic
// API which follows view_src. Our synthetic graph sets it on allocated tensors
// but not on ggml-created views (reshape/permute/view) - fill those from their
// root view_src.
static void fix_view_buffers(ggml_cgraph * gf) {
    auto root_buf = [](ggml_tensor * t) -> ggml_backend_buffer_t {
        while (t && t->buffer == nullptr && t->view_src) t = t->view_src;
        return t ? t->buffer : nullptr;
    };
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * n = gf->nodes[i];
        if (n->buffer == nullptr && n->view_src) n->buffer = root_buf(n->view_src);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = n->src[s];
            if (src && src->buffer == nullptr && src->view_src) src->buffer = root_buf(src->view_src);
        }
    }
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
                                                       const std::vector<expert_handle_t>& pins,
                                                       const ggml_tensor * ids,
                                                       uint32_t n_k, uint32_t n_t,
                                                       const std::vector<int32_t>& ids_compact) {
    if (n_t == 0 || n_k == 0) return GGML_STATUS_SUCCESS;
    const uint32_t n_expert = sched.topology().n_expert;
    const uint32_t n_pools  = sched.n_pools();

    // Per-expert pool from the pinned handles (pin_layer returns one per active
    // expert with its pool). -1 = not active / unknown.
    std::vector<int32_t> expert_pool(n_expert, -1);
    for (const auto & h : pins) {
        if (h.expert < n_expert) expert_pool[h.expert] = (int32_t) h.pool;
    }

    // Round list: real build_mix_plan rounds (single RAM pool degenerates to one
    // full round); STREAM_MOE_TEMP may force the test-only scattered split.
    // Flat storage lives in the per-layer scratch (grow-only, no per-round alloc).
    exec_scratch_t & sc = g_scratch;
    sc.reset();
    mix_plan_t plan;
#ifdef STREAM_MOE_TEMP
    const char * split_env = std::getenv("STREAM_MOE_TMP_BUCKET_ROUNDS");
    if (split_env && *split_env) {
        plan = test_split_rounds(ids_compact.data(), n_k, n_t, expert_pool.data(),
                                 n_expert, n_pools, sc);
    } else
#endif
    {
        plan = build_mix_plan(ids_compact.data(), n_k, n_t, expert_pool.data(),
                              n_expert, n_pools, sc.plan_ids, sc.plan_scatter, sc.plan_rounds);
    }
    const mix_round_t * rounds   = plan.rounds;
    const uint32_t      n_rounds = plan.n_rounds;
    if (n_rounds == 0) {
        LOG_ERROR("stream_moe: chain_buckets empty round list L" << layer);
#ifdef STREAM_MOE_TEMP
        {
            size_t id_bad = 0, pool_bad = 0, valid = 0, total = (size_t) n_k * n_t;
            int32_t id_min = 0, id_max = 0;
            bool first = true;
            for (uint32_t t = 0; t < n_t; ++t) {
                for (uint32_t k = 0; k < n_k; ++k) {
                    const int32_t e = ids_compact[(size_t) t * n_k + k];
                    if (first) { id_min = id_max = e; first = false; }
                    if (e < id_min) id_min = e;
                    if (e > id_max) id_max = e;
                    if (e < 0 || e >= (int32_t) n_expert) { ++id_bad; continue; }
                    const int32_t p = expert_pool[e];
                    if (p < 0 || p >= (int32_t) n_pools) { ++pool_bad; continue; }
                    ++valid;
                }
            }
            int n_pins = (int) pins.size(), n_pins_pool = 0;
            for (const auto & h : pins)
                if ((int) h.pool >= 0 && (int) h.pool < (int) n_pools) ++n_pins_pool;
            fprintf(stderr, "[chain_buckets] EMPTY L%d: n_k=%u n_t=%u n_expert=%u n_pools=%u "
                    "total=%zu id[min=%d max=%d bad=%zu] pool_bad=%zu valid=%zu pins=%d pins_pool=%d\n",
                    layer, n_k, n_t, n_expert, n_pools, total, id_min, id_max, id_bad,
                    pool_bad, valid, n_pins, n_pins_pool);
            for (const auto * cn : ex->compute) {
                if (!cn || cn->op != GGML_OP_MUL_MAT_ID) continue;
                const ggml_tensor * i2 = cn->src[2];
                fprintf(stderr, "[chain_buckets]   MMID '%s' w='%s' ne=[%lld,%lld,%lld] "
                        "ids ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] data=%p\n",
                        cn->name ? cn->name : "?",
                        cn->src[0] && cn->src[0]->name ? cn->src[0]->name : "?",
                        (long long) cn->ne[0], (long long) cn->ne[1], (long long) cn->ne[2],
                        i2 ? (long long) i2->ne[0] : -1, i2 ? (long long) i2->ne[1] : -1,
                        i2 ? (long long) i2->ne[2] : -1,
                        i2 ? i2->nb[0] : (size_t) 0, i2 ? i2->nb[1] : (size_t) 0,
                        i2 ? i2->nb[2] : (size_t) 0, i2 ? i2->data : nullptr);
            }
        }
#endif
        return GGML_STATUS_FAILED;
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG")) {
        fprintf(stderr, "[chain_buckets] L%d: n_k=%u n_t=%u -> %u rounds\n",
                layer, n_k, n_t, n_rounds);
    }
#endif

    // ---- targets: CPU (pool 0) + one graph per participating device pool ----
    int64_t d_out = 0;
    for (const auto * cn : ex->compute) {
        if (cn) d_out = std::max(d_out, (int64_t) cn->ne[0]);
    }
    if (d_out <= 0) return GGML_STATUS_FAILED;
    // Shared activation dimension (uploaded per round by the cur gather).
    int64_t d_in = 0;
    for (const auto * cn : ex->compute) {
        if (cn && cn->op == GGML_OP_MUL_MAT_ID && cn->src[1] && cn->src[1]->op == GGML_OP_NONE) {
            d_in = std::max(d_in, (int64_t) cn->src[1]->ne[0]);
        }
    }

    chain_ctx_t c;
    c.ctx   = ctx;   c.cpu   = cpu;   c.sched = &sched;
    c.topo  = &sched.topology();
    c.layer = layer; c.pins  = &pins; c.ex    = ex;
    c.scratch = &sc;
    c.fold_repl = true;
    c.w_b = n_k; c.n_t = n_t; c.d_out = d_out;
    c.cpu_buft = ggml_backend_get_default_buffer_type(cpu);
    // Free the per-round temporary buffers once the layer's graphs have run.
    struct owned_guard_t {
        chain_ctx_t & cc;
        ~owned_guard_t() {
            for (ggml_backend_buffer_t b : cc.owned) ggml_backend_buffer_free(b);
            cc.owned.clear();
        }
    } _owned_guard{ c };

    bucket_build_t b;
    b.c = &c; b.ex = ex; b.ids = ids; b.ids_data = ids;
    b.ids_ne0 = n_k; b.ids_ne1 = n_t;

    ggml_cgraph * gf_cpu = ggml_new_graph(ctx);

    // Resolve each distinct non-RAM pool to a device target (skip pools without
    // an exec context: their rounds fall back to the CPU host-map read).
    std::unordered_map<uint32_t, device_target_t> dev_targets;
#ifdef STREAM_MOE_TEMP
    const bool force_cpu_dev = std::getenv("STREAM_MOE_TMP_FORCE_CPU") != nullptr;
#else
    const bool force_cpu_dev = false;
#endif
    for (uint32_t ri = 0; ri < n_rounds; ++ri) {
        const mix_round_t & r = rounds[ri];
        if (force_cpu_dev) break;
        if (r.pool == 0 || r.width == 0 || r.n_active == 0) continue;
        if (dev_targets.count(r.pool)) continue;
        device_exec_ctx_t * dv = stream_moe_backend_device_exec(r.pool);
        if (!dv || !dv->be) continue;
        device_target_t t;
        t.pool = r.pool;
        t.be   = dv->be;
        dev_targets.emplace(r.pool, t);
    }

    const size_t acc_sz = (size_t)(d_out * n_t);
    const size_t acc_bytes = acc_sz * sizeof(float);
    size_t stage_est = 32u * 1024 * 1024;
    for (uint32_t ri = 0; ri < n_rounds; ++ri) {
        const mix_round_t & r = rounds[ri];
        if (r.width == 0 || r.n_active == 0) continue;
        const size_t cells = (size_t) r.width * r.n_active;
        stage_est += (size_t) d_in * n_t * sizeof(float)   // cur full source
                   + (size_t) n_k * n_t * sizeof(float)    // routing-weight flat source
                   + (size_t) n_expert * sizeof(float)     // per-expert scale table
                   + cells * sizeof(int32_t) * 4 + 4096;   // ids + idx leaves
    }
    // External leaves uploaded to the device staging (bucket_ext_leaf ->
    // bucket_upload_leaf): every closure-node src whose unwrapped producer is
    // outside the closure and is not a per-slot gather leaf. Each reference
    // uploads a fresh copy, and every round re-uploads, so the sum over
    // references x rounds is an upper bound. (Missed before -> a gating leaf such
    // as gemma `ffn_moe_gate` overran the staging buffer.) Over-estimation is
    // safe: the staging buffer is grow-only and persists across layers.
    size_t ext_ref = 0;
    {
        auto in_compute = [&](const ggml_tensor * p) -> bool {
            for (const auto * cn : ex->compute) if (cn == p) return true;
            return false;
        };
        for (const auto * cn : ex->compute) {
            if (!cn) continue;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                // src[0] of a routed mm is the expert weight table: it lives in
                // the pool shell, never uploaded (its graph nbytes is the whole
                // expert table, which would blow the estimate).
                if (cn->op == GGML_OP_MUL_MAT_ID && s == 0) continue;
                const ggml_tensor * t = cn->src[s];
                if (!t) continue;
                const ggml_tensor * p = t;
                while (p && (p->op == GGML_OP_VIEW || p->op == GGML_OP_RESHAPE ||
                             p->op == GGML_OP_TRANSPOSE || p->op == GGML_OP_PERMUTE ||
                             p->op == GGML_OP_CONT)) p = p->src[0];
                if (!p || in_compute(p)) continue;
                if (p->ne[0] == 1 && p->ne[1] == (int64_t) n_k) continue;   // per-slot: arena gather
                ext_ref += ggml_nbytes(p) + 4096;
            }
        }
        stage_est += ext_ref * (n_rounds ? n_rounds : 1);
    }
    // Device arena bump consumers (bind_fresh(..., false)): the fold scratch
    // (pc/s/acc) + the cur gather + one gather per per-(slot,token) routing-
    // weight leaf. The bump is never reset within a layer (all tensors stay valid
    // until the single graph_compute), so size it to the SUM over the device's
    // rounds. The arena is allocated BEFORE the build (a grow would invalidate
    // already-bound twin data pointers), so this estimate must be an upper bound.
    int n_per_slot = 0;
    {
        std::vector<const ggml_tensor*> seen;
        for (const auto & el : ex->external_leaves) {
            const ggml_tensor * et = el.tensor;
            if (!et || et->ne[0] != 1 || et->ne[1] != (int64_t) n_k) continue;
            bool dup = false;
            for (auto * s : seen) if (s == et) { dup = true; break; }
            if (!dup) { seen.push_back(et); ++n_per_slot; }
        }
        if (n_per_slot < 2) n_per_slot = 2;   // safety margin (tiny vs the fold)
    }
    // Executor scratch (P1-c): reserve the layer's exact i32/f32 need once so
    // the bump never reallocates a leaf's data pointer mid-build. CPU targets
    // fold into f32; device targets use their arena bump.
    {
        size_t i32_need = 0, f32_need = 0;
        for (uint32_t ri = 0; ri < n_rounds; ++ri) {
            const mix_round_t & r = rounds[ri];
            if (r.width == 0 || r.n_active == 0) continue;
            const size_t cells = (size_t) r.width * r.n_active;
            i32_need += 2 * cells + 2 * r.n_active + (size_t) n_per_slot * cells;
            device_target_t * dt = nullptr;
            if (r.pool != 0) {
                auto it = dev_targets.find(r.pool);
                if (it != dev_targets.end()) dt = &it->second;
            }
            if (!dt) {
                // CPU target: fold pc/s/acc + cur/per-slot gather outputs all
                // land in the f32 scratch (bind_fresh(..., false)).
                f32_need += cells * (size_t) d_out + 2 * (size_t) d_out * r.n_active
                          + (size_t) d_in * r.n_active + (size_t) n_per_slot * cells;
            }
        }
        if (!ex->layout_ok) {
            // twins fall back to the f32 scratch when verify had no layout
            for (const auto * cn : ex->compute) f32_need += (ggml_nbytes(cn) + 3) / 4;
        }
        sc.i32.resize(i32_need + 64);
        sc.f32.resize(f32_need + 64);
        // Device-source leaves read on a CPU round get their own grow-only host
        // block: one copy per reference per round (an upper bound on ext_ref).
        sc.leaf_host.resize(ext_ref * (n_rounds ? n_rounds : 1) + 64);
    }
    auto align64 = [](size_t x) { return (x + 63u) & ~size_t(63u); };
    std::unordered_map<uint32_t, size_t> pool_bump;
    for (uint32_t ri = 0; ri < n_rounds; ++ri) {
        const mix_round_t & r = rounds[ri];
        if (r.width == 0 || r.n_active == 0) continue;
        const size_t cells = (size_t) r.width * r.n_active;
        size_t b = align64(cells * (size_t) d_out * sizeof(float))            // pc [w,d,nt]
                 + align64((size_t) d_out * r.n_active * sizeof(float))       // s  [1,d,nt]
                 + align64((size_t) d_out * r.n_active * sizeof(float))       // acc[d,nt]
                 + align64((size_t) d_in * r.n_active * sizeof(float))        // cur gather
                 + (size_t) n_per_slot * align64(cells * sizeof(float));      // per-slot gathers
        pool_bump[r.pool] += b;
    }
    for (auto & kv : dev_targets) {
        device_target_t & t = kv.second;
        device_exec_ctx_t * dv = stream_moe_backend_device_exec(t.pool);
        const size_t base = ex->layout_ok ? ((ex->result_bytes + 63u) & ~size_t(63u)) : 0;
        t.acc_off    = base;
        t.arena_used = base + acc_bytes;
        const size_t arena_bytes = t.arena_used + pool_bump[t.pool];
        if (!stream_moe_backend_device_ensure(t.pool, arena_bytes, stage_est)) {
            LOG_ERROR("stream_moe: device arena/stage alloc failed pool " << t.pool);
            return GGML_STATUS_FAILED;
        }
        t.arena = dv->arena; t.stage = dv->stage;
        t.arena_map = dv->arena_map; t.stage_map = dv->stage_map;
        t.gf = ggml_new_graph(ctx);
        t.acc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_out, n_t);
        t.acc->nb[0] = 4; t.acc->nb[1] = (size_t) d_out * 4;
        t.acc->buffer = t.arena;
        t.acc->data   = stmoe_vk_buffer_host_offset(t.arena, t.acc_off);
        const std::vector<float> zeros(acc_sz, 0.0f);   // zero acc_d on the device
        ggml_backend_tensor_set(t.acc, zeros.data(), 0, acc_bytes);
    }

    if (c.acc_d.size() < acc_sz) c.acc_d.assign(acc_sz, 0.0f);
    std::fill(c.acc_d.begin(), c.acc_d.end(), 0.0f);

    // Single-target fast path (docs/BUCKET_FAST_PATH): one round covering the
    // whole (t,k) grid -> per_token IS the layer output. Skip ACC + host fold and
    // write it to moe_out directly (CPU: bind data; device: one D2H readback).
    const bool single_target = (n_rounds == 1 &&
                                rounds[0].n_active == n_t && rounds[0].width == n_k);
    ggml_tensor * moe_out = nullptr;
    for (const auto * cn : ex->compute) {
        if (cn && cn->name && strstr(cn->name, "ffn_moe_out") != nullptr) {
            moe_out = const_cast<ggml_tensor*>(cn); break;
        }
    }
    const bool moe_out_host = moe_out && moe_out->data &&
        (!moe_out->buffer ||
         ggml_backend_buft_is_host(ggml_backend_buffer_get_type(moe_out->buffer)));
    const bool direct_out = single_target && moe_out && moe_out->data &&
        moe_out->nb[0] == sizeof(float) && moe_out->nb[1] == (size_t) d_out * sizeof(float);

#ifdef STREAM_MOE_TEMP
    // ACC / SUM_ROWS micro-bench (STREAM_MOE_TMP_ACC_BENCH=1), min of 5, on the
    // device: ACC [d,k] += [d,k]; SUM_ROWS sums the k axis ([k,d] -> [1,d], the
    // real expert-fold shape). Shows fixed vs proportional cost in k.
    if (std::getenv("STREAM_MOE_TMP_ACC_BENCH") && !dev_targets.empty()) {
        static bool acc_benched = false;
        if (!acc_benched) {
            acc_benched = true;
            device_target_t & dt = dev_targets.begin()->second;
            const int64_t d = 2048;
            const int64_t ks[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
            auto run = [&](ggml_context * bctx, ggml_cgraph * gf) -> double {
                double best = 1e9;
                for (int rep = 0; rep < 5; ++rep) {
                    const auto t0 = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute_async(dt.be, gf);
                    ggml_backend_synchronize(dt.be);
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (ms < best) best = ms;
                }
                (void) bctx;
                return best;
            };
            for (int64_t k : ks) {
                {   // ACC [d,k] += [d,k]
                    ggml_init_params ip = { 8u * 1024u * 1024u, nullptr, true };
                    ggml_context * bctx = ggml_init(ip);
                    ggml_tensor * a = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, d, k);
                    ggml_tensor * b = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, d, k);
                    a->buffer = dt.arena; a->data = stmoe_vk_buffer_host_offset(dt.arena, 0);
                    b->buffer = dt.arena; b->data = stmoe_vk_buffer_host_offset(dt.arena, (size_t) (d * k) * 4);
                    ggml_tensor * node = ggml_acc_inplace(bctx, a, b, (size_t) d * 4, (size_t) d * k * 4, (size_t) d * k * 4, 0);
                    ggml_cgraph * gf = ggml_new_graph(bctx);
                    ggml_build_forward_expand(gf, node);
                    std::fprintf(stderr, "[accbench] ACC      d=%lld k=%-4lld min_ms=%.4f\n", (long long) d, (long long) k, run(bctx, gf));
                    ggml_free(bctx);
                }
                {   // SUM_ROWS over k: [k,d] -> [1,d]
                    ggml_init_params ip = { 8u * 1024u * 1024u, nullptr, true };
                    ggml_context * bctx = ggml_init(ip);
                    ggml_tensor * b = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, k, d);
                    b->buffer = dt.arena; b->data = stmoe_vk_buffer_host_offset(dt.arena, 0);
                    ggml_tensor * node = ggml_sum_rows(bctx, b);
                    ggml_cgraph * gf = ggml_new_graph(bctx);
                    ggml_build_forward_expand(gf, node);
                    std::fprintf(stderr, "[accbench] SUMROWS  k=%-4lld d=%lld min_ms=%.4f\n", (long long) k, (long long) d, run(bctx, gf));
                    ggml_free(bctx);
                }
            }
        }
    }
#endif

#ifdef STREAM_MOE_TEMP
    static tmr_acc_t g_rounds_dec{ "chain_rounds_dec" }, g_rounds_pre{ "chain_rounds_pre" },
                      g_tail_dec{ "chain_tail_dec" }, g_tail_pre{ "chain_tail_pre" };
    tmr_acc_t & _rt = (n_t == 1) ? g_rounds_dec : g_rounds_pre;
    tmr_acc_t & _tt = (n_t == 1) ? g_tail_dec : g_tail_pre;
    auto _ch_t0 = std::chrono::steady_clock::now();
#endif
    bool has_cpu_round = false;
    for (uint32_t ri = 0; ri < n_rounds; ++ri) {
        const mix_round_t & r = rounds[ri];
        if (r.width == 0 || r.n_active == 0) continue;
        device_target_t * dt = nullptr;
        if (r.pool != 0) {
            auto it = dev_targets.find(r.pool);
            if (it != dev_targets.end()) dt = &it->second;
        }
        if (!dt) has_cpu_round = true;
        c.dev = dt;
        c.gf  = dt ? dt->gf : gf_cpu;
        b.r = &r;
        b.w_b = r.width; b.n_active = r.n_active;
        b.tok_full  = (r.n_active == n_t);
        b.cell_full = b.tok_full && (r.width == n_k);

        // scatter_plan: t_round[a] = original token of round column a; order[i]
        // = round column of tight column i; segs = acc arithmetic runs.
        b.t_round = reinterpret_cast<uint32_t *>(c.scratch->a32(r.n_active));
        for (uint32_t a = 0; a < r.n_active; ++a) {
            b.t_round[a] = b.scat(a, 0).t;
        }
        scatter_plan_t sp = build_scatter_plan(b.t_round, r.n_active, n_t,
                                               c.scratch->sp_order, c.scratch->sp_segs);
        if (sp.n_order != r.n_active) {
            LOG_ERROR("stream_moe: chain_buckets scatter_plan failed L" << layer
                      << " round " << ri << " (n_active " << r.n_active << ")");
            return GGML_STATUS_FAILED;
        }
        b.order = sp.order;

        // tight-order ids (expert id + pool-local slot id).
        const size_t n_cells = (size_t) r.width * r.n_active;
        b.ids_exp  = c.scratch->a32(n_cells);
        b.ids_slot = c.scratch->a32(n_cells);
        std::fill(b.ids_exp,  b.ids_exp  + n_cells, 0);
        std::fill(b.ids_slot, b.ids_slot + n_cells, 0);
        bool ok = true;
        for (uint32_t i = 0; i < r.n_active && ok; ++i) {
            const uint32_t a = b.order[i];
            for (uint32_t s = 0; s < r.width; ++s) {
                const int32_t e = b.rid(a, s);
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
            fprintf(stderr, "[chain_buckets]   r%zu: pool=%u dev=%d w_b=%u n_active=%u segs=%u\n",
                    ri, r.pool, dt ? 1 : 0, r.width, r.n_active, sp.n_segs);
        }
#endif
        b.clear_round();
        b.n_arena = 0; b.n_heap = 0;
        ggml_tensor * last = append_bucket_chain_compact(b);
        if (!last) { LOG_ERROR("stream_moe: chain_buckets append failed L" << layer); return GGML_STATUS_FAILED; }
        ggml_tensor * per_token = append_expert_fold(b, last);
        if (!per_token) return GGML_STATUS_FAILED;
        if (per_token->ne[1] != (int64_t) r.n_active) {
            LOG_ERROR("stream_moe: chain_buckets fold width " << per_token->ne[1]
                      << " != n_active " << r.n_active << " L" << layer);
            return GGML_STATUS_FAILED;
        }
        if (direct_out) {
            // Single target: per_token is the whole answer -> write moe_out
            // directly, no ACC, no host fold.
            if (dt) dt->result = per_token;
            else if (moe_out_host) per_token->data = moe_out->data;
            else {
                b.bind_fresh(per_token, (size_t)(c.d_out * c.n_t) * sizeof(float), false);
            }
        } else {
            // acc_d[d_out, n_t] += per_token tight columns, one ggml_acc per run.
            ggml_tensor * acc = nullptr;
            if (dt) {
                acc = dt->acc;
            } else {
                acc = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, c.d_out, c.n_t);
                acc->nb[0] = 4; acc->nb[1] = (size_t)(c.d_out) * 4;
                acc->data = c.acc_d.data();
            }
            const size_t col = (size_t)(c.d_out) * 4;
            for (uint32_t si = 0; si < sp.n_segs; ++si) {
                const scatter_seg_t & seg = sp.segs[si];
                if (seg.len == 0) continue;
                ggml_tensor * pv = ggml_view_2d(c.ctx, per_token, c.d_out, seg.len,
                                                per_token->nb[1], (size_t) seg.src * per_token->nb[1]);
                // nb2/nb3 must be > the 2D src1 element span: the vulkan ACC shader
                // divides by them (the CPU kernel ignores them for a 2D src1).
                ggml_tensor * a = ggml_acc_inplace(c.ctx, acc, pv,
                                                   (size_t) seg.delta * col, acc_bytes, acc_bytes,
                                                   (size_t) seg.dst * col);
                ggml_build_forward_expand(c.gf, a);
            }
        }
    }
    c.dev = nullptr;

    // Submit device graphs async (overlap with the CPU graph), run the CPU
    // graph on the calling thread, then converge: sync devices, read each acc_d
    // back via DMA, and fold every target's partial sum into moe_out.
    for (auto & kv : dev_targets) {
        device_target_t & t = kv.second;
        if (!t.gf || t.gf->n_nodes == 0) continue;
        fix_view_buffers(t.gf);
#ifdef STREAM_MOE_TEMP
        if (const char * dg = std::getenv("STREAM_MOE_TMP_DEVGRAPH_DUMP")) {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                char fn[512]; std::snprintf(fn, sizeof(fn), "%s.txt", dg);
                if (FILE * f = std::fopen(fn, "w")) {
                    std::fprintf(f, "# device graph: pool=%u n_nodes=%d\n", t.pool, t.gf->n_nodes);
                    for (int i = 0; i < t.gf->n_nodes; ++i) {
                        const ggml_tensor * nd = t.gf->nodes[i];
                        std::fprintf(f, "%3d %-16s %-44s ne=[%lld,%lld,%lld,%lld] bytes=%zu\n", i,
                                ggml_op_name(nd->op), nd->name ? nd->name : "?",
                                (long long) nd->ne[0], (long long) nd->ne[1], (long long) nd->ne[2], (long long) nd->ne[3],
                                ggml_nbytes(nd));
                    }
                    std::fclose(f);
                }
            }
        }
#endif
        int submit_n = 1;
#ifdef STREAM_MOE_TEMP
        if (const char * sn = std::getenv("STREAM_MOE_TMP_SUBMIT_N")) {
            const int v = std::atoi(sn);
            if (v > 0) submit_n = v;
        }
#endif
        for (int si = 0; si < submit_n; ++si) {
            if (ggml_backend_graph_compute_async(t.be, t.gf) != GGML_STATUS_SUCCESS) {
                LOG_ERROR("stream_moe: device graph submit failed pool " << t.pool);
                return GGML_STATUS_FAILED;
            }
        }
        // Overlap: submit async and let the CPU graph run concurrently on the
        // calling thread. Verified stable: per-layer acc and final output are
        // IDENTICAL to the serialized variant (WIP O). STREAM_MOE_TMP_NO_OVERLAP
        // serializes for debugging.
#ifdef STREAM_MOE_TEMP
        if (std::getenv("STREAM_MOE_TMP_NO_OVERLAP")) ggml_backend_synchronize(t.be);
#endif
    }
#ifdef STREAM_MOE_TEMP
    { _rt.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _ch_t0).count();
      _rt.n++; _ch_t0 = std::chrono::steady_clock::now(); }
#endif
#ifdef STREAM_MOE_TEMP
    static tmr_acc_t g_tsync{ "tail_sync" }, g_tread{ "tail_read" }, g_tfold{ "tail_fold" };
    auto _tt0 = std::chrono::steady_clock::now();
#endif
    if (gf_cpu->n_nodes > 0 &&
        ggml_backend_graph_compute(cpu, gf_cpu) != GGML_STATUS_SUCCESS) {
        LOG_ERROR("stream_moe: CPU chain graph compute failed L" << layer);
        return GGML_STATUS_FAILED;
    }
    std::vector<std::vector<float>> dev_accs(dev_targets.size());
    std::vector<const float *> accs;
    accs.reserve(1 + dev_targets.size());
    if (!direct_out) accs.push_back(c.acc_d.data());
    size_t di = 0;
    for (auto & kv : dev_targets) {
        device_target_t & t = kv.second;
#ifdef STREAM_MOE_TEMP
        if (const char * sp = std::getenv("STREAM_MOE_TMP_SYNC_SPIN")) {
            const long spin_us = std::atol(sp);
            if (spin_us > 0) {
                const auto t_spin = std::chrono::steady_clock::now();
                while (std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t_spin).count() < spin_us) {}
            }
        }
        _tt0 = std::chrono::steady_clock::now();
#endif
        ggml_backend_synchronize(t.be);
#ifdef STREAM_MOE_TEMP
        { g_tsync.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tt0).count(); g_tsync.n++; _tt0 = std::chrono::steady_clock::now(); }
#endif
        const bool single_dev_target = (dev_targets.size() == 1 && !has_cpu_round);
        bool moe_out_same_dev = false;
        if (moe_out && moe_out->buffer) {
            ggml_backend_dev_t m_dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(moe_out->buffer));
            ggml_backend_dev_t t_dev = ggml_backend_get_device(t.be);
            if (m_dev && t_dev && m_dev == t_dev) moe_out_same_dev = true;
        }
        if (direct_out) {
            // single target: the device result IS the layer output
            if (t.result) {
                if (moe_out_same_dev) {
                    ggml_backend_tensor_copy(t.result, moe_out);
                } else if (moe_out_host) {
                    ggml_backend_tensor_get(t.result, moe_out->data, 0, acc_bytes);
                } else {
                    ggml_backend_tensor_copy(t.result, moe_out);
                }
            }
        } else if (single_dev_target) {
            // Single device target with all rounds: t.acc is the complete result.
            // Avoid roundtrip through host!
            if (t.acc) {
                if (moe_out_same_dev) {
                    ggml_backend_tensor_copy(t.acc, moe_out);
                } else if (moe_out_host) {
                    ggml_backend_tensor_get(t.acc, moe_out->data, 0, acc_bytes);
                } else {
                    ggml_backend_tensor_copy(t.acc, moe_out);
                }
            }
        } else {
            dev_accs[di].assign(acc_sz, 0.0f);
#ifdef STREAM_MOE_TEMP
            if (std::getenv("STREAM_MOE_TMP_DEVDBG")) {
                std::memcpy(dev_accs[di].data(), t.arena_map + t.acc_off, acc_bytes);
            } else {
                ggml_backend_tensor_get(t.acc, dev_accs[di].data(), 0, acc_bytes);
            }
#else
            ggml_backend_tensor_get(t.acc, dev_accs[di].data(), 0, acc_bytes);
#endif
            accs.push_back(dev_accs[di].data());
        }
#ifdef STREAM_MOE_TEMP
        { g_tread.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tt0).count(); g_tread.n++; _tt0 = std::chrono::steady_clock::now(); }
#endif
        ++di;
    }
#ifdef STREAM_MOE_TEMP
    if (const char * ad = direct_out ? nullptr : std::getenv("STREAM_MOE_TMP_ACC_DUMP")) {
        char fn[512];
        snprintf(fn, sizeof(fn), "%s/acc_L%d_cpu.bin", ad, layer);
        if (FILE * f = fopen(fn, "wb")) { fwrite(c.acc_d.data(), sizeof(float), acc_sz, f); fclose(f); }
        size_t q = 0;
        for (auto & kv : dev_targets) {
            snprintf(fn, sizeof(fn), "%s/acc_L%d_p%u.bin", ad, layer, kv.first);
            if (FILE * f = fopen(fn, "wb")) { fwrite(dev_accs[q].data(), sizeof(float), acc_sz, f); fclose(f); }
            ++q;
        }
    }
#endif
#ifdef STREAM_MOE_TEMP
    if (!direct_out && std::getenv("STREAM_MOE_TMP_DEVDBG")) {
        auto nrm = [](const std::vector<float> & v) {
            double s = 0; for (float x : v) s += (double) x * x; return std::sqrt(s);
        };
        fprintf(stderr, "[devdbg] L%d d_out=%lld n_t=%u cpu_nodes=%d devs=%zu cpu_acc=%.5g\n",
                layer, (long long) d_out, n_t, gf_cpu->n_nodes, dev_targets.size(), nrm(c.acc_d));
        for (auto & kv : dev_targets) {
            double sa = 0;
            if (kv.second.arena_map) {
                const float * am = (const float *) kv.second.arena_map;
                for (int i = 0; i < 262144; ++i) sa += (double) am[i] * am[i];
            }
            int n_acc = 0;
            if (kv.second.gf) {
                for (int i = 0; i < kv.second.gf->n_nodes; ++i) {
                    if (kv.second.gf->nodes[i]->op == GGML_OP_ACC) ++n_acc;
                }
            }
            fprintf(stderr, "[devdbg]   pool%u gf_nodes=%d acc_nodes=%d arena[0..1MB]=%.5g\n",
                    kv.first, kv.second.gf ? kv.second.gf->n_nodes : -1, n_acc, std::sqrt(sa));
            if (layer == 0 && kv.second.gf) {
                for (int i = kv.second.gf->n_nodes - 4; i < kv.second.gf->n_nodes; ++i) {
                    if (i < 0) continue;
                    ggml_tensor * n = kv.second.gf->nodes[i];
                    fprintf(stderr, "[devdbg]     node[%d] op=%s ne=[%lld,%lld] buf=%p data=%p\n",
                            i, ggml_op_name(n->op), (long long) n->ne[0], (long long) n->ne[1],
                            (void *) n->buffer, n->data);
                }
            }
        }
        for (size_t q = 0; q < dev_accs.size(); ++q) {
            fprintf(stderr, "[devdbg]   dev%zu acc=%.5g\n", q, nrm(dev_accs[q]));
        }
    }
#endif
#ifdef STREAM_MOE_TEMP
    _tt0 = std::chrono::steady_clock::now();
#endif
    const bool single_dev_target = (dev_targets.size() == 1 && !has_cpu_round);
    if (!direct_out && !single_dev_target) {
        if (!layer_fold(ex, d_out, n_t, accs)) return GGML_STATUS_FAILED;
    }
#ifdef STREAM_MOE_TEMP
    { g_tfold.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tt0).count(); g_tfold.n++; }
#endif
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
#ifdef STREAM_MOE_TEMP
    { _tt.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _ch_t0).count(); _tt.n++; }
#endif
    return GGML_STATUS_SUCCESS;
}

// Run this layer's cross-device transfers for one stage (docs/PER_DEVICE_ARENA.md
// SS8.3): ggml_backend_tensor_copy(src, dst) refreshes the consumer-side copy
// before the stage reads it. Synchronous - the consumer reads right after.
static void run_layer_xfers(int32_t layer, int stage) {
    for (const auto & r : route_b_relays()) {
        if (r.layer != layer || r.stage != stage || !r.src || !r.dst) continue;
        if (r.src->buffer) {
            ggml_backend_tensor_copy(r.src, r.dst);
        } else if (r.src->data) {
            // bufferless host leaf (positions / freq factors / mask): upload the
            // host bytes into the device shell.
            ggml_backend_tensor_set(r.dst, r.src->data, 0, ggml_nbytes(r.src));
        }
    }
}

// Burst one whole layer from its captured sequence.
static enum ggml_status exec_layer_burst(int32_t layer, ggml_context * ctx,
                                         ggml_backend_t cpu, expert_scheduler& sched,
                                         int /*n_threads*/) {
    const moe_layer_exec_t * ex = moe_chain_layer_exec(layer);
    if (!ex || ex->compute.empty()) return GGML_STATUS_SUCCESS;
#ifdef STREAM_MOE_TEMP
    if (layer == 0 && std::getenv("STREAM_MOE_TMP_BIN_DIR")) route_b_begin_ubatch();
#endif
#ifdef STREAM_MOE_TEMP
    // Debug: count ubatches (one per exec_layer_burst(layer=0)); dump all layers
    // of the 3rd token then stop.
    static int g_ubatch = 0;
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG") && layer == 0) {
        if (++g_ubatch > 3) { std::fflush(stderr); std::exit(0); }
    }
#endif
#ifdef STREAM_MOE_TEMP
    int64_t _n_t = 0;
    for (const auto * cn : ex->compute)
        if (cn && cn->op == GGML_OP_MUL_MAT_ID && cn->src[2]) { _n_t = cn->src[2]->ne[1]; break; }
    static tmr_acc_t g_burst_dec{ "burst_dec" }, g_burst_pre{ "burst_pre" },
                      g_pin_dec{ "pin_dec" }, g_pin_pre{ "pin_pre" },
                      g_unpin_dec{ "unpin_dec" }, g_unpin_pre{ "unpin_pre" };
    tmr_acc_t & _bt = (_n_t == 1) ? g_burst_dec : g_burst_pre;
    tmr_acc_t & _pt = (_n_t == 1) ? g_pin_dec : g_pin_pre;
    tmr_acc_t & _ut = (_n_t == 1) ? g_unpin_dec : g_unpin_pre;
    auto _burst_t0 = std::chrono::steady_clock::now();
    struct _burst_guard_t { tmr_acc_t & a; std::chrono::steady_clock::time_point t0;
        _burst_guard_t(tmr_acc_t & a_, std::chrono::steady_clock::time_point t0_) : a(a_), t0(t0_) {}
        ~_burst_guard_t() { a.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); a.n++; } } _burst_guard(_bt, _burst_t0);
#endif
    const moe_model_topology_t& topo = sched.topology();
#ifdef STREAM_MOE_TEMP
    // TEMP pre-warm (STREAM_MOE_TMP_PREWARM=1): pin+unpin every expert of every
    // layer once so the pool is fully resident before the timed work.
    if (std::getenv("STREAM_MOE_TMP_PREWARM")) {
        static bool prewarmed = false;
        if (!prewarmed) {
            prewarmed = true;
            size_t total = 0;
            for (uint32_t L = 0; L < topo.n_layer; ++L) {
                uint64_t all[BITMAP_WORDS] = { 0 };
                for (uint32_t e = 0; e < topo.n_expert; ++e) expert_scheduler::bit_set(all, e);
                batch_await_t aw;
                std::vector<expert_handle_t> pp(topo.n_expert);
                const int32_t np = sched.pin_layer(L, all, aw, pp.data(), static_cast<uint32_t>(pp.size()));
                if (np > 0) { for (int32_t i = 0; i < np; ++i) sched.unpin(pp[i]); total += static_cast<size_t>(np); }
            }
            if (std::getenv("STREAM_MOE_TMR"))
                std::fprintf(stderr, "[TMR] prewarm pinned %zu experts\n", total);
        }
    }
#endif
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
        const ggml_tensor * src = lt->src[0];
        mut->data = lt->op == GGML_OP_VIEW
                    ? static_cast<char*>(src->data) + lt->view_offs
                    : src->data;
        // A device backend resolves the memory via the owning buffer, so the
        // layout tensor must carry its source's buffer (CPU reads data directly).
        if (src->buffer) mut->buffer = src->buffer;
    }

    // L2 whole-layer ownership: run the dense head (layer nodes not in the MoE
    // closure and not downstream of ffn_moe_out) so cur/ids/weights are
    // materialised before pinning. The dense tail (downstream of ffn_moe_out)
    // runs after the burst. Layers without a captured whole-layer list keep the
    // MoE-only path (device dense).
    std::vector<ggml_tensor*> dense_head, dense_tail;
    const std::vector<ggml_tensor*> * lns = moe_chain_layer_nodes(layer);
    // Build-time plan (docs/LAYER_EXECUTOR_DESIGN.md 4.1): head/tail are computed
    // in moe_chain_assign_backend; the executor only consumes them.
    if (const moe_layer_plan_t * plan = moe_chain_layer_plan(layer)) {
        dense_head = plan->head;
        dense_tail = plan->tail;
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG") && layer == 0) {
        fprintf(stderr, "[split] L%d lns=%zu head=%zu tail=%zu closure=%zu\n", layer,
                lns ? lns->size() : 0, dense_head.size(), dense_tail.size(),
                ex ? ex->compute.size() : 0);
        for (auto * nd : dense_head) fprintf(stderr, "[split]  H %s\n", nd->name ? nd->name : "?");
        for (auto * nd : dense_tail) fprintf(stderr, "[split]  T %s\n", nd->name ? nd->name : "?");
        if (ex) for (auto * nd : ex->compute) fprintf(stderr, "[split]  C %s\n", nd->name ? nd->name : "?");
    }
#endif
    // Cross-device transfers (docs/PER_DEVICE_ARENA.md SS8.3): bring this layer's
    // incoming carry copies onto the layer's device before the head reads them.
    // The consumers were already rewired to the local copy by layout_arena.
    run_layer_xfers(layer, ROUTE_B_XFER_LAYER_FRONT);
    {
        const enum ggml_status hst = run_dense_by_device(ctx, cpu, dense_head, layer, "head");
        if (hst != GGML_STATUS_SUCCESS) { LOG_ERROR("stream_moe: dense head failed L" << layer); return hst; }
    }
#ifdef STREAM_MOE_TEMP
    dump_node_hash(layer, "head", dense_head);
#endif
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG") && layer == 0 && ex) {
        // Backend-agnostic value read (a device tensor's data is a fake offset).
        auto read3f = [](const ggml_tensor * t, float v[3]) {
            v[0] = v[1] = v[2] = 0.0f;
            if (!t || !t->data) return;
            float buf[4] = { 0, 0, 0, 0 };
            const size_t n = std::min<size_t>(3, (size_t) ggml_nelements(t));
            tensor_read_host(t, buf, 0, n * sizeof(float));
            v[0] = buf[0]; v[1] = buf[1]; v[2] = buf[2];
        };
        for (const auto * cn : ex->compute) {
            if (cn && cn->op == GGML_OP_MUL_MAT_ID && cn->src[1] && cn->src[1]->data) {
                float v[3]; read3f(cn->src[1], v);
                const ggml_tensor * vs = cn->src[1]->view_src;
                fprintf(stderr, "[cur] %s data=%p off=%zu view_src=%s(%p) v=%.6f %.6f %.6f\n",
                        cn->src[1]->name ? cn->src[1]->name : "?", cn->src[1]->data,
                        cn->src[1]->view_offs,
                        vs ? (vs->name ? vs->name : "?") : "-", vs ? vs->data : nullptr,
                        v[0], v[1], v[2]);
                break;
            }
        }
        for (auto * nd : dense_head) {
            if (nd->name && strncmp(nd->name, "norm-0", 6) == 0 && nd->op == GGML_OP_RMS_NORM &&
                nd->src[0] && nd->src[0]->data) {
                float v[3]; read3f(nd->src[0], v);
                fprintf(stderr, "[embd] %s %.6f %.6f %.6f\n",
                        nd->src[0]->name ? nd->src[0]->name : "?", v[0], v[1], v[2]);
                break;
            }
        }
        for (const auto & el : ex->external_leaves) {
            const ggml_tensor * t = el.tensor;
            if (!t || !t->data) continue;
            if (t->type == GGML_TYPE_F32) {
                float v[3]; read3f(t, v);
                fprintf(stderr, "[ext] %-6s %-26s f32 %.6f %.6f %.6f\n",
                        el.role ? el.role : "?", t->name ? t->name : "?", v[0], v[1], v[2]);
            } else if (t->type == GGML_TYPE_I32) {
                int32_t buf[4] = { 0, 0, 0, 0 };
                tensor_read_host(t, buf, 0, std::min<size_t>(3, (size_t) ggml_nelements(t)) * sizeof(int32_t));
                fprintf(stderr, "[ext] %-6s %-26s i32 %d %d %d\n",
                        el.role ? el.role : "?", t->name ? t->name : "?", buf[0], buf[1], buf[2]);
            }
        }
    }
#endif

    // Find the single routing ids tensor for this MoE layer.
    const ggml_tensor * ids = nullptr;
    for (const auto * cn : ex->compute) {
        if (cn && cn->op == GGML_OP_MUL_MAT_ID && cn->src[2]) { ids = cn->src[2]; break; }
    }
    if (!ids) return GGML_STATUS_FAILED;
    const uint32_t n_k = static_cast<uint32_t>(ids->ne[0]);
    const uint32_t n_t = static_cast<uint32_t>(ids->ne[1]);
    // Empty chain: 0-token no-op.
    if (n_t == 0 || n_k == 0) return GGML_STATUS_SUCCESS;
    if (!ids->data) return GGML_STATUS_FAILED;

    // Single D2H readback of routing ids per layer (iron rule: avoid redundant transfers).
    std::vector<uint8_t> ids_host;
    const uint8_t * ids_bytes = host_image(ids, ids_host);
    std::vector<int32_t> ids_compact((size_t) n_k * n_t, 0);
    for (uint32_t t = 0; t < n_t; ++t) {
        for (uint32_t k = 0; k < n_k; ++k) {
            ids_compact[(size_t) t * n_k + k] = moe_id_at(ids_bytes, ids, (int) t, (int) k);
        }
    }

    // Pin the layer's whole active expert set (all mm nodes share the ids).
    // Batch semantics: ONE request carrying the whole layer's expert bitmap;
    // missing experts load concurrently (IOCP n-way), exec wakes once.
    std::vector<keyed_expert_t> keys;
    for (int32_t e : ids_compact) {
        if (e >= 0 && e < static_cast<int32_t>(topo.n_expert)) {
            bool exists = false;
            for (const auto & x : keys) {
                if (x.expert == static_cast<uint32_t>(e)) { exists = true; break; }
            }
            if (!exists) keys.push_back({ static_cast<uint32_t>(layer), static_cast<uint32_t>(e) });
        }
    }
    // All keys belong to `layer` (burst is per-layer); build the needed bitmap.
    if (!keys.empty() && keys[0].layer != static_cast<uint32_t>(layer)) {
        LOG_ERROR("stream_moe: burst keys layer mismatch (" << keys[0].layer << " vs " << layer << ")");
        return GGML_STATUS_FAILED;
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_CHAIN_DEBUG"))
        fprintf(stderr, "[burst] L%d: keys=%zu n_expert=%u\n",
                layer, keys.size(), topo.n_expert);
#endif
    uint64_t needed[BITMAP_WORDS] = { 0 };
    for (const auto & k : keys) expert_scheduler::bit_set(needed, k.expert);
    batch_await_t await;
    std::vector<expert_handle_t> pins(keys.size());
#ifdef STREAM_MOE_TEMP
    auto _pin_t0 = std::chrono::steady_clock::now();
#endif
    const int32_t np = sched.pin_layer(static_cast<uint32_t>(layer), needed, await,
                                       pins.data(), static_cast<uint32_t>(pins.size()));
#ifdef STREAM_MOE_TEMP
    { _pt.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _pin_t0).count(); _pt.n++; }
#endif
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
    run_layer_xfers(layer, ROUTE_B_XFER_CLOSURE);   // cur on the closure's device
    const enum ggml_status st = exec_layer_burst_chain_buckets(layer, ctx, cpu, sched, ex, pins, ids, n_k, n_t, ids_compact);
#ifdef STREAM_MOE_TEMP
    dump_node_hash(layer, "moe", dense_head);   // head nodes still live at the tail?
#endif
    // L2 whole-layer ownership: run the dense tail (residual / post-norm / dense
    // MLP) after moe_out is materialised.
    if (st == GGML_STATUS_SUCCESS && !dense_tail.empty()) {
        run_layer_xfers(layer, ROUTE_B_XFER_TAIL);   // moe_out on the tail's device
        const enum ggml_status tst = run_dense_by_device(ctx, cpu, dense_tail, layer, "tail");
        if (tst != GGML_STATUS_SUCCESS) { LOG_ERROR("stream_moe: dense tail failed L" << layer); return tst; }
    }
#ifdef STREAM_MOE_TEMP
    dump_node_hash(layer, "tail", dense_tail);
#endif
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG") && layer >= 0) {
        const std::vector<ggml_tensor*> * all = moe_chain_layer_nodes_all(layer);
        if (all) {
            std::unordered_map<const ggml_tensor*, char> nodeset;
            for (auto * t : *all) if (t) nodeset[t] = 1;
            // Dump a tensor (backend-agnostic: a device tensor's data is a fake
            // offset pointer, never dereference it on the host). Name carries the
            // layer + node index (+ src slot) so dumps are unique and ordered.
            auto dump_one = [&](const ggml_tensor * t, const char * tag) {
                if (!t || !t->data) return;
                const size_t nb = ggml_nbytes(t);
                std::vector<uint8_t> host(nb);
                tensor_read_host(t, host.data(), 0, nb);
                char nm[160];
                std::snprintf(nm, sizeof(nm), "L%d_%s_%s", layer, tag, t->name ? t->name : "?");
                route_b_dump_node_bin(layer, nm, ggml_op_name(t->op), (int) t->type,
                                      t->ne[0], t->ne[1], host.data(), nb);
                uint64_t h = 1469598103934665603ull;
                for (size_t bi = 0; bi < nb; ++bi) { h ^= host[bi]; h *= 1099511628211ull; }
                double v0 = 0.0;
                if (t->type == GGML_TYPE_F32) v0 = *(const float *) host.data();
                else if (t->type == GGML_TYPE_I32) v0 = (double) *(const int32_t *) host.data();
                fprintf(stderr, "[node] L%d %-40s %-12s %016llx v0=%.6g\n", layer,
                        nm, ggml_op_name(t->op), (unsigned long long) h, v0);
            };
            for (size_t ni = 0; ni < all->size(); ++ni) {
                auto * nd = (*all)[ni];
                if (!nd || is_alias_op(nd) || !nd->data) continue;
                char tag[32]; std::snprintf(tag, sizeof(tag), "%03zu", ni);
                dump_one(nd, tag);
                // Inputs that are not layer nodes (weights / leaves): dump them too
                // so a node's operands can be compared offline.
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * src = nd->src[s];
                    if (!src || !src->data || nodeset.count(src)) continue;
                    char stag[40]; std::snprintf(stag, sizeof(stag), "%03zu_src%d", ni, s);
                    dump_one(src, stag);
                }
            }
        }
    }
#endif
#ifdef STREAM_MOE_TEMP
    auto _unpin_t0 = std::chrono::steady_clock::now();
#endif
    for (const auto & h : pins) sched.unpin(h);
#ifdef STREAM_MOE_TEMP
    { _ut.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _unpin_t0).count(); _ut.n++; }
#endif
    return st;
}

// graph_compute entry: dispatch to the whole-layer burst on the layer's first
// privatised node; later same-layer splits are no-ops (already produced by the
// burst). An un-captured MUL_MAT_ID (layer < 0) is a hard error - the legacy
// per-split path is gone.
// Whole-layer ownership entry (docs/ROUTE_B_LAYER_OWNERSHIP.md L2): the split
// may span many layers (all layer nodes are our backend -> one split for the
// whole graph). Run every layer present, once, in first-seen order. A view
// split is a no-op (its data pointers were fixed by the burst); an un-captured
// routed MUL_MAT_ID is a hard error.
enum ggml_status moe_exec_mul_mat_id(
    ggml_cgraph* cgraph,
    ggml_context* ctx,
    ggml_backend_t cpu_backend,
    expert_scheduler& sched,
    int n_threads)
{
    const ggml_tensor* const* nodes = cgraph->nodes;
    const int n_nodes = cgraph->n_nodes;
    if (n_nodes == 0) return GGML_STATUS_SUCCESS;
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG")) {
        int in = 0, shown = 0;
        for (int i = 0; i < n_nodes; ++i)
            if (nodes[i] && nodes[i]->data && route_b_in_arena(nodes[i]->data)) ++in;
        fprintf(stderr, "[arena_check] n_nodes=%d in_arena=%d (nodes[0]='%s' data=%p)\n",
                n_nodes, in, (nodes[0] && nodes[0]->name) ? nodes[0]->name : "?",
                nodes[0] ? nodes[0]->data : nullptr);
        for (int i = 0; i < n_nodes && shown < 12; ++i) {
            const ggml_tensor * nd = nodes[i];
            if (!nd || (nd->data && route_b_in_arena(nd->data))) continue;
            fprintf(stderr, "[arena_check] NOT-IN '%s' op=%s data=%p buf=%p\n",
                    nd->name ? nd->name : "(anon)", ggml_op_name(nd->op), nd->data,
                    (void*)(nd->buffer ? nd->buffer : nullptr));
            ++shown;
        }
    }
#endif

    std::vector<int32_t> layers;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * nd = nodes[i];
        if (!nd || is_alias_op(nd)) continue;
        const int32_t L = moe_chain_layer_of_node(nd);
        if (L < 0) {
            // An un-captured routed MUL_MAT_ID is a hard error: the legacy
            // per-split path is gone, so silently skipping it would leave the
            // layer's expert output uncomputed.
            if (nd->op == GGML_OP_MUL_MAT_ID) {
                fprintf(stderr,
                        "[stream_moe] un-captured MUL_MAT_ID: node='%s' w='%s' op=%s n_nodes=%d\n",
                        nd->name ? nd->name : "(anon)",
                        (nd->src[0] && nd->src[0]->name) ? nd->src[0]->name : "?",
                        ggml_op_name(nd->op), n_nodes);
                LOG_ERROR("stream_moe: un-captured MUL_MAT_ID split reached the executor (no legacy path)");
                return GGML_STATUS_FAILED;
            }
            continue;   // not a layer node (external / view / model I/O)
        }
        bool seen = false;
        for (int32_t x : layers) if (x == L) { seen = true; break; }
        if (!seen) layers.push_back(L);
    }
#ifdef STREAM_MOE_TEMP
    if (std::getenv("STREAM_MOE_TMP_DENSE_DEBUG")) {
        fprintf(stderr, "[exec] n_nodes=%d first=%s layers=", n_nodes,
                (nodes[0] && nodes[0]->name) ? nodes[0]->name : "?");
        for (int32_t L : layers) fprintf(stderr, "%d ", L);
        fprintf(stderr, "\n");
    }
#endif
    // Layer execution state for THIS graph_compute invocation
    // (docs/LAYER_EXECUTOR_DESIGN.md 4.4): each layer runs exactly once. The
    // state is per-call, not per-build: llama reuses the built graph across
    // decodes (llama-context.cpp:1379) and does not call moe_chain_assign_backend
    // again, so a state kept on the build-time plan would stay COMPLETE and skip
    // every later decode.
    enum class layer_state : uint8_t { NOT_STARTED, RUNNING, COMPLETE };
    std::unordered_map<int32_t, layer_state> exec_state;
    exec_state.reserve(layers.size() * 2 + 8);
    for (int32_t L : layers) {
        // Whole-layer ownership: the layer's nodes are one split, so the layer is
        // executed on first sight - correctness does not depend on a "first
        // node" happening to land in this split (L2 review A2).
        const moe_layer_plan_t * plan = moe_chain_layer_plan(L);
        if (!plan) {
            // MoE-only: the closure may span several splits. Burst the whole
            // layer only from the split that holds its FIRST node; later
            // same-layer splits are no-ops (the burst already produced them).
            const std::vector<ggml_tensor*> * ln = moe_chain_layer_nodes(L);
            const moe_layer_exec_t * ex = moe_chain_layer_exec(L);
            // First non-alias node: aliases are never assigned to our backend,
            // so they cannot appear in the split and would make has_first false.
            const ggml_tensor * first_node = nullptr;
            if (ln) for (auto * t : *ln) if (!is_alias_op(t)) { first_node = t; break; }
            if (!first_node && ex)
                for (auto * t : ex->compute) if (!is_alias_op(t)) { first_node = t; break; }
            if (!first_node) continue;
            bool has_first = false;
            for (int i = 0; i < n_nodes && !has_first; ++i) if (nodes[i] == first_node) has_first = true;
            if (!has_first) continue;
        }
        const auto sit = exec_state.find(L);
        if (sit != exec_state.end()) {
            if (sit->second == layer_state::COMPLETE) continue;
            LOG_ERROR("stream_moe: layer " << L << " re-entered while RUNNING");
            return GGML_STATUS_FAILED;
        }
        exec_state[L] = layer_state::RUNNING;
        const enum ggml_status st = exec_layer_burst(L, ctx, cpu_backend, sched, n_threads);
        if (st != GGML_STATUS_SUCCESS) return st;
        exec_state[L] = layer_state::COMPLETE;
    }
    return GGML_STATUS_SUCCESS;
}

} // namespace stream_moe
