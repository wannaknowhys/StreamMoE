#include "backend/moe_backend.h"
#include "backend/minigraph.h"
#include "backend/minigraph_exec.h"
#include "backend/scheduler.h"
#include "common/logger.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ggml-vulkan route-B extension: host mapping of a host-visible buffer.
void* stmoe_vk_buffer_host_ptr(ggml_backend_buffer_t buffer);
#include <string>
#include <vector>

namespace stream_moe {

namespace {

// ---- shared helpers ------------------------------------------------------

// Expert weight buft / buffer ---------------------------------------------

struct expert_buft_ctx {
    ggml_backend_dev_t dev = nullptr;
    char name[32] = {};
};

struct expert_buf_ctx {
    size_t size = 0;
};

const char* expert_buft_get_name(ggml_backend_buffer_type_t buft) {
    return static_cast<expert_buft_ctx*>(buft->context)->name;
}

ggml_backend_buffer_t expert_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* ctx = static_cast<expert_buf_ctx*>(std::calloc(1, sizeof(expert_buf_ctx)));
    // size is pure bookkeeping: gallocr pads each tensor to the buft alignment, so
    // keep a generous margin to satisfy the bounds assert without allocating.
    ctx->size = size + 16u * 1024u * 1024u;
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = [](ggml_backend_buffer_t b) { std::free(b->context); };
    // get_base: non-null sentinel (assert in ggml_backend_buffer_get_base);
    // never dereferenced - our graph_compute reads slots, not tensor->data.
    iface.get_base = [](ggml_backend_buffer_t) -> void* { static char sentinel[8]; return sentinel; };
    iface.memset_tensor = [](ggml_backend_buffer_t, ggml_tensor*, uint8_t, size_t, size_t) {};
    iface.set_tensor    = [](ggml_backend_buffer_t, ggml_tensor*, const void*, size_t, size_t) {};
    iface.get_tensor    = [](ggml_backend_buffer_t, const ggml_tensor*, void*, size_t, size_t) {};
    iface.clear = [](ggml_backend_buffer_t, uint8_t) {};
    return ggml_backend_buffer_init(buft, iface, ctx, ctx->size);
}

size_t expert_buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }

// One expert weight buft per model pool (multi-model support). Never static:
// each call allocates a fresh buft so buft identity == model identity.
ggml_backend_buffer_type_t make_expert_buft(ggml_backend_dev_t dev, int ordinal) {
    auto* buft = static_cast<ggml_backend_buffer_type*>(std::calloc(1, sizeof(ggml_backend_buffer_type)));
    auto* ctx  = new expert_buft_ctx();
    ctx->dev = dev;
    std::snprintf(ctx->name, sizeof(ctx->name), "STREAMMOE_EXPERT%s", ordinal > 0 ? std::to_string(ordinal).c_str() : "");
    buft->iface.get_name       = expert_buft_get_name;
    buft->iface.alloc_buffer   = expert_buft_alloc_buffer;
    buft->iface.get_alignment  = expert_buft_get_alignment;
    buft->iface.is_host        = [](ggml_backend_buffer_type_t) { return true; };
    buft->device               = dev;
    buft->context              = ctx;
    return buft;
}

// Compute (host) buft: plain malloc-backed buffer for node outputs ---------

struct host_buf_ctx {
    void* ptr = nullptr;
};

ggml_backend_buffer_t host_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* ctx = static_cast<host_buf_ctx*>(std::calloc(1, sizeof(host_buf_ctx)));
    if (size > 0) {
        ctx->ptr = aligned_alloc_ptr(size, 64);
        if (!ctx->ptr) { std::free(ctx); return nullptr; }
    }
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = [](ggml_backend_buffer_t b) {
        auto* c = static_cast<host_buf_ctx*>(b->context);
        aligned_free_ptr(c->ptr);
        std::free(c);
    };
    iface.get_base = [](ggml_backend_buffer_t b) -> void* {
        return static_cast<host_buf_ctx*>(b->context)->ptr;
    };
    iface.memset_tensor = [](ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t o, size_t s) {
        std::memset(reinterpret_cast<char*>(t->data) + o, v, s);
    };
    iface.set_tensor = [](ggml_backend_buffer_t, ggml_tensor* t, const void* d, size_t o, size_t s) {
        std::memcpy(reinterpret_cast<char*>(t->data) + o, d, s);
    };
    iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor* t, void* d, size_t o, size_t s) {
        std::memcpy(d, reinterpret_cast<const char*>(t->data) + o, s);
    };
    iface.clear = [](ggml_backend_buffer_t b, uint8_t v) {
        auto* c = static_cast<host_buf_ctx*>(b->context);
        if (c->ptr) std::memset(c->ptr, v, b->size);
    };
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

ggml_backend_buffer_type_t make_host_buft(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buft = {};
    static bool inited = false;
    if (!inited) {
        buft.iface.get_name      = [](ggml_backend_buffer_type_t) { return "STREAMMOE_HOST"; };
        buft.iface.alloc_buffer  = host_buft_alloc_buffer;
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return size_t(64); };
        buft.iface.is_host       = [](ggml_backend_buffer_type_t) { return true; };
        buft.device              = dev;
        inited = true;
    }
    return &buft;
}

// ---- backend / device ----------------------------------------------------

struct moe_dev_ctx {
    std::vector<ggml_backend_buffer_type_t> expert_bufts; // one per model pool
    ggml_backend_buffer_type_t host_buft = nullptr;
};

struct moe_backend_ctx {
    scratch_arena arena;
    ggml_backend_t cpu = nullptr; // stock CPU backend for mini-graph delegation
};

// Model pools bind their expert buft to a scheduler; graph_compute resolves
// the owning scheduler by the weight buft of the graph's MoE nodes.
struct buft_sched_binding {
    ggml_backend_buffer_type_t buft  = nullptr;
    expert_scheduler*          sched = nullptr;
};
std::vector<buft_sched_binding> g_bindings;
int g_threads = 1;

static size_t estimate_scratch(const ggml_tensor* const* nodes, int n_nodes) {
    size_t need = 16 * 1024 * 1024; // base + graph/overhead margin
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* nd = nodes[i];
        if (!nd || nd->op != GGML_OP_MUL_MAT_ID) continue;
        const ggml_tensor* w = nd->src[0];
        // leaf-only mini-graph: w3d reserves one expert slice, ids_slot small,
        // b_leaf small, result = dst bytes
        need += ggml_row_size(w->type, w->ne[0]) * w->ne[1];
        need += ggml_nbytes(nd);          // result written into dst
        need += 4 * ggml_tensor_overhead() * 8;
        need += 1 * 1024 * 1024;
        // Mixed-region split (J6): per-pool peel rounds each rebuild their own
        // ids_leaf + cur leaf ([d_in, width, n_active]) inside THIS graph_compute
        // call, on top of the single-mm budget above. Those leaves are NOT graph
        // nodes, so reserve per-token scratch: cur = d_in x n_t x 4 (worst case
        // a round is one token deep); ids = n_t x n_k x 4. Use src[2] (ids) ne
        // for the token count when available.
        const ggml_tensor* ids = nd->src[2];
        const int64_t n_tok = ids ? ids->ne[1] : 1024;
        need += static_cast<size_t>(w->ne[0]) * static_cast<size_t>(n_tok) * 4; // cur rebuild
        need += static_cast<size_t>(n_tok) * static_cast<size_t>(ids ? ids->ne[0] : 16) * 4; // ids rebuild
        need += 16 * 1024 * 1024;          // round graphs + mm result staging margin
    }
    return need * 2;
}

const char* moe_dev_get_name(ggml_backend_dev_t) { return "STREAMMOE"; }
const char* moe_dev_get_desc(ggml_backend_dev_t) {
    return "StreamMoE expert-pool backend (delegates MoE GEMM to CPU/GPU)";
}

void moe_dev_get_mem(ggml_backend_dev_t, size_t* free, size_t* total) {
    // Pool budget is managed by the scheduler; report nothing here.
    *free = 0; *total = 0;
}

enum ggml_backend_dev_type moe_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void moe_dev_get_props(ggml_backend_dev_t, ggml_backend_dev_props* props) {
    std::memset(props, 0, sizeof(*props));
    props->name = "STREAMMOE";
    props->description = "StreamMoE expert pool";
}

ggml_backend_t moe_dev_init_backend(ggml_backend_dev_t dev, const char*) {
    static ggml_guid guid = { 0x4d, 0x4f, 0x45, 0x53, 0x74, 0x72, 0x65, 0x61,
                              0x6d, 0x4d, 0x6f, 0x45, 0x42, 0x61, 0x63, 0x6b };
    auto* backend = static_cast<ggml_backend*>(std::calloc(1, sizeof(ggml_backend)));
    std::memcpy(&backend->guid, &guid, sizeof(ggml_guid_t));
    backend->device = dev;
    backend->context = new moe_backend_ctx();
    static_cast<moe_backend_ctx*>(backend->context)->cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (static_cast<moe_backend_ctx*>(backend->context)->cpu) {
        ggml_backend_cpu_set_n_threads(static_cast<moe_backend_ctx*>(backend->context)->cpu, g_threads);
    }
    backend->iface.get_name = [](ggml_backend_t) { return "STREAMMOE"; };
    backend->iface.free     = [](ggml_backend_t b) {
        auto* c = static_cast<moe_backend_ctx*>(b->context);
        if (c->cpu) ggml_backend_free(c->cpu);
        delete c;
        std::free(b);
    };
    backend->iface.synchronize = [](ggml_backend_t) {};
    backend->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph* cgraph) -> enum ggml_status {
        auto* bctx = static_cast<moe_backend_ctx*>(b->context);
        // This backend now receives whole privatised MoE splits: splits with
        // routed MUL_MAT_ID nodes AND weightless chain-compute splits (no mm -
        // the mm split ran first, srcs are already materialised). No early-out.
        // resolve the owning scheduler: by the weight buft of this graph's nodes
        // (each model pool has its own expert buft - see MULTI_MODEL_POOL.md),
        // falling back to the sole binding when this split carries no weights.
        expert_scheduler* sched = nullptr;
        for (int i = 0; i < cgraph->n_nodes && !sched; ++i) {
            const ggml_tensor* nd = cgraph->nodes[i];
            if (!nd->src[0] || !nd->src[0]->buffer || !nd->src[0]->buffer->buft) continue;
            ggml_backend_buffer_type_t buft = nd->src[0]->buffer->buft;
            for (const auto& bnd : g_bindings) {
                if (bnd.buft == buft) { sched = bnd.sched; break; }
            }
        }
        if (!sched && g_bindings.size() == 1) {
            sched = g_bindings[0].sched;   // single-pool: weightless chain split
        }
        if (!sched) {
            LOG_ERROR("stream_moe: graph_compute without a scheduler (expert backend not wired)");
            return GGML_STATUS_FAILED;
        }
        // collect our nodes (weights in our buft)
        std::vector<const ggml_tensor*> nodes;
        nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) nodes.push_back(cgraph->nodes[i]);

        if (!bctx->arena.ensure(estimate_scratch(nodes.data(), static_cast<int>(nodes.size())))) {
            LOG_ERROR("stream_moe: scratch arena alloc failed");
            return GGML_STATUS_FAILED;
        }
        if (!bctx->arena.reset()) return GGML_STATUS_FAILED;

        return moe_exec_mul_mat_id(bctx->arena.ctx(), bctx->cpu, *sched,
                                   nodes.data(), static_cast<int>(nodes.size()), g_threads);
    };
    return backend;
}

ggml_backend_buffer_type_t moe_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return static_cast<moe_dev_ctx*>(dev->context)->host_buft;
}

bool moe_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto* ctx = static_cast<moe_dev_ctx*>(dev->context);
    if (buft == ctx->host_buft) return true;
    for (const auto& eb : ctx->expert_bufts) {
        if (buft == eb) return true;
    }
    return false;
}

bool moe_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor* op) {
    // Take ownership ONLY of routed-expert MUL_MAT_ID (`*_exps`). Everything
    // else (dense, shared `*_shexp`, embeddings, ...) must stay on llama.cpp
    // defaults - otherwise the loader would select our ACCEL buft for them.
    if (!op || !op->src[0] || !op->src[0]->name) return false;
    const char* n = op->src[0]->name;
    if (!n[0]) return false;
    if (op->op == GGML_OP_MUL_MAT_ID) {
        return std::strstr(n, "_exps") != nullptr;
    }
    // View / layout ops whose source lives on our host compute buffer (e.g.
    // gemma4 slices the fused gate_up MUL_MAT_ID output). These are pure views
    // (no weight read) - the delegate executes them on the real buffer.
    if (op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE ||
        op->op == GGML_OP_CONT) {
        auto* ctx = static_cast<moe_dev_ctx*>(dev->context);
        return op->src[0]->buffer && op->src[0]->buffer->buft == ctx->host_buft;
    }
    return false;
}

bool moe_dev_offload_op(ggml_backend_dev_t, const ggml_tensor*) { return false; }

// ---- registration ---------------------------------------------------------

struct moe_reg_ctx {
    ggml_backend_device device;
    ggml_backend_reg reg;
    moe_dev_ctx dev_ctx;
};

size_t moe_reg_dev_count(ggml_backend_reg_t) { return 1; }

ggml_backend_dev_t moe_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto* ctx = static_cast<moe_reg_ctx*>(reg->context);
    return index == 0 ? &ctx->device : nullptr;
}

} // namespace

ggml_backend_buffer_type_t stream_moe_compute_buft() {
    return stream_moe_register_backend_helper_compute_buft();
}

void stream_moe_register_backend() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    static moe_reg_ctx ctx;
    std::memset(&ctx, 0, sizeof(ctx));

    // device
    ctx.device.reg = &ctx.reg;
    ctx.device.context = &ctx.dev_ctx;
    ctx.device.iface.get_name          = moe_dev_get_name;
    ctx.device.iface.get_description   = moe_dev_get_desc;
    ctx.device.iface.get_memory        = moe_dev_get_mem;
    ctx.device.iface.get_type          = moe_dev_get_type;
    ctx.device.iface.get_props         = moe_dev_get_props;
    ctx.device.iface.init_backend      = moe_dev_init_backend;
    ctx.device.iface.get_buffer_type   = moe_dev_get_buffer_type;
    ctx.device.iface.supports_buft     = moe_dev_supports_buft;
    ctx.device.iface.supports_op       = moe_dev_supports_op;
    ctx.device.iface.offload_op        = moe_dev_offload_op;

    ctx.dev_ctx.host_buft   = make_host_buft(&ctx.device);
    // expert bufts are created lazily per model pool (stream_moe_create_expert_buft)

    // reg
    ctx.reg.api_version = GGML_BACKEND_API_VERSION;
    ctx.reg.context = &ctx;
    ctx.reg.iface.get_name          = [](ggml_backend_reg_t) { return "stream_moe"; };
    ctx.reg.iface.get_device_count  = moe_reg_dev_count;
    ctx.reg.iface.get_device        = moe_reg_get_device;

    if (ggml_backend_reg_by_name("stream_moe") == nullptr) {
        ggml_backend_register(&ctx.reg);
        LOG_INFO("stream_moe backend registered (expert pool, ACCEL device)");
    } else {
        LOG_INFO("stream_moe backend already registered");
    }
}

// Create a NEW expert weight buft for one model pool (multi-model support).
ggml_backend_buffer_type_t stream_moe_create_expert_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("stream_moe");
    if (!reg) { stream_moe_register_backend(); reg = ggml_backend_reg_by_name("stream_moe"); }
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    if (!dev) return nullptr;
    auto* dctx = static_cast<moe_dev_ctx*>(dev->context);
    ggml_backend_buffer_type_t buft = make_expert_buft(dev, static_cast<int>(dctx->expert_bufts.size()));
    dctx->expert_bufts.push_back(buft);
    return buft;
}

// Dense weight buft (v2-chunk strip models): real malloc-backed host buffer
// (same allocator as the compute buft) with a distinct per-pool identity so the
// override skips llama.cpp's tensor_info read and route_b fills dense.srcs.
ggml_backend_buffer_type_t stream_moe_create_dense_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("stream_moe");
    if (!reg) { stream_moe_register_backend(); reg = ggml_backend_reg_by_name("stream_moe"); }
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    if (!dev) return nullptr;
    auto* dctx = static_cast<moe_dev_ctx*>(dev->context);
    auto* buft = static_cast<ggml_backend_buffer_type*>(std::calloc(1, sizeof(ggml_backend_buffer_type)));
    auto* ctx  = new expert_buft_ctx(); // name ctx reused from the expert buft
    ctx->dev = dev;
    const int ordinal = static_cast<int>(dctx->expert_bufts.size()) - 1; // pair with the model pool's expert buft
    std::snprintf(ctx->name, sizeof(ctx->name), "STREAMMOE_DENSE%s", ordinal >= 0 ? std::to_string(ordinal).c_str() : "");
    buft->iface.get_name      = expert_buft_get_name;
    buft->iface.alloc_buffer  = host_buft_alloc_buffer; // real malloc-backed
    buft->iface.get_alignment = expert_buft_get_alignment;
    buft->iface.is_host       = [](ggml_backend_buffer_type_t) { return true; };
    buft->device              = dev;
    buft->context             = ctx;
    return buft;
}

ggml_backend_buffer_type_t stream_moe_register_backend_helper_compute_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("stream_moe");
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    return dev ? static_cast<moe_dev_ctx*>(dev->context)->host_buft : nullptr;
}

void stream_moe_backend_bind_scheduler(ggml_backend_buffer_type_t buft, expert_scheduler* sched) {
    for (auto& bnd : g_bindings) {
        if (bnd.buft == buft) { bnd.sched = sched; return; }
    }
    g_bindings.push_back({ buft, sched });
}
void stream_moe_backend_set_threads(int threads) { g_threads = threads > 0 ? threads : 1; }

// ---- M2 device-exec resources -------------------------------------------------

namespace {
std::vector<device_exec_ctx_t> g_dev_execs;   // index = pool - 1 (pool 1 = first device)

device_exec_ctx_t* exec_ctx(uint32_t pool) {
    if (pool == 0) return nullptr;
    return (pool - 1 < g_dev_execs.size()) ? &g_dev_execs[pool - 1] : nullptr;
}

bool ensure_buffer(ggml_backend_buffer_t& buf, size_t& cap, uint8_t*& map,
                   ggml_backend_buffer_type_t buft, size_t need) {
    if (need <= cap) return true;
    if (buf) ggml_backend_buffer_free(buf);
    buf = ggml_backend_buft_alloc_buffer(buft, need);
    if (!buf) {
        cap = 0;
        map = nullptr;
        return false;
    }
    cap = need;
    map = static_cast<uint8_t*>(stmoe_vk_buffer_host_ptr(buf));   // null when not host-mapped
    return true;
}
} // namespace

void stream_moe_backend_bind_device_exec(uint32_t pool, ggml_backend_t be,
                                         ggml_backend_buffer_type_t arena_buft,
                                         ggml_backend_buffer_type_t stage_buft) {
    if (pool == 0 || !be) return;
    if (pool - 1 >= g_dev_execs.size()) g_dev_execs.resize(pool);   // 1-based
    auto& e = g_dev_execs[pool - 1];
    e.be         = be;
    e.arena_buft = arena_buft;
    e.stage_buft = stage_buft;
}

device_exec_ctx_t* stream_moe_backend_device_exec(uint32_t pool) {
    return exec_ctx(pool);
}

bool stream_moe_backend_device_ensure(uint32_t pool, size_t arena_bytes, size_t stage_bytes) {
    auto* e = exec_ctx(pool);
    if (!e || !e->be || !e->arena_buft || !e->stage_buft) return false;
    if (!ensure_buffer(e->arena, e->arena_cap, e->arena_map, e->arena_buft, arena_bytes)) return false;
    if (!ensure_buffer(e->stage, e->stage_cap, e->stage_map, e->stage_buft, stage_bytes)) return false;
    return true;
}

} // namespace stream_moe
