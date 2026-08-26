#include "backend/moe_backend.h"
#include "common/logger.h"

#include "ggml-backend-impl.h"

#include <cstdlib>
#include "backend/alloc.h"
#include <cstring>

namespace stream_moe {

namespace {

// ---- shared helpers ------------------------------------------------------

// Expert weight buft / buffer ---------------------------------------------

struct expert_buft_ctx {
    ggml_backend_dev_t dev = nullptr;
};

struct expert_buf_ctx {
    size_t size = 0;
};

const char* expert_buft_get_name(ggml_backend_buffer_type_t) { return "STREAMMOE_EXPERT"; }

ggml_backend_buffer_t expert_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* ctx = static_cast<expert_buf_ctx*>(std::calloc(1, sizeof(expert_buf_ctx)));
    ctx->size = size;
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = [](ggml_backend_buffer_t b) { std::free(b->context); };
    // get_base: non-null sentinel (assert in ggml_backend_buffer_get_base);
    // never dereferenced - our graph_compute reads slots, not tensor->data.
    iface.get_base = [](ggml_backend_buffer_t) -> void* { static char sentinel[8]; return sentinel; };
    iface.memset_tensor = [](ggml_backend_buffer_t, ggml_tensor*, uint8_t, size_t, size_t) {};
    iface.set_tensor    = [](ggml_backend_buffer_t, ggml_tensor*, const void*, size_t, size_t) {};
    iface.get_tensor    = [](ggml_backend_buffer_t, const ggml_tensor*, void*, size_t, size_t) {};
    iface.clear = [](ggml_backend_buffer_t, uint8_t) {};
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

size_t expert_buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }

ggml_backend_buffer_type_t make_expert_buft(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buft = {};
    static bool inited = false;
    if (!inited) {
        static expert_buft_ctx ctx;
        ctx.dev = dev;
        buft.iface.get_name       = expert_buft_get_name;
        buft.iface.alloc_buffer   = expert_buft_alloc_buffer;
        buft.iface.get_alignment  = expert_buft_get_alignment;
        buft.iface.is_host        = [](ggml_backend_buffer_type_t) { return true; };
        buft.device               = dev;
        buft.context              = &ctx;
        inited = true;
    }
    return &buft;
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
    ggml_backend_buffer_type_t expert_buft = nullptr;
    ggml_backend_buffer_type_t host_buft   = nullptr;
};

struct moe_backend_ctx {};

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
    backend->context = std::calloc(1, sizeof(moe_backend_ctx));
    backend->iface.get_name = [](ggml_backend_t) { return "STREAMMOE"; };
    backend->iface.free     = [](ggml_backend_t b) {
        std::free(b->context);
        std::free(b);
    };
    backend->iface.synchronize = [](ggml_backend_t) {};
    backend->iface.graph_compute = [](ggml_backend_t, ggml_cgraph*) -> enum ggml_status {
        // Skeleton: per-expert mini-graph delegation not implemented yet.
        LOG_ERROR("stream_moe backend graph_compute: not implemented - "
                  "route B delegation is not wired; disable the expert backend flag.");
        return GGML_STATUS_FAILED;
    };
    return backend;
}

ggml_backend_buffer_type_t moe_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return static_cast<moe_dev_ctx*>(dev->context)->host_buft;
}

bool moe_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto* ctx = static_cast<moe_dev_ctx*>(dev->context);
    return buft == ctx->expert_buft || buft == ctx->host_buft;
}

bool moe_dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    // Take ownership of MoE weight ops. Quant types: any ggml_type is accepted;
    // delegation will run them on the CPU backend's stock kernels.
    switch (op->op) {
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MUL_MAT:
            return true;
        default:
            return false;
    }
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

ggml_backend_buffer_type_t stream_moe_expert_buft() {
    // Requires registration first; fetch from the registered device.
    return stream_moe_register_backend_helper_expert_buft();
}

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

    ctx.dev_ctx.expert_buft = make_expert_buft(&ctx.device);
    ctx.dev_ctx.host_buft   = make_host_buft(&ctx.device);

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

// Expose bufts after registration (declared in moe_backend.h).
ggml_backend_buffer_type_t stream_moe_register_backend_helper_expert_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("stream_moe");
    if (!reg) { stream_moe_register_backend(); reg = ggml_backend_reg_by_name("stream_moe"); }
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    return dev ? static_cast<moe_dev_ctx*>(dev->context)->expert_buft : nullptr;
}

ggml_backend_buffer_type_t stream_moe_register_backend_helper_compute_buft() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("stream_moe");
    if (!reg) return nullptr;
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    return dev ? static_cast<moe_dev_ctx*>(dev->context)->host_buft : nullptr;
}

} // namespace stream_moe
