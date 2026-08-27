#include "server/route_b_inject.h"
#include "../../third_party/llama.cpp/src/tsc_timer.h"

#include "backend/moe_backend.h"
#include "backend/scheduler.h"
#include "common/types.h"
#include "io/async_dio.h"
#include "loader/moe_loader.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace stream_moe {

namespace {

// Process-lifetime state; freed by the OS at exit.
std::unique_ptr<moe_model_topology_t>     g_topo;
std::unique_ptr<async_dio_engine>         g_dio;
std::vector<dio_file_t*>                  g_shards;
std::unique_ptr<expert_scheduler>         g_sched;

// Model-agnostic llama.cpp MoE tensor patterns. Shared experts (ffn_*_shexp,
// plain MUL_MAT) stay on defaults until the backend supports them.
llama_model_tensor_buft_override g_overrides[] = {
    { "blk\\..*\\.ffn_.*_exps\\.weight", nullptr },
    { nullptr, nullptr },
};

} // namespace

llama_model_tensor_buft_override* route_b_setup(const char* model_path, size_t ram_pool_mb, int threads) {
    sm_tmr::timer _t("route_b_setup");
    if (g_sched) {
        return g_overrides; // already initialized (e.g. draft/MTP second context)
    }

    stream_moe_register_backend();

    ggml_backend_buffer_type_t eb = stream_moe_register_backend_helper_expert_buft();
    if (!eb) {
        std::fprintf(stderr, "route B: stream_moe expert buft unavailable - keeping llama.cpp defaults\n");
        return nullptr;
    }
    g_overrides[0].buft = eb;

    try {
        g_topo = std::make_unique<moe_model_topology_t>(moe_loader::parse_gguf_topology(model_path));
        if (g_topo->n_expert == 0) {
            std::fprintf(stderr, "route B: dense model - expert backend is a no-op\n");
            return nullptr;
        }
        g_dio  = async_dio_engine::create(1024);
        for (const auto& shard : g_topo->shard_paths) {
            dio_file_t* f = g_dio->open_file(shard);
            if (!f) {
                std::fprintf(stderr, "route B: cannot DIO-open shard %s\n", shard.c_str());
                return nullptr;
            }
            g_shards.push_back(f);
        }

        const size_t pool_bytes = ram_pool_mb > 0
            ? ram_pool_mb * 1024ull * 1024ull
            : (get_available_ram_bytes() * 3 / 4);

        g_sched = std::make_unique<expert_scheduler>();
        if (!g_sched->init(*g_topo, *g_dio, g_shards, pool_bytes)) {
            std::fprintf(stderr, "route B: scheduler init failed\n");
            g_sched.reset();
            return nullptr;
        }
        g_sched->start();

        stream_moe_backend_set_scheduler(g_sched.get());
        stream_moe_backend_set_threads(threads);

        std::fprintf(stderr, "route B: expert pool active, %.1f GB cap, %u slots\n",
                     (pool_bytes / (1024.0 * 1024.0 * 1024.0)), g_sched->num_slots());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "route B: setup failed: %s\n", e.what());
        return nullptr;
    }
    return g_overrides;
}

} // namespace stream_moe
