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

// One bounded expert pool per model (multi-model: main model + draft/MTP model
// each get their own pool, scheduler, DIO files and expert buft). Overrides are
// per-pool and point at that pool's own buft - never shared across models.
struct model_pool_t {
    std::string model_path;
    std::unique_ptr<moe_model_topology_t> topo;
    std::unique_ptr<async_dio_engine>     dio;
    std::vector<dio_file_t*>              shards;
    std::unique_ptr<expert_scheduler>     sched;
    ggml_backend_buffer_type_t            buft = nullptr;
    llama_model_tensor_buft_override      overrides[2] = {};
};

std::vector<std::unique_ptr<model_pool_t>> g_pools;

} // namespace

llama_model_tensor_buft_override* route_b_setup(const char* model_path, size_t ram_pool_mb, int threads, bool pool_full_when_zero) {
    sm_tmr::timer _t("route_b_setup");
    for (const auto& p : g_pools) {
        if (p->model_path == model_path) {
            return p->overrides; // already initialized
        }
    }

    auto pool = std::make_unique<model_pool_t>();
    pool->model_path = model_path;

    stream_moe_register_backend();

    pool->buft = stream_moe_create_expert_buft();
    if (!pool->buft) {
        std::fprintf(stderr, "route B: stream_moe expert buft unavailable - keeping llama.cpp defaults\n");
        return nullptr;
    }

    try {
        pool->topo = std::make_unique<moe_model_topology_t>(moe_loader::parse_gguf_topology(model_path));
        if (pool->topo->n_expert == 0) {
            std::fprintf(stderr, "route B: dense model - expert backend is a no-op\n");
            return nullptr;
        }
        pool->dio = async_dio_engine::create(1024);
        for (const auto& shard : pool->topo->shard_paths) {
            dio_file_t* f = pool->dio->open_file(shard);
            if (!f) {
                std::fprintf(stderr, "route B: cannot DIO-open shard %s\n", shard.c_str());
                return nullptr;
            }
            pool->shards.push_back(f);
        }

        size_t pool_bytes;
        if (ram_pool_mb > 0) {
            pool_bytes = ram_pool_mb * 1024ull * 1024ull;
        } else if (pool_full_when_zero) {
            pool_bytes = pool->topo->expert_total_bytes; // full residency, no eviction
        } else {
            pool_bytes = get_available_ram_bytes() * 3 / 4;
        }

        pool->sched = std::make_unique<expert_scheduler>();
        if (!pool->sched->init(*pool->topo, *pool->dio, pool->shards, pool_bytes)) {
            std::fprintf(stderr, "route B: scheduler init failed\n");
            return nullptr;
        }
        pool->sched->start();

        stream_moe_backend_bind_scheduler(pool->buft, pool->sched.get());
        stream_moe_backend_set_threads(threads);

        pool->overrides[0] = { "blk\\..*\\.ffn_.*_exps\\.weight", pool->buft };

        std::fprintf(stderr, "route B: expert pool active (model '%s'), %.1f GB cap, %u slots\n",
                     model_path, (pool_bytes / (1024.0 * 1024.0 * 1024.0)), pool->sched->num_slots());

        g_pools.push_back(std::move(pool));
        return g_pools.back()->overrides;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "route B: setup failed: %s\n", e.what());
        return nullptr;
    }
}

} // namespace stream_moe
