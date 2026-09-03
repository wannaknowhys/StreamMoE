#include "server/route_b_inject.h"
#include "../../third_party/llama.cpp/src/tsc_timer.h"

#include "backend/moe_backend.h"
#include "backend/scheduler.h"
#include "common/types.h"
#include "io/async_dio.h"
#include "io/staging_reader.h"
#include "loader/moe_loader.h"
#include "loader/model_builder.h"
#include "loader/topo_builder.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace stream_moe {

namespace {

// One bounded expert pool per model (multi-model: main model + draft/MTP model
// each get their own pool, scheduler and expert buft, but SHARE a single DIO
// engine and the process-wide global scheduler thread). Overrides are per-pool
// and point at that pool's own buft - never shared across models.
struct model_pool_t {
    std::string model_path;
    std::unique_ptr<moe_model_topology_t> topo;
    std::shared_ptr<async_dio_engine>     dio;   // shared across all pools
    std::vector<dio_file_t*>              shards;
    std::unique_ptr<expert_scheduler>     sched;
    ggml_backend_buffer_type_t            buft = nullptr;       // expert pool buft
    ggml_backend_buffer_type_t            dense_buft = nullptr; // v2-chunk dense buft
    std::unordered_map<std::string, std::vector<src_seg_t>> dense_srcs; // name -> strip segments (v2 chunk)
    llama_model_tensor_buft_override      overrides[3] = {};    // expert + dense + terminator
    // Device pools beyond the CPU-RAM scheduler (M1+: allocate, reserve for use).
    std::vector<std::pair<std::string, size_t>> vram_pending;   // dev, MB - allocated lazily (vulkan registers late)
    std::vector<ggml_backend_t>                 vram_backends;  // one ggml backend per VRAM pool
    std::vector<ggml_backend_buffer_t>          vram_buffers;   // its allocated buffer
};

std::vector<std::unique_ptr<model_pool_t>> g_pools;
std::shared_ptr<async_dio_engine> g_shared_dio;   // one IOCP engine for all pools

} // namespace

llama_model_tensor_buft_override* route_b_setup(
    const char* model_path,
    const std::vector<std::string>& extra_files,
    const std::vector<std::string>& pools,
    int threads, bool pool_full_when_zero) {
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
        // v2-chunk: full strip file list = main -m file + extra files
        std::vector<std::string> paths;
        paths.reserve(1 + extra_files.size());
        paths.push_back(model_path);
        for (const auto& f : extra_files) paths.push_back(f);
        model_t m = parse_model(paths);
        pool->topo = std::make_unique<moe_model_topology_t>(build_topology(m, paths[0]));
        if (pool->topo->n_expert == 0) {
            std::fprintf(stderr, "route B: dense model - expert backend is a no-op\n");
            return nullptr;
        }
        if (pool->topo->incomplete) {
            // v2-chunk: remember every dense tensor's strip segments for fill
            for (const auto& d : m.dense) pool->dense_srcs[d.name] = d.srcs;
        }
        if (!g_shared_dio) {
            g_shared_dio = async_dio_engine::create(1024);
        }
        pool->dio = g_shared_dio;
        for (const auto& shard : pool->topo->shard_paths) {
            dio_file_t* f = g_shared_dio->open_file(shard);
            if (!f) {
                std::fprintf(stderr, "route B: cannot DIO-open shard %s\n", shard.c_str());
                return nullptr;
            }
            pool->shards.push_back(f);
        }

        // Parse "<device>:<MB>" pool specs. RAM -> scheduler bytes; any other
        // device (Vulkan0, ...) -> allocate the buffer now (reserved; expert
        // residency on the device comes later). No RAM entry keeps the default
        // (75% free RAM, or full residency when pool_full_when_zero).
        size_t pool_bytes = 0;
        for (const auto& spec : pools) {
            const size_t colon = spec.find(':');
            if (colon == std::string::npos) continue;
            const std::string dev = spec.substr(0, colon);
            const size_t mb = (size_t) std::strtoull(spec.c_str() + colon + 1, nullptr, 10);
            if (dev == "RAM" || dev == "CPU") {
                pool_bytes = mb * 1024ull * 1024ull;
                continue;
            }
            // reserve a device pool: defer the real backend allocation - vulkan
            // (and other device backends) register after model load, while
            // route_b_setup runs before it. route_b_lazy_alloc_pools() (called
            // from the first graph_compute) performs the actual allocation.
            pool->vram_pending.push_back({dev, mb});
            std::fprintf(stderr, "route B: device pool queued %zu MB on '%s' (lazy alloc)\n", mb, dev.c_str());
        }
        if (pool_bytes == 0) {
            if (pool_full_when_zero) {
                pool_bytes = pool->topo->expert_total_bytes;
            } else {
                pool_bytes = get_available_ram_bytes() * 3 / 4;
            }
        }

        pool->sched = std::make_unique<expert_scheduler>();
        if (!pool->sched->init(*pool->topo, *pool->dio, pool->shards, pool_bytes)) {
            std::fprintf(stderr, "route B: scheduler init failed\n");
            return nullptr;
        }
        pool->sched->start();

        stream_moe_backend_bind_scheduler(pool->buft, pool->sched.get());
        stream_moe_backend_set_threads(threads);

        pool->overrides[0] = { ".*_exps\\.weight", pool->buft };
        if (pool->topo->incomplete) {
            // v2-chunk: dense tensors are strip-scattered - route them to a real
            // dense buft so llama.cpp skips its tensor_info read and route_b
            // fills them from dense.srcs after load.
            pool->dense_buft = stream_moe_create_dense_buft();
            if (!pool->dense_buft) {
                std::fprintf(stderr, "route B: dense buft unavailable\n");
                return nullptr;
            }
            pool->overrides[1] = { "^(?!.*_exps\\.weight)", pool->dense_buft };
        }

        std::fprintf(stderr, "route B: expert pool active (model '%s'), %.1f GB cap, %u slots%s\n",
                     model_path, (pool_bytes / (1024.0 * 1024.0 * 1024.0)), pool->sched->num_slots(),
                     pool->topo->incomplete ? ", v2-chunk dense takeover" : "");

        g_pools.push_back(std::move(pool));
        return g_pools.back()->overrides;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "route B: setup failed: %s\n", e.what());
        return nullptr;
    }
}

bool route_b_fill_dense(const char* tensor_name, void* data) {
    if (!tensor_name || !data) return false;
    const std::string name(tensor_name);
    for (auto& p : g_pools) {
        if (!p->topo || !p->topo->incomplete) continue; // only v2-chunk pools take over dense
        auto it = p->dense_srcs.find(name);
        if (it == p->dense_srcs.end()) continue; // expert / unknown: scheduler or default path
        std::vector<sub_tensor_req_t> reqs;
        reqs.reserve(it->second.size());
        for (const auto& s : it->second) {
            sub_tensor_req_t r;
            r.shard_idx = s.file;
            r.file_offset = s.off;
            r.byte_size = s.len;
            r.slot_offset = s.in_off; // segment position inside the dense tensor
            reqs.push_back(r);
        }
        const expert_read_plan_t plan = build_expert_read_plan(reqs.data(), static_cast<uint32_t>(reqs.size()));
        // DIO requires a 4K-aligned staging buffer (std::vector is not).
        uint8_t* staging = nullptr;
        const size_t ssz = (plan.total_staging_size + 4095u) & ~size_t(4095u);
        if (ssz) {
            staging = static_cast<uint8_t*>(async_dio_engine::alloc_aligned(ssz));
            if (!staging) return false;
        }
        uint8_t dummy = 0;
        const bool ok = read_expert_sync(p->dio.get(), p->shards, plan,
                                         staging ? staging : &dummy, static_cast<uint8_t*>(data));
        if (staging) async_dio_engine::free_aligned(staging);
        return ok;
    }
    return false;
}

// Device (VRAM) pools are queued at route_b_setup (vulkan registers only after
// model load). Perform the real allocations once the devices are available -
// called from the first graph_compute.
void route_b_lazy_alloc_pools() {
    for (auto& pool : g_pools) {
        for (const auto& req : pool->vram_pending) {
            const std::string& dev = req.first;
            const size_t mb = req.second;
            ggml_backend_dev_t vdev = ggml_backend_dev_by_name(dev.c_str());
            if (!vdev) {
                std::fprintf(stderr, "route B: lazy alloc: unknown device '%s'. Available:",
                             dev.c_str());
                for (int di = 0; di < ggml_backend_dev_count(); ++di) {
                    ggml_backend_dev_t dd = ggml_backend_dev_get(di);
                    if (dd) std::fprintf(stderr, " %s", ggml_backend_dev_name(dd));
                }
                std::fprintf(stderr, "\n");
                continue;
            }
            ggml_backend_t vbe = ggml_backend_dev_init(vdev, nullptr);
            if (!vbe) { std::fprintf(stderr, "route B: lazy alloc: cannot init '%s'\n", dev.c_str()); continue; }
            ggml_backend_buffer_t vbuf = ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(vbe), mb * 1024ull * 1024ull);
            if (!vbuf) {
                std::fprintf(stderr, "route B: lazy alloc: cannot allocate %zu MB on '%s'\n", mb, dev.c_str());
                ggml_backend_free(vbe);
                continue;
            }
            pool->vram_backends.push_back(vbe);
            pool->vram_buffers.push_back(vbuf);
            std::fprintf(stderr, "route B: device pool allocated %zu MB on '%s' (reserved)\n", mb, dev.c_str());
        }
        pool->vram_pending.clear();
    }
}

} // namespace stream_moe
