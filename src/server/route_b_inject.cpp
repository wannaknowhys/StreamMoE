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

#include <algorithm>
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
    // Device pools beyond the CPU-RAM scheduler (M1+): allocated up front in
    // route_b_setup, same point as the RAM scheduler. Every block is a whole
    // group of slot-aligned expert runs (an expert weight is never split); a
    // block that fails halves down to a single slot. vram_segs records each
    // buffer's {dev, group, slot span} so a future executor can address the
    // device pools directly.
    struct vram_seg_t {
        ggml_backend_t         be = nullptr;
        std::string            dev;
        uint32_t               group = 0;
        ggml_backend_buffer_t  buf = nullptr;
        size_t                 slot_size = 0;
        size_t                 n_slots = 0;
    };
    std::vector<ggml_backend_t>        vram_backends;  // keep device backends alive
    std::vector<ggml_backend_buffer_t> vram_buffers;   // every allocated buffer
    std::vector<vram_seg_t>            vram_segs;
};

// vram device spec parsed from --moe-expert-pools.
struct vram_spec_t { std::string dev; size_t mb; };

// Plan the group slots one device should host for a byte budget. Floors first
// (one full layer per group, biggest expert groups first - a device that can
// not host every group should host whole layers of the big ones), then the
// spare budget fills by the remaining model-wide byte need. plan[] is capped
// by remaining[]; remaining[] is only decremented by the caller after the real
// allocation lands (allocation can fall short of the plan).
void plan_device_slots(const moe_model_topology_t& topo, size_t budget_bytes,
                       const std::vector<size_t>& remaining, std::vector<size_t>& plan) {
    const size_t G = topo.groups.size();
    const uint32_t per_layer = topo.n_expert;
    plan.assign(G, 0);
    std::vector<size_t> order(G);
    for (size_t i = 0; i < G; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return topo.groups[a].expert_size > topo.groups[b].expert_size;
    });
    for (size_t gi : order) {
        if (budget_bytes < topo.groups[gi].expert_size) continue;   // below one slot
        const size_t want = remaining[gi] < per_layer ? remaining[gi] : per_layer;
        const size_t need = want * topo.groups[gi].expert_size;
        if (need <= budget_bytes) { plan[gi] += want; budget_bytes -= need; }
    }
    std::vector<size_t> rem_bytes(G);
    size_t total_rem = 0;
    for (size_t i = 0; i < G; ++i) {
        const size_t left = remaining[i] - plan[i];
        rem_bytes[i] = left * topo.groups[i].expert_size;
        total_rem += rem_bytes[i];
    }
    if (total_rem && budget_bytes) {
        for (size_t i = 0; i < G; ++i) {
            if (!rem_bytes[i]) continue;
            const size_t share = static_cast<size_t>(budget_bytes * (double) rem_bytes[i] / (double) total_rem);
            const size_t can = remaining[i] - plan[i];
            plan[i] += share / topo.groups[i].expert_size > can ? can : share / topo.groups[i].expert_size;
        }
    }
}

// Allocate one group's planned bytes on a device as slot-aligned blocks. On a
// failed request, halve (always to a whole-slot multiple) and retry; a
// successful block never shrinks the next attempt, so the loop keeps probing
// the device heap until it is full or a single slot fails.
size_t vram_alloc_group(model_pool_t* pool, const std::string& dev, ggml_backend_t be,
                        ggml_backend_buffer_type_t buft, uint32_t group,
                        size_t want_bytes, size_t slot_size) {
    size_t got = 0;
    size_t chunk = want_bytes;
    while (got < want_bytes) {
        const size_t ask = chunk < want_bytes - got ? chunk : want_bytes - got;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ask);
        if (buf) {
            pool->vram_buffers.push_back(buf);
            model_pool_t::vram_seg_t seg;
            seg.be = be; seg.dev = dev; seg.group = group;
            seg.buf = buf; seg.slot_size = slot_size; seg.n_slots = ask / slot_size;
            pool->vram_segs.push_back(seg);
            got += ask;
        } else {
            const size_t half = (ask / 2 / slot_size) * slot_size;  // whole slots only
            if (half < slot_size) break;                            // single slot failed too
            chunk = half;
        }
    }
    return got;
}

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

        // Parse "<device>:<MB>" pool specs. RAM -> scheduler bytes (bounded
        // residency, evict-managed). Every other device (Vulkan0, ...) gets a
        // reserved pool of slot-aligned whole-expert blocks, allocated HERE at
        // the same point as the RAM scheduler (never lazily at first compute).
        const uint32_t per_layer = pool->topo->n_expert;
        const size_t G = pool->topo->groups.size();
        std::vector<size_t> remaining(G);
        for (size_t i = 0; i < G; ++i) {
            remaining[i] = static_cast<size_t>(pool->topo->groups[i].layers.size()) * per_layer;
        }
        size_t pool_bytes = 0;
        std::vector<vram_spec_t> vspecs;
        for (const auto& spec : pools) {
            const size_t colon = spec.find(':');
            if (colon == std::string::npos) continue;
            const std::string dev = spec.substr(0, colon);
            const size_t mb = (size_t) std::strtoull(spec.c_str() + colon + 1, nullptr, 10);
            if (dev == "RAM" || dev == "CPU") {
                pool_bytes = mb * 1024ull * 1024ull;
                continue;
            }
            vspecs.push_back({dev, mb});
        }
        if (pool_bytes == 0) {
            if (pool_full_when_zero) {
                pool_bytes = pool->topo->expert_total_bytes;
            } else {
                pool_bytes = get_available_ram_bytes() * 3 / 4;
            }
        }

        // Real device allocations: one group plan per spec. Failures are logged
        // and the model still runs (RAM backs whatever a device cannot hold).
        for (const auto& vs : vspecs) {
            ggml_backend_dev_t vdev = ggml_backend_dev_by_name(vs.dev.c_str());
            if (!vdev) {
                std::fprintf(stderr, "route B: unknown pool device '%s' - ignoring (RAM backs this model)\n", vs.dev.c_str());
                continue;
            }
            ggml_backend_t vbe = ggml_backend_dev_init(vdev, nullptr);
            if (!vbe) {
                std::fprintf(stderr, "route B: cannot init backend '%s'\n", vs.dev.c_str());
                continue;
            }
            pool->vram_backends.push_back(vbe);
            ggml_backend_buffer_type_t vbuft = ggml_backend_get_default_buffer_type(vbe);
            const size_t budget = vs.mb * 1024ull * 1024ull;
            std::vector<size_t> plan;
            plan_device_slots(*pool->topo, budget, remaining, plan);
            size_t plan_bytes = 0, got_bytes = 0;
            std::string ginfo;
            for (size_t gi = 0; gi < G; ++gi) {
                if (!plan[gi]) continue;
                const size_t es = pool->topo->groups[gi].expert_size;
                plan_bytes += plan[gi] * es;
                const size_t got = vram_alloc_group(pool.get(), vs.dev, vbe, vbuft,
                                                    static_cast<uint32_t>(gi), plan[gi] * es, es);
                got_bytes += got;
                const size_t ns = got / es;
                remaining[gi] = ns >= plan[gi] ? 0 : remaining[gi] - ns;
                ginfo += (gi ? "," : "") + std::to_string(gi) + ":" + std::to_string(ns) + "/" + std::to_string(plan[gi]);
            }
            std::fprintf(stderr, "route B: device pool '%s' got %zu/%zu MB (planned %zu MB)%s\n",
                         vs.dev.c_str(), got_bytes / (1024 * 1024), vs.mb,
                         plan_bytes / (1024 * 1024), ginfo.empty() ? "" : (" groups:" + ginfo).c_str());
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

} // namespace stream_moe
