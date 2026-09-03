#pragma once

// Route B expert scheduler (docs/Backend.md §5, docs/LLAMA_MOE_NO_MMAP_RESEARCH.md §4).
// Owns the bounded expert pool, streams expert slices from GGUF shards via DIO
// on a background thread, and exposes a blocking "pin" API consumed by the
// backend's graph_compute.
//
// Model-agnostic: operates purely on moe_model_topology_t read plans (standard
// llama.cpp MoE schema). Cross-platform: async_dio_engine (Win IOCP / POSIX
// pread) + slot.h wait/wake (WaitOnAddress / futex).
//
// Memory guarantee: the whole pool is committed once at init (fixed budget),
// so expert residency can never exceed `pool_bytes` - dense/KV/compute buffers
// are the only other allocations.

#include "loader/moe_loader.h"
#include "io/async_dio.h"
#include "io/staging_reader.h"
#include "pool/expert_stats.h"
#include "backend/slot.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace stream_moe {

// A device (non-RAM) expert region added on top of the CPU-RAM pool: one
// contiguous run of slot-aligned whole-expert blocks, host-mapped so the
// scheduler can load/move expert bytes straight into it.
struct vram_region_t {
    uint32_t  pool   = 1;   // device pool index (0 = CPU RAM is not a region)
    uint32_t  group  = 0;   // expert group this region serves
    uint8_t*  base   = nullptr;   // host-mapped base (nullptr -> region skipped)
    uint32_t  n_slots = 0;
};

struct expert_handle_t {
    int32_t  slot       = -1;
    uint32_t generation = 0;
    uint32_t layer      = 0;
    uint32_t expert     = 0;
    uint32_t pool       = 0;   // device pool index (0 = CPU RAM)
    bool     pinned     = false;
};

// One in-flight expert load. The ringbuffer element is a contiguous 4K-aligned
// block [async_load_t header][staging data area]; `staging` points into that
// data area (0-sized for v2 which DIOs straight into the slot). `reqs` holds
// one async DIO request per sector-aligned slice; `pending` counts down to
// zero (IOCP has no batch-completion - we aggregate ourselves). One pool per
// model (draft/main sizes differ; even one may be v2 and the other original).
// `sched` names the owning pool so shared-dio completions are dispatched to it.
struct async_load_t {
    uint32_t layer = 0, expert = 0, pool = 0, slot = 0;
    expert_read_plan_t plan;
    int pending = 0;
    bool failed = false;
    bool direct = false;      // v2: reqs read straight into the slot
    uint8_t* staging = nullptr;
    uint64_t req_tsc = 0;     // [TMR] raw TSC: async load requested
    uint64_t dio_tsc = 0;     // [TMR] raw TSC: all DIO reads completed
    uint64_t done_tsc = 0;    // [TMR] raw TSC: drain settled the slot
    class expert_scheduler* sched = nullptr;
    aio_req_t reqs[MAX_SUB_TENSORS_PER_EXPERT];
};

class expert_scheduler {
public:
    expert_scheduler() = default;
    ~expert_scheduler();

    expert_scheduler(const expert_scheduler&) = delete;
    expert_scheduler& operator=(const expert_scheduler&) = delete;

    // Allocate the fixed pool and open the DIO engine/files. `pool_bytes` is the
    // hard cap on CPU-RAM expert residency (e.g. 70GB). `vregions` adds
    // host-mapped device (VRAM) regions as extra slot runs on top of the RAM
    // sub-pools - each region is one contiguous run of slot-aligned whole-expert
    // blocks serving one expert group (pool index 1, 2, ...). Call before
    // start(). RAM behaviour is unchanged when `vregions` is empty.
    bool init(const moe_model_topology_t& topo, async_dio_engine& dio,
              const std::vector<dio_file_t*>& files, size_t pool_bytes,
              const std::vector<vram_region_t>& vregions = {});

    // A single process-wide scheduler thread drives all pools (multi-model:
    // main + draft). start() registers this pool with it; stop() unregisters.
    void start();
    void stop();

    // Worker-driven execution hooks (called by the global scheduler thread):
    //  - drain_completions: finish IO completions dispatched to this pool by
    //    their owner (shared-dio completions carry the owning pool in the task)
    //  - accept_requests: pop the MPSC alloc queue and submit async loads
    //  - update_alpha: adapt eviction blend from hit-rate EMA
    void drain_completions(aio_req_t** done, uint32_t n);
    bool accept_requests();
    void update_alpha();
    bool is_running() const { return running_.load(std::memory_order_relaxed); }
    // The DIO engine this pool submits loads to. Pools share one engine in
    // production (route_b_inject); tests use their own. The global worker polls
    // each pool's engine and dispatches completions by owner (async_load_t::sched).
    async_dio_engine* dio_engine() const { return dio_; }

    // Compute-side API (called from graph_compute).
    // Block until (layer, expert) is resident, then pin it (refcount++).
    expert_handle_t pin_expert(uint32_t layer, uint32_t expert);

    // For the down-role split: the expert was pinned by the gate/up split and
    // eviction is blocked; just wait until READY (no refcount change).
    void wait_ready(uint32_t layer, uint32_t expert);

    // Release a pin (refcount--); slot becomes evictable at 0.
    void unpin(const expert_handle_t& h);

    // Sub-pool layout (docs/MULTI_SUBPOOL.md): the slot space is carved into
    // per-(group, pool) regions. Slot indices are global across all regions.
    // A group's CPU-RAM region comes first; device (VRAM) regions follow.
    struct subpool_t {
        uint32_t pool       = 0;   // 0 = CPU RAM, 1 = first device, ...
        uint32_t group      = 0;
        uint32_t slot_begin = 0;
        uint32_t n_slots    = 0;
        size_t   expert_size = 0;
        uint8_t* base       = nullptr;
    };
    // Group index owning `layer`, or (uint32_t)-1.
    uint32_t group_of(uint32_t layer) const;
    // CPU-RAM sub-pool of `group` (the executor host path), or nullptr when the
    // group has no RAM region. Device regions of the group are later in the
    // sub-pool list.
    const subpool_t* ram_subpool(uint32_t group) const;
    const subpool_t* subpool_at(size_t idx) const { return &subpools_[idx]; }
    size_t subpools_count() const { return subpools_.size(); }

    // Raw slot memory for a pinned/resident slot (compute side).
    uint8_t* slot_mem(int32_t slot) const {
        if (slot < 0 || slot >= static_cast<int32_t>(num_slots_)) return nullptr;
        for (const auto& sp : subpools_) {
            if (slot >= static_cast<int32_t>(sp.slot_begin) && slot < static_cast<int32_t>(sp.slot_begin + sp.n_slots)) {
                return sp.base + static_cast<size_t>(slot - sp.slot_begin) * sp.expert_size;
            }
        }
        return nullptr;
    }
    // Current slot index for (layer, expert) in any pool, or -1 if not resident.
    int32_t slot_of(uint32_t layer, uint32_t expert) const {
        uint32_t pool = 0;
        uint32_t s = dir_->scan(layer, expert, &pool);
        return s == SLOT_UNASSIGNED ? -1 : static_cast<int32_t>(s);
    }
    void unpin_slot(int32_t slot) {
        if (slot >= 0 && slot < static_cast<int32_t>(num_slots_)) slots_[slot].unpin();
    }
    // Sub-tensor layout inside a compact slot for a branch name
    // (e.g. "blk.5.ffn_gate_exps.weight"). Returns true + offset/size if found.
    bool branch_layout(uint32_t layer, uint32_t expert, const std::string& name,
                       size_t& slot_offset, size_t& byte_size) const {
        if (!topo_ || layer >= topo_->n_layer || expert >= topo_->n_expert) return false;
        const expert_info_t& info = topo_->get_expert(layer, expert);
        for (const auto& st : info.sub_tensors) {
            if (st.name == name) { slot_offset = st.slot_offset; byte_size = st.byte_size; return true; }
        }
        return false;
    }
    const moe_model_topology_t& topology() const { return *topo_; }

    // Telemetry (feeds profiler hits).
    uint64_t total_lookups() const { return n_lookups_.load(std::memory_order_relaxed); }
    uint64_t ram_hits()      const { return n_hits_.load(std::memory_order_relaxed); }
    uint64_t disk_misses()   const { return n_misses_.load(std::memory_order_relaxed); }

    size_t   pool_bytes() const { return pool_bytes_; }
    uint32_t num_slots() const { return num_slots_; }

private:
    int32_t alloc_or_evict(uint32_t layer, uint32_t expert);
    async_load_t* start_async_load(int32_t slot, uint32_t layer, uint32_t expert);
    void clear_directory(uint32_t layer, uint32_t expert);
    async_load_t* load_task(uint32_t idx) {
        return reinterpret_cast<async_load_t*>(load_pool_ + static_cast<size_t>(idx) * load_stride_);
    }

    const moe_model_topology_t* topo_   = nullptr;
    async_dio_engine*           dio_    = nullptr;
    std::vector<dio_file_t*>    files_;

    size_t   pool_bytes_ = 0;
    uint32_t num_slots_  = 0;
    uint32_t n_pools_    = 1;   // device pools; multi-device (GPU) comes in Phase B
    uint8_t* pool_base_  = nullptr;
    std::vector<subpool_t> subpools_;

    // Async-load ringbuffer pool: preallocated contiguous aligned block, element
    // stride = 4K-aligned header + 4K-aligned staging size. max_in_flight = pool
    // size = the actual concurrency limit (open it up for large prefill, e.g. 64).
    // When the pool is full the scheduler must keep draining completions to free
    // elements (it never blocks on submission); a full pool with no completions
    // is genuine backpressure (IO saturated) - surfaced by the sat report macro.
    uint32_t max_in_flight_ = 64;
    size_t   load_stride_ = 0;
    size_t   load_staging_size_ = 0;   // 0 when layout == V2
    uint8_t* load_pool_ = nullptr;
    std::vector<uint32_t> load_free_;   // free element indices (LIFO)
    std::vector<uint8_t>  load_in_use_;

    // Eviction scoring state (scheduler thread owns):
    //   recency = max(0, cur_token - last_used_token)   (dir_ holds last_used)
    //   score   = alpha * 1/(1+recency) + (1-alpha) * freq
    //   alpha   = clamp(1 - hit_rate_ema, 0.1, 0.9)
    std::atomic<uint64_t> token_{0};        // global pin counter (recency source)
    std::atomic<double>   alpha_{0.5};
    double hit_rate_ema_ = 0.0;
    uint64_t hit_rate_init_denom_ = 1;      // initial hit rate = pool bytes / total expert bytes

    std::unique_ptr<slot_meta[]>    slots_;              // std::atomic makes slot_meta non-copyable
    std::vector<std::pair<uint32_t, uint32_t>> owner_;   // scheduler-private
    expert_directory*               dir_ = nullptr;
    std::unique_ptr<expert_directory> dir_owned_;

    mpsc_alloc_queue        requests_{4096};
    std::atomic<bool>       running_{false};
    std::atomic<uint64_t>   seq_{0};

    std::atomic<uint64_t>   n_lookups_{0};
    std::atomic<uint64_t>   n_hits_{0};
    std::atomic<uint64_t>   n_misses_{0};

    expert_stats_tracker    stats_;
};

// ---- global scheduler thread (multi-level cache controller) ----------------
// One process-wide worker drives every registered pool: drains shared-dio
// completions (dispatched to the owning pool via async_load_t::sched), accepts
// alloc requests, and adapts eviction alpha. Pools register on start() and
// unregister on stop(); the worker runs while any pool is registered. Future
// GPU pools and CPU<->GPU migration ride the same controller.
class global_expert_scheduler {
public:
    static global_expert_scheduler& instance();
    ~global_expert_scheduler();

    void register_scheduler(expert_scheduler* s);
    void unregister_scheduler(expert_scheduler* s);

private:
    global_expert_scheduler() = default;
    void worker_loop();

    std::mutex mtx_;
    std::vector<expert_scheduler*> schedulers_;
    std::thread worker_;
    bool running_ = false;
};

} // namespace stream_moe
