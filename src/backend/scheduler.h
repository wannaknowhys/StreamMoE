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
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace stream_moe {

struct expert_handle_t {
    int32_t  slot       = -1;
    uint32_t generation = 0;
    uint32_t layer      = 0;
    uint32_t expert     = 0;
    uint32_t pool       = 0;   // device pool index (0 = CPU RAM)
    bool     pinned     = false;
};

class expert_scheduler {
public:
    expert_scheduler() = default;
    ~expert_scheduler();

    expert_scheduler(const expert_scheduler&) = delete;
    expert_scheduler& operator=(const expert_scheduler&) = delete;

    // Allocate the fixed pool and open the DIO engine/files. `pool_bytes` is the
    // hard cap on expert residency (e.g. 70GB). Call before start().
    bool init(const moe_model_topology_t& topo, async_dio_engine& dio,
              const std::vector<dio_file_t*>& files, size_t pool_bytes);

    void start();
    void stop();

    // Compute-side API (called from graph_compute).
    // Block until (layer, expert) is resident, then pin it (refcount++).
    expert_handle_t pin_expert(uint32_t layer, uint32_t expert);

    // For the down-role split: the expert was pinned by the gate/up split and
    // eviction is blocked; just wait until READY (no refcount change).
    void wait_ready(uint32_t layer, uint32_t expert);

    // Release a pin (refcount--); slot becomes evictable at 0.
    void unpin(const expert_handle_t& h);

    // Sub-pool layout (docs/MULTI_SUBPOOL.md): one contiguous pool carved into
    // per-expert-group regions. Slot indices are global (across all groups).
    struct subpool_t {
        uint32_t slot_begin = 0;
        uint32_t n_slots    = 0;
        size_t   expert_size = 0;
        uint8_t* base       = nullptr;
    };
    // Group index owning `layer`, or (uint32_t)-1.
    uint32_t group_of(uint32_t layer) const;
    const subpool_t& subpool(uint32_t gidx) const { return subpools_[gidx]; }
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
    // One in-flight expert load. The ringbuffer element is a contiguous 4K-aligned
    // block [async_load_t header][staging data area]; `staging` points into that
    // data area (0-sized for v2 which DIOs straight into the slot). `reqs` holds
    // one async DIO request per sector-aligned slice; `pending` counts down to
    // zero (IOCP has no batch-completion - we aggregate ourselves). One pool per
    // model (draft/main sizes differ; even one may be v2 and the other original).
    struct async_load_t {
        uint32_t layer = 0, expert = 0, pool = 0, slot = 0;
        expert_read_plan_t plan;
        int pending = 0;
        bool failed = false;
        bool direct = false;      // v2: reqs read straight into the slot
        uint8_t* staging = nullptr;
        aio_req_t reqs[MAX_SUB_TENSORS_PER_EXPERT];
    };

    void scheduler_loop();
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
    std::thread             worker_;
    std::atomic<uint64_t>   seq_{0};

    std::atomic<uint64_t>   n_lookups_{0};
    std::atomic<uint64_t>   n_hits_{0};
    std::atomic<uint64_t>   n_misses_{0};

    expert_stats_tracker    stats_;
};

} // namespace stream_moe
