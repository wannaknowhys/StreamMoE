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
#include <thread>
#include <utility>
#include <vector>

namespace stream_moe {

struct expert_handle_t {
    int32_t  slot       = -1;
    uint32_t generation = 0;
    uint32_t layer      = 0;
    uint32_t expert     = 0;
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

    // Telemetry (feeds profiler hits).
    uint64_t total_lookups() const { return n_lookups_.load(std::memory_order_relaxed); }
    uint64_t ram_hits()      const { return n_hits_.load(std::memory_order_relaxed); }
    uint64_t disk_misses()   const { return n_misses_.load(std::memory_order_relaxed); }

    size_t   pool_bytes() const { return pool_bytes_; }
    size_t   slot_size() const { return slot_size_; }
    uint32_t num_slots() const { return num_slots_; }

private:
    void scheduler_loop();
    int32_t alloc_or_evict(uint32_t layer, uint32_t expert);
    void load_slot(int32_t slot, uint32_t layer, uint32_t expert);
    void clear_directory(uint32_t layer, uint32_t expert);

    const moe_model_topology_t* topo_   = nullptr;
    async_dio_engine*           dio_    = nullptr;
    std::vector<dio_file_t*>    files_;

    size_t   pool_bytes_ = 0;
    size_t   slot_size_  = 0;
    uint32_t num_slots_  = 0;
    uint8_t* pool_base_  = nullptr;

    std::unique_ptr<uint8_t, aligned_buffer_deleter> staging_;
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
