#pragma once

#include "common/types.h"
#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "loader/moe_loader.h"
#include "io/async_dio.h"

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace stream_moe {

struct layer_routing_req_t {
    uint32_t              layer_idx = 0;
    std::vector<uint32_t> selected_experts;
    uint64_t              seq_id = 0;
    
    // Output split
    std::vector<int32_t>  hit_slots;     // Slots already ready and pinned
    std::vector<uint32_t> miss_experts;  // Experts requiring asynchronous fetch
};

class moe_scheduler {
public:
    moe_scheduler(
        const moe_model_topology_t& topo,
        expert_pool& pool,
        expert_stats_tracker& stats,
        async_dio_engine& dio_engine,
        const std::vector<dio_file_t*>& shard_files
    );
    ~moe_scheduler();

    // Disable copy/move
    moe_scheduler(const moe_scheduler&) = delete;
    moe_scheduler& operator=(const moe_scheduler&) = delete;

    // Start background scheduler thread
    void start();

    // Stop background scheduler thread
    void stop();

    // Called by Compute Thread at layer start:
    // Splits selected experts into Hit/Miss lists, pins Hit slots, and enqueues Miss fetches
    layer_routing_req_t route_and_prefetch(uint32_t layer_idx, const std::vector<uint32_t>& selected_experts, uint64_t seq_id);

    // Called by Compute Thread to wait for all Miss experts of a layer to become READY
    // Returns the newly ready slot IDs
    std::vector<int32_t> wait_miss_ready(const layer_routing_req_t& req, uint32_t timeout_ms = 5000);

    // Called by Compute Thread at layer finish: unpins all slots used by this layer
    void release_layer_slots(const std::vector<int32_t>& slots);

private:
    void scheduler_worker_loop();

    const moe_model_topology_t&    topo_;
    expert_pool&                   pool_;
    expert_stats_tracker&          stats_;
    async_dio_engine&              dio_engine_;
    std::vector<dio_file_t*>       shard_files_;

    std::unique_ptr<uint8_t, aligned_buffer_deleter> staging_buffer_;

    std::atomic<bool>              running_{false};
    std::thread                    worker_thread_;

    // Inter-thread queue & synchronizers
    struct fetch_task_t {
        uint32_t layer_idx;
        uint32_t expert_idx;
        int32_t  assigned_slot;
        uint64_t seq_id;
    };

    std::mutex                     queue_mutex_;
    std::condition_variable        queue_cv_;
    std::vector<fetch_task_t>      pending_tasks_;

    std::mutex                     sync_mutex_;
    std::condition_variable        sync_cv_;
    std::atomic<uint64_t>          completed_task_count_{0};
};

} // namespace stream_moe