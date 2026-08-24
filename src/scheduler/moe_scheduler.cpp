#include "scheduler/moe_scheduler.h"
#include "common/logger.h"

#include <cassert>
#include <chrono>

namespace stream_moe {

moe_scheduler::moe_scheduler(
    const moe_model_topology_t& topo,
    expert_pool& pool,
    expert_stats_tracker& stats,
    async_dio_engine& dio_engine,
    const std::vector<dio_file_t*>& shard_files
) : topo_(topo),
    pool_(pool),
    stats_(stats),
    dio_engine_(dio_engine),
    shard_files_(shard_files) {
    
    // Allocate shared aligned staging buffer for the scheduler thread
    size_t staging_sz = topo_.expert_dio_staging_size > 0 ? topo_.expert_dio_staging_size : (1024 * 1024 * 16);
    staging_buffer_ = make_aligned_buffer(staging_sz);
}

moe_scheduler::~moe_scheduler() {
    stop();
}

void moe_scheduler::start() {
    if (running_.exchange(true)) return;
    worker_thread_ = std::thread(&moe_scheduler::scheduler_worker_loop, this);
    LOG_INFO("Scheduler thread started.");
}

void moe_scheduler::stop() {
    if (!running_.exchange(false)) return;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    LOG_INFO("Scheduler thread stopped.");
}

layer_routing_req_t moe_scheduler::route_and_prefetch(
    uint32_t layer_idx,
    const std::vector<uint32_t>& selected_experts,
    uint64_t seq_id
) {
    layer_routing_req_t req;
    req.layer_idx = layer_idx;
    req.selected_experts = selected_experts;
    req.seq_id = seq_id;

    std::vector<fetch_task_t> new_tasks;

    for (uint32_t exp_id : selected_experts) {
        // Record access in stats tracker
        stats_.record_access(layer_idx, exp_id);

        int32_t slot = pool_.find_slot(static_cast<int32_t>(layer_idx), static_cast<int32_t>(exp_id));
        if (slot >= 0 && pool_.pin_slot(slot)) {
            // Hit! Slot is present and protected
            req.hit_slots.push_back(slot);
        } else {
            // Miss! Needs allocation & async fetch
            req.miss_experts.push_back(exp_id);
            int32_t target_slot = pool_.allocate_or_evict_slot(
                
                static_cast<int32_t>(layer_idx),
                static_cast<int32_t>(exp_id),
                stats_,
                seq_id
            );

            if (target_slot >= 0) {
                // Pin target slot immediately so other concurrent allocations don't re-evict it
                pool_.pin_slot(target_slot);
                fetch_task_t task;
                task.layer_idx     = layer_idx;
                task.expert_idx    = exp_id;
                task.assigned_slot = target_slot;
                task.seq_id        = seq_id;
                new_tasks.push_back(task);
            } else {
                LOG_ERROR("route_and_prefetch: failed to allocate slot for Layer " << layer_idx << " Expert " << exp_id);
            }
        }
    }

    // Submit miss tasks to scheduler thread
    if (!new_tasks.empty()) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_tasks_.insert(pending_tasks_.end(), new_tasks.begin(), new_tasks.end());
        queue_cv_.notify_one();
    }

    return req;
}

std::vector<int32_t> moe_scheduler::wait_miss_ready(const layer_routing_req_t& req, uint32_t timeout_ms) {
    std::vector<int32_t> ready_slots;
    if (req.miss_experts.empty()) {
        return ready_slots;
    }

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t exp_id : req.miss_experts) {
        int32_t slot_id = -1;
        while (true) {
            slot_id = pool_.find_slot(static_cast<int32_t>(req.layer_idx), static_cast<int32_t>(exp_id));
            if (slot_id >= 0) {
                // Verified ready
                ready_slots.push_back(slot_id);
                break;
            }

            // Wait on sync CV
            std::unique_lock<std::mutex> lock(sync_mutex_);
            if (sync_cv_.wait_for(lock, std::chrono::milliseconds(50)) == std::cv_status::timeout) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                if (elapsed >= timeout_ms) {
                    LOG_ERROR("wait_miss_ready: timeout waiting for Layer " << req.layer_idx << " Expert " << exp_id);
                    break;
                }
            }
        }
    }

    return ready_slots;
}

void moe_scheduler::release_layer_slots(const std::vector<int32_t>& slots) {
    for (int32_t slot_id : slots) {
        pool_.unpin_slot(slot_id);
    }
}

void moe_scheduler::scheduler_worker_loop() {
    while (running_) {
        fetch_task_t task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !running_ || !pending_tasks_.empty();
            });

            if (!running_) break;

            task = pending_tasks_.front();
            pending_tasks_.erase(pending_tasks_.begin());
        }

        // Execute DIO read for this expert
        if (task.layer_idx < topo_.n_layer && task.expert_idx < topo_.n_expert) {
            const expert_info_t& exp_info = topo_.get_expert(task.layer_idx, task.expert_idx);
            expert_slot_t& slot = pool_.get_slot(task.assigned_slot);

            // Determine which shard file to read from
            dio_file_t* file = nullptr;
            if (!exp_info.sub_tensors.empty()) {
                uint32_t s_idx = exp_info.sub_tensors[0].shard_idx;
                if (s_idx < shard_files_.size()) {
                    file = shard_files_[s_idx];
                }
            }

            if (file) {
                bool ok = read_expert_sync(
                    &dio_engine_,
                    file,
                    exp_info.read_plan,
                    staging_buffer_.get(),
                    slot.raw_ptr
                );

                if (ok) {
                    pool_.mark_ready(task.assigned_slot);
                } else {
                    LOG_ERROR("Worker: failed to read Layer " << task.layer_idx << " Expert " << task.expert_idx);
                }
            } else {
                // If mocked or no file attached, mark ready directly
                pool_.mark_ready(task.assigned_slot);
            }
        }

        // Notify waiting compute thread
        {
            std::lock_guard<std::mutex> lock(sync_mutex_);
            completed_task_count_++;
            sync_cv_.notify_all();
        }
    }
}

} // namespace stream_moe