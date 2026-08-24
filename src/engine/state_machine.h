#pragma once

#include "common/types.h"
#include <string>
#include <functional>
#include <atomic>
#include <mutex>

namespace stream_moe {

enum system_state_id_t {
    STATE_STEADY_5_0             = 500, // Balanced sweet spot
    STATE_CPU_BOUND_1_0          = 100, // CPU bound, promote L1 to VRAM / offload draft
    STATE_DRAFT_DEGRADED_1_5     = 150, // Low draft acceptance, reduce/disable draft
    STATE_DMA_BUS_CONTENTION_1_6 = 160, // DMA contending memory bus, throttle DMA
    STATE_PCIE_BOUND_2_0         = 200, // PCIe bottleneck, fall back marginal experts to CPU
    STATE_PCIE_THRASHING_2_1     = 210, // Slot churning, freeze VRAM cache
    STATE_GPU_BOUND_3_0          = 300, // GPU compute bound, offload secondary experts to CPU
    STATE_DISK_BOUND_4_0         = 400, // Disk IO bottleneck, increase prefetch depth
    STATE_RAM_BANDWIDTH_WALL_4_1 = 410, // DDR bus saturated, reduce CPU threads
    STATE_SYSTEM_THRASHING_5_1   = 510, // All saturated, trigger Emergency Reset
};

struct runtime_telemetry_t {
    double cpu_load       = 0.0; // [0.0, 1.0]
    double gpu_load       = 0.0; // [0.0, 1.0]
    double pcie_load      = 0.0; // [0.0, 1.0]
    double disk_load      = 0.0; // [0.0, 1.0]
    double draft_acc_rate = 0.0; // [0.0, 1.0]
    double cache_hit_rate = 0.0; // [0.0, 1.0]
};

struct engine_policy_actions_t {
    uint32_t draft_k_step         = 4;     // Active draft speculative step K
    bool     draft_enabled        = true;  // Draft model active
    uint32_t cpu_gemm_threads     = 16;    // Active threads for CPU GEMM
    uint32_t dma_throttle_us      = 0;     // Sleep delay per DMA transfer in microseconds
    bool     freeze_vram_cache    = false; // Freeze A/B slot replacement
    uint32_t prefetch_depth       = 2;     // Token prefetch lookahead
    bool     emergency_reset_flag = false; // Emergency reset triggered
};

class state_machine {
public:
    state_machine(uint32_t default_cpu_threads = 16);
    ~state_machine() = default;

    // Update telemetry and evaluate state transition
    system_state_id_t update_telemetry(const runtime_telemetry_t& telemetry);

    // Force trigger emergency reset
    void trigger_emergency_reset();

    // Query active state and policy
    system_state_id_t       current_state() const { return current_state_; }
    engine_policy_actions_t current_policy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return policy_;
    }

    const char* state_name(system_state_id_t state) const;

private:
    uint32_t                default_cpu_threads_ = 16;
    system_state_id_t       current_state_       = STATE_STEADY_5_0;
    engine_policy_actions_t policy_;
    mutable std::mutex      mutex_;
};

} // namespace stream_moe