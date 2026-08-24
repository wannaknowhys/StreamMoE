#include "engine/state_machine.h"
#include "common/logger.h"

namespace stream_moe {

state_machine::state_machine(uint32_t default_cpu_threads)
    : default_cpu_threads_(default_cpu_threads) {
    policy_.cpu_gemm_threads = default_cpu_threads_;
    policy_.draft_k_step     = 4;
    policy_.draft_enabled    = true;
}

system_state_id_t state_machine::update_telemetry(const runtime_telemetry_t& t) {
    std::lock_guard<std::mutex> lock(mutex_);

    system_state_id_t next_state = STATE_STEADY_5_0;

    // 1. Check for Emergency Thrashing (5.1): All saturated or severe thrashing
    if (t.cpu_load > 0.95 && t.gpu_load > 0.95 && t.pcie_load > 0.90 && t.disk_load > 0.90) {
        next_state = STATE_SYSTEM_THRASHING_5_1;
    }
    // 2. CPU Bound states (1.0, 1.5, 1.6)
    else if (t.cpu_load > 0.85 && t.gpu_load < 0.50) {
        if (t.pcie_load > 0.85) {
            next_state = STATE_DMA_BUS_CONTENTION_1_6;
        } else if (t.draft_acc_rate < 0.25) {
            next_state = STATE_DRAFT_DEGRADED_1_5;
        } else {
            next_state = STATE_CPU_BOUND_1_0;
        }
    }
    // 3. PCIe Bandwidth Bound states (2.0, 2.1)
    else if (t.pcie_load > 0.85) {
        if (t.cache_hit_rate < 0.40) {
            next_state = STATE_PCIE_THRASHING_2_1;
        } else {
            next_state = STATE_PCIE_BOUND_2_0;
        }
    }
    // 4. GPU Compute Bound (3.0)
    else if (t.gpu_load > 0.90 && t.cpu_load < 0.60) {
        next_state = STATE_GPU_BOUND_3_0;
    }
    // 5. Disk I/O Bound (4.0)
    else if (t.disk_load > 0.85) {
        next_state = STATE_DISK_BOUND_4_0;
    }
    // 6. Memory Bandwidth Wall (4.1): Moderate CPU load (0.5-0.7) but stalls
    else if (t.cpu_load > 0.60 && t.cpu_load < 0.80 && t.gpu_load < 0.30 && t.pcie_load < 0.30 && t.disk_load < 0.30) {
        next_state = STATE_RAM_BANDWIDTH_WALL_4_1;
    }
    // 7. Steady State (5.0)
    else {
        next_state = STATE_STEADY_5_0;
    }

    // Apply policy adjustments based on state transition
    if (next_state != current_state_) {
        LOG_INFO("State Machine Transition: [" << state_name(current_state_) << "] -> [" 
                 << state_name(next_state) << "]");
    }
    current_state_ = next_state;

    switch (current_state_) {
        case STATE_CPU_BOUND_1_0:
            policy_.draft_k_step      = 2;
            policy_.draft_enabled     = true;
            policy_.dma_throttle_us   = 0;
            policy_.freeze_vram_cache = false;
            break;

        case STATE_DRAFT_DEGRADED_1_5:
            policy_.draft_k_step      = 1;
            policy_.draft_enabled     = false; // Temporarily disable draft
            break;

        case STATE_DMA_BUS_CONTENTION_1_6:
            policy_.dma_throttle_us   = 200; // Throttle DMA to reduce bus contention
            break;

        case STATE_PCIE_BOUND_2_0:
            policy_.prefetch_depth    = 1; // Reduce prefetch lookahead
            break;

        case STATE_PCIE_THRASHING_2_1:
            policy_.freeze_vram_cache = true; // Lock current VRAM cache slots
            break;

        case STATE_GPU_BOUND_3_0:
            policy_.draft_k_step      = 8; // Buy GPU time with larger draft step
            break;

        case STATE_DISK_BOUND_4_0:
            policy_.prefetch_depth    = 4; // Deepen prefetch lookahead
            break;

        case STATE_RAM_BANDWIDTH_WALL_4_1:
            policy_.cpu_gemm_threads  = default_cpu_threads_ / 2; // Cut threads to reduce DDR contention
            break;

        case STATE_SYSTEM_THRASHING_5_1:
            trigger_emergency_reset();
            break;

        case STATE_STEADY_5_0:
        default:
            policy_.cpu_gemm_threads  = default_cpu_threads_;
            policy_.draft_k_step      = 4;
            policy_.draft_enabled     = true;
            policy_.dma_throttle_us   = 0;
            policy_.freeze_vram_cache = false;
            policy_.prefetch_depth    = 2;
            policy_.emergency_reset_flag = false;
            break;
    }

    return current_state_;
}

void state_machine::trigger_emergency_reset() {
    LOG_WARN("EMERGENCY RESET TRIGGERED: Purging prefetch queues and resetting dynamic cache slots to safe static mode!");
    policy_.draft_enabled        = false;
    policy_.draft_k_step         = 1;
    policy_.dma_throttle_us      = 500;
    policy_.freeze_vram_cache    = true;
    policy_.prefetch_depth       = 0;
    policy_.emergency_reset_flag = true;
    current_state_               = STATE_SYSTEM_THRASHING_5_1;
}

const char* state_machine::state_name(system_state_id_t state) const {
    switch (state) {
        case STATE_STEADY_5_0:             return "5.0 STEADY_STATE";
        case STATE_CPU_BOUND_1_0:          return "1.0 CPU_BOUND_PROMOTE";
        case STATE_DRAFT_DEGRADED_1_5:     return "1.5 DRAFT_DEGRADED_PAUSE";
        case STATE_DMA_BUS_CONTENTION_1_6: return "1.6 DMA_BUS_THROTTLE";
        case STATE_PCIE_BOUND_2_0:         return "2.0 PCIE_BOUND_LOCAL_GEMM";
        case STATE_PCIE_THRASHING_2_1:     return "2.1 PCIE_THRASHING_FREEZE";
        case STATE_GPU_BOUND_3_0:          return "3.0 GPU_BOUND_OFFLOAD_CPU";
        case STATE_DISK_BOUND_4_0:         return "4.0 DISK_BOUND_DEEPEN_PREFETCH";
        case STATE_RAM_BANDWIDTH_WALL_4_1: return "4.1 RAM_BANDWIDTH_CUT_THREADS";
        case STATE_SYSTEM_THRASHING_5_1:   return "5.1 SYSTEM_THRASHING_EMERGENCY_RESET";
        default: return "UNKNOWN_STATE";
    }
}

} // namespace stream_moe