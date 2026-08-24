#include "engine/state_machine.h"
#include "engine/speculative_engine.h"
#include "scheduler/moe_scheduler.h"
#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "loader/moe_loader.h"
#include "common/logger.h"

#include <iostream>
#include <cassert>
#include <vector>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_PASS(name) \
    std::cout << "[+] TEST PASSED: " << name << std::endl

// Test 1: State Machine Transitions across Resource Scenarios
bool test_state_transitions() {
    state_machine sm(32);
    TEST_ASSERT(sm.current_state() == STATE_STEADY_5_0, "Initial state should be STEADY 5.0");

    // 1. CPU Bound scenario (State 1.0)
    runtime_telemetry_t t1;
    t1.cpu_load = 0.95;
    t1.gpu_load = 0.20;
    t1.pcie_load = 0.30;
    t1.draft_acc_rate = 0.80;
    auto s1 = sm.update_telemetry(t1);
    TEST_ASSERT(s1 == STATE_CPU_BOUND_1_0, "Should transition to CPU_BOUND_1_0");

    // 2. Draft Degraded scenario (State 1.5)
    runtime_telemetry_t t2 = t1;
    t2.draft_acc_rate = 0.15; // Low acceptance
    auto s2 = sm.update_telemetry(t2);
    TEST_ASSERT(s2 == STATE_DRAFT_DEGRADED_1_5, "Should transition to DRAFT_DEGRADED_1_5");
    TEST_ASSERT(!sm.current_policy().draft_enabled, "Draft should be paused in state 1.5");

    // 3. DMA Bus Contention (State 1.6)
    runtime_telemetry_t t3 = t1;
    t3.pcie_load = 0.90; // High DMA traffic
    auto s3 = sm.update_telemetry(t3);
    TEST_ASSERT(s3 == STATE_DMA_BUS_CONTENTION_1_6, "Should transition to DMA_BUS_CONTENTION_1_6");
    TEST_ASSERT(sm.current_policy().dma_throttle_us > 0, "DMA should be throttled in state 1.6");

    // 4. PCIe Thrashing (State 2.1)
    runtime_telemetry_t t4;
    t4.pcie_load = 0.90;
    t4.cache_hit_rate = 0.30; // Low cache hit rate -> slot churn
    auto s4 = sm.update_telemetry(t4);
    TEST_ASSERT(s4 == STATE_PCIE_THRASHING_2_1, "Should transition to PCIE_THRASHING_2_1");
    TEST_ASSERT(sm.current_policy().freeze_vram_cache, "VRAM cache should be frozen in state 2.1");

    // 5. GPU Bound (State 3.0)
    runtime_telemetry_t t5;
    t5.gpu_load = 0.95;
    t5.cpu_load = 0.40;
    auto s5 = sm.update_telemetry(t5);
    TEST_ASSERT(s5 == STATE_GPU_BOUND_3_0, "Should transition to GPU_BOUND_3_0");
    TEST_ASSERT(sm.current_policy().draft_k_step == 8, "Draft step K should increase in GPU bound state");

    // 6. Disk Bound (State 4.0)
    runtime_telemetry_t t6;
    t6.disk_load = 0.92;
    auto s6 = sm.update_telemetry(t6);
    TEST_ASSERT(s6 == STATE_DISK_BOUND_4_0, "Should transition to DISK_BOUND_4_0");
    TEST_ASSERT(sm.current_policy().prefetch_depth >= 4, "Prefetch depth should increase in disk bound state");

    // 7. RAM Bandwidth Wall (State 4.1)
    runtime_telemetry_t t7;
    t7.cpu_load = 0.70;
    t7.gpu_load = 0.10;
    t7.pcie_load = 0.10;
    t7.disk_load = 0.10;
    auto s7 = sm.update_telemetry(t7);
    TEST_ASSERT(s7 == STATE_RAM_BANDWIDTH_WALL_4_1, "Should transition to RAM_BANDWIDTH_WALL_4_1");
    TEST_ASSERT(sm.current_policy().cpu_gemm_threads == 16, "CPU threads should be halved in state 4.1");

    // 8. Return to Steady State (5.0)
    runtime_telemetry_t t8;
    t8.cpu_load = 0.40;
    t8.gpu_load = 0.40;
    t8.pcie_load = 0.30;
    t8.draft_acc_rate = 0.85;
    auto s8 = sm.update_telemetry(t8);
    TEST_ASSERT(s8 == STATE_STEADY_5_0, "Should return to STEADY_5_0");

    TEST_PASS("test_state_transitions");
    return true;
}

// Test 2: Emergency Reset Trigger (State 5.1)
bool test_emergency_reset() {
    state_machine sm(32);
    
    // Saturated system
    runtime_telemetry_t t_disaster;
    t_disaster.cpu_load  = 0.98;
    t_disaster.gpu_load  = 0.98;
    t_disaster.pcie_load = 0.95;
    t_disaster.disk_load = 0.95;

    auto s = sm.update_telemetry(t_disaster);
    TEST_ASSERT(s == STATE_SYSTEM_THRASHING_5_1, "Should trigger STATE_SYSTEM_THRASHING_5_1");
    
    auto policy = sm.current_policy();
    TEST_ASSERT(policy.emergency_reset_flag, "Emergency reset flag must be active");
    TEST_ASSERT(!policy.draft_enabled, "Draft must be disabled in emergency reset");
    TEST_ASSERT(policy.freeze_vram_cache, "VRAM cache must be frozen in emergency reset");

    TEST_PASS("test_emergency_reset");
    return true;
}

// Test 3: Speculative Decoding Orchestrator
bool test_speculative_decoding_loop() {
    moe_model_topology_t topo;
    topo.arch_name = "test_moe";
    topo.n_layer = 4;
    topo.n_expert = 16;
    topo.expert_slot_size = 1024 * 64;

    expert_pool pool(topo.expert_slot_size, 8);
    expert_stats_tracker stats;
    stats.init("temp/test_spec_stats.bin", 4, 16);
    auto dio_engine = async_dio_engine::create(16);
    std::vector<dio_file_t*> files;
    moe_scheduler scheduler(topo, pool, stats, *dio_engine, files);
    state_machine sm(16);

    speculative_engine spec_engine(topo, sm, scheduler);

    // 1. Test draft expert pre-aggregation
    std::vector<int32_t> draft_tokens = { 101, 102, 103, 104 };
    auto aggregated = spec_engine.aggregate_draft_experts(0, draft_tokens, 2);
    TEST_ASSERT(!aggregated.empty(), "Aggregated draft experts should not be empty");
    TEST_ASSERT(aggregated.size() <= 8, "Aggregated experts count bound check");

    // 2. Test verify and accept (3 matches out of 4)
    std::vector<int32_t> target_top_tokens = { 101, 102, 103, 999 };
    auto verify_res = spec_engine.verify_and_accept(draft_tokens, target_top_tokens);

    TEST_ASSERT(verify_res.n_accepted == 3, "Accepted tokens count should be 3");
    TEST_ASSERT(verify_res.next_token == 999, "Next token should be target model correction 999");
    TEST_ASSERT(verify_res.acceptance_rate == 0.75, "Acceptance rate should be 0.75");

    TEST_PASS("test_speculative_decoding_loop");
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Running StreamMoE Phase 5 Unit Tests     " << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_state_transitions();
    all_passed &= test_emergency_reset();
    all_passed &= test_speculative_decoding_loop();

    std::cout << "===========================================" << std::endl;
    if (all_passed) {
        std::cout << "  ALL PHASE 5 TESTS PASSED SUCCESSFULLY!    " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  SOME PHASE 5 TESTS FAILED!                " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}