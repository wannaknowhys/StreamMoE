#include "scheduler/moe_scheduler.h"
#include "engine/subgraph_executor.h"
#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "loader/moe_loader.h"
#include "common/logger.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>

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

// Create a synthetic MoE model topology for scheduler testing
moe_model_topology_t create_mock_topology(uint32_t n_layer, uint32_t n_expert, uint32_t n_used) {
    moe_model_topology_t topo;
    topo.arch_name = "mock_moe";
    topo.n_layer = n_layer;
    topo.n_expert = n_expert;
    topo.n_expert_used = n_used;
    topo.expert_slot_size = 1024 * 64; // 64KB
    topo.expert_dio_staging_size = 1024 * 72; // 72KB
    topo.num_sub_tensors_per_expert = 3;

    topo.experts.resize(n_layer * n_expert);
    for (uint32_t l = 0; l < n_layer; ++l) {
        topo.moe_layers.push_back(l);
        for (uint32_t e = 0; e < n_expert; ++e) {
            size_t idx = l * n_expert + e;
            topo.experts[idx].layer_idx = static_cast<int32_t>(l);
            topo.experts[idx].expert_idx = static_cast<int32_t>(e);
            topo.experts[idx].total_expert_bytes = topo.expert_slot_size;
        }
    }
    return topo;
}

// Test 1: Hit/Miss routing split and Pin locking
bool test_routing_and_pin_locking() {
    auto topo = create_mock_topology(4, 16, 4);
    expert_pool pool(topo.expert_slot_size, 8);
    expert_stats_tracker stats;
    stats.init("temp/test_sched_stats.bin", 4, 16);
    auto dio_engine = async_dio_engine::create(16);

    std::vector<dio_file_t*> files;
    moe_scheduler scheduler(topo, pool, stats, *dio_engine, files);

    // Pre-fill slot 0 with (Layer 0, Expert 2) and slot 1 with (Layer 0, Expert 5)
    int32_t s0 = pool.allocate_or_evict_slot(0, 2, stats, 1);
    pool.mark_ready(s0);
    int32_t s1 = pool.allocate_or_evict_slot(0, 5, stats, 2);
    pool.mark_ready(s1);

    // Request routing for Layer 0 with selected experts [2, 5, 8, 9] (2 Hits, 2 Misses)
    scheduler.start();
    auto req = scheduler.route_and_prefetch(0, {2, 5, 8, 9}, 100);

    TEST_ASSERT(req.hit_slots.size() == 2, "Should have 2 hit slots");
    TEST_ASSERT(req.miss_experts.size() == 2, "Should have 2 miss experts");
    TEST_ASSERT(req.hit_slots[0] == s0 && req.hit_slots[1] == s1, "Hit slots should match s0, s1");

    // Verify hit slots are pinned
    TEST_ASSERT((pool.get_slot(s0).flags.load() & SLOT_PIN_LOCKED) != 0, "Slot s0 must be pinned");
    TEST_ASSERT((pool.get_slot(s1).flags.load() & SLOT_PIN_LOCKED) != 0, "Slot s1 must be pinned");

    // Wait for misses to be fetched by worker
    auto miss_slots = scheduler.wait_miss_ready(req, 2000);
    TEST_ASSERT(miss_slots.size() == 2, "Should receive 2 newly ready miss slots");

    // Release all slots
    std::vector<int32_t> all_used_slots = req.hit_slots;
    all_used_slots.insert(all_used_slots.end(), miss_slots.begin(), miss_slots.end());
    scheduler.release_layer_slots(all_used_slots);

    for (int32_t slot_id : all_used_slots) {
        TEST_ASSERT((pool.get_slot(slot_id).flags.load() & SLOT_PIN_LOCKED) == 0, "Slot must be unpinned");
    }

    scheduler.stop();
    TEST_PASS("test_routing_and_pin_locking");
    return true;
}

// Test 2: Dual-Thread Overlapping GEMM Computation & Pointer Rebind
bool test_dual_thread_overlapping_pipeline() {
    auto topo = create_mock_topology(4, 16, 4);
    expert_pool pool(topo.expert_slot_size, 8);
    expert_stats_tracker stats;
    stats.init("temp/test_sched_overlap.bin", 4, 16);
    auto dio_engine = async_dio_engine::create(16);

    std::vector<dio_file_t*> files;
    moe_scheduler scheduler(topo, pool, stats, *dio_engine, files);
    subgraph_executor executor(topo, pool, 64, 128);

    scheduler.start();

    const uint32_t n_tokens = 4;
    const uint32_t n_embd = 64;
    std::vector<float> input(n_tokens * n_embd, 1.0f);
    std::vector<float> output_accum(n_tokens * n_embd, 0.0f);

    // Pre-seed 1 hit
    int32_t s0 = pool.allocate_or_evict_slot(1, 0, stats, 1);
    pool.mark_ready(s0);

    // Route Layer 1 with experts [0 (hit), 1 (miss), 2 (miss)]
    auto req = scheduler.route_and_prefetch(1, {0, 1, 2}, 50);

    // 1. Immediately compute Hit GEMM concurrently while misses are in-flight
    std::vector<expert_compute_item_t> hit_items;
    for (int32_t slot : req.hit_slots) {
        hit_items.push_back({slot, 0.5f});
    }
    executor.compute_batch_rebind(hit_items, input.data(), output_accum.data(), n_tokens);

    // 2. Wait for Miss experts to complete
    auto miss_slots = scheduler.wait_miss_ready(req, 2000);
    TEST_ASSERT(miss_slots.size() == 2, "Miss slots count mismatch");

    // 3. Compute Miss GEMM and accumulate
    std::vector<expert_compute_item_t> miss_items;
    for (int32_t slot : miss_slots) {
        miss_items.push_back({slot, 0.25f});
    }
    executor.compute_batch_rebind(miss_items, input.data(), output_accum.data(), n_tokens);

    // Verify accumulation: hit (0.5) + miss1 (0.25) + miss2 (0.25) = 1.0 per element
    for (size_t i = 0; i < output_accum.size(); ++i) {
        TEST_ASSERT(std::abs(output_accum[i] - 1.0f) < 1e-5f, "Accumulation output mismatch");
    }

    std::vector<int32_t> all_slots = req.hit_slots;
    all_slots.insert(all_slots.end(), miss_slots.begin(), miss_slots.end());
    scheduler.release_layer_slots(all_slots);

    scheduler.stop();
    TEST_PASS("test_dual_thread_overlapping_pipeline");
    return true;
}

// Test 3: Multi-Layer Multi-Step Simulation with Dynamic Cache Eviction
bool test_multi_step_simulation() {
    auto topo = create_mock_topology(4, 32, 4);
    expert_pool pool(topo.expert_slot_size, 6); // Very constrained pool (6 slots for 4 layers x 32 experts)
    expert_stats_tracker stats;
    stats.init("temp/test_multistep.bin", 4, 32);
    auto dio_engine = async_dio_engine::create(16);

    std::vector<dio_file_t*> files;
    moe_scheduler scheduler(topo, pool, stats, *dio_engine, files);
    subgraph_executor executor(topo, pool, 32, 64);

    scheduler.start();

    // Simulate 20 decode steps
    for (uint64_t step = 1; step <= 20; ++step) {
        for (uint32_t l = 0; l < topo.n_layer; ++l) {
            // Selected experts for this step (some recurring hot experts, some shifting)
            uint32_t e0 = (l * 2) % topo.n_expert;
            uint32_t e1 = (l * 2 + 1) % topo.n_expert;
            uint32_t e2 = (l * 3 + step) % topo.n_expert;

            auto req = scheduler.route_and_prefetch(l, {e0, e1, e2}, step * 100 + l);
            auto miss_slots = scheduler.wait_miss_ready(req, 2000);

            std::vector<int32_t> all_slots = req.hit_slots;
            all_slots.insert(all_slots.end(), miss_slots.begin(), miss_slots.end());
            scheduler.release_layer_slots(all_slots);
        }
        stats.notify_tokens_generated(1);
    }

    scheduler.stop();
    TEST_PASS("test_multi_step_simulation");
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Running StreamMoE Phase 4 Unit Tests     " << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_routing_and_pin_locking();
    all_passed &= test_dual_thread_overlapping_pipeline();
    all_passed &= test_multi_step_simulation();

    std::cout << "===========================================" << std::endl;
    if (all_passed) {
        std::cout << "  ALL PHASE 4 TESTS PASSED SUCCESSFULLY!    " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  SOME PHASE 4 TESTS FAILED!                " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}