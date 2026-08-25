#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "common/logger.h"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace stream_moe;

void test_pinned_pool_allocation() {
    std::cout << "[Test 1] Testing Pinned Pool VirtualAlloc & VirtualLock...\n";
    size_t slot_size = 1024 * 1024 * 2; // 2MB
    uint32_t num_slots = 8;
    expert_pool pool(slot_size, num_slots);

    assert(pool.num_slots() == 8);
    assert(pool.slot_size() == 2 * 1024 * 1024);
    assert(pool.total_bytes() == 16 * 1024 * 1024);
    assert(pool.base_ptr() != nullptr);

    std::cout << "[+] TEST PASSED: test_pinned_pool_allocation\n";
}

void test_expert_stats_lifecycle() {
    std::cout << "[Test 2] Testing EST1 Binary File Lifecycle...\n";
    std::string test_stats_path = "temp/test_expert_stats.bin";
    if (std::filesystem::exists(test_stats_path)) {
        std::filesystem::remove(test_stats_path);
    }

    expert_stats_tracker stats;
    stats.init(test_stats_path, 16, 64, 8192);

    for (int i = 0; i < 100; ++i) {
        stats.record_access(2, 5);
        stats.record_access(3, 10);
    }

    assert(stats.get_global_count(2, 5) == 100);
    assert(stats.get_global_count(3, 10) == 100);
    assert(stats.get_global_count(0, 0) == 0);

    stats.flush();

    expert_stats_tracker loaded_stats;
    loaded_stats.init(test_stats_path, 16, 64, 8192);
    assert(loaded_stats.get_global_count(2, 5) == 100);
    assert(loaded_stats.get_global_count(3, 10) == 100);

    loaded_stats.notify_tokens_generated(8192);

    std::cout << "[+] TEST PASSED: test_expert_stats_lifecycle\n";
}

void test_pure_lru_and_hybrid_eviction() {
    std::cout << "[Test 3] Testing Pure LRU vs Hybrid Eviction Policies...\n";
    std::string test_stats_path = "temp/test_eviction_stats.bin";
    expert_stats_tracker stats;
    stats.init(test_stats_path, 4, 16, 8192);

    // Give expert (0, 0) a very high historical frequency
    for (int i = 0; i < 1000; ++i) {
        stats.record_access(0, 0);
    }

    // 1. Test PURE_LRU (Pure LRU MUST evict (0,0) if it is the oldest, despite high frequency!)
    {
        expert_pool lru_pool(1024 * 1024, 3, eviction_policy_t::PURE_LRU);
        int32_t s0 = lru_pool.allocate_or_evict_slot(0, 0, stats, 10); // seq=10 (oldest)
        lru_pool.mark_ready(s0);

        int32_t s1 = lru_pool.allocate_or_evict_slot(0, 1, stats, 20); // seq=20
        lru_pool.mark_ready(s1);

        int32_t s2 = lru_pool.allocate_or_evict_slot(0, 2, stats, 30); // seq=30
        lru_pool.mark_ready(s2);

        assert(s0 == 0 && s1 == 1 && s2 == 2);

        // Now allocate (0, 3) at seq=40 -> Under PURE LRU, slot 0 (seq=10) MUST be evicted!
        int32_t victim = lru_pool.allocate_or_evict_slot(0, 3, stats, 40);
        assert(victim == 0);
        assert(lru_pool.find_slot(0, 0) == -1); // (0,0) was evicted!
    }

    // 2. Test HYBRID_EST1 (Hybrid will protect (0,0) because its frequency is 1000!)
    {
        expert_pool hybrid_pool(1024 * 1024, 3, eviction_policy_t::HYBRID_EST1);
        int32_t s0 = hybrid_pool.allocate_or_evict_slot(0, 0, stats, 10); // seq=10, freq=1000
        hybrid_pool.mark_ready(s0);

        int32_t s1 = hybrid_pool.allocate_or_evict_slot(0, 1, stats, 20); // seq=20, freq=0
        hybrid_pool.mark_ready(s1);

        int32_t s2 = hybrid_pool.allocate_or_evict_slot(0, 2, stats, 30); // seq=30, freq=0
        hybrid_pool.mark_ready(s2);

        // Allocate (0, 3) at seq=40 -> Under Hybrid, slot 1 (seq=20, freq=0) is evicted instead of slot 0!
        int32_t victim = hybrid_pool.allocate_or_evict_slot(0, 3, stats, 40);
        assert(victim == 1); // Slot 1 evicted!
        assert(hybrid_pool.find_slot(0, 0) != -1); // (0,0) was protected by frequency!
    }

    std::cout << "[+] TEST PASSED: test_pure_lru_and_hybrid_eviction\n";
}

int main() {
    std::cout << "===========================================\n"
              << "  Running StreamMoE Phase 2 Unit Tests     \n"
              << "===========================================\n";

    test_pinned_pool_allocation();
    test_expert_stats_lifecycle();
    test_pure_lru_and_hybrid_eviction();

    std::cout << "===========================================\n"
              << "  ALL PHASE 2 TESTS PASSED SUCCESSFULLY!    \n"
              << "===========================================\n";
    return 0;
}