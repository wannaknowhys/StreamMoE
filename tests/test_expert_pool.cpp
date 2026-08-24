#include "pool/expert_pool.h"
#include "pool/expert_stats.h"
#include "common/logger.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <thread>

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

// Test 1: Pinned Pool Memory & Alignment
bool test_pinned_pool_allocation() {
    const size_t slot_size = 1024 * 1024 * 2; // 2MB
    const uint32_t num_slots = 8;

    expert_pool pool(slot_size, num_slots);
    TEST_ASSERT(pool.num_slots() == num_slots, "num_slots mismatch");
    TEST_ASSERT(pool.slot_size() == slot_size, "slot_size mismatch");
    TEST_ASSERT(pool.total_bytes() == slot_size * num_slots, "total_bytes mismatch");
    TEST_ASSERT(pool.base_ptr() != nullptr, "base_ptr is null");
    TEST_ASSERT(is_aligned(pool.base_ptr(), 4096), "base_ptr not 4KB aligned");

    // Verify all slots are correctly initialized and writeable
    for (uint32_t i = 0; i < num_slots; ++i) {
        auto& slot = pool.get_slot(i);
        TEST_ASSERT(slot.layer_idx == -1, "slot layer_idx should be -1");
        TEST_ASSERT(slot.flags.load() == SLOT_EMPTY, "slot flags should be SLOT_EMPTY");
        TEST_ASSERT(slot.raw_ptr == pool.base_ptr() + i * slot_size, "slot raw_ptr mismatch");
        
        // Write pattern to verify memory page committed
        std::memset(slot.raw_ptr, 0x55, 4096);
        TEST_ASSERT(slot.raw_ptr[0] == 0x55, "Memory write check failed");
    }

    TEST_PASS("test_pinned_pool_allocation");
    return true;
}

// Test 2: Expert Stats Lifecycle (EST1 file format, auto-create, 8192-token sync, exit flush)
bool test_expert_stats_lifecycle() {
    const std::string stats_file = "temp/test_expert_stats.bin";
    std::filesystem::remove(stats_file);

    const uint32_t n_layer = 16;
    const uint32_t n_expert = 64;

    // Scope 1: Initialize and generate accesses
    {
        expert_stats_tracker tracker;
        bool ok = tracker.init(stats_file, n_layer, n_expert, 8192);
        TEST_ASSERT(ok, "tracker.init failed");
        TEST_ASSERT(std::filesystem::exists(stats_file), "stats file should be created on init");

        // Record accesses to Layer 0 Expert 5, Layer 2 Expert 10
        for (int i = 0; i < 100; ++i) {
            tracker.record_access(0, 5);
        }
        for (int i = 0; i < 30; ++i) {
            tracker.record_access(2, 10);
        }

        TEST_ASSERT(tracker.get_global_count(0, 5) == 100, "count(0,5) should be 100");
        TEST_ASSERT(tracker.get_global_count(2, 10) == 30, "count(2,10) should be 30");
        TEST_ASSERT(tracker.get_global_count(0, 0) == 0, "count(0,0) should be 0");

        // Simulate generating 4000 tokens (less than 8192 threshold)
        tracker.notify_tokens_generated(4000);
        // Destructor will automatically flush upon leaving scope
    }

    // Scope 2: Reload and verify persistence
    {
        expert_stats_tracker tracker;
        bool ok = tracker.init(stats_file, n_layer, n_expert, 8192);
        TEST_ASSERT(ok, "tracker.init reload failed");

        TEST_ASSERT(tracker.get_global_count(0, 5) == 100, "reloaded count(0,5) should be 100");
        TEST_ASSERT(tracker.get_global_count(2, 10) == 30, "reloaded count(2,10) should be 30");

        // Test periodic 8192-token sync trigger
        tracker.record_access(5, 20);
        tracker.notify_tokens_generated(9000); // Exceeds 8192 threshold
    }

    // Scope 3: Verify 8192 sync was written
    {
        expert_stats_tracker tracker;
        tracker.init(stats_file, n_layer, n_expert, 8192);
        TEST_ASSERT(tracker.get_global_count(5, 20) == 1, "synced count(5,20) should be 1");
    }

    TEST_PASS("test_expert_stats_lifecycle");
    return true;
}

// Test 3: Adaptive Frequency Convergence (Decaying towards recent window)
bool test_adaptive_frequency_convergence() {
    expert_stats_tracker tracker;
    const std::string stats_file = "temp/test_adaptive_decay.bin";
    std::filesystem::remove(stats_file);

    tracker.init(stats_file, 8, 32);

    // Expert A (old hot expert, 500 accesses in the past)
    for (int i = 0; i < 500; ++i) {
        tracker.record_access(1, 0);
    }
    double score_a_old = tracker.get_adaptive_frequency(1, 0);
    TEST_ASSERT(score_a_old > 0.0, "Score A should be positive");

    // Simulate 4000 tokens generated where Expert B is actively used while Expert A is completely inactive
    for (int step = 0; step < 100; ++step) {
        for (int k = 0; k < 5; ++k) {
            tracker.record_access(1, 1); // Expert B active
        }
        tracker.notify_tokens_generated(40);
    }

    double score_a_new = tracker.get_adaptive_frequency(1, 0);
    double score_b_new = tracker.get_adaptive_frequency(1, 1);

    // Old Expert A's adaptive score should have decayed significantly
    TEST_ASSERT(score_a_new < score_a_old, "Old expert score should decay");
    TEST_ASSERT(score_b_new > score_a_new, "Recent expert score should surpass decayed old expert");
    TEST_PASS("test_adaptive_frequency_convergence");
    return true;
}

// Test 4: Pin Lock Protection & Eviction Policy
bool test_pin_lock_and_eviction() {
    const size_t slot_size = 1024 * 1024; // 1MB
    const uint32_t num_slots = 4;        // Small pool of 4 slots

    expert_pool pool(slot_size, num_slots);
    expert_stats_tracker stats;
    stats.init("temp/test_eviction_stats.bin", 4, 16);

    uint64_t seq = 1;

    // 1. Fill all 4 slots with Layer 0 Experts [0, 1, 2, 3]
    int32_t s0 = pool.allocate_or_evict_slot(0, 0, stats, seq++);
    int32_t s1 = pool.allocate_or_evict_slot(0, 1, stats, seq++);
    int32_t s2 = pool.allocate_or_evict_slot(0, 2, stats, seq++);
    int32_t s3 = pool.allocate_or_evict_slot(0, 3, stats, seq++);

    TEST_ASSERT(s0 == 0 && s1 == 1 && s2 == 2 && s3 == 3, "Slots should be 0,1,2,3");
    for (int i = 0; i < 4; ++i) pool.mark_ready(i);

    // 2. Pin slots 0 and 1 (they are currently computing in GEMM)
    TEST_ASSERT(pool.pin_slot(s0), "Pin s0");
    TEST_ASSERT(pool.pin_slot(s1), "Pin s1");

    // Make Expert 2 frequently accessed so its adaptive frequency is high
    for (int i = 0; i < 50; ++i) stats.record_access(0, 2);

    // Now request slot for a new Expert (0, 4)
    // Slot 0 & 1 are pinned (protected).
    // Slot 2 has high frequency and recent access.
    // Slot 3 has low frequency and older access -> Slot 3 MUST be evicted!
    int32_t victim = pool.allocate_or_evict_slot(0, 4, stats, seq++);
    TEST_ASSERT(victim == s3, "Slot 3 should be selected as victim for eviction");
    TEST_ASSERT(pool.find_slot(0, 3) == -1, "Evicted expert should no longer be found");
    pool.mark_ready(victim);

    // 3. Verify pinned slots 0 and 1 were NOT evicted
    TEST_ASSERT(pool.find_slot(0, 0) == s0, "Pinned slot 0 must remain present");
    TEST_ASSERT(pool.find_slot(0, 1) == s1, "Pinned slot 1 must remain present");

    // Unpin slot 0
    pool.unpin_slot(s0);
    TEST_ASSERT((pool.get_slot(s0).flags.load() & SLOT_PIN_LOCKED) == 0, "Slot 0 should be unpinned");

    TEST_PASS("test_pin_lock_and_eviction");
    return true;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Running StreamMoE Phase 2 Unit Tests     " << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_pinned_pool_allocation();
    all_passed &= test_expert_stats_lifecycle();
    all_passed &= test_adaptive_frequency_convergence();
    all_passed &= test_pin_lock_and_eviction();

    std::cout << "===========================================" << std::endl;
    if (all_passed) {
        std::cout << "  ALL PHASE 2 TESTS PASSED SUCCESSFULLY!    " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 0;
    } else {
        std::cerr << "  SOME PHASE 2 TESTS FAILED!                " << std::endl;
        std::cout << "===========================================" << std::endl;
        return 1;
    }
}