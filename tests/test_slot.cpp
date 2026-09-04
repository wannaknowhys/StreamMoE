// Unit tests for the route B control plane (src/backend/slot.h).
// Model-agnostic, cross-platform (no llama.cpp dependency).
#include "backend/slot.h"

#include <algorithm>
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { std::printf("[-] ASSERTION FAILED: %s (line %d)\n", msg, __LINE__); std::fflush(stdout); return false; } \
} while(0)

static bool test_slot_meta_lifecycle() {
    slot_meta s;
    TEST_ASSERT(slot_word_state(s.load()) == SLOT_EMPTY, "initial state EMPTY");
    TEST_ASSERT(slot_word_refcount(s.load()) == 0, "initial refcount 0");
    TEST_ASSERT(slot_word_generation(s.load()) == 0, "initial generation 0");

    // pin on non-ready must fail
    TEST_ASSERT(s.try_pin() < 0, "pin on EMPTY fails");

    // reload -> IO_INFLIGHT, generation bumps
    uint32_t g1 = s.begin_reload();
    TEST_ASSERT(slot_word_state(s.load()) == SLOT_IO_INFLIGHT, "reload -> IO_INFLIGHT");
    TEST_ASSERT(g1 == 1, "generation 1 after first reload");
    TEST_ASSERT(s.try_pin() < 0, "pin on IO_INFLIGHT fails");

    s.mark_ready();
    TEST_ASSERT(slot_word_state(s.load()) == SLOT_READY, "ready after mark_ready");

    int64_t gen = s.try_pin();
    TEST_ASSERT(gen == 1, "pin returns generation 1");
    TEST_ASSERT(slot_word_refcount(s.load()) == 1, "refcount 1 after pin");
    s.unpin();
    TEST_ASSERT(slot_word_refcount(s.load()) == 0, "refcount 0 after unpin");

    // second reload bumps generation; old pin generation invalidates
    uint32_t g2 = s.begin_reload();
    TEST_ASSERT(g2 == 2, "generation 2 after second reload");
    s.mark_ready();
    int64_t gen2 = s.try_pin();
    TEST_ASSERT(gen2 == 2, "pin returns generation 2");

    // wait_ready with matching generation succeeds (already ready)
    TEST_ASSERT(s.wait_ready(g2), "wait_ready on READY ok");
    s.unpin();

    // eviction blocks new pins (EVICTING is terminal until begin_reload reuses it)
    TEST_ASSERT(s.begin_evict(), "evict CAS wins");
    TEST_ASSERT(s.try_pin() < 0, "pin fails during EVICTING");
    TEST_ASSERT(slot_word_state(s.load()) == SLOT_EVICTING, "EVICTING state");

    // FAILED state: wait_ready returns false
    s.begin_reload();       // reuse EVICTING slot
    s.mark_failed();
    TEST_ASSERT(slot_word_state(s.load()) == SLOT_FAILED, "FAILED state");
    TEST_ASSERT(!s.wait_ready(3), "wait_ready on FAILED returns false");

    std::printf("[+] test_slot_meta_lifecycle PASSED\n"); std::fflush(stdout);
    return true;
}

static bool test_slot_wake_thread() {
    // thread A waits READY; main thread loads + marks ready -> A wakes
    slot_meta s;
    uint32_t g = s.begin_reload();

    std::atomic<bool> done{false};
    std::thread waiter([&] {
        bool ok = s.wait_ready(g);
        done.store(ok, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TEST_ASSERT(!done.load(std::memory_order_acquire), "waiter still blocked before ready");
    s.mark_ready();

    waiter.join();
    TEST_ASSERT(done.load(std::memory_order_acquire), "waiter woke and saw READY");

    std::printf("[+] test_slot_wake_thread PASSED\n"); std::fflush(stdout);
    return true;
}

static bool test_expert_directory() {
    expert_directory dir(4, 256, 2, 1024);   // n_pools = 2 (multi-pool layout)
    TEST_ASSERT(dir.n_pools() == 2, "pool count");
    TEST_ASSERT(dir.find(0, 0, 0) == SLOT_UNASSIGNED, "unassigned initially (pool 0)");
    TEST_ASSERT(dir.scan(0, 0, nullptr) == SLOT_UNASSIGNED, "unassigned across pools");

    dir.set(2, 17, 0, 42);
    TEST_ASSERT(dir.find(2, 17, 0) == 42, "find after set (pool 0)");
    TEST_ASSERT(dir.find(2, 17, 1) == SLOT_UNASSIGNED, "other pool untouched");
    uint32_t scan_pool = 99;
    TEST_ASSERT(dir.scan(2, 17, &scan_pool) == 42 && scan_pool == 0, "scan returns slot + pool");
    TEST_ASSERT(dir.find(0, 0, 0) == SLOT_UNASSIGNED, "other entry untouched");

    uint32_t v0 = dir.version(2, 17);
    dir.set(2, 17, 0, 43);
    TEST_ASSERT(dir.version(2, 17) == v0 + 1, "version bumped on remap (pool 0)");
    dir.set(2, 17, 1, 55);                    // second pool mapping
    TEST_ASSERT(dir.version(2, 17) == v0 + 2, "version bumped on cross-pool remap");
    TEST_ASSERT(dir.scan(2, 17, &scan_pool) == 43 && scan_pool == 0, "scan picks lowest pool first");

    dir.touch_last_used(2, 17, 1000);
    TEST_ASSERT(dir.last_used(2, 17) == 1000, "last_used stored");
    TEST_ASSERT(dir.last_used(0, 0) == 0, "other entry last_used zero");

    std::printf("[+] test_expert_directory PASSED\n"); std::fflush(stdout);
    return true;
}

static bool test_mpsc_queue() {
    mpsc_alloc_queue q(8);
    for (uint32_t i = 0; i < 8; ++i) {
        slot_request_t r;
        r.layer = 1;
        r.n_load_target = i + 1;
        r.needed[0] = 1ull << i;   // expert i
        q.push(r);
    }
    slot_request_t r;
    uint32_t n = 0;
    while (q.pop(r)) {
        TEST_ASSERT(r.layer == 1, "layer preserved");
        TEST_ASSERT((r.needed[0] >> n) & 1ull, "pop order preserved");
        ++n;
    }
    TEST_ASSERT(n == 8, "all 8 popped");

    // multi-producer: 4 threads push distinct expert bits, single consumer pops
    mpsc_alloc_queue q2(64);
    std::atomic<int> producers_done{0};
    auto producer = [&](uint32_t base) {
        for (uint32_t i = 0; i < 16; ++i) {
            slot_request_t r;
            r.layer = 2;
            r.needed[0] = 1ull << (base + i);
            q2.push(r);
        }
        producers_done.fetch_add(1, std::memory_order_relaxed);
    };
    std::thread t0(producer, 0), t1(producer, 16), t2(producer, 32), t3(producer, 48);
    std::vector<uint64_t> seen;
    while (producers_done.load(std::memory_order_relaxed) < 4 || seen.size() < 64) {
        slot_request_t x;
        while (q2.pop(x)) seen.push_back(x.needed[0]);
        if (seen.size() < 64) std::this_thread::yield();
    }
    t0.join(); t1.join(); t2.join(); t3.join();
    // drain any remaining
    slot_request_t x;
    while (q2.pop(x)) seen.push_back(x.needed[0]);
    TEST_ASSERT(seen.size() == 64, "64 messages from 4 producers");
    std::sort(seen.begin(), seen.end());
    for (uint32_t i = 0; i < 64; ++i) {
        TEST_ASSERT(seen[static_cast<size_t>(i)] == (1ull << i), "no loss/dup");
    }

    std::printf("[+] test_mpsc_queue PASSED\n"); std::fflush(stdout);
    return true;
}

int main() {
    bool ok = true;
    ok = test_slot_meta_lifecycle() && ok;
    ok = test_slot_wake_thread() && ok;
    ok = test_expert_directory() && ok;
    ok = test_mpsc_queue() && ok;
    if (ok) {
        std::printf("ALL SLOT CONTROL PLANE TESTS PASSED\n");
        return 0;
    }
    std::printf("SOME SLOT CONTROL PLANE TESTS FAILED\n");
    return 1;
}
