// Unit tests for the mixed-region split planner (src/backend/mix_split.cpp).
// Pure, model-agnostic: feeds synthetic routing ids + a per-expert pool map
// and checks the peel rounds tile every (token, routed expert) exactly once
// (zero waste) with llama-compatible ids/scatter layout.
#include "backend/mix_split.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { std::printf("[-] ASSERTION FAILED: %s (line %d)\n", msg, __LINE__); std::fflush(stdout); return false; } \
} while(0)

// Reference coverage: every (t,k) with a valid expert must appear in exactly
// one round of its pool, with the same expert id, in llama slot order.
static bool check_coverage(const mix_plan_t& plan,
                           const std::vector<int32_t>& ids,
                           const std::vector<int32_t>& expert_pool,
                           uint32_t n_k, uint32_t n_t) {
    // covered[t][k] = round that produced this slot, or -1
    std::vector<std::vector<int>> covered(n_t, std::vector<int>(n_k, -1));
    uint32_t total_emitted = 0;
    for (const auto& r : plan.rounds) {
        TEST_ASSERT(r.ids.size() == (size_t)r.width * r.n_active, "ids size == width*n_active");
        TEST_ASSERT(r.scatter.size() == r.ids.size(), "scatter size == ids size");
        for (uint32_t a = 0; a < r.n_active; ++a) {
            for (uint32_t s = 0; s < r.width; ++s) {
                const size_t idx = (size_t)a * r.width + s;
                const auto& sc = r.scatter[idx];
                TEST_ASSERT(sc.t < n_t && sc.k < n_k, "scatter within bounds");
                // expert id must equal the source ids at (t,k)
                const int32_t expect = ids[sc.t * n_k + sc.k];
                TEST_ASSERT(r.ids[idx] == expect, "round ids matches source");
                TEST_ASSERT(covered[sc.t][sc.k] == -1, "each (t,k) covered once");
                covered[sc.t][sc.k] = 1;
                ++total_emitted;
            }
        }
    }
    // every valid (t,k) must be covered
    for (uint32_t t = 0; t < n_t; ++t)
        for (uint32_t k = 0; k < n_k; ++k) {
            const int32_t e = ids[t * n_k + k];
            if (e >= 0 && expert_pool[e] >= 0) {
                TEST_ASSERT(covered[t][k] != -1, "valid (t,k) covered");
            } else {
                TEST_ASSERT(covered[t][k] == -1, "invalid (t,k) never covered");
            }
        }
    (void)total_emitted;
    return true;
}

// ---- single pool: everything lives in pool 0 (pure CPU) -------------------
static bool test_pure_single_pool() {
    const uint32_t n_k = 4, n_t = 3, n_expert = 8;
    std::vector<int32_t> ids = {
        // t0        t1        t2
        0,1,2,3,  1,3,5,7,  2,4,6,0
    };
    std::vector<int32_t> expert_pool(n_expert, 0);   // all in pool 0
    mix_plan_t plan = build_mix_plan(ids.data(), n_k, n_t, expert_pool.data(), n_expert, 1);
    // one pool, every token hits 4 -> buckets[4] = 3 -> one round width 4
    TEST_ASSERT(plan.rounds.size() == 1, "one round");
    if (!plan.rounds.empty()) {
        TEST_ASSERT(plan.rounds[0].width == 4, "width 4");
        TEST_ASSERT(plan.rounds[0].n_active == 3, "3 active tokens");
    }
    TEST_ASSERT(check_coverage(plan, ids, expert_pool, n_k, n_t), "coverage");
    return true;
}

// ---- two pools, matching the doc example {2:19, 5:48} shape ---------------
// n_t = 5: two tokens hit pool-0 twice, three tokens hit it five times.
static bool test_mixed_two_pools_bucket_example() {
    const uint32_t n_k = 8, n_t = 5, n_expert = 16;
    // tokens 0,1 route 2 experts to pool 0 (6 to pool 1); tokens 2..4 route 5
    // to pool 0 (3 to pool 1).
    std::vector<int32_t> ids = {
        // t0: 2x pool0(e0,e1) + 6x pool1(e8..e13)
        0,1,8,9,10,11,12,13,
        // t1: same shape
        2,3,8,9,10,11,12,13,
        // t2: 5x pool0(e0..e4) + 3x pool1(e8,e9,e10)
        0,1,2,3,4, 8,9,10,
        // t3
        0,1,2,3,4, 8,9,10,
        // t4
        0,1,2,3,4, 8,9,10,
    };
    std::vector<int32_t> expert_pool(n_expert, 1);   // pool 1 default
    for (int e = 0; e < 8; ++e) expert_pool[e] = 0;  // experts 0..7 in pool 0
    mix_plan_t plan = build_mix_plan(ids.data(), n_k, n_t, expert_pool.data(), n_expert, 2);

    // pool 0: buckets c2=2, c5=3 -> bounds v=[2,5], w=[2,3]; two rounds:
    //   round1 width 2, active 5; round2 width 3, active 3.
    // pool 1: tokens hit 6/3 -> c3=3, c6=2 -> bounds v=[3,6], w=[3,3]; two rounds
    //   round1 width 3 active 5, round2 width 3 active 2.
    uint32_t p0_rounds = 0, p1_rounds = 0;
    uint32_t p0_flops = 0, p1_flops = 0;
    for (const auto& r : plan.rounds) {
        if (r.pool == 0) { ++p0_rounds; p0_flops += r.width * r.n_active; }
        else             { ++p1_rounds; p1_flops += r.width * r.n_active; }
    }
    TEST_ASSERT(p0_rounds == 2, "pool0 has 2 rounds");
    TEST_ASSERT(p1_rounds == 2, "pool1 has 2 rounds");
    // FLOPS: pool0 = 5*2 + 3*3 = 19 == 2*2 + 3*5 = 19 ; pool1 = 5*3+2*3=21 == 3*3+2*6=21
    TEST_ASSERT(p0_flops == 19, "pool0 zero waste");
    TEST_ASSERT(p1_flops == 21, "pool1 zero waste");
    TEST_ASSERT(check_coverage(plan, ids, expert_pool, n_k, n_t), "coverage");
    return true;
}

// ---- per-token worst case: every token different count ---------------------
static bool test_per_token_histogram() {
    const uint32_t n_k = 4, n_t = 4, n_expert = 8;
    // token t hits pool0 (t+1) times -> counts 1,2,3,4 -> 4 rounds.
    std::vector<int32_t> ids = {
        0,0,0,0,    // t0: all in pool0? use distinct below
    };
    (void)ids;
    // build ids so token t routes (t+1) experts from pool0 and the rest from pool1
    std::vector<int32_t> ids2(n_k * n_t, 0);
    for (uint32_t t = 0; t < n_t; ++t) {
        for (uint32_t k = 0; k < n_k; ++k) {
            // first (t+1) slots -> pool0 experts 0..; rest -> pool1 expert 8+
            const int32_t e = (k <= t) ? (int32_t)k : 8;
            ids2[t * n_k + k] = e;
        }
    }
    std::vector<int32_t> expert_pool(16, 1);
    for (int e = 0; e < 8; ++e) expert_pool[e] = 0;
    mix_plan_t plan = build_mix_plan(ids2.data(), n_k, n_t, expert_pool.data(), 16, 2);
    uint32_t p0_rounds = 0, p1_rounds = 0;
    for (const auto& r : plan.rounds) (r.pool == 0 ? p0_rounds : p1_rounds)++;
    TEST_ASSERT(p0_rounds == 4, "pool0 4 rounds (counts 1..4)");
    // pool1 count = 4-(t+1) = 3,2,1,0 -> buckets 3:1 2:1 1:1 -> 3 rounds
    TEST_ASSERT(p1_rounds == 3, "pool1 3 rounds");
    TEST_ASSERT(check_coverage(plan, ids2, expert_pool, n_k, n_t), "coverage");
    return true;
}

// ---- unknown experts (expert_pool = -1) are skipped, no crash ---------------
static bool test_unknown_experts_skipped() {
    const uint32_t n_k = 4, n_t = 2, n_expert = 4;
    std::vector<int32_t> ids = {
        0,1,2,3,  0,1,2,3
    };
    std::vector<int32_t> expert_pool = { 0, 1, -1, -1 };
    mix_plan_t plan = build_mix_plan(ids.data(), n_k, n_t, expert_pool.data(), n_expert, 2);
    // experts 2,3 unknown -> each token has 1 hit in pool0 and 1 in pool1
    uint32_t p0_rounds = 0, p1_rounds = 0;
    for (const auto& r : plan.rounds) {
        if (r.pool == 0) { ++p0_rounds; TEST_ASSERT(r.width == 1, "p0 width 1"); }
        else             { ++p1_rounds; TEST_ASSERT(r.width == 1, "p1 width 1"); }
    }
    TEST_ASSERT(p0_rounds == 1, "pool0 1 round");
    TEST_ASSERT(p1_rounds == 1, "pool1 1 round");
    TEST_ASSERT(check_coverage(plan, ids, expert_pool, n_k, n_t), "coverage");
    return true;
}

// ---- edge: no experts at all in a pool -> no round --------------------------
static bool test_empty_pool_no_round() {
    const uint32_t n_k = 4, n_t = 2, n_expert = 4;
    std::vector<int32_t> ids = { 0,1,2,3, 0,1,2,3 };
    std::vector<int32_t> expert_pool = { 0, 0, 0, 0 };
    mix_plan_t plan = build_mix_plan(ids.data(), n_k, n_t, expert_pool.data(), n_expert, 2);
    for (const auto& r : plan.rounds) TEST_ASSERT(r.pool == 0, "only pool0 rounds");
    TEST_ASSERT(check_coverage(plan, ids, expert_pool, n_k, n_t), "coverage");
    return true;
}

int main() {
    bool ok = true;
    struct { bool (*fn)(); const char* name; } tests[] = {
        { test_pure_single_pool, "pure_single_pool" },
        { test_mixed_two_pools_bucket_example, "mixed_two_pools_bucket_example" },
        { test_per_token_histogram, "per_token_histogram" },
        { test_unknown_experts_skipped, "unknown_experts_skipped" },
        { test_empty_pool_no_round, "empty_pool_no_round" },
    };
    for (const auto& t : tests) {
        const bool pass = t.fn();
        std::printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        ok = ok && pass;
    }
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
