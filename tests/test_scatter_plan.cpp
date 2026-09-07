// Unit tests for the bucket scatter planner (src/backend/scatter_plan.cpp),
// following docs/SCATTER_PLAN.md §6: virtual full-width "truth" array -> feed
// the hit token ids to build_scatter_plan -> rebuild a compact array via the
// returned order -> scatter it back through segs into a fresh sentinel target
// -> elementwise equivalence with the truth. Sentinels stand for "acc column
// kept its original value, this bucket never touched it".
#include "backend/scatter_plan.h"
#include "common/tsc.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace stream_moe;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { std::printf("[-] ASSERTION FAILED: %s (line %d)\n", msg, __LINE__); std::fflush(stdout); return false; } \
} while(0)

namespace {

const int32_t SENT = -1;   // sentinel mark: "no expert / not touched"

// Deterministic LCG so the grid is reproducible.
struct lcg_t {
    uint64_t s;
    explicit lcg_t(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s >> 32;
    }
};

// Build the virtual truth array: miss -> sentinel, hit -> mark 1..k in scan
// order. Returns the ascending hit token list in `out_t`.
void make_truth(uint32_t n_t, uint32_t hit_permille, uint64_t seed,
                std::vector<int32_t>& truth, std::vector<uint32_t>& out_t) {
    lcg_t rng(seed);
    truth.assign(n_t, SENT);
    out_t.clear();
    int32_t mark = 0;
    for (uint32_t v = 0; v < n_t; ++v) {
        if ((rng.next() % 1000u) < hit_permille) {
            ++mark;
            truth[v] = mark;
            out_t.push_back(v);
        }
    }
}

// Full procedure of §6 steps 4-8 for one (n_t, hit list) task, given the
// pre-built truth. `t` may be any permutation of the hit token ids.
bool run_scatter_check(const std::vector<uint32_t>& t, const std::vector<int32_t>& truth,
                       uint32_t n_t) {
    const uint32_t n_active = static_cast<uint32_t>(t.size());
    uint32_t k = 0;
    for (int32_t x : truth) if (x != SENT) { TEST_ASSERT(x == (int32_t)(k + 1), "marks are 1..k in scan order"); ++k; }
    TEST_ASSERT(k == n_active, "hit count == active count");

    const uint64_t r0 = tsc_now();
    scatter_plan_t plan = build_scatter_plan(t.data(), n_active, n_t);
    const uint64_t r1 = tsc_now();

    // step 4: order is a permutation of [0, n_active).
    TEST_ASSERT(plan.order.size() == n_active, "order size == n_active");
    std::vector<uint8_t> seen(n_active, 0);
    for (uint32_t i = 0; i < n_active; ++i) {
        TEST_ASSERT(plan.order[i] < n_active, "order[i] in range");
        TEST_ASSERT(!seen[plan.order[i]], "order is a permutation (no dup)");
        seen[plan.order[i]] = 1;
    }

    // steps 5-6: compact via order; must hold each mark 1..k exactly once.
    std::vector<int32_t> compact(n_active, SENT);
    std::vector<uint8_t> have(n_active, 0);   // have[m-1] for mark m
    for (uint32_t i = 0; i < n_active; ++i) {
        const uint32_t tv = t[plan.order[i]];
        TEST_ASSERT(tv < n_t, "t[order[i]] < n_t");
        const int32_t val = truth[tv];
        TEST_ASSERT(val != SENT, "compact has no sentinel (only hits present)");
        TEST_ASSERT(val >= 1 && (uint32_t)val <= k, "compact mark in [1,k]");
        TEST_ASSERT(!have[val - 1], "each mark exactly once");
        have[val - 1] = 1;
        compact[i] = val;
    }

    // step 7: scatter compact back through segs into a fresh sentinel target.
    std::vector<int32_t> target(n_t, SENT);
    uint32_t src_next = 0;
    for (const auto& seg : plan.segs) {
        TEST_ASSERT(seg.src == src_next, "segs are contiguous, cover all src");
        TEST_ASSERT(seg.len >= 1, "seg len >= 1");
        TEST_ASSERT(seg.delta >= 1, "seg delta >= 1");
        TEST_ASSERT(seg.dst + (uint64_t)(seg.len - 1) * seg.delta < n_t, "seg dst arithmetic fits n_t");
        for (uint32_t i = 0; i < seg.len; ++i) {
            const uint32_t si = seg.src + i;
            const uint32_t di = seg.dst + i * seg.delta;
            TEST_ASSERT(si < n_active, "seg src in range");
            TEST_ASSERT(di < n_t, "seg dst in range");
            TEST_ASSERT(di == t[plan.order[si]], "seg dst == token id of that tight column");
            target[di] = compact[si];
        }
        src_next = seg.src + seg.len;
    }
    TEST_ASSERT(src_next == n_active, "segs cover all n_active columns");

    // step 8: full-width elementwise equivalence.
    for (uint32_t v = 0; v < n_t; ++v) TEST_ASSERT(target[v] == truth[v], "target == truth elementwise");

    std::printf("total=%u active=%u segs=%zu dt=%lldns\n",
                n_t, n_active, plan.segs.size(),
                (long long)tsc_delta_ns(r1 - r0));
    return true;
}

} // namespace

// ---- grid: n_t x hit rate ------------------------------------------------
static bool test_grid() {
    const uint32_t n_ts[]     = { 1, 2, 3, 4, 16, 1024 };
    const uint32_t hit_ppm[]  = { 0, 100, 500, 900, 1000 };   // 0 / .1 / .5 / .9 / 1
    uint64_t seed = 0x51A7E9u;
    bool ok = true;
    for (uint32_t n_t : n_ts) {
        for (uint32_t ppm : hit_ppm) {
            std::vector<int32_t> truth;
            std::vector<uint32_t> t;
            make_truth(n_t, ppm, seed++, truth, t);
            // ascending t (the real producer's order: mix_split scans t asc)
            if (!run_scatter_check(t, truth, n_t)) ok = false;
        }
    }
    return ok;
}

// ---- fixed: fully consecutive active set -> one delta-1 seg, order identity --
static bool test_full_consecutive() {
    const uint32_t n_t = 6;
    std::vector<uint32_t> t = { 0, 1, 2, 3, 4, 5 };
    std::vector<int32_t> truth = { 1, 2, 3, 4, 5, 6 };
    scatter_plan_t plan = build_scatter_plan(t.data(), (uint32_t)t.size(), n_t);
    TEST_ASSERT(plan.segs.size() == 1, "one seg");
    if (!plan.segs.empty()) {
        TEST_ASSERT(plan.segs[0].len == 6, "seg len 6");
        TEST_ASSERT(plan.segs[0].delta == 1, "seg delta 1");
        TEST_ASSERT(plan.segs[0].dst == 0, "seg dst 0");
    }
    for (uint32_t i = 0; i < t.size(); ++i) TEST_ASSERT(plan.order[i] == i, "order identity");
    return run_scatter_check(t, truth, n_t);
}

// ---- fixed: motivating sparse case {2,4,5,6,9} -> 2 segs -------------------
static bool test_motivating_sparse() {
    const uint32_t n_t = 10;
    std::vector<uint32_t> t = { 2, 4, 5, 6, 9 };
    std::vector<int32_t> truth(n_t, SENT);
    // scan-order marks at the hit positions (ascending truth, marks 1..5)
    uint32_t j = 0;
    for (uint32_t v = 0; v < n_t; ++v)
        if (v == 2 || v == 4 || v == 5 || v == 6 || v == 9) truth[v] = (int32_t)(++j);

    scatter_plan_t plan = build_scatter_plan(t.data(), (uint32_t)t.size(), n_t);
    TEST_ASSERT(plan.segs.size() == 2, "2 segs (better than natural 3)");
    if (plan.segs.size() == 2) {
        // greedy: longest run {4,5,6} d=1 (len3 beats {2,4,6} by smaller delta
        // on the len-3 tie), then pair {2,9} d=7.
        TEST_ASSERT(plan.segs[0].len == 3 && plan.segs[0].delta == 1 && plan.segs[0].dst == 4, "seg0 {4,5,6} d1");
        TEST_ASSERT(plan.segs[1].len == 2 && plan.segs[1].delta == 7 && plan.segs[1].dst == 2, "seg1 {2,9} d7");
    }
    return run_scatter_check(t, truth, n_t);
}

// ---- plan depends only on the value set, not input order -------------------
static bool test_input_order_invariant() {
    const uint32_t n_t = 10;
    std::vector<uint32_t> asc  = { 0, 2, 4, 5, 6, 9 };
    std::vector<uint32_t> rev  = { 9, 6, 5, 4, 2, 0 };
    std::vector<uint32_t> shuf = { 2, 0, 9, 5, 4, 6 };

    const auto mk = [&](const std::vector<uint32_t>& t) {
        scatter_plan_t p = build_scatter_plan(t.data(), (uint32_t)t.size(), n_t);
        // the value sequence of the tight order (order applied to t)
        std::vector<uint32_t> vals;
        for (uint32_t i = 0; i < t.size(); ++i) vals.push_back(t[p.order[i]]);
        return vals;
    };
    const auto v0 = mk(asc), v1 = mk(rev), v2 = mk(shuf);
    TEST_ASSERT(v0 == v1 && v1 == v2, "tight value sequence identical across input orders");
    const auto segs_asc  = build_scatter_plan(asc.data(), (uint32_t)asc.size(), n_t).segs;
    const auto segs_shuf = build_scatter_plan(shuf.data(), (uint32_t)shuf.size(), n_t).segs;
    bool same = segs_asc.size() == segs_shuf.size();
    for (size_t i = 0; same && i < segs_asc.size(); ++i)
        same = segs_asc[i].len == segs_shuf[i].len && segs_asc[i].delta == segs_shuf[i].delta &&
               segs_asc[i].dst == segs_shuf[i].dst;
    TEST_ASSERT(same, "segs identical across input orders");

    std::vector<int32_t> truth(n_t, SENT);
    for (uint32_t i = 0; i < asc.size(); ++i) truth[asc[i]] = (int32_t)(i + 1);
    return run_scatter_check(shuf, truth, n_t);
}

// ---- duplicate / out-of-range token ids are rejected -----------------------
static bool test_reject_invalid() {
    std::vector<uint32_t> dup = { 1, 1, 2 };
    scatter_plan_t p1 = build_scatter_plan(dup.data(), 3, 4);
    TEST_ASSERT(p1.order.empty() && p1.segs.empty() && p1.n_active == 3, "duplicate rejected (n_active kept)");

    std::vector<uint32_t> oor = { 0, 4 };
    scatter_plan_t p2 = build_scatter_plan(oor.data(), 2, 4);
    TEST_ASSERT(p2.order.empty() && p2.segs.empty(), "out-of-range rejected");

    scatter_plan_t p3 = build_scatter_plan(nullptr, 3, 4);
    TEST_ASSERT(p3.order.empty() && p3.segs.empty(), "null t rejected");

    scatter_plan_t p4 = build_scatter_plan(nullptr, 0, 4);
    TEST_ASSERT(p4.order.empty() && p4.segs.empty() && p4.n_active == 0, "empty input valid empty plan");
    return true;
}

// ---- consistency: order + segs recompute to the exact dst sequences --------
static bool test_order_seg_consistency() {
    const uint32_t n_t = 16;
    std::vector<uint32_t> t = { 1, 3, 5, 9, 10, 11, 0 };
    scatter_plan_t plan = build_scatter_plan(t.data(), (uint32_t)t.size(), n_t);
    uint32_t tight = 0;
    for (const auto& seg : plan.segs) {
        TEST_ASSERT(seg.src == tight, "segs contiguous");
        for (uint32_t i = 0; i < seg.len; ++i) {
            const uint32_t expected_dst = t[plan.order[seg.src + i]];
            TEST_ASSERT(expected_dst == seg.dst + i * seg.delta, "segs dst arithmetic == token ids");
        }
        tight += seg.len;
    }
    TEST_ASSERT(tight == (uint32_t)t.size(), "segs cover all");
    return true;
}

int main() {
    bool ok = true;
    struct { bool (*fn)(); const char* name; } tests[] = {
        { test_grid, "grid (n_t x hit rate, 30 tasks)" },
        { test_full_consecutive, "full_consecutive" },
        { test_motivating_sparse, "motivating_sparse_{2,4,5,6,9}" },
        { test_input_order_invariant, "input_order_invariant" },
        { test_reject_invalid, "reject_invalid" },
        { test_order_seg_consistency, "order_seg_consistency" },
    };
    for (const auto& t : tests) {
        const bool pass = t.fn();
        std::printf("[%s] %s\n", pass ? "PASS" : "FAIL", t.name);
        ok = ok && pass;
    }
    std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
