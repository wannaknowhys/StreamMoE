#include "backend/scatter_plan.h"

#include <algorithm>
#include <cstddef>

namespace stream_moe {

namespace {

// Find the best remaining arithmetic run over the present values, per the §4
// greedy: primary = longest run length, tie = smallest delta, then smallest
// start value. out_delta / out_start / out_len receive the winner. Requires
// m >= 2. in_set[v]==1 marks present; rem is the ascending present values.
// For one delta, every chain {v, v+delta, v+2delta, ...} has a unique start
// v with v-delta absent; walking forward from each start touches each present
// value at most once, so a delta sweep is O(m). Candidates are pruned once the
// bound range/delta + 1 cannot beat the current best length.
void best_run(const uint8_t* in_set, const std::vector<uint32_t>& rem,
              uint32_t n_t, uint32_t minv, uint32_t maxv, uint32_t m,
              uint32_t& out_delta, uint32_t& out_start, uint32_t& out_len) {
    uint32_t best_len = 1, best_delta = 1, best_start = minv;
    const uint32_t range = maxv - minv;

    for (uint32_t delta = 1; delta <= range; ++delta) {
        // No run with this delta can beat the best we already hold.
        if (range / delta + 1 <= best_len) break;

        uint32_t d_best_len = 0, d_best_start = 0;
        for (const uint32_t v : rem) {
            // Not a chain start: v-delta is present, v belongs to a longer chain.
            if (v >= delta && in_set[v - delta]) continue;
            uint32_t len = 0;
            for (uint32_t u = v; u < n_t && in_set[u]; u += delta) ++len;
            if (len > d_best_len) { d_best_len = len; d_best_start = v; }
        }
        if (d_best_len == 0) continue;   // cannot happen for m >= 2; keep safe

        if (d_best_len > best_len ||
            (d_best_len == best_len && (delta < best_delta ||
                                        (delta == best_delta && d_best_start < best_start)))) {
            best_len   = d_best_len;
            best_delta = delta;
            best_start = d_best_start;
        }
        if (best_len == m) break;        // whole remaining set is one run
    }

    out_delta = best_delta;
    out_start = best_start;
    out_len   = best_len;
}

} // namespace

scatter_plan_t build_scatter_plan(const uint32_t* t, uint32_t n_active, uint32_t n_t) {
    scatter_plan_t plan;
    plan.n_active = n_active;
    plan.n_t      = n_t;
    if (n_active == 0 || n_t == 0 || !t) return plan;

    // Validate: token ids in [0, n_t), distinct. Membership + original-index
    // maps indexed by token id (value domain is [0, n_t)).
    std::vector<uint8_t>   seen(n_t, 0);
    std::vector<uint8_t>   in_set(n_t, 0);
    std::vector<uint32_t>  idx_of(n_t, 0);
    std::vector<uint32_t>  rem;
    rem.reserve(n_active);
    for (uint32_t i = 0; i < n_active; ++i) {
        const uint32_t v = t[i];
        if (v >= n_t || seen[v]) {
            // Reject: out-of-range or duplicate token id -> empty plan.
            plan.order.clear();
            plan.segs.clear();
            return plan;
        }
        seen[v]  = 1;
        in_set[v] = 1;
        idx_of[v] = i;
        rem.push_back(v);
    }
    std::sort(rem.begin(), rem.end());

    uint32_t tight = 0;   // running tight-order column offset == order.size()
    while (!rem.empty()) {
        const uint32_t minv = rem.front();
        const uint32_t maxv = rem.back();
        const uint32_t m    = static_cast<uint32_t>(rem.size());

        uint32_t d, start, len;
        if (m == 1) {
            d = 1; start = minv; len = 1;   // singleton run; delta unused by acc
        } else {
            best_run(in_set.data(), rem, n_t, minv, maxv, m, d, start, len);
        }

        // Emit one seg over tight columns [tight, tight+len).
        scatter_seg_t seg;
        seg.src   = tight;
        seg.dst   = start;
        seg.len   = len;
        seg.delta = d;
        plan.segs.push_back(seg);

        // The run's values ascending become the next tight columns; append
        // each one's original index to `order`.
        for (uint32_t i = 0; i < len; ++i) {
            const uint32_t v = start + i * d;
            plan.order.push_back(idx_of[v]);
            in_set[v] = 0;
        }
        tight += len;

        // Rebuild the remaining sorted value list.
        rem.clear();
        for (uint32_t v = minv; v <= maxv; ++v) {
            if (in_set[v]) rem.push_back(v);
        }
    }

    return plan;
}

} // namespace stream_moe
