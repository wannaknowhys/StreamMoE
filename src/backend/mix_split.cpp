#include "backend/mix_split.h"

#include <algorithm>
#include <cstring>

namespace stream_moe {

namespace {

// Histogram-peeling: given bucket[] (bucket[c] = number of tokens that hit the
// pool exactly c times, c in [0, n_k]), produce the layer bounds v[1..m] of the
// nonzero counts and, per round, the set of tokens still active. Because tokens
// are only identified by their hit count here, the per-round token set is
// expressed as (count, n_tokens) pairs in the plan builder which keeps real
// token indices.
struct peel_bounds_t {
    std::vector<uint32_t> v;   // v[0]=0, v[j] = j-th distinct nonzero count
    std::vector<uint32_t> w;   // w[j] = v[j] - v[j-1]
};

peel_bounds_t peel_bounds(const std::vector<uint32_t>& bucket) {
    peel_bounds_t b;
    std::vector<uint32_t> counts;
    for (uint32_t c = 1; c < bucket.size(); ++c) {
        if (bucket[c] != 0) counts.push_back(c);
    }
    // counts is ascending already (we scanned c=1..), so no sort needed.
    b.v.assign(counts.size() + 1, 0);
    b.w.assign(counts.size(), 0);
    for (size_t j = 0; j < counts.size(); ++j) {
        b.v[j + 1] = counts[j];
        b.w[j]     = counts[j] - (j == 0 ? 0 : counts[j - 1]);
    }
    return b;
}

} // namespace

mix_plan_t build_mix_plan(const int32_t* ids, uint32_t n_expert_used, uint32_t n_tokens,
                          const int32_t* expert_pool, uint32_t n_expert, uint32_t n_pools) {
    mix_plan_t plan;
    plan.n_expert_used = n_expert_used;
    plan.n_tokens      = n_tokens;
    plan.n_pools       = n_pools;

    const size_t n_bucket = static_cast<size_t>(n_expert_used) + 1;
    plan.buckets.assign(static_cast<size_t>(n_pools) * n_bucket, 0);

    if (!ids || n_tokens == 0 || n_expert_used == 0 || !expert_pool) return plan;

    // Per (pool, token) list of routed experts: slots[k] of token t whose
    // expert lives in pool p. Prefer compact storage: one vector per pool of
    // (token, k) pairs ordered as encountered (t ascending, k ascending), plus
    // per-(pool,token) counts for peeling.
    struct pool_token_hits_t {
        std::vector<uint32_t> ks;   // original k per hit
        std::vector<uint32_t> ts;   // original t per hit (aligned with ks)
    };
    std::vector<pool_token_hits_t> pool_hits(static_cast<size_t>(n_pools));

    std::vector<std::vector<uint32_t>> hit_count(
        static_cast<size_t>(n_pools), std::vector<uint32_t>(n_tokens, 0));

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            const int32_t e = ids[static_cast<size_t>(t) * n_expert_used + k];
            if (e < 0 || static_cast<uint32_t>(e) >= n_expert) continue;
            const int32_t p = expert_pool[e];
            if (p < 0 || static_cast<uint32_t>(p) >= n_pools) continue;
            const size_t pi = static_cast<size_t>(p);
            pool_hits[pi].ks.push_back(k);
            pool_hits[pi].ts.push_back(t);
            hit_count[pi][t]++;
        }
    }

    for (uint32_t p = 0; p < n_pools; ++p) {
        const size_t pi = static_cast<size_t>(p);
        // bucket histogram for this pool
        std::vector<uint32_t> bucket(n_bucket, 0);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const uint32_t c = hit_count[pi][t];
            if (c > n_expert_used) continue;   // defensive
            bucket[c]++;
            plan.buckets[pi * n_bucket + c]++;
        }
        // The pool's per-token hits, in (t, k) order, grouped by count value:
        // reorder hits so rounds can slice contiguous runs. We rebuild as a
        // vector of (token, k) sorted by (hit_count desc, t asc, k asc) so the
        // peel rounds consume from the front.
        std::vector<uint32_t> order(n_tokens);
        for (uint32_t t = 0; t < n_tokens; ++t) order[t] = t;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return hit_count[pi][a] > hit_count[pi][b];
        });

        // Build a per-token sorted k list for fast slicing.
        std::vector<std::vector<uint32_t>> ks_of(n_tokens);
        const auto & ph = pool_hits[pi];
        for (size_t i = 0; i < ph.ts.size(); ++i) ks_of[ph.ts[i]].push_back(ph.ks[i]);

        const peel_bounds_t bounds = peel_bounds(bucket);
        if (bounds.v.size() <= 1) continue;   // all-zero bucket: no work for this pool

        // Round j consumes, for each token with count >= v[j], the slice of its
        // ks from v[j-1] to v[j]. Tokens with a higher count keep the tail for
        // later rounds.
        for (size_t j = 1; j < bounds.v.size(); ++j) {
            const uint32_t w = bounds.w[j - 1];
            const uint32_t vj = bounds.v[j];
            // tokens with count >= vj
            std::vector<uint32_t> active;
            for (uint32_t t = 0; t < n_tokens; ++t) {
                if (hit_count[pi][t] >= vj) active.push_back(t);
            }
            if (active.empty() || w == 0) continue;
            mix_round_t r;
            r.pool     = p;
            r.width    = w;
            r.n_active = static_cast<uint32_t>(active.size());
            r.ids.reserve(static_cast<size_t>(r.n_active) * w);
            r.scatter.reserve(static_cast<size_t>(r.n_active) * w);
            // llama layout: row = token (a), k fastest. For each active token,
            // its ks slice [v[j-1], v[j]) are this round's slots.
            const uint32_t vprev = bounds.v[j - 1];
            for (const uint32_t t : active) {
                const auto & ks = ks_of[t];
                for (uint32_t s = vprev; s < vj; ++s) {
                    const uint32_t k = ks[s];
                    r.ids.push_back(ids[static_cast<size_t>(t) * n_expert_used + k]);
                    r.scatter.push_back({ t, k });
                }
            }
            plan.rounds.push_back(std::move(r));
        }
    }
    return plan;
}

} // namespace stream_moe
