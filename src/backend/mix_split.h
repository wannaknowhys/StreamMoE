#pragma once

// Route B mixed-region plan: split one mul_mat_id's routing ids so each device
// pool computes exactly its own active experts, as the MINIMAL set of full
// rectangles (docs/M2_DEVICE_EXECUTOR.md, mixed J6). Pure, deterministic,
// model-agnostic - no llama.cpp / scheduler dependency, unit-testable offline.
//
// Problem: ids is a rectangle [n_expert_used, n_tokens]; every token must
// contribute the same number of slots, but the number of experts that land in
// a given pool differs per token. Split per (pool): bucket tokens by how many
// of their routed experts live in that pool, then peel the histogram with the
// classic minimum-rectangle-cover so each round is one full rectangle with
// ZERO waste:
//
//   buckets e.g. c={2:19, 5:48}  (19 tokens hit 2 experts in pool p, 48 hit 5)
//   layer bounds v = [2, 5];  widths w = [2, 3];
//   round 1: width 2, all 67 tokens        -> mm [2, 67]
//   round 2: width 3, the 48 tokens with c>=5 -> mm [3, 48]
//   total FLOPS = 67*2 + 48*3 = 278 == 19*2 + 48*5 (zero waste)
//
// mm calls per pool = number of distinct nonzero bucket counts (<= n_k).
//
// Storage is flat + caller-owned (grow-only, reused across layers): each round
// is a span {off, width, n_active} into out_ids / out_scatter, so the planner
// does no per-round allocation. ids are in llama layout (row = token, k
// fastest): ids[off + a*width + s] = expert id of the s-th slot of round token
// a; scatter[off + a*width + s] records the original (t,k) that slot came from.

#include <cstdint>
#include <vector>

namespace stream_moe {

// One (original token, original k-slot) an expert column must be written back to.
struct mix_scatter_t {
    uint32_t t = 0;   // original token index (into the layer's n_t)
    uint32_t k = 0;   // original slot index within that token (into n_k)
};

// One full-rectangle sub-mm for a pool: a span into the flat ids/scatter arrays.
struct mix_round_t {
    uint32_t pool     = 0;   // device pool this round computes
    uint32_t width    = 0;   // rectangle height = slots per token
    uint32_t n_active = 0;   // tokens in this round
    uint32_t off      = 0;   // cell offset into ids/scatter (off + a*width + s)
};

// Per-pool peeling result: pointers into the caller-owned flat storage.
struct mix_plan_t {
    const int32_t *       ids     = nullptr;
    const mix_scatter_t * scatter = nullptr;
    const mix_round_t *   rounds  = nullptr;
    uint32_t              n_rounds = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_tokens      = 0;
    uint32_t n_pools       = 0;
};

// Split `ids` ([n_expert_used, n_tokens], compact row-major data[t*n_k + k],
// k fastest) by the pool each routed expert lives in (`expert_pool[e]`).
// A token counts a pool hit when expert_pool[ids[t][k]] == pool. Experts with
// expert_pool[] == -1 (unknown) are never scheduled. Pools with an all-zero
// bucket produce no rounds.
//
// out_ids / out_scatter / out_rounds are cleared and filled; they are the
// caller's grow-only buffers (capacity retained across calls). The returned
// plan points into them and is valid until the next call.
mix_plan_t build_mix_plan(const int32_t* ids, uint32_t n_expert_used, uint32_t n_tokens,
                          const int32_t* expert_pool, uint32_t n_expert, uint32_t n_pools,
                          std::vector<int32_t>& out_ids,
                          std::vector<mix_scatter_t>& out_scatter,
                          std::vector<mix_round_t>& out_rounds);

} // namespace stream_moe
