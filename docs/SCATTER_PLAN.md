# Bucket token reorder + acc scatter plan (scatter_plan)

[English](SCATTER_PLAN.md) | [简体中文](SCATTER_PLAN.zh-CN.md)

> Status: **design, 2026-09**. Feeds the CPU two-bucket compact-chain prototype
> (docs/WORK_IN_PROGRESS.md M2-5) and the eventual per-device mini-graph
> (docs/M2_DEVICE_EXECUTOR.md SS7.8). A bucket here = one `mix_round_t` of a
> `mix_plan_t` (docs/Backend.md J6 / mix_split): a full rectangle `[w_b,
> n_active]` of routed (token, slot) columns.

## 1. Problem

A routed MoE layer produces, per token, `n_k` expert columns. A bucket holds
only the columns its pool owns - a subset of tokens, and for each token a
subset of its slots. The compact chain (docs/M2_DEVICE_EXECUTOR.md SS7.5/7.8)
computes the layer's MoE ffn for a bucket's columns only and folds them per
token, so after `append_expert_fold` we have one partial-sum column per
**active** token: `per_token[d_out, a]` belongs to the original token `t_a`
recorded in the round's `scatter[a*w_b]`. These `t_a` are scattered (a = peel
order, not token order), and different tokens may appear in several rounds.

The fold result must be added into the layer's output accumulator
`acc_d[d_out, n_t]` (CPU prototype) / the main-graph `moe_out` dst columns
(GPU): `acc_d[:, t_a] += per_token[:, a]` for every active `a`.

A single `ggml_acc` call can express: src1's contiguous columns `0..len-1`
added to dst columns `base, base+delta, ...` (verified in the CPU kernel
ggml-cpu/ops.cpp `ggml_compute_forward_acc_f32`: `dst[offset + i1*nb1 + ...]
+= src1[...]`, with `nb1 = delta * d_out * esize`, `offset = base * d_out *
esize`; intermediate dst columns are untouched). So one acc per **arithmetic
run** of dst columns is enough - provided the src columns feeding that run are
**contiguous** in `per_token`.

That contiguity does not exist in peel order. `scatter_plan` decides the
**tight order**: a reordering of the bucket's active tokens such that after
reordering, the token-to-acc-dst mapping decomposes into few contiguous src
runs, each run a pure arithmetic progression of dst columns. Each run then
needs exactly one `ggml_acc_inplace`.

## 2. Why the reorder must be consumed before the chain (cur copy)

The compact chain's intermediate tensors all carry the bucket's token axis in
the order it was given at the first routed mm (`cur` for the gate/up mm). If
that axis is the tight order from the start, every downstream tensor - gate_up
dst, GLU output, down mm dst, down_scaled, weighted, and finally the fold -
inherits the tight order for free, and `per_token` comes out already grouped
into the runs. No second gather at the tail.

The first routed mm reads `cur` (the norm output, one column per original
token). For a full-token bucket `cur` can be referenced in place. For a token
subset / tight reorder it must be **copied into a compact staging buffer in
tight token order** before the mm reads it (this is the "forced cur copy").
That copy is where the reorder is physically applied.

### Per-token inputs affected by bucketing (audit)

| data | role | shape | bucket effect |
|---|---|---|---|
| `cur` (norm out) | mm src1 | `[d,1,n_t]` (shared) / `[d,n_k,n_t]` (per-slot) | token axis must be copied in tight order -> `[d,..,n_active]` |
| routing `ids` | mm src2, GET_ROWS src1 | `[n_k,n_t]` | token subset + slot subset already staged (`ids_exp`/`ids_slot`); reorder token blocks to tight order |
| per-slot routing weights (`ffn_moe_weights_norm`) | weighted mul src1 | `[1,n_k,n_t]` | slot slice `[k_lo..k_hi)` AND token axis in tight order |
| expert weights, per-expert scale (REPEAT table) | mm src0 / GET_ROWS src0 | per-expert | token-independent - untouched |
| chain-internal tensors (GLU/down/weighted) | - | `[d,w_b,n_t]` | inherit the order fixed at the first cur copy |

## 3. Module interface (pure, no ggml/llama dependency)

New files `src/backend/scatter_plan.h/.cpp`, same style as `mix_split` (pure,
deterministic, offline-unit-testable). The module owns the run definition and
the greedy policy - callers only supply the token sequence.

```cpp
#pragma once
// StreamMoE bucket scatter planning (docs/SCATTER_PLAN.md).
// Pure / deterministic / model-agnostic - no llama.cpp or ggml dependency.
#include <cstdint>
#include <vector>

namespace stream_moe {

// One arithmetic run of acc-dst columns fed by a contiguous src run.
//   acc_d[.., dst + i*delta] += per_token[.., src + i]   for i in [0, len)
// dst/delta are dst-column indices into acc_d[d_out, n_t]; the executor turns
// them into ggml_acc params: nb1 = delta*d_out*esize, offset = dst*d_out*esize.
struct scatter_seg_t {
    uint32_t src   = 0;    // tight-order src start column (== run index offset)
    uint32_t dst   = 0;    // acc-dst column of src[0]
    uint32_t len   = 0;    // columns in this run (>= 1)
    uint32_t delta = 0;    // dst column step between consecutive src columns
};

struct scatter_plan_t {
    // Tight order: new per-token column i (0..n_active-1) was originally the
    // active-token entry `order[i]` (index into the input `t` sequence). The
    // cur-copy layer gathers cur columns in this order.
    std::vector<uint32_t> order;
    // Acc runs over the tight order (in tight order, disjoint, cover all
    // n_active columns). segs.back() ends at src == n_active.
    std::vector<scatter_seg_t> segs;
    uint32_t n_active = 0;
    uint32_t n_t      = 0;
};

// Build the plan for one bucket. `t[0..n_active)` = original token id of each
// active column (a order, i.e. t[a] = scatter[a*w_b].t). n_t = layer token
// count (bounds: dst+delta*(len-1) < n_t).
scatter_plan_t build_scatter_plan(const uint32_t* t, uint32_t n_active, uint32_t n_t);

} // namespace stream_moe
```

`order` is the primary output (the cur copy consumes it). `segs` is derivable
from `order` but precomputed so the acc-loop executor does no re-derivation;
both are unit-tested for consistency (re-apply order, slice segs, verify they
cover [0,n_active) disjointly).

## 4. Greedy maximum-run extraction

Goal: cover the token set with the fewest arithmetic runs (one acc per run),
each run = a strictly increasing arithmetic progression of dst columns whose
src columns can be made contiguous by one reorder.

Pure set-cover over arithmetic progressions is NP-hard in general, so use a
greedy: **repeatedly take the longest remaining run**, then drop its tokens.
Because the dst columns of a run must be distinct and increasing and every
token is used once, the same token can never appear in two runs.

Algorithm (n_active tokens, values in [0, n_t)):

1. Group tokens by value (handle duplicate token ids defensively: a token
   appearing twice in one bucket folds to two distinct partial columns; each
   is its own run element with the same dst - only possible when the pool owns
   two disjoint slot slices of the same token in one round, which the
   rectangle peel never produces; reject as an input error otherwise).
2. Repeat until the value set is empty:
   a. For every candidate delta from 1 upward (delta such that the token value
      can appear again, i.e. any two values in the set), count the longest
      chain of equal-delta in-set values: `len(delta) = max over v of
      #{k >= 0 : v + k*delta in set}`. delta is unbounded above but only
      deltas <= (max-min) matter; scan deltas that actually divide a pair.
   b. Pick the delta with the largest `len(delta)` (ties: smallest delta, then
      smallest start value - deterministic).
   c. Emit one seg for that run; remove its values from the set.
3. The emission order (segs in the order the runs are extracted) defines the
   tight order: concatenate each run's values ascending.

Complexity is acceptable for CPU/decode sizes (n_k <= 16, n_active <= n_t);
an O(delta_range * n_active) delta sweep per extraction with n_active <= a few
hundred is fine. The module stays heuristic by design - "fewest runs" is a
quality goal, not a guarantee.

## 5. Executor consumption (design sketch, not implemented yet)

- **cur copy** (`build_cur_sub`-equivalent in the compact chain): allocate
  staging `cur_tight[d, 1|w_b?, n_active]`, copy column `t[order[i]]` of the
  main `cur` to staging column `i`. The first routed mm then reads staging.
  (Full-token bucket, no reorder: order is identity, reference in place.)
- **ids**: reorder the already-staged `ids_exp`/`ids_slot` token blocks by
  `order` (block = the token's slot slice).
- **per-slot weights** (`weights_norm`): slice slots `[k_lo..k_hi)` and reorder
  the token axis by `order`.
- **acc loop** (replaces the offset-0 single acc in
  `exec_layer_burst_chain_buckets`): for each seg, one
  `ggml_acc_inplace(acc_d, per_token_col_slice, nb1=delta*d_out*4,
  offset=dst*d_out*4)`. The src slice is contiguous in `per_token` because the
  chain ran in tight order.

## 6. Unit tests (planned)

`tests/test_scatter_plan.cpp`, added to the build.bat test list:

1. full-token bucket, token ids already consecutive -> one seg delta 1, order
   identity.
2. `t = {0,2,3,4}` (the motivating example) -> one run {0,2,4} delta 2 (len 3)
   + one singleton {3}; order reorders 3 after 4.
3. scattered far-apart tokens -> each a singleton or a small arithmetic
   grouping; verify every seg's dst arithmetic fits n_t and src ranges are
   disjoint and cover [0,n_active).
4. reverse order input -> identical plan to ascending (order only depends on
   values).
5. duplicate / out-of-range token ids -> rejected (returned plan empty / flag).
6. consistency: applying `order` to `t` then walking `segs` yields the exact
   dst sequences the segs claim.
