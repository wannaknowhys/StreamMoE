# Bucket execution with token-subset rounds (bucket_exec token-subset)

[English](BUCKET_EXEC_TOKEN_SUBSET.md) | [简体中文](BUCKET_EXEC_TOKEN_SUBSET.zh-CN.md)

> Status: **design, 2026-09**. Serves the M2-5 compact-chain engine convergence:
> after delete-A (2026-09-07) the ONLY executor is `exec_layer_burst_chain_buckets`
> (minigraph_exec.cpp). Its current buckets are **full-token k-slot slices**
> (env `STREAM_MOE_TMP_CHAIN_BUCKETS`, default one full-width bucket). This doc
> upgrades the bucket source to **`build_mix_plan` rounds = true token-subset
> buckets** (`[w_b, n_active]`, active tokens are a subset of n_t, each token
> owns `w_b` of its `n_k` slots), and wires the `scatter_plan` module
> (docs/SCATTER_PLAN.md) as the accumulator writeback. Companion:
> docs/M2_DEVICE_EXECUTOR.md SS7.5/SS7.6/SS7.7 (per-device accumulator, compact
> per-bucket chains, ggml_acc as the fold writeback).

## 1. Current engine shape (what we change)

`exec_layer_burst_chain_buckets` today:

- bucket list = k-slot half-open ranges over **ALL tokens** (`bk = {[k0,k1)}`,
  `n_active == n_t`), from env cut or default single full-width `[0, n_k)`.
- per bucket: build `ids_exp`/`ids_slot` = expert ids / pool-local slot ids over
  `(t, k_lo+s)` for every token `t` -> `[w_b, n_t]` compact chain.
- `append_mm_bucket`: mm shell over `[w_b, n_t]`; cur is the main-graph shared
  activation referenced in place (`bucket_ext_leaf`), compact dst pinned into the
  arena at `out_off[seq]` (M2 SS7.2.1 serial reuse).
- weightless twins narrow the slot axis to `w_b`, token axis stays `n_t`.
- `append_expert_fold` folds `w_b` -> `[d_out, n_t]`; acc = `ggml_acc_inplace`
  offset 0 into the persistent `acc_d` (n_active == n_t, so same-position add).
- token-subset buckets are mock-gated (n_active != n_t aborts).

## 2. Target shape (this doc)

The bucket source becomes **`mix_plan_t.rounds`** from `build_mix_plan`
(backend/mix_split.h): each round is a TRUE token-subset bucket

- `r.ids`        = expert ids, llama layout `[w_b, n_active]` (`r.ids[a*w_b+s]`),
  `w_b = r.width`, `n_active = r.n_active`.
- `r.scatter`    = per `(a,s)`, the original `(t, k)` the column belongs to.
- pool (device) = `r.pool`.

`round.scatter[a*w_b].t` = original token of active column `a` -> this is the
`t` input to `build_scatter_plan`.

The engine loops rounds like it loops buckets today, but each round builds a
compact chain on **`[w_b, n_active]`** and folds into the accumulator at the
token positions `t_a` (scattered, not offset-0).

### 2.1 Cur copy per device / per bucket (GPU shape)

M2 SS7.5: each device owns a cur copy; the per-bucket mini-graph PREPENDS a
**cur copy node** that gathers the main full cur `[d, 1, n_t]` into the bucket's
compact cur `[d, 1, n_active]` (tight order, columns = `t[order[i]]`). In the
one-cgraph CPU engine the same node is appended at the head of each bucket
chain (before the first routed mm of that bucket). Every downstream chain tensor
then inherits the tight order for free (SCATTER_PLAN SS2).

ggml-native gather: `ggml_get_rows(cur, col_ids)` where `col_ids` is an i32 leaf
of the tight-order token ids (`[1, n_active]`). This is the chosen CPU-phase
implementation (user decision 2026-09-07) - the gather is an in-graph node, not
a host memcpy. Requires the CPU kernel to gather over cur's token axis (ne2 of
`[d,1,n_t]`), not only the leading dim - verified before use. The cur copy node
writes `[d,1,n_active]` into the arena (its own out_off region or fold_buf).

### 2.2 Chain geometry on the round

- mm ids = pool-local slot translation of `r.ids` (`slot = pin(e); slot-slot_begin`).
- mm cur = the bucket's compact cur copy (tight order) for the first routed mm;
  down mm reads its own compact GLU twin as today.
- weightless twins narrowed to `w_b` as today, but token axis = `n_active`.
- `append_expert_fold` -> `per_token[d_out, n_active]` in TIGHT order.

### 2.3 Scatter-add into the accumulator (scatter_plan)

For each round:
1. `t[a] = r.scatter[a*r.width].t` (a order, original token per active column).
2. `plan = build_scatter_plan(t, n_active, n_t)` -> `order[]`, `segs[]`.
3. The cur copy gathered cur in `order`, so `per_token` is already tight:
   `per_token[:, i]` corresponds to original token `t[order[i]]`.
4. acc: for each seg, one `ggml_acc_inplace`:
   `acc_d[:, dst + i*delta] += per_token[:, src + i]`,
   `nb1 = delta*d_out*4`, `offset = dst*d_out*4` (src contiguous because tight).

The plan is computed host-side (pure function, cheap); `order` feeds the cur
copy, `segs` feeds the acc loop.

### 2.4 Acc_d is the per-device accumulator

`c.acc_d` (chain_ctx, `[d_out, n_t]`) is the device accumulator (M2 SS7.6.1):
incremental per-bucket in-place adds, zeroed per layer before the round loop
(existing). Exit = `chain_exit` (acc_d -> add_in -> moe_out) unchanged.

## 3. Bucket split function (one bucket -> two, for validation)

A single-pool `build_mix_plan` produces exactly ONE round (all tokens, full k),
which would not exercise the token-subset path. A deterministic **split
function** turns that one round into two SCATTERED token-subset buckets so the
subset path (cur copy + scatter_plan reorder) is validated on the CPU engine
before real multi-pool exists.

`src/backend/bucket_split.h/.cpp` (pure, no llama dep, offline unit-testable):

```cpp
// Split one full round into two SCATTERED token-subset rounds.
// Round 0 = tokens {0, 2, 4, ...} (even), round 1 = {1, 3, 5, ...} (odd);
// each keeps full k (width = n_k). The interleaved token ids give scatter_plan
// arithmetic runs with delta > 1 and a real reorder - the subset path's order
// machinery is what we validate.
std::vector<mix_round_t> split_round_tokens_parity(const int32_t* ids, uint32_t n_k,
                                                   uint32_t n_t,
                                                   const int32_t* expert_pool,
                                                   uint32_t n_expert, uint32_t pool);
```

The executor accepts either the real `build_mix_plan` rounds or the split
function's output as its round list; both are `mix_round_t`. Validation: the two
parity buckets accumulated into acc_d must equal the single full round (relaxed
ULP gate; both sides already differ from llama's k-order fold).

## 4. Executor change sketch (exec_layer_burst_chain_buckets)

- build the round list from `build_mix_plan` by default (single pool = one
  full round, degenerate; multi-pool = true per-pool rounds). Validation split
  (env `STREAM_MOE_TMP_BUCKET_ROUNDS=parity`) replaces that list with the split
  function's two scattered buckets.
- loop rounds; per round set `b.w_b = r.width`, `b.n_active = r.n_active`,
  `b.ids_exp/ids_slot` from `r.ids` + pin, `b.t` from `r.scatter`.
- prepend the cur copy node (`ggml_get_rows`, tight-gather) to the chain.
- `append_expert_fold` -> tight `[d_out, n_active]`.
- acc via `build_scatter_plan` order/segs (multiple `ggml_acc_inplace`, one per
  seg, chained in place on `acc_d`).
- keep arena out_off pinning; compact `[d, w_b, n_active]` fits the full-width
  out_off region because `w_b <= n_k && n_active <= n_t`.

## 5. Open questions / risks

1. ggml_get_rows over cur's token axis (ne2 of `[d,1,n_t]`) must be verified
   before use; if the CPU kernel only gathers the leading dim, the cur copy
   falls back to a reshape to `[d*n_t]`-style 2D + leading-dim get_rows + reshape
   (still an in-graph node, not host staging).
2. The existing single-pool full-round numeric baselines (gemma_129_l0,
   deepseek_hi) must stay byte-identical with the degenerate single-round path
   (default = build_mix_plan single pool = one round, no split env).
3. `scatter_plan` currently rejects duplicate token ids in a round - rectangle
   peel never produces them; the split function must keep each token in exactly
   one bucket.

## 6. TODO

1. `bucket_split.h/.cpp` split function (parity -> 2 scattered rounds) + test.
2. Verify ggml_get_rows gathers cur's token axis on CPU (or the 2D-reshape
   fallback works).
3. Executor round loop: accept mix_round_t list; cur copy node prepend;
   per-round tight ids; fold -> tight per_token; acc via scatter_plan.
4. Validate: default single round unchanged vs baselines; `parity` split ->
   relaxed gate vs single round.
