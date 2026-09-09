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
>
> **2026-09-08 revision (user):** cur gather = 2D reshape + get_rows (SS2.1);
> arbitrary (t,k) weights = general flat index-gather node (SS2.1a); the test
> split is a macro-gated throwaway helper inside minigraph_exec.cpp, not a module
> (SS3); `tmp_split_blocks` is deleted (SS4).

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

- `r.ids` = expert ids, llama layout `[w_b, n_active]` (`r.ids[a*w_b+s]`),
  `w_b = r.width`, `n_active = r.n_active`.
- `r.scatter` = per `(a,s)`, the original `(t, k)` the column belongs to.
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

ggml-native gather (decided 2026-09-08): `ggml_get_rows` gathers the ROW axis
(ne1), never ne2, so `[d,1,n_t]` cannot be gathered directly. Instead reshape the
(contiguous, ne1 == 1) cur to 2D `[d, n_t]` and `get_rows` with an i32 leaf of the
tight token ids `[n_active]` -> `[d, n_active]`, then reshape back to
`[d,1,n_active]`. Free reshape, standard kernel, in-graph node (not a host
memcpy). The copy node writes `[d,1,n_active]` into the arena (its own out_off
region or fold_buf).

### 2.1a General index-gather node (arbitrary (t,k))

A `mix_plan` round selects, per token, an ARBITRARY subset of its k slots
(`ks_of[t][vprev..vj)`, mix_split.cpp) - not a contiguous k range. So the per-slot
routing weight `weights_norm[0,k,t]` (`[1, n_k, n_t]`) cannot be expressed as an
affine leaf slice (`bucket_ext_leaf`'s current `data += k_lo*nb1`). One general
**flat index-gather** node covers it (user decision 2026-09-08):

- reshape the contiguous `[1, n_k, n_t]` to 2D `[1, n_k*n_t]` (free);
- build an i32 leaf `idx[width*n_active]` of FLAT element offsets in tight order:
  `idx[i*width + s] = k + t*n_k` for `(t,k) = scatter[order[i]*width + s]`;
- `ggml_get_rows(flat, idx)` -> `[1, width*n_active]`, reshape `[1, width, n_active]`.

The cur gather (SS2.1) is the same helper with row_size = d instead of 1. This is
the node that replaces the contiguous-k `bucket_ext_leaf` slice for weights_norm;
the forced test split (SS3) deliberately selects non-contiguous k so it is
exercised on CPU before real multi-pool exists.

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

## 3. Test-only forced-split function (validation harness)

A single-pool `build_mix_plan` produces exactly ONE round (all tokens, full k),
which would not exercise the token-subset / arbitrary-(t,k) path. A test-only
**forced-split function** turns that one round into several SCATTERED token-subset
rounds so the subset path (cur gather + index-gather weights + scatter_plan
reorder) is validated on the CPU engine before real multi-pool exists.

Placement and gating (decided 2026-09-08): a `static` helper in
`minigraph_exec.cpp`; BOTH its definition and its call site are wrapped in
`#ifdef STREAM_MOE_TEMP` (the only tag defining that macro is
`StreamMoE_dump_dbg`, build.bat). It is throwaway validation code - delete it once
the subset path is verified. No separate `bucket_split.h/.cpp`, no offline UT.

Split shape: split BOTH axes so arbitrary (t,k) is exercised:

- k axis by parity -> rounds with non-contiguous k (a valid rectangle when n_k is
  even: each token contributes n_k/2);
- token axis by parity -> token subset + scatter delta > 1;
- combined -> 4 rounds, each `width = n_k/2, n_active = n_t/2`, `(t,k)` arbitrary.

`r.scatter` records the original (t,k); `r.pool` = the source round's pool.

## 4. Executor change sketch (exec_layer_burst_chain_buckets)

- default round list = `build_mix_plan(ids, n_k, n_t, expert_pool, n_expert,
n_pools).rounds`, with `expert_pool[e] = handle.pool` filled from the pinned
  handles (`pin_layer` already returns per-expert pool). Single RAM pool = one
  full round (degenerate, byte-identical to today's default single bucket).
- the macro-gated forced split replaces that list with the split rounds.
- delete `tmp_split_blocks` / `tmp_blk_t` (the old test cut family, currently
  compiled unconditionally) - two bucket sources must not coexist.
- loop rounds; per round set `b.w_b = r.width`, `b.n_active = r.n_active`,
  `b.ids_exp/ids_slot` from `r.ids` + pin, `b.t` from `r.scatter`.
- prepend the cur copy node (2D reshape + get_rows, SS2.1).
- per-slot routing weights via the general index-gather node (SS2.1a).
- `append_expert_fold` -> tight `[d_out, n_active]`.
- acc via `build_scatter_plan` order/segs (one `ggml_acc_inplace` per seg on
  `acc_d`, chained in place).
- keep arena out_off pinning; compact `[d, w_b, n_active]` fits the full-width
  out_off region because `w_b <= n_k && n_active <= n_t`.

## 5. Open questions / risks

1. RESOLVED (2026-09-08): `ggml_get_rows` gathers ne1 only; cur is gathered via a
   free 2D reshape to `[d, n_t]` + get_rows + reshape back (SS2.1). No kernel
   change, no host staging.
2. Arbitrary (t,k) weights use the flat index-gather node (SS2.1a), not an affine
   slice; the forced test split exercises it.
3. The single-pool full-round numeric baselines (gemma_129_l0, deepseek_hi) must
   stay byte-identical under the degenerate single-round default (build_mix_plan
   single pool = one round, no split macro).
4. `scatter_plan` rejects duplicate token ids in a round - rectangle peel never
   produces them; the forced-split function must keep each token in exactly one
   bucket.

## 6. TODO

1. General index-gather helper (SS2.1a): flat reshape + get_rows for arbitrary
   (t,k) weights; the cur gather (SS2.1) is the row_size = d instance.
2. Executor round loop over `mix_round_t`: expert_pool from pins, default
   build_mix_plan rounds, per-round tight ids, fold -> tight per_token, acc via
   scatter_plan; delete tmp_split_blocks.
3. Macro-wrapped test-only forced split (k-parity x t-parity = 4 rounds) + call.
4. Verify: default single round unchanged vs baselines; forced split -> relaxed
   gate (maxAbs <= 1e-5 / cos ~= 1.0).
