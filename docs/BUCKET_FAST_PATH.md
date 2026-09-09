# Full-width / Single-device Bucket Fast Path (BUCKET_FAST_PATH)

[English](BUCKET_FAST_PATH.md) | [简体中文](BUCKET_FAST_PATH.zh-CN.md)

> Status: design discussion (2026-09-09), **not implemented**. Corresponds to
> `docs/WORK_IN_PROGRESS.md` P1. Goal: in the full-width / single-device case,
> remove the fixed overhead of identity ops and single-target reduction in the
> only executor `exec_layer_burst_chain_buckets` (`src/backend/minigraph_exec.cpp`).
> This doc records the conclusions, exact predicates, the transpose (CONT)
> analysis, risks and the validation gate.

## 1. Background

- The only executor is the compact-bucket engine; rounds come from
  `build_mix_plan` (real per-pool token-subset rounds).
- The general shape rebuilds a compact chain `[d, w_b, n_active]` per round,
  folds the expert axis, and scatter-adds into `acc_d`.
- When a round degenerates to / approaches full width, the chain contains many
  **identity permutations** and a **single-target reduction**, which are pure
  overhead:
  - `GET_ROWS(cur)` / `GET_ROWS(weights)` are identities at full width;
  - with a single device, `ACC` + host `layer_fold` are just a copy / a zero add.
- Note: **buckets only exist with multiple pools.** A single pool (all experts in
  one pool) is always **one full-width bucket** (every token hits that pool n_k
  times, so hit_count is constant); with multiple pools, different hit counts on
  one pool (device) produce several rounds (see §3).

## 2. Omission predicates (down to code)

> The predicate is evaluated on the **current bucket (round)**, independent of
> which pool it belongs to; the per-pool peel in §3 only explains how the round
> list is produced.

| Omitted | Predicate (per round / per plan) | Hook | Reason |
| :--- | :--- | :--- | :--- |
| `GET_ROWS(cur)` | per round `r.n_active == n_t` | `bucket_gather_cur` (minigraph_exec.cpp:665) | `active` is built in ascending t (mix_split.cpp:113); `build_scatter_plan` yields an identity `order` for all tokens (scatter_plan.cpp:84); independent of width |
| `GET_ROWS(weights)` | per round `r.n_active == n_t && r.width == n_k` | `bucket_gather_per_slot` (minigraph_exec.cpp:631) | the whole `(t,k)` cell must be present in this round for the target layout to match the source `[1,n_k,n_t]` |
| `ACC` as copy | `single_target && exactly 1 round` | acc loop (minigraph_exec.cpp:1300) | with several rounds ACC is a real accumulation; `width==n_k && n_active==n_t` implies a single round |
| host `layer_fold` | `single_target` (any round count) | `layer_fold` (minigraph_exec.cpp:465) | with one target the host side is only "drop that target's acc into moe_out" |

- `single_target` = number of **distinct pools in `rounds` == 1** (not
  `dev_targets.size()<=1`: a CPU round and a device round coexist -> 2 targets).
- The two classes are **independent**: full width controls the widening gathers;
  single target controls the reduction merge. A single pool is always one
  full-width bucket, satisfying both; with multiple pools test each bucket
  separately.
- `cell_full` (`width==n_k && n_active==n_t`) means every token hits that pool and
  hits it n_k times (single pool / layer-equivalent single pool) -> one bucket;
  `tok_full` (`n_active==n_t`) does **not** imply one bucket (with multiple pools
  a pool can be token-full while `width<n_k`).

### 2.1 What `weights` is (why it needs a gather)

In `build_moe_ffn` (vendored `llama-graph.cpp`), `weights` is the **routing
coefficient (gating probability)**, not the expert weight matrix:

- gate logits -> softmax/sigmoid -> `probs` `[n_expert, n_tokens]` (:1991-2008);
- `selected_experts = argsort_top_k(...)` `[n_k, n_t]` (:2057-2060) = the `ids`
  the executor uses;
- `weights = get_rows(probs, selected_experts)` `[1, n_k, n_t]` (:2071);
- with `norm_w`: reshape `[n_k,n_t]` -> `sum_rows` -> clamp -> `weights /= weights_sum`
  = `ffn_moe_weights_norm` (:2082-2095), optional `w_scale` = `_scaled` (:2097-2100);
- consumed as `experts = experts * weights` (after the down mm, :2267-2270; or on
  the input cur when `weight_before_ffn`, :2107-2111), then the anonymous fold
  sums the n_k expert slots -> `moe_out`.

`bucket_gather_per_slot` handles exactly this `[1,n_k,n_t]`
`ffn_moe_weights_norm`/`_scaled` external leaf: the compact bucket's per-slot
multiply needs the values at the bucket's `(t,k)` subset in the bucket's tight
order (an affine k-slice cannot express an arbitrary subset, hence the flat
index-gather). At full width the whole `(t,k)` cell is present -> identity ->
reference the source directly. The model expert matrices (gate_up/down) are the
pool w3d shells, unrelated to `weights`.

## 3. Why cur and weights have different predicates

In one sentence: **a bucket is a definite rectangle `[width, n_active]`; the
question is whether that rectangle is the whole source grid.** The source
`ffn_moe_weights_norm` is the full `[1, n_k, n_t]` (contiguous, flat index
`t*n_k+k`); the bucket needs `[1, width, n_active]` (flat `a*width+s`). Identity
requires covering the whole grid.

| Quantity | Direct reference (skip gather) | Gather shape |
| :--- | :--- | :--- |
| `cur` | `n_active == n_t` | one whole row per token (d contiguous floats) |
| `weights` | `n_active == n_t && width == n_k` | one element per `(t,k)` (flat index) |

Three combinations:

- `n_active==n_t && width==n_k`: both skipped (single pool, single round);
- `n_active==n_t && width<n_k`: cur skipped, weights still gathers (round 1 of
  one pool in a multi-pool setup);
- `n_active<n_t` (any width): both gather.

`cur` depends on the token axis only (one row per token, slot-independent);
`weights` depends on both token and slot.

Peel mechanics (mix_split.cpp:21-35, 108-135): per pool, tokens are bucketed by
hit_count; `counts` = distinct nonzero hit counts ascending; round j has
`width = v[j]-v[j-1]` (`v[0]=0`) and `active = {t : hit_count[t] >= v[j]}`.
Both directions decouple:

- **Case A: `n_active==n_t` but `width<n_k`**
  The pool covers every token, but tokens hit it a different number of times.
  Example: two pools, n_k=3, one device pool holds a subset with counts=2/3/3
  -> round1 `width=2`, `active=all`.
  -> cur is identity (token axis full); weights is **not** (each token brings only
  2 of its 3 k slots).
- **Case B: `width==n_k` but `n_active<n_t`**
  Every token that hits this pool hits it n_k times (all its experts live here),
  but zero-hit tokens are excluded.
  Example: two pools, n_k=2; token A has both experts on Vulkan0, token B has
  both on RAM -> Vulkan0 round1 `width=2=n_k`, `active={A}`.
  -> neither cur nor weights is identity.
- **Case C: both hold (`cell_full`)**
  All tokens hit the pool and hit it n_k times -> single pool, single round ->
  both cur and weights are identities.

Consequences: `width==n_k` does not imply `n_active==n_t` (Case B);
`n_active==n_t` does not imply `width==n_k` (Case A). cur elision needs only
`n_active==n_t`; weights elision needs both.

## 4. The transpose (CONT) problem

### 4.1 Location and root cause

`append_expert_fold` (minigraph_exec.cpp:608-624):

```
weighted [d_out, w_b, n_active]
  -> permute(1,0,2,3)            // view [w_b, d_out, n_active], new nb0 = d_out*4
  -> ggml_cont                   // :615 real data movement (transpose)
  -> sum_rows                    // reduce new ne0 (= w_b) -> [1, d_out, n_active]
  -> ggml_cont_2d                // :619 -> [d_out, n_active]
```

Root cause: ggml `sum_rows` **only reduces a contiguous ne0**.

- CPU (`ggml-cpu/ops.cpp:1470-1486`): `GGML_ASSERT(src0->nb[0] == sizeof(float))`,
  `ggml_vec_sum_f32(ne00, ...)` reads with unit stride.
- Vulkan (`vulkan-shaders/sum_rows.comp:27-33`):
  `src_idx = i01*nb01 + i02*nb02 + i03*nb03`, sum `data_a[src_idx + i]`
  (`i < n_cols = src0->ne[0]`); the push constants have **no nb00**
  (ggml-vulkan.cpp:1972-1998 lists only nb01..03).

After `permute(1,0,2,3)` the view has `nb0 = d_out*4`, which neither backend can
consume -> `ggml_cont` is mandatory, not conservative.

### 4.2 Cost

Transposed tensor = `w_b * d_out * n_active * 4`:

| Case | Size | Note |
| :--- | ---: | :--- |
| prefill 3k (w_b=8, d=2048) | ~196 MB/layer | cont read+write ~392MB, sum_rows reads 196MB again -> ~600MB/layer |
| decode (n_active=1) | ~64 KB | negligible |

The P1b measurement of `CONT 986us` for a 2-token layer **cannot be bandwidth**
(the data is a few tens of KB); it must be fixed dispatch/graph overhead or
perf-logger attribution, which supports the P2 conclusion that per-layer
submit+sync dominates.

### 4.3 Can it be removed

1. **Replace with a per-k view + add chain** (the anonymous-fold shape):
   `acc = view_2d(weighted, d_out, n_active, weighted->nb[2], 0);`
   `for w: acc = add(acc, view_2d(..., w*weighted->nb[1]))`
   - no transpose, no 196MB temp, bucket-general (no original k mapping needed);
     cost is `w_b-1` add kernels.
   - numerically a different summation order (inside the relaxed gate).
2. **Reuse llama's original anonymous fold at full width** (closure replay):
   zero rebuild, but `bucket_src_leaf` must handle general view shapes (it only
   slices d today) -- that is the other fork, larger change.
3. **Keep as is**: the MoE mm FLOPs dominate prefill; the transpose is probably
   not the bottleneck there.

### 4.4 Two incidental points

- `ggml_cont_2d(s, ne0, nt)` (:619) is a redundant copy when the sum_rows output
  is already contiguous; `ggml_reshape_2d` suffices (saves 24MB/layer prefill).
- **To verify**: the fold scratch (`pc`/`s`/`acc`) goes through the device bump of
  `bind_fresh(..., false)`, but the arena size is fixed **before** the round loop
  as `arena_used(=base+acc) + 32MB` (minigraph_exec.cpp:1139-1143); the build-time
  bump growth is not included. `device_ensure` is called once before the loop and
  `ensure_buffer` only reallocates when `need>cap` (moe_backend.cpp:424-437). At
  prefill a single `pc` is 196MB, far beyond the 32MB slack -> the device arena
  may be overrun. **Not reproduced; confirm during P1.**

## 5. Implementation hooks

- Compute `tok_full` / `cell_full` per round into `bucket_build_t`;
  `bucket_gather_cur` checks `tok_full`, `bucket_gather_per_slot` checks
  `cell_full`.
- Full-width early return: CPU references `m->data`+`m->nb` (zero copy); device
  goes through `bucket_upload_leaf` (drops the GET_ROWS kernel, keeps one upload).
- **Device upload precondition**: `tensor_write_host` (tensor_io.h:22) does a
  contiguous memcpy of `ggml_nbytes` and ignores strides. Skipping the gather
  requires a "source is compact" condition; otherwise a strided source (argsort
  layer ids are strided; weights_norm TBD) would be copied silently wrong ->
  fall back to the original gather. CPU reference does not care about layout and
  is always safe.
- With `single_target`, skip acc/fold: CPU target pins the final
  `per_token->data` to `moe_out->data` (CPU compute only reads data; verified in
  §7.4); device target keeps `per_token` in the arena and does one
  `tensor_get` into `moe_out` after sync. Numerically equivalent (the CPU acc in
  `layer_fold` is always zero; `0+x` is bit-identical).
- `gather_cache` is already cleared per round (:1279), so full-width direct
  references and later subset gathers do not mix.

## 6. Risks and open items

- `moe_out` must be contiguous `[d_out, n_t]` (`layer_fold` already assumes so);
  otherwise fall back to fold.
- `STREAM_MOE_TMP_BUCKET_ROUNDS` (diagnostic forced split) must still take the
  old path -- the per-round predicates cover it.
- The B38 0-token no-op returns before rounds are built; unaffected.
- Device upload source compactness (see §5).
- Device arena slack (see §4.4).
- `layer_fold` dereferences `moe_out->data` directly, in tension with the iron
  rule "never dereference tensor->data on the host"; confirm moe_out is always
  host-resident.

## 7. Validation gate

- Default full-width path: the omitted ops are identity/zero, so it must be
  **byte-IDENTICAL to a clean build of the current HEAD** (same flavor, same
  input).
- `STREAM_MOE_TMP_BUCKET_ROUNDS=1`: still takes the old path, relaxed gate
  (maxAbs <= 1e-5 / cos ~= 1.0).
- Keep the existing TMR diagnostics to compare before/after (device graph node
  count, `chain_tail` / `tail_sync`).

## 8. Tasks (design in §9)

- **P1-a** conditional emission: cur/weights gather identity-skip + contiguous
  optimization.
- **P1-b** single target: ACC direct-writes `moe_out`, skip host `layer_fold`.
- **P1-c** scratch: per-layer grow-only host scratch, kill executor keep-alive
  small vectors.
- **P1-d** merge `mix_plan` into the scratch (flat ids/scatter, rounds become
  spans).
- **P1-e** replace `twin`/`gather_cache` with fixed arrays.
- **P1-f** regression (default single pool byte-IDENTICAL vs a clean HEAD build +
  forced-bucket relaxed gate + TMR before/after).
- **P1b** (later) CONT measurement / per-k add chain.
- **P2** (later) per-layer submit+sync.

## 9. Unified path + build-time conditional emission (agreed 2026-09-09)

Goal: keep the **single executor**; for a single pool, do not emit the redundant
nodes at build time (equivalent to graph optimization) instead of adding a second
path. The same change eliminates all keep-alive small vectors.

### 9.1 Conditional emission list (per bucket)

| Node | Do not emit when | Replacement |
| :--- | :--- | :--- |
| `cur` GET_ROWS | `n_active == n_t` | leaf referencing the source |
| `weights` GET_ROWS | `cell_full` | leaf referencing the source |
| `weights` GET_ROWS | single pool (uniform contiguous k set) | k-slice view + token gather |
| `ACC` | `single_target && one bucket` | `sum_rows` output pinned to `moe_out` |
| host `layer_fold` | `single_target` | acc bound to output / one D2H |

Single pool => one full-width bucket => the graph reduces to "original closure +
one expert fold", almost no redundant nodes.

### 9.2 Scratch (kill the small vectors)

Fixed grow-only host scratch (one per layer, process lifetime; capacity set from
the current `n_t`/`n_k` at rebuild, `n_k` constant), bump-allocated with per-round
offsets:

- i32 region: every index/ids that must live until `graph_compute`
  (`ids_exp`/`ids_slot`/cur idx/weights idx/scale ids), replacing `mm_ids_pool`;
  bound `sum <= ~4*n_k*n_t + n_t`, one allocation.
- f32 region: CPU fold temporaries, replacing `fold_buf`; devices still use the
  arena.
- `t_round`/`order`/local `idx` become scratch slices (pointer + length), no
  `push_back`.
- `twin`/`gather_cache`: closure is <= ~20 nodes/bucket, so fixed small arrays
  with linear scan can replace the `unordered_map`s.
- **Ownership / sizing (decided 2026-09-09)**: not per-layer; a
  `thread_local exec_scratch_t` (one per executing thread, reused within the
  single-layer-serial exec). Within a layer it is **written at build, then
  read-only**; reset and overwritten across layers. Per layer the exact `i32`/
  `f32` need is computed from `n_t` and the actual rounds and **reserved once**;
  exec only bumps (no grow; reserving first keeps leaf data pointers stable).
  **Implementation trap**: the `f32` need must include the gather outputs (CPU
  `bind_fresh(...,false)` also uses `f32`); missing them overruns `sum_rows`.
- **mix_plan storage (decided 2026-09-09)**: flat `ids` (i32) + `scatter`
  (`mix_scatter_t`), used length = actual `sum width*n_active`; `mix_round_t`
  becomes `{pool,width,n_active,ids_off,scatter_off}` and rounds use a **fixed
  span array** (cap `n_pools*n_k`).

### 9.3 The one non-trivial node: the fold's CONT (kept in P1)

At a single pool the fold still runs (expert-axis sum), it is not identity/zero;
the CONT in `permute+cont+sum_rows` is an artifact of `sum_rows` requiring a
contiguous ne0, not a mathematical necessity. **Decision (2026-09-09): keep CONT
in P1**; measure in P1b before deciding whether to rewrite the fold as a per-k
view+add chain (`w_b-1` adds, no transpose; for small decode tensors the extra
add nodes may be worse).

### 9.4 Decisions (2026-09-09)

- **Fold unchanged**: P1 only does conditional emission; CONT stays.
- **`mix_round_t::ids/scatter` move into the scratch**: `build_mix_plan` writes
  flat ids/scatter into the scratch, rounds become spans; update `test_mix_plan`
  accordingly.
