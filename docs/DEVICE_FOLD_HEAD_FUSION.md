# Device-Side Fold + Per-Layer Single Submission (design)

[English](DEVICE_FOLD_HEAD_FUSION.md) | [简体中文](DEVICE_FOLD_HEAD_FUSION.zh-CN.md)

> Status: **design proposal, 2026-10-26; not implemented.** Two changes to the
> only executor `exec_layer_burst_chain_buckets` (`src/backend/minigraph_exec.cpp`):
>
> - **A. Device-side fold**: keep the per-pool partial accumulator and `moe_out`
>   on the device that owns the tail, instead of reading every device partial
>   back to the host and folding there (the multi-target path only).
> - **B. Per-layer single submission**: fuse `[MoE(L) + Tail(L) + Head(L+1)]`
>   into one device graph, which needs per-layer scratch parity because the
>   arena currently shares one scratch window across layers.
>
> Related: `M2_DEVICE_EXECUTOR.md` (§7.8 exit, §7.9 device rounds),
> `PER_DEVICE_ARENA.md` (§6 regions, §7 open items), `DEVICE_DENSE_CLOSURE.md`
> (3.6 per-device `moe_out`, 3.8 grow-only), `ASYNC_PIPELINE_ROADMAP.md`
> (Phase 3), `BUCKET_FAST_PATH.md` (P1-b single target), `BENCHMARK.md`
> (2026-10-26 dispatch measurements).

## 1. Why, in numbers

2026-10-26 measurements (RX590 Vulkan, olmoe, 23 decode tokens, 16 layers):

| config | eval/token | head | fused tail+head | MoE closure | relay |
| :--- | ---: | ---: | ---: | ---: | ---: |
| place-cpu | 20.6 ms | 0.4 | 5.5 (15x0.37) | 12.7 (16x0.79) | 0 |
| place-c1 | 75.7 ms | 2.7 | 49.3 (15x3.29) | 21.2 (16x1.33) | 4.4 |
| place-exp5 | 60.0 ms | 1.1 | 19.4 (15x1.29) | 37.3 (16x2.33) | 0 |
| place-c1c2-exp5 | 80.8 ms | 40.8 (16x2.55) | - (Tier-2) | 40.1 (16x2.50) | 0.1 |

Launch probe (`diagnostics/vk_launch_probe.cpp`): one Vulkan graph submission
costs ~0.17 ms + **~39 us per node** of GPU time plus ~17 us per node of CPU
recording; the per-node cost dominates for small ops (13% at 256^2 MUL_MAT,
0.5% at 2048^2). Consequences:

- A/B cannot remove the per-node cost. They remove **sync round-trips, one
  submission per layer, and the host fold**, i.e. ~5-10% on this driver and
  proportionally less when the per-node compute is larger (deepseek/P100).
- They are still worth doing because they are the structural prerequisite for
  (a) the mixed-pool target (DeepSeek on RAM + P100) and (b) per-layer graph
  replay on CUDA, where the gain is much larger.

## 2. Current code map (what actually runs today)

Exit of `exec_layer_burst_chain_buckets` (`minigraph_exec.cpp`):

- `direct_out` (one full-width round) -> the round's `per_token` **is** the
  answer: in-place into `moe_out` when `moe_out_same_dev` ([:2406-2413](src/backend/minigraph_exec.cpp:2406),
  [:2632-2642](src/backend/minigraph_exec.cpp:2632)); one D2H when `moe_out` is
  host.
- `single_dev_target` (all rounds on one device, no CPU round) -> `t.acc` is
  complete, in-place / D2H / D2D ([:2643-2656](src/backend/minigraph_exec.cpp:2643)).
- **everything else (multi-target: RAM round + device round(s))** -> each device
  `ggml_backend_tensor_get(t.acc, dev_accs[di])` ([:2658-2670](src/backend/minigraph_exec.cpp:2658)),
  then `layer_fold()` ([:2727-2728](src/backend/minigraph_exec.cpp:2727)) sums the
  host copies and `tensor_write_host(moe_out, ...)` ([:1290-1310](src/backend/minigraph_exec.cpp:1290)).
  The CPU round accumulates in `c.acc_d` ([2437-2441](src/backend/minigraph_exec.cpp:2437)).

`moe_out` (`ffn_moe_out`) is a captured node; when the closure device is a real
device its `buffer`/`data` point into that device's `plan.buf`
([route_b_chain.cpp:640-646](src/backend/route_b_chain.cpp:640)). So on the GPU
path the fold costs a D2H + a host fold + an H2D back into the device buffer.

Scratch: `plan.scratch_size = max over layers` and every layer's offsets are
packed from 0 inside that single window
([route_b_chain.cpp:565-578](src/backend/route_b_chain.cpp:565)); node data =
`base + carry1 + carryN + scratch_off`. This is why cross-layer fusion aliases.

Tier-2 already fuses `MoE(L)+Tail(L)` when a single device target exists
([minigraph_exec.cpp:2464-2530](src/backend/minigraph_exec.cpp:2464)); head
fusion is hard-disabled at [:2526](src/backend/minigraph_exec.cpp:2526) and
`out_fused_head_next` is never set true, so the `continue` branch at
[:3348](src/backend/minigraph_exec.cpp:3348) is dead.

## 3. Design A - device-side fold (multi-target)

### 3.1 Goal / non-goals

Goal: for a layer whose rounds touch both host and one or more devices, produce
`moe_out` on the device that owns the tail without a host fold round-trip, and
with one device sync instead of two.

Non-goals: changing the single-target paths (already elided); peer-to-peer
device copies (no semaphore exposure); moving the CPU round to the device.

### 3.2 Owner selection

`owner = dev_of[moe_out]` (the tail device), when `moe_out->buffer` is a device
buffer. If `moe_out` is host (C1 on CPU), keep the current host fold: that path
already ends with one D2H per device and no H2D. If there is no device round at
all, nothing changes.

### 3.3 Transport of the non-owner partials

- CPU partial (`c.acc_d`): a **host-visible staging tensor** on the owner device
  (same host buft as `g_pinned_h2d`, grow-only, one slice per layer, not shared
  with the transfer staging ring). The host writes it with a plain memcpy after
  the CPU graph finishes; the fold graph then reads it as an operand. Vulkan
  supports host-buft graph operands (the stage leaves already rely on this), so
  no `set_tensor` and no extra queue sync is needed.
- Another **physical** device's partial: read it back to a host-visible tensor
  (D2H + fence of that device) before the fold graph is submitted; the fold
  graph reads that host-visible tensor directly. This costs one sync of the
  source device, unavoidable without semaphores; it is still cheaper than today
  (no H2D of `moe_out`, no host fold, and the tail can be fused).
- Multiple chunks/pools on the **same physical device**: their acc tensors live
  in the same device buffer, so the fold graph reads them directly (no copies).

### 3.4 Graph shape

```
fold(L) on owner D:
    moe_out = acc_D
            + staged_cpu_partial              (host-visible operand)
            + staged_other_dev_partial_i ...  (host-visible operands)
```

Implemented with `ggml_add` nodes in a small `ggml_cgraph` appended to the
device target's graph (`single_dt->gf` when tail is fused, or its own graph when
not). The node count is tiny (~1 + N_partials), so the per-node cost is not the
issue; the win is removing the host fold and one sync.

For the `direct_out`/`single_dev_target` cases the existing elision stays: do not
run the fold graph when the owner accumulator already equals `moe_out`.

### 3.5 Control flow (multi-target, one physical device)

Today:

```
submit device MoE graphs (async)
run CPU graph                       -> c.acc_d
sync device(s)                      # 1
D2H each device acc                 # copies
host fold -> host vector
tensor_write_host(moe_out)          # H2D
submit tail+head graph (Tier-1)
sync                                # 2
```

With A:

```
submit device MoE graphs (async)
run CPU graph                       -> c.acc_d
write c.acc_d into the host-visible partial (memcpy, no device work)
[other physical devices: D2H their acc -> host-visible partial + fence]
submit fold+tail(+head) graph on D  # reads acc_D and the host partials
sync                                # 1
```

### 3.6 Data structures

```cpp
struct fold_plan_t {
    int device_pool = -1;            // owner pool (or -1 = host fold, current path)
    bool needs_cpu_partial = false;  // c.acc_d -> host-visible staging
    std::vector<int> foreign_pools;  // pools whose acc is read back first
    // graph fragment is appended to the owner target, no new buffer type
};
```

Staging: one grow-only host-visible buffer per device, size `n_d_out * n_t *
4 * (1 + n_foreign)`; slots reset per layer (layer execution is serial on the
calling thread).

### 3.7 Correctness invariants

1. `moe_out` is written exactly once per layer, by the owner fold graph (or by
   the existing elision).
2. Every participant partial is fully written before the fold graph is
   submitted: device partials by the queue order on D, foreign devices by the
   explicit fence, CPU partial by the host memcpy before submit.
3. The fold graph is submitted on D's queue after the MoE graph, so ordering is
   by queue and not by wall clock.
4. Summation order changes (device partial + staged partials instead of
   CPU-first host order). This is a ulp-level difference and is covered by the
   project's relaxed gate (maxAbs <= 1e-5 / cos ~= 1.0), not by byte equality.
5. Last layer / export: the final `moe_out` (and any exported tensor) is still
   read back to the host exactly as today; A does not remove the final D2H.
6. If any predicate fails (owner is host, `moe_out` not contiguous, an operand
   is unsupported by the device), fall back to the current host fold for that
   layer - a structural fallback, not a build-time gate.

### 3.8 Failure modes and disposition

| failure | disposition |
| :--- | :--- |
| staging allocation fails | fall back to host fold for this layer, log once |
| foreign device acc readback fails | propagate (GGML_STATUS_FAILED) |
| fold node unsupported by device (`supports_op`) | host fold fallback |
| `moe_out` not contiguous / host | host fold fallback (current path) |
| cross-physical-device fence missing | must be an assert: the plan carries `foreign_pools` and the submit path fences each before the fold |

### 3.9 Verification

- `build.bat llamalibs dual` + `build.bat test main` (6/6).
- `run_baseline.bat moe_129_8192_vk`: the moe/upstream gates must stay PASS; the
  vk gate is the relaxed one (cos), so the fold-order change is allowed but must
  not change the ratio materially.
- `verify_places --flavor StreamMoE`: 12 topologies, 0 ERROR; the 8 byte-identical
  cases must stay byte-identical **except** the multi-pool topologies where the
  fold order changes (document the expected 4 known diffs; no new ones).
- TMR before/after on the mixed-pool topologies: `tail_read`/`tail_fold` should
  drop to ~0 and `tail_sync` count should drop by one.
- Expert history unchanged (routing does not depend on the fold).

## 4. Design B - per-layer single submission (scratch parity + head fusion)

### 4.1 The real blocker

`layout_arena` gives every layer the same scratch window (`scratch_size = max`,
offsets from 0). Fusing `MoE(L)+Tail(L)` with `Head(L+1)` puts both layers'
temporaries in one graph and therefore in the same offsets; the CHECKPOINT entry
of 2026-10-07 recorded GPU address collisions (garbled output) when that was
attempted. The router-ids readback is **not** a blocker: ids(L+1) are read after
the fused graph returns, which is exactly where the current per-layer sync
already happens.

### 4.2 Parity layout

- `plan.scratch_stride = max_L(lb_L)` (today's `scratch_size`).
- `plan.scratch_size = 2 * scratch_stride` (parity 0/1).
- Layer L's base = `off_scratch + (L & 1) * scratch_stride`; `scratch_off[node]`
  stays relative to that base.
- `ex->out_off[i]` for a closure node stays relative; the executor's
  `moe_chain_fullalloc_buffer`/`bind_fresh` base adds the same parity offset, and
  `result_bytes` becomes `scratch_stride`.
- CPU full-alloc block (`moe_chain_set_full_alloc`) is sized 2x for the same
  reason (the CPU side can also be fused later; for now keep it consistent).
- `xfer` shells are already keyed by consumer layer, so they are untouched.
- Memory cost: +scratch_stride per device (olmoe decode 8 MB -> 16 MB, prefill
  128 MB -> 256 MB; gemma/deepseek similar order). Acceptable.

### 4.3 Eligibility predicate (per layer, runtime structural)

```
fuse_head_next(L) =
    dense placement puts Tail(L), Head(L+1), and moe_out on the same device D
    && Head(L+1) exists and all its non-alias nodes are on D
    && all those nodes pass ggml_backend_supports_op(D)
    && no cross-device input for Head(L+1) that is produced by the graph itself
    && the layer is not the last layer
```

If it is false, keep today's behavior (Tier-1 fused tail+next head, or separate
head).

### 4.4 Construction

- In `exec_layer_burst_chain_buckets`'s Tier-2 block
  ([:2464-2530](src/backend/minigraph_exec.cpp:2464)), after appending the tail
  nodes, append `Head(L+1)`'s dense nodes in topological order.
- Dispatch incoming relays for Head(L+1) **before** submitting (same-device
  placement means most of them are internal; external ones are KV/positions
  which are already resident). The graph must not contain a relay whose source
  is produced later in the same graph.
- Set `out_fused_head_next = true`; the caller loop already handles
  `fused_tail && fused_head_next -> continue` ([:3348](src/backend/minigraph_exec.cpp:3348)).
- ids(L+1) are read by the next `exec_moe_closure(L+1)` after this layer's sync;
  no change to the pin path.
- Keep Tier-1 as the fallback: the existing `exec_fused_tail_and_head` path stays
  for layers where the head is not fusable.

### 4.5 Interaction with A

- Same-device (single pool): A is a no-op (already elided), B works alone.
- Mixed pool with one physical device: B needs A, because the fold must happen
  on D before the tail, and the tail is inside the fused graph. Implement A
  first, then B's predicate can include the mixed case.
- Multi-physical-device: B's predicate stays false (the head may be on a
  different device); keep Tier-1.

### 4.6 Correctness invariants

1. No two nodes in the fused graph share a scratch byte range unless they are
   provably dead at the same time. Parity gives this per layer; within a layer
   the existing `pack_interval` order still applies.
2. The fused node list is topologically ordered; the tail (L) comes before the
   head (L+1), and no head node reads a tail node that has not been emitted.
3. `executed_relays` bookkeeping must be shared across the fused call so the
   tail/head relays are dispatched exactly once.
4. Tier-1 and Tier-2 must be mutually exclusive per layer (assert
   `!(fused_tail && fused_head_next)` in the Tier-1 branch).
5. The final layer has no `Head(L+1)`; the last `exec_moe_closure(-1)` path is
   unchanged.
6. Numerical behavior: fusing does not change any arithmetic, only the grouping
   of submissions, so the output should be byte-identical to the two-graph path
   (same kernels, same order). Treat any difference as a bug.

### 4.7 Failure modes and disposition

| failure | disposition |
| :--- | :--- |
| head node unsupported on D | predicate false -> Tier-1 fallback |
| scratch parity not allocated (device OOM) | predicate false + log (never silently alias) |
| relay source produced inside the graph | predicate false (check before building) |
| fused graph exceeds `gf->size` | grow the graph (no silent drop; today's [:2508](src/backend/minigraph_exec.cpp:2508) drop must be turned into a hard error) |
| ids(L+1) missing/zero tokens | keep the existing B38 0-token early return |

### 4.8 Verification

- Same gates as A, plus: compare `STREAM_MOE_TMP_DUMP` `moe_out`/hidden per layer
  against the pre-change build on gemma + deepseek (byte-identical expected for
  B).
- TMR: `head` bucket should drop to 1 call/token (currently 16 at c1c2exp5), and
  per-layer submissions should be 1.
- `verify_places` 12/12; `run_baseline` PASS; smoke_cli PASS.

## 5. Order, cost, expected gain

| step | touches | risk | expected |
| :--- | :--- | :--- | :--- |
| A. device-side fold | exit block + fold plan + staging + arena sizing | medium (multi-target path) | mixed-pool: -1 sync, -1 D2H/H2D per layer (~0.2-0.4 ms/layer on olmoe; more on deepseek) |
| B. scratch parity + head fusion | `layout_arena` + Tier-2 block + loop | medium-high (aliasing proof) | -1 submission/layer (~0.2-0.3 ms/layer on RX590; bigger once CUDA replay exists) |

A is the prerequisite for B on mixed pools; on single-device placements B can
land independently. Neither replaces the per-node dispatch cost measured in
`BENCHMARK.md`: on RX590 the decode ceiling stays ~20-40 t/s even with zero
breaks. The target (P100/CUDA) must be measured with the same probe before
deciding how far to push B.

## 6. Open questions

1. Should the foreign-device partial path (multi-GPU) be implemented now, or
   only the same-physical-device case? (The target is one P100 + RAM, so the
   former is not needed for the first milestone.)
2. Parity 2 or parity N (N = max fusion depth)? 2 is enough for one-layer
   lookahead; N>2 only if cross-layer batching grows.
3. Do we need a TEMP dump (`STREAM_MOE_TMP_FOLD_DUMP`) to compare the device
   fold against the host fold per layer? Recommend yes, used only for bring-up.
4. For CUDA replay later: is the round-shape cache keyed by
   `(layer, stage, width, n_active)` enough? (Decode: width in [0..n_k].)
