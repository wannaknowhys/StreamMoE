# Device-Side Fold + Submission-Unit Arena + Head Fusion (design)

[English](DEVICE_FOLD_HEAD_FUSION.md) | [简体中文](DEVICE_FOLD_HEAD_FUSION.zh-CN.md)

> Status: **A accepted (2026-10-26); B0b implemented and verified (2026-10-27).**
> `layout_arena` now plans per submission unit and pins the head->closure set in
> a new `sync` region - no parity, no carry reclassification (see §4.2). B1
> (head fusion) is the remaining step. Background: two changes to the only
> executor `exec_layer_burst_chain_buckets` (`src/backend/minigraph_exec.cpp`):
>
> - **A. Device-side fold** (accepted): keep the per-pool partial accumulator and
>   `moe_out` on the device that owns the tail, instead of reading every device
>   partial back to the host and folding there (the multi-target path only).
> - **B. Submission-unit arena + per-layer single submission**: plan the arena per
>   *submission unit* `[closure(L) + tail(L) + head(L+1)]` (liveness packing inside
>   the unit, window reused across units), then fuse that unit into one device
>   graph. This replaces the earlier "per-layer scratch parity" idea: parity
>   doubles scratch memory to dodge aliasing, while unit planning *proves* the
>   reuse and costs no extra memory.
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
  answer: in-place into `moe_out` when `moe_out_same_dev`
  ([:2406-2413](src/backend/minigraph_exec.cpp:2406),
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
([route_b_chain.cpp:506-578](src/backend/route_b_chain.cpp:506)); node data =
`base + carry1 + carryN + scratch_off`; `ex->result_bytes` is that same layer
window (`lb`), and the closure's runtime bump (`pool_bump`, [:2195-2211](src/backend/minigraph_exec.cpp:2195))
starts at `base = result_bytes`, i.e. **after** the static region. This is why
cross-layer fusion aliases, and also why the runtime bump is already outside the
static layout.

Tier-2 already fuses `MoE(L)+Tail(L)` when a single device target exists
([minigraph_exec.cpp:2464-2530](src/backend/minigraph_exec.cpp:2464)); head
fusion is hard-disabled at [:2526](src/backend/minigraph_exec.cpp:2526) and
`out_fused_head_next` is never set true, so the `continue` branch at
[:3348](src/backend/minigraph_exec.cpp:3348) is dead.

Carry: `carry1` is the cross-1 boundary set, **double-buffered by layer parity**
([route_b_chain.cpp:462-481](src/backend/route_b_chain.cpp:462)); `carryN` is
packed as retained ([483-494](src/backend/route_b_chain.cpp:483)); both are
relayed at device boundaries. KV cache is llama-owned and outside the arena.

## 3. Design A - device-side fold (accepted)

**Decision (2026-10-26): accepted as designed; implement before B.**

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

## 4. Design B - submission-unit arena + head fusion

### 4.1 The real blocker

`layout_arena` gives every layer the same scratch window (`scratch_size = max`,
offsets from 0). Fusing `MoE(L)+Tail(L)` with `Head(L+1)` puts both layers'
temporaries in one graph and therefore in the same offsets; the CHECKPOINT entry
of 2026-10-07 recorded GPU address collisions (garbled output) when that was
attempted. The router-ids readback is **not** a blocker: ids(L+1) are read after
the fused graph returns, which is exactly where the current per-layer sync
already happens.

### 4.1b Finding (2026-10-27): the ids barrier cuts inside the layer

`layout_arena` packs per layer because the **whole layer is one allocation
window**: head, closure and tail share it, so tensors crossing the ids barrier
(`ffn_norm-*` = the closure's `cur`, `ffn_moe_*` ids/topk/weights, the residual)
are ordinary scratch nodes with one offset.

A unit `[closure(L) + tail(L) + head(L+1)]` puts `head(L+1)` in a **different
unit** than `closure(L+1)`, so head(L+1)'s outputs consumed by closure(L+1)
cross a unit boundary. They are **not** `carry` today (carry means "consumed in a
different layer", [route_b_chain.cpp:277-287](src/backend/route_b_chain.cpp:277)),
and they are not relayed - they are plain scratch. Reusing the unit window across
units would let unit k+1's temporaries overwrite them before closure(L+1) reads
them.

Evidence from the plan dump (`STREAM_MOE_TMP_DUMP_PLAN=1`, olmoe, place-cpu):

```
[ 30] ffn_norm-0        MUL  L0  scratch off=41943040 dev=CPU(host)
[ 38] ffn_moe_gate-0    MUL_MAT_ID L0  scratch off=8388608 ...
```

`ffn_norm-0` (the closure's `cur`) is an L0 scratch node and its consumer
`ffn_moe_gate-0` is in the same L0 closure - so `carry` does not contain it.

Two viable B0s follow.

### 4.2a Option B0a - per-layer parity (NOT TAKEN)

Kept for the record: this was the low-risk fallback (2x static scratch). B0b
below was implemented instead, so there is no parity slot anywhere.

Keep the existing per-layer window and give layer L the slot `L & 1`:

- `stride = max over layers (packed layer size)`; `scratch_size = 2 * stride`;
- node offset = `off_scratch + (L & 1) * stride + scratch_off_L[node]`;
- `ex->result_bytes = 2 * stride` (the runtime bump base must be above **both**
  slots, otherwise a prefill-size bump of layer L can spill into head(L+1)'s
  slot);
- `plan.need` follows from `scratch_size`; the CPU full-alloc block grows to cover
  the parity offsets because it indexes the same `out_off`;
- apply only to real-device plans (`D != ""`); the host plan keeps 1x (no device
  graph can fuse, so a CPU-only run is unchanged).

Cost: 2x static scratch per device (olmoe prefill 71 MB -> 142 MB; gemma prefill
128 MB -> 256 MB; bump unchanged). No reclassification, no relay changes, and the
layout change is byte-identical (offsets only).

### 4.2 Option B0b - submission-unit window (IMPLEMENTED 2026-10-27)

**Result:** `layout_arena` now plans per unit and pins the head->closure set in a
new `sync` region. No parity, no carry reclassification: `carry1`/`carryN` keep
their meaning and relay path. B0a was not needed.

The executor's per-token structure is:

```
U0      = [ head(L0) ]
U_k     = [ closure(L_k) + tail(L_k) + head(L_k+1) ]      k = 0 .. n-2
U_last  = [ closure(L_{n-1}) + tail(L_{n-1}) ]
```

Each unit ends at the ids readback (the only host break per layer), i.e. at a
**sync**. Within a unit the nodes run in exactly the order
closure -> tail -> head; across units the window is reused because the sync
retires every in-flight read.

The implementation (all in `layout_arena`):

- units are built from `g_layer_nodes` + the existing `moe_layer_plan_t.head/tail`
  and `moe_layer_exec_t.compute` lists, mirroring the executor's pairing;
- `sync` = captured tensors whose producer unit differs from a consumer unit and
  which are not already in `carry`. Pinned in `[carry1][carryN][sync][scratch]`;
  one slot per unit (`sync_size = max over units of the unit's set`), reused
  across units;
- every unit's non-view, non-carry, non-sync nodes are packed with
  `pack_interval` on a time axis that spans the unit (`head(L+1)` after
  `tail(L)`); `scratch_size = max over units`; the same window (offset 0) is
  reused across units;
- the closure's `out_off` and `result_bytes` are the unit's packed closure part
  (`result_bytes = unit_static`), so the runtime bump starts above every unit
  static; the CPU full-alloc block and the device arena both index the same
  `out_off`;
- the `sync` slot is reused across units because unit k+1 writes its own set
  only at its **end** (the head nodes), after unit k's consumers ran. That
  ordering is what makes one slot enough - see the pitfall below.

**Pitfall found during bring-up (must stay fixed):** the consumer scan must skip
alias/view consumers (`view_src != NULL`). `build_layer_plans` can attribute a
view of a closure output (e.g. `ffn_moe_weighted`) to the head list of the same
layer, i.e. a *different* unit; counting that view as a consumer marks closure
outputs as `sync`. They then share the single sync slot with the previous unit's
live set (`ffn_norm`/`ffn_inp`), and the closure writes them mid-layer, so later
rounds read clobbered `cur` - a deterministic numeric change (observed as
`place-c1c2-exp1` flipping from PASS to DIFF, 8/4 -> 7/5). Skipping view
consumers and following the real consumer (`ffn_moe_out`) fixes it.

### 4.3 Why this is legal

- Reuse is legal **between** units because a sync separates them. Tensors live
  across the boundary (`sync`) are pinned; everything else must die inside the
  unit, which is what `pack_interval` proves.
- The unit boundary is fixed by the *placement and the graph* (one head per
  layer, one ids barrier per layer), **not** by the runtime routing: whether a
  token's experts land in RAM or VRAM changes which rounds run, not which nodes
  exist or their order. So the plan is build-time deterministic and stable
  across tokens.
- `carry` is unchanged: tail(L)->head(L+1) is cross-layer and stays in
  `carry1`, even though it is now unit-internal. Demoting it to scratch is a
  separate, later change (it would touch the relay sets).

### 4.4 What the unit plan covers

| buffer | inside the unit? | handled by |
| :--- | :--- | :--- |
| dense head/tail temporaries | yes | unit liveness pack |
| closure static nodes (`out_off`, `result_bytes`) | yes | unit liveness pack |
| closure runtime bump (`pool_bump`: twins/gathers/fold) | yes (black box) | reserved after `unit_static`, bounded per device |
| tensor produced by tail(L), consumed by head(L+1) | **yes** | unit scratch (it is carry today; in B0b it is internal and its relay disappears when both are on D) |
| tail/head cross-device edge | no (two device plans) | xfer shell; the region is split by stage (TAIL vs LAYER_FRONT/CLOSURE) so a fused tail+head's two layers never share a slot (§4.6b) |
| `cur` for closure(L) from head(L) | no (crosses the ids barrier) | **`sync` region** (new; one slot per unit) |
| router ids/topk/weights (head -> closure) | no (same boundary) | `sync` region |
| cross-N retained state | no | `carryN` |
| KV cache | no | llama-owned |

So: everything whose whole live range fits inside the unit collapses into one
plan; only what crosses the ids barrier (`sync`) or is cross-layer (`carry1`) /
retained (`carryN`) stays special. Measured on olmoe prefill 2048 (place:
C1/C2 Vulkan, RAM+Vulkan0 pools): `sync` = 8.5 MB/unit, `scratch` = 102 MB,
`carry1` = 8 MB - i.e. B0b costs +8.5 MB, not the +102 MB of parity.

### 4.5 Multiple residual streams

The plan is per tensor, so k residual streams are just k tensors per boundary:

- streams consumed inside the unit are unit scratch;
- streams crossing the ids barrier are `sync`; the slot is per unit and holds the
  whole set, so k streams cost k tensors once, not k windows;
- retained streams stay in `carryN`.

No new mechanism is needed. Two assumptions remain outside this design:
**one MoE closure per layer** (`g_layer_exec` has one `moe_layer_exec_t` per
layer and the closure takes the first `MUL_MAT_ID`), and one accumulator
(`acc_d`/`moe_out`) per layer. An architecture with several MoE blocks per layer
would need a separate change.

### 4.6 Caveats / non-goals

1. **Runtime bump region**: it is not part of the static liveness model. Keep
   today's rule (`base = unit_static + acc`, bump after) so it can never overlap
   tail/head statics; the bound per device is `max over layers(pool_bump)`.
2. **Two buffers per device**: device arena and the CPU full-alloc block both
   index by `out_off`; both must be sized with the unit window.
3. **xfer shells** are not absorbed in the first step (keep the separate region,
   packed per consumer layer as today).
4. The unit plan is a **layout-only** change and must be verified byte-identical
   on the current two-graph/two-submission paths before fusion is enabled.

### 4.6b xfer slots (finding 2026-10-29)

A fused `tail(L)+head(L+1)` graph dispatches the head's incoming H2D relays
**before** the segment, while the tail's `moe_out` shell is still live. The xfer
region used to pack one slot per layer and reuse it across layers; with two
layers in one graph, layer L+1's 64KB `ffn_moe_weighted` shell
(0x3C640..0x4C640) covered layer L's 8KB `moe_out` shell at 0x4C600, and the
head relay overwrote the first 16 floats of `moe_out`. That was the c1c2 text
nondeterminism: the tail output is `corrupted_moe_out + ffn_inp`, the stale 16
floats depended on what the slot held before, so runs diverged (and the 1B model
degraded into repetition loops). Fix: `xfer_size = stride_tail + stride_other`;
TAIL shells in slot 0, LAYER_FRONT/CLOSURE shells in slot 1; same-slot reuse
across layers remains legal because a sync separates the calls.

### 4.7 Head fusion construction (after the unit plan lands)

- In the Tier-2 block ([:2464-2530](src/backend/minigraph_exec.cpp:2464)),
  after appending the tail nodes, append `Head(L+1)`'s dense nodes in
  topological order (they are inside the unit, so their offsets are already
  planned).
- Dispatch incoming relays for `Head(L+1)` **before** submitting (same-device
  placement means most are internal; external ones are KV/positions already
  resident). The graph must not contain a relay whose source is produced later
  in the same graph.
- Set `out_fused_head_next = true`; the caller loop already handles
  `fused_tail && fused_head_next -> continue` ([:3348](src/backend/minigraph_exec.cpp:3348)).
- ids(L+1) are read by the next `exec_moe_closure(L+1)` after this layer's sync;
  no change to the pin path.
- Keep Tier-1 as the fallback for layers where the head is not fusable.

### 4.8 Eligibility predicate (per layer, runtime structural)

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

### 4.9 Correctness invariants

1. No two nodes in a unit share a scratch byte range unless their liveness
   intervals are disjoint under the unit's execution order. No reuse across the
   unit boundary unless a sync separates them (by construction it does).
2. The fused node list is topologically ordered; the tail (L) comes before the
   head (L+1), and no head node reads a tail node that has not been emitted.
3. `executed_relays` bookkeeping is shared across the fused call so tail/head
   relays are dispatched exactly once.
4. Tier-1 and Tier-2 are mutually exclusive per layer (assert
   `!(fused_tail && fused_head_next)` in the Tier-1 branch).
5. The final layer has no `Head(L+1)`; the last `exec_moe_closure(-1)` path is
   unchanged.
6. Numerical behavior: neither the unit plan (layout only) nor the fusion
   (submission grouping only) changes arithmetic, so both should be
   **byte-identical** to the current path. Any difference is a bug.

### 4.10 Failure modes and disposition

| failure | disposition |
| :--- | :--- |
| head node unsupported on D | predicate false -> Tier-1 fallback |
| unit window allocation fails (device OOM) | fail loudly; never silently alias |
| relay source produced inside the graph | predicate false (check before building) |
| fused graph exceeds `gf->size` | grow the graph (no silent drop; today's [:2508](src/backend/minigraph_exec.cpp:2508) drop must become a hard error) |
| ids(L+1) missing / 0 tokens | keep the existing B38 0-token early return |

### 4.11 Verification

- Step B0b (unit window + sync region, no fusion): **done 2026-10-27.**
  `run_baseline moe_129_8192_vk` PASS: moe/upstream **IDENTICAL**, embd gate
  121/129 = 93.8% (unchanged), KL max 4.027535e-01 (unchanged), places **8 PASS /
  4 DIFF** (the historical set), smoke PASS; `build.bat test main` 6/6 + guard
  PASS; `STREAM_MOE_TMP_DUMP_PLAN` shows `sync`=8.5 MB, `scratch`=102 MB on
  olmoe prefill 2048.
- Step B1 (fusion): per-layer `STREAM_MOE_TMP_DUMP` `moe_out`/hidden vs B0b on
  gemma + deepseek must be byte-identical; TMR `head` bucket drops to 1
  call/token and per-layer submissions become 1; 12/12 topologies, 0 ERROR.
- TMR before/after: `chain_tail`/`tail_sync` and the submission count.

## 5. Order, cost, expected gain

| step | touches | risk | expected |
| :--- | :--- | :--- | :--- |
| B0b. unit window + `sync` region (layout only) | `layout_arena` (units, sync set, pack, `result_bytes`) + `region_plan_t` | medium; byte-identical | **done 2026-10-27**; +8.5 MB `sync` instead of +102 MB parity; unblocks B1 |
| B1. head fusion | Tier-2 block + loop | medium (aliasing gone by construction) | -1 submission/layer (~0.2-0.3 ms/layer on RX590; bigger once CUDA replay exists) |
| A. device-side fold | exit block + fold plan + staging | medium (multi-target path) | mixed-pool: -1 sync, -1 D2H/H2D per layer (~0.2-0.4 ms/layer on olmoe; more on deepseek) |

B0b landed first (behavior-neutral, strong gate); A is independent and needed
for mixed pools; B1 turns the currently dead `can_fuse_head_next` path live. None
of them replaces the per-node dispatch cost measured in `BENCHMARK.md`: on RX590
the decode ceiling stays ~20-40 t/s even with zero breaks. The target (P100/CUDA)
must be measured with the same probe before deciding how far to push B1.

## 6. Open questions

1. Should the foreign-device partial path (multi-GPU, A) be implemented now, or
   only the same-physical-device case? (The target is one P100 + RAM, so the
   former is not needed for the first milestone.)
2. ~~B0a vs B0b~~ **resolved 2026-10-27:** B0b implemented (unit window + `sync`
   region); B0a (parity) dropped.
3. Do we absorb `xfer` shells into the unit pack in B0b, or keep the separate
   xfer region and do it later? (Recommend later - one change at a time.)
4. Do we demote the internal tail(L)->head(L+1) carry to unit scratch in B0b?
   (Cheap once the unit exists, but it changes the relay sets; recommend a
   separate step after B0b is verified byte-identical.)
5. For CUDA replay later: is the round-shape cache keyed by
   `(layer, stage, width, n_active)` enough? (Decode: width in [0..n_k].)

## 7. Pending work (detailed)

Status 2026-10-27: B0b is in (unit arena + `sync` region). The two remaining items
are recorded here with their exact anchors. Line numbers are from 2026-10-27
(`src/backend/minigraph_exec.cpp`) and will drift; the function names are stable.

### 7.1 A - device-side fold (multi-target exit)

**Current path (what to remove):** with a mixed pool (some rounds on the CPU, some
on a device) the exit does, per layer:

1. `ggml_backend_synchronize(t.be)` - [:2626](src/backend/minigraph_exec.cpp:2626);
2. `ggml_backend_tensor_get(t.acc, dev_accs[di], ...)` - D2H - [:2666](src/backend/minigraph_exec.cpp:2666);
3. `layer_fold(ex, d_out, n_t, accs)` - host sum - [:2727](src/backend/minigraph_exec.cpp:2727), definition [:1290](src/backend/minigraph_exec.cpp:1290);
4. `tensor_write_host(moe_out, ...)` - H2D write-back - [:1308](src/backend/minigraph_exec.cpp:1308).

The sync is needed for the tail anyway; A removes the two copies and the host
fold, not the sync. `direct_out` and `single_dev_target` are already elided
([:2632-2656](src/backend/minigraph_exec.cpp:2632)), so this is **multi-target
only** (`!direct_out && !single_dev_target`).

**Design:**

1. `fold_plan_t` - owner = the device that owns `moe_out` (the target whose
   `moe_out_same_dev` is true; if none, fall back to host). Fields: owner pool,
   `foreign_pools` (accs on other devices), `needs_cpu_partial`.
2. Staging: one grow-only host-visible buffer per device,
   `d_out * n_t * 4 * (1 + n_foreign)`, reset per layer (layer execution is serial
   on the calling thread).
3. Fold graph on the owner, submitted **after** the MoE graph on the owner queue:
   start from the owner acc, then one `ACC` per foreign acc (D2D or staged) and
   one per CPU partial; the result lands in `moe_out` directly (it is already on
   the owner device).
4. CPU partial: the CPU rounds write `c.acc_d`; upload it into the owner's staging
   slot before submit (or have the CPU rounds write a host-visible slot the owner
   reads - zero copy when the arena is host-visible).
5. Delete the `tensor_get`/`layer_fold`/`tensor_write_host` sequence for
   multi-target; keep `layer_fold` as the structural fallback (owner is host,
   `moe_out` not contiguous, or an operand op unsupported by the device).

**Invariants:**

- `moe_out` is written exactly once, by the owner fold;
- every partial is fully written before the fold is submitted: device partials by
  queue order on the owner, foreign devices by an explicit fence, CPU by the host
  copy;
- the fold is submitted on the owner queue after the MoE graph;
- summation order changes (owner-first instead of host-first) -> ulp-level
  differences; the gate is the relaxed one (cos/KL), **not** byte-identical;
  document the expected diffs on the multi-pool topologies;
- the final `moe_out` readback (export/sampling) is unchanged.

**Failure modes:** staging allocation fails -> host fold fallback (log once);
foreign readback fails -> propagate (GGML_STATUS_FAILED); ACC unsupported ->
host fold fallback; `moe_out` non-contiguous/host -> host fold fallback.

**Verification:** `run_baseline` PASS with the same cos gate (embd/KL), places
8 PASS / 4 DIFF (same set; multi-pool entries are the documented ulp diffs), TMR
`tail_read`/`tail_fold` drop to ~0 and the `tail_sync` count drops by one;
~0.2-0.4 ms/layer on olmoe mixed, more on deepseek.

### 7.2 B1 - head fusion (Tier-2)

**Current path (why it is dead):** the tail fusion is built at
[:2491-2512](src/backend/minigraph_exec.cpp:2491), then
`can_fuse_head_next = false` is hard-coded at
[:2526](src/backend/minigraph_exec.cpp:2526) with the aliasing comment
[:2514-2526](src/backend/minigraph_exec.cpp:2514). B0b removed that reason (unit
window + `sync` region). The ids readback is not a blocker: it happens after the
graph returns, exactly where the per-layer sync already is.

**Scope:** `single_dev_target` + `can_fuse_tail` + `next_layer == layer + 1` +
`plan_head_next != nullptr`; everything else keeps the Tier-1 path.

**Prerequisite (4.6b):** the xfer region's stage slots must stay - the fused
graph keeps the tail (slot 0) and the head (slot 1) shells disjoint. Without
that split the head's pre-segment relays overwrite the tail's `moe_out`.

**Steps:**

1. After the tail append ([:2505-2511](src/backend/minigraph_exec.cpp:2505)),
   append `plan_head_next->head` in graph order (skip `is_alias_op`, `!data`,
   `GGML_OP_NONE`).
2. Make graph overflow a **hard error** (`n_nodes + n_head > gf->size`); also fix
   the tail append's silent drop at
   [:2508](src/backend/minigraph_exec.cpp:2508) (it must not silently lose nodes).
3. Dispatch **incoming** head relays before submit: `r.layer == next_layer`,
   stage `LAYER_FRONT`/`CLOSURE`, dst device == D; **skip** relays whose src is a
   node of the fused graph (tail outputs are internal scratch now - copying them
   from a stale source would be wrong; the head reads them directly).
4. Set `can_fuse_head_next = true` and `out_fused_head_next = true`; the loop's
   `continue` at [:3348](src/backend/minigraph_exec.cpp:3348) then skips
   `exec_dense_head(L_next)`.
5. Keep the outgoing-relay block at
   [:2771-2789](src/backend/minigraph_exec.cpp:2771) as-is: it dispatches
   head->next-layer copies **after** the graph, when the head output exists.

**Predicates:** every non-alias head node is on D and
`ggml_backend_supports_op(D)`; no external cross-device src that is not dispatched
(KV/positions/embd are device-side or staged at setup; tail->head is internal);
`moe_out_same_dev` is already required by `can_fuse_tail`.

**Invariants:** head nodes topologically after the tail; no head node reads a tail
node that has not been emitted; every relay dispatched exactly once; ids read
after the graph returns; byte-identical expected (same nodes, order, buffers).

**Failure modes:** unsupported head op -> Tier-1 fallback; graph overflow -> hard
error (never silent drop); 0 tokens -> existing B38 early return.

**Verification:** per-layer `moe_out`/hidden dumps byte-identical to B0b; TMR
`head` bucket 1 call/token and 1 submission/layer; places 12/12; `run_baseline`
PASS. Expected: -1 submission/layer (~0.2-0.3 ms on RX590, larger once CUDA
replay exists).

### 7.3 Order

B1 first (direct continuation of B0b, no new data movement), then A (mixed-pool
path); they are independent. CUDA replay / P100 measurements come after both.
