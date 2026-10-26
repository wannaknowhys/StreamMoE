# Device-Side Fold + Submission-Unit Arena + Head Fusion (design)

[English](DEVICE_FOLD_HEAD_FUSION.md) | [简体中文](DEVICE_FOLD_HEAD_FUSION.zh-CN.md)

> Status: **A accepted (2026-10-26). B revised: arena planning per submission unit
> instead of per-layer scratch parity.** Two changes to the only executor
> `exec_layer_burst_chain_buckets` (`src/backend/minigraph_exec.cpp`):
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

### 4.2 Submission unit: definition

The executor's per-token structure is:

```
U0      = [ head(L0) ]
U_k     = [ closure(L_k) + tail(L_k) + head(L_k+1) ]      k = 0 .. n-2
U_last  = [ closure(L_{n-1}) + tail(L_{n-1}) ]
```

Each unit ends at the ids readback (the only host break per layer), i.e. at a
**sync**. Within a unit the nodes run in exactly the order
closure -> tail -> head; across units the window may be reused because the sync
retires every in-flight read.

The plan therefore becomes:

- pack every unit's non-carry, non-view nodes with the existing
  `pack_interval` liveness packing, per device, using a time axis that spans the
  unit (`head(L+1)`'s nodes are after `tail(L)`'s);
- `unit_stride = max over units of (packed unit size + runtime bump bound)`;
- `scratch_size = unit_stride` and every unit uses the same window (offset 0),
  exactly like today's per-layer reuse, but the unit is bigger;
- the closure's `out_off` and `result_bytes` are the unit's packed closure part;
  the runtime bump (`pool_bump`) keeps starting at `base = unit_static`, i.e.
  after the static region, as today;
- CPU full-alloc (`moe_chain_set_full_alloc`) is sized by the same unit window.

### 4.3 Why this is legal (and why parity was a hack)

- Reuse is legal **between** units because a sync separates them; it is illegal
  **inside** a unit unless liveness says so, which is exactly what
  `pack_interval` already computes.
- The unit boundary is fixed by the *placement and the graph* (one head per
  layer, one ids barrier per layer), **not** by the runtime routing: whether a
  token's experts land in RAM or VRAM changes which rounds run, not which nodes
  exist or their order. So the unit plan is build-time deterministic and stable
  across tokens.
- Parity (2x scratch) only exists to make an unsynced cross-layer pair safe; a
  unit plan proves the same property for the actual pair and uses one window.
  Memory: no doubling (unit ~ max layer size, plus the internal tail->head
  tensor that is now live across the junction, one hidden state).

### 4.4 What the unit plan covers

| buffer | inside the unit? | handled by |
| :--- | :--- | :--- |
| dense head/tail temporaries | yes | unit liveness pack |
| closure static nodes (`out_off`, `result_bytes`) | yes | unit liveness pack |
| closure runtime bump (`pool_bump`: twins/gathers/fold) | yes (black box) | reserved after `unit_static`, bounded per device |
| tensor produced by tail(L), consumed by head(L+1) | **yes** | unit scratch (not carry); its relay disappears when both are on D |
| tail/head cross-device edge | no (two device plans) | xfer shell (today: separate region; can be absorbed into the unit pack later) |
| `cur` for closure(L) from head(L) | no (crosses the ids barrier) | `carry1` (parity) |
| cross-N retained state | no | `carryN` |
| KV cache | no | llama-owned |

So: everything whose whole live range fits inside the unit collapses into one
plan; only what crosses the ids barrier (`carry1`) or is retained (`carryN`)
stays special. That matches "everything except the cross buffers" - with the
addition that cross-1 is forced by the ids barrier, not by the plan.

### 4.5 Multiple residual streams

The plan is per tensor, so k residual streams are just k tensors per boundary:

- streams consumed inside the unit are unit scratch;
- streams crossing the ids barrier are `carry1`, whose boundary set is already a
  set (parity applies to the whole set, so 2x the set, not 2x per stream);
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

- Step B0 (unit plan only, no fusion): `run_baseline` PASS with **IDENTICAL**
  moe/upstream; `verify_places` unchanged (8 exact + the same 4 known diffs);
  smoke PASS. This is the strong gate: layout changes must not move numerics.
- Step B1 (fusion): per-layer `STREAM_MOE_TMP_DUMP` `moe_out`/hidden vs B0 on
  gemma + deepseek must be byte-identical; TMR `head` bucket drops to 1
  call/token and per-layer submissions become 1; 12/12 topologies, 0 ERROR.
- TMR before/after: `chain_tail`/`tail_sync` and the submission count.

## 5. Order, cost, expected gain

| step | touches | risk | expected |
| :--- | :--- | :--- | :--- |
| B0. submission-unit arena plan (layout only, no fusion) | `layout_arena` scratch section + `result_bytes`/windows + CPU full-alloc + device `arena_used` | medium (layout), zero numeric risk | removes the aliasing constraint; **no extra scratch memory** (unlike parity's 2x); no perf claim |
| A. device-side fold | exit block + fold plan + staging | medium (multi-target path) | mixed-pool: -1 sync, -1 D2H/H2D per layer (~0.2-0.4 ms/layer on olmoe; more on deepseek) |
| B1. head fusion | Tier-2 block + loop | medium (aliasing now provable) | -1 submission/layer (~0.2-0.3 ms/layer on RX590; bigger once CUDA replay exists) |

B0 is a prerequisite for B1 and can land first (it is behavior-neutral); A is
independent and needed for mixed pools; B1 turns the currently dead
`can_fuse_head_next` path live. None of them replaces the per-node dispatch cost
measured in `BENCHMARK.md`: on RX590 the decode ceiling stays ~20-40 t/s even
with zero breaks. The target (P100/CUDA) must be measured with the same probe
before deciding how far to push B1.

## 6. Open questions

1. Should the foreign-device partial path (multi-GPU, A) be implemented now, or
   only the same-physical-device case? (The target is one P100 + RAM, so the
   former is not needed for the first milestone.)
2. Unit plan: should `head(L0)` and the final `[closure+tail]` be separate units
   as described (yes by the sync argument), or is there value in a smaller
   second window for them? (Recommend same window; they are max-layer sized.)
3. Do we absorb `xfer` shells into the unit pack in B0, or keep the separate
   xfer region and do it later? (Recommend later - one change at a time.)
4. Do we demote the internal tail(L)->head(L+1) carry to unit scratch in B0?
   (Cheap once the unit exists, but it changes the relay sets; recommend a
   separate step after B0 is verified byte-identical.)
5. For CUDA replay later: is the round-shape cache keyed by
   `(layer, stage, width, n_active)` enough? (Decode: width in [0..n_k].)
