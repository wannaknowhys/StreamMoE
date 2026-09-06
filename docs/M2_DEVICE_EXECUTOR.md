# M2 Device Executor - per-device expert-column chains

[English](M2_DEVICE_EXECUTOR.md) | [简体中文](M2_DEVICE_EXECUTOR.zh-CN.md)

> Status: design, 2026-09 (revised: no transitional code - implement the final
> shape directly, component by component). Builds on `docs/ROUTE_B_GPU_PHASE.md`
> (whole-layer self-scheduling, §1/§3) and the landed VRAM data layer
> (docs/WORK_IN_PROGRESS.md J: host-map channel, experts resident on device,
> demote-to-RAM).

## 0. Decision supplement 2026-09-05 (user) - buffer layout via verify, not the
##     execution path; whole-chain per device

History lesson first: the current result-buffer scheme is a maze of band-aids.
`hide_burst` forces a full-alloc per node and ping-pong was disabled
(`g_pingpong_ok=false`) after a "long-range dep: L0 compute#4 reads compute#1"
panic that was patched by turning EVERYTHING to full-alloc. That was a
whack-a-mole fix under a wrong numeric result - the real root cause was
suspected elsewhere (a DIO bug). So the buffer-layout question must be answered
by ANALYSIS, not by trial patches in the executor.

User's target shape:

1. **Layout analysis lives in build-time verify** (the same pass that already
   collects the per-layer closure `compute` sequence). For every compute-node
   RESULT, walk consumers BACKWARD from the last node (reverse order) to find
   its free moment = the last node that still reads it (last-use). This gives a
   per-result live range `[produce, last_use]`.
2. **Closure-internal cross-node dependency graph** (NOT cross-layer - layers are
   independent, verified by Check1). Two shapes:
   - chained/neighbour (result read only by the next node): ping-pong (two
     alternating result buffers) is enough;
   - long-range (a producer read many nodes later, e.g. compute#4 reads
     compute#1): needs a larger set or an interval-allocated buffer.
   The analysis must be a DEBUG DUMP + `exit 0` first (print each result's
   free/last-use + the dependency graph), THEN decide the allocator.
3. **Per-device whole chain to the closure exit**: each device that holds active
   experts runs its OWN column mini-graph ALL THE WAY to the per-expert
   contribution / closure output - not "mm on GPU, then read back for the
   weightless tail on CPU". Each device computes to the final exit.
4. Buffer sizing + per-step input/output RELATIVE offsets are derived from the
   verify analysis (not guessed at exec time). Result: `moe_node_plan` with
   per-device arena size and each step's out_off / in(producer+offset).
5. Build-time: one topology TEMPLATE per layer (closure shape is static). At
   exec entry, INSTANTIATE it from the verify product + THIS run's pinned expert
   distribution (fill data pointers / ids / offsets, no structural rebuild).
6. Exec: submit other devices' instantiated mini graphs ASYNC, run the local
   path on the calling thread, then a CONVERGE point that forces a full wait
   (sync every device) and folds all contributions back out of the closure.

Rationale to keep: device mini graph computes the per-expert column path on the
device; only the (small) contributions / merge cross the boundary. Big
intermediates never leave the device arena. graph_compute returns only after the
layer converge sync (§3.4).

Build order note: the ANALYSIS (reverse last-use dump) is step 1 and is purely
observational - do it, dump, `exit 0`, and decide the allocator from real
dependency data before writing any executor code.

### 0.1 Landed layout + measured gap to ideal (2026-09-05, commit e6995dd)

Landed: `moe_chain_assign_backend` now computes a node-level interval layout for
every layer (always, not debug): per result take the LAST reader index
(last_use), then greedy first-fit in exec order - a node reuses a slot whose
occupant died (last_use < current index). Output `moe_layer_exec_t.out_off[i]`
+ `result_bytes` + `layout_ok`; exec `hide_burst` writes compute[i]'s output to
`out_off[i]` (per-node bump fallback when no layout). Verified: gemma + deepseek
run IDENTICAL to the CPU baseline; `STREAM_MOE_CAP_DUMP` prints the layout.

**Layout scales with ubatch automatically**: layout lives in `g_layer_exec`,
rebuilt by `moe_chain_assign_backend` on every graph REBUILD. llama's
`process_ubatch` calls `can_reuse(gparams)` - gparams carries the ubatch size,
so a changed ubatch fails reuse, rebuilds the graph, and re-runs assign -> the
layout is recomputed with the new tensor byte sizes. Same-shape decodes reuse
the graph (and the layout, which is then correct because the shape is
unchanged). Measured on a 12610-token prefill-from (-b 13000, -ub 2048, ctx
20000): result_bytes varied across four tiers ~766KB / 12MB / 62MB / 392MB,
tracking the batch the graph was built for. Slot TOPOLOGY (which nodes share a
slot) is dependency-only; only slot BYTE sizes scale with ubatch.

**Gap to ideal (measured, gemma 14 nodes/layer)**:
- layout = 191488 B/layer vs full-alloc reference 417312 B (saves 54%).
- 3 slots/layer: slot0 90112 B (gate_up 45K + down-mm 90K + weighted 90K),
  slot1 90112 B (geglu 22K + down_scaled 90K + 5 small ADD/scale nodes),
  slot2 11264 B (GET_ROWS 32B + 3 ADDs).
- Ideal (best-fit / interval-coloring minimizing total area) would put the ADD
  chain in ONE small slot (~11K) and the large mm outputs in the big slots,
  reaching ~130K/layer -> current is ~47% above ideal.
- Cause: greedy FIRST-FIT is legal (no live-range conflict) but not minimal -
  small ADDs get glued into a large slot because first-fit never considers
  "share with the existing small slot". The allocator is dependency-correct,
  not area-optimal. Node order (exec order) decides the fit.

**Ideal algorithm** (for later): interval scheduling / one-dimensional bin
packing over the result buffer viewed as a stack - assign by (produce,
last_use) and place nodes best-fit-decreasing (large blocks land first), i.e.
minimize peak simultaneous-live BYTES, equivalent to coloring the interval
graph by size class. Current first-fit is the conservative general-DAG
fallback; the ideal keeps the same dependency analysis but groups nodes by
size class before packing. Generality: first-fit handles any DAG shape safely
(gemma's fused gate_up + deepseek's split gate/up/down both verified); the
ideal packs tighter but needs a size-class grouping heuristic per arch. Buffer
budget note: arena is reused across layers (exec is one layer at a time), so
the cost is ONE layer's layout, scaled by ubatch - not layers x layout.

## 1. Goal

Execute a privatised MoE layer as **per-device expert-column chains**, in the
final shape from the start (no CPU-only transitional executor). Every device
that has active experts owns a **whole-column mini graph** for its experts; VRAM
devices submit it once asynchronously, the CPU pool computes its columns on the
main graph_compute thread with the existing per-node path, and only the exit
merge synchronises across devices. Timing per device is recorded at completion
for the future prefetch policy.

## 2. Execution unit: per-device expert-column mini graph

The privatised MoE chain is executed **per device**, one mini graph per device
that actually holds active experts:

- a device mini graph rebuilds the layer's column path for THAT device's
  experts: routed mm (gate_up / down) columns plus the per-expert
  view/silu/geglu/down/weighted operations that follow, with all live tensors
  (weights-view shells, intermediates, per-expert contribution outputs) placed
  in that device's own arena buffers;
- leaves are tensor shells: weights point at the resident pool buffer slot
  (fake-base offset), cur/ids point at a host-visible staging copy (uploaded
  per layer; dense is on CPU until dense offload lands);
- the device chain ends at the **per-expert contribution** (the last
  expert-independent operation before the first cross-expert op). Cross-expert
  summation stays outside the per-device graphs;
- **CPU pool columns** run on the main thread with the existing per-node path,
  restricted to the CPU pool's experts (column-slice semantics - the current
  executor is "whole-chain sole owner", which must become slice-aware once a
  layer's active set spans pools).

## 2.1 Layer-local convergence: the anonymous per-k adds are privatised

Audit fact (gemma4): the layer chain is delivered as one graph_compute **per
node** (each `n_nodes=1`), and the per-topk-k convergence of the weighted
views is a tree of **anonymous add nodes** (no `ffn_moe_` name) that llama's
graph leaves on its CPU default backend, reading our hidden intermediates by
their host pointer. That only works while hidden outputs live in host memory;
once weighted columns live in VRAM (vulkan shell data) those dense-side adds
would read a fake pointer. Decision: **extend privatisation to the anonymous
convergence adds** - a weightless op whose srcs trace (through views) to a
privatised intermediate is privatised too, so the whole per-layer convergence
runs inside our merge (exits via moe_out to the main dst as today). The
privatisable predicate becomes recursive/closure-based, consistent with the
verify chain BFS instead of the current name heuristic.

## 2.2 Execution trigger: whole-layer burst at the first split

Because the layer arrives as one split per node, the per-device column mini
graph has no whole-layer view at execution time. Decision: **cache the
privatised layer topology at build/verify** (the per-layer node sequence,
shapes and edges are already traversable on `gf`), then at execution the FIRST
privatised split of a layer triggers the whole layer: build each device's
column mini graph, async-submit VRAM devices, run CPU columns on the main
thread, merge at the layer tail into the main dst. Subsequent same-layer
privatised splits of that layer are served as no-op returns (their data was
already produced by the burst). The burst must run strictly after the layer's
dense-side inputs (cur/ids) are ready - they are produced before llama invokes
the first privatised split.

Column mapping (which device produced global column (k,t)) is recorded by
route B and consumed by the exit merge.

## 3. Parallelism and synchronisation (no extra worker threads)

Facts (verified): `ggml_backend_graph_compute_async` is a pass-through to
`iface.graph_compute` - async behaviour is backend-owned. Vulkan submits to the
device queue and returns; CPU computes synchronously (multi-threaded inside its
own graph call).

Execution model (final):

- VRAM device mini graph: `ggml_backend_graph_compute_async(vulkan, mini)` -
  submitted once, the GPU runs it without per-node sync;
- CPU pool columns: computed on the main graph_compute thread (no dedicated CPU
  worker needed - CPU work is on the caller thread anyway, so the GPU and CPU
  parts overlap naturally);
- `graph_compute` returns only after the exit merge: synchronise the VRAM
  device(s), read back the contribution columns, run the cross-expert add into
  the external dst. Async never leaks past the takeover boundary.

## 4. Per-device arena and staging

- VRAM arena buffers (vk buft) hold the mini graph's intermediates and
  contribution outputs; sized from the verify per-layer ping-pong budget. Two
  ping-pong regions per executing device is the theoretical minimum (in-place
  single-buffer writes are impossible in general - outputs differ in shape from
  inputs and readers lag writers by a step).
- Host-visible staging (vk host buft) receives cur + ids each layer; it must
  stay stable until the layer's async work completes (no in-layer reuse).
- VRAM arena buffers are host-mapped (host-map channel landed), so the exit
  merge can read contribution columns straight from the mapping while dense is
  on CPU.

## 4.1 Node plan: topology resolution decoupled from byte layout (2026-09)

The whole-layer closure (anonymous convergence adds AND moe_out included) executes
entirely on the device - reading per-expert contributions back to the host for the
merge is too expensive to run. Main-graph tensors are never mutated (a b4-2 crash:
llama's scheduler keeps bookkeeping the mutated buffers after the burst).

The builder is fed a **per-layer node plan** that separates *what the graph is*
from *where bytes go*:

```cpp
enum class moe_in_kind { k_chain, k_external };
struct moe_resolved_in {
    moe_in_kind   kind = moe_in_kind::k_external;
    int32_t       prod = -1;   // k_chain: producer index (compute order)
    int64_t       off  = 0;    // k_chain: cumulative view-chain byte offset
    ggml_tensor * src  = nullptr; // actual tensor for shape/type/nb
};
struct moe_node_plan {
    ggml_tensor * node = nullptr;
    size_t        out_off   = 0;  // arena region (allocator fills)
    size_t        out_bytes = 0;
    bool          is_out    = false;  // moe_out: exit readback to main dst
    moe_resolved_in in[GGML_MAX_SRC];
};
```

- Chain inputs are stored as **references** (`prod` + view offset), never absolute
  arena offsets - the builder derives `plan[prod].out_off + off` when it creates
  the arena leaf for an input, so a reuse analysis only ever rewrites the `out_off`
  array and the builder does not change.
- mm / moe_out are special-cased by op in the builder: mm src[0] is the weight
  shell (pool device buffer + branch offset, not the arena) and ids need slot
  localisation; moe_out owns an arena region plus the exit-readback flag.
- **Now**: `out_off` is filled by the landed node-level BEST-FIT-DECREASING
  layout (compute in `moe_chain_assign_backend`, diagnostics/layout_sim):
  one per-layer result block reused across layers, per-compute absolute
  offsets reaching the peak-simultaneous-live lower bound (gemma ~180KB/layer,
  deepseek ~197KB/layer @1 token; auto-scales with ubatch on graph rebuild).
- **Future**: this layout extends from "whole-layer one block" to per-device
  slices (each device's arena only holds the columns it computes); per-device
  ping-pong pairs are a special case of the interval layout. Builder consumes
  it unchanged.

## 5. Exit merge and scatter (generalised)

- The exit (cross-expert add into moe_out) reads every device's contribution
  columns and writes the external consumer's dst. Target buffer = wherever the
  external (dense) data lives - CPU main-graph dst today, VRAM once dense
  offloads. No host assumption.
- route B keeps the mapping device-pingpong-column -> main-graph global (k,t)
  column; the exit scatters/reads contributions at the correct global offsets,
  then the chain add node runs as today.
- VRAM contributions are read back via the host mapping (per-column copy) until
  dense moves to GPU.

## 6. Profile channel (feeds the future prefetch policy)

- Alloc-request ring (existing MPSC) stays single-purpose; load requests carry
  `total_tokens` + batch `start_rdtsc` (POD `slot_request_t` gains two fields).
- Profile ring (new, non-blocking): compute never stalls the scheduler's alloc
  processing on profile traffic.
- Per-backend aggregator in `moe_backend_ctx` records per device
  {n_expert, submit_rdtsc, done_rdtsc}; the layer exit pushes ONE profile event
  per device per layer. No per-node flood.
- Payload (POD): {layer, device_id, n_expert, delta_cycles,
  batch_total_tokens, batch_start_rdtsc}. IO events already carry req->done
  rdtsc. Policy consumers (future): per-layer lag + io lag -> prefetch depth;
  per-expert average cost (delta/n_expert) -> skew prefetch to idle devices.

## 7. Component build order (final shape, no transitional code)

1. **arena + shells**: per-device arena buffer (vk buft, verify budget) +
   host-visible staging; tensor-shell helper (fake-base offset, buffer binding).
2. **VRAM column-chain mini-graph builder**: rebuild the per-device column path
   in the arena (weight-view shells, uploaded cur/ids, arena intermediates and
   contribution outputs), one cgraph per device, async submit.
3. **CPU column-slice**: the CPU executor runs its own pool's columns on the
   existing per-node path (slice semantics; contribution out to the shared
   merge region).
4. **exit merge**: cross-expert add from every device's contribution columns
   (VRAM read back via host map) into the external dst, in main-graph column
   order; graph_compute synchronises at the end.
5. **numeric gate**: pure-CPU run (single slice) and all-VRAM run (single
   device) both IDENTICAL; mixed pools pass the same gate.
6. **profile plumbing**: ctx aggregator + profile ring + event struct + IO and
   device completion timestamps.

Design questions that come up during implementation are asked before coding.

## 7.1 Gap analysis vs current code -> actionable roadmap (2026-09-05)

Current code (landed): `exec_layer_burst` runs one layer synchronously on the
main thread, node by node. `MUL_MAT_ID` goes through `exec_mixed_mm`
(per-pool peel rounds - CPU or Vulkan, each round a SYNC graph_compute + read
back); weightless chain nodes run as manual 1-node CPU graphs. Result buffer =
the §4.1 best-fit layout as ONE whole-layer block. No async, no per-device
arena, no device-side whole chain.

Gap to the final design (per-device expert-column mini graphs, §1-§7). Items
below, roughly in dependency order:

**Analysis layer (verify/assign) - mostly landed, needs device-ization**
- [x] closure collection + internal dependency/last-use + layout
      (best-fit out_off / result_bytes) - done (e6995dd, 488930f).
- [ ] `moe_node_plan` (per-step in[prod+off] relative offsets) - layout covers
      per-node out offsets; inputs are still resolved ad-hoc in the executor.
- [ ] **per-device whole-chain execution plan**: which experts / contribution
      columns each device owns (the mix_plan pool partition), expressed per
      compute node (which slice of a node's output a device produces and which
      of its inputs it must have). Every device runs the FULL node chain over
      ITS OWN columns (whole chain to the contribution), not just the mm.

**Per-device arena (user decision 2026-09-05): NO per-device shrinking**
- [ ] each device that participates in a layer allocates a FULL whole-layer
      result block (result_bytes from the existing best-fit layout) - NOT a
      per-device column-sliced arena. Reason: keep layout identical across
      devices (same out_off[]), simple and uniform; per-device columns are a
      runtime partition that only affects which slice each device actually
      fills, never the block geometry. Buffer cost is one layer's best-fit
      block per participating device (small; arena reused across layers).
- [ ] bucket execution: within its block, a device computes only its own
      buckets (columns), leaving other slices untouched (see executor items).

**Executor resources**
- [ ] per-device ping-pong / event tracking (async GPU must not let the next
      layer overwrite in-flight results - §3 sync discipline).
- [ ] template instantiation at exec entry from verify product + THIS run's
      pinned expert distribution (fill data / ids / offsets, no rebuild).
- [ ] per-device column execution: a device walks its own slice of every
      compute node in the layer (same node chain, restricted to its columns),
      so the device reaches its contribution without host round-trips.

**Async execution skeleton**
- [ ] `exec_round_vk` -> async submit (`graph_compute_async`) + completion
      tracking instead of per-round sync + read back.
- [ ] CPU/VK overlap: submit device graphs async, continue CPU columns on the
      calling thread, converge at layer end.
- [ ] converge point: force-sync every device, read back contributions via host
      map, fold into moe_out (generalised exit merge/scatter, §5).
- [ ] each participating device holds its OWN full whole-layer result block
      (same §4.1 geometry, one block per device, not column-sliced).

**Verification gates**
- [ ] pure-device numeric gate once device execution lands (K6 shape; note GPU
      has no absolute fidelity vs CPU - validate structural equivalence, not
      byte identity, per BACKEND_DIVERGENCE_ANALYSIS.md §6).
- [ ] M8 UTs (layout self-check, device plan, merge) - test link fix is the
      blocker (stmoe_vk_* symbols need ggml-vulkan at link, B33).

**Profile (deferred, §6)**
- [ ] profile ring + per-device completion timestamps; slot_request_t already
      carries total_tokens / start_rdtsc fields.

Suggested next step: make each participating device run the whole node chain
over ITS OWN buckets (columns) into a full per-device result block (same
best-fit layout, one block per device). Build-time: express the device column
partition (which experts/columns per device, per node slice) so the executor
can instantiate the per-device whole-chain template; exec-time: async submit
per device, run the CPU bucket on the calling thread, converge at layer end.
Pure analysis-layer addition first (verifiable via CAP_DUMP/CSV + sim.js),
then the async skeleton on top.

## 7.2 Bucket-chain serialization + per-device output region (design agreed
##     2026-09-06, user)

Converged design for the multi-bucket concatenated mini graph (shape B). Three
decisions, all confirmed:

1. **A long graph IS serial** - if one device runs one graph whose nodes are
   executed in topological order on the same backend with NO cross-node
   parallelism, concatenating the buckets' chains into that one graph is a
   strict serialisation: bucket 2's chain starts only after bucket 1's chain
   ended. No extra submit per bucket is needed and no artificial edges must be
   forced - serialisation is simply what a single graph on one backend means.
   CAVEAT (why this holds): the executor does NOT rely on ggml's automatic
   buffer allocator. It is `no_alloc` + tensor shells with manually assigned
   `nd->data = arena_base + out_off` (this is what hide_burst already does on
   the CPU path). A ggml/vulkan auto-allocator would NOT see cross-bucket
   reuse (the two chains have no data dependency, so it treats them as
   parallel and gives each a disjoint buffer, needing the sum of all buckets).
   Manual `out_off` is what makes serial reuse of the intermediate region
   safe and free. This is device-agnostic: vulkan (and CUDA) derive a tensor's
   buffer offset as `(data - buffer_base) + view_offs` and never dereference
   `data` itself (`stmoe_vk_buffer_host_offset` returns `vk_ptr_base + off`;
   exec_round_vk already uses shells this way). So per-node
   `data = arena_base + out_off` works on vulkan/CUDA unchanged.
2. **Each bucket chain ends by ADDing its contribution into the device's OWN
   output region** (a reserved area of the same device arena block / the
   per-device accumulator). Once the bucket's tail add consumed its results,
   that bucket's intermediate region is dead and the next bucket chain reuses
   the SAME `out_off[]` byte regions. Final output = one read-back / cross-
   device merge of each device's output region at the layer end (SS5: only the
   contributions cross the device boundary, once, after every device finished).
3. **Per-device arena = a FULL whole-layer block** (result_bytes of the landed
   best-fit layout, same geometry as CPU, one per participating device). This
   is a worst-case upper bound: a single bucket chain's peak live bytes are
   <= the whole-layer peak, so any bucket packing inside is safe, and the
   geometry is byte-identical across devices (same `out_off[]`, no per-device
   offset rescaling). Simple and uniform; not column-sliced per device.

Downstream implications:
- moe_out / the anonymous per-topk convergence adds are NOT run inside any
  bucket chain. In the multi-bucket shape the fixed llama ADD tree (built for
  contiguous per-token k) does not match the split k columns, so the fold is a
  converge step that gathers each device's contribution columns by the
  original (t,k) mapping and adds into the external dst (SS2.1/SS5 - the
  privatised converge graph, phase 2). CPU phase 1 is safe precisely because
  scatter re-materialises full k rows in the host main dst, letting the
  existing captured anonymous ADD tree run unchanged.
- Device-computed bucket chains use REDUCED column shapes (mm output
  [d, w_b, n_active], not the full [d, n_k, n_t]); the full-width byte-level
  `out_off` from verify is therefore NOT directly the bucket nodes' byte
  layout. What carries over is the interval/slot STRUCTURE; per-device arena
  bytes scale linearly with the device's column span when reduced shapes run
  (deferred to the GPU bucket phase - CPU phase 1 stays full-width and only
  validates numerics).

## 7.3 CPU phase-1 prototype: whole-layer burst graph + per-device accumulator
##     (user decision 2026-09-06)

SUPERSEDES the CPU-phase-1 remark in SS7.2 ("scatter re-materialises full k
rows, let the captured anonymous ADD tree run unchanged"). The user wants the
FINAL shape exercised on CPU from the start: llama's graph is ALL no-op except
its entry and exit, and one burst graph per layer runs every bucket's chain
plus one dedicated accumulator that REPLACES the anonymous per-topk fold.

Anatomy of the anonymous fold (verified, llama-graph.cpp ~2289): it is a
SEQUENTIAL per-k accumulation, not a binary tree:
`moe_out = cur_experts[0]; for i = 1..n_k-1: moe_out += view_2d(experts, i)`.
So an accumulator that adds contribution columns row by row is the SAME shape
of computation - only the ORDER differs (bucket order vs k order), which lands
inside the accepted relaxed gate.

Exec-time flow (one layer burst):

1. Entry (first privatised split of the layer) triggers the burst:
   - pin_layer the layer's whole active expert set (existing M5 single-pass);
     afterwards every expert's owning pool/device is known.
   - Build the buckets from the (token, pool) hit-count peel; the plan is
     computed ONCE and reused across the layer's mm nodes (same ids).
   - Assemble ONE graph holding every bucket chain, serialised (manual
     `out_off` region reuse per SS7.2.1), plus one per-device ACCUMULATOR
     buffer `[d_out, n_t]`. Each bucket chain = compact-column
     gate/up mm -> clamp/glu -> down mm -> scale; its tail ADDs (by the
     bucket's (t,k) scatter map) the weighted contribution columns into the
     accumulator rows. Accumulation order = bucket order, NOT llama's k order
     -> ULP-level difference vs the baseline is EXPECTED and ACCEPTED: the
     numeric gate is relaxed (epsilon / structural equality), not bit
     IDENTICAL.
   - Run one CPU graph_compute.
2. Middle privatised splits of the same layer: no-op (already the case).
3. Exit (moe_out split / burst tail): memcpy the accumulator into the main-
   graph moe_out dst (llama owns that buffer; it is never hidden).

This burst graph is the CPU prototype of ONE device running ONE whole layer.
Once it matches, the GPU phase copies the same construction per device: the
device's bucket chains feed that device's OWN accumulator (= the SS7.2 output
region), and the exit becomes the converge step reading every device's
accumulator and merging. No second design pass.

Per-device resource lifecycle (arena + accumulator) - user confirmed, OK to
keep and reuse, no per-layer free:

- **Per-device arena** (verify layout `out_off[]` / `result_bytes`):
  process-lifetime, grow-only, reused across layers by offset reset
  (`reset_layer`) + overwrite - the same mode as today's `g_fullalloc_buf`.
  Reuse is safe ONLY because a layer ends with full sync (graph_compute
  returns after the converge). Release point = device teardown / process exit.
- **Per-device accumulator** `[d_out, n_t]`: process-lifetime, grow-only
  (d_out fixed; n_t grows with the largest ubatch seen). MUST be zeroed before
  each layer - accumulation is `dst[t] += col`, and a token row that no bucket
  touched this layer would otherwise keep stale upper-layer data. One
  `d_out x n_t` memset per layer, cheap.
- No per-layer free of either. Future async cross-layer overlap needs per-
  device ping-pong pairs (SS3/SS4 minimum) - M2-2 work, not now. Both are held
  per device (the existing `device_exec_ctx_t` / `g_dev_execs`).

Both resources are per-device: bucket-chain intermediates placed per the
verify layout inside the device's arena block; the accumulator is that
device's own output region.

**Relaxed-gate calibration (measured 2026-09-06, 129-token gemma L0 dumps,
temp/bucket_acc_calib.js)**: the accumulator is CONSTRUCTIVELY identical to
llama's linear per-k fold (k-order reference reproduces moe_out byte-for-byte,
0 diff). Reordering the fold changes only float summation order:
- `buckets 2+3+rest natural` (same-bucket k stays ascending) = 0 diff - real
  peel buckets that preserve in-bucket k order are EXACT;
- any cross-bucket reorder / shuffle: maxAbs <= 7.6e-6 (~1 f32 ulp of the
  value scale), cos = 1.000000000, ~55% of elements differ by exactly 1 ulp;
- per-token random shuffle worst case: maxAbs <= 3.8e-6.
Gate to use downstream: **maxAbs <= 1e-5 (f32 1-2 ulp), cos ~= 1.0**.
