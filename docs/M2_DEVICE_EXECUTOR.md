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
- **Now**: `out_off` is a fixed full-alloc bump over the compute sequence (each
  node its own region; the layer's whole arena+stage is reserved once up front -
  growing the arena mid-layer frees already-hidden outputs, b4-2 lesson).
- **Future**: verify runs a live-range / interval reuse analysis and fills the
  same `out_off` array (per-device ping-pong pairs are a special case of the
  interval layout). Builder consumes it unchanged.

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
