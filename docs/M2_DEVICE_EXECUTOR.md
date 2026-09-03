# M2 Device Executor - per-device expert-column chains

[English](M2_DEVICE_EXECUTOR.md) | [简体中文](M2_DEVICE_EXECUTOR.zh-CN.md)

> Status: design, 2026-09 (revised: no transitional code - implement the final
> shape directly, component by component). Builds on `docs/ROUTE_B_GPU_PHASE.md`
> (whole-layer self-scheduling, §1/§3) and the landed VRAM data layer
> (docs/WORK_IN_PROGRESS.md J: host-map channel, experts resident on device,
> demote-to-RAM).

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
