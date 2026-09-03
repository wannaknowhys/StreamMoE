# M2 Device Executor - per-device expert-column chains

[English](M2_DEVICE_EXECUTOR.md) | [简体中文](M2_DEVICE_EXECUTOR.zh-CN.md)

> Status: design, 2026-09. Builds on `docs/ROUTE_B_GPU_PHASE.md` (whole-layer
> self-scheduling, §1/§3) and the VRAM data layer already landed
> (docs/WORK_IN_PROGRESS.md J: host-map channel, experts resident on device,
> demote-to-RAM). This document fixes the M2 executor shape and the profile
> channel that feeds the future prefetch policy.

## 1. Goal

Execute a privatised MoE layer by **expert column** across real devices: the
active experts of each MUL_MAT_ID are split by their resident pool, every device
computes its own experts' **full column chain** locally (mm -> silu/geglu ->
down mm -> weighted contribution), intermediates never leave the device arena,
results stay in that device's ping-pong buffers, and only the **exit node**
(moe_out) gathers the per-device columns into the external consumer's buffer.
Meanwhile the execution produces per-device, per-layer timing that a future
prefetch policy consumes.

## 2. Execution unit: device-local expert-column chain

Today (`minigraph_exec.cpp`) every privatised compute node runs one-at-a-time as
a CPU single-node mini-graph; hidden outputs land in two global host ping-pong
buffers. Under M2 the active set of a MUL_MAT_ID is partitioned by the resident
**pool/device** of each expert:

- one sub-compute per device = the device's experts x the *whole column path*
  that touches them (the routed mm columns plus the per-expert view/silu/geglu
  and down-multiply that follow inside the layer chain);
- each sub-compute runs **inside that device** (its ggml backend, its arena);
- the sub-results stay in the device's own ping-pong buffers;
- column mapping (which device produced global column (k,t)) is recorded by
  route B; only the exit node scatters/merges (§5).

The gating segment stays dense-side (unchanged). ids fan out per device; the
activation broadcast `cur` is one copy per device.

## 3. Parallelism and the graph_compute contract

Facts verified in ggml:

- `ggml_backend_graph_compute_async` is a straight pass-through to
  `backend->iface.graph_compute` (ggml-backend.cpp:450) - there is **no generic
  async layer**. Whether it returns before finishing is purely backend-owned.
- GPU backends (Vulkan/CUDA/Metal): submit to the device queue and return -
  genuinely async; `ggml_backend_synchronize` / events wait.
- CPU backend: its `graph_compute` multi-threads internally and returns when
  done - synchronous, it has no device queue.

Consequence: cross-backend true parallelism is not free. The device sub-computes
are submitted with the unified primitive `ggml_backend_graph_compute_async`;
Vulkan is natively async, and the CPU-pool sub-compute needs **one route-B
worker thread** to overlap with Vulkan. Per-device completion is awaited by
`ggml_backend_synchronize` (Vulkan) and the CPU worker join/sync.

**Contract (unchanged, required)**: `graph_compute` returns only after every
device sub-compute finished and the exit merge wrote the external dst (§4 of
ROUTE_B_GPU_PHASE). No asynchronous leak past the takeover boundary.

## 4. Per-device arena and ping-pong buffers

- verify (`route_b_chain.cpp`, moe_chain_verify_graph) already derives the
  per-layer odd/even ping-pong **budget** (largest compute-node output on each
  parity). Runtime keeps global host buffers today; under M2 each device gets
  its **own** odd/even pair sized to what that device's dispatched chain needs
  (a device arena: staging cur copy | exec region | result columns).
- In-place single-buffer writes are **not** possible in general: chain compute
  outputs differ in shape from their inputs (mul_mat_id dst != cur), and readers
  lag writers by a step. With the long-range-dependency check already enforcing
  liveness <= 1 compute step, **two ping-pong regions are the theoretical
  minimum** per executing device.
- VRAM arena buffers are host-mapped (host-map channel landed); while dense is
  on CPU the exit scatter can read device results straight from the mapping.
  Once dense moves to GPU the same scatter targets VRAM (§5).

## 5. Output scatter, generalised

The exit node (moe_out) is never hidden - it must produce a real value for the
dense side. Generalisation: **scatter to wherever the external consumer's buffer
lives**, never assume host:

- route B keeps the mapping device-pingpong-column -> main-graph global (k,t)
  column (the per-device sub-result local column order differs from the main
  ids order);
- at the exit, each device's result columns are copied (scattered) into the
  exit node's input region at the correct global offsets, then the exit chain
  node runs as today (sum + write main dst);
- today dense is on CPU -> the target is the CPU main-graph dst; when dense
  moves to GPU -> the target is that VRAM buffer. The scatter is parameterised
  by the external data location, not by a host assumption.
- host phase: per-column memcpy (all pools host-readable). Vulkan phase: real
  cross-device column transfers.

## 6. Profile channel (feeds the future prefetch policy)

Two rings instead of one tagged union:

- **alloc-request ring** (existing MPSC) stays single-purpose; load requests
  carry `total_tokens` + a batch `start_rdtsc` (extend the POD
  `slot_request_t` - it is an atomic-stored struct, adding two fields keeps
  that property).
- **profile ring** (new, non-blocking low-water push): compute threads must
  never stall the scheduler's alloc processing on profile traffic.

Aggregation: a **per-backend aggregator in `moe_backend_ctx`** (not globals)
keeps, per device, `{n_expert_this_layer, accumulated delta_rdtsc}`; the layer
exit node pushes ONE profile event per device per layer. No per-node flood.

Profile event payload (POD): `{ layer, device_id, n_expert, delta_cycles,
batch_total_tokens, batch_start_rdtsc }`. IO completion events already carry
req->done rdtsc (async_load fields). Policy consumers (future):
- per-layer compute lag (steady-state, aligned by token progress via
  total_tokens) + io lag -> how many experts to prefetch;
- per-device per-expert average cost (delta / n_expert, so a busy device is not
  mistaken for idle) -> skew prefetch to the idle device.

The policy itself is deferred (scheduler strategy rework).

## 7. Migration steps

1. **M2-1** all-experts-on-vram, single device: real Vulkan column-chain
   execution (device arena/ping-pong in VRAM, mini-leaf graphs submitted to the
   Vulkan backend), IDENTICAL to the vk baseline. Introduces one new variable
   set at a time (Vulkan execution) on top of the data layer.
2. **M2-2** two channels run truly in parallel: CPU-pool worker + Vulkan async,
   sub-computes split by pool; graph_compute finishes with full
   synchronisation. IDENTICAL.
3. **M2-3** exit scatter generalisation: write back to the external data
   location (dense position becomes a parameter), column map + scatter. Mixed
   RAM/VRAM active sets (J6) fall out of this framework naturally.
4. **M2-4** profile plumbing: ctx aggregator, profile ring, event struct, IO +
   device completion timestamps. Independent of 1-3; can land first if needed.

Design questions that come up during implementation are asked before coding.
