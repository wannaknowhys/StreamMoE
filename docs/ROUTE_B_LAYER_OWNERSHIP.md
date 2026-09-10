# Route B Layer Ownership - whole-layer execution, device-local buffers, dynamic C1 migration

[English](ROUTE_B_LAYER_OWNERSHIP.md) | [简体中文](ROUTE_B_LAYER_OWNERSHIP.zh-CN.md)

> Status: **design, 2026-09-10**. Converged from the "cur copy / seam" discussion.
> Supersedes the per-GPU-MoE-backend sketch (that sketch fixes only static
> placement). Related: `ROUTE_B_GPU_PHASE.md` (whole-layer self-scheduling),
> `M2_DEVICE_EXECUTOR.md` (per-device executor + arena), `DENSE_PLACEMENT.md`
> (C1/C2 + Phase 2 migration), `EXPERT_MOVE_PIPELINE.md` (move machinery),
> `BUCKET_FAST_PATH.md`, `WORK_IN_PROGRESS.md` P (sync measurement),
> `VRAM_DMA_MOVE.md`.

## 1. Goal

Route B owns the **whole layer**: every compute node of a layer (C1 dense,
gating, MoE chain) runs through our backend, and the layer's activations live in
our own buffers placed on the layer's device. This is the target shape; it
delivers three things closure-only ownership cannot.

1. **No redundant seam copies.** cur and moe_out become internal to our graph and
   device-local; the scheduler never inserts a host copy and the executor never
   re-uploads. Same-device buckets read cur in place; only remote pools copy.
2. **One backend split per layer** (all layers owned -> one split for the whole
   graph): the per-layer cross-backend submit/sync disappears.
3. **Dynamic C1 migration.** Route B owns C1 placement, so C1 (and its KV) can
   move between devices at runtime and the layer graph is rebuilt to the new
   placement.

Out of scope: C2 placement policy (already expressible with
`--dense-placement`); writing our own dense kernels (standard ops are delegated
to the device backend).

## 2. Background: the three costs

### 2.1 Seam copies
The MoE backend advertises a host compute buft (`STREAMMOE_HOST`,
`moe_backend.cpp:264`). The scheduler inserts a copy only when the consuming
backend does not support the producer's buffer type (`ggml-backend.cpp:1026`,
copy condition `:1325`/`:1399`). So C1-produced cur is copied to host, then the
executor uploads it again to the device stage (`bucket_upload_leaf`,
`minigraph_exec.cpp:635`); moe_out crosses back to the dense device. Two
crossings per layer that carry no information.

### 2.2 Per-layer sync
Dense (llama backend) and MoE (our backend) alternate every layer; the scheduler
synchronises at every backend boundary. Measured `tail_sync 1.68 ms/layer` on
decode (`WORK_IN_PROGRESS.md` P2) - the fixed cost that makes the CPU path
faster than the device path.

### 2.3 Static C1
`dev_layer` is fixed at load (`llama-model.cpp:1492-1494`). C1 cannot move.
`DENSE_PLACEMENT.md` §5 Phase 2: dynamic migration requires route B to own the
C1 weights and execution.

## 3. Architecture

### 3.1 Whole-layer closure
Extend the privatised set from the MoE chain to **the whole layer**. Layer
attribution already exists (`moe_chain_layer_of_node`); the closure becomes all
layer-L compute nodes (dense, gating, MoE). The verify gate (Check 1) becomes:
no layer-L intermediate has a consumer outside layer L, except the layer output
consumed by the next layer / residual.

### 3.2 One backend, internal placement
The whole graph is owned by our backend, so the scheduler produces **one split**.
`graph_compute` receives the whole graph and runs it **layer by layer**,
internally placing each layer's activations on that layer's device. The
scheduler's allocation is reduced to the model input (embd, forced CPU) and the
model output (logits); every layer activation is ours.

This is why per-GPU MoE backends are **not** needed: layer activations no longer
depend on the scheduler's per-backend compute buft. (`supports_buft` accepting
all device bufts is still useful while layers are only partially owned.)

### 3.3 Dense delegation
We do not write dense kernels. Every dense node is cloned (fresh tensor, same op
/ op_params) and run on the layer's device backend; dense weights are referenced
in place (Phase 1) or come from the route-B C1 pool (migration phase). Attention
references llama's KV tensors in place. The MoE chain runs through the existing
bucket engine.

### 3.4 Device-local buffers and the fold
Each layer has a per-device arena (the existing verify interval layout, extended
from the MoE closure to the whole layer). cur is the layer input; moe_out is the
layer output and lives on the layer's device. The expert fold writes moe_out on
the device (the host `layer_fold`, `minigraph_exec.cpp:492`, is replaced by a
device-side reduction).

### 3.5 Cross-device transport
Experts are pinned per pool; `build_mix_plan` guarantees a bucket's experts are
on the bucket's pool. The only cross-device data is:

- **cur**: owner (layer device) -> remote pool. Full-width bucket -> whole copy;
  subset bucket -> index-gather.
- **acc_d**: remote pool -> owner, for the device-side fold.

Both are device transfers (D2D or the transfer queue), never host staging.

### 3.6 ids and the residual host round-trip
The scheduler (pin, `build_mix_plan`) runs on host and needs the current token's
ids, which are device-computed. This is the one unavoidable per-layer host
round-trip (the ids tensor is tiny; the cost is the wait). Levers:

- **prefetch / pipeline**: use historical routing to load experts ahead, so the
  current ids are not on the critical path (only misses stall);
- **all-resident layer**: if a layer's needed expert set is already resident,
  bypass the bucket engine and run the standard on-device `MUL_MAT_ID` with ids
  as a device tensor - zero host round-trip for that layer.

### 3.7 C1 vs C2
C1 is the layer's dense; its device is the layer's device and defines where
cur/moe_out live. C2 (`token_embd` / `output` / `output_norm`) sits outside the
layers and does not touch the seam; it stays a placement policy
(`--dense-placement C2:<dev>`), placed on a device only when it is a net win.

### 3.8 Dynamic C1 migration
Route B owns the C1 weights in a managed pool (reuse the
`EXPERT_MOVE_PIPELINE` machinery: move worker, hysteresis, copy-then-release).
On a migration decision, C1 weights and the layer's KV move together, and the
layer graph is rebuilt for the new placement. Because we own the layer's
execution and buffers, this is a placement parameter, not a llama structural
change.

### 3.9 ids join: static prefix / dynamic suffix

Whole-layer ownership does **not** remove the data dependency
`gate -> ids -> expert load / slot map`. Every layer is therefore two parts
joined at ids:

- **static prefix** (buildable ahead, no ids): dense prefix (attention / norm) +
  gating up to `topk` (which produces ids). Dispatched by C1's device.
- **ids join**: the routing ids are device-computed and must reach the host,
  because the host owns (a) the load decision (which experts to pin) and (b) the
  expert-id -> pool-slot map (`pin_slot`, the `w3d` slot axis). This is the one
  residual per-layer host round-trip; the ids tensor is tiny, the cost is the
  wait.
- **dynamic suffix** (needs ids): `weights = get_rows(probs, ids)`, the expert
  mm, the fold, and the tail (residual / next-layer prefix). Rebuilt per ids.

Why the host round-trip is forced: route B repacks experts into a bounded slot
pool with partial residency, so ids cannot be consumed by a device-side
`MUL_MAT_ID` alone - the slot mapping and the load decision are host-side. The
only escapes are all-resident experts in original GGUF order (impractical for a
bounded pool) or prefetching so the load decision is off the critical path. A
GPU has no "look up ids, page in on miss" mechanism; residency is host/DMA
managed.

Consequences:

- "whole-layer graph" = **static prefix graph + runtime suffix graph**, not one
  graph; the ids D2H is the join.
- single device: the suffix can absorb the tail (and even the next layer's
  prefix) into one submission, so the only sync is the ids wait (1 per layer).
- multi-device: the remote pools' expert subgraphs + fold add one converge
  (1 ids + 1 converge per layer).

The expert-closure analysis is still needed after takeover, but its role changes
from "privatisation boundary + external-consumer check" to **intra-layer
execution partition** (expert-domain -> bucket engine, dense-domain -> delegated
to the device backend); the external-consumer check simplifies to "no layer-L
intermediate escapes layer L except the layer output".

## 4. Execution flow (per token)

1. `graph_compute` is called once for the whole graph.
2. For each layer L, split at the ids join:
   - **static prefix**: run the dense prefix (attention / norm) + gating up to
     `topk` on L's device (clone + delegate);
   - **ids join**: D2H ids; pin/resolve experts; build the mix plan;
   - **dynamic suffix**: build and submit the suffix (routing weights, expert
     mm, fold, tail) on L's device; per-pool expert graphs are submitted async
     and the CPU pool graph runs on the calling thread; converge, fold `acc_d`
     on L's device, write moe_out, then run the tail.
3. Output logits to the model output.

## 5. Milestones

- **L1 - whole-layer capture + verify.** Extend closure capture/verify to the
  whole layer; dump under a debug flag. No execution change.
- **L2 - dense delegation.** Layer executor runs the dense head + tail (clone +
  delegate) around the existing MoE burst; ids D2H. Gate: backend-gate numerics.
- **L3 - device-local buffers + fold.** Layer activations in our arena on the
  layer's device; device-side fold; cross-device cur/acc transport.
- **L4 - dynamic C1 migration.** C1 weight pool + move + KV follow + rebuild.
- **L5 - ids pipeline.** Prefetch from history; all-resident bypass.

## 6. Open questions

1. Dense weights: keep llama-loaded and reference in place until L4, or move to
   the route-B pool from L2?
2. KV: reference llama's KV in place; confirm the move path for L4.
3. Dense subgraph: clone node-by-node and delegate (reuses the existing clone
   machinery), or hand the whole dense subgraph to the device backend in one
   call?
4. Claim the whole graph at once, or layer-by-layer during the L2 transition?

## 7. Validation gates

- Pure-RAM path stays IDENTICAL (no behavior change until L2 lands).
- After L2: same flavor, relaxed backend gate (cos ~0.999x) vs the current split
  path; `SM_COPY_TMR` / `[seam]` shows no cur/moe_out copy.
- Device graph node counts and per-layer timing via the existing TMR hooks.

## 8. Risks

- Owning attention means owning KV ordering; a mistake is a wrong number, not a
  crash. Keep the pure-RAM gate as the anchor.
- Cloning the whole layer multiplies the graph-build surface; verify Check 2
  (chain integrity) must extend to the whole layer.
- Buffer sizing: the whole-layer interval layout must be recomputed (it currently
  covers the MoE closure only).
