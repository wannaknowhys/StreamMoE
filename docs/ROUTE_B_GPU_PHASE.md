[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B Multi-Device Expert Pool - Design (Phase B)

> Status: design, **revised 2026-09** (supersedes the 2026-08-29 draft - the old
> "cb_eval pin/unpin + per-node hijack + HOST_VISIBLE placement" model is replaced
> by the whole-layer self-scheduling mainline below). Builds on route B third
> path (official `ggml_mul_mat_id` kernel + uniform-stride slot pool,
> `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7).
> Verified facts carried over: Vulkan and CUDA natively support `MUL_MAT_ID`
> with the pool-slot 3D layout; RX 590 exposes a real 8 GiB
> `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` heap; GGML_VULKAN=ON builds clean.
> Related: `docs/MULTI_MODEL_POOL.md`, `docs/MULTI_SUBPOOL.md`, `docs/Backend.md`.

## 1. Goal and mainline

Selected experts live across multiple device pools (CPU RAM + GPU VRAM), each
bounded by its pool budget; GPU is a first-class compute node (not storage).
MoE execution is **not** per-node on the llama graph: the stream_moe backend
takes a **whole MoE layer** in `graph_compute` and **self-schedules** it - read
ids, resolve each expert's pool, dispatch per-device subsets, run each subset's
expert chain inside that device, merge results, write the layer output back.

Confirmed mainline decision (2026-09): **compute takes the whole layer and
self-schedules to each device** - this is required, not optional.

## 2. Current mechanisms (what stays, what extends)

- **Why a backend**: registering a ggml backend/device gives us the scheduler's
  formal `graph_compute` callback. `supports_op` claims only `*_exps`
  `MUL_MAT_ID` (+ views whose source is our host compute buft); the scheduler
  hands us exactly those nodes, once each, correctly ordered. Replacing this
  with graph surgery is not planned.
- **mini-graph today**: each original `MUL_MAT_ID` is re-executed in a private
  leaf-only mini-graph with the **official kernel**: `w3d` leaf fakes the pool's
  compact slots as `[ne0, ne1, group_slots]` (data = pool base + branch offset,
  nb[2] = slot stride), `ids_slot` translates expert ids -> slot index, `b_leaf`
  wraps the main-graph activation, result writes the main dst. CPU-only, zero
  copy (all one RAM).
- **Notify protocol**: compute -> scheduler = MPSC push + sleep on a per-(L,E)
  version word; scheduler -> compute = version bump + `WakeByAddressAll`/futex
  wake (directory `set`/`clear`, `slot_meta` state machine). This protocol is
  **device-agnostic and stays** - compute waits for "(L,E) available", wakes and
  rescans all pools.

The gap to self-scheduling is not "collect more nodes": it is **execution of the
in-layer activation chain** (gate_up -> view gate/up -> silu -> weighting -> down)
which today runs on llama buffers between nodes. Until we own that chain,
per-device dispatch would still copy big intermediates between backends.

## 3. Target model: whole-layer self-scheduling

The llama main graph still builds every MoE layer node (`can_reuse`/plan
stable). `supports_op` extends to the layer's whole MoE domain
(`blk.N.ffn_moe_*` names: gate_up / silu / down / views / adds / mul). At
execution, `graph_compute` receives one layer's MoE split and runs:

1. read ids (`ffn_moe_topk-N`; the gate softmax/topk stays on the graph / dense
   side - ids arrive computed)
2. resolve every (L,E) to its pool slot (directory scan, pin/wait as today)
3. choose the **lead device** for the layer (see §5)
4. dispatch: per non-lead device a subset (ids slice + weights slice)
5. per device: run its expert chain in the device's fixed arena (official
   `mul_mat_id` + official silu/mul/add on that backend); big intermediates never
   leave the arena
6. merge expert contributions (embedding-dim, small) back to the lead device,
   write main `ffn_moe_out-N` dst

Dynamic (which experts, which device, lead choice) lives entirely in the
execution layer; the graph/reuse shape is static. This is the core architectural
win of self-scheduling.

## 4. device_pool[] + fixed per-device execution arena

Current single-pool code already keeps a pool dimension in the directory
(`entries` is (L,E) x pool, `scan` iterates pools) - keep that. What changes is
the **physical slot addressing** (today one base carved into layer groups):

```
device_pool[] = {
  ggml_backend handle,
  physical base (VRAM or RAM), slot stride, capacity,
  per-pool layer-group carve (subpools_),
  per-pool slot meta (state/refcount/generation) or global index with pool dim,
  fixed execution arena,
}
```

**Fixed arena per device** (the point of "one reusable result region"): one
preallocated block sized for the worst-case layer the device will process:

```
arena = [ staging(cur broadcast copy) | exec region (chain intermediates
          gate_up/silu/down overwrite in place) | result (expert contributions) ]
```

Value is **not** saving allocation (llama gallocr + our scratch arena already
reuse) - it is making per-device expert chains run with intermediates never
leaving the block, so cross-device traffic is only the small items (§5). Watch:
in-place overwrite ordering (read-before-overwrite per chain step; layers are
serial so safe, but explicit read/write windows needed).

## 5. Computation placement rules (data flow is minimal by construction)

- **Gate** (`cur x gate_proj`, softmax, topk) is a **dense** computation ->
  runs on the layer's **dense device** (static, decided by layer placement -
  dense weights are never moved/copied for gating). ids + per-expert weights
  then fan out to the executing pools (small: n_used x tokens).
- **Lead device** (merge point) = the pool holding the **most resident experts
  of this layer** (heuristic from the directory - pool residency changes slowly;
  we cannot use this run's ids to choose, they do not exist before gating). Most
  expert contributions are summed locally at the lead.
- **cur broadcast**: every pool that computes experts needs cur - one copy per
  executing pool (unavoidable minimum).
- **Merge**: lead device sums its own + returning non-lead contributions
  (embedding-dim, small) -> `ffn_moe_out-N` -> back to the dense device for
  residual/add/norm.

Big intermediates (gate_up / silu / down outputs) never cross devices; only ids /
weights / cur-broadcast / expert-contributions / moe_out move - all small.

## 6. dense / KV per-layer assignment (independent follow-up)

Decouple dense weights per layer and assign a device per layer (static, no
runtime migration), KV follows the layer's dense device. Example: put the last N
layers' dense + KV on the RX 590. Value: dense device = gate device = ideal
lead device for that layer -> zero cross-device for those layers.

Two notes:
- llama.cpp's native layer offload is **contiguous** (`-ngl N`, split-mode
  layer) - an arbitrary per-layer device table ("layer 27-29 on GPU") is a
  loader/`dev_layer` change, an independent work item.
- Only the expert pool is dynamic/migratable; dense + KV are fixed at
  load/init. Size the VRAM budget (dense/KV per layer) before doing this.

Not blocking the mainline - do M1/M2 first, exercise native contiguous offload
to learn the interactions, then decide whether a custom layer table earns its
keep.

## 7. Numerical / correctness baseline

- Per-expert results use official kernels + same weights -> identical; the
  expert-add order may differ from the official graph -> ulp level. Gate on
  v2/deepseek alignment (route IDENTICAL + cos ~ ulp) - the toolchain exists.
- Pool size must never change numerics: different `--moe-ram-pool` sizes must
  produce IDENTICAL exports (already verified 8192 vs 71680) - keep as a
  regression guard across the GPU work.
- The 2026-09 divergence analysis (docs/BACKEND_DIVERGENCE_ANALYSIS.md): any
  two backends show ~5% routing flips + hidden amplification - it is inherent,
  not a bug; GPU phases will show the same class of noise.

## 8. Milestones (risk-decreasing)

- **M1** - CPU-only self-scheduling skeleton: two "fake device_pools" (both CPU,
  independent pools + arenas) to pin down ids-grouping / dispatch / per-pool
  subset chain / merge logic and its numerics == official (v2 alignment). No
  real GPU yet; this defines the self-scheduler interface.
- **M2** - device_pool[] abstraction lands (backend handle / slot addressing /
  fixed arena); one real Vulkan device + CPU mixed (some layers on GPU via
  native offload), exercise gate-follows-dense + lead-device merge live.
- **M3** - same-layer cross-device dispersion (2 experts GPU0 / 2 GPU1 / 3 CPU
  style) + arena read/write-window discipline at full speed.
- **M4** - device migration (directory set/clear already supports the protocol).

## 9. Kept verified facts (build / hardware)

- Vulkan + CUDA native `MUL_MAT_ID` with the slot-3D layout (src0 ne[2]=n_slots,
  nb[2]=slot stride) - the pool-slot w3d trick is backend-portable.
- RX 590 8 GiB BAR1 is a real `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` heap -
  zero-copy DIO into mapped VRAM remains available for expert loading once a
  device pool exists (fallback: RAM staging + copy).
- `GGML_VULKAN=ON` builds clean (vendored CMake hook forwards toolchain);
  `libomp.dll` must sit next to the exe. Realistic target stays: dense on
  Vulkan + hot experts on GPU + most experts in RAM.
