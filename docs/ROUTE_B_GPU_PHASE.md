[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B GPU Mixed Pool - Design (Phase B)

> Status: design (2026-08-29). Builds on the route B third path (official
> `ggml_mul_mat_id` kernel + uniform-stride slot pool, `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7).
> Verified against vendored llama.cpp @ 5ab785cf8: Vulkan and CUDA both natively
> support `MUL_MAT_ID` with the pool-slot 3D layout.
> Related: `docs/MULTI_MODEL_POOL.md` (per-model pools), `docs/MULTI_SUBPOOL.md` (per-expert-kind subpools), `docs/Backend.md` §1 (CPU/GPU dual-pool concurrency).

## Goal

Extend route B so selected experts can live in a GPU pool (VRAM) while the rest
stay in the CPU pool, with physical memory still bounded by pool budgets
(`--moe-ram-pool` / `--moe-vram-pool`). Model-agnostic, no change to llama.cpp
math / graph topology / scheduling.

## Architecture: three pillars, each owns one concern

| Pillar | Owner | Responsibility |
| :--- | :--- | :--- |
| Compute hijack | `stream_moe` backend | Every `MUL_MAT_ID` that consumes ids is hijacked - src0 in-place re-pointed at a pool slot layout, executed by the official kernel. |
| pin/unpin lifecycle | `cb_eval` hooks | pin where ids are produced, unpin where experts are consumed (merge node). Decoupled from computation. |
| Async load engine | scheduler | IOCP (Win) / io_uring (Linux) / io_submit fallback - concurrent in-flight reads, wait-on-address wake. |

## 1. Compute hijack (in-place pool-slot repoint)

The MoE nodes stay in the main graph (`build_moe_ffn`'s `ggml_mul_mat_id`). The
backend's `graph_compute` collects them and executes the official kernel over the
pool-slot layout:

- src0 in-place: `ne[2]: n_expert -> n_slots`, `nb[2]: expert stride -> slot stride`,
  `data: dummy -> pool_base + branch_off` (`ne[0]/ne[1]` unchanged - per-expert
  weight shape is identical; the official kernel indexes `e*nb[2]`, consuming slot ids).
- ids translated expert id -> slot index (mapping fixed at pin time).
- Run `ggml_backend_graph_compute(backend, subgraph)` with the official kernel.

CPU-only today: one subgraph, delegate `cpu_backend`. GPU mixed: split ids by the
expert's pool into `cpu_ids` / `gpu_ids` -> **one subgraph per pool** (a single
src0 cannot describe two pools) - CPU subgraph to `cpu_backend`, GPU subgraph to
`vulkan_backend`, dst merged by token/expert ownership. The two run concurrently.

Key facts (verified in vendored llama.cpp @ 5ab785cf8):

- Vulkan: `supports_op` handles `MUL_MAT_ID` (`ggml-vulkan.cpp:18120`); AMD vendor
  branch enables `mul_mat_id_s/m` (small/medium warp tile, `mul_mat_id_l=false`);
  src0 types include Q4_K / Q5_1 / Q8_0 (all of gemma's heterogeneous types).
- CUDA: native `MUL_MAT_ID` support (`ggml-cuda.cu:2252` supports_op + dedicated
  `mul_mat_id` / fused up-gate kernels at `:1684`).
- Caveat: a backend's `supports_buft` rejects the stream_moe pool buft, so the GPU
  subgraph's src0 must be a **backend-owned buffer (the GPU pool)**, not the pool
  buft. GPU pool = a Vulkan buffer (DEVICE_LOCAL or HOST_VISIBLE) into which slot
  weights are loaded (DIO -> host -> device, or HOST_VISIBLE direct write).

## 2. pin/unpin lifecycle (cb_eval)

pin needs the ids *values*, which are runtime tensors - so "pin where ids are
produced" means "pin at the graph execution point where ids are ready":

- **Hash layers** (`il < dsv4_hash_layer_count`, `deepseek4.cpp:1279` - deterministic
  `get_rows(ffn_gate_tid2eid, tokens)`): ids are known at **build time** (tokens
  known) -> pin asynchronously during graph build (free perfect prefetch). Note:
  hash is the *shallow* layers, argsort the *deep* layers.
- **Argsort layers**: `cb_eval` has no post-execution callback, so pin lands on
  the **first `MUL_MAT_ID` of the layer** (ids ready by topo order): read `src[2]`,
  pin misses asynchronously, wait ready (wake-on-ready, not busy-wait), then
  release the node.
- **unpin** on the **merge node** (`mul(expert_out, weights)` / `add`): these are
  CPU elementwise nodes that consume the down output - sched's boundary event
  synchronization guarantees the GPU subgraph is done before they run, so unpin
  here is the authoritative GPU-completion point (rides sched's event sync, no
  extra `backend_synchronize`). Late unpin is safe (only holds a slot a bit longer).

`cb_eval` is a per-node callback that forces synchronization - fine for CPU-only,
but it would eat GPU async gains in the mixed phase. For the GPU phase, pin/unpin
may move back inside `graph_compute` (which naturally has per-split before/after
positions and couples with backend event sync), keeping the async load engine.

## 3. Async load engine (scheduler)

pin = submit an async read (scheduler MPSC queue) and return immediately - the
compute thread is never blocked on IO. scheduler runs IOCP (Win) / io_uring
(Linux) / io_submit fallback with concurrent in-flight reads (QD sweep: 1/4/16/64).
IO completion writes the slot READY (memory_order_release) + WakeByAddressAll /
futex_wake; compute threads wait on `slot_meta[slot]` (state+refcount+generation
64-bit word) with wait-on-address (acquire), never busy-wait. GPU slots add a
host->device copy after IO (or direct write into HOST_VISIBLE VRAM, ReBAR in
Phase C).

## 4. Multi-GPU

`ggml_backend` is natively multi-device (sched `n_backends`, vulkan enumerates
`physical_devices.size()`). Pools gain a device dimension: **one pool instance
per model x per device**. Pool selection (at pin time) picks which GPU (free /
hotness / layer preference). Each GPU has its own vulkan backend; MoE subgraphs
are delegated to the backend of the experts' assigned GPU. The dense `-ngl`
split mechanism is untouched.

## 5. Generic ggml backend abstraction

The whole scheme runs through `ggml_backend_graph_compute(backend, subgraph)` -
the GPU phase only swaps `cpu_backend` for `vulkan_backend`. No shader / kernel /
backend-specific memory code is written: the pool (slots / DIO / pin / translate /
pool selection) is backend-agnostic and lives in the stream_moe layer.

Performance notes:

- **Weight transfer is the main cost**: GPU slot weights come from DIO reads and
  must be copied into the vulkan buffer. Mitigate with HOST_VISIBLE buffers (AMD
  direct write) and EST1 hotness keeping frequently-used experts resident in VRAM;
  ReBAR (HOST_VISIBLE|DEVICE_LOCAL zero-copy) is Phase C.
- **Per-layer delegation overhead**: one vulkan `graph_compute` per MoE subgraph
  (command submission / descriptor binding). llama.cpp's vulkan backend has
  pipeline caching; for compute-bound large GEMMs the GPU gain dominates.
- Trade-off: no fused/custom shaders (e.g. swiglu fused into GEMM) - acceptable,
  matches the "no llama math changes / generic backend" stance.

## 6. Heterogeneous experts

`docs/MULTI_SUBPOOL.md` grouping (per-expert-kind subpools, budget by byte
fraction) still applies. Two GPU additions:

- The GPU pool is also grouped; pool selection (at pin) must consider "group
  quantization x target backend capability" (a group may only fit CPU if vulkan
  does not support its type).
- src0 `nb[2]` (slot stride) resolves per layer/group (existing
  `branch_layout(layer, ...)`); CPU/GPU slots each aligned per group/backend.

## 7. Numerical equivalence

GPU kernels have different float accumulation paths - expect ulp-level
differences (same class as the repack-vs-plain divergence). Regression must cover
three distributions per `docs/Backend.md` go-live gate: all-CPU / all-GPU / mixed.

## 8. Prerequisites (build)

- Vulkan SDK (glslc shader compiler, vulkan.hpp, vulkan-1.lib) - required at build
  time; build.bat already forwards `GGML_VULKAN=ON` (default OFF). Runtime needs
  only the GPU driver (vulkan-1.dll is present at 1.3.250.0).
- RX 590 8GB is Vulkan-only; VRAM pool budget is therefore small - realistic
  target is dense on Vulkan + small experts on GPU + most experts in RAM.
