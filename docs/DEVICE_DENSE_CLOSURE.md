# Device-Dense Whole-Layer Closure (C1 / MoE / C2 on their own devices)

[English](DEVICE_DENSE_CLOSURE.md) | [简体中文](DEVICE_DENSE_CLOSURE.zh-CN.md)

> Status: **design + implementation in progress, 2026-09-15**. Extends whole-layer
> ownership (`ROUTE_B_LAYER_OWNERSHIP.md`, `LAYER_EXECUTOR_DESIGN.md`) to layers
> whose dense weights live on a **device** (`--dense-placement C1:<dev>`), and
> makes every cross-device data movement explicit. Related:
> `PER_DEVICE_ARENA.md` (per-device arena), `M2_DEVICE_EXECUTOR.md` (device
> executor), `DENSE_PLACEMENT.md` (C1/C2), `GRAPH_PARTITION.md` (iron rule).

## 1. Goal

Whole-layer ownership currently captures a layer only when its dense weights are
**host-resident** (`moe_chain_assign_backend`). With `--dense-placement
C1:Vulkan0` the layer is not captured, so dense runs through llama's scheduler and
the MoE closure through route B - two execution paths. The goal is to capture
device-dense layers too, so the executor runs the **dense head (C1), the MoE
closure, and the dense tail (C2) each on its placement device**, with all
cross-device movement explicit and in-graph (GPU-equivalent, no host round-trips
in the middle).

## 2. Defects found when relaxing the filter

Relaxing the capture filter exposed a chain of issues (all fixed or being fixed):

1. **Scheduler dups device weights to our host buft.** Because
   `moe_dev_supports_buft` rejected device bufts, `ggml_backend_sched_split_graph`
   inserted `%s#%s#%d` copies (`ggml-backend.cpp:1405`) of every device weight
   onto `STREAMMOE_HOST`; the device subgraph then received a host operand
   (invalid Vulkan subbuffer).
2. **KV cache on the wrong device.** `llama_model::dev_layer` returned `STREAMMOE`
   whenever whole-layer was active, so llama placed the KV cache (and fused ops)
   on host while C1's weights were on the device.
3. **Bucket-engine leaf reads assumed host.** Several sites passed `m->data` as a
   host pointer into `bucket_upload_leaf`; with device-resident sources that is a
   fake `0x1000+off` pointer.
4. **Scratch/staging sizing was hand-rolled** and did not cover the device<->host
   reads that device-dense + mixed pools imply.

## 3. Design

### 3.1 Buft masquerade (`moe_dev_supports_buft`)

Accept the device bufts we execute on (the registered device-exec arena/stage
bufts). This stops the scheduler's weight->host dup. Safe because the executor
runs each node where its operands actually live.

### 3.2 `dev_layer` follows the placement

`llama_model::dev_layer(il)` returns the layer's placement device when it is a
real device, and `STREAMMOE` only for host-dense layers. So the KV cache and
fused ops follow `--dense-placement C1:<dev>`.

### 3.3 `ids` forced to RAM

The routing `ids` are always read to host (the mix plan is built on host:
`build_mix_plan`). This is a forced D2H; the per-device graphs use the host-built
index, not the device ids.

### 3.4 Cross-device movement: option 1 (front bulk copies + local gather)

A ggml graph runs on **one** backend and every operand must already be on it;
there is no cross-device `get_rows` / `cpy` node. So:

- **front bulk copies** (executor-level `ggml_backend_tensor_copy`, D2D / D2H /
  H2D) bring each closure input onto the device once per layer, amortized over
  that device's rounds;
- the **gather** (`get_rows`) is then a local graph node reading device-local
  memory.

"All cross-device gather" (option 2) is infeasible and slower: it degenerates
into per-row/per-element copies or a host bounce.

### 3.5 Per-device closure input/output buffers

For each device, allocate the closure's **input buffers** and its `moe_out`
partial with `ggml_backend_buft_alloc_buffer` (device buft), and **zero them at
the start of each layer** (so partials never accumulate stale data). The
intermediate compute nodes keep using the verify-built plan (`ex->out_off` /
`layout_ok`).

### 3.6 Per-device `moe_out` partials + the anonymous add

The closure already folds/accumulates within itself. Each device folds its own
buckets into **its own `moe_out` partial**. The **anonymous add** that consumes
`ffn_moe_out` is *outside* the closure (`collect_chain` stops at `moe_out`'s
consumers, `route_b_chain.cpp:1746`) and is a **tail** node, so it runs on C1's
device. It gets **one extra `src` per device partial** and sums them
(single device => one input, no extra work). This replaces the host-side
`layer_fold` over per-device `acc_d`.

### 3.7 Backend-agnostic leaf reads

The bucket engine reads every source leaf through a single helper
`bucket_source_leaf` (device staging on a device target, host scratch on a CPU
target, backend-agnostic `tensor_read_host` in both cases). No site may
dereference `src->data` on the host (iron rule, `GRAPH_PARTITION.md`).

### 3.8 Independent grow-only buffers

Scratch / staging become independent grow-only buffers instead of the hand-rolled
`af32`/`a32`/`stage_used` bump with a size estimate, removing the OOB class.
Grow (reallocate) always happens **before** pointers are bound for the current
build.

### 3.9 Strict cross-device audit

`layout_arena` audits every captured node's operands: any operand whose device
differs from the node's device and has no transfer is a hard error. It prints the
problem nodes and the full graph, then `exit(1)` in production; under
`STREAM_MOE_TEMP` + `STREAM_MOE_TMP_AUDIT_CONTINUE` it prints and continues (dev
survey).

### 3.10 Keep the originals (no dense clone)

The dense head/tail keep using the **original** graph nodes (as
`run_dense_subgraph` does), with outputs bound to the per-device arena. No clone:
the MoE bucket engine clones for shape narrowing (bucketing), not for buft
reasons, and the dense path has no shape narrowing.

## 4. Status (2026-09-15)

- **Landed:** buft masquerade (3.1), `dev_layer` (3.2), generic xfer with
  leaf/bufferless coverage and closure-node exclusion, `bucket_source_leaf`
  (3.7), view `buffer` propagation, host full-alloc for the CPU closure path,
  strict audit (3.9).
- **In progress:** per-device closure input/output buffers (3.5), front copies
  (3.4), per-device `moe_out` + anonymous add (3.6), grow-only buffers (3.8).
- **Not done:** device-dense numerical validation (`C1:Vulkan0` + Vulkan pool).

## 5. Files

- `src/backend/route_b_chain.cpp` / `.h` - capture, dev_of, per-device plan,
  generic xfer, audit.
- `src/backend/minigraph_exec.cpp` - bucket engine (`bucket_source_leaf`,
  per-device rounds, exit merge).
- `src/backend/moe_backend.cpp` - `moe_dev_supports_buft`.
- `third_party/llama.cpp/src/llama-model.cpp` - `dev_layer` (via
  `route-b-inject.patch`).
