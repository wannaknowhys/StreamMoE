# StreamMoE Graph Partition - whole-graph map and the closure cut

[English](GRAPH_PARTITION.md) | [简体中文](GRAPH_PARTITION.zh-CN.md)

> Status: **analysis, 2026-09-09**. Maps the full llama.cpp compute graph of
> olmoe / gemma-4-26B-A4B / deepseek-v4-flash into the regions route B owns,
> tags every leaf with its buffer type, and frames the next executor step
> (same-device seam, C1 closure-ization).
>
> Related: `docs/ROUTE_B_GPU_PHASE.md` (closure/privatisation), `docs/DENSE_PLACEMENT.md`
> §2/§7 (C1/C2 categories, closure analysis), `docs/STREAMMOE_GGUF_FORMAT.md` §3
> (C1/C2/C3/C4), `docs/M2_DEVICE_EXECUTOR.md` (per-device executor),
> `docs/BUCKET_EXEC_TOKEN_SUBSET.md`.

> **Iron rule - data movement (2026-09-09)**:
> 1. **ggml tensor bytes**: never dereference or `memcpy` `ggml_tensor::data` on
>    the host. Use the backend-agnostic `ggml_backend_tensor_get` / `_set`
>    (`_2d`; `_async` + `ggml_backend_synchronize` mid-graph, backend via
>    `ggml_backend_sched_get_tensor_backend`; see `export_capture_experts`,
>    `llama-context.cpp:1806`).
> 2. **Our own raw arenas** (expert-pool slots, DIO staging, scratch): route
>    through the existing move pipeline (async DIO + scheduler copy worker +
>    per-pool DMA reader), not ad-hoc `memcpy`. That pipeline owns those bytes;
>    ad-hoc copies duplicate its accounting and lose the device path.
> 3. **Backend `iface`** (`get_tensor`/`set_tensor`/`clear`): this IS the
>    backend's own engine (device DMA or host memcpy) - legitimate.
> 4. **Non-tensor memory** (struct init/`memset`, guid/name copies): negligible
>    and libc is fine, but keep it out of per-token hot paths.
> See §7.4 for the bug this rule comes from.

## 1. How the map is produced

`moe_chain_verify_graph(gf)` runs in the route-b `graph_reserve` frag
(`patches/route-b/common/stmoe_routeb_lctx_reserve.frag`) and therefore sees the
whole `ggml_cgraph` before the scheduler splits. A `STREAM_MOE_TEMP`-gated dump
(`STREAM_MOE_TMP_GRAPH_DUMP=1`, dbg build only) prints:

- every compute node: `N idx op name ne bytes chain=<L|-1> buft=<name|-|>`
- every leaf (weight/input/KV): `L idx op name ne bytes buft=<name>`

`chain=L` marks the privatised expert closure of layer L (see
`collect_chain`, `src/backend/route_b_chain.cpp`). `buft` is resolved through
`ggml_backend_buffer_get_type` when the tensor already has a buffer (weights do;
compute nodes get theirs later at sched split). Reproduce:

```
STREAM_MOE_TMP_GRAPH_DUMP=1 build\StreamMoE_dump_dbg\llama-build\bin\llama-cli.exe ^
  -m <model> -p hi -n 1 -c 2048 -t 16 --expert-backend --fit off ^
  --moe-expert-pools RAM:<N> --dense-placement C1:RAM,C2:RAM --no-warmup < nul > dump.txt 2>&1
node temp/analyze_gdump.js dump.txt
```

## 2. Four regions

The whole graph partitions into four parts. The cut between (2) and (3) is
**hard** (topology-proven by `moe_chain_verify_graph`: no node outside the
closure reads a closure intermediate except `ffn_moe_out`).

| # | Region | What | Executor |
|:-:|:-------|:-----|:---------|
| 1 | **Dense trunk = C1** | per-layer attention (GQA or MLA+DSA), norms, KV writes, dense MLP (gemma/deepseek), shared expert (deepseek `_shexp`), hyper-connections (deepseek) | llama.cpp native (placement only) |
| 2 | **Gating** | `ffn_moe_logits/probs/argsort/topk/weights` (+ weight-norm for gemma/deepseek) | llama.cpp native (dense side) |
| 3 | **Expert closure** | routed `MUL_MAT_ID` + swiglu/geglu + down + weighted + convergence adds + `ffn_moe_out`; weights = `_exps.weight` in the pool | **route B** (privatised) |
| 4 | **Output head = C2** | final norm + lm_head | llama.cpp native (placement only) |

Region 3 is exactly the closure: anchor-driven forward BFS from routed
`MUL_MAT_ID` to `ffn_moe_out` (`collect_chain`). Gating (2) is upstream of the
anchors and stays dense. The closure includes view/reshape aliases of its
producer outputs (they are not compute).

## 3. Per-model structure

Measured with the dump above (reserve graph, `-c 2048`; closure bytes are the
layer's hidden-intermediate block).

| | olmoe-1b-7b | gemma-4-26B-A4B | deepseek-v4-flash |
|:--|--:|--:|--:|
| layers | 16 | 30 | 43 |
| compute nodes | 934 | 2644 | 8521 |
| leaves | 234 | 729 | 1606 |
| closure nodes | 320 (20/layer) | 660 (22/layer) | 774 (18/layer) |
| closure hidden / layer | 344 KB | 578 KB | 608 KB |
| `MUL_MAT_ID` / layer | 3 (gate/up/down) | 2 (gate_up, down) | 3 (gate/up/down) |
| C3 expert weights | 48 / 3720 MB | 60 / 13688 MB | 129 / 140352 MB |
| C1 per-layer dense | 144 / 160.8 MB | 565 / 1711.2 MB | 1193 / 11993.5 MB |
| C2 embedding (`token_embd`) | 1 / 55.3 MB | 2 / 1496 MB | 1 / 1010 MB |
| C4 scale | - | 30 / 15 KB | - |

Model-specific notes:

- **olmoe**: pure MoE, no dense MLP; C1 is attention + norms + router only.
- **gemma**: dense MLP (`ffn_gate/up/geglu/mlp`) + MoE in every layer; C4
  `ffn_down_exps.scale` (512 B/layer) is replicated to `STREAMMOE_HOST`.
- **deepseek**: MLA + DSA lightning-indexer + hyper-connections (`DSV4_HC_*`) +
  shared expert (`ffn_*_shexp`, dense/C1) + routed experts (3 mm/layer).

## 4. The seam (closure external leaves)

`STREAM_MOE_CAP_DUMP=1` prints each layer's external leaves - the tensors the
closure consumes from the dense side, and the one it hands back:

| role | tensor | producer | consumer |
|:-----|:-------|:---------|:---------|
| `w` | `blk.L.ffn_*_exps.weight` | expert pool (scheduler pin) | the closure's `MUL_MAT_ID` |
| `cur` | `ffn_norm-L` (normed hidden) | C1 norm | first `MUL_MAT_ID` |
| `ids` | `ffn_moe_argsort-L` | gating | `MUL_MAT_ID` src[2] |
| `scale` | `ffn_moe_weights-L` (routing weights) | gating | `ffn_moe_weighted` |
| `scale` | `blk.L.ffn_down_exps.scale` (gemma only, C4) | C4 resident | down weightless op |
| **out** | `ffn_moe_out-L` | closure | dense residual add |

So the entire dense<->closure interface is: **`cur` in, `ids` in, routing-weight
`scale` in, expert weights in, `ffn_moe_out` out**. Nothing else crosses.

## 5. Buffer-type marks

| buft name | meaning | where |
|:----------|:--------|:------|
| `STREAMMOE_EXPERT` | expert-pool buft; every routed `_exps.weight` | C3, the closure's weight leaves |
| `STREAMMOE_HOST` | host-mapped buft for C4-replicated small leaves | gemma `_exps.scale` |
| `STREAMMOE_DENSE` | v2-chunk dense strip buft (`topo.incomplete` only) | not present for these complete GGUFs |
| `STREAMMOE` (backend) | the device that receives the privatised closure splits | `graph_compute` of region 3 |

`--dense-placement C1/C2` is **orthogonal** to these marks: it moves dense
weights to a device buft (e.g. `Vulkan0`), which is not a STREAMMOE name.

## 6. C1 / C2 / embedding

`docs/DENSE_PLACEMENT.md` §2 defines **C2 = global dense, non `blk.*`:
`token_embd`, `output`, `output_norm`, `rope_freqs` (+ deepseek `output_hc_*`)**.
So **embedding is part of C2 by category**, but the two C2 members are at
opposite ends of the graph:

- `token_embd` is the **input** (upstream: tokens -> embedding -> layers). llama.cpp
  hard-pins the input layer to CPU (`llama-model.cpp:1489`, "little benefit to
  offloading the input layer"), so `token_embd` stays on CPU regardless of
  `C2:<dev>`.
- `output` + `output_norm` are the **output head** (downstream). They follow the
  output layer device (`route_b_dense_device(spec, il == n_layer_all)`), i.e. C2.

They are the same class only because both are layer-independent; they are not
adjacent and have no direct producer/consumer relation. (Some models tie their
weights; these three keep separate `token_embd` and `output` tensors.)

## 7. Design implications (next executor steps)

### 7.1 Same-device seam: no host round-trip

Today's per-device executor (`M2_DEVICE_EXECUTOR.md` §7.9) stages `cur`/`ids`
host->device and reads `acc_d` device->host at the layer tail, even when C1 and
the pool are on the same device. The rule to add: **when the producer and the
consumer live on the same device, the seam tensor never leaves it.**

- `cur` `[d, T]` is the expensive H2D. If the C1 norm is on the pool's device,
  the closure can bind the norm's output directly.
- `ffn_moe_out` `[d, T]` is the expensive D2H: leave `acc_d` on the device and do
  the fold + residual add on-device (generalises M2-3).
- `ids` is control data: the host-side round planner (`build_mix_plan`,
  `scatter_plan`) reads it to partition tokens. Keeping `ids` on-device requires
  moving round planning on-device, or a fixed/known routing. Flag this: `cur` and
  `ffn_moe_out` are pure data and can stay resident today; `ids` cannot until the
  planner moves.

Fallback: when producer/consumer devices differ, stage as today. The device
identity must be resolved at physical-device granularity (llama `dev_layer` vs
the pool device), not `ggml_backend_t` identity.

### 7.2 C1 closure-ization (whole-layer closure)

C1 alone is **not** an anchor-driven closure: its nodes interleave with the
expert chain (pre-attn norm is C1, the chain is the middle, the tail residual
add is C1 and consumes `ffn_moe_out`). The right unit is the **whole layer**
(`docs/DENSE_PLACEMENT.md` §7): take every layer-L node, seed = layer input,
stop = layer output. Then `cur`/`ids`/`ffn_moe_out` become internal edges and the
seam disappears by construction.

- Cost: route B must execute attention + norms + KV + dense MLP itself (today
  llama.cpp does). That is the Phase 2 "managed C1" change, not a small refactor.
- Mechanical refactor when it lands: generalise `collect_chain` to
  `collect_closure(gf, seed_pred, stop_pred, ...)`; expert closure =
  `(is_routed_mm, is_output_name)`, whole-layer closure = `(is_layer_node,
  layer_output)`. Verify becomes "no external consumer of any layer-L
  intermediate except the layer output".
- Sequencing: 7.1 captures most of the win for **static all-resident C1** (one
  device, no round-trip) without the takeover. 7.2 is the end-state for
  **dynamic / split C1** and multi-device, where the layer unit is already the
  placement unit.

### 7.3 Multi-outlet whole-layer package

C1 is atomic per layer and does **not** cross devices (`DENSE_PLACEMENT.md`
§3.2); experts **do** cross devices (RAM + Vulkan0 + ...). So the layer package
is a fan-out / fan-in shape:

```
C1 prefix (device A) --cur--> [ expert closure pool0 (device A) ]--\
                      \--cur--> [ expert closure pool1 (device B) ]--+--> C1 tail (residual, device A)
                      \--cur--> [ expert closure pool2 (RAM)      ]--/
```

- `cur` is **multi-outlet**: the C1 prefix writes it once; each per-device
  expert sub-package consumes a copy (or the same on-device buffer when the
  device matches).
- `ffn_moe_out` is **multi-inlet**: each sub-package folds its `acc_d`, and the
  C1 tail sums the partials (M2-2's per-pool `acc_d` fan-in, generalised).
- N=1 (C1 and all experts on one device) degenerates to "same-device seam, no
  round-trip" (§7.1).
- `ids` stays control data (host round planner) regardless of N - a whole-layer
  package does not remove that until the planner moves device-side.

### 7.4 B39 (FIXED): raw memcpy on a device leaf

The callout rule comes from a real crash. gemma `C1:Vulkan0` + Vulkan expert
pool crashed at load with `0xC0000005`. lldb stack: `memcpy` <-
`stream_moe_backend_replicate_leaf` (`moe_backend.cpp:478`) <-
`moe_chain_assign_backend` (C4 replication) <- `graph_reserve`. The C4 leaf
(`blk.N.ffn_down_exps.scale`) was allocated on Vulkan (C1 on GPU) and the code
did `memcpy(dev, t->data, bytes)` on the host. **Not** a VRAM-space issue: a
128 MB pool + `-ub 16` still crashed. C1:RAM keeps the leaf host-resident, and
with no Vulkan pool the replication loop is empty - both work, which is why only
`C1:Vulkan0 + pool` hit it. Fix: `ggml_backend_tensor_get(t, dev, 0, bytes)`.
Both crashing configs (`place-c1-exp2`, `place-c1c2-exp2`) now load clean.

Raw-memcpy audit (2026-09-09, `src/`): only `moe_backend.cpp:478` was unsafe.
The rest are legal: backend `iface` implementations (`moe_backend.cpp:98-108`),
non-tensor memory (`moe_backend.cpp:190/199/327`, `async_dio_win.cpp:191`),
name slicing (`route_b_chain.cpp:536`), our own byte arenas - expert-pool slots
and staging (`scheduler.cpp:322/724`, `staging_reader.cpp:114`), and the
`STREAM_MOE_TMP_DEVDBG` diagnostic read of the device arena
(`minigraph_exec.cpp:1256`). The staging upload `minigraph_exec.cpp:564`
(`memcpy(stage_map + off, host_data, bytes)`) writes a host-mapped stage buffer;
it should move to `ggml_backend_tensor_set` when the seam becomes tensor-based.

## 8. Open questions

1. Device-identity map: how to prove "same device" across llama `dev_layer` and
   the route-B pool device, and bind one buffer in both backends.
2. `ids` residency: can the token-subset planner run device-side, or do we accept
   a host round-trip for the small int tensor only?
3. Whole-layer closure vs llama.cpp dense execution: how much of attention/KV can
   be reused as-is (per-device graph) vs must be re-implemented.
4. gemma `C1:Vulkan0 + pool` load crash (BUG_TRACKER B39) - **FIXED** 2026-09-09
   by the backend-agnostic copy rule (§7.4); the seam can now be exercised on a
   shared device.

## 9. Implementation checklist

Ordered by dependency; each step is independently verifiable.

### A. Data-movement hygiene (do first, small)

- [x] B39: `stream_moe_backend_replicate_leaf` reads its source via
  `ggml_backend_tensor_get` (§7.4).
- [x] `minigraph_exec.cpp:564` staging upload -> `tensor_write_host`
  (`ggml_backend_tensor_set`); the fake base pointer makes the stage buffer's
  `set_tensor` resolve the offset.
- [x] `minigraph_exec.cpp:1256` DEVDBG arena read - kept gated (reads a host
  mapping of our own arena; diagnostic only).
- [x] One helper `src/backend/tensor_io.h` (`tensor_read_host` /
  `tensor_write_host`); used by B39, the staging upload, and the ids read.
- [x] Grep guard: `scripts/check_tensor_data.js` (Node, zero-dep) scans `src/` +
  `patches/` for `memcpy/memmove/memset(...->data...)`; wired into `build.bat test`
  and runnable standalone. Exempt a line (or the next line) with the marker
  `iron-rule-exempt` (backend iface). Current: 0 violations.

### B. `ids` host copy (unblocks device-resident routing)

- [x] `exec_layer_burst_chain_buckets` and `exec_layer_burst` pin keys: read
  ids through `host_image()` + `moe_id_at()` (host copy only when the ids buffer
  is not host-resident; compact-ids build and round planning stay host-side).
- [x] Device smoke: gemma `C1:Vulkan0` + Vulkan pool runs clean (exit 0). Note
  the ids-copy branch is defensive: the current sched copies ids into our host
  backend, so it is not yet hit in production; it fires once the closure runs
  as a device graph with ids device-resident.

### C. Same-device seam: `cur` / `ffn_moe_out` no round-trip

- [ ] Device-identity map: `ggml_backend_get_device` / `ggml_backend_dev_name`;
  expose dense `dev_layer[il].dev` and the pool device.
- [ ] `cur`: if the producing norm device == pool device, bind the closure's cur
  leaf to the producer buffer (no staging); else stage.
- [ ] `ffn_moe_out`: keep `acc_d` on device, do the fold + residual add
  on-device; read back only the final layer output (or keep it resident when the
  consumer shares the device).
- [ ] Verify: same-device path vs staged path byte-IDENTICAL.

### D. Multi-outlet whole-layer package (end-state)

- [ ] Generalise `collect_chain` -> `collect_closure(gf, seed_pred, stop_pred,
  ...)`; expert = `(is_routed_mm, is_output_name)`, whole-layer =
  `(is_layer_node, layer_output)`.
- [ ] Package shape: C1 prefix (1 device) -> fan-out `cur` to per-device expert
  closures -> fan-in `ffn_moe_out` (§7.3).
- [ ] Reuse the M2-2 per-device graph + `EXPERT_MOVE_PIPELINE` move machinery;
  C1 execution takeover (attention/KV/dense MLP) is the large piece - gate it on
  dynamic/split C1 only.
- [ ] Verify: per-device graph correctness + no external consumer of any layer-L
  intermediate except the layer output.
