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

## 8. Open questions

1. Device-identity map: how to prove "same device" across llama `dev_layer` and
   the route-B pool device, and bind one buffer in both backends.
2. `ids` residency: can the token-subset planner run device-side, or do we accept
   a host round-trip for the small int tensor only?
3. Whole-layer closure vs llama.cpp dense execution: how much of attention/KV can
   be reused as-is (per-device graph) vs must be re-implemented.
4. gemma `C1:Vulkan0 + pool` load crash (BUG_TRACKER B39, open) sits exactly on
   the C1<->closure seam and must be fixed before either step is exercised on a
   shared device.
