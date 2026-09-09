# Dense Placement - C1/C2 residency and migration design

[English](DENSE_PLACEMENT.md) | [简体中文](DENSE_PLACEMENT.zh-CN.md)

> Status: **Phase 1a landed** (2026-09-08). Route B owns expert placement; this
> doc makes dense placement an explicit, parameter-controlled concern too.
> Static placement (Phase 1) is the main path; dynamic C1 migration (Phase 2) is
> a capacity-constrained fallback.
>
> Related: `docs/STREAMMOE_GGUF_FORMAT.md` §3 (C1/C2/C3/C4), `docs/ROUTE_B_GPU_PHASE.md`
> §6 (per-layer dense/KV), `docs/M2_DEVICE_EXECUTOR.md` §7.8 (final fold follows
> dense), `docs/TODO.md` stage 7, `docs/EXPERT_MOVE_PIPELINE.md` (move machinery),
> `docs/VRAM_DMA_MOVE.md` (measured device bandwidth).

## 1. Goal

`--expert-backend` currently forces `no_op_offload=true` and `n_gpu_layers=0`
(`patches/route-b/common/stmoe_routeb_args.frag`): the expert pool is placed by
route B, but dense placement is left entirely to the user's explicit `-ngl`
(which is contiguous from the tail, `llama-model.cpp:1455`).

This design makes dense placement a first-class, parameter-controlled concern in
the same style as `--moe-expert-pools`, with four rules:

1. **C2 is static and never migrated.**
2. **C1 is per-layer, may be split across devices at layer granularity; KV
   follows the C1 layer.**
3. **A C1 layer is migrated only when the lag comparison says it is worth it;
   otherwise it stays where it is (lazy migration).**
4. **`-ngl` is rejected under `--expert-backend`** (warn + exit): route B owns
   dense placement; dense is expressed only through `--dense-placement`.

## 2. Categories and measured sizes

From `docs/STREAMMOE_GGUF_FORMAT.md` §3.1/§3.1 table (measured):

| Class | Meaning | deepseek4 (43 layers / 256 experts) | gemma4 (30 / 128) |
| :-- | :-- | :-- | :-- |
| C1 | per-layer dense (`blk.N.*` non-expert: attn, norms, dense ffn, router, shexp, indexer/hc) | 9920.6 MB total, 209.0-245.6 MB/layer | 1711.2 MB, 54.6-69.2 MB/layer |
| C2 | global dense (non `blk.*`: `token_embd`, `output`, `output_norm`, `rope_freqs`, `output_hc_*`) | 2020.3 MB | 748.0 MB |
| C3 | per-expert (`_exps.weight`, `ne[2]==n_expert`) | 137.1 GB (12.8 MB/expert) | 13.4 GB |
| C4 | closure-used non-per-expert small table (gemma `_exps.scale`) | empty | 15 KB |

**Key fact**: deepseek C1+C2 = 11940.9 MB ~= 11.66 GiB, which fits in 128 GiB
RAM. Dense streaming is therefore **not** a capacity necessity for RAM; it is a
GPU-acceleration decision. (The "dense 162 GB" line in `docs/TODO.md` stage 5 is
stale - 162 GB is the expert total.)

## 3. Placement principle

### 3.1 C2 static global

C2 (`token_embd`/`output`/`output_norm`/`rope_freqs`) is used every token and
never migrated. `output` is one large per-token matmul and is the highest-value
dense tensor to pin on a GPU. Note: llama forces the input layer to CPU
(`llama-model.cpp:1470`, "little benefit to offloading the input layer"), so
`token_embd` stays on CPU by design; `output` follows the output layer device.

### 3.2 C1 atomic per layer, split at layer granularity

C1 is organised as one unit per layer and may be assigned to different devices
per layer. **Do not split within a layer**: attention on one device and shexp on
another reintroduces the weightless-op cross-device copy problem (see §7). The
layer is the atomic placement unit.

### 3.3 KV follows C1

The KV cache is per-layer state produced and consumed by that layer's attention.
It must live on the same device as the layer (`llama-kv-cache.cpp:215`:
`dev = model.dev_layer(il)` when `offload_kqv` is on). A migrated C1 layer takes
its KV with it; this KV cost is part of the migration cost (§4).

### 3.4 No C1 eviction pool

C1 is used every token, every layer, with **zero intra-token reuse** and a
strictly sequential access order. An LRU/eviction pool would miss on every layer
and thrash. Therefore C1 has exactly three states:

- **static resident** (RAM, or VRAM when it fits) - decode default;
- **sequential prefill stream** (double-buffered, prefetch L+1 while computing
  L) - prefill only, this is a stream, not a cache pool;
- **dynamic migration** (Phase 2 fallback) - only when a layer does not fit and
  the lag comparison justifies the move.

## 4. Decision rule: compute lag vs migration lag

For one C1 layer with `Np` parameters, `b` bytes/param, effective CPU/GPU
throughput `P_cpu`/`P_gpu`, link bandwidth `BW`, and a decision window of `T`
tokens:

```
compute saved  = T * 2 * Np * (1/P_cpu - 1/P_gpu)
migration cost = (Np*b + KV_bytes) / BW
migrate  <=>  T > (b + KV_bytes/Np) / (2 * BW * (1/P_cpu - 1/P_gpu))
```

With `b ~= 1`, `P_cpu ~= 0.5 TFLOPS`, `P_gpu ~= 3 TFLOPS`, `BW = 21 GB/s`,
`T* ~= 14 tokens`. On the measured 8 GB/s ReBAR write of the RX 590
(`docs/VRAM_DMA_MOVE.md`), `T* ~= 38 tokens`. This is why prefill (hundreds to
thousands of tokens) migrates and decode (T=1) does not.

Three refinements:

1. **KV is part of the migration cost.** For deepseek MLA the KV term is small
   (~1.15 KB/token/layer, ~49.5 KB/token over 43 layers); for GQA models it is
   large and can dominate.
2. **Hysteresis + cooldown.** Near the threshold the decision oscillates. Reuse
   the `EXPERT_MOVE_PIPELINE` machinery (margin + cooldown + copy-then-release);
   do not build a second mechanism.
3. **Window = batch/phase, not per token.** "Do not migrate if not worth it" is
   lazy migration: once placed, only move if the round-trip pays. Per-token
   decisions thrash.

Prefetch changes the criterion: if L+1 is prefetched while L computes, the
transfer is hidden and the effective cost is lower (pipelining across layers).

## 5. Implementation

### Phase 1 - static placement (main path, small change)

**`-ngl` is rejected under `--expert-backend`.** The loader's contiguous `-ngl`
conflicts with route-B-owned dense placement, so if `--expert-backend` is set and
`-ngl`/`--gpu-layers` was given, route B warns and exits. Detection: leave
`n_gpu_layers` at its default `-1` (do not overwrite it in the `--expert-backend`
handler), then validate after parse - any other value means the user touched
`-ngl`. Edge: an explicit `-ngl auto` sets the same value as the default and is
treated as not given.

**Phase 1a (first milestone): whole-set device selection.**

```
--dense-placement <spec>
  spec  := item[,item...]
  item  := C1:<dev>            # ALL C1 layers (whole)
         | C2:<dev>            # ALL C2 (whole; alias GLOBAL)
  dev   := RAM | CPU | Vulkan0 | CUDA0 | ...
```

- Examples: `--dense-placement C1:Vulkan0,C2:Vulkan0` (all dense on the GPU),
  `--dense-placement C2:Vulkan0,C1:RAM` (C2 on GPU, C1 on CPU).
- Default when `--dense-placement` is absent: all dense on CPU (today's
  behavior - a RAM-only run allocates no GPU dense buffer).

**Landed (2026-09-08)**: `--dense-placement C1:<dev>,C2:<dev>` fills
`llama_model_params.dense_placement`; `get_layer_buft_list` consults
`stream_moe::route_b_dense_device` (route-b hook in `llama-model.cpp`). `-ngl`
under `--expert-backend` is rejected (warn + exit); a bad device name is
rejected. Verified on gemma-4-26B (129-token prefill-from): default and
`C1:RAM,C2:RAM` byte-IDENTICAL; `C1:Vulkan0,C2:Vulkan0` places 30 C1 + 1 output
layer on Vulkan0 and diverges at the known backend-noise scale (hidden cos
~0.986); `C1:RAM,C2:Vulkan0` / `C1:Vulkan0,C2:RAM` diverge ~0.9999. DeepSeek C1
on the RX 590 8G is not feasible (11.66 GB dense > 8 GB) - it is the P100 16G
target.

**Phase 1b (later): per-layer table.**

```
  item  := ... | L<a>[-<b>]:<dev> | LAYER:<dev> | OUTPUT:<dev>
```

- P100 16G holds all dense: `C1:Vulkan0,C2:Vulkan0` (Phase 1a already covers it).
- Partial: `C2:Vulkan0,LAYER:RAM,L27-42:Vulkan0` (Phase 1b).

Implementation (shared by 1a/1b): a route-b hook in `get_layer_buft_list`
(`llama-model.cpp:1457`) consults the device table and returns
`{dev, gpu_buft_list}`; the output layer uses the `OUTPUT:`/`C2:` device. Dense
execution stays llama-native; we only decide placement. This also moves KV
automatically (`offload_kqv`).

- **Do not use `-ot` for this**: `-ot` changes only the weight buft, not
  `dev_layer`, so KV and weightless ops stay on CPU and every attention node
  crosses devices.
- `op_offload` interaction: route B forces `no_op_offload=true` to keep RAM-only
  runs off the GPU. When any dense layer is on a GPU device, weightless dense ops
  (rope/softmax/...) must be allowed to follow, so `op_offload` must be enabled
  for that run. Rule: `op_offload = (any dense layer on a GPU device)`. Accept
  the resulting Vulkan compute buffer (~1.3 GB) and the GPU numeric form.

### Phase 2 - dynamic C1 migration (fallback, large change)

- Dynamic migration cannot be expressed by llama's static `dev_layer`; C1 weights
  must be owned and moved by route B (a managed C1 weight pool), with our
  executor taking over the layer. Reuse `EXPERT_MOVE_PIPELINE` (move worker,
  hysteresis, copy-then-release, per-pool eviction).
- Entry point: only when C1 does not fit the target device (or KV growth eats
  the budget). Prefer Phase 1 all-resident whenever it fits.
- Attention + KV move together; keep the layer atomic.

## 6. Capacity math (deepseek4)

| Config | Capacity | C1+C2 = 11.66 GB fits? | Notes |
| :-- | :-- | :-- | :-- |
| P100 16G alone | ~15.5 GB usable | **yes** | ~3.8 GB spare for KV + small expert pool |
| RX590 8G alone | ~7 GB usable | **no** | C2 (2.02 GB) + ~20 C1 layers, or all to expert pool |
| RX590 8G + P100 16G | ~24 GB | **yes, easily** | put all dense on P100, 590 for expert pool |

Once a P100 16G is available, **all dense should be statically resident on it**
(HBM2 ~732 GB/s vs PCIe 21 GB/s), which removes the dense bottleneck from both
prefill and decode. Dynamic migration then only matters when dense does not fit
(e.g. 590-only, or very long context where KV exhausts the P100).

KV budget on P100 (deepseek MLA, ~1.15 KB/token/layer x 43 ~= 49.5 KB/token):

| Context | KV | C1+C2+KV | Spare of 15.5 GB |
| --: | --: | --: | --: |
| 4k | 0.20 GB | 11.86 GB | ~3.6 GB |
| 32k | 1.62 GB | 13.28 GB | ~2.2 GB |
| 64k | 3.24 GB | 14.90 GB | ~0.6 GB |
| 128k | 6.48 GB | 18.14 GB | does not fit |

Pascal caveat: P100 is CC 6.0 with no tensor cores and no display output; verify
ggml-vulkan runs `MUL_MAT_ID` and the quantized kernels on it, and keep a display
GPU (the RX 590) for the desktop.

## 7. Verify / closure analysis

**Decision: do not change `moe_chain_verify_graph` / `collect_chain` now, and do
not build "two forward closures".**

- `collect_chain` is an anchor-driven forward BFS from expert `MUL_MAT_ID` to
  `ffn_moe_out` (`src/backend/route_b_chain.cpp:552`). C1 is not such a closure:
  it is the layer's dense nodes, which interleave with the expert chain (pre-attn
  /norm are C1, the expert chain is the middle, the tail residual add is C1 and
  consumes `moe_out`). There is no single anchor and no single direction.
- The verify job is only to prove that privatised intermediates have no external
  consumer. Phase-1 static placement does not privatise C1 (llama executes it),
  so verify is untouched.
- When Phase 2 takes over C1 execution, the right abstraction is a **whole-layer
  closure** (`layer[]` already gives ownership; take all layer-L nodes), and
  verify becomes "no external consumer of any layer-L intermediate except the
  layer output" - which converges to the whole-layer self-scheduling in
  `docs/ROUTE_B_GPU_PHASE.md` §3.
- The only mechanical refactor worth doing then: extract the BFS skeleton into
  `collect_closure(gf, seed_pred, stop_pred, out_chain, out_layer)`; expert
  closure = `(is_routed_mm, is_output_name)`. Defer until C1 takeover.

## 8. Open questions

1. **`-ngl` vs `--dense-placement`** (resolved): under `--expert-backend`,
   `-ngl` is rejected (warn + exit); dense is expressed only through
   `--dense-placement`. See §5.
2. **Partial static placement** (some layers VRAM, some RAM) is allowed by the
   Phase 1b table; confirm the residual hidden-state cross-device copy per
   boundary layer is acceptable (it is one `[d, T]` tensor).
3. **P100 Vulkan viability** (MUL_MAT_ID + quantized kernels on Pascal) - must be
   measured before committing to "all dense on P100".
4. **KV slot paging** (single-slot-into-VRAM, forbid cross-slot ubatch) is a
   separate KV feature; out of scope here. See `docs/TODO.md` stage 5.
