# Route B Layer Executor Design (whole-layer ownership)

[English](LAYER_EXECUTOR_DESIGN.md) | [简体中文](LAYER_EXECUTOR_DESIGN.zh-CN.md)

> Status: **proposal, 2026-09-13**. Replaces the ad-hoc debug whole-layer path
> (`STREAM_MOE_TEMP`). Related: `ROUTE_B_LAYER_OWNERSHIP.md` (target shape),
> `M2_DEVICE_EXECUTOR.md` (per-device executor + arena), `L2_WHOLE_LAYER_REVIEW.md`
> (bug inventory + located divergences), `DENSE_PLACEMENT.md` (C1/C2).

## 1. Problem

The debug whole-layer path has three blockers, all from the same root: route B
**claims nodes through the scheduler** instead of **owning the layer and
executing it internally**.

1. **DeepSeek OOM / crash.** Assigning every layer node to our backend makes the
   scheduler reserve a single `STREAMMOE_HOST` compute buffer (~97 GiB) for the
   whole graph (no cross-layer reuse); on top of the expert pool this exceeds
   RAM -> allocation failure (`--moe-ram-pool 81920`) or `0xC0000005` (`8192`).
2. **olmoe divergence.** The last layer contains the `inp_out_ids` output-token
   reduction (`get_rows`); it is captured into the layer and executed by
   `run_dense_nodes`, corrupting the output. Bisect: L0-L14 alone are identical
   to baseline, L15 alone diverges; excluding L15 restores correct output.
3. **Flash attention disabled.** Whole-layer mode turns `FLASH_ATTN_EXT` into a
   manual `kq/kqv` path, materializing the `O(B^2)` score matrix (see 4.9).

## 2. Goal / non-goals

Goal: route B owns each layer (and, as a later stage, C2); the scheduler sees
only model I/O; each layer is executed internally with one reusable arena; the
last-layer narrowing is handled explicitly; flash attention is preserved.

Non-goals: writing dense kernels (dense ops are delegated to the device/CPU
backend); changing C1/C2 placement semantics.

## 3. Architecture

- Our backend owns the whole graph -> **one split**.
- `graph_compute` walks the graph **layer by layer** using a build-time
  `LayerPlan`.
- Per layer: `dense head -> MoE burst -> dense tail`, then the layer is complete.
- **One arena buffer per device, sized to the worst-case (largest) layer, reused
  across layers.** "Per-layer" below refers to the *layout analysis* (offsets),
  not to separate backing allocations.
- Dataflow: the KV cache is resident (llama-owned, not in our arena); our arena
  holds only the transient per-layer activations; only the hidden state `X` and
  the residual cross layer boundaries. Hence worst-case-layer sizing suffices.
- Why the debug path gives ~97 GiB: `ggml_backend_sched` sizes its compute buffer
  over the whole graph and does not reuse across layers. **We do not rely on
  gallocr**: our own arena plus pre-allocated buffers (4.2) bound the memory to
  one worst-case layer.
- The scheduler's compute buffer covers only model I/O (`embd`, `logits`).

## 4. Mechanisms

### 4.1 LayerPlan (build time)

Replace the runtime heuristics (`collect_layer_nodes` + name suffix +
`strstr("ffn_moe_out")` + `down[]` propagation + `has_first`/`first_node`) with a
structure computed once at graph build:

```cpp
struct moe_layer_plan {
    int32_t layer;
    std::vector<ggml_tensor*> all;
    std::vector<ggml_tensor*> head;   // dense before MoE input
    std::vector<ggml_tensor*> moe;    // MoE closure
    std::vector<ggml_tensor*> tail;   // dense after MoE output
    ggml_tensor* input;               // layer input
    ggml_tensor* moe_out;             // MoE anchor (ffn_moe_out)
    ggml_tensor* output;              // layer output
    bool host_owned;                  // dense weights host-resident
};
```

Source of truth: a **side channel** registered by `llm_build_context` while it
builds each layer / MoE subgraph (see 4.4), not string matching.

### 4.2 Pre-allocated buffers (fixes DeepSeek)

The scheduler treats a tensor that **already has a buffer** as pre-allocated
(`ggml-backend.cpp:911` assigns it to the buffer's backend; `:927` never
allocates it).

**Prototype result** (`temp/proto_prealloc.cpp`, CPU backend, 32 live
intermediates of 256 KiB): control compute buffer 8,912,896 B -> with the
intermediates pre-allocated 786,432 B, and 32/32 preset `data` pointers survive
`ggml_backend_sched_alloc_graph`. Mechanism confirmed.

Plan:

- one **arena** `ggml_backend_buffer` per device (buft registered to our backend);
- before scheduling, point each layer node's `buffer`/`data` at the arena;
- the scheduler assigns those tensors to our backend and does not allocate them;
- `graph_compute` executes them in place.

**CPU caveat:** the current closure path sets only `data`
(`minigraph_exec.cpp:730`), not `buffer`. Both must be set, or the scheduler
still allocates. The device path already sets `buffer` (`:724`).

### 4.3 Arena: three regions (carry / compact / closure)

One arena buffer per device, sized once (never re-grown - a grow would
invalidate already-set `data` pointers), split into three fixed sub-regions:

```
[ carry region (fixed base) ][ compact region ][ closure block ]
```

- **carry region (FIRST)**: every tensor whose live range crosses a layer
  boundary (any depth) - the residual stream / layer output and boundary
  transforms (e.g. the last layer's `inp_out_ids` narrowing). Fixed addresses,
  never reused. Placed first so a change in the compact/closure sizes cannot move
  carry addresses (carry pointers are live across layers).
- **compact region**: within-layer temporaries, packed per layer and **reused**
  across layers; size = max over layers. Same addresses every layer.
- **closure block**: the existing MoE-closure layout (`ex.out_off` /
  `ex.result_bytes`) + bump, reused across layers; size = max over layers.

Global (whole-graph) liveness is used only to **classify** carry vs within-layer,
not to assign global addresses. Any cross-layer tensor goes to the carry region:
keeping a cross-layer tensor in the compact region would let the next layer
overwrite it before consumption (order-fragile - rejected).

Reuse the existing packing: extract the best-fit interval packing currently in
`moe_chain_verify_graph` (`route_b_chain.cpp:282-370`) into
`pack(nodes, last_use) -> offsets, size`; call it for the closure block and the
compact region. Do not write a second allocator.

The closure stays separate because its execution is special:

- the bucket engine rebuilds a mini-graph per bucket/round (transient tensors,
  bump region);
- `ffn_moe_out` / the fold is a dedicated output;
- the anonymous per-topk adds (`node_NNN ADD`) are closure nodes.

**Heterogeneous layers.** Every layer's plan and packing are computed
independently; only the region sizes are shared (max / peak). Edge rules:

- pure dense layer (no MoE): empty closure, the whole layer is dense;
- no `ffn_moe_out`: use the closure's last node as the anchor;
- different attention types (sliding-window / SSM): the head/tail split depends
  only on the closure anchor, not on the attention op;
- special layer indices: the official layer channel handles them;
- multiple MoE subgraphs in one layer: currently one `ex` per layer; extend the
  plan to a list if needed (open).

Motivating example (consumer check, R1): the last layer's `inp_out_ids`
`get_rows(inpSA, ...)` reads the previous layer's output and is consumed by the
last layer; producer propagation attributes it to the previous layer. It is a
carry (cross-1) and must live until the last layer consumes it.

### 4.4 No-clone dense execution (fixes olmoe)

Execute the **original nodes** (`nd`), not `ggml_dup` clones with hand-filled
`data`/`nb`/`view`. Because route B owns the whole graph (one split), the
scheduler never executes our nodes; we run them ourselves, so there is no double
execution. A `LayerExecutionState {NOT_STARTED, RUNNING, COMPLETE}` guarantees
once-only and asserts on duplicate concurrent execution.

### 4.5 Explicit layer boundary (side channel)

`llm_build_context` knows the layer index and the MoE span while building.
Register `layer_span_t {begin, moe_begin, moe_end, end}` per layer into a side
channel. This deletes `name_layer_suffix`, `last_suffixed`,
`moe_chain_layer_of_node`, `ffn_moe_out` string matching, and the
`has_first`/`first_node` split heuristic (review items A1/A2/A5/E1/E2/E4).

### 4.6 op classification

Three classes (absorbs review item E3):

- `PURE_ALIAS`: `VIEW / RESHAPE / TRANSPOSE / PERMUTE` - no compute, must
  guarantee data/layout.
- `MATERIALIZING`: `CONT / DUP / CAST` - real copy, must execute.
- `COMPUTE`: everything else.

### 4.7 Last-layer output-token narrowing (explicit)

The last layer contains (e.g. `olmoe.cpp`):

```cpp
if (il == n_layer - 1 && inp_out_ids) {
    cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
```

These `GET_ROWS` narrow the token dimension to `n_outputs`. The executor
**explicitly supports** this:

- the narrowing nodes are `COMPUTE` (not alias) and run in the head;
- from the narrowing onward the layer runs with `n_tokens = n_outputs`
  (`ffn_inp`, MoE pin / mix-plan / arena, tail all use the narrowed count);
- the arena / pin budget for that layer is computed from the narrowed count.

### 4.8 tracer separation + clean allocation failure

Move canary / dump / `fprintf` out of `run_dense_nodes` / `exec_layer_burst`
into an injectable tracer (no-op by default). On arena / buffer allocation
failure, return a clear `GGML_STATUS_FAILED` with an actionable message (never a
bare `0xC0000005`).

### 4.9 Flash attention MUST be preserved

The current whole-layer path turns `FLASH_ATTN_EXT` into manual `kq / softmax /
kqv`, which materializes the `O(B^2)` score matrix. Per-layer estimates (f32,
`h` heads):

| model | B | head (linear + attn) | closure | tail |
|---|---|---|---|---|
| gemma | 2048 | **900 MB (132 + 768)** | 77 MB | 60 MB |
| olmoe | 2048 | **864 MB (96 + 768)** | 48 MB | 40 MB |
| deepseek | 2048 | **3264 MB (192 + 3072)** | 96 MB | 80 MB |

The attention term dominates and scales as `B^2`. Route B owns the layer but
**delegates** ops it does not implement; `FLASH_ATTN_EXT` must be delegated to
the device/CPU backend so the score matrix is never materialized. This is a hard
requirement (also a D item).

### 4.10 C2 takeover (later stage)

Today `--dense-placement C2:<dev>` only sets the placement buft
(`route_b_dense_device`, `route_b_inject.cpp:445`); the C2 nodes are still
executed by llama, **not owned** by route B. So the route-B-layer -> C2 seam can
still copy even on the same device.

Target: route B also owns the C2 activations (pre-allocated in the arena), so
C1 and C2 on the same device are copy-free. Space is independent until a unified
liveness analysis merges C1/C2 (they do not overlap in time).

## 5. D problems (must be designed, not automatic)

| D problem | How this design addresses it |
|---|---|
| Whole-graph single-backend compute buffer | One worst-case-layer arena, reused (3, 4.2) |
| Last-layer `inp_out_ids` semantics | Explicit narrowing support (4.7) |
| Flash attention must stay on | Delegate `FLASH_ATTN_EXT` (4.9) |
| C2 seam copies | C2 takeover (4.10) |
| `supports_buft` / `supports_op` vs ownership | route B declares the arena buft + expert bufts |
| dense execution order / data readiness | internal order + `LayerExecutionState` + asserts |
| debug scaffold on the hot path | tracer separation (4.8) |
| allocation failure crash | clean error path (4.8) |

## 6. Milestones

- **R1** LayerPlan + explicit boundary (no behavior change; keep MoE-only).
- **R2** No-clone dense execution + `LayerExecutionState` -> fixes olmoe.
- **R3** Pre-allocated three-region arena (carry first / compact reused / closure
  block) -> fixes DeepSeek.
- **R4** Last-layer narrowing explicit support.
- **R5** Preserve flash attention (delegate `FLASH_ATTN_EXT`).
- **R6** C2 takeover (own C2 activations) -> copy-free C1/C2 on one device.
- **R7** op classification + tracer separation.

Each milestone keeps the production (MoE-only) path numerically IDENTICAL.

## 7. Validation gates

- Production pure-RAM **IDENTICAL** (gemma, olmoe) via `baseline_regression`.
- Whole-layer olmoe equals baseline (coherent output, hidden cos ~1.0).
- DeepSeek whole-layer loads and passes the cos gate (`baseline/deepseek_hi_up`).
- Logged scheduler compute-buffer size stays bounded (model I/O only).
- **Build-time debug log** of per-layer head / closure / tail bytes (to see the
  real numbers for the three models at a chosen ubatch).

## 8. Open questions

1. Arena liveness/sizing: one grow-only arena vs exact worst-case measurement.
2. Device-local arena (DMA) for the GPU phase - reuse `M2_DEVICE_EXECUTOR`.
3. Multi-device: per-device arenas + the ids D2H join.
4. C2 ownership: own C2 activations only, or also its weights (C1 pool reuse)?
