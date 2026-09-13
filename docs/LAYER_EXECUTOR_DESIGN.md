# Route B Layer Executor Design (whole-layer ownership)

[English](LAYER_EXECUTOR_DESIGN.md) | [简体中文](LAYER_EXECUTOR_DESIGN.zh-CN.md)

> Status: **proposal, 2026-09-13**. Replaces the ad-hoc debug whole-layer path
> (`STREAM_MOE_TEMP`). Related: `ROUTE_B_LAYER_OWNERSHIP.md` (target shape),
> `M2_DEVICE_EXECUTOR.md` (per-device executor + arena), `L2_WHOLE_LAYER_REVIEW.md`
> (bug inventory + located divergences), `DENSE_PLACEMENT.md` (C1/C2).

## 1. Problem

The debug whole-layer path has two blockers, both from the same root: route B
**claims nodes through the scheduler** instead of **owning the layer and
executing it internally**.

1. **DeepSeek OOM / crash.** Assigning every layer node to our backend makes the
   scheduler reserve a single `STREAMMOE_HOST` compute buffer (~97 GiB) for the
   whole graph; on top of the expert pool this exceeds RAM -> allocation failure
   (`--moe-ram-pool 81920`) or `0xC0000005` (`8192`).
2. **olmoe divergence.** The last layer contains the `inp_out_ids` output-token
   reduction (`get_rows`); it is captured into the layer and executed by
   `run_dense_nodes`, corrupting the output. Bisect: L0-L14 alone are identical
   to baseline, L15 alone diverges; excluding L15 restores correct output.

## 2. Goal / non-goals

Goal: route B owns each layer; the scheduler sees only model I/O; each layer is
executed internally with its own arena; the last-layer narrowing is handled
explicitly.

Non-goals: C2 (`token_embd` / `output`) placement policy; writing dense kernels
(dense is delegated to the device/CPU backend).

## 3. Architecture

- Our backend owns the whole graph -> **one split**.
- `graph_compute` walks the graph **layer by layer** using a build-time
  `LayerPlan`.
- Per layer: `dense head -> MoE burst -> dense tail`, then the layer is complete.
- Layer activations live in **route B's own arena**, not in the scheduler's
  compute buffer.
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

### 4.2 Per-layer arena + pre-allocated buffers (fixes DeepSeek)

The scheduler treats a tensor that **already has a buffer** as pre-allocated
(`ggml-backend.cpp:911` assigns it to the buffer's backend; `:927` never
allocates it). Therefore:

- route B owns a **per-layer arena** (a `ggml_backend_buffer` whose buft is
  registered to our backend);
- before scheduling, route B points each layer node's `buffer`/`data` at the
  arena;
- the scheduler assigns those tensors to our backend and **does not allocate
  them** in the compute buffer -> the compute buffer stays bounded (model I/O
  only);
- `graph_compute` executes them in place.

Open: arena sizing (layer liveness), one arena vs one grow-only arena, device
variant (device-local arena, reuse `M2_DEVICE_EXECUTOR`).

### 4.3 No-clone dense execution (fixes olmoe)

Execute the **original nodes** (`nd`), not `ggml_dup` clones with hand-filled
`data`/`nb`/`view`. Because route B owns the whole graph (one split), the
scheduler never executes our nodes; we run them ourselves, so there is no double
execution. A `LayerExecutionState {NOT_STARTED, RUNNING, COMPLETE}` guarantees
once-only and asserts on duplicate concurrent execution.

This removes the second execution semantics that broke the last-layer reduction.

### 4.4 Explicit layer boundary (side channel)

`llm_build_context` knows the layer index and the MoE span while building.
Register `layer_span_t {begin, moe_begin, moe_end, end}` per layer into a
side channel. This deletes `name_layer_suffix`, `last_suffixed`,
`moe_chain_layer_of_node`, `ffn_moe_out` string matching, and the
`has_first`/`first_node` split heuristic (review items A1/A2/A5/E1/E2/E4).

### 4.5 op classification

Three classes (absorbs review item E3):

- `PURE_ALIAS`: `VIEW / RESHAPE / TRANSPOSE / PERMUTE` - no compute, must
  guarantee data/layout.
- `MATERIALIZING`: `CONT / DUP / CAST` - real copy, must execute.
- `COMPUTE`: everything else.

### 4.6 Last-layer output-token narrowing (explicit)

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

This is what makes whole-layer ownership correct for the last layer.

### 4.7 tracer separation

Move canary / dump / `fprintf` out of `run_dense_nodes` / `exec_layer_burst`
into an injectable tracer (no-op by default). Keeps the hot path clean and
removes the long-lived debug globals.

### 4.8 Clean allocation failure

When an arena / buffer allocation fails, return a clear `GGML_STATUS_FAILED`
with an actionable message (never a bare `0xC0000005`).

## 5. D problems (must be designed, not automatic)

| D problem | How this design addresses it |
|---|---|
| Whole-graph single-backend compute buffer | Pre-allocated per-layer arena buffers (4.2) -> scheduler compute buffer stays bounded |
| Last-layer `inp_out_ids` semantics | Explicit narrowing support (4.6) |
| `supports_buft` / `supports_op` vs ownership | route B declares the arena buft + expert bufts; no blanket accept |
| dense execution order / data readiness | route B controls execution order internally; `LayerExecutionState` + asserts |
| debug scaffold on the hot path | tracer separation (4.7) |
| allocation failure crash | clean error path (4.8) |

## 6. Milestones

- **R1** LayerPlan + explicit boundary (no behavior change; keep MoE-only).
- **R2** No-clone dense execution + `LayerExecutionState` -> fixes olmoe.
- **R3** Per-layer pre-allocated arena -> fixes DeepSeek.
- **R4** Last-layer narrowing explicit support.
- **R5** op classification + tracer separation.

Each milestone keeps the production (MoE-only) path numerically IDENTICAL.

## 7. Validation gates

- Production pure-RAM **IDENTICAL** (gemma, olmoe) via `baseline_regression`.
- Whole-layer olmoe equals baseline (coherent output, hidden cos ~1.0).
- DeepSeek whole-layer loads and passes the cos gate (`baseline/deepseek_hi_up`).
- Logged scheduler compute-buffer size stays bounded (model I/O only).

## 8. Open questions

1. Arena liveness/sizing: one arena per layer vs one grow-only arena.
2. Device-local arena (DMA) for the GPU phase - reuse `M2_DEVICE_EXECUTOR`.
3. Scheduler split behavior when most tensors are pre-allocated.
4. Multi-device: per-device arenas + the ids D2H join.
