# Build Graph Entry Points & output[] Control

> [English](GRAPH_BUILD_OUTPUT.md) | [简体中文](GRAPH_BUILD_OUTPUT.zh-CN.md)
>
> Line numbers below are for the **clean upstream `f280b2698`** (vendored working tree with no patches applied). Applying phase-1 / prefill / route-b patches shifts them; search by symbol, not line number.

## 1. The three places that build the graph

`model.build_graph()` is called from exactly three sites, all in `src/llama-context.cpp`:

| #   | Site (file:line)                                          | Upper callers                                                                                                                                                    | Purpose                                                                                                                         | Graph built                                                               |
| :-- | :-------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------ |
| 1   | `graph_reserve` : `src/llama-context.cpp:2431`            | `sched_reserve()` (581) -> `graph_reserve` at 633 (PP worst-case), 653 (TG), 668 (PP again); `resolve_fused_ops` (513, fused-op probe); post-memory-update (830) | **Reserve** worst-case graphs to size sched splits / compute buffers; **never executed**                                        | Full-size PP graph (n_tokens = min(n_ctx,n_ubatch)) + single-seq TG graph |
| 2   | `process_ubatch` : `src/llama-context.cpp:1358`           | `llama_decode` main loop (1816, once per ubatch); `llama_encode` (1463)                                                                                          | **Real inference** graph. First checks `can_reuse` (1339): if the params match the previous graph it reuses, otherwise rebuilds | Actual ubatch shape (prefill = many tokens, decode = n_seqs tokens)       |
| 3   | `llama_encode` direct path : `src/llama-context.cpp:3418` | `llama_encode` API                                                                                                                                               | encode/embedding path, **forced** `res->reset()` + build every ubatch (no reuse), own compute context                           | Actual ubatch graph                                                       |

Both prefill and decode run through **entry #2** (`process_ubatch`). They differ only by the ubatch parameters (n_tokens / n_seqs / n_outputs), never by a distinct prefill-vs-decode entry.

## 2. Where output[] (batch.logits / ubatch.output) is set

The "which tokens get logits/embd" decision flows: server decides -> `batch.logits[]` -> `llama_decode` -> `ubatch.output[]` -> `n_outputs` -> LM-head gather.

| Layer                                 | File:line                                                                                                                                                | What it does                                                         |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| Server decides per-token output       | `tools/server/server-context.cpp:151-154` (`server_batch::set_output`) — callers in update_slots / process mark the **last** token true by default       | Stores `tokens[idx].output`                                          |
| Batch render -> `batch.logits[]`      | `common/common.cpp:1851` (`common_batch_add`): `batch.logits[batch.n_tokens] = logits;` — invoked from `server_batch::render()` (server-context.cpp:156) | Writes the output flag into the llama_batch                          |
| `--prefill-from` mode (prefill patch) | `tools/server/server.cpp`: `lg.back() = 1;`                                                                                                              | Forces output on the last token for the prefill-from one-shot decode |
| llama_batch -> llama_ubatch           | `src/llama-context.cpp` (`llama_decode` -> ubatch prep)                                                                                                  | Copies `batch.logits` into `ubatch.output[]`                         |
| `n_outputs` count                     | `src/llama-context.cpp:1800-1811`                                                                                                                        | `n_outputs = sum(ubatch.output[i])`                                  |
| Graph: out_ids tensor                 | `src/llama-graph.cpp:2425-2444` (`build_inp_out_ids`), `199-224` (`llm_graph_input_out_ids::set_input` collects indices where `ubatch.output[i]`)        | LM head only computes the `n_outputs` output rows                    |

Note: hidden (`t_h_nextn`) and embd (`result_norm` = `t_embd`) are **whole-tensor** graph nodes — they always cover every token (layers run for all), independent of output[]. Only logits (LM head) is pruned by n_outputs. The prefill export (`prefill-export-llama.patch`) therefore captures all-token embd/hidden even with output[] = last-token-only; `--logits-all` / setting all output[] is only needed for all-token **logits**.

## 3. Patch directly, or phase-1 anchor + frag? (2026-10-09 revised)

Rule of thumb: **pure insertions and new code go through phase-1 include-anchors + main-repo frags; only edits to existing logic lines stay as direct patch hunks.**

### Phase-1 anchor + frag (insertions / new code, either feature)

- **prefill export (2b patch eliminated 2026-10-09)**: `src/llama-context.cpp` (~400-line export body + 9 hook sites), `src/llama-context.h` (`export_*` members), `src/llama-kv-cache.h/.cpp` (`get_v_storage`), `tools/server/server.cpp` (prefill-only + /shutdown) — all phase-1 anchors + `patches/prefill-export/common/*.frag`, gated by `#ifdef STREAM_MOE_PREFILL_EXPORT`.
- **route-b hooks (2a slimmed 2026-10-09)**: `common/arg.cpp` / `common/preset.cpp` (option tracking), `common/speculative.cpp/.h` (draft pool + stats), `src/llama-context.cpp` (`route_b_begin_graph()`), `common/CMakeLists.txt` (sources moved to `src/cmake/stmoe_routeb_sources.cmake`) — all phase-1 anchors + frags.
- Shared structs both features inject fields into (`common_params`, `llama.h`) keep the original anchor + per-feature frag scheme.

### Direct patch edit (modifying existing logic lines — 2a remainder)

- `src/llama-model-loader.cpp/.h` (bounds-check skip, route-B load branch), `src/llama-model.cpp/.h` (logical/physical device split, dense placement), `src/llama-kv-cache*.cpp` + `llama-memory-recurrent.cpp` (`dev_layer_physical` + no-kv-offload rejection), `src/llama.cpp` (timer). A frag can only shrink these to 1-line anchor calls, not eliminate the edit — so they stay in `route-b-inject.patch`.
- `llama-kv-cache.cpp` is the single file shared by two patches (phase-1 `get_v_storage` anchor + route-b ctor logic, disjoint hunks, fixed order macros -> route-b).

### Shared-struct pattern (example: `common_params`)

- **Phase 1 (`streammoe-macros.patch`)** adds only anchor `#include`s to the shared struct:

  ```cpp
  struct common_params {
  #ifdef STREAM_MOE_PREFILL_EXPORT
  #include "stmoe_prefill_common_params.frag"
  #endif
      int32_t n_predict = -1;
  #ifdef STREAM_MOE_ROUTE_B
  #include "stmoe_routeb_common_params.frag"
  #endif
      ...
  };
  ```

- Features only ADD `.frag` files — they never edit the shared file again.
- Macros are defined at **compile time** by `build.bat llamalibs <tag>`: `main` -> `-DSTREAM_MOE_ROUTE_B`; `upstream_dump` -> `-DSTREAM_MOE_PREFILL_EXPORT`; `StreamMoE_dump` -> both; undefined macro -> the include line is skipped by the preprocessor (phase-1 alone compiles as pure upstream).

### Decision checklist (2026-10-09)

1. Pure insertion / new function / new field -> **phase-1 anchor + frag** (no direct vendored code).
2. Modifies an existing logic line (branch condition, call target, bounds check) -> **direct patch hunk** in `route-b-inject.patch` (`#ifdef STREAM_MOE_ROUTE_B` gate so other-tag builds stay upstream-identical).
3. New file shared by two patches (only `llama-kv-cache.cpp` today) -> split hunks per patch on regeneration, fixed order macros -> route-b.
