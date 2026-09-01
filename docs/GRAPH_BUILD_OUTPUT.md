# Build Graph Entry Points & output[] Control

> [English](GRAPH_BUILD_OUTPUT.md) | [简体中文](GRAPH_BUILD_OUTPUT.zh-CN.md)
>
> Line numbers below are for the **clean upstream `f280b2698`** (vendored working tree with no patches applied). Applying phase-1 / prefill / route-b patches shifts them; search by symbol, not line number.

## 1. The three places that build the graph

`model.build_graph()` is called from exactly three sites, all in `src/llama-context.cpp`:

| # | Site (file:line) | Upper callers | Purpose | Graph built |
|---|---|---|---|---|
| 1 | `graph_reserve` : `src/llama-context.cpp:2431` | `sched_reserve()` (581) -> `graph_reserve` at 633 (PP worst-case), 653 (TG), 668 (PP again); `resolve_fused_ops` (513, fused-op probe); post-memory-update (830) | **Reserve** worst-case graphs to size sched splits / compute buffers; **never executed** | Full-size PP graph (n_tokens = min(n_ctx,n_ubatch)) + single-seq TG graph |
| 2 | `process_ubatch` : `src/llama-context.cpp:1358` | `llama_decode` main loop (1816, once per ubatch); `llama_encode` (1463) | **Real inference** graph. First checks `can_reuse` (1339): if the params match the previous graph it reuses, otherwise rebuilds | Actual ubatch shape (prefill = many tokens, decode = n_seqs tokens) |
| 3 | `llama_encode` direct path : `src/llama-context.cpp:3418` | `llama_encode` API | encode/embedding path, **forced** `res->reset()` + build every ubatch (no reuse), own compute context | Actual ubatch graph |

Both prefill and decode run through **entry #2** (`process_ubatch`). They differ only by the ubatch parameters (n_tokens / n_seqs / n_outputs), never by a distinct prefill-vs-decode entry.

## 2. Where output[] (batch.logits / ubatch.output) is set

The "which tokens get logits/embd" decision flows: server decides -> `batch.logits[]` -> `llama_decode` -> `ubatch.output[]` -> `n_outputs` -> LM-head gather.

| Layer | File:line | What it does |
|---|---|---|
| Server decides per-token output | `tools/server/server-context.cpp:151-154` (`server_batch::set_output`) — callers in update_slots / process mark the **last** token true by default | Stores `tokens[idx].output` |
| Batch render -> `batch.logits[]` | `common/common.cpp:1851` (`common_batch_add`): `batch.logits[batch.n_tokens] = logits;` — invoked from `server_batch::render()` (server-context.cpp:156) | Writes the output flag into the llama_batch |
| `--prefill-from` mode (prefill patch) | `tools/server/server.cpp`: `lg.back() = 1;` | Forces output on the last token for the prefill-from one-shot decode |
| llama_batch -> llama_ubatch | `src/llama-context.cpp` (`llama_decode` -> ubatch prep) | Copies `batch.logits` into `ubatch.output[]` |
| `n_outputs` count | `src/llama-context.cpp:1800-1811` | `n_outputs = sum(ubatch.output[i])` |
| Graph: out_ids tensor | `src/llama-graph.cpp:2425-2444` (`build_inp_out_ids`), `199-224` (`llm_graph_input_out_ids::set_input` collects indices where `ubatch.output[i]`) | LM head only computes the `n_outputs` output rows |

Note: hidden (`t_h_nextn`) and embd (`result_norm` = `t_embd`) are **whole-tensor** graph nodes — they always cover every token (layers run for all), independent of output[]. Only logits (LM head) is pruned by n_outputs. The prefill export (`prefill-export-llama.patch`) therefore captures all-token embd/hidden even with output[] = last-token-only; `--logits-all` / setting all output[] is only needed for all-token **logits**.

## 3. Patch directly, or phase-1 macro wrap?

Rule of thumb: **only shared structures that BOTH feature patches modify need the phase-1 include-anchor scheme.** Everything else is edited directly in the feature patch.

### Direct edit in the feature patch (one owner)
- **prefill (`prefill-export-llama.patch`)** owns: `src/llama-context.cpp` (cb_eval swap, export_t_* publish, capture_* , export_token_seq, dtor flush), `src/llama-context.h` (export_* members), `src/llama-kv-cache.h/.cpp`, `tools/server/server.cpp`, `tools/server/server-context.cpp` (export_dir mapping). Prefill-specific code inside those is wrapped in `#ifdef STREAM_MOE_PREFILL_EXPORT` so a route-B-only build compiles nothing extra.
- **route-b (`route-b-inject.patch`)** owns: `common/speculative.cpp/.h`, `src/llama-model-loader.cpp`, `src/llama-model.cpp`, `src/llama.cpp`, `tools/server/server-context.cpp` (route_b_setup injection), `common/CMakeLists.txt`. `llama-context.cpp` is **not** touched by route-b today — no overlap with prefill there.
- The three build entry points and the output[]/n_outputs/out_ids logic above are all `llama-context.cpp` / `llama-graph.cpp` / `server-context.cpp` / `common.cpp` — currently owned by prefill only, so hooking them is a **direct patch edit** (wrapped in `#ifdef STREAM_MOE_PREFILL_EXPORT`), no phase-1 needed.

### Phase-1 macro wrap (BOTH patches touch the same shared struct)
The only true shared-structure collision is `common_params` (declared in `common/common.h`, parsed in `common/common.cpp` / `common/arg.cpp`, `llama.h` params) — both features add fields:

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
- **Feature patches only ADD `.frag` files** (`common/stmoe_routeb_*.frag`, `common/stmoe_prefill_*.frag`, `include/stmoe_prefill_llama_*.frag`) — they never edit `common.h/common.cpp/arg.cpp/llama.h` again, so apply order between phase-2a/2b is irrelevant and patches never conflict.
- Macros are defined at **compile time** by `build.bat llamalibs <tag>`: `main` -> `-DSTREAM_MOE_ROUTE_B`; `upstream_dump` -> `-DSTREAM_MOE_PREFILL_EXPORT`; `StreamMoE_dump` -> both; undefined macro -> the include line is skipped by the preprocessor (phase-1 alone compiles as pure upstream).

### Decision checklist
1. Does the file/site get edited by **both** route-b and prefill?
   - Yes, and it is a shared struct/header both inject fields into (common_params / llama.h) -> **phase-1 anchor + feature .frag**.
   - Yes, in disjoint locations (e.g. llama-context.cpp) -> each feature patch edits its own region directly, `#ifdef`-gated; still keep apply order phase-2b vs 2a independent by not overlapping hunks.
2. Owned by exactly one feature -> **direct patch edit** (`#ifdef STREAM_MOE_*` gate recommended so other-tag builds stay byte-identical to upstream).
