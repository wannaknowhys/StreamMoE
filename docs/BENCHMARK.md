# Placement Benchmark - run_bench.js

[English](BENCHMARK.md) | [简体中文](BENCHMARK.zh-CN.md)

> Status: **landed 2026-09-09**. Measures how much VRAM residency of C1
> (per-layer dense), C2 (global dense) and the expert pool improves throughput,
> singly or combined, and compares against the stock upstream llama.cpp.
>
> Related: `docs/DENSE_PLACEMENT.md` (`--dense-placement`), `docs/ROUTE_B_GPU_PHASE.md`
> (expert pools / `--moe-expert-pools`), `tools/run_export.js` (same spec model),
> `docs/TEST_FLOW.md`.

## 1. Metrics

`llama-server` reports its own `timings`; the runner reads two numbers:

- **pp** = prompt processing (prefill), `timings.prompt_per_second`.
- **tg** = token generation (decode), `timings.predicted_per_second`.

Two task kinds:

- **single**: request 1 = **cold** (includes cold expert load from disk), then
  `warmup` requests are discarded, then `repeat` requests are measured and the
  **median** is reported. Uses `/completion` with `cache_prompt=false`, so every
  repeat re-prefills the whole prompt.
- **jsonl**: chained multi-turn via `/v1/chat/completions` with
  `cache_prompt=true`. Every turn is one JSONL record + one console line; a
  final summary record carries the aggregate. `prompt_tokens` is that turn's
  **increment** (prefix is reused, visible as `cache_n`), not the cumulative
  context length.

The same methodology drives route B and the stock upstream binary.

## 2. Runner

`tools/run_bench.js` reuses the `run_specs` model/engine/task cartesian product
and `${VAR}` env expansion from `run_export.js`.

```
node tools/run_bench.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]>
                        [--port N] [--health-timeout S] [--out FILE] [--dry-run]
```

- Env: `SM_BENCH_PORT` (default 8994), `SM_BENCH_OUT` (default
  `benchmark/results/bench_<ts>.jsonl`).
- One server process per run (placement is a process-level argument).
- A failed run (health timeout, early exit, request error) is recorded as a
  `FAIL` row and the batch **continues** - this is what makes the matrix
  capacity-aware.
- Every run stores a `server_log_tail` for offload / OOM diagnosis.
- Results are appended line by line, so partial batches survive.

## 3. Spec structure

Three disjoint spec categories; any duplicate key across them aborts.

| Category | Keys |
| :------- | :--- |
| model | `model`, `modelPath`, `draft?`, `pool?` |
| engine | `engine`, `bin` \| `binPath`, `extra?` |
| task  | `input`, plus single/jsonl fields below |

Engine = binary + placement args:

- route B: `"bin": "StreamMoE"`,
  `extra = ["--expert-backend","--fit","off","--moe-expert-pools","RAM:${pool}[,Vulkan0:N]","--dense-placement","C1:<dev>,C2:<dev>"]`
- stock: `"binPath": "${SM_UPSTREAM_BIN}"`,
  `extra = ["--fit","off","--n-gpu-layers","<N>"]`

Task fields:

- single: `prompt` | `promptFile`, `promptRepeat?`, `nPredict?`, `warmup?`,
  `repeat?`, `ctx?`, `threads?`
- jsonl: `feed:{type:"jsonl", path, maxTurns}`, `nPredict?`, `ctx?`, `threads?`

## 4. Placement matrix (`tools/run_specs/engines/`)

All route-B specs use `bin: StreamMoE`; `${pool}` comes from the model spec.

| Spec | `--dense-placement` | `--moe-expert-pools` |
| :--- | :--- | :--- |
| `place-cpu.json` | `C1:RAM,C2:RAM` | `RAM:${pool}` |
| `place-c1.json` | `C1:Vulkan0,C2:RAM` | `RAM:${pool}` |
| `place-c2.json` | `C1:RAM,C2:Vulkan0` | `RAM:${pool}` |
| `place-exp2.json` | `C1:RAM,C2:RAM` | `RAM:${pool},Vulkan0:2048` |
| `place-exp5.json` | `C1:RAM,C2:RAM` | `RAM:${pool},Vulkan0:5120` |
| `place-c1-exp2.json` | `C1:Vulkan0,C2:RAM` | `RAM:${pool},Vulkan0:2048` |
| `place-c2-exp2.json` | `C1:RAM,C2:Vulkan0` | `RAM:${pool},Vulkan0:2048` |
| `place-c1c2.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool}` |
| `place-c1c2-exp2.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool},Vulkan0:2048` |
| `place-c1c2-exp1.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool},Vulkan0:1024` |

Stock upstream: `stock-cpu.json` (`-ngl 0`), `stock-vulkan.json` (`-ngl 99`),
both via `binPath: ${SM_UPSTREAM_BIN}`.

Tasks: `bench.json` (~1k-token prompt), `bench_long.json` (~4k via
`promptRepeat`), plus the existing `en.json` / `cn.json` multi-turn sets.

## 5. Private env (machine paths stay out of the repo)

`private/` is gitignored. `private/env.bat` sets model paths,
`SM_UPSTREAM_BIN` (stock Vulkan build) and `SM_BENCH_OUT`; `private/bench.bat`
loads it and calls the runner. The committed template is
`scripts/sm_env.example.bat`.

## 6. Result schema (`benchmark/results/*.jsonl`, gitignored)

- single: `{kind:"single", cold, warmup[], steady[], median:{prompt_n,prompt_tps,decode_tps}, server_log_tail}`
- jsonl: one `{kind:"turn", turn, prompt_tokens, prompt_tps, decode_tps, gen_n, cache_n}`
  per turn, then `{kind:"summary", turns, total_prompt_tokens, total_gen_tokens,
  avg_decode_tps, median_decode_tps, avg_prompt_tps, server_log_tail}`

## 7. Examples

```bat
rem stock GPU vs CPU on olmoe
private\bench.bat --models tools\run_specs\models\olmoe.json ^
  --engines tools\run_specs\engines\stock-cpu.json,tools\run_specs\engines\stock-vulkan.json ^
  --tasks tools\run_specs\tasks\bench.json

rem gemma placement: all-CPU baseline vs C1+C2+2G experts in VRAM
private\bench.bat --models tools\run_specs\models\gemma.json ^
  --engines tools\run_specs\engines\place-cpu.json,tools\run_specs\engines\place-c1c2-exp2.json ^
  --tasks tools\run_specs\tasks\bench_long.json
```

## 8. Notes and early findings

- RX590 8GB (~7GB usable): gemma C1+C2 = 2.46GB fits; deepseek C1 = 9.9GB does
  not, so `C1:Vulkan0` rows are expected to `FAIL` (recorded, batch continues).
  Run dense rows on gemma and C2/expert rows on deepseek.
- Dense on a GPU re-enables `op_offload` (~1.3GB Vulkan compute buffer); that
  counts against the same VRAM budget as the expert pool.
- 2026-09-09 smoke (olmoe, 8-token decode): stock CPU 52.5 tg vs stock Vulkan
  43.5 tg. `-lv 4` confirmed `offloaded 17/17 layers to GPU` (3961 MiB on
  Vulkan0), so the GPU path is real; a tiny decode is launch-overhead bound.
  Use `bench.json` / `bench_long.json` for representative numbers.
