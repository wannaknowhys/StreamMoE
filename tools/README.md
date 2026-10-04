# StreamMoE Tools & Runner Suite

[English](README.md) | [简体中文](README.zh-CN.md)

This directory contains diagnostic utilities, automated runners, benchmark harnesses, and run specifications for StreamMoE.

---

## 1. Directory Structure

```
tools/
├── run_specs/                   # 4D Cartesian execution matrix configurations
│   ├── flavors/                 # Target binary flavors and execution capabilities
│   ├── models/                  # Model configurations, weights, and training context limits
│   ├── engines/                 # Placement topology definitions (C1/C2 & MoE pools)
│   └── tasks/                   # Evaluation tasks (chat, benchmarks, prefill export)
├── run_export.js                # Core test runner executing 4D cartesian specifications
├── run_bench.js                 # Throughput and latency benchmarking harness
├── smoke_cli.js                 # Rapid CLI mixed-pool smoke verification runner
├── freshness_check.js           # Binary compilation timestamp guard vs source changes
└── ...
```

---

## 2. 4D Cartesian Run Specifications (`run_specs/`)

Runners evaluate combinations across 4 decoupled, orthogonal dimensions:

### 2.1 Flavors (`tools/run_specs/flavors/<name>.json`)
Defines the binary executable and its diagnostic/export capabilities:
- `StreamMoE.json`: Production release binary (`hasExport: false`). Optimal inference throughput.
- `StreamMoE_dump.json`: Test and regression binary with tensor export hooks (`hasExport: true`).
- `StreamMoE_latest.json`: Latest features build (`hasExport: true`).
- `StreamMoE_dump_dbg.json`: Diagnostic build with verbose logging (`hasExport: true`).
- `upstream_dump.json` / `upstream_vulkan_dump.json`: Reference stock llama.cpp baselines.

### 2.2 Models (`tools/run_specs/models/<name>.json`)
Defines model file location and training specifications:
- `model`: Short model identifier (e.g., `olmoe`, `gemma`, `deepseek`).
- `modelPath`: Model path or environment variable reference (e.g., `${SM_OLMOE}`).
- `modelCtx`: Architecture training context limit (e.g. `4096` for OLMoE, `8192` for Gemma/DeepSeek).
- `pool`: Recommended expert pool size in MB.

### 2.3 Engines (`tools/run_specs/engines/<name>.json`)
Defines placement topology and hardware allocation. **Completely decoupled from binary paths**:
- `engine`: Engine topology name (e.g., `place-cpu`, `place-c1c2-exp5`).
- `extra`: Command line placement flags (`--expert-backend`, `--moe-expert-pools`, `--dense-placement`).

### 2.4 Tasks (`tools/run_specs/tasks/<name>.json`)
Defines evaluation prompts, dialogues, and context limits:
- `input`: Task identifier (e.g., `hi`, `en`, `cn`, `bench`, `prefill3000`).
- `taskCtx`: Task-desired context size.
- `feed`: Optional chat feed (`jsonl` or `prefill`).
- Generation parameters (`temp`, `nPredict`).

---

## 3. Runners & Diagnostic Tools

### 3.1 `run_export.js`
Executes combinations across flavors, models, engines, and tasks:
```bash
node tools/run_export.js --flavors <flavor> --models <model> --engines <engine> --tasks <task>
```
- **Context auto-negotiation**: Computes `effectiveCtx = Math.min(modelCtx, taskCtx)`. Avoids context overflow warnings and multi-slot VRAM exhaustion.
- **Argument deduplication**: Atomically deduplicates flags (`-c`, `--fit`, `--temp`).
- **Conditional export protection**: `--export-dir` is passed ONLY when `flavor.hasExport === true` AND the task is a prefill test (`feed.type === 'prefill'`). Chat and generation tasks never pass `--export-dir`, avoiding expensive decode-time GPU synchronization.

### 3.2 `run_bench.js`
Evaluates cold/warm prompt processing (pp) and token generation (tg) throughput:
```bash
node tools/run_bench.js --models <spec> --engines <spec> --tasks <spec>
```

### 3.3 `smoke_cli.js`
Quick smoke test for CLI direct inference under mixed memory pools (e.g. `RAM:8192,Vulkan0:5120`):
```bash
node tools/smoke_cli.js
```
Ensures CLI inference produces readable, fluent text without server overhead.

### 3.4 `freshness_check.js`
Compares the last modified time of the target binary against `src/` and `patches/`:
```bash
node tools/freshness_check.js <path/to/binary>
```
Exits with code 1 if the binary is stale, enforcing compilation before testing.
