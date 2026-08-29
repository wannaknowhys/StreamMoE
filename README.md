[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE

**StreamMoE** is a memory-bounded Mixture-of-Experts (MoE) inference engine. For
large MoE models (DeepSeek-V4 / Qwen2-MoE / Mixtral) it keeps physical memory
bounded by routing **expert weights away from `mmap`** into a compact private
slot pool (custom ggml backend handling `MUL_MAT_ID`), while dense layers keep
the default `llama.cpp` behavior.

---

## Key Features

1. **Real inference engine**: vendored libllama (deepseek4 architecture, 43
   layers / 256 experts) drives the full forward pass — MLA, DSA indexer,
   hyper-connections, Sinkhorn, hash + argsort routing, shared expert.
2. **MoE without mmap (route B)**: expert weights live in a self-managed slot
   pool backed by Direct I/O (DIO); physical memory is bounded by the pool
   budget (`--moe-ram-pool`, e.g. 70 GB on a 128 GB host). Numerically
   identical to the stock graph (verified per-element).
3. **Fast startup**: weights are `mmap`-mapped, not copied; experts are streamed
   on demand via DIO. First-token latency is dominated by cold N: disk reads,
   not model load (model load ~1.5 s).
4. **On-demand expert loading + adaptive eviction (EST1)**: hit experts compute
   immediately; miss experts are prefetched asynchronously; LRU/LFU/EST1 policy
   with bounded pool residency.
5. **OpenAI-compatible API server**: upstream `llama-server` (with the route B
   plugin injected via `--expert-backend`) - `/v1/chat/completions` (SSE
   streaming), `/v1/completions`, `/v1/models`, `/health`, `/metrics`.
6. **Interactive REPL CLI**: upstream `llama-cli` (with the route B plugin) -
   multi-turn streaming chat with exact KV cache footprint reporting.
7. **Speculative decoding (draft model)**: planned (see `docs/BUG_TRACKER.md`
   B11); the draft model file is used once implemented.

---

## Toolchain & Roadmap

Architecture comparison with upstream `llama.cpp` tools: [`docs/LLAMA_EXE_ROADMAP.md`](docs/LLAMA_EXE_ROADMAP.md).

Project layout, build sub-path pattern, and test/result archiving conventions: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

Upstream tool migration plan (route B as a plugin): [`docs/UPSTREAM_TOOLS_MIGRATION.md`](docs/UPSTREAM_TOOLS_MIGRATION.md).

| Binary | Description | Status |
| :--- | :--- | :--- |
| **`build/main/llama-build/bin/llama-cli.exe`** | Upstream CLI + route B plugin (interactive REPL / single-shot prompt) | **Ready** |
| **`build/main/llama-build/bin/llama-server.exe`** | Upstream OpenAI-compatible HTTP/SSE server + route B plugin | **Ready** |
| **`build/main/bin/stream_moe_convert.exe`** | 4KB sector-aligned zero-copy GGUF optimizer (`stream_moe_convert`) | Planned |
| **`build/main/bin/stream_moe_bench.exe`** | Multi-dimensional MoE benchmark suite | Planned |

---

## Build & Test Instructions

### Prerequisites
- Clang / LLVM (or MSVC / GCC) supporting C++17 and OpenMP
- Windows (PowerShell / `cmd`) or Linux / POSIX

### Windows Build
```powershell
# Build vendored libllama + upstream llama-cli/llama-server (route B plugin linked in)
.\build.bat llamalibs main
.\build.bat build main

# Run unit test suites
.\build.bat test

# Clean build artifacts (all tags)
.\build.bat clean
```

### Linux Build
```bash
make test
```

---

## Usage

> Both binaries are the upstream `llama-cli` / `llama-server` with the route B
> plugin injected. Enable the expert pool with `--expert-backend`; without it
> the model runs in stock llama.cpp mode. MoE models should add `--fit off
> --no-warmup` (skip the empty startup forward pass so boot does not cold-read
> experts from disk).

### 1. Interactive Multi-Turn CLI Mode
```powershell
# Launch interactive REPL with a 70 GB expert pool and 16 physical cores
build\main\llama-build\bin\llama-cli.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --expert-backend --moe-ram-pool 71680 `
    -c 8192 -t 16 `
    --fit off --no-warmup `
    -p "Hello" -i
```

### 2. OpenAI-Compatible API Server Mode
```powershell
# Start HTTP/SSE API server on port 8080
build\main\llama-build\bin\llama-server.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --expert-backend --moe-ram-pool 71680 `
    --host 127.0.0.1 --port 8080 `
    -c 8192 -t 16 `
    --fit off --no-warmup --no-webui
```

#### API Endpoints (upstream llama-server)
- `POST /v1/chat/completions` (OpenAI format, supports `"stream": true`)
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`
- `GET /metrics` / `GET /slots` (upstream observability)

---

## License
MIT License.
