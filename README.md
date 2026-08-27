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
5. **OpenAI-compatible API server (`stream_moe_server`)**: lightweight C++
   HTTP server with `/v1/chat/completions` (SSE streaming), `/v1/models`,
   `/health`, `/stats`.
6. **Interactive REPL CLI (`stream_moe`)**: multi-turn streaming chat with
   exact KV cache footprint reporting.
7. **Speculative decoding (draft model)**: planned (see `docs/BUG_TRACKER.md`
   B11); the draft model file is used once implemented.

---

## Toolchain & Roadmap

Architecture comparison with upstream `llama.cpp` tools: [`docs/LLAMA_EXE_ROADMAP.md`](docs/LLAMA_EXE_ROADMAP.md).

Project layout, build sub-path pattern, and test/result archiving conventions: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

| Binary | Description | Status |
| :--- | :--- | :--- |
| **`build/main/bin/stream_moe.exe`** | Interactive REPL CLI & Single-shot prompt runner | **Ready** |
| **`build/main/bin/stream_moe_server.exe`** | OpenAI-compatible HTTP/SSE API server | **Ready** |
| **`build/main/bin/stream_moe_bench.exe`** | Multi-dimensional MoE benchmark suite | Planned |
| **`build/main/bin/stream_moe_convert.exe`** | 4KB sector-aligned zero-copy GGUF optimizer | Planned |

---

## Build & Test Instructions

### Prerequisites
- Clang / LLVM (or MSVC / GCC) supporting C++17 and OpenMP
- Windows (PowerShell / `cmd`) or Linux / POSIX

### Windows Build
```powershell
# Build all binaries into build\main\bin\
.\build.bat build

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

### 1. Interactive Multi-Turn CLI Mode
```powershell
# Launch interactive REPL with a 70 GB expert pool and 16 physical cores
build\main\bin\stream_moe.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --moe-ram-pool 71680 `
    -c 4096 `
    -t 16 `
    -i
```

### 2. OpenAI-Compatible API Server Mode
```powershell
# Start HTTP/SSE API server on port 8080
build\main\bin\stream_moe_server.exe `
    -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --host 127.0.0.1 `
    --port 8080 `
    --moe-ram-pool 71680 `
    -t 16
```

#### API Endpoints
- `POST /v1/chat/completions` (OpenAI format, supports `"stream": true`)
- `GET /v1/models`
- `GET /health`
- `GET /stats` (real-time pool usage, hit counters)

---

## License
MIT License.
