[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE (OffloadMoE)

**StreamMoE** is a high-performance, memory-optimized Mixture-of-Experts (MoE) inference engine designed to execute massive MoE models (e.g. 100GB+ DeepSeek-V4 / Qwen2-MoE / Mixtral) on physically memory-constrained hardware (e.g., 32GB RAM / 8GB VRAM).

---

## Key Features

1. **Extreme Memory Offload**: Run 150GB+ models on 4GB-32GB RAM systems via Sector-Aligned Direct I/O (DIO) and Non-Inclusive Non-Exclusive Cache (NINEC) pools.
2. **Instant Engine Startup (< 0.15s)**: Zero heavy upfront weight loading (`--moe-preload none`).
3. **Dual-Thread Overlapping Pipeline**: Compute thread immediately executes Hit Expert GEMM while Scheduler thread asynchronously streams Miss Experts via IOCP / `io_uring`.
4. **Adaptive Frequency Cache Eviction (`EST1`)**: Combines LRU with exponential decay moving averages towards recent routing hotspots.
5. **Zero-Overhead Subgraph Pointer Rebinding**: Pre-allocated static computation graphs with in-place pointer swapping.
6. **5D Adaptive Resource State Machine**: Dynamic self-tuning across CPU, GPU, PCIe, Disk, and Speculative Decoding yield with automatic Thrashing Emergency Reset.
7. **OpenAI-Compatible Streaming API Server (`stream_moe_server`)**: Lightweight C++ HTTP server supporting `/v1/chat/completions` (SSE streaming), `/v1/models`, `/health`, and `/stats`.
8. **Interactive REPL CLI (`stream_moe`)**: Multi-turn conversation chat with real-time streaming output, context size, and exact KV cache footprint calculation.

---

## Toolchain & Roadmap

See [`LLAMA_EXE_ROADMAP.md`](LLAMA_EXE_ROADMAP.md) for the full architecture comparison with upstream `llama.cpp` tools (`llama-cli`, `llama-server`, `llama-bench`, `llama-quantize`).

| Binary | Description | Status |
| :--- | :--- | :--- |
| **`bin/stream_moe.exe`** | Interactive REPL CLI & Single-shot prompt runner | **Ready** |
| **`bin/stream_moe_server.exe`** | OpenAI-compatible HTTP/SSE API server | **Ready** |
| **`bin/stream_moe_bench.exe`** | Multi-dimensional MoE benchmark suite | Planned |
| **`bin/stream_moe_convert.exe`** | 4KB sector-aligned zero-copy GGUF optimizer | Planned |

---

## Build & Test Instructions

### Prerequisites
- Clang / LLVM (or MSVC / GCC) supporting C++17 and OpenMP
- Windows (PowerShell / `cmd`) or Linux / POSIX

### Windows Build
```powershell
# Build all binaries (stream_moe.exe, stream_moe_server.exe)
.\build.bat build

# Run all Phase 1-5 unit test suites
.\build.bat test

# Clean artifacts
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
# Launch interactive REPL with auto 75% available RAM pool allocation and 16 physical cores
bin\stream_moe.exe `
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --draft-model "path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" `
    --moe-ram-pool auto `
    -c 4096 `
    -t 16 `
    -i
```

### 2. OpenAI-Compatible API Server Mode
```powershell
# Start HTTP/SSE API server on port 8080
bin\stream_moe_server.exe `
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" `
    --host 127.0.0.1 `
    --port 8080 `
    --moe-ram-pool auto `
    -t 16
```

#### API Endpoints
- `POST /v1/chat/completions` (OpenAI format, supports `"stream": true`)
- `GET /v1/models`
- `GET /health`
- `GET /stats` / `GET /metrics` (real-time Cache Hit Rate, Pool usage, and state machine mode)

---

## License
MIT License.