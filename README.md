[English](README.md) | [简体中文](README.zh-CN.md)

# StreamMoE (OffloadMoE)

**StreamMoE** is a high-performance, memory-optimized Mixture-of-Experts (MoE) inference engine designed to execute massive MoE models (e.g. 100GB+ DeepSeek-V4 / Qwen2-MoE / Mixtral) in physical RAM/VRAM-constrained environments (e.g., 32GB RAM / 8GB VRAM).

---

## Key Features

1. **Extreme Memory Offload**: Run 150GB+ models on 4GB-32GB RAM systems via Sector-Aligned Direct I/O (DIO) and Non-Inclusive Non-Exclusive Cache (NINEC) pools.
2. **Instant Engine Startup (< 0.2s)**: Zero heavy upfront weight loading (`--moe-preload none`).
3. **Dual-Thread Overlapping Pipeline**: Compute thread immediately executes Hit Expert GEMM while Scheduler thread asynchronously streams Miss Experts via IOCP / `io_uring`.
4. **Adaptive Frequency Cache Eviction (`EST1`)**: Combines LRU with exponential decay moving averages towards recent routing hotspots.
5. **Zero-Overhead Subgraph Pointer Rebinding**: Pre-allocated static computation graphs with in-place pointer swapping.
6. **5D Adaptive Resource State Machine**: Dynamic self-tuning across CPU, GPU, PCIe, Disk, and Speculative Decoding yield with automatic Thrashing Emergency Reset.

---

## Build & Test Instructions

### Prerequisites
- Clang / LLVM (or MSVC / GCC) supporting C++17 and OpenMP
- Windows (PowerShell / `cmd`) or Linux / POSIX

### Windows Build
```powershell
# Build the main CLI executable (bin\stream_moe.exe)
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

```bash
# Run 150GB DeepSeek-V4 MoE with a 4GB Pinned Host RAM pool budget and speculative decoding
bin/stream_moe \
    -m "path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" \
    --draft-model "path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" \
    --moe-ram-pool 4096 \
    --moe-preload none \
    -p "Explain the physics behind MoE sparse routing." \
    -n 64
```

### Command Line Options
| Option | Description | Default |
| :--- | :--- | :--- |
| `-m, --model <path>` | Path to primary GGUF model (single or multi-shard `-00001-of-00005.gguf`) | *Required* |
| `--draft-model <path>` | Path to Dense draft model for speculative decoding | None |
| `-ngl, --gpu-layers <N>` | Number of Dense backbone layers to offload to GPU | `0` |
| `--moe-vram-pool <MB>` | Pinned VRAM MoE Expert Cache Pool size in MB | `4096` |
| `--moe-ram-pool <MB>` | Pinned Host RAM MoE Expert Cache Pool size in MB | `8192` |
| `--moe-preload <policy>` | Preload policy (`none`, `ram`, `vram`, `all`) | `none` |
| `--stats-file <path>` | Path to `EST1` expert usage statistics file | `<model_name>.bin` |
| `-n, --n-predict <N>` | Maximum tokens to generate | `32` |
| `-t, --threads <N>` | CPU worker threads for GEMM kernels | `16` |

---

## Architecture Overview

```
[ GGUF Multi-Shard Storage ]
        |
        v (Direct I/O Sector-Aligned Stream)
[ Async DIO IOCP / io_uring Engine ]
        |
        v (Payload Staging memcpy)
[ Pinned RAM Pool (VirtualAlloc/VirtualLock) ] <---> [ Pinned VRAM Pool (Vulkan/CUDA) ]
        |                                                     |
        +------------------+----------------------------------+
                           |
          [ Router Hit / Miss Partitioning ]
          /                                \
[ Hit: Immediate GEMM ]            [ Miss: Async Scheduler Fetch ]
          \                                /
           +---------> [ Layer Output Reduction ]
```

---

## License
MIT License.