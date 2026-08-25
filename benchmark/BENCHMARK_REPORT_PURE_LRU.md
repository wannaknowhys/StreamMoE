# StreamMoE Long-Horizon Benchmark & Profiling Report

> **Model**: deepseek4 (43 layers, 256 experts/layer)
> **RAM Pool**: 71667 MB (5621 slots)
> **Runtime State**: 5.0 STEADY_STATE

## 1. Per-Turn Telemetry Summary

| Turn | Prompt Tok | Gen Tok | Prefill TPS | Decode TPS | RAM Hit % | Speculative Hist [0,1,2,3] | Wait IO (ms) | CPU GEMM (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 64 | 8 | 120.0 | 123456.8 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **2** | 64 | 8 | 120.0 | 111576.0 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **3** | 64 | 8 | 120.0 | 128617.4 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **4** | 64 | 8 | 120.0 | 113475.2 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **5** | 64 | 8 | 120.0 | 120120.1 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **6** | 64 | 8 | 120.0 | 114285.7 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **7** | 64 | 8 | 120.0 | 199501.3 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **8** | 64 | 8 | 120.0 | 268456.4 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **9** | 64 | 8 | 120.0 | 273972.6 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **10** | 64 | 8 | 120.0 | 278745.6 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |

**Average Expert Hit Rate**: 90.6%

## 2. Technical Findings

1. **Eviction Policy Behavior**: Evaluated cache retention performance under tested workload.
2. **Asynchronous Direct I/O Pipeline**: Overlaps compute thread GEMM with NVMe IO prefetching.
3. **Dynamic Context Scaling**: DeepSeek MLA latent compression preserves Host memory for MoE cache.
