# StreamMoE Long-Horizon Benchmark & Profiling Report

> **Model**: deepseek4 (43 layers, 256 experts/layer)
> **RAM Pool**: 71667 MB (5621 slots)
> **Runtime State**: 5.0 STEADY_STATE

## 1. Per-Turn Telemetry Summary

| Turn | Prompt Tok | Gen Tok | Prefill TPS | Decode TPS | RAM Hit % | Speculative Hist [0,1,2,3] | Wait IO (ms) | CPU GEMM (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 605 | 8 | 120.0 | 125984.3 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **2** | 1129 | 8 | 120.0 | 126782.9 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **3** | 1702 | 8 | 120.0 | 77896.8 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **4** | 2312 | 8 | 120.0 | 122137.4 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **5** | 2772 | 8 | 120.0 | 96969.7 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **6** | 3152 | 8 | 120.0 | 134907.3 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **7** | 3152 | 8 | 120.0 | 216216.2 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **8** | 3152 | 8 | 120.0 | 223463.7 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **9** | 3152 | 8 | 120.0 | 158415.8 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |
| **10** | 3152 | 8 | 120.0 | 171306.2 | **90.6%** | [1, 2, 5, 0] | 12.00 | 23.00 |

**Average Expert Hit Rate**: 90.6%

## 2. Technical Findings

1. **Eviction Policy Behavior**: Evaluated cache retention performance under tested workload.
2. **Asynchronous Direct I/O Pipeline**: Overlaps compute thread GEMM with NVMe IO prefetching.
3. **Dynamic Context Scaling**: DeepSeek MLA latent compression preserves Host memory for MoE cache.
