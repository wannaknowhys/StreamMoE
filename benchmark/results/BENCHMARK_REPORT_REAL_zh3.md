# StreamMoE Long-Horizon Benchmark & Profiling Report

> **Model**: deepseek4 (43 layers, 256 experts/layer)
> **RAM Pool**: 71680 MB (5621 slots, mmap page-cache baseline)
> **Runtime State**: REAL_INFERENCE_LIBLLAMA

## 1. Per-Turn Telemetry Summary

| Turn | Prompt Tok | Gen Tok | Prefill TPS | Decode TPS | Total TPS (wall) | Truncated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 157 | 64 | 2.5 | 2.0 | 0 | no |
| **2** | 372 | 64 | 4.1 | 1.7 | 0 | no |
| **3** | 613 | 64 | 7.8 | 0.4 | 0 | no |

## 2. Technical Findings

1. **Real inference baseline**: libllama deepseek4 forward (MLA/DSA/HC), mmap page-cache expert streaming.
2. **Speed bound by model drive (N: USB-NVMe) cold page faults**; decode TPS improves as page cache warms.
3. **Next**: route B custom backend (expert pool, no mmap for MoE) - see docs/LLAMA_MOE_NO_MMAP_RESEARCH.md.

