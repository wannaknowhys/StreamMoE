# StreamMoE 待办任务清单 (TODO.md)

> 本文档用于持续跟踪 StreamMoE 下一阶段的核心研发、淘汰调度策略与评测实验任务。

---

## 阶段 1：长程测试基建与 KV Cache 状态持久化（✅ 已完成）

- [x] **1.1 动态上下文扩容架构设计 (Dynamic Chunked KV Cache)**
  - 逻辑最大上下文上限设为 1,048,576（初始化 RoPE / YaRN 频率缩放）。
  - 物理显存/内存按 4096 tokens 为分块单位按需动态追加，避免静态 1M 浪费 48GB 内存。
- [x] **1.2 KV Cache 与 Prompt 历史状态落盘/快照模块 (`kv_cache_manager`)**
  - 设计二进制快速快照格式（`SMKV`）：支持保存/恢复 Prompt 文本历史、Token 序列与全层 KV Cache 潜在张量。
  - 实测保存耗时 **2.82 ms**，恢复耗时 **13.65 ms**，避免长程（50k~200k tokens）测试中断时重新执行漫长的 Prefill 计算。
- [x] **1.3 构造 10 轮长程递进式测试脚本与数据集 (`benchmark/long_horizon_prompts.jsonl`)**
  - 以 `AGENTS.md` 规则与 StreamMoE 架构设计为基准，构造 10 轮连续递进的高密度工程提问。

---

## 阶段 2：高精度零开销 Profiler 埋点与日志记录 (`--profile-log`)（✅ 已完成）

- [x] **2.1 热路径汇编级纳秒埋点 (`__rdtscp` / `read_timestamp_ns`)**
  - 在推理关键节点（Prefill、Prefix Match、各层 Attention、专家 IO 等待、CPU/GPU GEMM、PCIe 同步、合并计算）进行纯寄存器无锁纳秒埋点，开销 < 10ns。
- [x] **2.2 定长模板零拷贝 JSONL 序列化引擎**
  - 采用预分配固定大小 Buffer，指针直接填数，杜绝动态内存分配与字符串拼接惩罚。
  - 每次收到输入（`request_ingest`）与完成回复（`response_finish`）各写一条结构化日志。
  - 字段：`turn_id`, `tps`, `hits` (GPU/RAM/Total), `speculative_hist` (如 `[1, 2, 5, 0]`), `timings_ns`。

---

## 阶段 3：缓存淘汰策略可选化（Pure LRU vs Hybrid EST1）（✅ 已完成）

- [x] **3.1 消除先验偏差的纯 LRU 调度策略 (`--eviction-policy lru`)**
  - 支持 `HYBRID_EST1`（默认：历史频率 + 近期衰减）、`PURE_LRU`（纯零先验 LRU 剔除）、`PURE_LFU`（会话内频次）。
  - 单元测试已 100% 验证：在 `PURE_LRU` 下，高频但访问时间最老的专家被严格剔除，彻底杜绝历史频次引起的命中率偏见。
- [x] **3.2 独立 Pure-LRU 冷启动双击测试脚本 (`run_pure_lru_benchmark.bat`)**
  - 纯冷启动、不加载任何 KV Cache 历史快照、使用 `--eviction-policy lru`。
  - 完整对话文本实时落盘至 `benchmark/conversation_pure_lru.txt`，供人工比对模型生成质量。

---

## 阶段 4：建议的深入基准测试矩阵 (Recommended Benchmark Matrix)

- [ ] **4.1 专家池容量扩展与命中率饱和曲线 (Cache Sizing & Hit Saturation)**
  - 扫描 `4GB` $\to$ `16GB` $\to$ `32GB` $\to$ `70GB` 的 RAM 池，绘制命中率与 TPS 增长曲线。
- [ ] **4.2 KV Cache 存放介质实测 (RAM vs VRAM under 4k / 32k / 200k Context)**
  - 对比 `--kv-placement ram`（CPU Attention）与 `--kv-placement vram`（GPU Attention）。
- [ ] **4.3 异步 Direct I/O 队列深度与并发吞吐 (NVMe QD Sweep)**
  - 测试 IOCP 并发队列深度 $QD = 1, 4, 16, 64$。
- [ ] **4.4 投机推理步长与接受率实测 ($K=0, 1, 2, 4, 8$)**
  - 配合 Dense 草稿模型评估打字机交互与长代码生成的加速比。
- [ ] **4.5 4KB 扇区对齐零拷贝 (Zero-Copy DMA vs Staging Buffer)**
  - 对比标准 GGUF 与 4KB 扇区对齐 GGUF 的直读性能与 CPU 占用率。
- [ ] **4.6 多 Agent 并发多会话槽位共享压力测试 (Multi-Slot Concurrency)**
  - 评估 4~16 个并发对话流复用 70GB Pinned 专家池时的局部性放大效应。