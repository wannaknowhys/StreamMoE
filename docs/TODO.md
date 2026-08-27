# StreamMoE 待办任务清单 (TODO.md)

> 本文档用于持续跟踪 StreamMoE 下一阶段的核心研发、淘汰调度策略与评测实验任务。
> 阶段 1/2/3 为 mock 时代任务，已随真实引擎落地而完成；描述已更新为现状（原 kv_cache_manager / SMKV 等模块已删除）。

---

## 阶段 1：长程测试基建与 KV Cache 持久化（✅ 已完成，形态已变）

- [x] **1.1 逻辑最大上下文 1,048,576 tokens**
  - `-c 1048576` 由 libllama 按需分配 KV（deepseek4 MLA latent，非静态 1M 浪费）；启动打印实际 KV Cache 内存。
- [x] **1.2 KV Cache 与 Prompt 历史**
  - 原 `kv_cache_manager` / SMKV 快照模块已删；KV 由 libllama 管理 + 跨轮前缀复用（server 多轮连续对话累计上下文）。
- [x] **1.3 10 轮长程递进式测试脚本与数据集（`benchmark/prompts/long_horizon_prompts*.jsonl`）**
  - 已落地 `scripts/run_long_horizon_test.bat [en|zh]`（server 连续对话 + `tools/bench_agent.js`）。

---

## 阶段 2：高精度零开销 Profiler 埋点与日志记录（✅ 已完成）

- [x] **2.1 热路径纳秒埋点（`__rdtscp`）** — `src/profile/profiler.cpp` 活跃使用。
- [x] **2.2 定长模板零拷贝 JSONL 序列化** — `--profile-log` 每次 request/response 各写一条；字段：`turn_id`、`tps`、`hits`、`speculative_hist`（预留）、`timings_ns`。部分字段（gpu_hits 等）为 mock 遗留，route B 后接真实命中/IO 计时。

---

## 阶段 3：缓存淘汰策略可选化（✅ 已完成，并入 route B）

- [x] **3.1 LRU / LFU / EST1 可选** — `src/pool/expert_stats.cpp`（EST1 归一化已修）+ route B `src/backend/scheduler.cpp` 驱逐策略。
- [x] **3.2 冷启动测试** — 策略模拟器 `tools/simulate_cache.js`（LRU/LFU/EST1/OPT × 池大小曲线）已落地（见 `docs/EXPERT_TRACE_SIMULATION.md`）。

---

## 阶段 5：route B 完善与启动体验（2026-08-27 增补）

- [ ] **彻底杜绝 mmap 的边界确认**：dense 权重 162GB > 128GB RAM，物理装载不可行——mmap 是 dense 唯一现实路径（专家已 100% 走池）。记录设计边界到 RESEARCH/README（澄清"杜绝 mmap"= 专家杜绝，dense 保持 mmap + OS 页缓存）。
- [ ] **不预加载参数**：`--moe-preload` 是 mock 时代参数（真实引擎无）。route B 专家按需 DIO 已实现"不预加载"；如需预热选项（`--moe-warmup <topk热专家>`）再议。
- [ ] **deepseek 快速启动优化**：N: USB-NVMe 冷盘是首启瓶颈（5-10 分钟）。候选：专家热度预热（复用 simulate_cache 结果）、DIO 并行装载、`--mlock`（dense 部分，RAM 不够则免）、或提示权重放快盘。
- [ ] **短程测试统一 `-c 8192`**（小模型 gemma/Qwen），见 `docs/SMOKE_TESTING.md`。
- [ ] **KV 集合（multi-replica）**：`--kv-placement RAM,VRAM0,VRAM1` 语法已支持，但**只走第一个元素**（多元素 warning）。多副本镜像需 Phase B（多后端并行 attention）才合理——写放大 N×、读单份，纯镜像无收益；推荐实现路径为**按层分片/分层**（hot 层 VRAM，冷层 RAM，每层一份）。参数落地见 common.cpp `common_context_params_to_llama`。

---

## 阶段 4：建议的深入基准测试矩阵（待办）

- [ ] **4.1 专家池容量扩展与命中率饱和曲线 (Cache Sizing & Hit Saturation)**
  - 用 `simulate_cache.js` 扫池大小 → 命中率曲线（模拟器已就绪，待长历史数据）。
- [ ] **4.2 KV Cache 存放介质实测 (RAM vs VRAM under 4k / 32k / 200k Context)**
  - `--kv-placement ram` 可用；`vram` 待 Vulkan 后端接入。
- [ ] **4.3 异步 Direct I/O 队列深度与并发吞吐 (NVMe QD Sweep)**
  - IOCP 并发队列深度 QD = 1, 4, 16, 64。
- [ ] **4.4 投机推理步长与接受率实测（B11，draft 未实现）**
  - 配合草稿模型 `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`（在 N:\AI_LLM\DeepSeek-V4-Flash-0731\）评估加速比。
- [ ] **4.5 4KB 扇区对齐零拷贝 (Zero-Copy DMA vs Staging Buffer)**
  - 对比标准 GGUF 与 4KB 扇区对齐 GGUF 的直读性能（`stream_moe_convert` 规划）。
- [ ] **4.6 多 Agent 并发多会话槽位共享压力测试 (Multi-Slot Concurrency)**
  - 4~16 个并发对话流复用专家池时的局部性放大效应。
