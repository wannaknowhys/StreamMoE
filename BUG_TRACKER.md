# StreamMoE Bug 追踪清单 (BUG_TRACKER.md)

> 来源：`Bugs.txt` 外部审计 + 2026-08-25 对当前 main 分支源码逐条复核（全部属实，已标注 file:line）。
> 状态图例：`OPEN` 未修 / `FIXED` 已修并提交 / `WONTFIX` 有意保留 / `GONE` 由架构替换消除。

---

## P0 推理真实性（generation 为模拟，输出不可信）

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B01 | OPEN | tokenize() 是贪心最长匹配，未执行 BPE merge；`bpe_ranks_` 为死代码，prompt token IDs 错误 | src/tokenizer/tokenizer.cpp:154-194 | Phase A: libllama 内置 tokenizer+chat template 替代 |
| B02 | OPEN | 输出 token ID = `(step*17+100) % vocab_size` 公式伪造 | src/main.cpp:191 | Phase A: 真实 logits 采样 |
| B03 | OPEN | expert routing = `(step*3+l*7+k*13) % n_expert` 公式伪造 | src/main.cpp:161 | Phase A: 模型真实 router |
| B04 | OPEN | prefill = `sleep_for(5ms)`，prefill TPS 无意义 | src/main.cpp:146 | Phase A |
| B05 | OPEN | decode 每 token `sleep_for(8ms)`，TPS 含人为延迟 | src/main.cpp:195 | Phase A |
| B06 | OPEN | subgraph_executor 是标量乘 mock：`out += in * w`，slot.raw_ptr 未参与任何计算；无 dequant/SwiGLU/gate/up/down GEMM | src/engine/subgraph_executor.cpp:58-66 | Phase A 移除主路径依赖；Phase B 以真实 MUL_MAT_ID 后端取代 |
| B07 | OPEN | 无 embedding/attention/RoPE/router/LM head/sampling 前向链 | 全仓 | Phase A |
| B08 | OPEN | KV cache 只有 storage/snapshot，与前向零集成 | src/kv/* | Phase A: libllama KV 管理（含 dsv4 压缩 KV）|
| B09 | OPEN | MLA 仅 metadata 识别，无推理实现 | src/loader/moe_loader.cpp:139-143 | Phase A: 上游 deepseek4 MLA/DSA/HC 实现 |
| B10 | OPEN | GPU 执行不存在：`prof.gpu_hits=0` 写死；无任何 GPU kernel | src/main.cpp:207 | Phase A: 参数化 -ngl/Vulkan（可选后端）|
| B11 | OPEN | speculative decoding 仅接口模拟：load_draft_model 只查文件存在；acceptance=`step%4` 公式 | src/engine/speculative_engine.cpp, src/main.cpp:185 | Phase A 后置：libllama --model-draft (dflash/MTP) |
| B12 | OPEN | 遥测硬编码 cpu_load=0.70/gpu_load=0.40/cache_hit_rate=0.85 | src/engine/state_machine.cpp 等 | Phase A: 接真实计时/计数 |
| B13 | OPEN | state machine 无遥测闭环（主循环不喂 update_telemetry） | src/main.cpp | Phase A |

## P1 正确性/资源管理 bug（独立于推理路线，需修复）

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B14 | OPEN | `--moe-preload ram/all` 未读 GGUF 即 mark_ready，空内存被当合法权重且后续 find_slot 永久 hit、永不触发磁盘读 | src/main.cpp:325-337 | 修复：preload 改为走 scheduler 真实读取路径 |
| B15 | OPEN | IO 失败后 worker 只 LOG 不复位 slot -> 永久 IO_INFLIGHT\|PIN_LOCKED，泄漏且永不驱逐，可耗尽整池 | src/scheduler/moe_scheduler.cpp:158-175 | 修复：失败时 abort_io/复位 slot 状态 |
| B16 | OPEN | wait_miss_ready 超时的 miss 不进返回列表但 slot 已 pin -> 永久 PIN_LOCKED 泄漏 | src/scheduler/moe_scheduler.cpp:100-130, src/main.cpp:171-177 | 修复：超时也返回 slot 引用供 release |
| B17 | OPEN | timeout_ms 被每个 miss 单独消耗，非整体 deadline | 同 B16 | 修复：共享 deadline |
| B18 | OPEN | VirtualLock/mlock 返回值忽略，"pinned" 只是尝试而非确认，误导 benchmark | src/pool/expert_pool.cpp:39,45 | 修复：检查并告警/降级说明 |
| B19 | OPEN | shard 缺失仅 WARN 继续，150GB 模型静默缺数据 | src/loader/moe_loader.cpp:186-190 | 修复：split.count 存在时严格校验分片完整性 |
| B20 | OPEN | expert 切片假设等大连续布局，未校验 shape/quant block 对齐 | src/loader/moe_loader.cpp:249-250 | 修复：校验 total_size % n_expert == 0 且 slice 为 block 整数倍 |
| B21 | OPEN | route_and_prefetch 对重复 expert id 会 double-pin 并重复入队 | src/scheduler/moe_scheduler.cpp:62-89 | 修复：请求内去重 |
| B22 | OPEN | POSIX "async DIO" 实为同步 pread，README 宣称 io_uring 不实（Windows IOCP 为真） | src/io/async_dio_posix.cpp | 文档修正；io_uring 另行规划 |
| B23 | OPEN | POSIX completed_queue_ 无锁非线程安全；max_in_flight_ 未使用 | src/io/async_dio_posix.cpp | 修复：加 mutex |
| B24 | OPEN | scheduler 单 worker：expert 内 batch 并发、expert 间串行，无跨 expert 流水 | src/scheduler/moe_scheduler.cpp:30-34 | Phase B 再评估（多 worker/队列深度参数化）|
| B25 | OPEN | compute/IO 重叠未接入主路径（executor 从未被 generation loop 调用） | src/main.cpp | Phase A/B |
| B26 | OPEN | `-ngl/--gpu-layers`、`--moe-vram-pool` 解析后完全未接线 | src/main.cpp:34-35 | Phase A: 映射到 libllama 参数 |

## P2 统计与测试可信度

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B27 | OPEN | hybrid 打分 freq_val 无 [0,1] 归一化，可到 ~50，LRU 项失效退化为 frequency dominance | src/pool/expert_pool.cpp:187-189, src/pool/expert_stats.cpp | 修复：归一化 adaptive frequency |
| B28 | OPEN | 历史 global_counts 初始化与 session 自增分数尺度不一致，无重归一化 | src/pool/expert_stats.cpp | 与 B27 一并修 |
| B29 | OPEN | "EMA" 实为 score *= pow(0.999,n) 再 +1 的累加衰减，语义混叠 | src/pool/expert_stats.cpp | 修复为标准 decayed counter |
| B30 | OPEN | cache hit rate 基于 fake routing，只能回答"人工访问模式下缓存表现" | src/main.cpp | Phase A 后自动转为真实 routing 数据 |
| B31 | OPEN | 测试断言过弱：tokenizer 只验 roundtrip 含子串；scheduler 测试空 shard_files 时 miss 直接 mark_ready 不做 IO；overlap 测试无时间验证 | tests/test_tokenizer.cpp, tests/test_scheduler.cpp | 修复：加入参考 token IDs 黄金用例 + 真实小 GGUF IO 用例 |
| B32 | OPEN | prompt_tokens 分母不可信（依赖 B01），污染全部 TPS 统计 | src/main.cpp:130 | Phase A |

## P3 保留资产（审计确认真实可用）

GGUF 元数据/多分片发现、per-expert offset/read plan、4KB sector staging reader、Windows IOCP DIO、VirtualAlloc+VirtualLock 槽池、pin/unpin 生命周期、LRU/LFU/EST1 驱逐骨架、scheduler 骨架、RDTSCP profiler + JSONL、HTTP server 骨架、bench_agent.js 测评线。详见 EXPERT_OFFLOAD_INTEGRATION.md。

---

## 修复批次记录

| 批次 | Commit | 内容 | 关联 Bug |
|------|--------|------|----------|
| 0 | (本文件) | 建立追踪清单 + 架构分析文档 | - |
