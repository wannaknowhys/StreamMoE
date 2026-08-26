# StreamMoE 代码库审计 (CODEBASE_AUDIT.md)

> 日期：2026-08-26。范围：src/ 全部 + tests/。
> 分类口径（四类）：
> - **MOCK**：伪造行为（公式/硬编码/sleep），必须替换，不能复用。
> - **可用**：真实代码，在当前真实推理路径中活跃使用。
> - **架构废弃**：真实代码，但其所处的架构槽位已被新架构取代，模块整体不再接入；内部可能有可借鉴算法。
> - **架构变了但函数本身没问题**：函数/结构本身成立且可复用，但周边架构变了，需要改接线/适配，而非重写。

---

## 1. 文件夹层

| 目录 | 判定 | 说明 |
|---|---|---|
| `src/common/` | ✅ 可用 | types.h（对齐/内存发现/线程探测）、logger.h，全部活跃使用 |
| `src/engine/` | 🟡 混合 | llama_engine 可用（真实推理核心）；subgraph_executor / speculative_engine / state_machine 为 MOCK |
| `src/io/` | ✅ 可用 | async_dio（Win IOCP 真异步/POSIX 同步 pread）、staging_reader（扇区对齐读计划）|
| `src/loader/` | ✅ 可用 | moe_loader：拓扑解析 + 专家发现 + read plan（B19/B20 已修）|
| `src/pool/` | 🟡 混合 | expert_stats（EST1）可用；expert_pool 存储/控制面架构废弃（Backend.md 取代），驱逐算法可借鉴 |
| `src/scheduler/` | ⚠️ 架构废弃 | moe_scheduler 单 worker + queue 被 Backend.md 调度线程（决策快循环 + IOCP）取代 |
| `src/kv/` | ⚠️ 架构废弃 | kv_cache_manager 是纯存储/快照，真实 KV 已由 libllama 管理；SMKV 快照格式可作适配参考 |
| `src/tokenizer/` | ⚠️ 架构废弃 | 贪心匹配（B01），已由 libllama tokenizer + chat template 取代 |
| `src/profile/` | ✅ 可用 | profiler（RDTSCP + JSONL）活跃使用；turn_profile_t 部分字段为 mock 时代遗留 |
| `src/server/` | ✅ 可用 | http_server（真实 OpenAI 兼容 + SSE）+ server_main，活跃使用 |
| `src/main.cpp` | ✅ 可用 | 真实 CLI 入口（llama_engine）|

## 2. 文件层

| 文件 | 判定 | 备注 |
|---|---|---|
| `engine/llama_engine.h/.cpp` | ✅ 可用 | 新增；chat template / KV 前缀复用 / 采样链 / 流式回调 |
| `server/http_server.h/.cpp` | ✅ 可用 | 重写；nlohmann JSON、SSE、真实遥测 |
| `server_main.cpp` / `main.cpp` | ✅ 可用 | 重写；参数映射（-ngl / --kv-placement / --mlock / 采样）|
| `io/async_dio_win.cpp` | ✅ 可用 | IOCP 真异步；route B 调度线程直接复用 |
| `io/async_dio_posix.cpp` | ✅ 可用(待升级) | 同步 pread（B22：io_uring 未实现），completed_queue 已加锁（B23）|
| `io/staging_reader.h/.cpp` | 🟡 架构变了但函数没问题 | 扇区对齐数学/plan 结构成立；`slot_offset`/紧凑槽目标布局需适配 route B 的"每 tensor 每专家"布局 |
| `loader/moe_loader.h/.cpp` | 🟡 架构变了但函数没问题 | 专家发现/切片校验成立；read plan 目标偏移需按 route B 布局重写 |
| `pool/expert_stats.h/.cpp` | ✅ 可用 | EST1（B27-29 已修：读时归一化）；route B 热度追踪直接复用 |
| `pool/expert_pool.h/.cpp` | ⚠️ 架构废弃 | 存储（独立 VirtualAlloc 槽）+ 控制面（mutex/flags）被 Backend.md slot_meta/expert_directory/MPSC 取代；LRU/LFU/EST1 驱逐打分可作算法参考 |
| `scheduler/moe_scheduler.h/.cpp` | ⚠️ 架构废弃 | route/wait/release 语义被 graph_compute 内部就绪等待取代；单 worker 队列被调度线程取代 |
| `engine/subgraph_executor.h/.cpp` | 🔴 MOCK | 标量乘 mock；**接口概念（按槽指针重绑定的 expert 计算）正是 route B mini-graph 委托的雏形**，但实现 100% 弃用 |
| `engine/speculative_engine.h/.cpp` | 🔴 MOCK | 文件存在=已加载；公式路由；硬编码遥测；greedy 匹配。由 libllama draft 模型支持取代 |
| `engine/state_machine.h/.cpp` | 🔴 MOCK(逻辑可救) | 输入遥测硬编码（cpu=0.70 等）；状态分类/策略表逻辑本身成立，接真实遥测后可复用（架构变了但函数没问题）|
| `kv/kv_cache_manager.h/.cpp` | ⚠️ 架构废弃 | SMKV 快照；可适配 libllama `llama_state_seq_save_file` 等序列化接口，但当前不接入 |
| `tokenizer/tokenizer.h/.cpp` | 🔴 MOCK + 架构废弃 | BPE 未实现（贪心匹配），由 libllama 取代 |
| `profile/profiler.h/.cpp` | ✅ 可用 | `turn_profile_t` 中 gpu_hits/ram_hits/spec_accept_hist/t_expert_* 为 mock 时代遗留，现多为 0；route B 后可接真实专家命中/IO 计时 |
| `common/types.h` / `logger.h` | ✅ 可用 | 无异议 |

## 3. 关键结构定义 / 函数级

| 结构/函数 | 判定 | 说明 |
|---|---|---|
| `aio_req_t` / `async_dio_engine`（submit_batch/wait_events）| ✅ 可用 | 抽象正确，route B 调度线程直接复用 |
| `expert_read_plan_t` / `build_expert_read_plan` / `read_expert_sync` | 🟡 函数没问题，适配布局 | 4KB 对齐数学、staging 大小计算成立；`tensor_slice_read_t.copy_dst_offset` 按紧凑槽布局，route B 改为"每 tensor 大区域内 `e*slice` 偏移" |
| `expert_info_t` / `sub_tensor_info_t` / `moe_loader::parse_gguf_topology` | 🟡 函数没问题，适配布局 | 专家发现/同质校验成立；目标偏移适配 route B |
| `expert_slot_t` / `find_slot` / `pin_slot` / `allocate_or_evict_slot` | ⚠️ 架构废弃 | 被 Backend.md 64 位原子字（state/refcount/generation）+ expert_directory + CAS 取代 |
| `get_adaptive_frequency` / `record_access` / `notify_tokens_generated` | ✅ 可用 | EST1 语义已修，route B 直接复用 |
| `route_and_prefetch` / `wait_miss_ready` / `release_layer_slots` | ⚠️ 架构废弃 | 就绪等待/释放语义被 graph_compute 内 pin/unpin（§4.8 角色式）取代 |
| `compute_expert_rebind` / `compute_batch_rebind` | 🔴 MOCK | 标量乘；概念上被 route B mini-graph 委托取代 |
| `llama_engine::chat` / `decode_tokens` / `init` | ✅ 可用 | 真实推理主路径 |
| `turn_profile_t` | 🟡 字段过时 | 保留 JSONL 骨架；删/改 mock 字段，route B 补真实命中/IO 计时 |
| `smkv_header_t` / `save_snapshot` / `load_snapshot` | ⚠️ 架构废弃 | 适配 libllama state 序列化或删除 |

## 4. tests/

| 文件 | 判定 |
|---|---|
| `test_async_dio.cpp` | ✅ 可用（测真实 IOCP）|
| `test_expert_pool.cpp` | 🟡 测已废弃模块（pool）；驱逐算法若保留可参考 |
| `test_moe_loader.cpp` | ✅ 可用（拓扑解析）|
| `test_kv_cache.cpp` | ⚠️ 测已废弃模块 |
| `test_profiler.cpp` | ✅ 可用 |
| `test_scheduler.cpp` | ⚠️ 测已废弃模块（且 B31：miss 路径无真实 IO）|
| `test_state_machine.cpp` | ⚠️ 测 mock 模块 |
| `test_tokenizer.cpp` | ⚠️ 测已废弃模块（B31：断言弱）|

## 5. 结论摘要

- **活跃真实路径**：engine/llama_engine + server/* + main.cpp + profile + common + (loader/io 待适配)。
- **route B 直接复用**：async_dio（Win）、expert_stats、loader/staging_reader 的扇区数学（需改目标布局）。
- **必须弃用**：subgraph_executor（mock 实现）、speculative_engine、state_machine（输入 mock）、tokenizer、kv_cache_manager、moe_scheduler、expert_pool 的存储/控制面。
- **可救**：state_machine 的策略表（接真实遥测）、SMKV 快照（适配 libllama state）、expert_pool 驱逐打分、subgraph_executor 的"槽指针重绑定"概念。
