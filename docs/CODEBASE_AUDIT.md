# StreamMoE 代码库审计 (CODEBASE_AUDIT.md)

> 日期：2026-08-26（原始审计）；2026-08-27 更新为仓库重组后视图（mock/废弃模块已删除，route B backend 落地）。
> 分类口径（四类）：
> - **MOCK**：伪造行为（公式/硬编码/sleep），必须替换，不能复用。
> - **可用**：真实代码，在当前真实推理路径中活跃使用。
> - **架构废弃（已删）**：真实代码，但其所处的架构槽位已被新架构取代，模块整体已删除；算法思想保留在本文档供参考。
> - **架构变了但函数本身没问题**：函数/结构本身成立且可复用，但周边架构变了，需要改接线/适配，而非重写。

---

## 1. 文件夹层（现状）

| 目录 | 判定 | 说明 |
|---|---|---|
| `src/common/` | ✅ 可用 | types.h（对齐/内存发现/线程探测）、logger.h，全部活跃使用 |
| `src/engine/` | ✅ 可用 | llama_engine 真实推理核心（唯一模块；mock 时代 subgraph_executor / speculative_engine / state_machine 已删）|
| `src/backend/` | ✅ 可用（route B）| 自定义 ggml backend：moe_backend（buft/backend 注册）、minigraph_exec（MUL_MAT_ID 委托 + MOE_ID_AT）、scheduler（槽控制面/DIO/EST1）|
| `src/io/` | ✅ 可用 | async_dio（Win IOCP 真异步/POSIX 同步 pread）、staging_reader（扇区对齐读计划）|
| `src/loader/` | ✅ 可用 | moe_loader：拓扑解析 + 专家发现 + read plan（B19/B20 已修）|
| `src/pool/` | ✅ 可用 | expert_stats（EST1，B27-29 已修）；expert_pool 存储/控制面已删（route B 槽池取代）|
| `src/profile/` | ✅ 可用 | profiler（RDTSCP + JSONL）活跃使用；turn_profile_t 部分字段为 mock 时代遗留 |
| `src/server/` | ✅ 可用 | http_server（真实 OpenAI 兼容 + SSE + /stats）+ server_main |
| `src/main.cpp` | ✅ 可用 | 真实 CLI 入口（llama_engine）|

已删除的 mock/废弃模块（历史保留于此供参考）：`src/scheduler/`、`src/kv/`、`src/tokenizer/`、`engine/subgraph_executor`、`engine/speculative_engine`、`engine/state_machine`、`pool/expert_pool`。

## 2. 文件层（现状）

| 文件 | 判定 | 备注 |
|---|---|---|
| `engine/llama_engine.h/.cpp` | ✅ 可用 | chat template / KV 前缀复用 / 采样链 / 流式回调 |
| `backend/moe_backend.h/.cpp` | ✅ 可用 | route B：自定义 buft（轻量句柄 + no-op set_tensor）+ backend 注册，supports_op(MUL_MAT_ID/MUL_MAT) |
| `backend/minigraph_exec.h/.cpp` | ✅ 可用 | route B：MUL_MAT_ID 委托执行（官方内核 + 槽 stride 布局 + `MOE_ID_AT` 读 ids）|
| `backend/scheduler.h/.cpp` | ✅ 可用 | route B：槽控制面（slot_meta/expert_directory/refcount）、DIO 装载、EST1 驱逐、pin 生命周期（首触 pin 末触 unpin）|
| `server/http_server.h/.cpp` | ✅ 可用 | nlohmann JSON、SSE、/stats、handler 异常回 400 + crash log |
| `server_main.cpp` / `main.cpp` | ✅ 可用 | 参数映射（-ngl / --kv-placement / --mlock / 采样 / --expert-backend / --moe-ram-pool）|
| `io/async_dio_win.cpp` | ✅ 可用 | IOCP 真异步；route B 调度线程直接复用 |
| `io/async_dio_posix.cpp` | ✅ 可用(待升级) | 同步 pread（B22：io_uring 未实现），completed_queue 已加锁（B23）|
| `io/staging_reader.h/.cpp` | 🟡 架构变了但函数没问题 | 扇区对齐数学/plan 结构成立；`slot_offset` 已适配 route B 槽布局 |
| `loader/moe_loader.h/.cpp` | 🟡 架构变了但函数没问题 | 专家发现/切片校验成立；read plan 目标偏移已按 route B 布局重写 |
| `pool/expert_stats.h/.cpp` | ✅ 可用 | EST1（读时归一化）；route B 热度追踪直接复用 |
| `profile/profiler.h/.cpp` | ✅ 可用 | `turn_profile_t` 中 gpu_hits/ram_hits/spec_accept_hist 为 mock 时代遗留，现多为 0；route B 后可接真实专家命中/IO 计时 |
| `common/types.h` / `logger.h` | ✅ 可用 | 无异议 |

## 3. 已删模块的可借鉴资产（route B 已/未落地）

| 原结构/函数 | 去向 | 可借鉴 |
|---|---|---|
| `aio_req_t` / `async_dio_engine` | 保留（src/io/）| route B DIO 直接复用 |
| `expert_read_plan_t` / `build_expert_read_plan` | 保留（src/loader/）| 4KB 对齐数学、staging 大小计算成立 |
| `get_adaptive_frequency` / `record_access` / `notify_tokens_generated` | 保留（src/pool/expert_stats）| EST1 语义，route B 复用 |
| `expert_pool`（独立 VirtualAlloc 槽 + mutex 控制面）| 已删 | LRU/LFU/EST1 驱逐打分 → route B scheduler 参考 |
| `moe_scheduler`（route_and_prefetch / wait_miss_ready）| 已删 | 就绪等待/释放语义 → route B graph_compute 内 pin/unpin（§4.8 角色式）|
| `subgraph_executor`（槽指针重绑定 + 标量乘 mock）| 已删 | "按槽指针重绑定的 expert 计算"概念 → route B mini-graph 委托雏形 |
| `speculative_engine`（接口模拟）| 已删 | 由 libllama draft 模型支持取代（B11 待接）|
| `state_machine`（遥测硬编码）| 已删 | 状态分类/策略表逻辑可参考，接真实遥测后可复用 |
| `kv_cache_manager`（SMKV 快照）| 已删 | 可适配 libllama `llama_state_seq_save_file` 等序列化接口，当前不接入 |
| `tokenizer`（贪心匹配 mock）| 已删 | 由 libllama tokenizer + chat template 取代 |

## 4. tests/（现状）

| 文件 | 判定 |
|---|---|
| `test_async_dio.cpp` | ✅ 可用（测真实 IOCP）|
| `test_moe_loader.cpp` | ✅ 可用（拓扑解析）|
| `test_profiler.cpp` | ✅ 可用 |
| `test_scheduler.cpp` | ✅ 可用（route B 调度控制面，UT 2/2）|
| `test_slot.cpp` | ✅ 可用（route B 槽语义，UT 4/4）|

已删除 mock 时代模块的 UT（pool/scheduler/state_machine/tokenizer/kv）。

## 5. 结论摘要

- **活跃真实路径**：engine/llama_engine + backend（route B）+ server/* + main.cpp + profile + common + loader/io。
- **route B 直接复用**：async_dio（Win）、expert_stats、loader/staging_reader 的扇区数学。
- **已弃用（已删）**：subgraph_executor、speculative_engine、state_machine、tokenizer、kv_cache_manager、moe_scheduler、expert_pool 存储/控制面。
- **可救（算法参考）**：state_machine 策略表、SMKV 快照（适配 libllama state）、expert_pool 驱逐打分、subgraph_executor 槽重绑定概念。
