# StreamMoE 项目检查点 (CHECKPOINT.md)

> **用途**：opencode 会话上下文被压缩/重开时，先读本文件 + `docs/PROJECT_STRUCTURE.md` 恢复状态。
> **最近更新**：2026-08-26。由维护者每次阶段收尾更新"当前状态"与"下一步"。

---

## 1. 项目目标（一句话）

DeepSeek4 等 MoE 模型，**MoE 专家权重完全不走 mmap、走自研紧凑槽专家池**（route B，自定义 ggml backend 接管 MUL_MAT_ID），dense 保持 llama.cpp 默认；物理内存有界（池预算）。

## 2. 当前状态（已完成 ✅）

- **真实推理引擎**（vendored libllama @ f280b2698 完整 deepseek4 前向）：`build\main\bin\stream_moe.exe`（CLI）+ `stream_moe_server.exe`（OpenAI 兼容 SSE）。
  - 双语 long-horizon 输出已验证可读（`benchmark/results/conversation_real_*.txt`）。
- **内存问题根因已定位**（INC-1/2/3，见 `docs/BUG_TRACKER.md`）：INC-1 repack 已关；INC-2 seq_id 已修；INC-3 file-backed working set 是真凶 → route B 私有池为解。
- **memwatch 变补丁**（分支已删）：`patches/`，用 `build.bat build memwatch` 出特殊版，见 `patches/README.md`。
- **仓库重组完成**：docs/ scripts/ benchmark/{prompts,results}/ patches/ build/<tag>/；废弃模块已删。见 `docs/PROJECT_STRUCTURE.md`。
- **route B 骨架 + graph_compute 委托（本轮：官方 mul_mat_id 第三条路）**：
  - 采纳第三路径：槽池是单块连续内存 + 均匀 stride → 每个原始 MUL_MAT_ID 节点直接用**官方 `ggml_mul_mat_id` 内核**执行：权重叶子 `[ne00,ne01,num_slots]` data=pool+branch_off、`nb[2]=slot_size`；ids 在私有 mini-graph 里由专家 id 翻译成槽号；b_leaf 包装主图激活；结果直写主图 dst。**数值与官方逐位一致**，删掉了 per-expert gather/scatter 重建。
  - 崩溃根因已修：`ggml_build_forward_expand` 会递归捕获主图祖先（ggml.c:7165）——旧实现 view/reshape 了主图节点把原始 MUL_MAT_ID 拖进 mini-graph，CPU backend 执行它读到 buft 的悬空 sentinel → OpenMP AV。新实现全部用**叶子包装**（op==NONE + 手动 data/nb），gf->n_nodes==1 验证无捕获。
  - 顺带修：expert buft 记账 size 余量、ACCEL buft 被 dense 选走（supports_op 只认 `_exps` 的 MUL_MAT_ID）、g_threads 接入 ggml_backend_cpu_set_n_threads、pin_expert FAILED 防御。
  - **实测状态**：`--expert-backend`（70G 池）43 层 prefill + decode 完整跑通，输出与基线逐字一致；数值等价回归（`tools/compare_trace.js`）1247 条记录逐位 IDENTICAL。冷 N: DIO 装载仍是性能瓶颈（非正确性）。
- **待办（进行中）**：① 并行化专家装载（pin 循环当前逐个阻塞，scheduler 单 worker 一次一个请求）；② 更大的池/预热页缓存；③ 长上下文/多轮命中率曲线（任务一模拟器已就绪，待长历史数据）；④ 清理/还原一次性 patch。

### 数值等价根因已解决（2026-08-26）

- **根因**：`minigraph_exec.cpp` 的 delegate 用紧凑索引 `ids_data[t*n_ids+k]` 读路由 ids，但 argsort 层（L3+）的 ids 张量 `nb[1]=1024`（稀疏，nbytes=4120），t≥1 读到 t0 行 padding 假值 → 选错专家 → 输出错。L0-2（hash）ids 紧凑（nb[1]=24）→ 正常；t0 两种布局重叠 → 永远对。
- **修复**：`MOE_ID_AT(ids,t,k)` 按 `ids->nb[1]` 真实行步长读（gate/up/down 三处统一）。
- **验证**：`--expert-backend --temp 0 -p "Say hi." -n 8` 输出 `Hi! How can I help you today`（与基线逐字一致）；`compare_trace.js` base vs moe **IDENTICAL**；完整跑通诗 prompt 24 token 与基线一致。
- 排查全记录：`docs/DEBUG_DELEGATION.md`；方法论：`docs/BUG_TRACKER.md` §0。

### 任务一/二完成（2026-08-26）

- **任务一（专家访问历史 + 策略模拟器）**：`docs/EXPERT_TRACE_SIMULATION.md`。decode 主线程遍历执行后主图 MUL_MAT_ID 节点读路由 ids（nb[1] stride），累积 `(layer, token, expert)`，析构写 `LLM_EXPORT_DIR/expert_history.bin`；`tools/simulate_cache.js` 重放历史，扫描池大小输出 LRU/LFU/EST1/OPT 命中率曲线（不重跑模型）。短对话实测：4644 条、43 层、1148 唯一专家；OPT 略优于 LRU/LFU（理论界）。
- **任务二（Prefill 交叉验证）**：`docs/PREFILL_CROSS_VALIDATION.md`。`LLM_EXPORT_DIR` 时析构导出 LM head 输入（result_norm）+ hidden（t_h_nextn）+ 最终 dsv4 KV 逐层张量；`tools/verify_prefill.js` 按 token 对齐，cos+maxAbs 双门槛对比。实测 std vs `--expert-backend` **IDENTICAL**。
- 实现均为一次性 patch：`patches/prefill-export-llama.patch`（vendored）+ `patches/prefill-export-streammoe.patch`（llama_engine 全 token output）；`LLM_EXPORT_DIR` 触发，无 env 零开销。
- 批跑：`scripts/run_prefill_verify.bat [en|zh] [n]`（jsonl 批量）；`scripts/verify_prefill.bat ["prompt"]`（单 prompt 快速验证）。

### 此前已确认（route B 骨架，未变）

- DIO 装载字节全等（layer3 全部 256 专家 + 运行时槽整段校验和 `BCFB9B786148910B`）；pin 生命周期泄漏已修（首触 pin、末触 unpin，§4.8 角色式）。
- route B 骨架模块（slot 控制面 / scheduler / backend / minigraph arena / `--expert-backend` 开关）均已编译 + UT 通过（`tests/test_slot.cpp` 4/4、`tests/test_scheduler.cpp` 2/2）。
- profiler JSONL schema 完整；expert 命中遥测等 route B 填充（scheduler 的 total_lookups/ram_hits/disk_misses 已就绪）。

## 3. 你可以跑的验证

| 动作 | 命令 |
|---|---|
| 构建（先 llamalibs 一次） | `build.bat llamalibs main` 然后 `build.bat build main` |
| 单元测试（含 slot 控制面） | `build.bat test main`（4/4 通过）|
| 单 prompt 冒烟 | `build\main\bin\stream_moe.exe -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" --moe-ram-pool 71680 -c 4096 -t 16 -p "Say hi." -n 8` |
| 整轮 long-horizon（手动盯内存） | `scripts\run_long_horizon_test.bat en` / `zh` |
| 内存诊断版 | 应用 `patches/` 后 `build.bat build memwatch`，看 `%TEMP%\memwatch_*.log` |
| 看性能 profile | `benchmark\results\profile_real_<tag>.jsonl`（每轮 prompt/gen/prefill_tps/decode_tps）|
| 看输出正确性 | `benchmark\results\conversation_real_<tag>.txt`（转写，合法 UTF-8）|
| 重生成报告 | `node tools/regenerate_report.js <tag>` |
| Prefill 交叉验证（单 prompt） | `scripts\verify_prefill.bat "Say hi."`（std vs moe 导出 + verify_prefill）|
| Prefill/专家历史批量验证（jsonl） | `scripts\run_prefill_verify.bat en 1` / `zh 1`（读 `benchmark\prompts\long_horizon_prompts*.jsonl`，需先应用导出 patch）|
| 策略模拟器（命中率曲线） | `node tools\simulate_cache.js temp\export_std\expert_history.bin 11008`（LRU/LFU/EST1/OPT × 池大小）|

## 4. 下一步（route B 实施，按序）

1. ✅ `src/backend/` 骨架（slot 控制面 + backend/buft 注册 + minigraph arena + 开关接线）——**已完成**。
2. ✅ **scheduler**：DIO 调度线程 + 有界池 + EST1 驱逐 + pin/wait_ready/unpin + 遥测——**已完成**（UT 通过）。
3. ✅ **graph_compute 委托**（官方 mul_mat_id 叶子 mini-graph）——**已完成**；数值等价回归 IDENTICAL、端到端输出与基线一致。
4. ✅ **pin 生命周期 §4.8**（首触 pin 末触 unpin）——**已完成**。
5. ✅ **数值等价回归**（`compare_trace.js` 逐层 IDENTICAL）——**已完成**。
6. ✅ **任务一（专家历史 + 策略模拟器）**、**任务二（Prefill 交叉验证）**——**已完成**（一次性 patch，见 `patches/README.md`）。
7. **并行化专家装载**：pin 循环当前逐个阻塞（scheduler 单 worker），改为多请求并发 DIO（IOCP QD）——**下一步**。
8. **长上下文命中率曲线**：用 `scripts\run_prefill_verify.bat en/zh` + `long_horizon_prompts*.jsonl` 采长历史，跑 `simulate_cache.js` 出 LRU/LFU/EST1/OPT 曲线——**下一步**。
9. **内存验证**：memwatch 版确认 PRIV ≤ 池预算（预期已达成，待复跑）。
10. Phase B：GPU 混合池（backend 抽象 cpu/vulkan/cuda 委托，unpin 挂 split 边界事件）。
11. 收尾：B11 投机解码（libllama draft）、TODO.md 基准矩阵、长程 10 轮实测；清理/还原一次性 patch。

## 5. 已记录的未来任务（设计确认后实施）

### 任务一（已完成）→ 后续：长上下文数据 + 策略对比
- 采集 + 模拟器已落地（`docs/EXPERT_TRACE_SIMULATION.md`、`tools/simulate_cache.js`，LRU/LFU/EST1/OPT）。
- 下一步：长 prompt/多轮对话采集真实专家访问历史，得到"池容量 ↔ 命中率"完整曲线，指导池预算/驱逐策略选择。

### 任务二（已完成）→ 后续：超长上下文 prefill 交叉验证
- 导出 + 验证已落地（`docs/PREFILL_CROSS_VALIDATION.md`、`tools/verify_prefill.js`，cos+maxAbs 双门槛）。
- 下一步：用超长 Agent 对话记录（大 batch prefill）验证长上下文一致性，把后续问题收敛到 Decode/调度/缓存/性能。

> 两个任务的前置（route B 跑通 + 数值等价回归）均已达成。

## 6. 关键文档速查- `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` — route B 设计（§3 路线对比、§4 实现要点、§5 阶段）
- `docs/Backend.md` — 原始架构（slot 位分配、eviction 顺序、MPSC 三信道）
- `docs/PROJECT_STRUCTURE.md` — 目录/产物规范
- `docs/BUG_TRACKER.md` — bug 清单 + INC 事故记录
- `docs/CODEBASE_AUDIT.md` — 四类划分（已删模块的算法参考）
- `docs/TEST_FLOW.md` — 测试流程铁律
- `patches/README.md` — 内存哨兵补丁用法

## 6. 环境与坑（记住）

- **测试约定（2026-08-26 起）**：所有 route B 池测试统一 `--moe-ram-pool 71680`（70GB）。不用小池子。
- 模型盘 N: = **USB 转接 NVMe**（非 iSCSI）；162GB 冷页拉取慢（decode 0.3~2 tok/s）。
- GPU = Radeon RX 590 8GB（Vulkan only，无 CUDA）。
- RAM 128GB（空闲约 99GB），70GB 池参数可行。
- OpenMP：`F:\Dev\LLVM\bin\libomp.dll` + `libomp.lib`（构建已在 build.bat/CMake 接线）。
- 中文文档编码：**只用 write/edit 工具**，严禁 PowerShell Set/Add-Content（会破坏 UTF-8）。
- 后台进程：跑批前 `taskkill /F /IM stream_moe_server.exe`。
