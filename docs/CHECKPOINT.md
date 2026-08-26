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
  - **实测状态**：`--expert-backend`（8GB 池）5-token prefill 已跑到第 20/43 层，gate/up/down 全算通、无误导；但**冷 N: 顺序 DIO 装载是性能瓶颈**（~70s/层 → 全程约 1h）。卡死印象 = 慢，非死锁。
- **待办（下一个会话）**：① 并行化专家装载（pin 循环当前逐个阻塞，scheduler 单 worker 一次一个请求）；② 更大的池/预热页缓存；③ 跑通完整 prefill + decode 验证正确性（对比非 expert-backend 输出）；④ 数值等价回归；⑤ 移除 [moe] 调试日志。
- **route B 骨架模块（细节，均已编译 + UT 通过）**：
  - `src/backend/slot.h`：跨平台控制面——`slot_meta` 64 位原子字（state/refcount/generation + CAS pin/unpin/evict/reload/failed）、`expert_directory`（二维原子数组+版本号）、有界 MPSC 分配队列；等待唤醒走 Windows WaitOnAddress / Linux futex，无忙等。UT：`tests/test_slot.cpp`（4/4 通过）。
  - `src/backend/scheduler.h/.cpp`：**专家调度器（本轮新增）**——固定大小提交内存池（`pool_bytes` 硬上限，初始即提交 = 70G 保证）、后台线程从 MPSC 取请求、复用 `io/async_dio` + `staging_reader::read_expert_sync` + `loader/moe_loader` read plan 装载专家切片进槽、EST1/LRU 驱逐（READY+refcount==0）、计算侧 `pin_expert`（阻塞装载+refcount++）/`wait_ready`（down 角色不 pin）/`unpin`、命中/缺失遥测（喂 profiler hits）。UT：`tests/test_scheduler.cpp`（合成拓扑 + 临时文件，装载/驱逐/遥测，2/2 通过）。
  - `src/backend/moe_backend.h/.cpp`：自定义 ggml backend 注册（`stream_moe`，ACCEL 设备）+ 轻量 expert buft（记账句柄 + no-op set_tensor）+ host compute buft；`graph_compute` **已实现**（官方 mul_mat_id 委托，见上）。
  - `src/backend/minigraph.h`（scratch arena）+ `alloc.h`（跨平台对齐分配）。
  - `--expert-backend` 开关（main + server），`llama_engine` 接线 `tensor_buft_overrides`（pattern 覆盖 `ffn_*_exps` + `ffn_*_shexp`，模型无关；dense 无匹配自动无影响）。**默认 OFF**。
- **profiler 核对**：JSONL schema 完整（hits/speculative_hist/timings_ns 全在序列化中），每轮 2 次 flush（turn 级，低频率无性能影响）；expert 相关值等 route B 填充（scheduler 的 total_lookups/ram_hits/disk_misses 已就绪）。

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

## 4. 下一步（route B 实施，按序）

1. ✅ `src/backend/` 骨架（slot 控制面 + backend/buft 注册 + minigraph arena + 开关接线）——**已完成**。
2. ✅ **scheduler**：DIO 调度线程 + 有界池 + EST1 驱逐 + pin/wait_ready/unpin + 遥测——**已完成**（UT 通过）。
3. **graph_compute 实现**：MUL_MAT_ID / MUL_MAT 的 mini-graph 委托（scratch arena + 外部指针权重包装 → ggml-cpu 执行），先单专家后 k 专家批处理；就绪等待接 scheduler（pin_expert）。**（当前主路径瓶颈）**
4. **pin 生命周期 §4.8**：首触 pin（非 down 角色）末触 unpin（down 角色），同 ids 自洽。
5. 开启 `--expert-backend` 端到端跑通 + **数值等价回归**（全命中/混合 vs 官方图逐元素 diff）。
6. **内存验证**：memwatch 版确认 PRIV ≤ 池预算 + libllama（dense/KV/图缓冲）——预期一旦 graph_compute 只从槽读、且池固定提交即达成。
7. Phase B：GPU 混合池（backend 抽象为 cpu/vulkan/cuda 委托，unpin 挂 split 边界事件）。
8. 收尾：B11 投机解码（libllama draft）、TODO.md 基准矩阵、长程 10 轮实测。

## 5. 已记录的未来任务（设计确认后实施）

### 任务一：专家访问历史采集 patch + 策略模拟器
- 目的：不改现有调度/缓存逻辑，在特殊编译版本里记录一次完整 prompt→generation 的**实际专家访问路径**（token、layer、expert ID），存成独立 trace 文件；用独立策略模拟器读该历史，**不重跑模型**即可测试不同池大小 / LRU/其他淘汰策略 / 预取策略 / 专家分布下的预期命中率，得到 cache size ↔ hit rate 关系曲线。
- 形态：独立 patch（不入主线，类似 memwatch），编译出"采集历史专用版本"。
- 前置：route B 跑通（能真实推理出路由 ids）。

### 任务二：Prefill 交叉验证基准（llama.cpp vs StreamMoE）
- 目的：以超长真实 Agent 对话记录为固定输入，在**标准 llama.cpp** 和 **StreamMoE** 上分别做大规模 Prefill（忽略首字延迟，允许大 batch），逐 token 导出 **LM Head 输入向量 + 对应 KV Cache** 到文件；独立验证程序按 token/位置对齐，算余弦相似度、最大绝对误差、MSE，定位首个明显差异的 token 和位置。
- 若两套在长上下文范围高度一致 → 证明 StreamMoE 在 Prefill 的模型加载/权重读取/专家调度/计算/KV 写入与 llama.cpp 基本一致 → 把后续问题收敛到 Decode/调度/缓存/性能。
- 形态：两个互相独立的 patch（改造 llama.cpp 导出；改造 StreamMoE 同样导出）+ 独立验证程序。
- 前置：route B 端到端跑通 + 数值等价回归。

> 注意：上述两个任务都依赖 route B 主线先跑通。当前正卡在 graph_compute mini-graph 委托的 OpenMP 崩溃（见 §4 第 3 步）。

## 6. 关键文档速查- `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` — route B 设计（§3 路线对比、§4 实现要点、§5 阶段）
- `docs/Backend.md` — 原始架构（slot 位分配、eviction 顺序、MPSC 三信道）
- `docs/PROJECT_STRUCTURE.md` — 目录/产物规范
- `docs/BUG_TRACKER.md` — bug 清单 + INC 事故记录
- `docs/CODEBASE_AUDIT.md` — 四类划分（已删模块的算法参考）
- `docs/TEST_FLOW.md` — 测试流程铁律
- `patches/README.md` — 内存哨兵补丁用法

## 6. 环境与坑（记住）

- 模型盘 N: = **USB 转接 NVMe**（非 iSCSI）；162GB 冷页拉取慢（decode 0.3~2 tok/s）。
- GPU = Radeon RX 590 8GB（Vulkan only，无 CUDA）。
- RAM 128GB（空闲约 99GB），70GB 池参数可行。
- OpenMP：`F:\Dev\LLVM\bin\libomp.dll` + `libomp.lib`（构建已在 build.bat/CMake 接线）。
- 中文文档编码：**只用 write/edit 工具**，严禁 PowerShell Set/Add-Content（会破坏 UTF-8）。
- 后台进程：跑批前 `taskkill /F /IM stream_moe_server.exe`。
