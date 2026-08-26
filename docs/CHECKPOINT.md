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
- **route B 骨架（本轮新增，已编译+UT 通过）**：
  - `src/backend/slot.h`：跨平台控制面——`slot_meta` 64 位原子字（state/refcount/generation + CAS pin/unpin/evict/reload/failed）、`expert_directory`（二维原子数组+版本号）、有界 MPSC 分配队列；等待唤醒走 Windows WaitOnAddress / Linux futex，无忙等。UT：`tests/test_slot.cpp`（4/4 通过）。
  - `src/backend/moe_backend.h/.cpp`：自定义 ggml backend 注册（`stream_moe`，ACCEL 设备，`ggml_backend_register`），轻量 expert buft（记账句柄 + no-op set_tensor）+ host compute buft；`graph_compute` 目前**拒绝执行（GGML_STATUS_FAILED）**——mini-graph 委托未接。
  - `src/backend/minigraph.h`：scratch arena（固定 buffer ggml context，reset 复用免 malloc）+ 外部指针 2D 权重包装。
  - `src/backend/alloc.h`：跨平台对齐分配（Win `_aligned_malloc` / POSIX `posix_memalign`）。
  - `--expert-backend` 开关（main + server），`llama_engine` 接线 `tensor_buft_overrides`（pattern 覆盖 `ffn_*_exps` + `ffn_*_shexp`，模型无关；dense 无匹配自动无影响）。**默认 OFF**，开启后加载会走自定义 buft 但 decode 会因 graph_compute 未实现而失败——这是预期。
- **profiler 核对**：JSONL schema 完整（hits/speculative_hist/timings_ns 全在序列化中），每轮 2 次 flush（turn 级，低频率无性能影响）；expert 相关值等 route B 填充。

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
2. **scheduler**：DIO 调度线程（复用 `io/async_dio` + `loader/moe_loader` read plan），装载专家切片进槽内存；slot_meta 状态机驱动（EMPTY→IO_INFLIGHT→READY/FAILED）。
3. **graph_compute 实现**：MUL_MAT_ID / MUL_MAT 的 mini-graph 委托（scratch arena + 外部指针权重包装 → ggml-cpu 执行），先单专家后 k 专家批处理；就绪等待接 scheduler。
4. **pin 生命周期 §4.8**：首触 pin（非 down 角色）末触 unpin（down 角色），同 ids 自洽。
5. 开启 `--expert-backend` 端到端跑通 + **数值等价回归**（全命中/混合 vs 官方图逐元素 diff）。
6. Phase B：GPU 混合池（backend 抽象为 cpu/vulkan/cuda 委托，unpin 挂 split 边界事件）。
7. 收尾：B11 投机解码（libllama draft）、TODO.md 基准矩阵。

## 5. 关键文档速查

- `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` — route B 设计（§3 路线对比、§4 实现要点、§5 阶段）
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
