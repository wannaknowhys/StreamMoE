# StreamMoE 项目检查点 (CHECKPOINT.md)

> **用途**：opencode 会话上下文被压缩/重开时，先读本文件 + `docs/PROJECT_STRUCTURE.md` 恢复状态。
> **最近更新**：2026-08-30。维护者每阶段收尾更新"当前状态"与"下一步"。

---

## 1. 项目目标（一句话）

DeepSeek4 等 MoE 模型，**MoE 专家权重完全不走 mmap、走自研紧凑槽专家池**（route B，自定义 ggml backend 接管 MUL_MAT_ID），dense 保持 llama.cpp 默认；物理内存有界（池预算）。

---

## 2. 当前状态（✅ 已完成）

### 转换器重构完成（2026-08-30，统一抽象 + 哑物理服务）
- **架构**：JS 逻辑单点（`tools/stream_moe_layout.js`：`buildModel` 任意源 → 统一 Model 描述 + `writeV1/V2/V2chunk` 布局计划）→ **convertd 哑物理服务**（`tools/stream_moe_convertd.cpp`，**裸 TCP**，JSON-lines，5 原语 `open/write_meta/copy/fill/close`）。
- **5 源 × 3 目标矩阵逐字节一致**：官方分片/原版/v1/v2/v2切 → v1/v2/v2切，`scripts/verify_convert_matrix.bat <workdir>` 全 PASS（N 盘原版源 → R 盘 ramdisk）。
- 修掉的 bug：scale 统一归 dense、v1 单数 KV 不再泄漏、`computeV2Layout` 块偏移、`rangeToSegs` 条带局部偏移、BOOL 数组复制、TCP/UTF-8 行分帧。
- 文档：`docs/STREAMMOE_GGUF_FORMAT.md` §7-9 已更新。

### deepseek prefill 交叉验证（2026-08-30）
- **prefill upstream vs moe：数值一致**（token#0 最后一层 hidden/embd cos≈1，4e-9 ulp 级），**非 bit 级**（route B delegate 累加顺序差异，与 gemma 的 repack 对齐路径不同）——**不做**（用户明确 A1 忽略）。
- **专家历史导出**：upstream 侧 ids 全 0（sched 回收 ids buffer）——**已知限制，不修**（B4 忽略）。
- **prefill10000 产物保留为回归基准**（upstream 跑太慢，勿清）：`O:\1\deepseek\upstream\prefill10000\` + `O:\1\deepseek\moe\prefill10000\`。
- 验证工具：`tools/verify_prefill.js`（PREFEXP1：embd=LM head 输入/result_norm + hidden=t_h_nextn 末层 hidden + KV caches）、`tools/verify_expert_history.js`（EXPHIST1）。

### 引擎 spec（2026-08-30）
- `moe.json`/`upstream.json`：`--temp 1.0`；`moe-temp0`/`upstream-temp0`：测试专用（`--temp 0 --top-k 1`）。
- **KV 无显式 cache-type 选项**（尊重 llama 默认 f16）。
- 由 `temp/gen_engines.js` 用 node 生成（无 BOM）。

### 早期里程碑（M1-M3 / 多子池 / 零 mmap / 调度 / Vulkan）
- 构建：clang-cl + VS2026 MSVC STL + libomp，上游 `llama-cli/llama-server`（`build\main\llama-build\bin\`）。
- route B 注入（`src/server/route_b_inject.*`）：--expert-backend / 池 / tensor_buft_overrides；moe 强制 `use_mmap=false`，**零 mmap**（gemma 8G 池 WS 5.94GB / PM 12.11GB）。
- 多子池（MULTI_SUBPOOL）：专家按布局分组，预算按字节占比切分；gemma 2297 槽/8GB 正常。
- 调度第1-4步：dir 二维、打分公式、异步装载 ringbuffer、全局调度线程、共享 DIO 引擎。
- Vulkan：`GGML_VULKAN=ON` 构建 OK（RX 590 8GB，HOST_VISIBLE|DEVICE_LOCAL heap 实测）。
- repack 根因闭环：gemma gate_up 的 up 走 repack、rb 普通内核；补 rb repack 路径后与 up **bit 完全一致**（1408/1408）。

---

## 3. 你可以跑的验证

| 动作 | 命令 |
|---|---|
| 构建 | `build.bat llamalibs main` 然后 `build.bat build main` |
| 转换矩阵 | `scripts\verify_convert_matrix.bat <workdir>`（N 原版源 → R 盘，产物 ~3x 模型）|
| deepseek 交互 CLI | 双击 `scripts\run_deepseek_cli.bat` |
| gemma 冒烟 | `build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --cache-type-k q8_0 --cache-type-v q8_0 --fit off --no-warmup --no-webui` |
| hi KV 导出对比 | `node tools/run_export.js --models tools/run_specs/models/deepseek.json --engines tools/run_specs/engines/upstream-temp0.json,tools/run_specs/engines/moe-temp0.json --tasks tools/run_specs/tasks/hi.json` + `node tools/verify_prefill.js <up>/hi/prefill_export_main.bin <moe>/hi/prefill_export_main.bin` |
| 单测 | `build.bat test main` |
| ASan 构建 | 见 `docs/ASAN_BUILD.md` |

---

## 4. 下一步（TODO，按用户最新决策 2026-08-30）

**主线方向**
1. **GPU 池**（vulkan Phase B：HOST_VISIBLE DIO、EMA 放置，见 `docs/ROUTE_B_GPU_PHASE.md`）。
2. **mini-graph / 主图就地池化**（route B 计算路径重构，待聊设计）。

**转换器（C6 待评估）**
3. **v1 张量级分片**（单文件 → `-00001-of-0000N` 分片，原版可合并；反向于合分片）——**未实现，待确认是否要做**。

**明确不做**（用户 2026-08-30 决策）
- deepseek prefill 追 bit 级一致（A1）。
- 专家历史 upstream 导出修复（B4）。
- prefill 全 token 层状态验证（B5）。

**搁置/历史**（见早期 CHECKPOINT，未到期不追）
- deepseek KV 实测（8192→100k→1M）、长程回归、M4 删自研 src、--moe-preload/--moe-eviction、KV 多副本、设备后端共存。

---

## 5. 环境与坑（记住）

- **测试约定**：route B 池统一 `--moe-ram-pool 71680`（70GB）大 MoE / gemma 用 8192；短程 `-c 8192`。
- **moe 必须 `--fit off --no-warmup`**；`--no-warmup` 避免启动空跑装载。
- **R: 是 ramdisk，会掉**（曾两次消失）——矩阵基准/测试产物重启即丢；重建后需重跑。
- **O: 盘慢**——避免把 source 放 O；模型源用 N 盘；小产物可放 O。`SM_OUT_ROOT` 默认 `O:\1`。
- **prefill10000 产物勿清**——upstream 跑一次 10000-token prefill 很慢，保留作回归基准（`O:\1\deepseek\*`）。
- 模型盘 N: = USB 转接 NVMe（162GB 冷盘慢）；GPU = Radeon RX 590 8GB（Vulkan only）；RAM 128GB。
- OpenMP：`F:\Dev\LLVM\lib\libomp.lib` + `-Xclang -fopenmp`（clang-cl）。
- 编译器：clang-cl + VS2026 MSVC STL。
- 转换器：convertd 编译需 `ws2_32.lib`（TCP）；`temp/stream_moe_convertd.exe` 是编译产物（gitignored）。
- 中文文档编码：只用 write/edit；PowerShell `Set-Content` 会写 BOM/乱码。
- 文档：`docs/STREAMMOE_GGUF_FORMAT.md`（转换器/中间格式，已更新）、`docs/PROJECT_STRUCTURE.md`、`docs/UPSTREAM_TOOLS_MIGRATION.md`、`docs/ROUTE_B_GPU_PHASE.md`、`docs/VENDORED_MODIFICATIONS.md`、`docs/REPACK_DIVERGENCE_DEBUG.md`。
