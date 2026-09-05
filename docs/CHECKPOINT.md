# StreamMoE 项目检查点 (CHECKPOINT.md)

> **用途**：opencode 会话上下文被压缩/重开时，先读本文件 + `docs/PROJECT_STRUCTURE.md` + `patches/README.md` 恢复状态。
> **最近更新**：2026-09-05（NO_VICTIM 驱逐死锁修复 + stall 兜底，5d08bb3；dbg tag 改 RelWithDebInfo）。维护者每阶段收尾更新"当前状态"与"下一步"。

---

## 1. 项目目标（一句话）

DeepSeek4 等 MoE 模型，**MoE 专家权重完全不走 mmap、走自研紧凑槽专家池**（route B，自定义 ggml backend 接管 MUL_MAT_ID），dense 保持 llama.cpp 默认；物理内存有界（池预算）。

---

## 2. 当前状态（✅ 已完成）

### M4 收编：自研主项目删除（2026-08-31）
- 删除 `src/main.cpp`、`src/server_main.cpp`、`src/engine/llama_engine.*`、`src/server/http_server.*`、`patches/prefill-export-streammoe.patch`（见 docs/UPSTREAM_TOOLS_MIGRATION.md）。
- CMakeLists 去 `stream_moe`/`stream_moe_server` 目标（保留 test_*）；build.bat/Makefile 去 `build` 子命令（保留 llamalibs/test/convertd/clean）。
- 推理/导出全走 vendored `llama-server`/`llama-cli`（route B 插件经 `src/server/route_b_inject.*` 注入）。

### vendored patch 体系（2026-09-03 重构：frag 全主仓库 + features 宏机制）
- **vendored HEAD = 纯上游 `f280b2698`**，工作区干净（5 patch 全部 apply 为工作态）。
- **features 宏机制**：`build.bat llamalibs <tag>` 传 `-DSTREAM_MOE_FEATURES`（route_b / prefill_export / route_b,prefill_export）→ vendored 根 `CMakeLists.txt` features 块全局 `add_compile_definitions` + `include_directories`（主仓库 frag 目录）。**宏不拼 CXX_FLAGS**。宏对当次构建全部 target 生效（防静默丢弃）。
- **frag 全在主仓库**（随主仓库 commit）：`patches/route-b/common/`、`patches/prefill-export/common/`、`patches/prefill-export/include/`——vendored `include/` 已清空。
- **5 个 patch**（`patches/`，phase 结构，干净 worktree apply 验证逐字节一致）：
  - **Phase 1（必选，互不依赖）**：`streammoe-macros.patch`（根 CMakeLists features 块 + 共享文件 include 锚点：arg/common.cpp/h/llama.h/server-context 3 锚点）+ `tsc_timer.patch`（[TMR] `sm_tmr`，`STREAM_MOE_TMR` env 门控）
  - **Phase 2a（可选）**：`route-b-inject.patch`（route-b 专属：common/CMakeLists STREAM_MOE_SRC + speculative + llama-model-loader.cpp/h + llama-model.cpp + llama.cpp）——**无 frag new-file、无 server-context 段**
  - **Phase 2b（可选）**：`prefill-export-llama.patch`（prefill 专属：llama-context.cpp/h + llama-kv-cache.cpp/h + server.cpp）——**无 frag new-file**
  - **`gguf-alignment.patch`**（独立）：ggml gguf.h/cpp（convertd 工具用）
- **应用顺序**：macros → tsc_timer → route-b-inject → gguf-alignment → prefill-export-llama（临时 worktree 逐字节一致验证过，21 文件）。
- **宏隔离**：无宏（features 空 / 只 phase1）= 纯上游等价（include 行预处理跳过）。编译目录：`main`→route_b；`StreamMoE`→route_b（无导出代码的旗舰对话 build，见下）；`upstream_dump`/`upstream_vulkan_dump`→prefill_export；`StreamMoE_dump`→两者；`asan`→route_b（MSVC cl，`build.bat asan`）。**GGML_VULKAN 默认 ON 的 tag**：`StreamMoE` + `upstream_vulkan_dump` + `StreamMoE_dump`（route-B 的 Vulkan0 device-pool 路径需要设备注册；`--expert-backend` **隐含 no-op-offload**，见下）；`upstream_dump`/`main` 默认 OFF。env `GGML_VULKAN=OFF` 可覆盖。
- **op_offload 与数值形态**：llama 默认 `op_offload=true`（把 host 计算自动 offload 到 device，-ngl 0 也占 Vulkan0 compute buffer ~1.3G）。`--expert-backend` 在 frag 里隐含 `--no-op-offload`（route B 拥有专家放置权，3932d33）——Vulkan0 splits=0 实测。但 **GGML_VULKAN=ON 编译本身改变数值**（CPU buft 换 Vulkan0 host buft 等 host 内存形态，gate 边界 expert-flip 级噪声）——回归按构建形态选基线：CPU-only 编对 `baseline_regression\baseline\moe_129_8192`，默认 vulkan 编对 `moe_129_8192_vk`（run_baseline.bat 首参，见该 README）。
- **当前任务追踪**：`docs/WORK_IN_PROGRESS.md`。
- **驱逐死锁修复（2026-09-05，5d08bb3）**：alloc_or_evict 改组内 ring 扫描（修 layer-0 候选集空导致的 NO_VICTIM 单核自旋 + exec 永久等 batch 死锁）+ accept_requests 进展感知 + wall-clock 2s stall 兜底 fail-settle（唤醒 exec 上抛错误）。CPU 基线回归 IDENTICAL PASS。详见 WIP N 节。
- vendored 子模块有 `backup-20260830` 分支（整理前状态，保险）。

### VRAM 数据层（2026-09，路线 A：数据层先行）
- **vram 池真驻留 + CPU 从 vram 读权重执行**（`--moe-expert-pools RAM:N,Vulkan0:N`）：分配（slot 对齐降档，seg 登记）→ host map 通道（ggml-vulkan.cpp 在 **phase1 macros.patch 的 `STREAM_MOE_ROUTE_B` 锚点** include `stmoe_routeb_vk_hostmap.frag`（函数体在主仓库）导出 `stmoe_vk_buffer_host_ptr` 返回真 vkMapMemory ptr；`get_base` 仍是假 base `0x1000` 不改）→ scheduler 槽空间并入 vram 区（subpool 变 per-(group,pool)）→ 请求优先装 vram → CPU 执行从 vram map 读权重（"reads pool 1"，IDENTICAL）→ **vram 驱逐 demote 回 RAM**（Vulkan0:1024 触发 1987 demote 仍 IDENTICAL）。
- 开发模型 = **v2 chunk**（专家独立化 direct，moe(v2) 与 v1 基线 IDENTICAL）；upstream 对照仍 v1（不认识 v2）。
- **v2 布局改造定案（2026-09，见 §4 下一步）**：v2 expert-blocks 整专家紧凑块是 vulkan 不兼容槽布局（专家 stride=expert_size）的源头。解法 = **v2 文件块内每个张量切片独立 4K 对齐** + **pool 改张量列区（struct-of-array）**。文件侧 DIO 源对齐，pool 侧槽 stride=单张量紧凑大小（vulkan 硬编码步长）。v1 sections-v1（张量分散 GGUF 原生可读）已否决——GGUF tensor offset 须紧凑单调，无法在张量内做 4K 切片 stride，writeV1 的 per-expert reflow 产出非法 GGUF（llama 加载 offset 校验失败）。
- 已知限制：执行器**单区**（同层激活集须同池）；mixed 分区执行（WIP J6）与 GPU 每-device 分区同构，GPU 阶段前铺。
- **下轮重构（2026-09 定稿，见 docs/EXPERT_MOVE_PIPELINE.md）**：directory 加宽 (L,E) 生命周期状态、删 owner_、驱逐改 (L,E) 层距、demote/r2v 异步 move worker、批记账移调度线程——解 L6b 的 vram demote 同步阻塞。**落地状态**：M1-M4 已 commit（6b6300e/dbaff8f/45b14de/fdf4982）；M5（活跃槽记账）未做；M6 部分；M7 文档未同步。
- **mixed 执行落地（2026-09）**：所有 MUL_MAT_ID 统一走 per-pool peel rounds（exec_mixed_mm），单池/混合通吃；scratch arena no_alloc（大 batch ctx 不爆）；删老单池快路径（exec_mm_vk/exec_device_chain）。129-token mixed 列级 0 BAD。
- **v2r demote DMA 提速（2026-09，docs/VRAM_DMA_MOVE.md）**：RX590 rebar host **读 0.02 GB/s**（158ms/专家）是 demote 卡死根因；transfer-queue DMA（ggml_vk_buffer_read + cached staging）**~1ms/专家**（快 100-150×）。r2v 无需改（rebar **写 8 GB/s**）。实测 4-token 64MB（238 demote）内容 0 BAD；129-token 64MB 跑通（DIO/计算主导）。
- 代码全部主仓库 src（scheduler/minigraph_exec/route_b_inject）+ 唯一 vendored = ggml-vulkan host map + dma frag（patch 记录）。细节：`docs/WORK_IN_PROGRESS.md` J/M 节。

### prefill 导出（2026-08-31，cb_eval 图内抓取 + 参数化）
- **机制**：`--export-dir <dir>` 参数替代 `LLM_EXPORT_DIR` env（llama_context_params.export_dir，`common_context_params_to_llama` 传递；`run_export.js` 已适配传参）。
- **抓取**：llama.cpp 现成的 `cparams.cb_eval` 评估回调（每个图节点算完触发）——图内抓 embd（result_norm）/ hidden（t_h_nextn）/ **top-4 logits + logsumexp**（t_logits）/ 路由 ids（MUL_MAT_ID src[2]）。**不依赖强制输出 + ids 时机在 compute 流内**（根治 sched 回收）。
- **关键**：`export_t_embd/hidden/logits` 在 **build_graph 后、compute 前**发布（回调比对）——设在 compute 后会抓不到。
- **导出文件**（析构时写 `--export-dir`）：
  - `prefill_export_main/draft.bin`（PREFEXP1）：embd + hidden + KV caches + **top-4(id,logit)+logsumexp**（向后兼容）
  - `expert_history_main/draft.bin`（EXPHIST1）：逐 (layer,token,expert) 路由
  - `tokens_id.bin`（u32 采样 token id）+ `tokens_text.txt`（detokenize）——采样在 server 层（`common_sampler`），经 `llama_export_add_token(ctx,id)` API 喂入
- **`--prefill-from <PATH>`** 专用 prefill 模式（server）：读 `.bin`（u32 数组）/ `.txt`（自动 tokenize）→ `llama_batch_get_one` + 补 logits（最后 token）→ `llama_decode` 一次 → 析构导出 → 退出（不进 HTTP server）。gemma 验证 rc=0（embd=1/hidden=21/top4=1）。
- `--export-dir` 目录不存在自动 `create_directories`。
- 调试开关：`STREAM_MOE_DBG=1` env 打印导出诊断（门控）。

### 转换器（2026-08-30 完成）
- **统一抽象**：`tools/stream_moe_layout.js`（buildModel 任意源 → Model 描述 + 写 v1/v2/v2chunk）→ convertd 哑物理服务（裸 TCP：open/write_meta/copy/fill/close）。
- **5 源 × 3 目标矩阵逐字节一致**（`scripts/verify_convert_matrix.bat <workdir>` 全 PASS；N 原版源 → R 盘）。
- 文档：`docs/STREAMMOE_GGUF_FORMAT.md` §7-9。

### repack 实证（2026-08-30，原版行为）
- 原版加载时对匹配 repack 的权重（gemma gate_up：Q4_K 3D）**全量 repack 成重排布局常驻**（285MB/层，内存 dump vs GGUF 原始 **99.4% 字节不同**）；**替换 mmap 驻留 → 净零额外内存**（WS ≈ 模型大小）。
- 触发靠 **buft 身份**（`buft == repack_buffer_type()`），数据布局 8x8 交错。route B 槽（原始字节 + plain buft）不匹配 → 普通内核。
- **gemma gate_up 走 repack 内核 vs route B 普通内核 → 单层 ulp、累积后 logits cos 0.508**（REPACK_DIVERGENCE_DEBUG.md 已验证补 repack 可 bit 一致，但生产未落地）。
- **结论**：这是"浮点累加路径差异 + 逐层放大"，非正确性 bug；deepseek Q8 不匹配 repack → 两边同内核 → 仅 ulp。

### deepseek prefill 交叉验证（2026-08-30）
- **prefill 数值一致**（token#0 hidden/embd cos≈1，4e-9 ulp），非 bit 级。
- KV（f16 默认）逐层 cos ~0.98（f16 精度 + 生成部分）。
- **prefill10000 产物保留为回归基准**（upstream 跑太慢）：`O:\1\deepseek\upstream|moe\prefill10000\`。
- **专家历史 upstream 导出（2026-08-31 修复）**：VULKAN=ON 时 prefill 大图（如 119-token）的 MUL_MAT_ID 被 sched 分配到 Vulkan0，权重名带 `Vulkan0#` 前缀，旧的 `name[0]=='b'` / `strncmp("blk.",4)` 检查失败 + `ids->data` 不可直接读（GPU 内存）→ 中间 token 专家路由全缺。**修复（3 处，已并入 prefill-export-llama.patch）**：① 条件改 `strstr(name, "_exps")` 容忍前缀；② 层号用 `strstr("blk.")` 定位；③ ids 读取改 sched `get_async`（Vulkan0 GPU 内存不可直接 ids->data）。验证：131-token prefill + 16 decode = 148 token 专家历史完整（n=71040 = 148x30x16，ids 0-127 合法），与 VULKAN=OFF 一致。

### 引擎 spec（2026-08-30）
- `moe.json`/`upstream.json`：`--temp 1.0`；`moe-temp0`/`upstream-temp0`：测试专用（temp 0 + top-k 1）。KV 无显式 cache-type（默认 f16）。`temp/gen_engines.js` 生成。

---

## 3. 你可以跑的验证

| 动作 | 命令 |
|---|---|
| route-b 完整推理 | `build.bat llamalibs main` → `build\main\llama-build\bin\llama-server.exe` |
| prefill 导出（上游基准）| `build.bat llamalibs upstream_dump` → `build\upstream_dump\llama-build\bin\llama-server.exe` |
| 完整栈导出 | `build.bat llamalibs StreamMoE_dump` → `build\StreamMoE_dump\llama-build\bin\llama-server.exe`（含 vulkan，见 §2）|
| 转换器服务 | `build.bat convertd` → `build\convertd\convertd.exe` |
| 转换矩阵 | `scripts\verify_convert_matrix.bat <workdir>`（N 原版源 → R 盘）|
| gemma 冒烟 | `build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup --no-webui` |
| prefill 导出（--export-dir）| `llama-server -m <gemma> --export-dir <dir> ...` + 喂 prompt + shutdown → 导出 prefill_export/tokens_id/tokens_text |
| prefill-from | `llama-server -m <gemma> --prefill-from <prompt.txt|tokens.bin> --export-dir <dir> -c 1024 -t 8` |
| **基线回归** | **`baseline_regression\run_baseline.bat`**（改代码+编译后跑：129-token prefill-from 三组 IDENTICAL-vs-baseline + per-token KL 报告 + kv_cos，直接出 PASS/FAIL；CPU 编对 `moe_129_8192`、默认 vulkan 编对 `moe_129_8192_vk`，见其 README）|
| vram 池驻留 | `llama-server -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --prefill-from baseline_regression\baseline\upstream_129\tokens_id.bin --export-dir <dir> -c 2048 -t 16 --expert-backend --moe-expert-pools RAM:8192,Vulkan0:4096 --fit off --no-warmup`（全量专家进 vram；池 1024 触发 demote）→ 产物对 `moe_129_8192_vk` IDENTICAL |

**run_export 前台窗口启动**（跑 cn/en/prefill10000 导出任务——脱离 opencode 管控但用户可见）：
```bat
agy-run -c "start cmd /k temp\run_export_win.bat"
```
- agy-run 绑 `WinSta0\Default` 在**交互桌面**弹新 cmd 窗口 → llama-server 在窗口里跑（**用户实时看**）。
- `start` 立即返回 → **opencode/bash 不阻塞**。别用 `start /b` 后台（那才真正脱离管控且看不到）。
- `temp\run_export_win.bat`：`call temp\sm_env.bat`（机器路径 env）+ `node tools\run_export.js --models ... --engines ... --tasks ...`。

---

## 4. 下一步（TODO）

**主线方向（2026-09 决策：vulkan 兼容布局改造 = v2 块内张量对齐 + SoA pool）**
0. **b4-3 实验线已归档**（HEAD 回退 860f9f4，存档 debug_patch/b4-3-arena-clone/）——arena-clone 整层 GPU 执行是死路：ggml-vulkan MUL_MAT_ID 专家步长硬编码 `ne0*ne1`、忽略 `nb[2]`，任何槽 stride=expert_size 的伪 3D 壳 vulkan 都不认（CPU 读 nb02 才正确）。
1. **v2 文件布局改造（核心，替代 b4-3）**：vulkan 只认"每张量一个连续区、区内专家 stride=单张量紧凑大小"。两步：
   - **文件侧（v2 转换器）**：保留 expert-blocks 块架构，但**块内每张量切片独立 4K 对齐**（gate_up 段、down 段各自起点 4K；原紧凑拼接 down 起点非 4K）。块内 offset 计算对齐到 4096；pad 空洞由 fill 补。所有专家切片源 offset 均 4K 对齐 → DIO 源对齐成立。
   - **pool 侧（SoA）**：subpool 内按张量分成列（gate_up 列、down 列……每列是某张量的专家切片序列，stride = 单张量紧凑 perExpert）。槽 = (列, 专家切片)。vulkan mm 以单张量为权重单位（列基址 + e×stride，nb[2]=perExpert）→ 天然匹配。
   - **装载**：专家 e 的 gate_up 切片 = 文件 gate_up 段 offset 一次 DIO → 池 gate_up 列。perExpert 是 4K 倍数 → DIO 直写；非 4K → DIO 读 4K 窗口 + staging move。每专家 N 次 DIO（每张量一次）。
   - 数值门：改后 w3d 壳 stride=perExpert → CPU/vulkan 同读同构造逐字节验证。
2. deepseek `--prefill-from` KV 预构建实测（gemma 非 dsv4 无 KV；deepseek 才有，验证 KV 导出 + 预构建价值）。
3. **vulkan 作为 backend dll**（`GGML_BACKEND_DL` + `BUILD_SHARED_LIBS=ON`）——**需与 moe 适配商议**。

**顺手**
4. `llama-tokenize` 一次性编译（`ninja llama-tokenize`）。
5. 3 个 direct_fill task spec（`tools/run_specs/tasks/direct_fill_cn_txt/en_txt/10000_txt.json`，引用同名 txt）。
6. **OpenSSL**（build 报 "OpenSSL not found, HTTPS support disabled"）——需要 `OPENSSL_ROOT_DIR` 提供 `libcrypto` 才能编入 HTTPS；不急但看着烦。

**明确不做**（用户决策）
- deepseek prefill 追 bit 级（A1）。
- prefill 全 token 层状态验证（B5）。
- v1 张量级分片（C6）。

---

## 5. 环境与坑（记住）

- **测试约定**：route B 池 `--moe-ram-pool 71680`（70GB）大 MoE / gemma 8192；`--fit off --no-warmup`。
- **编译目录**：`build\<tag>\llama-build\bin\`（llama-server + libomp.dll 已自动 copy）；tag 决定宏（见 §2）。
- **AVX flags**：build.bat llamalibs 显式 `-mavx2 -mfma`（5950X；clang-cl 的 ggml 指令集检测否则 Failed 落 SSE2）。
- **R: 是 ramdisk，会掉**（多次消失）——矩阵基准/测试产物重启即丢。
- **O: 盘慢**——source 避免 O；模型源用 N。`SM_OUT_ROOT` 默认 `O:\1`。
- **prefill10000 产物勿清**（回归基准）：`O:\1\deepseek\*`。
- 模型盘 N: = USB 转接 NVMe（冷盘慢）；GPU = RX 590 8GB（Vulkan only）；RAM 128GB。
- OpenMP：`F:\Dev\LLVM\lib\libomp.lib` + `-Xclang -fopenmp`（clang-cl）。
- 编译器：clang-cl + VS2026 MSVC STL；convertd 编译需 `ws2_32.lib`。
- 中文文档编码：只用 write/edit；PowerShell `Set-Content` 写 BOM/乱码。
- **patch 纪律**：改 vendored 前备份（`git -C third_party/llama.cpp branch backup`）；生成 patch 用 `cmd /c "git diff -- <文件列表> > patches\x.patch"`（PS `>` 写 UTF-16）。
- **临时调试**：`STREAM_MOE_DBG=1` env 开导出诊断；临时脚本进 temp/（gitignored）。
- 文档：`patches/README.md`（patch 体系）、`docs/STREAMMOE_GGUF_FORMAT.md`、`docs/REPACK_DIVERGENCE_DEBUG.md`、`docs/PROJECT_STRUCTURE.md`。
