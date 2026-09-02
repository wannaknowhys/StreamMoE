# StreamMoE 项目检查点 (CHECKPOINT.md)

> **用途**：opencode 会话上下文被压缩/重开时，先读本文件 + `docs/PROJECT_STRUCTURE.md` + `patches/README.md` 恢复状态。
> **最近更新**：2026-09-03（patch 体系重构收尾）。维护者每阶段收尾更新"当前状态"与"下一步"。

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
- **宏隔离**：无宏（features 空 / 只 phase1）= 纯上游等价（include 行预处理跳过）。编译目录：`main`→route_b；`upstream_dump`/`upstream_vulkan_dump`→prefill_export；`StreamMoE_dump`→两者；`asan`→route_b（MSVC cl，`build.bat asan`）。
- **当前任务追踪**：`docs/WORK_IN_PROGRESS.md`。
- vendored 子模块有 `backup-20260830` 分支（整理前状态，保险）。

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
| 完整栈导出 | `build.bat llamalibs StreamMoE_dump` → `build\StreamMoE_dump\llama-build\bin\llama-server.exe` |
| 转换器服务 | `build.bat convertd` → `build\convertd\convertd.exe` |
| 转换矩阵 | `scripts\verify_convert_matrix.bat <workdir>`（N 原版源 → R 盘）|
| gemma 冒烟 | `build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup --no-webui` |
| prefill 导出（--export-dir）| `llama-server -m <gemma> --export-dir <dir> ...` + 喂 prompt + shutdown → 导出 prefill_export/tokens_id/tokens_text |
| prefill-from | `llama-server -m <gemma> --prefill-from <prompt.txt|tokens.bin> --export-dir <dir> -c 1024 -t 8` |

**run_export 前台窗口启动**（跑 cn/en/prefill10000 导出任务——脱离 opencode 管控但用户可见）：
```bat
agy-run -c "start cmd /k temp\run_export_win.bat"
```
- agy-run 绑 `WinSta0\Default` 在**交互桌面**弹新 cmd 窗口 → llama-server 在窗口里跑（**用户实时看**）。
- `start` 立即返回 → **opencode/bash 不阻塞**。别用 `start /b` 后台（那才真正脱离管控且看不到）。
- `temp\run_export_win.bat`：`call temp\sm_env.bat`（机器路径 env）+ `node tools\run_export.js --models ... --engines ... --tasks ...`。

---

## 4. 下一步（TODO）

**主线方向**
1. **deepseek `--prefill-from` KV 预构建实测**（gemma 非 dsv4 无 KV；deepseek 才有，验证 KV 导出 + 预构建价值）。
2. **vulkan 作为 backend dll**（`GGML_BACKEND_DL` + `BUILD_SHARED_LIBS=ON`）——**需与 moe 适配商议**。
3. **mini-graph / 主图就地池化**（route B 计算路径重构，待聊设计）。

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
