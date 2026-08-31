# Patches (临时诊断/实验补丁)

> 用法：这些是**一次性补丁**，不入主线。需要时临时应用 → 构建特殊版本 → 用完后 `git apply -R` 还原。
> 不再用分支承载（历史教训：分支长期干扰合并）。遇内存问题时：应用补丁 → `build.bat build memwatch` → 跑测试 → 还原。

## 叠加纪律（叠加应用 patch 前必须 commit + 快照）

> 问题根源（2026-08-27）：`route-b-inject.patch`（长期）与 `prefill-export-llama.patch`（临时）
> 同时 apply 在 vendored 工作区，且应用 prefill **前没有记录已有状态** → 两个 patch 的改动混合在
> 同一个未提交工作区：route-b-inject.patch 重生成时把 prefill 改动也抄进去（`git diff` 是全量），
> prefill patch 因同一文件被后续改动污染而 `git apply -R` 失败。**根因 = 叠加前未 commit/快照。**

铁律：
1. **在已有 patch 应用的基础上叠加第二个 patch 前，必须先**：
   a. 提交父仓库当前状态（patch 文件 + 相关改动 commit + push）；
   b. 快照 vendored 工作区：
      `git -C third_party/llama.cpp diff > temp/patch_backup_<date>/working-tree-full.patch`。
2. **生成 patch 用 `git diff -- <文件列表>` 限定范围**，各 patch 各管各的文件，绝不 `git diff >` 全量抄
   （那会把其他 patch 的改动混进来）。文件归属（2026-08-30 整理，route B 栈收敛为 4 个 patch）：
   - `tsc_timer.patch`（[TMR] 计时头，独立）：`src/tsc_timer.h`。被 route-b（common）和 prefill
     （llama-context.cpp 的 sm_tmr）共享，独立成 patch。
   - `route-b-inject.patch`（route B 核心 + 附属）：`common/CMakeLists.txt`、`common/arg.cpp`（route-b 参数 +
     `--prompt-log`/`--kv-placement`，原 export-args.patch 已并入）、`common/common.cpp`、`common/common.h`、
     `common/speculative.cpp/h`（route B draft 池绑定 + draft 统计，原 spec-stats.patch 已并入）、
     `src/llama-model-loader.cpp`、`src/llama-model.cpp`、`src/llama.cpp`、
     `tools/server/server-context.cpp`（spec-stats 析构打印 + [TMR]）。
   - `gguf-alignment.patch`（GGUF 转换器依赖）：`ggml gguf.h/cpp`（`gguf_set_alignment`）。
   - `prefill-export-llama.patch`：`src/llama-context.cpp/h`（prefill 导出 + 专家历史 + export_token_seq +
     cb_eval 图内抓取：top-4 logits + logsumexp + 采样 tokens）、`src/llama-kv-cache.h/cpp`、
     `tools/server/server.cpp`（/shutdown 端点）。

**vulkan 构建修复（无 patch）**：`GGML_VULKAN=ON` 时由 `build.bat` 通过上游注入点
`-DVULKAN_SHADER_GEN_CMAKE_ARGS` 把 ninja/clang 工具链传给 shader-gen 子 cmake——不改 vendored。

**route B 栈应用顺序（phase 结构）**（`git apply --check` 已验证可在 vendored HEAD=f280b2698 上叠加应用）：
   ```
   Phase 1（必选，互不依赖，顺序可互换）：streammoe-macros → tsc_timer
   Phase 2a（可选）：route-b-inject
   Phase 2b（可选）：prefill-export-llama
   ```
- **Phase 1 = 占位 + tsc**：占位改共享文件（common.h/common.cpp/arg.cpp/llama.h 的宏 include 块），tsc 新增 `src/tsc_timer.h`（[TMR]）——不同文件，先谁后谁无所谓；但**必须在所有功能 patch 前**（占位 = include 锚点；tsc = route-b/prefill 都 include 的 [TMR] 头）。**Phase 1 单独 apply + 无宏编译 = 纯上游等价物**（include 行被预处理跳过）。
- **Phase 2a = route-b**（可选）：只新增 route 的 `include/*.frag` + 专属文件，不碰共享文件。
- **Phase 2b = prefill-export-llama**（可选）：只新增 prefill 的 `include/*.frag` + llama 层（llama-context/kv-cache/server-context/server.cpp）。prefill 只依赖 tsc（phase 1 已含），不依赖 route-b。
- **`gguf-alignment.patch` 独立**：转换器工具（`tools/stream_moe_layout.js` 需要 `gguf_set_alignment`），改 ggml gguf.h/cpp——**与推理构建无关，不入 phase 2b 构建过程**；可任意时 apply。
- **组合任意**：只 2a / 只 2b / 2a+2b / 都不——各功能 patch 只新增自己的文件 + 各自命名的 .frag（不同名），apply 层零交集；编译时 `-D` 宏选功能（`STREAM_MOE_ROUTE_B` / `STREAM_MOE_PREFILL_EXPORT`）。

**编译目录（vendored llama-server 变体，主项目 stream_moe 已废弃）**：
   ```
   build/main             phase 2a（route-b 完整推理，生产主线）
   build/upstream_dump    phase 2b（prefill 导出——标准上游行为基准，无 route-b）
   build/StreamMoE_dump   2a+2b（完整 StreamMoE 导出，与 upstream_dump 对比）
   build/convertd         转换器哑服务（独立工具，裸 TCP + ws2_32.lib）
   ```
   `build.bat llamalibs/build <tag>` 按 tag 传宏：`main`→`-DSTREAM_MOE_ROUTE_B`；`upstream_dump`→`-DSTREAM_MOE_PREFILL_EXPORT`；`StreamMoE_dump`→两者；`convertd` 无宏。

**当前状态（2026-08-30 整理）**：vendored 子模块 HEAD 已回滚到纯上游 `f280b2698`，工作区完全干净；
StreamMoE 全部改动只存在于上述 patch（含 spec-stats/export-args 并入项，vulkan 走 build.bat），按顺序 apply 可完整复现。

## 共享文件宏隔离（2026-08-31 约定，冲突文件一律这样搞）

两个功能（route B 推理引擎 / prefill 导出）都挂在 `common_params` + context 参数传递——**patch 文本在同一结构相邻插入 → apply 顺序冲突**。解法：**占位 patch 在共享文件只加宏保护的 include 片段，高层 patch 只新增 `.frag` 文件**（不碰共享文件）：

```cpp
// 占位 patch（streammoe-macros.patch）改共享文件：只加 include 宏块
struct common_params {
#ifdef STREAM_MOE_PREFILL_EXPORT
#include "stmoe_prefill_common_params.frag"   // 编译期文本替换进结构
#endif
    int32_t n_predict = -1;
#ifdef STREAM_MOE_ROUTE_B
#include "stmoe_routeb_common_params.frag"
#endif
    int32_t n_ctx = 0;
};
// route-b patch / prefill patch：只新增 include/ 下的 *.frag 文件（片段内容），不碰 common.h/common.cpp/arg.cpp/llama.h
```

- **宏**：`STREAM_MOE_ROUTE_B` / `STREAM_MOE_PREFILL_EXPORT`（`build.bat` 编译时传；宏未定义时 include 行被预处理跳过，占位单独可编译）。
- **效果**：include 片段 = 编译期文本替换 = 等价 diff；共享文件只被占位改一次；高层 patch 只新增 `.frag`（放 `include/`）——**无 apply 冲突**；`-D` 选功能（prefill 独立 = 占位 + prefill + 只开 PREFILL）。
- **注意**：片段文件放 vendored `include/`（共享文件的 include 路径）；新增共享改动一律"占位 include + 高层 .frag"。
3. 应用前 `git apply --check`，应用后 `git apply --check -R`（验证可还原）。
4. 涉及同一文件的两个 patch 位置不重叠，靠 `git apply` 的上下文 fuzz 叠加，应用顺序无关，但**还原顺序必须逆序**（先 -R 后应用的）。

## prefill-export（Prefill 交叉验证导出 + 专家访问历史）

任务二（`docs/PREFILL_CROSS_VALIDATION.md`）+ 任务一（`docs/EXPERT_TRACE_SIMULATION.md`）：在 `llama_context` 累积每步 LM head 输入（`result_norm`）+ hidden state（`t_h_nextn`）与**专家访问历史**，析构时一次性导出：
- `LLM_EXPORT_DIR/prefill_export.bin`：LM head 输入 + hidden + 最终 KV 全部子缓存逐层张量（`tools/verify_prefill.js` 双门槛对比）。
- `LLM_EXPORT_DIR/expert_history.bin`：逐 token/层/专家的路由访问序列（`tools/simulate_cache.js` 策略命中率曲线）。

| 文件 | 内容 |
|---|---|
| `prefill-export-llama.patch` | vendored llama.cpp：`llama-kv-cache.h/.cpp` 加 `get_v_storage`；`llama-context.h/.cpp` 加累积成员 + 析构导出 `export_prefill_final()` / `export_expert_history_final()` + decode 循环内累积（embd/hidden/专家历史）；`LLM_EXPORT_DIR` 时强制 `embeddings_nextn` |
| `prefill-export-streammoe.patch` | StreamMoE `llama_engine.cpp`：`LLM_EXPORT_DIR` 时 prefill 全部 token `logits=true`（LM head 输入覆盖全部 token）|

**应用/还原**：
```bat
git -C third_party/llama.cpp apply patches\prefill-export-llama.patch
git apply patches\prefill-export-streammoe.patch
build.bat llamalibs main && build.bat build main
rem 用完还原
git -C third_party/llama.cpp apply -R patches\prefill-export-llama.patch
git apply -R patches\prefill-export-streammoe.patch
```
注意：cmd 设 env 必须 `set "LLM_EXPORT_DIR=path"`（引号），否则尾随空格导致 fopen 失败。

## memwatch（内存哨兵）

定位"内存又被撑爆"的插桩。在所有 ggml 内存分配点打 log + 检查进程 WorkingSet，超 90GB 自动 `ExitProcess`。

| 文件 | 内容 |
|---|---|
| `memwatch-ggml.patch` | vendored llama.cpp 内：新增 `ggml/src/ggml-memwatch.h` + 在 `ggml-backend.cpp` 的 4 个分配点插桩（`buft_alloc_buffer` / `buft_get_alloc_size` / `buffer_free` / `dev_buffer_from_host_ptr`）|
| `memwatch-build.patch` | ~~改 build.bat 加 `-lpsapi`~~ **已失效**：build.bat 重构为 CMake 薄壳后不再含链接参数。如需 psapi（`GetProcessMemoryInfo`），临时在 `CMakeLists.txt` 的链接列表加 `psapi`（`target_link_libraries(... PRIVATE psapi)`）|

**应用**：
```bat
rem 1) vendored llama.cpp 内应用 ggml 补丁
git -C third_party/llama.cpp apply patches\memwatch-ggml.patch
rem 2) （如需 psapi）临时改 CMakeLists.txt 链接列表加 psapi；memwatch-build.patch 已失效
rem 3) 构建 memwatch 版（独立产物目录，不碰 main 版）
build.bat llamalibs memwatch
build.bat build memwatch
```

**运行与观测**：
- 单 prompt：`build\memwatch\bin\stream_moe.exe -m <model> --moe-ram-pool 71680 -c 4096 -t 16 -p "..." -n 24`
- 长测试：`scripts\run_long_horizon_test.bat en memwatch`
- 日志：`%TEMP%\memwatch_<pid>.log`，每行含 `调用点|size|ptr|WS|PRIV|COMMIT|FILEBK|SYS_AVAIL|SYS_TOTAL`。
- **判定**：`WS` 大而 `PRIV` 小 → file-backed 页驻留；`PRIV` 也大 → 真实私有拷贝/泄漏。`FILEBK` = WS-PRIV。

**还原**：
```bat
git -C third_party/llama.cpp apply -R patches\memwatch-ggml.patch
git apply -R patches\memwatch-build.patch
rem 子模块回上游
git -C third_party/llama.cpp checkout f280b2698
git submodule update --init
```

**注意**：应用补丁会弄脏 vendored 子模块工作区；还原后务必 `git -C third_party/llama.cpp status` 确认干净。
