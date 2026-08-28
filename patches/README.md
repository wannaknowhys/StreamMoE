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
   （那会把其他 patch 的改动混进来）。参考文件归属：
   - `route-b-inject.patch`：common/*、arg.cpp、llama-model-loader/model/llama.cpp、llama-context.cpp
     （仅 [TMR] 计时行）、server-context.cpp、speculative.*。
   - `prefill-export-llama.patch`：llama-context.h/cpp（prefill 部分，含 `_main/_draft` 分文件）、
     llama-kv-cache.h/cpp。
3. 应用前 `git apply --check`，应用后 `git apply --check -R`（验证可还原）。
4. 涉及同一文件的两个 patch（如 llama-context.cpp 的 [TMR] 与 prefill）位置不重叠，靠 `git apply` 的
   上下文 fuzz 叠加，应用顺序无关，但**还原顺序必须逆序**（先 -R 后应用的）。

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
