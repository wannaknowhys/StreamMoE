# Patches (临时诊断/实验补丁)

> 用法：这些是**一次性补丁**，不入主线。需要时临时应用 → 构建特殊版本 → 用完后 `git apply -R` 还原。
> 不再用分支承载（历史教训：分支长期干扰合并）。遇内存问题时：应用补丁 → `build.bat build memwatch` → 跑测试 → 还原。

## prefill-export（Prefill 交叉验证导出）

任务二（`docs/PREFILL_CROSS_VALIDATION.md`）：在 `llama_context` 累积每步 LM head 输入（`result_norm`）+ hidden state（`t_h_nextn`），析构时一次性导出到 `LLM_EXPORT_DIR/prefill_export.bin`（含最终 KV 全部子缓存逐层张量）。配合 `tools/verify_prefill.js`（cos+maxAbs 双门槛）对比标准 vs `--expert-backend`。

| 文件 | 内容 |
|---|---|
| `prefill-export-llama.patch` | vendored llama.cpp：`llama-kv-cache.h/.cpp` 加 `get_v_storage`；`llama-context.h/.cpp` 加累积成员 + 析构导出 `export_prefill_final()` + decode 循环内累积；`LLM_EXPORT_DIR` 时强制 `embeddings_nextn`（导出 `t_h_nextn`）|
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
| `memwatch-build.patch` | 仓库根：`build.bat` 链接参数加 `-lpsapi`（`GetProcessMemoryInfo` 需要）|

**应用**：
```bat
rem 1) vendored llama.cpp 内应用 ggml 补丁
git -C third_party/llama.cpp apply patches\memwatch-ggml.patch
rem 2) 仓库根应用 build 补丁（若 build.bat 已有 -lpsapi 则跳过）
git apply patches\memwatch-build.patch
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
