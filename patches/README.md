# Patches (临时诊断/实验补丁)

> 用法：这些是**一次性补丁**，不入主线。需要时临时应用 → 构建特殊版本 → 用完后 `git apply -R` 还原。
> 不再用分支承载（历史教训：分支长期干扰合并）。遇内存问题时：应用补丁 → `build.bat build memwatch` → 跑测试 → 还原。

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
