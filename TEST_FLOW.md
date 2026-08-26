# StreamMoE 测试流程规范 (TEST_FLOW.md)

> 本文件定义统一的测试工作流，避免"后台进程 + 无人值守跑长任务"导致的失控（内存爆满 / 僵尸进程 / 日志丢失）。
> 原则：**先单 prompt 验证，再整轮评测；长任务一律交给用户手动运行的 .bat，内存异常由人眼停止。**

---

## 0. 铁律

1. **单 prompt 优先**：任何改动后，先用 `bin\stream_moe.exe -p "..." -n 32` 直接观察返回文本与行为，**不要**直接跑 server + 长测试。
2. **长任务交 .bat**：完整 `long_horizon_prompts*.jsonl` 跑批只用 `.bat` 脚本，由用户自己双击运行；用户看到内存异常会手动关闭（Ctrl+C / 任务管理器）。
3. **规避 PowerShell 脚本**：日常命令尽量单行、直接调用 exe；不要写复杂 PS 脚本文件。
4. **后台进程必须可回收**：server 进程要能被清晰终止；不再有"僵尸占端口"（历史教训：旧二进制占 8999 端口服务了错误数据）。
5. **产物落盘可追溯**：每次测试的 server 日志、profile JSONL、conversation 文件按 `_<tag>` 后缀隔离，避免互相覆盖。

---

## 1. 分支工作流：debug/memguard = 主线 + memwatch 补丁

**memwatch（内存哨兵）是一个补丁，不是独立主线。** 规则：

- `debug/memguard` 分支 = `mainline` + 仅两个改动：
  1. vendored llama.cpp 子模块：`ggml/src/ggml-memwatch.h`（哨兵 log 函数）+ `ggml-backend.cpp` 里 4 个分配点插桩调用。
  2. `build.bat` 的 `-lpsapi`（`GetProcessMemoryInfo` 需要）。
- **其它一切改动（文档、测试脚本、报告、架构研究）都直接提交到 `main`。**
- **主线随时合并进补丁分支**：`git checkout debug/memguard && git merge main`。子模块指针冲突时：保留 memwatch 子模块提交，把主线子模块指针合进来后重放插桩（若 llama.cpp 升级导致插桩点行号变化）。
- 验收流程：主线修复 → `git merge main` 到 `debug/memguard` → `build.bat llamalibs && build.bat build` → 跑单 prompt → 看 `%TEMP%\memwatch_<pid>.log` 验证内存行为 → 结果反馈到 `BUG_TRACKER.md`（提交到 main）。

## 2. 快速验证：单 prompt（几十秒~几分钟）

直接命令行调用，无后台进程、无脚本文件：

```bat
bin\stream_moe.exe -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" --moe-ram-pool 71680 -c 4096 -t 16 -p "Say hello in one short sentence." -n 32
```

- `-n` 先给 16~32 观察；首 token 慢（模型在 N: 冷页拉取）。
- 观察点：返回文本是否可读/切题；进程是否崩；`%TEMP%\memwatch_<pid>.log`（诊断分支）是否出现异常。

## 3. 完整评测：long_horizon_prompts（.bat，用户运行）

使用 `run_long_horizon_test.bat`（仓库根目录）：

```bat
run_long_horizon_test.bat en    rem 英文 10 轮
run_long_horizon_test.bat zh    rem 中文 10 轮
```

脚本行为：
1. 启动 `bin\stream_moe_server.exe`（端口 8992，`--moe-ram-pool 71680`，`-n 384`，`--temp 0.6 --top-p 0.95`，`--profile-log benchmark\profile_real_<tag>.jsonl`）。
2. 等待 /health 就绪（轮询 120s，超时自动杀进程并退出）。
3. 运行 `node tools\bench_agent.js --prompts benchmark\long_horizon_prompts[_zh].jsonl ...`，输出到 `benchmark\conversation_real_<tag>.txt`。
4. 结束后杀掉 server，server 日志落 `temp\server_<tag>.log`。

**用户职责**：运行期间盯住内存（任务管理器）。发现内存接近爆满，立即关掉 CMD 窗口或杀 `stream_moe_server.exe`——这本身就是重要的诊断信号。

## 4. 产物与日志位置速查

| 产物 | 路径 |
|---|---|
| 单 prompt 输出 | 直接终端 |
| server 运行日志 | `temp\server_<tag>.log` / `.err` |
| 逐轮遥测 JSONL | `benchmark\profile_real_<tag>.jsonl` |
| 完整对话转写 | `benchmark\conversation_real_<tag>.txt` |
| 报告 | `benchmark\BENCHMARK_REPORT_REAL_<tag>.md` |
| memwatch 哨兵日志 | `%TEMP%\memwatch_<pid>.log`（诊断分支）|

## 5. 已知坑

- **端口冲突**：旧 server 可能残留占用端口。运行 .bat 前先 `taskkill /F /IM stream_moe_server.exe`。
- **N: 盘 = USB 转接 NVMe**：模型 162GB 在 N:，冷页拉取慢（decode 0.3~2 tok/s），页缓存随运行变热。基准数字需注明介质。
- **中文显示**：PowerShell 控制台 CP936 会把 UTF-8 显示成乱码，转写文件本身是合法 UTF-8，用编辑器/`git show` 查看。
- **CRLF 幻影改动**：`core.autocrlf=true` 会让 `git status` 显示大量无内容差异的 `M` 文件；提交前用 `git diff` 确认内容。
