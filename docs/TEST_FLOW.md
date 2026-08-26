# StreamMoE 测试流程规范 (TEST_FLOW.md)

> 原则：**先单 prompt 验证，再整轮评测；长任务交给用户手动运行的 .bat，内存异常由人眼停止。**
> 目录/产物规范见 `PROJECT_STRUCTURE.md`；内存哨兵见 `patches/README.md`。

---

## 0. 铁律

1. **单 prompt 优先**：任何改动后，先用 `build\main\bin\stream_moe.exe -p "..." -n 32` 直接观察返回文本与行为，**不要**直接跑 server + 长测试。
2. **长任务交 .bat**：完整 `long_horizon_prompts*.jsonl` 跑批只用 `scripts\run_long_horizon_test.bat`，由用户自己双击运行；看到内存异常手动关闭（Ctrl+C / 任务管理器）。
3. **规避 PowerShell 脚本**：日常命令尽量单行、直接调用 exe；不要写复杂 PS 脚本文件。
4. **后台进程必须可回收**：跑批前后 `taskkill /F /IM stream_moe_server.exe`，杜绝僵尸占端口。
5. **产物落盘可追溯**：同一次运行的 conversation / profile / report 同后缀，落在 `benchmark\results\`。

## 1. 内存诊断：临时应用 memwatch 补丁（不再用分支）

遇到"内存被撑爆"时：
1. 按 `patches/README.md` 应用 memwatch 补丁（`git -C third_party/llama.cpp apply patches\memwatch-ggml.patch` + `git apply patches\memwatch-build.patch`）。
2. 构建特殊版：`build.bat llamalibs memwatch && build.bat build memwatch`（独立 `build\memwatch\` 产物，不碰 `build\main\`）。
3. 单 prompt 复现：`build\memwatch\bin\stream_moe.exe ...`。
4. 看 `%TEMP%\memwatch_<pid>.log`：`WS` 大而 `PRIV` 小 → file-backed 页驻留；`PRIV` 也大 → 真实私有拷贝/泄漏。
5. 用后还原补丁，确认 `git -C third_party/llama.cpp status` 干净。

## 2. 快速验证：单 prompt（几十秒~几分钟）

```bat
build\main\bin\stream_moe.exe -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" --moe-ram-pool 71680 -c 4096 -t 16 -p "Say hello in one short sentence." -n 32
```

- `-n` 先给 16~32；首 token 慢（模型在 N: 冷页拉取）。
- 观察：返回是否可读/切题；是否崩溃；`%TEMP%\memwatch_*.log`（若应用了补丁）。

## 3. 完整评测：long_horizon_prompts（.bat，用户运行）

```bat
scripts\run_long_horizon_test.bat en        rem 英文 10 轮（build tag 默认 main）
scripts\run_long_horizon_test.bat zh memwatch  rem 指定 build tag
```

脚本行为：起 server（端口 8992）→ 等 /health 就绪（120s）→ 跑 `node tools\bench_agent.js` → 杀 server。
产物：`benchmark\results\conversation_real_<tag>.txt` / `profile_real_<tag>.jsonl` / `BENCHMARK_REPORT_REAL_<tag>.md`；server 日志 `build\<btag>\server_<tag>.log`。

**用户职责**：盯内存；异常立即关窗/杀 `stream_moe_server.exe`。

## 4. 产物与日志位置速查

| 产物 | 路径 |
|---|---|
| 单 prompt 输出 | 直接终端 |
| server 运行日志 | `build\<btag>\server_<tag>.log` |
| 逐轮遥测 JSONL | `benchmark\results\profile_real_<tag>.jsonl` |
| 完整对话转写 | `benchmark\results\conversation_real_<tag>.txt` |
| 报告 | `benchmark\results\BENCHMARK_REPORT_REAL_<tag>.md` |
| memwatch 哨兵日志 | `%TEMP%\memwatch_<pid>.log`（memwatch 版构建）|

## 5. 已知坑

- **端口冲突**：跑批前先 `taskkill /F /IM stream_moe_server.exe`。
- **N: 盘 = USB 转接 NVMe**：模型 162GB 在 N:，冷页拉取慢（decode 0.3~2 tok/s），页缓存随运行变热。
- **中文显示**：PowerShell 控制台 CP936 会把 UTF-8 显示成乱码；文件本身是合法 UTF-8，用编辑器/`git show` 查看。
- **CRLF 幻影改动**：`core.autocrlf=true` 会让 `git status` 显示无内容差异的 `M`；提交前 `git diff` 确认。
- **文档编码**：含中文的 md 只能用 write/edit 工具改；严禁 PowerShell Set/Add-Content。
