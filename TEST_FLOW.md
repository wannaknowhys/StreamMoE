# StreamMoE 测试流程规范 (TEST_FLOW.md)

> 本文件定义统一的测试工作流，避�?后台进程 + 无人值守跑长任务"导致的失控（内存爆满 / 僵尸进程 / 日志丢失）�?> 原则�?*先单 prompt 验证，再整轮评测；长任务一律交给用户手动运行的 .bat，内存异常由人眼停止�?*

---

## 0. 铁律

1. **�?prompt 优先**：任何改动后，先�?`bin\stream_moe.exe -p "..." -n 32` 直接观察返回文本与行为，**不要**直接�?server + 长测试�?2. **长任务交 .bat**：完�?`long_horizon_prompts*.jsonl` 跑批只用 `.bat` 脚本，由用户自己双击运行；用户看到内存异常会手动关闭（Ctrl+C / 任务管理器）�?3. **规避 PowerShell 脚本**：日常命令尽量单行、直接调�?exe；不要写复杂 PS 脚本文件�?4. **后台进程必须可回�?*：server 进程要能被清晰终止；不再�?僵尸占端�?（历史教训：旧二进制�?8999 端口服务了错误数据）�?5. **产物落盘可追�?*：每次测试的 server 日志、profile JSONL、conversation 文件�?`_<tag>` 后缀隔离，避免互相覆盖�?
---

## 1. 快速验证：�?prompt（几十秒~几分钟）

直接命令行调用，无后台进程、无脚本文件�?
```bat
bin\stream_moe.exe -m "N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf" --moe-ram-pool 71680 -c 4096 -t 16 -p "Say hello in one short sentence." -n 32
```

- `-n` 先给 16~32 观察；首 token 慢（模型�?iSCSI N: 盘冷页拉取）�?- 观察点：返回文本是否可读/切题；进程是否崩；`temp\memwatch_*.log`（诊断分支）是否出现异常�?
## 2. 完整评测：long_horizon_prompts�?bat，用户运行）

使用 `run_long_horizon_test.bat`（见仓库根目录）。用法：

```bat
run_long_horizon_test.bat en    rem 英文 10 �?run_long_horizon_test.bat zh    rem 中文 10 �?```

脚本行为�?1. 启动 `bin\stream_moe_server.exe`（端�?8992，`--moe-ram-pool 71680`，`-n 384`，`--temp 0.6 --top-p 0.95`，`--profile-log benchmark\profile_real_<tag>.jsonl`）�?2. 等待 /health 就绪（轮�?120s，超时自动杀进程并退出）�?3. 运行 `node tools\bench_agent.js --prompts benchmark\long_horizon_prompts[_zh].jsonl ...`，输出到 `benchmark\conversation_real_<tag>.txt`�?4. 结束后杀�?server，汇�?server 日志�?`temp\server_<tag>.log`�?
**用户职责**：运行期间盯住内存（任务管理器）。发现内存接近爆满，立即关掉 CMD 窗口或杀 `stream_moe_server.exe`——这本身就是重要的诊断信号，把现象反馈给开发者�?
## 3. 诊断分支：memwatch 内存哨兵（debug/memguard�?
当怀�?内存又被撑爆 / file-backed cache"时，切到 `debug/memguard` 分支重新编译，得到带内存哨兵的测试版本：

```bat
git checkout debug/memguard
build.bat llamalibs   rem 用哨兵插桩重新构�?vendored llama
build.bat build
```

- 哨兵在所�?ggml 内存分配点打 log 并检查进程总内存，�?90GB 自动 `ExitProcess`�?- 日志写入 `temp\memwatch_<pid>.log`，每行含：调用点 / 分配大小 / 指针 / 进程 WorkingSet / PrivateBytes / Commit / 系统可用 / 系统 Cache�?- **判定方法**：WorkingSet 大�?PrivateBytes �?�?file-backed cache（mmap 页驻留）是主因；PrivateBytes 也大 �?存在真实私有拷贝/泄漏�?- 插桩点：`ggml_backend_buft_alloc_buffer` / `ggml_backend_buft_get_alloc_size` / `ggml_backend_buffer_free` / `ggml_backend_dev_buffer_from_host_ptr`（及额外补充点）�?- 主线修复后，`git merge main` 合并回诊断分支重新编译，确认问题是否消失�?
## 4. 产物与日志位置速查

| 产物 | 路径 |
|---|---|
| �?prompt 输出 | 直接终端 |
| server 运行日志 | `temp\server_<tag>.log` / `.err` |
| 逐轮遥测 JSONL | `benchmark\profile_real_<tag>.jsonl` |
| 完整对话转写 | `benchmark\conversation_real_<tag>.txt` |
| 报告 | `benchmark\BENCHMARK_REPORT_REAL_<tag>.md` |
| memwatch 哨兵日志 | `temp\memwatch_<pid>.log` |

## 5. 已知�?
- **端口冲突**：旧 server 可能残留占用端口。运�?.bat 前先 `taskkill /F /IM stream_moe_server.exe`�?- **iSCSI N: �?*：模�?162GB �?LIO-ORG 网络盘上，冷页拉取极慢（decode 0.2~2 tok/s）；页缓存会随运行变热。基准数字需注明介质�?- **中文显示**：PowerShell 控制�?CP936 会把 UTF-8 显示成乱码，转写文件本身是合�?UTF-8，用编辑�?`git show` 查看�?