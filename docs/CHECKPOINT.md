# StreamMoE 项目检查点 (CHECKPOINT.md)

> **用途**：opencode 会话上下文被压缩/重开时，先读本文件 + `docs/PROJECT_STRUCTURE.md` 恢复状态。
> **最近更新**：2026-08-27。维护者每阶段收尾更新"当前状态"与"下一步"。

---

## 1. 项目目标（一句话）

DeepSeek4 等 MoE 模型，**MoE 专家权重完全不走 mmap、走自研紧凑槽专家池**（route B，自定义 ggml backend 接管 MUL_MAT_ID），dense 保持 llama.cpp 默认；物理内存有界（池预算）。

## 2. 当前状态（✅ 已完成）

### 架构：迁移到上游 llama.cpp tools + route B 注入（M1-M3 完成）
- **构建**：clang-cl（`F:\Dev\LLVM\bin\clang-cl.exe`）+ **VS2026 MSVC STL** + libomp（`-Xclang -fopenmp`）+ `/EHsc`。`build.bat llamalibs main` 产出上游 `llama-cli/llama-server`（`build\main\llama-build\bin\`）。
- **M2 参数**：用原版 `--cache-type-k/v`、`--swa-full`（默认 windowed）、`-ngl`；我们的 `--kv-placement`（集合语法，单元素生效，多元素 warning）已加。
- **M3 route B 注入**：`common/arg.cpp` 加 `--expert-backend/--moe-ram-pool/--moe-vram-pool/--prompt-log`；`common_init_from_params` 调 `route_b_setup`（`src/server/route_b_inject.*`）注册后端 + 池 + 挂 `tensor_buft_overrides`。
- vendored 改动 5 文件记录于 `docs/VENDORED_MODIFICATIONS.md` + `patches/route-b-inject.patch`。

### route B 多子池（MULTI_SUBPOOL）
- 专家按布局分组成子池，预算按字节占比切分（槽数比例 = 专家数比例）。
- gemma-4-26B-A4B（2 组：29 层 Q4_K+Q5_1 + layer29 Q8_0）全层进池，2297 槽/8GB，chat 正常。
- deepseek-4（1 组）642 槽/8GB 池初始化正常。

### use_mmap=false + overrides 修复
- **`init_mappings` 检测 `n_expert>0` → 强制 `use_mmap=false`**（moe 不 mmap，mmap/mlock 忽略 + warning）。
- **修复**：`common_params_parse` 会把 `tensor_buft_overrides` pad 成 4096 null（fit 预留）——注入前 `clear()`；`load_all_data` 对池化专家跳过 plain-read（dummy data）。
- 实测 gemma 8G 池：**WS 5.94GB / PM 12.11GB，Mapped File ≈ 0（零 mmap）**，启动 12s（`--no-warmup` 后 8.1s，CLI 4.1s 可交互）。

### 启动性能（[TMR] RDTSC/chrono 计时，`src/tsc_timer.h` 临时）
- 模型加载 3.0s（`load_all_data` dense 冷盘读 2.1s）+ warmup 6.5s（已用 `--no-warmup` 跳过）+ server 框架 3.8s。
- `--no-warmup`：启动 8.1s；CLI 4.1s 出提示符。首次回复含冷盘专家装载（44s gemma）。

### KV 实测（q8_0 + windowed SWA，llama_get_memory_breakdown）
- gemma-4-26B-A4B：`-c 8192`=563MB；`-c 262144`=3.2GB。
- deepseek-4：待测（见 `docs/SMOKE_TESTING.md` 表）。

### 脚本
- `start_server.bat` / `run_long_horizon_test.bat` / `run_prefill_verify.bat` / `verify_prefill.bat` → 迁移到上游 server/cli + `--no-warmup`（产物两件套 conversation+profile）。
- `run_deepseek_cli.bat`（双击交互 CLI，70G 池）。

## 3. 你可以跑的验证

| 动作 | 命令 |
|---|---|
| 构建 | `build.bat llamalibs main` 然后 `build.bat build main` |
| deepseek 交互 CLI | 双击 `scripts\run_deepseek_cli.bat` |
| gemma 冒烟 | `build\main\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8997 -c 8192 -t 16 --expert-backend --moe-ram-pool 8192 --cache-type-k q8_0 --cache-type-v q8_0 --fit off --no-warmup --no-webui` |
| 单测 | `build.bat test main` |
| ASan 构建 | 见 `docs/ASAN_BUILD.md` |

## 4. 下一步（TODO，按序）

**A. 用户体验（你想试的，优先）**
1. 草稿推理（B11）：`--model-draft N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-...Q8_0.gguf`（上游原生支持）。
2. GPU 加速：`build.bat llamalibs main` 设 `GGML_VULKAN=ON`（RX 590）+ 设备后端；Phase B 专家池 GPU。
3. 自造 4K 对齐 GGUF（`stream_moe_convert`）：dense/expert 分区 + 4K 对齐 → DIO 整段读。

**B. 验证/回归**
4. deepseek KV 实测（8192→100k→1M）。
5. 数值等价（gemma/deepseek std vs `--expert-backend`）。
6. 长程回归 + 命中率曲线（`run_long_horizon` + `simulate_cache`）。

**C. 迁移收尾**
7. M4：删自研 `src/main.cpp` / `server_main.cpp` / `server/http_server.*` / `engine/llama_engine.*`。
8. 导出功能（prefill/KV/expert_history）重新适配上游（patch A 通用 + 新 patch B 于 server/cli）。
9. `--moe-preload` / `--moe-eviction` 参数。
10. KV 多副本（Phase B 后）。
11. 设备后端开关验证（Vulkan/CUDA 共存）。

## 5. 环境与坑（记住）

- **测试约定**：route B 池测试统一 `--moe-ram-pool 71680`（70GB）大 MoE / gemma 用 8192；短程 `-c 8192`。
- **moe 必须 `--fit off --no-warmup`**；专家 `--no-warmup` 避免启动空跑装载。
- 模型盘 N: = **USB 转接 NVMe**（162GB 冷盘慢）；GPU = Radeon RX 590 8GB（Vulkan only）。
- RAM 128GB；池 70GB 可行。
- OpenMP：`F:\Dev\LLVM\lib\libomp.lib` + `-Xclang -fopenmp`（clang-cl）。
- 编译器：clang-cl + VS2026 MSVC STL（VS2019 STL 有 cast-qual 硬错；libc++ on Windows 不成熟）。
- 中文文档编码：只用 write/edit；后台进程：`taskkill /F /IM llama-server.exe`。
- 文档：`docs/UPSTREAM_TOOLS_MIGRATION.md`（迁移计划）、`docs/VENDORED_MODIFICATIONS.md`（vendored 改动）、`docs/LLAMA_MMAP_CALLS.md`（mmap 地图 + lldb）、`docs/MULTI_SUBPOOL.md`、`docs/SMOKE_TESTING.md`、`docs/ASAN_BUILD.md`、`docs/SAMPLING.md`。
