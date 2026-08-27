# 迁移到原版 llama.cpp Tools：CLI + Server 重构计划 (UPSTREAM_TOOLS_MIGRATION.md)

> 状态：**计划（草案）**。目标：废弃自研 `main.cpp` / `server_main.cpp` / `http_server`，改用 vendored llama.cpp 原版 `llama-cli`（CLI）与 `llama-server`（OpenAI 兼容服务端），复用原版 `common/` 库，route B 作为插件注入。
> 前提结论：**我们的 KV cache 一直是原版**（llama.cpp 原版流程），之前"自研 server 报告"是唯一偏离点，已改为原版 API。本次迁移把"外壳"也全部交给原版。

---

## 1. 背景与动机

| 现状 | 问题 |
|---|---|
| 自研 `src/server/http_server.*` | 手写 HTTP/SSE，功能少：单槽、无 tools/函数调用、无 /v1/completions、无 API key、无 metrics/日志体系、无 context-shift/slot 管理 |
| 自研 `src/server_main.cpp` / `src/main.cpp` | 参数解析手写；`--cache-type`/`--no-swa-full` 与上游参数冲突 |
| vendored `common/`（28 文件）编译了但**未链接** | 浪费；且原版已实现全部推理/采样/对话/参数基建 |

**决策**：CLI/server 外壳整体走原版 `llama-cli` / `llama-server`，`common/` 直接链接使用；**route B（`src/backend/`）保留为插件**，通过公共扩展点注入。

---

## 2. 目标架构

```text
stream_moe.exe          = llama-cli（原版）+ route B 插件
stream_moe_server.exe   = llama-server（原版）+ route B 插件
```

- **直接复用（零改动）**：`llama-server`（server.cpp + server-http/context/task/queue/stream/chat/models/tools/schema/mcp/cors 全模块）、`llama-cli`、`common/`（arg.cpp 参数体系、chat.cpp 模板、sampling.cpp、speculative.cpp、json-schema→grammar、llguidance、reasoning-budget、preset、log.cpp 等）。
- **route B 插件**：`src/backend/`（moe_backend/minigraph_exec/scheduler/slot/alloc）+ `src/io/` + `src/pool/` + `src/loader/` 保留，经 `llama_model_params.tensor_buft_overrides` + `ggml_backend_reg` 注册注入（原路线不变）。
- **删除**：`src/server/http_server.*`、`src/server_main.cpp`、`src/main.cpp`、自研 `src/engine/llama_engine.*`（原版 server/cli 直接用 libllama 推理，route B 由 backend 承担；engine 若仅用于 trace/验证则降级为 `diagnostics/` 工具）。
- **保留**：`src/common/`（types/logger/crash——crash 兜底仍有用）、`src/profile/`（如需自定义 JSONL）、`tools/`（js 脚本）、`scripts/`、`patches/`、`tests/`。

---

## 3. 参数对比与去留（我们现有选项 → 原版）

| 我们的参数 | 原版 llama-cli/server | 处置 |
|---|---|---|
| `-m/--model` | `--model` | ✅ 用原版 |
| `--host/--port` | `--host/--port`（server）| ✅ 用原版 |
| `-c/--ctx-size` | `--ctx-size` | ✅ 用原版 |
| `-ngl/--gpu-layers` | `--n-gpu-layers` | ✅ 用原版 |
| `-t/--threads` | `--threads` | ✅ 用原版 |
| `-n/--n-predict` | `--n-predict` | ✅ 用原版 |
| `--temp/--top-p/--top-k` | 同 | ✅ 用原版（还有 --samplers 等全链）|
| `--mlock` | `--mlock` | ✅ 用原版 |
| `--kv-placement ram/vram` | `-ngl` + `--flash-attn` + `--cache-type-*` 组合 | ❌ **去掉**，用原版机制 |
| `--cache-type`（合并）| `--cache-type-k` / `--cache-type-v`（-ctk/-ctv）| ❌ **去掉**，用原版 |
| `--no-swa-full` | `--swa-full`（**默认 false=windowed**）| ❌ **去掉**，用原版 `--swa-full` |
| `--expert-backend` | 无 | 🔧 小改：arg.cpp 加参数 + 注入 |
| `--moe-ram-pool` | 无 | 🔧 小改：arg.cpp 加参数 |
| `--moe-vram-pool` | 无 | 🔧 小改：arg.cpp 加参数（Phase B 用）|
| `--profile-log` | server `--metrics`（Prometheus）+ `--log-*` | 🔧 映射或小改保留 JSONL |
| `--prompt-log` | 无（server 有 `--prompt-cache` 缓存、`--log-prompt`）| 🔧 小改：arg.cpp 加参数 + server 钩子 |
| `-i/-p` | `llama-cli` 原生 REPL / `-p` | ✅ 用原版 |

> **重要发现**：原版 `common_params.swa_full` 默认 **false（windowed）**，`--swa-full` 显式开启（arg.cpp:1679）。我们自研 server 用 `llama_context_default_params()`（swa_full=true）**默认反而比原版大**。迁移后默认即回到原版 windowed 行为（KV 更小），无需 `--no-swa-full`。
>
> **默认推荐参数**（SAMPLING.md/start_server.bat 同步改为原版）：`--temp 1.0 --top-p 0.95 -c 1048576 -n 384000 --cache-type-k q8_0 --cache-type-v q8_0 --no-warmup`（不加 `--swa-full` = windowed；`--no-warmup` 跳过空跑前向，MoE 下避免启动时冷盘专家装载）。

---

## 4. llama.cpp 原版评估：能用 / 小改 / 替换

### 4.1 能用（零改动）✅
- `llama-server`：HTTP/SSE/OpenAI API、`/v1/chat/completions`（stream、tools、json-schema、MCP、CORS）、`/v1/models`、`/v1/completions`、`/health`、`/metrics`、`/slots`、multi-slot（`--parallel --slots`）、context-shift、undo/reasoning、prompt-cache、`--draft/--model-draft`（**B11 投机解码直接白拿**）。
- `llama-cli`：REPL/`-p`/`-f`、采样全链、grammar、`--mlock`、多轮缓存。
- `common/`：arg.cpp 参数解析（含 `--swa-full`、`--cache-type-k/v`、`--ctx-checkpoints`）、chat template（Jinja）、sampling、speculative、json-schema→grammar、preset、log、console。
- KV/上下文：`--cache-type-k/v`（量化）、`--swa-full`、`--ctx-size`、`--flash-attn`、`--offload-kqv`（若 CLI 存在）。

### 4.2 小改（加 route B 接线，不改上游语义）🔧
| 改动点 | 内容 |
|---|---|
| `common/arg.cpp` | 新增 4 个参数：`--expert-backend`、`--moe-ram-pool <MB>`、`--moe-vram-pool <MB>`、`--prompt-log <path>`；`common_params` 加对应字段 |
| `common/common.h/cpp` | 字段 + 默认值 + 传给 model params 的映射 |
| `tools/server/server-context.cpp` `load_model()`（:958）| model 加载前：`ggml_backend_reg` 注册 route B backend；`mparams.tensor_buft_overrides` 指向 route B weight buft（复用 `src/backend/moe_backend` 的注册逻辑）|
| `tools/server/server-context.cpp` / `tools/server/server.cpp` | `--prompt-log`：在 `/v1/chat/completions` 入口追加请求体（~10 行，挂 server-http 回调）|
| `common/arg.cpp`（可选）| `--profile-log`：若保留 JSONL，在 server task 完成处回调 `src/profile/profiler` |
| `CMakeLists.txt` / `build.bat` | 构建目标改为 `llama-cli llama-server` + 链接 route B 静态库 |

### 4.3 用我们的实现替换/补充
| 对象 | 替换为 |
|---|---|
| `src/server/http_server.*` | 原版 `tools/server/server-http.*`（删）|
| `src/server_main.cpp` / `src/main.cpp` | 原版 `tools/server/main.cpp` / `llama-cli`（删）|
| `src/engine/llama_engine.*` | 原版推理循环 + route B backend 注入（删；如 trace/导出需要，迁 `diagnostics/`）|
| `src/backend/*`、`src/io/*`、`src/pool/*`、`src/loader/*` | **保留**（route B 插件本体）|
| `src/common/{types,logger,crash}` | 保留（crash 兜底可注入原版入口）|
| `--kv-placement`/`--cache-type`/`--no-swa-full` | 原版 `-ngl`/`--cache-type-k/-v`/`--swa-full` |

---

## 5. 构建：models/*.cpp 保留 + 全设备后端可配置

- **models/*.cpp**：保持 `GLOB`（全部 150 个编译进 libllama，不精简）。
- **设备后端可配置**：`build.bat` / `CMakeLists.txt` 把 vendored 的 ggml 后端开关暴露为可选参数（当前默认全关=CPU）：
  - CUDA `-DGGML_CUDA=ON`、HIP `-DGGML_HIP=ON`、Metal `-DGGML_METAL=ON`、SYCL `-DGGML_SYCL=ON`、Vulkan `-DGGML_VULKAN=ON`（均需对应 SDK/工具链；本机 RX 590 走 Vulkan，Phase B 开 `-DGGML_VULKAN=ON`）。
  - 同时保留 OpenMP/NATIVE 现状；开关写成 `build.bat [llamalibs|build|test] <tag> [--vulkan] [--cuda] ...` 或环境变量透传。
- `llama-common-base` 冗余构建：改为 `llama-cli llama-server` 直接构建并链接 `llama-common`（不再单独 build 未链接的 base）。

---

## 6. route B 注入设计（小改核心）

```text
llama-server --expert-backend --moe-ram-pool 71680 -m deepseek4.gguf
  -> common/arg.cpp 解析 -> common_params.expert_backend / moe_ram_pool
  -> server-context.cpp load_model():
       if (expert_backend):
          route_b_init(pool_mb)          // 分配槽池 + 注册 ggml backend/buft（复用 moe_backend）
          mparams.tensor_buft_overrides = route_b_overrides(ffn_*_exps / ffn_*_shexp)
       llama_model_load_from_file(...)   // 原版加载，exps 走自定义 buft（no-op set_tensor）
       llama_new_context_with_model(...) // 原版；MUL_MAT_ID 派给我们的 backend
```

- 图/数学/调度：零改动（与现有 route B 一致，已验证数值等价）。
- 错误处理：注入失败在 load_model 阶段报错退出（对齐原版加载失败语义）。
- crash 兜底：入口处 install_crash_handlers()（server 版 + cli 版各一行）。

---

## 7. 实施步骤（里程碑）

1. **M0 论证/冻结**：本计划评审通过；SAMPLING.md/start_server.bat 参数同步为原版命名。
2. **M1 构建接入**：`build.bat` 构建 `llama-cli llama-server`（链接 llama-common），产出到 `build/<tag>/bin/`；跑通原版 server 加载 deepseek4（无 route B）。
3. **M2 参数迁移**：删我们 `--cache-type/--no-swa-full/--kv-placement`；脚本/文档改用原版 `--cache-type-k/-v/--swa-full/-ngl`；验证 KV 报告（原版 `/metrics` 与启动日志 cells）。
4. **M3 route B 注入**：common/arg.cpp 4 参数 + server-context 注入；`--expert-backend` 跑通并复跑数值等价回归（compare_trace / verify_prefill）。
5. **M4 收编**：删 `http_server/server_main/main/llama_engine`；profile/prompt-log 按 §4.2 挂钩；crash 兜底接入；tests 适配（原版参数名）。
6. **M5 长程回归**：`run_long_horizon_test.bat` / `run_prefill_verify.bat` 改用原版 server 命令 + `--swa-full` 与否两组对比。
7. **M6 设备后端**：CMake/构建暴露 GGML_VULKAN/CUDA/... 开关（本机先 Vulkan）。

---

## 8. 风险与取舍

- **注入点稳定性**：server-context.cpp 内部 API 随上游演进；注入面收敛到一个函数（load_model），升级时只查一处。
- **原版 server 依赖**：`llama-server` 需要 `llama-common` + `llama-common-server`（server 特有模块），构建链接全量 common——比现在 exe 大，可接受。
- **测试/脚本适配**：原版 server 参数/输出格式变化，`bench_agent.js`/`chat_cli.js` 基本兼容（OpenAI API 不变），bat 命令需重写参数。
- **engine 删除**：`LLM_EXPORT_DIR` 导出（任务一/二）依赖 llama_engine 的 patch——迁移到"注入原版 server + 原 patch"或临时保留 diagnostics 工具。
- **route B 与原版 server 多槽**：`--parallel > 1` 时槽池并发命中局部性放大是 bonus；首期保持 `--parallel 1` 验证。

---

## 9. 决策点（评审时拍板）

1. 删除 `llama_engine` 的导出功能（任务一/二）降级为 diagnostics 工具，还是迁移到原版 server 注入？
2. `--profile-log`（JSONL 遥测）：保留自定义 JSONL，还是直接用原版 `--metrics`（Prometheus）？
3. 首期是否直接支持 `--parallel` 多槽，还是先 `--parallel 1`？
4. models/*.cpp 保留（已定）；设备后端开关是否首期就加 Vulkan？
