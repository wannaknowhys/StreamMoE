# 依赖与源码地图 (DEPENDENCY_MAP.md)

> 用途：回答"vendored llama.cpp 哪些被用、哪些没用"与"我们自研文件各是什么"。维护者新增文件前先查本表。
> 更新：2026-08-27。基于 vendored llama.cpp @ f280b2698 + 当前 main 分支。

---

## 1. vendored llama.cpp 使用情况

### 1.1 编译进 libllama 并链接使用（exe 直接依赖）

构建：`build.bat llamalibs <tag>` → cmake+ninja over `third_party/llama.cpp`，`ninja llama` 产出静态库；StreamMoE exe 链接 `llama.lib + ggml.lib + ggml-base.lib + ggml-cpu.lib + libomp`（见 CMakeLists.txt `streammoe_target`）。

| 库 | 编译的源 | 说明 |
|---|---|---|
| `llama` | `src/llama.cpp` + `src/llama-*.cpp`（adapter/arch/batch/chat/context/cparams/grammar/graph/hparams/impl/io/kv-cache×7/memory×4/mmap/model-loader/model-saver/model/quant/sampler/vocab/unicode×2）| 真实推理核心；**我们的 KV cache 就是原版**（dsv4/iswa 等 7 个 kv-cache 文件全编译进库，运行时按架构选择）|
| `llama` | `src/models/*.cpp`（**GLOB 全部 150 个**）| 编译进库，但运行时仅 `deepseek4.cpp` 被实例化（LLM_ARCH_DEEPSEEK4）|
| `ggml` / `ggml-base` / `ggml-cpu` | `ggml/src/ggml.c/.cpp`、`ggml-alloc.c`、`ggml-backend.cpp`、`ggml-backend-meta.cpp`、`ggml-opt.cpp`、`ggml-threading.cpp`、`ggml-quants.c`、`gguf.cpp`、`ggml-backend-reg.cpp`、`ggml-backend-dl.cpp` | 图/张量/后端/量化/GGUF 解析；CPU backend（ggml-cpu）承担全部计算 |
| 公共头 | `include/llama.h`（公共 API）、`ggml/include/ggml.h` 等 | 仅头文件，编译期使用 |
| 一次性 patch | `src/` 内 `llama-kv-cache-dsv4.*`、`llama-context.*`、`deepseek4.cpp` 的导出钩子（LLM_EXPORT_DIR）| 见 `patches/prefill-export-llama.patch`；当前 vendored 已还原（无 patch）|

### 1.2 编译了但 exe 未使用（冗余，可优化）

| 对象 | 说明 | 建议 |
|---|---|---|
| `models/*.cpp` 除 `deepseek4.cpp` 外约 149 个 | GLOB 编译进 libllama，链接器会剔除未引用符号，但编译期耗时 | 可改为仅编译 deepseek4，暂不改（上游结构）|
| `llama-common-base`（`common/` 28 个 .cpp）| `build.bat` 显式 `ninja llama llama-common-base`，但 CMakeLists 未链接 common → 纯冗余构建 | 建议把 `llama-common-base` 从 build.bat 的 ninja 目标中移除 |

### 1.3 完全未编译/未使用

| 顶层对象 | 状态 |
|---|---|
| `tools/`（llama-server / llama-cli / llama-bench / llama-quantize / llama-perplexity / llama-gguf* 等）| `LLAMA_BUILD_TOOLS=OFF`，未编译（我们用自研 main/server 替代）|
| `examples/`（llama-eval / llama-parallel / llama-batched 等）| `LLAMA_BUILD_EXAMPLES=OFF`，未编译 |
| `tests/`（上游 UT）| `LLAMA_BUILD_TESTS=OFF`，未编译 |
| ggml 设备后端：CUDA / HIP / METAL / SYCL / Vulkan（`ggml_add_backend`）| 未启用（本机 RX 590 仅 Vulkan，Phase B 接入时开 Vulkan）|
| `gguf-py/`、`benches/`、`grammars/`、`conversion/`、`pocs/`、`skills/`、`app/`、`ci/`、`media/`、`models/`、`docs/`（上游）、`requirements/`、`scripts/`、`.devops/`、`.github/` | 未使用（其中 gguf-py 仅本地研究时用 uv 临时引入）|
| `vendor/` | 我们 include 路径引用（hash/sse 等），直接使用其中头文件 |

---

## 2. 我们自研的文件（src/）与功能

| 文件 | 功能 |
|---|---|
| `main.cpp` | CLI 入口：参数解析 → `llama_engine`；交互 REPL（`-i`）或单 prompt（`-p`）；采样覆盖 `--temp/--top-p/--top-k`；KV 选项 `--cache-type/--no-swa-full` |
| `server_main.cpp` | API server 入口：参数解析；GGUF 拓扑解析；banner 显示模型/池/上下文；**KV 实际内存用 `llama_get_memory_breakdown` 报告**；启动 `http_server`；Ctrl+C 优雅关闭 |
| `engine/llama_engine.h/.cpp` | **真实推理核心**：tokenize、chat template、KV 前缀复用、采样链（temp/top-p/top-k/seed）、decode、流式回调；`LLM_EXPORT_DIR` 导出钩子；`kv_memory_bytes()`；cparams 透传（type_k/v、swa_full） |
| `backend/moe_backend.h/.cpp` | **route B**：自定义 ggml backend + weight buft 注册（轻量句柄、no-op set_tensor、supports_op MUL_MAT_ID/MUL_MAT、graph_compute 分发）|
| `backend/minigraph_exec.h/.cpp` | **route B**：MUL_MAT_ID 委托执行——官方 `ggml_mul_mat_id` 内核 + 槽 stride 布局 + `MOE_ID_AT`（按 ids 真实 stride 读）+ b_leaf 叶子包装防祖先捕获 + pin/unpin |
| `backend/minigraph.h` | mini-graph 构造辅助（叶子/包装张量）|
| `backend/scheduler.h/.cpp` | **route B**：槽控制面（slot_meta/expert_directory/refcount/generation CAS）、DIO 装载、EST1 驱逐、命中遥测、pin 生命周期（首触 pin 末触 unpin）|
| `backend/slot.h` | 槽布局/64 位原子字定义、池常量、`MOE_ID_AT` 宏 |
| `backend/alloc.h` | 对齐/分配辅助 |
| `server/http_server.h/.cpp` | 轻量 C++ HTTP/SSE：`/v1/chat/completions`（stream:true）、`/v1/models`、`/health`、`/stats`；`--prompt-log`；handler 异常回 400 + 写 crash log |
| `loader/moe_loader.h/.cpp` | GGUF 拓扑解析：架构/层/专家/read plan、KV 元数据（is_mla/kv_lora_rank/compress_ratios）、分片校验 |
| `profile/profiler.h/.cpp` | RDTSCP 纳秒计时 + 定长 JSONL 遥测（turn_id/tps/hits/timings）|
| `common/types.h` | 对齐/内存发现/线程探测 |
| `common/logger.h` | 分级日志（INFO/WARN/ERROR）|
| `common/crash.h/.cpp` | 全局崩溃兜底：SEH/signal/terminate 捕获 + CaptureStackBackTrace 栈 + `temp/stream_moe_fatal.log`，`_Exit(1)` |
| `io/async_dio.h` + `async_dio_win.cpp` + `async_dio_posix.cpp` | 扇区对齐 DIO：Windows IOCP 真异步；POSIX 同步 pread（B22 io_uring 待办）|
| `io/staging_reader.h/.cpp` | 4KB 扇区对齐读计划 / staging 大小计算 |
| `pool/expert_stats.h/.cpp` | EST1 热度：recency-weighted decaying counter（读时归一化）|

### 2.1 tests/（UT，`build.bat test`）

| 文件 | 覆盖 |
|---|---|
| `test_async_dio.cpp` | IOCP 真异步 + 真实 GGUF 读 |
| `test_moe_loader.cpp` | GGUF 拓扑/分片解析 |
| `test_profiler.cpp` | JSONL 序列化 |
| `test_scheduler.cpp` | route B 槽控制面（2/2）|
| `test_slot.cpp` | route B 槽语义（4/4）|

### 2.2 tools/（Node 工具）

| 文件 | 功能 |
|---|---|
| `compare_trace.js` | base vs moe 逐层 trace 逐位对比（数值等价回归）|
| `verify_prefill.js` | prefill 交叉验证（cos+maxAbs 双门槛）|
| `simulate_cache.js` | 专家访问历史重放 → LRU/LFU/EST1/OPT 命中率曲线 |
| `chat_cli.js` | OpenAI 兼容交互客户端（真流式 SSE，/quit /reset /stats）|
| `bench_agent.js` | 多轮基准客户端（累积上下文）|

### 2.3 scripts/（批处理）

| 文件 | 功能 |
|---|---|
| `start_server.bat` | 启动 server（`--temp 1.0 --top-p 0.95 -c 1048576 -n 384000 --cache-type q8_0 --no-swa-full`，推荐默认）|
| `run_long_horizon_test.bat` | 整轮 long-horizon 基准（en/zh）|
| `run_prefill_verify.bat` | 批量 prefill/专家历史验证（std→moe→verify→simulate）|
| `verify_prefill.bat` | 单 prompt 快速 prefill 交叉验证 |

### 2.4 其他

- `diagnostics/trace_dump.*`：`STREAM_MOE_TEMP` 下每层张量 trace 回调（`stream_moe_trace_cb`），见 `diagnostics/README.md`。
- `patches/`：一次性补丁（prefill-export×2、memwatch×2），用法见 `patches/README.md`。
- `CMakeLists.txt` / `build.bat` / `Makefile`：构建规则唯一来源 + Windows/POSIX 薄壳转发。

---

## 3. 快速判断规则

- 新增上游能力（KV 类型、draft、Vulkan）优先走 **llama.cpp 公共 API**（`llama.h` / `llama-ext.h`），避免改 vendored。
- vendored 只在一次性实验时临时 patch（`patches/`），用完还原；**llama.cpp 的 AGENTS.md 禁止向子模块提交**。
- 自己写的东西只进 `src/`（或 `tools/`/`diagnostics/`），不再进 `src/scheduler|kv|tokenizer`（已删）。
