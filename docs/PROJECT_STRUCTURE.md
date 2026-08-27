# StreamMoE 项目结构与产物布局 (PROJECT_STRUCTURE.md)

> 本文件定义仓库目录规范、编译产物子路径模式、测试/结果归档约定。所有新文件必须遵循。

---

## 1. 顶层目录

```text
StreamMoE/
├── README.md / README.zh-CN.md   双语入口（唯一保留在根的中文/英文文档对）
├── build.bat                     构建入口（Windows）
├── Makefile                      构建入口（Linux）
├── docs/                         设计文档（全部 md 集中于此，见 §2）
├── scripts/                      运行脚本（.bat，见 §3）
├── benchmark/
│   ├── prompts/                  测试 prompt 数据集（.jsonl）
│   └── results/                  测试结果归档（见 §4）
├── patches/                      一次性诊断/实验补丁（见 §5）
├── src/                          源码（见 §6）
├── tests/                        UT 单元测试（见 §7）
├── tools/                        Node/脚本工具（bench_agent.js 等）
├── third_party/                  vendored 依赖（llama.cpp 子模块）
└── build/                        编译产物（gitignored，见 §8）
```

## 2. docs/（设计文档集中）

| 文件 | 内容 |
|---|---|
| `Backend.md` | DeepSeek4 自定义 backend / expert pool 调度设计（用户原始架构文档）|
| `LLAMA_MOE_NO_MMAP_RESEARCH.md` | MoE 去 mmap 可行性研究 + route B 实现要点（含 pin 生命周期 §4.8、shexp §4.9）|
| `EXPERT_OFFLOAD_INTEGRATION.md` | llama.cpp 集成追踪早期分析 |
| `DESIGN_REVIEW.md` | 各设计文档矛盾与待决策问题 |
| `BUG_TRACKER.md` | bug 追踪清单（P0/P1/P2/P3 + INC 事故记录 + 修复批次）|
| `CODEBASE_AUDIT.md` | 代码库四类划分（mock/可用/废弃/变了但函数没问题）|
| `TEST_FLOW.md` | 测试流程规范（单 prompt 优先 → .bat 整轮 → 用户手动盯内存）|
| `PROJECT_STRUCTURE.md` | 本文件 |
| `DEPENDENCY_MAP.md` | vendored llama.cpp 使用情况（编译/冗余/未用）+ 自研文件功能表 |
| `UPSTREAM_TOOLS_MIGRATION.md` | 迁移到原版 llama-cli/llama-server 的重构计划（route B 插件注入）|
| `TODO.md` / `LLAMA_EXE_ROADMAP.md` | 待办 / 可执行程序路线图 |

**约定**：所有文档 UTF-8；编辑只用 write/edit 工具，**严禁 PowerShell Set-Content 追加中文**（会破坏编码）。

## 3. scripts/（运行脚本）

| 文件 | 用途 |
|---|---|
| `start_server.bat` | 启动 API server（`--temp 1.0 --top-p 0.95 -n 384000 -c 1048576`，见 docs/SAMPLING.md），`--prompt-log temp\server_prompts.log` |
| `run_long_horizon_test.bat` | 整轮 long-horizon 基准（en/zh），用户手动运行、盯内存 |
| `run_prefill_verify.bat` | 批量 Prefill/专家历史验证（std → moe 连续跑 jsonl → verify_prefill → simulate_cache）|
| `verify_prefill.bat` | 单 prompt 快速 Prefill 交叉验证（std vs moe 导出 + verify_prefill）|

其余 mock 时代 runner（run_benchmark_experiments / run_chinese_benchmark / run_pure_lru_benchmark）已删除——统一收敛为 `run_long_horizon_test.bat [en|zh] [build-tag]`。

## 4. benchmark/（prompts 与 results 分离）

- `benchmark/prompts/*.jsonl`：输入数据集（`long_horizon_prompts.jsonl` / `_zh.jsonl`）。
- `benchmark/results/`：每次运行的两件套按 `_<tag>` 后缀命名，同一次运行的产物同后缀：
  - `conversation_real_<tag>.txt`（完整对话转写）
  - `profile_real_<tag>.jsonl`（逐轮遥测）
- 过期的 mock 时代结果已删除（BENCHMARK_REPORT*.md / conversation_* / profile_70G_ram* 等）。

## 5. patches/（一次性补丁）

- `README.md`：补丁用法总览。
- `memwatch-ggml.patch` / `memwatch-build.patch`：内存哨兵（详见 patches/README.md）。
- **规则**：补丁不入主线；临时应用 → `build.bat build <tag>` → 用后 `git apply -R` 还原。不再用分支承载。

## 6. src/（源码，route A 重组后）

```text
src/
├── main.cpp               CLI 入口（llama_engine）
├── server_main.cpp        API server 入口
├── common/                types.h（对齐/内存发现）、logger.h
├── backend/               route B：moe_backend（buft/backend 注册）、minigraph_exec（MUL_MAT_ID 委托）、scheduler（槽控制面/DIO/EST1）
├── engine/                llama_engine（真实推理核心）
├── io/                    async_dio（Win IOCP 真异步）、staging_reader（扇区对齐读计划）
├── loader/                moe_loader（GGUF 拓扑 + 专家 read plan）
├── pool/                  expert_stats（EST1 热度）
├── profile/               profiler（RDTSCP + JSONL）
└── server/                http_server（OpenAI 兼容 + SSE）
```

已删除 mock/废弃模块：`src/scheduler/`、`src/kv/`、`src/tokenizer/`、`engine/subgraph_executor`、`engine/speculative_engine`、`engine/state_machine`、`pool/expert_pool`。可救逻辑（驱逐算法、状态策略表、SMKV、槽重绑定概念）见 docs/CODEBASE_AUDIT.md §3。

## 7. tests/（UT）

- `test_async_dio.cpp` / `test_moe_loader.cpp` / `test_profiler.cpp` / `test_scheduler.cpp` / `test_slot.cpp`。
- 已删除 mock 时代模块的 UT（pool/scheduler/state_machine/tokenizer/kv）。
- 运行：`build.bat test <tag>`（CMake/ctest，产物在 `build\<tag>\bin\`，5/5 通过）。

## 8. build/（产物，多版本共存）

**子路径模式**：`build\<tag>\`，`<tag>` 描述构建风味/版本，例如：

| tag | 含义 |
|---|---|
| `main` | 主线默认构建（`build.bat build`）|
| `memwatch` | 应用了内存哨兵补丁的构建 |
| `v0.2` / 任意名称 | 版本/实验标记 |

每个 `build\<tag>\` 固定含：

```text
build\<tag>\
├── bin\            可执行文件 + libomp.dll
├── cmake\          构建中间产物（CMake/Ninja，含 ctest）
├── llama-build\    vendored libllama 的 CMake 构建目录（.lib 等）
└── server_<tag>.log  （runner 落 server 日志于此）
```

- 用 `build.bat build <tag>` 构建（缺 tag 默认 `main`）。
- 不同 tag 互不干扰，可共存；`build.bat clean` 一次性清空整个 build/。
- `build/` 已被 .gitignore 排除，永不入库。

## 9. 变更纪律

1. 每个逻辑变更独立 commit + push，消息用 conventional style。
2. 文档改动进 main；补丁进 patches/（不入 main 逻辑）。
3. 临时文件/产物进 `build\<tag>\` 或系统 %TEMP%，严禁散落根目录。
4. 删除文件用 `git rm`（保留历史），不要直接 `Remove-Item` 后失联。

## 10. 临时代码规范（已修复 bug 的打桩/诊断代码）

- **优先：独立文件夹 + 独立 .cpp**——短期诊断代码放 `diagnostics/` 下自成一文件（如 `trace_dump.cpp`），单独编译、不进入主构建、不污染 `src/`。该文件夹的代码可直接进 git 跟踪。
- **次选：宏包裹**——只有**必须写进主体代码文件**的（如 llama_engine 里 1 行 cb_eval 钩子），才用 `STREAM_MOE_TEMP` 宏包裹（`#ifdef STREAM_MOE_TEMP`），默认编译不带该宏。
- **长期有意义的诊断** → 做成 `patches/` 里的独立 patch（如 memwatch），不入主线。
- 已修复 bug 的验证代码用完即删（`git rm`）。
- 命名：`STREAM_MOE_TEMP` 为总开关；细化用 `STREAM_MOE_TEMP_<NAME>` 子宏，受总开关约束。
- 诊断编译：`diagnostics/README.md` 说明各自的构建命令。
