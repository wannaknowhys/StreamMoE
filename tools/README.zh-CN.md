# StreamMoE 工具与运行器套件

[English](README.md) | [简体中文](README.zh-CN.md)

本目录包含 StreamMoE 的诊断工具、自动化运行器、性能基准套件及多维运行规范配置。

---

## 1. 目录结构

```
tools/
├── run_specs/                   # 4D 笛卡尔积执行矩阵规范
│   ├── flavors/                 # 产物类型与能力定义（Release、Dump、Dbg 等）
│   ├── models/                  # 模型路径与训练上下文上限
│   ├── engines/                 # 硬件放置拓扑（C1/C2 放置与 MoE 专家池）
│   └── tasks/                   # 评测任务（对话、Benchmark、Prefill 导出）
├── run_export.js                # 核心测试运行器（解析并执行 4D 笛卡尔积）
├── run_bench.js                 # 吞吐与延迟基准评测运行器
├── smoke_cli.js                 # CLI 混合池直跑端到端冒烟测试
├── freshness_check.js           # 二进制编译时间戳新鲜度拦截守卫
└── ...
```

---

## 2. 4D 笛卡尔积规范 (`run_specs/`)

运行器基于 4 个相互正交解耦的维度进行笛卡尔积组合测试：

### 2.1 Flavors (`tools/run_specs/flavors/<name>.json`)
定义目标可执行文件及其调试/导出能力：
- `StreamMoE.json`：生产 Release 产物（`hasExport: false`），纯推断满血吞吐。
- `StreamMoE_dump.json`：测试与回归产物，包含张量导出钩子（`hasExport: true`）。
- `StreamMoE_latest.json`：最新特性构建（`hasExport: true`）。
- `StreamMoE_dump_dbg.json`：调试诊断构建，包含细粒度日志（`hasExport: true`）。
- `upstream_dump.json` / `upstream_vulkan_dump.json`：官方 stock 基准参照。

### 2.2 Models (`tools/run_specs/models/<name>.json`)
定义模型文件与训练规格：
- `model`：模型短标识符（如 `olmoe`、`gemma`、`deepseek`）。
- `modelPath`：模型路径或环境变量引用（如 `${SM_OLMOE}`）。
- `modelCtx`：模型架构训练上下文限制（例如 OLMoE 为 `4096`，Gemma/DeepSeek 为 `8192`）。
- `pool`：推荐专家池大小（MB）。

### 2.3 Engines (`tools/run_specs/engines/<name>.json`)
定义硬件放置拓扑与内存分配。**彻底解耦二进制路径**：
- `engine`：拓扑名称（如 `place-cpu`、`place-c1c2-exp5`）。
- `extra`：命令行 placement 参数（`--expert-backend`、`--moe-expert-pools`、`--dense-placement`）。

### 2.4 Tasks (`tools/run_specs/tasks/<name>.json`)
定义评测 Prompt、对话集与上下文目标：
- `input`：任务标识符（如 `hi`、`en`、`cn`、`bench`、`prefill3000`）。
- `taskCtx`：任务期望上下文长度。
- `feed`：对话流配置（`jsonl` 或 `prefill`）。
- 生成参数（`temp`、`nPredict`）。

---

## 3. 运行器与诊断工具

### 3.1 `run_export.js`
解析并执行 flavors、models、engines 与 tasks 的笛卡尔积：
```bash
node tools/run_export.js --flavors <flavor> --models <model> --engines <engine> --tasks <task>
```
- **上下文双端协商**：自动计算 `effectiveCtx = Math.min(modelCtx, taskCtx)`。消除上下文溢出告警，防止多 slot 显存撑爆。
- **参数原子去重**：自动排重 `-c`、`--fit`、`--temp` 等核心参数。
- **条件式导出保护**：`--export-dir` 仅在 `flavor.hasExport === true` 且任务为 prefill 测试（`feed.type === 'prefill'`）时传递。常规对话与生成任务绝不传递，杜绝 decode 期间每 token 的 GPU 显卡同步与回读开销。

### 3.2 `run_bench.js`
评估 Prompt 处理速度（pp）与 Token 生成吞吐（tg）：
```bash
node tools/run_bench.js --models <spec> --engines <spec> --tasks <spec>
```

### 3.3 `smoke_cli.js`
在混合内存池（如 `RAM:8192,Vulkan0:5120`）下使用 `llama-cli` 直跑推理快速冒烟：
```bash
node tools/smoke_cli.js
```
验证输出文本的可读性与流畅度，防止因服务端 dump 验证遗漏 CLI 推理错乱。

### 3.4 `freshness_check.js`
对比二进制最后修改时间与 `src/` 及 `patches/` 源码时间：
```bash
node tools/freshness_check.js <path/to/binary>
```
若二进制早于源码修改时间则立即退出并报错（Exit code 1），强制必须重新编译后方可验证。
