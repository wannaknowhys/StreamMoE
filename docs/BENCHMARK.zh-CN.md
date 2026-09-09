# 布局基准测试 - run_bench.js

[English](BENCHMARK.md) | [简体中文](BENCHMARK.zh-CN.md)

> 状态：**2026-09-09 落地**。测量 C1（逐层 dense）、C2（全局 dense）、专家池
> 常驻 VRAM 单独或组合时吞吐提升多少，并与纯粹原版 llama.cpp 对比。
>
> 相关：`docs/DENSE_PLACEMENT.md`（`--dense-placement`）、`docs/ROUTE_B_GPU_PHASE.md`
>（专家池 / `--moe-expert-pools`）、`tools/run_export.js`（同一套 spec 模型）、
> `docs/TEST_FLOW.md`。

## 1. 指标

`llama-server` 自带 `timings`，runner 读两个数：

- **pp** = prompt processing（prefill，预填充），`timings.prompt_per_second`。
- **tg** = token generation（decode，生成），`timings.predicted_per_second`。

两类任务：

- **single**：第 1 次请求 = **冷启动**（含从磁盘冷载专家），随后 `warmup` 次丢弃，
  再 `repeat` 次取**中位数**。走 `/completion` + `cache_prompt=false`，所以每轮都
  全量重算 prefill。
- **jsonl**：多轮链式，走 `/v1/chat/completions` + `cache_prompt=true`。每轮一条
  JSONL 记录 + 一行屏幕输出，最后一条 summary 记录聚合结果。`prompt_tokens` 是
  该轮**增量**（前缀复用，见 `cache_n`），不是累计上下文长度。

route B 与纯粹原版用同一套方法学。

## 2. Runner

`tools/run_bench.js` 复用 `run_export.js` 的 model/engine/task 笛卡尔 + `${VAR}`
环境变量展开。

```
node tools/run_bench.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]>
                        [--port N] [--health-timeout S] [--out FILE] [--dry-run]
```

- 环境变量：`SM_BENCH_PORT`（默认 8994）、`SM_BENCH_OUT`（默认
  `benchmark/results/bench_<ts>.jsonl`）。
- 每个 run 起一个 server 进程（placement 是进程级参数）。
- 失败（health 超时、提前退出、请求出错）记一条 `FAIL` 后**继续**——这正是矩阵
  容量感知的来源。
- 每个 run 记录 `server_log_tail`，便于查 offload / OOM。
- 结果逐条追加，部分跑批也能留存。

## 3. Spec 结构

三类互斥 spec，跨类重复键直接报错。

| 类别 | 键 |
| :--- | :--- |
| model | `model`、`modelPath`、`draft?`、`pool?` |
| engine | `engine`、`bin` \| `binPath`、`extra?` |
| task | `input`，加下面的 single/jsonl 字段 |

engine = 二进制 + placement 参数：

- route B：`"bin": "StreamMoE"`，
  `extra = ["--expert-backend","--fit","off","--moe-expert-pools","RAM:${pool}[,Vulkan0:N]","--dense-placement","C1:<dev>,C2:<dev>"]`
- 原版：`"binPath": "${SM_UPSTREAM_BIN}"`，
  `extra = ["--fit","off","--n-gpu-layers","<N>"]`

task 字段：

- single：`prompt` | `promptFile`、`promptRepeat?`、`nPredict?`、`warmup?`、
  `repeat?`、`ctx?`、`threads?`
- jsonl：`feed:{type:"jsonl", path, maxTurns}`、`nPredict?`、`ctx?`、`threads?`
- prefill：`feed:{type:"prefill", path?, tokens?}`、`nPredict?`、`ctx?`、`threads?`。
  `path` 是 JSON 消息数组，会扁平化成纯文本走 `/completion`（模型无关，tool_calls
  模板不会把它顶掉）；`tokens` 是无 path 时的合成 filler 兜底。`ctx` 不设时按 prompt
  自动放大；但模型 `n_ctx_train` 仍会封顶 slot（olmoe = 4096，所以 `prefill10000`
  要用长上下文模型）。

## 4. 布局矩阵（`tools/run_specs/engines/`）

所有 route-B spec 用 `bin: StreamMoE`；`${pool}` 来自 model spec。

| Spec | `--dense-placement` | `--moe-expert-pools` |
| :--- | :--- | :--- |
| `place-cpu.json` | `C1:RAM,C2:RAM` | `RAM:${pool}` |
| `place-c1.json` | `C1:Vulkan0,C2:RAM` | `RAM:${pool}` |
| `place-c2.json` | `C1:RAM,C2:Vulkan0` | `RAM:${pool}` |
| `place-exp2.json` | `C1:RAM,C2:RAM` | `RAM:${pool},Vulkan0:2048` |
| `place-exp5.json` | `C1:RAM,C2:RAM` | `RAM:${pool},Vulkan0:5120` |
| `place-c1-exp2.json` | `C1:Vulkan0,C2:RAM` | `RAM:${pool},Vulkan0:2048` |
| `place-c2-exp2.json` | `C1:RAM,C2:Vulkan0` | `RAM:${pool},Vulkan0:2048` |
| `place-c1c2.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool}` |
| `place-c1c2-exp2.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool},Vulkan0:2048` |
| `place-c1c2-exp1.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool},Vulkan0:1024` |
| `place-c2-exp5.json` | `C1:RAM,C2:Vulkan0` | `RAM:${pool},Vulkan0:5120` |
| `place-c1c2-exp5.json` | `C1:Vulkan0,C2:Vulkan0` | `RAM:${pool},Vulkan0:5120` |

纯粹原版：`stock-cpu.json`（`-ngl 0`）、`stock-vulkan.json`（`-ngl 99`），都走
`binPath: ${SM_UPSTREAM_BIN}`。

任务：`bench.json`（约 1k token prompt）、`bench_long.json`（`promptRepeat` 约
4k），另有现成的 `en.json` / `cn.json` 多轮集。

## 5. 私有 env（本机路径不入库）

`private/` 已 gitignore。`private/env.bat` 设模型路径、`SM_UPSTREAM_BIN`（原版
Vulkan 构建）、`SM_PREFILL_10K`（`prefill10000` 用的 ~10.4k token 聊天快照）和
`SM_BENCH_OUT`；`private/bench.bat` 加载后调用 runner。提交的模板是
`scripts/sm_env.example.bat`。

## 6. 结果结构（`benchmark/results/*.jsonl`，已 gitignore）

- single：`{kind:"single", cold, warmup[], steady[], median:{prompt_n,prompt_tps,decode_tps}, server_log_tail}`
- jsonl：每轮一条 `{kind:"turn", turn, prompt_tokens, prompt_tps, decode_tps, gen_n, cache_n}`，
  末尾 `{kind:"summary", turns, total_prompt_tokens, total_gen_tokens,
  avg_decode_tps, median_decode_tps, avg_prompt_tps, server_log_tail}`

## 7. 示例

```bat
rem olmoe 上原版 GPU vs CPU
private\bench.bat --models tools\run_specs\models\olmoe.json ^
  --engines tools\run_specs\engines\stock-cpu.json,tools\run_specs\engines\stock-vulkan.json ^
  --tasks tools\run_specs\tasks\bench.json

rem gemma 布局：全 CPU 基线 vs C1+C2+2G 专家进 VRAM
private\bench.bat --models tools\run_specs\models\gemma.json ^
  --engines tools\run_specs\engines\place-cpu.json,tools\run_specs\engines\place-c1c2-exp2.json ^
  --tasks tools\run_specs\tasks\bench_long.json
```

## 8. 注意事项与早期实测

- RX590 8GB（可用约 7GB）：gemma C1+C2 = 2.46GB 放得下；deepseek C1 = 9.9GB 放不
  下，所以 `C1:Vulkan0` 的格预期 `FAIL`（记下、跑批继续）。dense 格用 gemma，C2/专家
  格用 deepseek。
- dense 上 GPU 会重新打开 `op_offload`（Vulkan compute buffer 约 1.3GB），这笔和
  专家池抢同一块显存预算。
- 2026-09-09 冒烟（olmoe，8-token decode）：原版 CPU 52.5 tg vs 原版 Vulkan
  43.5 tg。用 `-lv 4` 确认了 `offloaded 17/17 layers to GPU`（Vulkan0 占 3961 MiB），
  显卡路径是真的；这么小的 decode 受启动开销主导。代表性数字请用 `bench.json` /
  `bench_long.json`。

## 9. 发现记录

### 2026-09-09 - olmoe (1B-7B)，en.json 10 轮，RX590 8GB

10 轮平均（tg = decode tok/s，pp = prefill tok/s）：

| engine | tg | pp | 布局 |
| :--- | ---: | ---: | :--- |
| stock-cpu | 42.38 | 102.3 | 原版，CPU |
| stock-vulkan | 41.32 | 221.9 | 原版，全部层 Vulkan0 |
| place-cpu | 40.00 | 171.5 | route B 基线（dense + 专家全 RAM） |
| place-c2 | 36.87 / 36.54 | 143.7 / 144.6 | C2 进 Vulkan0（跑了两次） |
| place-c1 | 19.50 | 188.4 | C1 进 Vulkan0 |
| place-c1c2 | 20.26 | 195.8 | C1+C2 进 Vulkan0 |
| place-exp2 | 27.03 | 101.3 | 专家 2G VRAM |
| place-exp5 | 25.42 | 140.9 | 专家 5G VRAM |
| place-c2-exp2 | 28.00 | 101.0 | C2 + 2G 专家 |
| place-c2-exp5 | 26.21 | 145.8 | C2 + 5G 专家 |
| place-c1-exp2 | 16.69 | 111.4 | C1 + 2G 专家 |
| place-c1c2-exp1 | 17.77 | 143.5 | C1+C2 + 1G 专家 |
| place-c1c2-exp2 | 17.12 | 109.1 | C1+C2 + 2G 专家 |
| place-c1c2-exp5 | 16.39 | 160.5 | C1+C2 + 5G 专家 |

初步判断（**不是结论**）：

- **prefill 明显吃 GPU 红利**：原版 102 -> 222 pp，C1+C2 196 pp。
- **decode 在这个模型上不吃**：RX590 与 16 线程 CPU 持平甚至更慢（原版 41.3 vs
  42.4），而 C1 进 VRAM、专家池进 VRAM 都出现较大 decode 损耗（C1 40 -> ~20；专家
  40 -> 25-27；叠加 40 -> 16-17）。
- 疑似原因：1B-active 模型上 GPU decode 本来就不快，加上 route B VRAM 池路径的
  跨设备/同步/搬迁开销；dense C1 上 GPU 还会重新打开 `op_offload`。
- **下一步：换 gemma 重测**（dense 更重、专家更多、C1+C2 能进 VRAM）后再下结论。
