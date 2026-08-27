[English](SAMPLING.md) | [简体中文](SAMPLING.zh-CN.md)

# 采样参数建议

StreamMoE 面向 **agentic / 长推理** 用途；以下遵循 DeepSeek 官方部署建议：

| 场景 | temperature | top_p | 最大输出长度 |
| :--- | :--- | :--- | :--- |
| Agentic / 工具调用 / 长推理 | **1.0** | **0.95** | high/max effort：**384K tokens** |
| 非 agentic（普通对话/续写） | 1.0 | 1.0 | 按需 |

## 落地位置

- `scripts/start_server.bat`：运行**上游 llama-server**，以 `--temp 1.0 --top-p 0.95 --n-predict 384000 --no-warmup`（`--no-warmup` 跳过启动时空跑前向——MoE 下避免启动时冷盘专家装载，专家改为首次请求时装载）。
- 请求里的 `max_tokens` 只能**下调**服务端上限（`min(server --n-predict, request.max_tokens)`）。
- `scripts/run_long_horizon_test.bat` / `scripts/run_prefill_verify.bat` 同样用 `--temp 1.0 --top-p 0.95`。

## KV Cache 与上下文（推荐默认）

长上下文场景下默认启动脚本还设置：

- `-c 1048576`：模型原生最大上下文（1,048,576）。
- `--cache-type-k q8_0 --cache-type-v q8_0`：量化 KV cache（上游 `-ctk/-ctv`），相对 f16 减半。
- SWA cache 默认**窗口式**（上游 `common_params.swa_full=false`）；`--swa-full` 才开启全尺寸。窗口式符合 DeepSeek-V4 设计意图（滑窗 raw KV + CSA/HCA 压缩层长程）。

实测 `-c 1048576` 的 KV 占用（llama.cpp `memory_breakdown`）：

| 配置 | KV 内存 |
| :--- | :--- |
| f16 + 全尺寸 SWA（`--swa-full`）| ~49.7 GB |
| **q8_0 + 窗口式（默认）** | **~3.6 GB** |

> 注：窗口式 SWA 不改滑窗注意力 mask（计算语义不变），只把 raw KV 存储改为窗口式。短 prompt 冒烟输出一致——长依赖生成仍需留意。

## 理由

- temperature 1.0 保留完整概率质量，利于多步长推理。
- top_p 0.95（nucleus）只裁剪不合理的尾部，保留 agentic 循环所需的多样性。
- 384K token 上限避免截断 high/max reasoning effort 输出；本机 decode 为 CPU 受限（0.5-3 tok/s），超长生成本身就慢——上限只是天花板，模型在 EOS 自然停止。
