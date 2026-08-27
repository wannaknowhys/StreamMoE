[English](SAMPLING.md) | [简体中文](SAMPLING.zh-CN.md)

# 采样参数建议

StreamMoE 面向 **agentic / 长推理** 用途；以下遵循 DeepSeek 官方部署建议：

| 场景 | temperature | top_p | 最大输出长度 |
| :--- | :--- | :--- | :--- |
| Agentic / 工具调用 / 长推理 | **1.0** | **0.95** | high/max effort：**384K tokens** |
| 非 agentic（普通对话/续写） | 1.0 | 1.0 | 按需 |

## 落地位置

- `scripts/start_server.bat`：以 `--temp 1.0 --top-p 0.95 -n 384000` 启动。
- 请求里的 `max_tokens` 只能**下调**服务端上限（`min(server -n, request.max_tokens)`）。
- `scripts/run_long_horizon_test.bat` / `scripts/run_prefill_verify.bat` 同样用 `--temp 1.0 --top-p 0.95`。

## 理由

- temperature 1.0 保留完整概率质量，利于多步长推理。
- top_p 0.95（nucleus）只裁剪不合理的尾部，保留 agentic 循环所需的多样性。
- 384K token 上限避免截断 high/max reasoning effort 输出；本机 decode 为 CPU 受限（0.5-3 tok/s），超长生成本身就慢——上限只是天花板，模型在 EOS 自然停止。
