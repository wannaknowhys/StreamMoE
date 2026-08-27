[English](SAMPLING.md) | [简体中文](SAMPLING.zh-CN.md)

# Sampling Parameter Recommendations

StreamMoE targets **agentic / long-reasoning use**; the following follow the
DeepSeek official deployment guidance:

| Scenario | temperature | top_p | max output length |
| :--- | :--- | :--- | :--- |
| Agentic / tool-use / long reasoning | **1.0** | **0.95** | high/max effort: **384K tokens** |
| Non-agentic (plain chat / generation) | 1.0 | 1.0 | as needed |

## Where it is applied

- `scripts/start_server.bat`: starts the server with `--temp 1.0 --top-p 0.95 -n 384000`.
- The per-request `max_tokens` field can only **lower** the server cap
  (`min(server -n, request.max_tokens)`).
- `scripts/run_long_horizon_test.bat` / `scripts/run_prefill_verify.bat` also use
  `--temp 1.0 --top-p 0.95`.

## KV cache & context (recommended defaults)

For long-context use the default launcher also sets:

- `-c 1048576`: the model's native max context (1,048,576).
- `--cache-type q8_0`: quantized KV cache (K=V; deepseek4/MLA requires equal
  types). Halves the KV footprint vs f16.
- `--no-swa-full`: windowed raw cache (n_swa=128) instead of full-size SWA.
  This matches the DeepSeek-V4 design intent (sliding-window raw KV +
  compressed long-range via the CSA/HCA layers). See `docs/BUG_TRACKER.md`
  or the KV investigation notes (CHECKPOINT) for details.

Measured KV footprint at `-c 1048576` (llama.cpp `memory_breakdown`):

| config | KV memory |
| :--- | :--- |
| f16 + full-size SWA (upstream default) | ~49.7 GB |
| **q8_0 + `--no-swa-full` (recommended)** | **~3.6 GB** |

> Note: `--no-swa-full` keeps the sliding-window attention mask unchanged (the
> computation semantics are the same); only the raw KV storage is windowed.
> Long-context output was spot-checked consistent on short prompts - keep an
> eye on long-dependency generations.

## Why

- temperature 1.0 keeps the full probability mass for long multi-step reasoning.
- top_p 0.95 (nucleus) trims only the implausible tail while preserving diversity
  that agentic loops need.
- 384K-token cap avoids truncating high/max reasoning-effort outputs; on this
  machine decoding is CPU-bound (0.5-3 tok/s), so very long generations are slow
  by design - the cap is an upper bound, the model stops at EOS naturally.
