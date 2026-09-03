# Baseline Regression（基线回归）

> 每次改代码 + 编译完后跑一遍，确认 129-token prefill-from 产物仍与 known-good 基线一致。
> 用法：仓库根跑 `baseline_regression\run_baseline.bat`（需先 `build.bat llamalibs StreamMoE_dump` 与 `upstream_dump`）。

## 布局

```
baseline_regression/
  baseline/
    moe_129_8192/     known-good：moe(route-B, 8GB 池) 129-token prefill-from 产物
    upstream_129/     known-good：upstream(无 route-B) 129-token prefill-from 产物
  tools/              verify_prefill.js / verify_expert_history.js / kv_cos.js /
                      verify_kl.cpp（自编 C++ 工具，bat 自动编译出 exe）
  run_baseline.bat    全套：跑 moe+upstream prefill-from -> 对基线比较 -> 结论
  temp/               （gitignored）本轮产物与日志
```

- baseline 产物是**大二进制，gitignored**（本地盘持有，等同模型盘；换机/重拉后重建见下）。
- 喂入 tokens = `baseline/upstream_129/tokens_id.bin`（129 tokens，与基线同输入）。

## run_baseline.bat 检查项与期望

| # | 检查 | 期望 |
|---|---|---|
| 3a | verify_prefill moe vs baseline moe | **IDENTICAL**（embd/hidden/KV 逐字节同）——池大小/verify 等改动不得影响数值 |
| 3b | verify_expert_history moe vs baseline | **IDENTICAL**（路由逐条目同）|
| 3c | verify_prefill upstream vs baseline upstream | **IDENTICAL** |
| 4 | verify_kl baseline-upstream vs new moe | 报告 per-token KL（thresh 1e-2 宽松；同路由 0.9996 / 路由翻放大的固有噪声下不要当 bug，见 docs/BACKEND_DIVERGENCE_ANALYSIS.md）|
| 5 | kv_cos baseline moe vs new moe | 全 ~1.0（IDENTICAL 保证）|

**PASS 判定 = 3a/3b/3c 全 IDENTICAL**；4/5 是报告参考。

## 重建基线（known-good 更新时）

```bat
rem 用当时验证过的 build 跑两遍 prefill-from 产物拷入
build\StreamMoE_dump\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --prefill-from baseline_regression\baseline\upstream_129\tokens_id.bin --export-dir <tmp_moe> -c 2048 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup
build\upstream_dump\llama-build\bin\llama-server.exe -m ... --prefill-from ... --export-dir <tmp_up> -c 2048 -t 16
rem 拷 prefill_export_main.bin / expert_history_main.bin / prefill_meta.json / tokens_id.bin 到 baseline\<moe_129_8192|upstream_129>\
```

## verify_kl 单独用法

`tools\verify_kl.exe <model.gguf> <ref.bin> <cand.bin> [--thresh T]`——源 `tools/verify_kl.cpp`，用法细节见 `tools/verify_kl.md`（编译由 bat 自动；手动编译见该 md）。
