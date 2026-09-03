# Baseline Regression（基线回归）

> 每次改代码 + 编译完后跑一遍，确认 129-token prefill-from 产物仍与 known-good 基线一致。
> 用法：仓库根 `baseline_regression\run_baseline.bat [基线moe目录] [待验证目录]`（需先 `build.bat llamalibs StreamMoE_dump` 与 `upstream_dump`）。

## 数值形态（flavor）与基线选择——先读

- **CPU-only 构建**（`GGML_VULKAN=OFF` 重编的 StreamMoE_dump）数值 = CPU 基线：用默认基线 `moe_129_8192`。
- **默认构建（`GGML_VULKAN=ON`）只有匹配自己的数值形态**：`--expert-backend` 隐含 `--no-op-offload`，但 `GGML_VULKAN=ON` 编译本身仍改变数值（CPU backend 的 buft 换成 Vulkan0 host buft 等 host 内存形态差异，gate 边界产生 expert-flip 级噪声）——**默认 vulkan 构建请用 `moe_129_8192_vk` 作基线**。
- 两个 moe 基线之间是 flip 型差异（div_match 归因 0 unexplained，非 bug）；upstream 对照永远用固定的 `upstream_129`（CPU 干净上游，作 KL 参照）。

## 布局

```
baseline_regression/
  baseline/
    moe_129_8192/     known-good：moe(route-B, 8GB 池) 129-token 产物 —— CPU 基线
    moe_129_8192_vk/  known-good：同款产物 —— GGML_VULKAN=ON 构建基线
    upstream_129/     known-good：upstream(无 route-B) 129-token 产物（CPU 干净，固定参照）
  tools/              verify_prefill.js / verify_expert_history.js / kv_cos.js /
                      div_match.js / verify_kl.cpp（自编 C++ 工具，bat 自动编译出 exe）
  run_baseline.bat    全套：跑 moe+upstream prefill-from -> 对基线比较 -> 结论
  temp/               （gitignored）本轮产物与日志
```

- baseline 产物是**大二进制，gitignored**（本地盘持有，等同模型盘；换机/重拉后重建见下）。
- 喂入 tokens = `baseline/upstream_129/tokens_id.bin`（129 tokens，与基线同输入）。

## run_baseline.bat 参数与检查项

```
run_baseline.bat [baseline_moe_dir] [verify_dir]
  baseline_moe_dir   moe 参考基线目录（默认 baseline\moe_129_8192；vulkan 构建用 moe_129_8192_vk）
  verify_dir         本轮 moe/up 导出写入目录（默认 baseline_regression\temp）
```

| # | 检查 | 期望 |
|---|---|---|
| 3a | verify_prefill moe vs baseline moe | **IDENTICAL**（embd/hidden/KV 逐字节同）——池大小/verify 等改动不得影响数值 |
| 3b | verify_expert_history moe vs baseline | **IDENTICAL**（路由逐条目同）|
| 3c | verify_prefill upstream vs baseline upstream | **IDENTICAL** |
| 4 | verify_kl baseline-upstream vs new moe | 报告 per-token KL（`moe_129_8192` thresh 1.0；`*_vk` 基线自动放宽到 4.5——vk flavor 相对 CPU upstream 参照恒虚高）|
| 5 | kv_cos baseline moe vs new moe | 全 ~1.0（IDENTICAL 保证）|

**PASS 判定 = 3a/3b/3c 全 IDENTICAL**；4/5 是报告参考。moe 与基线 flavor 不匹配时 3a/3b 会 DIVERGED（div_match 归因 0 unexplained）——选对基线即可。

## 重建基线（known-good 更新时）

```bat
rem 用当时验证过的 build 跑两遍 prefill-from 产物拷入
build\StreamMoE_dump\llama-build\bin\llama-server.exe -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf --prefill-from baseline_regression\baseline\upstream_129\tokens_id.bin --export-dir <tmp_moe> -c 2048 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup
build\upstream_dump\llama-build\bin\llama-server.exe -m ... --prefill-from ... --export-dir <tmp_up> -c 2048 -t 16
rem 拷 prefill_export_main.bin / expert_history_main.bin / prefill_meta.json / tokens_id.bin 到
rem   baseline\<moe_129_8192 | moe_129_8192_vk | upstream_129>\  —— *_vk 目录只能由 GGML_VULKAN=ON 构建产物填
```

## verify_kl 单独用法

`tools\verify_kl.exe <model.gguf> <ref.bin> <cand.bin> [--thresh T]`——源 `tools/verify_kl.cpp`，用法细节见 `tools/verify_kl.md`（编译由 bat 自动；手动编译见该 md）。
