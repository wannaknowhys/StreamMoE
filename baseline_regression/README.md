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
    gemma_129_l0/     known-good：A 引擎（默认 exec_one_burst，无 CHAIN env）L0 per-node
                      MM_ONLY dump + 同源 tokens_id.bin —— 删 A 路径前冻结的每节点数值基准
                      （来源 temp/dump_c129b，2026-09-07；用于唯一收敛引擎的宽松 gate）
    deepseek_hi_up/   known-good：upstream（纯上游 llama CPU）deepseek UD-00001 `-p hi -st`
                      prefill-export（hi 完整对话流 95 tokens，PREFEXP2）—— 上游 CPU 参照
    deepseek_hi_moe/  known-good：同输入下桶引擎（唯一执行器，删 A 后）prefill-export
                      —— 引擎数值 gate 参照（hidden cos≈1.0 判据，见下）
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
| 3a | verify_prefill moe vs baseline moe | CPU 基线：**IDENTICAL**；`_vk` 基线：**cos 门**（embd cos≥0.99 的 token 比例 ≥ 90%，脚本打印 cos 区间比例）|
| 3b | verify_expert_history moe vs baseline | CPU 基线：**IDENTICAL**；`_vk` 基线：**仅报告**（路由 flip 是 vk flavor 噪声）|
| 3c | verify_prefill upstream vs baseline upstream | **IDENTICAL** |
| 4 | verify_kl baseline-upstream vs new moe | 报告 per-token KL（`moe_129_8192` thresh 1.0；`*_vk` 基线自动放宽到 4.5——vk flavor 相对 CPU upstream 参照恒虚高）|
| 5 | kv_cos baseline moe vs new moe | 全 ~1.0（IDENTICAL 保证；vk 下为参考）|

**PASS 判定**：3c upstream 必须 IDENTICAL；3a CPU 必须 IDENTICAL / `_vk` 必须过 cos 门；3b `_vk` 仅报告。4/5 是报告参考。`_vk` 的 cos 门参数在 `run_baseline.bat`（`--cos-floor 0.99 --min-ratio 0.9`），要收紧改这两个数即可。

## DeepSeek gate（cos 判据，非逐字节）

Gemma 的 run_baseline：CPU 基线**逐字节 IDENTICAL**；vk 基线（`moe_129_8192_vk`）因 host 内存
形态差异只能走 **cos 门**（`--cos-floor 0.99 --min-ratio 0.9`）。DeepSeek 因路由
expert-flip 噪声（gate 边界专家序号翻转，~8% 条目，累加结果不变），只能 **cos gate**：
- 参照 = `baseline/deepseek_hi_up`（upstream 纯 CPU llama，UD-00001，`-p hi -st` 生成流
  95 tokens 导出 prefill_export_main.bin）。
- 引擎产物放 `baseline/deepseek_hi_moe`，同 tokens_id.bin 喂入。
- 判据：`verify_prefill up vs moe` 的 **hidden cos ≈ 1.0（近 0.9999999）** = 桶引擎层输出与
  上游一致；embd 多数 >0.999；expert_history 允许翻转（div_match 归因 0 unexplained）；
  KV 因布局不同不做字节比较。文本层若走 `-p hi` 解码，翻转在下游放大可能产生不同措辞
  （"Hi there!" vs "Hello!"）——非引擎 bug，是 gate 噪声。
- 构建：`deepseek_hi_up` 只可由 upstream_dump 填；`deepseek_hi_moe` 由当前引擎
  （StreamMoE_dump_dbg 或唯一收敛引擎）填。两者必须用同 tokens_id.bin（来自 up_hi_prefill）。

## 已知现象：冻结 moe 基线与当前引擎的部分 token 差异（2026-09-08）

`baseline/moe_129_8192_vk`（及 CPU 版 `moe_129_8192`）与当前 HEAD 引擎的 embd/hidden 存在
**部分 token 不一致**（如 token#0 cos≈0.986）——这是 K5（SoA 池）/删 A 之后引擎与旧冻结基线
之间的既有 token 级差异，**非单次改动引入**；用户已知，不重建该基线。

因此验证"某次改动是否影响数值"时，口径应为**对拍当前 HEAD 的干净构建**（同输入、同 flavor），
而不是直接对旧冻结基线；`run_baseline.bat` 对旧基线报 DIVERGED 属已知，不作该改动的回归判据。
（upstream 固定参照 `upstream_129` 与 DeepSeek 的 cos gate 判据不受影响。）

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
