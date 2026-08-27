# 任务二：Prefill 交叉验证基准 (PREFILL_CROSS_VALIDATION.md)

> 目的：以相同输入分别在**标准 llama.cpp**（StreamMoE 不带 `--expert-backend`，即 stock 图）与 **StreamMoE**（`--expert-backend`）做 Prefill，逐 token 导出 **LM Head 输入向量** + **KV Cache**，独立验证程序对齐对比，定位首个明显差异。
> 若两套一致 → 证明 StreamMoE 在 Prefill 的加载/权重读取/专家调度/计算/KV 写入与 llama.cpp 一致。

## 形态

- **patch A** `patches/prefill-export-llama.patch`：改造 vendored llama.cpp，在 `llama_context` 累积每步的 LM head 输入（`result_norm` / `t_embd`）+ hidden state（`t_h_nextn`），析构时一次性导出到文件（含最终 KV cache 全部子缓存逐层张量）。受环境变量 `LLM_EXPORT_DIR` 控制，不设则零开销。
- **patch B（StreamMoE 侧）**：无需独立代码——StreamMoE 复用同一 `llama_context`，设 `LLM_EXPORT_DIR` 即同样导出。运行差异仅 `--expert-backend` 与否。
- **独立验证程序** `tools/verify_prefill.js`：读两个导出文件，按 token 位置对齐，算 cosine/MAD/MSE，定位首个差异 + KV 逐层字节 diff。

## 用法

```bat
rem 1) 应用 patch A 并重建（build.bat llamalibs main）
git -C third_party/llama.cpp apply patches\prefill-export-llama.patch
build.bat llamalibs main && build.bat build main

rem 2) 标准 run（无 --expert-backend）
set "LLM_EXPORT_DIR=temp\export_std"
build\main\bin\stream_moe.exe -m <model> --moe-ram-pool 71680 --temp 0 -p "..." -n 2

rem 3) StreamMoE run（--expert-backend）
set "LLM_EXPORT_DIR=temp\export_moe"
build\main\bin\stream_moe.exe -m <model> --moe-ram-pool 71680 --expert-backend --temp 0 -p "..." -n 2

rem 4) 对比
node tools\verify_prefill.js temp\export_std\prefill_export.bin temp\export_moe\prefill_export.bin
```

> 注意：cmd 的 `set VAR=path && cmd` 会把 `&&` 前的空格算进值导致 fopen 失败，必须 `set "VAR=path" && cmd`（引号包裹整个 set）。

## 导出文件格式（PREFEXP1，析构时一次性写）

```
u32 n_embd_rows, u32 n_embd_dim
per row: u32 token_pos, float embd[n_embd_dim]       ; LM head 输入（result_norm，output tokens）
u32 n_hid_rows, u32 n_hid_dim
per row: u32 token_pos, float hidden[n_hid_dim]      ; hidden state（t_h_nextn，全部 tokens）
u32 n_cache
per cache: u32 nl, name[nl], u32 n_layer
  per layer: u32 il, u32 k_type, u64 k_nbytes, k bytes, u32 v_type, u64 v_nbytes, v bytes
```

deepseek4 的 KV 由 `llama_kv_cache_dsv4` 承载（raw_base/raw_swa/csa/hca/lid 5 个子缓存）；`raw_base` 层数为 0（DSV4 raw 只使用 SWA 半区），V 张量为空（MLA 无显式 V 存储）。

## 2026-08-26 实测（prompt "Say hi.", -n 2）

- embd（LM head 输入，2 行：prefill 末 token + decode 首 token）与 KV 全部子缓存逐层：**IDENTICAL（0 差异）**。
- 与 `tools/compare_trace.js`（逐层 ffn/attn/gate/up/down + #ids/#cur 逐位一致）互相印证。

## 已知限制

- `t_h_nextn`（hidden state，全 token）当前为 0 行：deepseek4 的 `t_h_nextn` 仅部分路径设置，导出尚未拿到；LM head 输入（result_norm）仅对 **output tokens** 输出（prefill 只有末 token 的 logits=true）。若需 prefill 全 token 的 LM head 输入，需让 batch 全部 token 为 output（专用 prefill 测试，非引擎默认）。
- 导出 patch 是一次性补丁（`patches/`）；还原 `git -C third_party/llama.cpp checkout .` 后需重建。
