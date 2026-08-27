# 任务二：Prefill 交叉验证基准 (PREFILL_CROSS_VALIDATION.md)

> 目的：以相同输入分别在**标准 llama.cpp**（StreamMoE 不带 `--expert-backend`，即 stock 图）与 **StreamMoE**（`--expert-backend`）做 Prefill，逐 token 导出 **LM Head 输入向量** + **KV Cache**，独立验证程序对齐对比，定位首个明显差异。
> 若两套一致 → 证明 StreamMoE 在 Prefill 的加载/权重读取/专家调度/计算/KV 写入与 llama.cpp 一致。

## 形态

- **patch A** `patches/prefill-export-llama.patch`：改造 vendored llama.cpp——
  - `llama_context` 累积每步的 LM head 输入（`result_norm` / `t_embd`）+ hidden state（`t_h_nextn`），析构时一次性导出到文件（含最终 KV cache 全部子缓存逐层张量）；
  - `llama_new_context_with_model` 在设置 `LLM_EXPORT_DIR` 时**强制 `cparams.embeddings_nextn=true`**，使 `t_h_nextn` 被计算并导出（见"已知限制"）。
- **patch B** `patches/prefill-export-streammoe.patch`：改造 StreamMoE `src/engine/llama_engine.cpp` 的 `decode_tokens`——在设置 `LLM_EXPORT_DIR` 时给 batch **全部 token 标记 `logits=true`**，使 prefill 的 LM head 输入（result_norm）覆盖全部 token。
- **独立验证程序** `tools/verify_prefill.js`：读两个导出文件，按 token 位置对齐，**cosine 与 maxAbs 双门槛**判定（方向 + 幅度都一致才算一致），定位首个差异 + KV 逐层字节 diff。

## 判定条件（verify_prefill.js）

- 余弦相似度只衡量**方向**，对整体缩放不敏感——所以**同时要求 `|1-cos| <= 1e-6` 且 `maxAbs <= 1e-4`**，两者都过才算一致，任一超限即报 DIFF。
- **两个全零向量**时 cos 公式会算出 0（而非 1），导致误报——先判 `normA < 1e-30 && normB < 1e-30` 视为一致。
- 报告每 token 的 cos / MAD / MSE / maxAbs@pos 及命中的门槛（cos/abs）。

## 用法

```bat
rem 1) 应用两个 patch 并重建（build.bat llamalibs main）
git -C third_party/llama.cpp apply patches\prefill-export-llama.patch
git apply patches\prefill-export-streammoe.patch
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

- **LM head 输入（embd）**：6 行 = prefill 全部 5 token + decode 首 token（patch B 使 prefill 全 token 为 output）；**hidden（t_h_nextn）**：8 行（patch A 强制 embeddings_nextn 后导出）。std vs moe 均 **IDENTICAL（cos+maxAbs 双门槛 0 差异）**。
- **KV**：全部子缓存（raw_swa 43 层 / csa 21 / hca 20 / lid 21）逐层字节 0 差异。
- 与 `tools/compare_trace.js`（逐层 ffn/attn/gate/up/down + #ids/#cur 逐位一致）互相印证。

## 已知限制

### 限制 1：`t_h_nextn`（hidden state）——已由 patch A 解除

原因为 `src/models/deepseek4.cpp:1327-1331` 仅在 `cparams.embeddings_nextn` 开启时设置 `res->t_h_nextn`。patch A 在 `llama_new_context_with_model` 检测到 `LLM_EXPORT_DIR` 时强制开启，故导出 run 能拿到全 token hidden state。注意这会多算 `h_nextn` 节点（仅在导出 run 生效，正确性不变）。

### 限制 2：LM head 输入只含 output tokens——已由 patch B 解除

原因为 `src/models/deepseek4.cpp:1333-1342` 在 `inp_out_ids` 非空时只保留 output tokens 计算 hc_head/output_norm，而引擎 prefill 只给末 token 设 `logits=true`。patch B 使导出 run 的 prefill 全部 token `logits=true`（`n_outputs = n_tokens`），`result_norm` 即覆盖全部 token。

### 其他

- 导出 patch 是一次性补丁（`patches/`）；还原：`git -C third_party/llama.cpp apply -R patches\prefill-export-llama.patch && git apply -R patches\prefill-export-streammoe.patch` 后重建。
