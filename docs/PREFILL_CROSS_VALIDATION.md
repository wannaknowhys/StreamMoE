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

### 限制 1：`t_h_nextn`（hidden state）当前为 0 行

**代码原因**（`src/models/deepseek4.cpp:1327-1331`）：

```cpp
if (cparams.embeddings_nextn) {
    ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;
}
```

`res->t_h_nextn` **只在 `cparams.embeddings_nextn` 为 true 时才会被设置**（该开关供 next-token embeddings 功能使用，llama_engine 默认关闭）。未开启时 `get_h_nextn()` 返回 `nullptr` → 累积代码直接跳过 → 导出 hidden=0 行。

**要启用**：调用 `llama_set_embeddings_nextn(ctx, true)`（引擎侧需暴露参数）后，`t_h_nextn` 才会被计算并导出。

### 限制 2：LM head 输入（result_norm）只含 output tokens

**代码原因**（`src/models/deepseek4.cpp:1324-1342`）：

```cpp
ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
...
if (inp_out_ids) {
    inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);   // 只保留 output tokens
}
cur = build_hc_head(inpL, ...);        // 只对 n_outputs 计算
cur = build_norm(cur, model.output_norm, ...);   // result_norm = [n_embd, n_outputs]
res->t_embd = cur;
```

当 `inp_out_ids` 非空（图里存在"只需 output token 参与后续计算"的需求）时，`inpL` 被 `get_rows` 提取成**只有 output tokens** 的 `[n_embd, hc, n_outputs]`，随后 hc_head + output_norm（= LM head 输入 `result_norm`）**只对 output tokens 计算** → `t_embd = [n_embd, n_outputs]`。

而 llama_engine 的 prefill 只给最后 1 个 token 设 `logits=true`（`src/engine/llama_engine.cpp:240` `need_logits_last && (k == end-1)`），所以 prefill 的 `n_outputs=1` → 导出的 embd 只有 1 行（末 token）；decode 每 token 都是 output → 每次 1 行。

**要 prefill 全 token 的 LM head 输入**：让 batch 全部 token 的 `logits=true`（`n_outputs = n_tokens`），`inp_out_ids` 为空/全取，`result_norm` 即为 `[n_embd, n_tokens]`——这需要专用 prefill 测试（如 `llama_batch` 全 logits 的 prefill run），非引擎默认行为。

### 其他

- 导出 patch 是一次性补丁（`patches/`）；还原 `git -C third_party/llama.cpp checkout .` 后需重建。
