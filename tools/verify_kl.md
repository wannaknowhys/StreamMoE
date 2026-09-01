# verify_kl — 逐 token KL 散度验证工具

> 用途：对比两个 prefill 导出（PREFEXP1/PREFEXP2 的 embd 段 = LM head 输入 `result_norm`）的**逐 token 分布一致性**，用模型自身的 `output.weight`（LM head）经 ggml 计算 logits，再算 softmax 后的 KL 散度。动态检测 embd 类型（f16/f32）与 `output.weight` 的类型（任意量化），用 `ggml_mul_mat` 做类型匹配的矩阵乘（量化自动 dispatch）。

## 基准（谁是 reference）

- **file1 = 基准（reference）**，file2 = 被验证（candidate）。
- 主指标：**KL(file1 || file2)**（file2 相对 file1 的信息损失）。
- 每行顺带输出**反向 KL(file2 || file1)** 作为参考（KL 非对称，两边都看）。
- 用法：`verify_kl <model.gguf> <file1.bin> <file2.bin>`

## token 语义与缺失

- token 是**连续序列**：`pos = 0 .. max(file1 最后 token_pos, file2 最后 token_pos)`。
- 每 pos：
  - 两边都有 → 算 KL，输出一行数字
  - 只有 file1 → `file 2 missing`（file2 缺该 token）
  - 只有 file2 → `file 1 missing`
  - 两边都没有 → `both file missing`（范围内空洞）

## 编译

链接 vendored ggml 静态库（先 `build.bat llamalibs main` 有 `build\main\llama-build`）：

```bat
set L=build\main\llama-build
F:\Dev\LLVM\bin\clang++.exe -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -fopenmp tools\verify_kl.cpp ^
    -I third_party\llama.cpp\ggml\include ^
    %L%\ggml\src\ggml.lib %L%\ggml\src\ggml-base.lib %L%\ggml\src\ggml-cpu.lib ^
    F:\Dev\LLVM\lib\libomp.lib -ladvapi32 ^
    -o temp\verify_kl.exe

rem 运行需要 libomp.dll（在 PATH 或 exe 同目录）：
copy /Y F:\Dev\LLVM\bin\libomp.dll temp\ >nul
```

## 用法

```bat
temp\verify_kl.exe <model.gguf> <ref.bin> <cand.bin> [--thresh <T>]
```

- `--thresh T`：任一 token 的 `KL(ref||cand) > T` 则 RESULT=FAIL（退出码 1）；不传则只报告。

示例（对比上游基准 vs route-B，gemma4）：

```bat
temp\verify_kl.exe N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf ^
    O:\1\exports\gemma\upstream\cn\prefill_export_main.bin ^
    O:\1\exports\gemma\moe\cn\prefill_export_main.bin
```

## 输出说明

```
[verify_kl] LM head token_embd.weight (tied) type=q8_0 ne=[2816 x 262144]
[verify_kl] ref  (...prefill_export_main.bin): rows=10241 dim=2816 dtype=f32
[verify_kl] cand (...prefill_export_main.bin): rows=10241 dim=2816 dtype=f32
token#0: KL(ref||cand)=3.2e-05  KL(cand||ref)=3.1e-05
...
token#5: file 2 missing
token#7: both file missing
[verify_kl] tokens=132 shared=129 ref_missing=1 cand_missing=2 both_missing=0
[verify_kl] mean_KL(ref||cand)=2.8e-05  max_KL(ref||cand)=9.1e-05 @token#3
[verify_kl] RESULT: PASS (thresh=1.0e-03, max_KL=9.1e-05)
```

- **LM head**：优先 `output.weight`；无则用 `token_embd.weight`（tied embeddings，如 gemma——布局已是 `[n_embd x n_vocab]`）。`type` 是权重的量化类型，`ne=[n_embd x n_vocab]`（gemma vocab=262144）。
- `dtype=f32/f16`：该导出 embd 段的实际类型（动态检测）。
- **pos 空洞**：导出端只记录实际算过的 token——例如 embd 段 pos 可能是 `0, 172, 176, 177...`（空洞 1-171 两边都没有），按连续序列语义这些报 `both file missing`。
- 数值小 = 分布接近（route-B 与上游应到 ulp 级/1e-5 量级）；`INF` 出现在 ref 有概率而 cand 概率为 0 的维度。
- `--list-tensors` 可单列模型的全部张量（诊断 LM head 名称/形状）。
