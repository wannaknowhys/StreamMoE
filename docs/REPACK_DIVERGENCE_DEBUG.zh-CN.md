[English](REPACK_DIVERGENCE_DEBUG.md) | [简体中文](REPACK_DIVERGENCE_DEBUG.zh-CN.md)

# Route B vs 原生：gate_up 数值差异（cos 0.508）bit 级排查与 repack 根因确认

> 状态：已排查、根因确认、bit 级验证通过、清理收束（2026-08-29）。
> 关联代码：`third_party/llama.cpp/ggml/src/ggml-cpu/repack.cpp`（repack 内核 / extra buffer type）、`src/backend/minigraph_exec.cpp`（route B delegate mini-graph）。

## 背景

- 目标：验证 route B（专家池，物理内存有界）与原生 llama.cpp 在 gemma-4-26B-A4B（Q4_K_M）上的数值一致性。
- 现象：30 层 MoE 叠加后，LM head cos 掉到 **0.508**，而 attention 部分 bit 完全一致。

## 排查链路

1. **首个 bit 分叉**：`ffn_moe_geglu-0`（cur，rb 第 #24 步）。
2. **cur（gate/up 切片）与原生差 1-13 ulp**。每层有 4-5 个 rms_norm，大值 up 分支上 1e-7 的差异被逐层放大 -> LM head cos 0.508。
3. **与 stride 无关**：改专家 stride（3717120 / 1486848 / 7929856）输出不变。
4. **与 delegate 传参无关**：delegate -> 内核参数与原生 52/52 完全一致（nb、data 头部、ids）。
5. **down 输出与原生一致**；L1 down 的差异来自前层累积。

## 关键教训：fnv 不能用于数值对比

- fnv 是字节级哈希：1 ulp 的差异会翻转几乎全部 bit -> 即使值几乎相同也显示"完全不同"。
- 应使用 **cos / maxDiff / ulp-bit 分布**（逐值对两个 float bit 模式做 XOR 后统计 popcount）。
- fnv 只在明确想判断"字节序列是否 bit 相同"时才可用。

## 根因：repack vs 普通内核路径差异

- 原生对 gemma 的 `ffn_gate_up_exps.weight`（Q4_K_M，3D，ne[0]=1408 / ne[1]%16==0）触发 **repack 权重重打包**（`ggml::cpu::repack` extra_buffer_type 接管 MUL_MAT_ID），走 repack 的 `forward_mul_mat_id`（`block_q4_Kx8` 布局）。
- `down_exps` 不匹配 repack 支持类型，走普通 `ggml_compute_forward_mul_mat_id`。
- route B 的 mini-graph（`w3d` = 池槽，普通 CPU buffer）永远不匹配 repack 的 `supports_op`，gate_up/down 都走普通内核。
- => 两边 gate_up 走不同内核路径 -> 1-16 ulp 差异，放大到 cos 0.508。
- deepseek（Q8_K_XL）不匹配任何 repack 类型 -> 两边同路径 -> cos 0.99999999。

## Bit 级验证

L0 gate_up 输出，token 0，专家列（1408 floats）：

- **rb 普通 vs 原生 repack**：bit 一致 20.8%，有差异 79.2%。bit 差分布：1bit=7110、2-4bit=9357、5-16bit=1378、17+bit=1、max=20bit。值显示完全相同（-0.05954 等）-> 1e-6 级浮点抖动，不是瞎算。
- **rb repack 路径 vs 原生**：**bit 完全一致 1408/1408（100.0%），0 差异**。

结论：输入 bit 一致 + 权重字节一致 + 同内核路径 => 输出 bit 一致。整个差异就是 repack-vs-普通 的路径差异。

## 验证实现方式

### 思路 1：模拟 dispatch
构造单专家 `MUL_MAT_ID`，src0 用池槽权重的 repack 副本，并给它 repack 的 `buffer->buft` + `extra`，让 `ggml_backend_graph_compute` 走 repack 的 extra buffer type。

### 思路 2（最终采用）：直接驱动内核
在 `repack.cpp` 暴露辅助函数：
1. 分配 repack buffer（`ggml_backend_buft_alloc_buffer`），设 `t->buffer/data/extra`，用 `traits->repack(...)` 重打包（必须与内核匹配，见下）；
2. 在栈上构造 `ggml_tensor`（src0/src1/ids/op，显式 ne/nb/data）、`ggml_compute_params`（wdata/wsize、threadpool、ith=0、nth=1），然后调 `((tensor_traits*)src0->extra)->compute_forward(&params, &op)`。

### 踩坑记录（均已解决）
- x86 AVX2 的 Q4_K repack 内核是 **`q4_K_8x8_q8_K`（8x8 布局）**；`q4_K_16x1_q8_K` 是 RISC-V 专属。用错 repack 布局会产生巨大值/NaN。
- 用内核自身的 `traits->repack()`，不要硬编码 repack 函数。
- `ggml_compute_params` 没有 `type` 字段。
- `params->threadpool` 必须非 null（`ggml_barrier` 会解引用）；用 `ggml_threadpool_params_default(1)` + `ggml_threadpool_new` 创建。
- `wsize = GGML_PAD(nbw3, 8) + n_as*(ne12+1)*8`；保险起见加余量。
- `no_alloc` 的 ggml context 里所有 tensor `data` 都是 null，需手动赋值（cur 切片、ids、dst）。
- `block_q4_Kx16` 是 2304 字节（d[16]+dmin[16]+scales[192]+qs[2048]）；16 行交错恰好写满原 Q4_K 张量的 `ggml_nbytes`。

## 调试方法论（工程经验）

- 在累积处读 `MUL_MAT_ID` 输出无效（缓冲被复用）；要在执行处（delegate / 内核入口 / repack）打 log。
- 直接用 cmd 重定向跑 `llama-cli`：`cmd /c "... > file 2>&1"`。PS 的 `*>` 流重定向会缓冲挂起；`Start-Process`+`Sleep`+`Stop-Process` 有竞态。
- 看 `$LASTEXITCODE`：`-1073741819`（0xC0000005）= 访问冲突，`-1073740791`（0xC0000409）= 断言/栈缓冲越界。
- stderr 重定向到文件不是行缓冲；进程可能崩溃时，诊断打印后要 `fflush(stderr)`。

## repack 内部要点（供以后参考）

- `supports_op`（MUL_MAT_ID）条件：`src[0]->buffer && buft == ggml_backend_cpu_repack_buffer_type() && ggml_n_dims(src[0]) == 3 && ggml_repack_get_optimal_repack_type(src[0]) && src[1]->type == GGML_TYPE_F32`。
- `get_tensor_traits` 返回 `src[0]->extra`；buffer 的 `init_tensor` 设 `tensor->extra = ggml_repack_get_optimal_repack_type(tensor)`。
- repack 数据 buffer 存重排后的权重；张量 `ne`/`nb` 保持原值（Q4_K 时 `nb00 == ggml_type_size(src0->type) == 144` 断言成立）。

## 清理

- 无待提交的正式改动：submodule HEAD `5ab785cf8`（export）与主仓库 `36e2b99`/`113c60f` 已含全部生产代码。
- `git checkout` 重置了 6 个文件的调试改动（minigraph_exec.cpp + submodule 的 ggml-cpu.c / repack.cpp / ggml.c / llama-context.cpp / llama-graph.cpp）。
- 删除 102 个临时文件 + `temp/export_ds`、`temp/sequences`、`temp/gemma_export`。保留 `temp/sm_env.example.bat`、`temp/patches_split/`、`temp/patch_backup_*/`、`temp/llama_verify/`。
