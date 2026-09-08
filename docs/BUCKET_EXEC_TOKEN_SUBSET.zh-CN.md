# 桶执行：token 子集 round（bucket_exec token-subset）

[English](BUCKET_EXEC_TOKEN_SUBSET.md) | [简体中文](BUCKET_EXEC_TOKEN_SUBSET.zh-CN.md)

> 状态：**设计，2026-09**。服务于 M2-5 紧凑链引擎收敛：删 A（2026-09-07）后唯一执行器 =
> `exec_layer_burst_chain_buckets`（minigraph_exec.cpp）。它现在的桶是**全 token k-slot 切**
> （env `STREAM_MOE_TMP_CHAIN_BUCKETS`，默认单满宽桶）。本文把桶源升级为
> **`build_mix_plan` rounds = 真 token 子集桶**（`[w_b, n_active]`，active token 是 n_t 的
> 子集，每 token 持有其 n_k slot 中的 w_b 个），并把 `scatter_plan` 模块
> （docs/SCATTER_PLAN.md）接为累加器写回。配套：docs/M2_DEVICE_EXECUTOR.md
> §7.5/§7.6/§7.7（per-device 累加器、每桶紧凑链、ggml_acc 做 fold 写回）。

## 1. 当前引擎形态（要改什么）

`exec_layer_burst_chain_buckets` 现在：

- 桶列表 = 覆盖**全部 token** 的 k-slot 半开区间（`bk = {[k0,k1)}`，`n_active == n_t`），
  来自 env cut 或默认单满宽 `[0, n_k)`。
- 每桶：对全部 `(t, k_lo+s)` 建 `ids_exp`/`ids_slot`（专家 id / pool 本地 slot id）→
  `[w_b, n_t]` 紧凑链。
- `append_mm_bucket`：`[w_b, n_t]` mm 壳；cur 是主图共享激活原地引用（`bucket_ext_leaf`），
  紧凑 dst 钉 arena `out_off[seq]`（M2 §7.2.1 串行复用）。
- weightless 孪生把 slot 轴窄到 w_b，token 轴保持 n_t。
- `append_expert_fold` 折 w_b → `[d_out, n_t]`；acc = `ggml_acc_inplace` offset 0 进常驻
  `acc_d`（n_active == n_t，同位加）。
- token 子集桶被 mock 挡（n_active != n_t 即 abort）。

## 2. 目标形态（本文）

桶源变为 **`mix_plan_t.rounds`**（`build_mix_plan`，backend/mix_split.h）：每个 round 是
真 token 子集桶：

- `r.ids`        = 专家 id，llama 布局 `[w_b, n_active]`（`r.ids[a*w_b+s]`），
  `w_b = r.width`，`n_active = r.n_active`。
- `r.scatter`    = 每 `(a,s)` 列对应的原 `(t, k)`。
- pool（device）= `r.pool`。

`round.scatter[a*w_b].t` = active 列 a 的原 token → 这就是喂给 `build_scatter_plan` 的 `t`。

引擎像现在循环桶一样循环 rounds，但每桶在 **`[w_b, n_active]`** 上建紧凑链，并折进累加器
在 token 位置 `t_a`（散列，非 offset-0）。

### 2.1 每 device / 每桶的 cur 拷贝（GPU 形态）

M2 §7.5：每 device 拥有一份 cur 拷贝；每桶 mini-graph 在**最前面放一个 cur 拷贝节点**，
把主图 full cur `[d,1,n_t]` gather 成桶的紧凑 cur `[d,1,n_active]`（tight 序，列 =
`t[order[i]]`）。CPU 单图引擎里同一节点 append 到每桶链头（该桶首个路由 mm 之前）。
下游所有链张量免费继承 tight 序（SCATTER_PLAN §2）。

ggml 原生 gather：`ggml_get_rows(cur, col_ids)`，col_ids 是 tight 序 token id 的 i32
leaf（`[1, n_active]`）。这是已定的 CPU 阶段实现（用户决策 2026-09-07）——gather 是图内
节点，非 host memcpy。要求 CPU 内核能对 cur 的 token 轴（`[d,1,n_t]` 的 ne2）gather，
不只前导维——使用前先验证。cur 拷贝节点写 `[d,1,n_active]` 进 arena（自己的 out_off 区
或 fold_buf）。

### 2.2 round 上的链几何

- mm ids = `r.ids` 的 pool 本地 slot 翻译（`slot = pin(e); slot-slot_begin`）。
- mm cur = 桶的紧凑 cur 拷贝（tight 序）供首个路由 mm；down mm 读自己的紧凑 GLU 孪生如旧。
- weightless 孪生窄到 w_b 如旧，但 token 轴 = n_active。
- `append_expert_fold` → `per_token[d_out, n_active]`（TIGHT 序）。

### 2.3 累加器 scatter-add（scatter_plan）

每个 round：

1. `t[a] = r.scatter[a*r.width].t`（a 序，每个 active 列的原 token）。
2. `plan = build_scatter_plan(t, n_active, n_t)` → `order[]`、`segs[]`。
3. cur 拷贝按 `order` gather → `per_token` 已是 tight 序：
   `per_token[:, i]` 对应原 token `t[order[i]]`。
4. acc：每个 seg 一次 `ggml_acc_inplace`：
   `acc_d[:, dst + i*delta] += per_token[:, src + i]`，
   `nb1 = delta*d_out*4`、`offset = dst*d_out*4`（src 因 tight 连续）。

plan 在 host 算（纯函数、便宜）；`order` 喂 cur 拷贝，`segs` 喂 acc 循环。

### 2.4 acc_d 就是 per-device 累加器

`c.acc_d`（chain_ctx，`[d_out,n_t]`）即 device 累加器（M2 §7.6.1）：每桶就地增量加，
层首（round 循环前）清零（已有）。出口 = `chain_exit`（acc_d → add_in → moe_out）不变。

## 3. 分桶函数（一桶变两桶，用于验证）

单 pool 的 `build_mix_plan` 只产一个 round（全 token 全 k），测不到 token 子集路径。
需要一个确定性**分桶函数**把那个 round 切成两个**散落** token 子集桶，让子集路径
（cur 拷贝 + scatter_plan 重排）在真 multi-pool 之前就能在 CPU 引擎上被验证。

`src/backend/bucket_split.h/.cpp`（纯、无 llama 依赖、可离线单测）：

```cpp
// 把一个满 round 切成两个散落 token 子集 round。
// round0 = tokens {0, 2, 4, ...}（偶），round1 = {1, 3, 5, ...}（奇）；
// 各保持满 k（width = n_k）。交错 token id 让 scatter_plan 产生 delta>1 的等差段
// 与真重排——我们要验证的正是子集路径的 order 机制。
std::vector<mix_round_t> split_round_tokens_parity(const int32_t* ids, uint32_t n_k,
                                                   uint32_t n_t,
                                                   const int32_t* expert_pool,
                                                   uint32_t n_expert, uint32_t pool);
```

执行器接受 `build_mix_plan` 真 rounds 或分桶函数输出作为 round 列表，两者都是
`mix_round_t`。验证：两个奇偶桶累进 acc_d 必须等于单满 round（宽松 ULP gate——两边
本就与 llama 的 k 序 fold 不同）。

## 4. 执行器改动草图（exec_layer_burst_chain_buckets）

- 默认从 `build_mix_plan` 建 round 列表（单 pool = 单满 round 退化；multi-pool = 真
  per-pool rounds）。验证分桶（env `STREAM_MOE_TMP_BUCKET_ROUNDS=parity`）用分桶函数的
  两个散落桶替换该列表。
- 循环 rounds；每 round 设 `b.w_b = r.width`、`b.n_active = r.n_active`、
  `b.ids_exp/ids_slot` 来自 `r.ids` + pin、`b.t` 来自 `r.scatter`。
- 链头 prepend cur 拷贝节点（`ggml_get_rows`，tight-gather）。
- `append_expert_fold` → tight `[d_out, n_active]`。
- acc 走 `build_scatter_plan` order/segs（每个 seg 一次 `ggml_acc_inplace`，在 `acc_d`
  上就地链式）。
- 保留 arena out_off 钉址；紧凑 `[d,w_b,n_active]` 落在满宽 out_off 区内，因
  `w_b <= n_k && n_active <= n_t`。

## 5. 未决问题 / 风险

1. `ggml_get_rows` 对 cur token 轴（`[d,1,n_t]` 的 ne2）gather 需先验证；若 CPU 内核只
   支持前导维 gather，cur 拷贝退化为 reshape 成 `[d*n_t]` 式 2D + 前导维 get_rows +
   reshape 回来（仍是图内节点，非 host staging）。
2. 现有单 pool 满 round 数值基线（gemma_129_l0、deepseek_hi）必须在退化单 round 路径
   （默认 = build_mix_plan 单 pool = 1 round、无分桶 env）下保持逐字节一致。
3. `scatter_plan` 目前拒绝 round 内重复 token id——矩形 peel 不会产生；分桶函数须保证
   每 token 只在一个桶。

## 6. TODO

1. `bucket_split.h/.cpp` 分桶函数（parity → 2 散落 round）+ 测试。
2. 验证 ggml_get_rows 能在 CPU 上 gather cur 的 token 轴（或 2D-reshape 兜底可行）。
3. 执行器 round 循环：接受 mix_round_t 列表；cur 拷贝节点 prepend；每 round tight ids；
   fold → tight per_token；acc 走 scatter_plan。
4. 验证：默认单 round 对基线不变；`parity` 分桶 → 对单 round 宽松 gate。
