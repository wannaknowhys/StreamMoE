# 桶执行：token 子集 round（bucket_exec token-subset）

[English](BUCKET_EXEC_TOKEN_SUBSET.md) | [简体中文](BUCKET_EXEC_TOKEN_SUBSET.zh-CN.md)

> 状态：**设计，2026-09**。服务于 M2-5 紧凑链引擎收敛：删 A（2026-09-07）后唯一执行器 =
> `exec_layer_burst_chain_buckets`（minigraph_exec.cpp）。它现在的桶是**全 token k-slot 切**
> （env `STREAM_MOE_TMP_CHAIN_BUCKETS`，默认单满宽桶）。本文把桶源升级为
> **`build_mix_plan` rounds = 真 token 子集桶**（`[w_b, n_active]`，active token 是 n_t 的
> 子集，每 token 持有其 n_k slot 中的 w_b 个），并把 `scatter_plan` 模块
> （docs/SCATTER_PLAN.md）接为累加器写回。配套：docs/M2_DEVICE_EXECUTOR.md
> §7.5/§7.6/§7.7（per-device 累加器、每桶紧凑链、ggml_acc 做 fold 写回）。
>
> **2026-09-08 修订（用户）**：cur gather = 2D reshape + get_rows（§2.1）；任意 (t,k)
> 权重 = 通用 flat index-gather 节点（§2.1a）；测试分桶 = minigraph_exec.cpp 内宏包裹的
> 一次性助手，不是模块（§3）；删 `tmp_split_blocks`（§4）。

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

ggml 原生 gather（2026-09-08 定）：`ggml_get_rows` 只 gather 行轴（ne1）、永不
gather ne2，所以 `[d,1,n_t]` 不能直接 gather。改法：把（连续、ne1 == 1 的）cur reshape
成二维 `[d, n_t]`，用 tight token id 的 i32 leaf `[n_active]` 做 get_rows →
`[d, n_active]`，再 reshape 回 `[d,1,n_active]`。reshape 免费、标准内核、图内节点
（非 host memcpy）。拷贝节点写 `[d,1,n_active]` 进 arena（自己的 out_off 区或 fold_buf）。

### 2.1a 通用 index-gather 节点（任意 (t,k)）

一个 `mix_plan` round 给每个 token 选的是它 k 个 slot 的**任意子集**
（`ks_of[t][vprev..vj)`，mix_split.cpp）——不是连续 k 区间。所以 per-slot 路由权重
`weights_norm[0,k,t]`（`[1, n_k, n_t]`）无法用 affine leaf 切片
（`bucket_ext_leaf` 现在的 `data += k_lo*nb1`）表达。一个通用 **flat index-gather**
节点解决（用户决策 2026-09-08）：

- 把连续的 `[1, n_k, n_t]` reshape 成二维 `[1, n_k*n_t]`（免费）；
- 构造 i32 leaf `idx[width*n_active]`，内容为 tight 序的**扁平元素偏移**：
  `idx[i*width + s] = k + t*n_k`，其中 `(t,k) = scatter[order[i]*width + s]`；
- `ggml_get_rows(flat, idx)` → `[1, width*n_active]`，reshape `[1, width, n_active]`。

cur gather（§2.1）是同一助手的 row_size = d 特例。这个节点取代 weights_norm 的连续-k
`bucket_ext_leaf` 切片；强制测试分桶（§3）刻意选非连续 k，让它在真 multi-pool 之前
就在 CPU 上被打到。

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

## 3. 测试专用强制分桶函数（验证脚手架）

单 pool 的 `build_mix_plan` 只产一个 round（全 token 全 k），测不到 token 子集 / 任意
(t,k) 路径。一个测试专用**强制分桶函数**把那个 round 切成若干**散落** token 子集桶，
让子集路径（cur gather + index-gather 权重 + scatter_plan 重排）在真 multi-pool 之前
就能在 CPU 引擎上被验证。

放置与门控（2026-09-08 定）：`minigraph_exec.cpp` 里的 `static` 助手；**函数定义和调用点
都用 `#ifdef STREAM_MOE_TEMP` 包裹**（唯一定义该宏的 tag 是 `StreamMoE_dump_dbg`，
build.bat）。它是一次性验证码——子集路径验证完即删。不新建 `bucket_split.h/.cpp`，不写
离线 UT。

分桶形状：两轴都切，才能打到任意 (t,k)：
- k 轴按奇偶 → 非连续 k 的 round（n_k 为偶数时是合法矩形：每 token 贡献 n_k/2）；
- token 轴按奇偶 → token 子集 + scatter delta > 1；
- 叠加 → 4 个 round，每个 `width = n_k/2, n_active = n_t/2`，`(t,k)` 任意。

`r.scatter` 记原始 (t,k)；`r.pool` = 源 round 的 pool。

## 4. 执行器改动草图（exec_layer_burst_chain_buckets）

- 默认 round 列表 = `build_mix_plan(ids, n_k, n_t, expert_pool, n_expert,
  n_pools).rounds`，`expert_pool[e] = handle.pool` 由已 pin 的 handle 填
  （`pin_layer` 本就返回 per-expert pool）。单 RAM 池 = 一个满 round（退化，与今天默认
  单桶逐字节一致）。
- 宏门控的强制分桶替换该列表。
- 删 `tmp_split_blocks` / `tmp_blk_t`（旧测试 cut 族，现在还是无条件编译）——两套分桶源
  不能并存。
- 循环 rounds；每 round 设 `b.w_b = r.width`、`b.n_active = r.n_active`、
  `b.ids_exp/ids_slot` 来自 `r.ids` + pin、`b.t` 来自 `r.scatter`。
- 链头 prepend cur 拷贝节点（2D reshape + get_rows，§2.1）。
- per-slot 路由权重走通用 index-gather 节点（§2.1a）。
- `append_expert_fold` → tight `[d_out, n_active]`。
- acc 走 `build_scatter_plan` order/segs（每个 seg 一次 `ggml_acc_inplace`，在 `acc_d`
  上就地链式）。
- 保留 arena out_off 钉址；紧凑 `[d,w_b,n_active]` 落在满宽 out_off 区内，因
  `w_b <= n_k && n_active <= n_t`。

## 5. 未决问题 / 风险

1. 已解决（2026-09-08）：`ggml_get_rows` 只 gather ne1；cur 用免费的 2D reshape 成
   `[d, n_t]` + get_rows + reshape 回来（§2.1）。不改内核、不做 host staging。
2. 任意 (t,k) 权重走通用 flat index-gather 节点（§2.1a），不是 affine 切片；强制测试
   分桶会打到它。
3. 单 pool 满 round 数值基线（gemma_129_l0、deepseek_hi）必须在退化单 round 默认路径
   （build_mix_plan 单 pool = 1 round、无分桶宏）下保持逐字节一致。
4. `scatter_plan` 拒绝 round 内重复 token id——矩形 peel 不会产生；强制分桶函数须保证
   每 token 只在一个桶。

## 6. TODO

1. 通用 index-gather 助手（§2.1a）：任意 (t,k) 权重用 flat reshape + get_rows；cur
   gather（§2.1）是 row_size = d 的特例。
2. 执行器 round 循环：expert_pool 从 pins 取、默认 build_mix_plan rounds、每 round
   tight ids、fold → tight per_token、acc 走 scatter_plan；删 tmp_split_blocks。
3. 宏包裹的测试专用强制分桶（k 奇偶 × t 奇偶 = 4 round）+ 调用。
4. 验证：默认单 round 对基线不变；强制分桶 → 宽松 gate（maxAbs <= 1e-5 / cos ~= 1.0）。
