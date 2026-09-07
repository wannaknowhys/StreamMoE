# 桶 token 重排 + acc scatter 计划（scatter_plan）

[English](SCATTER_PLAN.md) | [简体中文](SCATTER_PLAN.zh-CN.md)

> 状态：**设计，2026-09**。服务于 CPU 双桶紧凑链原型（docs/WORK_IN_PROGRESS.md M2-5）
> 以及未来的 per-device mini-graph（docs/M2_DEVICE_EXECUTOR.md §7.8）。此处"桶"=
> `mix_plan_t` 里的一个 `mix_round_t`（docs/Backend.md J6 / mix_split）：路由 (token, slot)
> 列的一个满矩形 `[w_b, n_active]`。

## 1. 问题

路由 MoE 层每个 token 产生 `n_k` 个专家列。一个桶只持有其 pool 拥有的列——token 的一个
子集，且每 token 只持有其 slot 的一个子集。紧凑链（docs/M2_DEVICE_EXECUTOR.md §7.5/7.8）
只对桶的列计算本层 MoE ffn 并按 token 折叠，所以 `append_expert_fold` 之后每个**活跃**
token 得到一列部分和：`per_token[d_out, a]` 属于 round 的 `scatter[a*w_b]` 记录的原 token
`t_a`。这些 `t_a` 是散落的（a 是 peel 序而非 token 序），且同一 token 可能出现在多个 round。

折叠结果必须加进本层的输出累加器 `acc_d[d_out, n_t]`（CPU 原型）/ 主图 `moe_out` 的列
（GPU）：对每个活跃 `a`：`acc_d[:, t_a] += per_token[:, a]`。

一次 `ggml_acc` 调用能表达：src1 的连续列 `0..len-1` 加进 dst 等差列 `base,
base+delta, ...`（已在 CPU 内核验证，ggml-cpu/ops.cpp `ggml_compute_forward_acc_f32`：
`dst[offset + i1*nb1 + ...] += src1[...]`，其中 `nb1 = delta * d_out * esize`、
`offset = base * d_out * esize`；中间的 dst 列不受影响）。所以每个 dst 列的**等差 run**
只需一次 acc——前提是供给该 run 的 src 列在 `per_token` 里**连续**。

peel 序下不存在这种连续性。`scatter_plan` 决定 **tight 序**：对桶的活跃 token 做一次重排，
使重排后的 token→acc-dst 映射能分解成少数几个连续 src run，每个 run 是 dst 列的纯等差
序列。这样每个 run 恰好需要一次 `ggml_acc_inplace`。

## 2. 为什么重排必须在链之前消费（cur 拷贝）

紧凑链的所有中间张量都带桶的 token 轴，且顺序由第一个路由 mm 的 `cur`（gate/up mm 的
输入）给定。如果从一开始就让该轴是 tight 序，那么下游每个张量——gate_up dst、GLU 输出、
down mm dst、down_scaled、weighted，直到折叠——都免费继承 tight 序，`per_token` 出来时
已经按 run 分组。末尾无需二次 gather。

第一个路由 mm 读 `cur`（norm 输出，每个原 token 一列）。对全 token 桶可以直接引用
`cur`；对 token 子集 / tight 重排，必须先把 `cur` **按 tight token 序拷贝进紧凑 staging
缓冲**，mm 再读它（这就是"强制 cur 拷贝"）。重排正是在这次拷贝中物理落地。

### 受桶切割影响的 per-token 输入（盘点）

| 数据 | 角色 | 形状 | 桶影响 |
|---|---|---|---|
| `cur`（norm 输出） | mm src1 | `[d,1,n_t]`（共享）/ `[d,n_k,n_t]`（per-slot） | token 轴必须按 tight 序拷贝 → `[d,..,n_active]` |
| 路由 `ids` | mm src2、GET_ROWS src1 | `[n_k,n_t]` | token 子集 + slot 子集已暂存（`ids_exp`/`ids_slot`）；token 块需按 tight 重排 |
| per-slot 路由权重（`ffn_moe_weights_norm`） | weighted mul src1 | `[1,n_k,n_t]` | slot 切片 `[k_lo..k_hi)` **且** token 轴按 tight 序 |
| 专家权重、per-expert scale（REPEAT 表） | mm src0 / GET_ROWS src0 | per-expert | 与 token 无关——不动 |
| 链内张量（GLU/down/weighted） | - | `[d,w_b,n_t]` | 继承首次 cur 拷贝定下的顺序 |

## 3. 模块接口（纯计算，无 ggml/llama 依赖）

新文件 `src/backend/scatter_plan.h/.cpp`，风格同 `mix_split`（纯、确定、可离线单测）。
模块自含 run 定义与贪心策略——调用方只需给 token 序列。

```cpp
#pragma once
// StreamMoE bucket scatter planning (docs/SCATTER_PLAN.md).
// Pure / deterministic / model-agnostic - no llama.cpp or ggml dependency.
#include <cstdint>
#include <vector>

namespace stream_moe {

// 一个 acc-dst 列的等差 run，由一段连续 src 供给。
//   acc_d[.., dst + i*delta] += per_token[.., src + i]   对 i in [0, len)
// dst/delta 是 acc_d[d_out, n_t] 里的 dst 列下标；执行器转成 ggml_acc 参数：
// nb1 = delta*d_out*esize，offset = dst*d_out*esize。
struct scatter_seg_t {
    uint32_t src   = 0;    // tight 序 src 起点列（== 前序 run 的累计列数）
    uint32_t dst   = 0;    // src[0] 对应的 acc-dst 列
    uint32_t len   = 0;    // 本 run 列数（>= 1）
    uint32_t delta = 0;    // 相邻 src 列对应的 dst 列步长
};

struct scatter_plan_t {
    // Tight 序：重排后第 i 个 per-token 列（0..n_active-1）原是输入 `t` 序列的
    // 活跃项 `order[i]`。cur 拷贝层按此序收集 cur 列。
    std::vector<uint32_t> order;
    // 对 tight 序切出的 acc run（按 tight 序、互斥、覆盖全部 n_active 列）。
    // segs.back() 结束时 src == n_active。
    std::vector<scatter_seg_t> segs;
    uint32_t n_active = 0;
    uint32_t n_t      = 0;
};

// 为一个桶建计划。`t[0..n_active)` = 每个活跃列的原 token id（a 序，
// 即 t[a] = scatter[a*w_b].t）。n_t = 本层 token 数（边界：dst+delta*(len-1) < n_t）。
scatter_plan_t build_scatter_plan(const uint32_t* t, uint32_t n_active, uint32_t n_t);

} // namespace stream_moe
```

`order` 是主输出（cur 拷贝消费它）。`segs` 可由 `order` 推出，但预计算出来让 acc 循环
执行器不用再推；两者都做一致性单测（重放 order、切 segs、验证互斥覆盖 [0,n_active)）。

## 4. 贪心最大 run 抽取

目标：用最少的等差 run 覆盖 token 集（每 run 一次 acc），每个 run = 严格递增的 dst
列等差序列，其 src 列经一次重排可连续。

对等差序列做集合覆盖一般是 NP-hard，故用贪心：**反复取剩余的最长 run**，然后去掉其
token。run 的 dst 列必须互异且递增、每个 token 只用一次，所以同一 token 不会出现在两个
run 里。

算法（n_active 个 token，值域 [0, n_t)）：

1. 按值分组（防御性处理重复 token id：一个 token 在一桶里出现两次 = 两个独立部分列，
   各自一个 dst 相同的 run 元素——只有 pool 在同一个 round 里持有该 token 两个不相交
   slot 切片才会发生，而矩形 peel 永不产生这种情形；否则当输入错误拒绝）。
2. 直到值集为空：
   a. 对每个候选 delta（从 1 向上，只要 token 值还能再出现，即集中任意两值之差）统计
      等 delta 的最长链：`len(delta) = max over v of #{k >= 0 : v + k*delta in set}`。
      delta 无上界，但只有 delta <= (max-min) 有意义；只扫真正整除某对值的 delta。
   b. 选 `len(delta)` 最大的 delta（平手：delta 最小，再起点最小——确定性）。
   c. 为该 run 发一个 seg，从集合去掉其值。
3. 发出顺序（run 被抽取的先后）即 tight 序：把每个 run 的值升序拼接。

复杂度对 CPU/decode 规模可接受（n_k <= 16，n_active <= n_t）；每次抽取做
O(delta_range * n_active) 扫描、n_active 最多几百，没问题。模块按设计是启发式的——
"run 最少"是质量目标而非保证。

## 5. 执行器消费点（设计草稿，未实现）

- **cur 拷贝**（紧凑链里 `build_cur_sub` 的对应物）：分配 staging `cur_tight[d, 1|w_b?,
  n_active]`，把主图 `cur` 的第 `t[order[i]]` 列拷到 staging 第 `i` 列。第一个路由 mm 读
  staging。（全 token 桶、无重排：order 为恒等，原地引用。）
- **ids**：把已暂存的 `ids_exp`/`ids_slot` token 块按 `order` 重排（块 = 该 token 的 slot
  切片）。
- **per-slot 权重**（`weights_norm`）：切 slot `[k_lo..k_hi)` 并把 token 轴按 `order`
  重排。
- **acc 循环**（取代 `exec_layer_burst_chain_buckets` 里的 offset-0 单次 acc）：对每个
  seg 做一次 `ggml_acc_inplace(acc_d, per_token_col_slice, nb1=delta*d_out*4,
  offset=dst*d_out*4)`。src 切片在 `per_token` 里连续，因为链按 tight 序跑。

## 6. 单元测试（计划）

`tests/test_scatter_plan.cpp`，加入 build.bat test 列表：

1. 全 token 桶、token id 已连续 → 单个 delta 1 seg，order 恒等。
2. `t = {0,2,3,4}`（动机例子）→ 一个 run {0,2,4} delta 2（len 3）+ 一个单例 {3}；
   order 把 3 排到 4 之后。
3. 散落很远 token → 各为单例或小的等差组合；验证每个 seg 的 dst 等差落在 n_t 内、
   src 区间互斥且覆盖 [0,n_active)。
4. 逆序输入 → 与升序得到相同计划（order 只依赖值）。
5. 重复 / 越界 token id → 拒绝（返回空 plan / 标志位）。
6. 一致性：把 `order` 应用到 `t` 后沿 `segs` 走，得到 seg 声称的确切 dst 序列。
