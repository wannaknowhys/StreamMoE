# 推理后端分歧分析（BACKEND_DIVERGENCE_ANALYSIS.md）

> 2026-09-03。背景：`--prefill-from`（129-token 单次 decode）+ route-B @ 8GB 池卡死，
> 修复后做"与原版是否基本一致"验证，发现 moe 与 upstream 有分歧——本文件记录完整调查链与结论。
> 相关：E3（docs/WORK_IN_PROGRESS.md）、修复 commit `6111cc7`、verify 修复 `4bf25d3`。

## 1. prefill-from 死锁（已修复 6111cc7）

- **症状**：StreamMoE_dump（route-b + prefill）+ 8192MB 池，129-token prefill-from 卡死；
  日志停在 init_mappings 后，2.5GB 日志全是同一死循环 `ALLOC_FAIL L29/E106 requeued` + `NO_VICTIM (g=1)`。
- **根因**：每组池按 byte-fraction 分配。group 1（仅 layer 29）只分到 76 slots（60%），
  而 129-token 单次 decode 在 L29 一层被 gate 选中的不同专家可 >76。compute 把整层专家
  `pin_expert` 并 hold 到层结束（rc>0 不可驱逐）→ 第 77 个专家无空槽、无可逐 victim →
  worker 无限 requeue + compute 死等。17-token 一层活跃集 <76 不触发（HTTP 分片 decode 因此一直没暴露）。
- **修复**：每组分池**保底 = 一层全量专家**（`per_layer x expert_size`，组容量 ≥ 单层最坏活跃集，
  驱逐永不需动本层活跃 pin → 死锁物理不可能）；剩余预算按 byte-fraction 给多层组保留驱逐弹性；
  总池 < Σ 保底 → init fail-fast 报错退出（`pool X MB too small: needs at least Y MB`）。
- **验证**：8192 池 129-token rc=0（group1 76→128 slots）；512 池 fail-fast；**8192 vs 71680 池
  产物 IDENTICAL**（证明修复只改容量分配、零数值影响）。

## 2. moe vs upstream 分歧调查（gemma original 布局）

用 verify/关联分析（temp/analyze_div.js）对比同输入（129-token prefill-from）的 moe 与 upstream 产物：

| 指标 | 值 |
|---|---|
| 专家历史条目分歧 | 3344/61920（5.4%）|
| 路由集合一致的 token | 25/129 |
| 路由一致 token 的 hidden cos | mean **0.9996**（min 0.9987）|
| 路由分歧 token 的 hidden cos | 低至 0.975 |
| 最早路由分歧位置 | 全部在 layer > 1 |
| KV cos（kv_cos.js）| mean 0.9998（数值级一致；verify_prefill 字节 diff 是 f16 表示微小差）|

**机制链**：同路由下 moe 与 upstream 就有微小浮点差（cos 0.9996，非 repack 0.508 级严重分歧）
→ 门控概率近等处在较深层发生 top-k rank 翻转/选相邻专家（路由分叉）→ 后续 hidden/KV 分歧放大。
**路由分叉是"放大器"不是起点**；起点是 ~0.04% 的同路由计算差。

## 3. 关键对照：CPU vs Vulkan（纯上游内部）

同输入跑 upstream_dump（CPU）vs upstream_vulkan_dump（`-ngl 2`，GPU offload 2 层）——**两个纯上游原版之间**：

| 指标 | moe vs CPU | CPU vs Vulkan |
|---|---|---|
| 专家历史分歧 | 5.4% | 5.5% |
| 路由一致 hidden cos | mean 0.9996 | mean 0.9996 |
| 路由分歧 cos 下限 | 0.975 | 0.960 |
| 最早路由分歧 | 全部 >L1 | L0/L1 就有 15 个 |
| embd/hidden token0 cos | 0.9817 | 0.9830 |
| KV diff 处 | 58 | 60 |

**结论**："同路由浮点小差 → 路由边界翻转 → 下游放大"是**任何两个计算后端对比的固有属性**
（上游自己 CPU↔Vulkan 就产生同量级分歧）。moe 与上游 CPU 的差异完全落在该固有噪声框架内，
甚至更温和（最早路由分歧在深层，CPU-vs-Vulkan 在 L0 就翻）。**moe 引擎实现非分歧来源**，
是浮点路径 + 路由边界的固有点火/放大链。

## 4. 验证工具修复

- **verify_prefill.js**（`4bf25d3`）：PREFEXP2 KV 段每个 tensor 头含 4×ne + 4×nb（int64，1708f20 起）
  ——旧 reader 把这 64 字节布局头当数据读，KV 段整体错位、字节对比无意义。已补解析。
- **kv_cos.js**（temp/）：工具本身验证 OK——同文件自洽全 cos=1（258 rows）；真实差异数据
  cos mean 0.9998、无 min 0。之前 v2 场景的 0.94/min 0 是旧文件/旧工具版本场景，非当前逻辑 bug。

## 5. 可用结论（后续排查别再当 bug）

- moe 相对 upstream 的 KV/hidden/logits 差异若落在"路由一致 ~0.9996、路由翻放大 ~0.96-0.98、
  专家历史 ~5% 条目"范围 → **任何后端对比的固有噪声**，不是引擎 bug。
- 需要"路由 IDENTICAL + cos→ulp"的强一致验证 → 用 v2 布局（不触发边界翻转）或 deepseek Q8
  （不匹配 repack → 同内核）。
- 池容量验证口径：moe 不同池大小产物必须 IDENTICAL（`--moe-ram-pool` 只影响驻留不影响数值）。

## 6. 2026-09-05 补充：GPU（vulkan）执行无"绝对理想还原"——路线 B 也不例外

> 触发：K6 vram GPU 数值门。route-B 在 RAM:1024+Vulkan0:2048 下跑 129-token prefill-from，
> 对 CPU 基线 DIVERGED（embd cos 0.988、expert flip 3336、10 unexplained）。一度怀疑
> v2align 文件/loader/route-B 混算 bug。判别实验证明与 route-B 无关。

**判别实验链**：
1. **同 build 纯 RAM**：v2.gguf 与 v2align.gguf 对 CPU 基线 `moe_129_8192` 均 **IDENTICAL**
   （embd/hidden 逐字节 + expert history 全同）→ 装载/寻址/SoA 列/route-B 无回归，两文件等价。
2. **route-B GPU 混算**（RAM:1024+Vulkan0:2048）：164 vk round + 152 cpu round 混跑 →
   expert flip 3336、10 unexplained。差异落在跨 backend 浮点混算。
3. **纯 upstream_vulkan_dump（无 route-B、原版 gguf、`-ngl 12` GPU 执行）对 CPU upstream 基线**：
   embd token#0 cos **0.983**、expert flip **3436/61920（105/129 tokens = 81.4%）**、
   16 unexplained——**与 route-B GPU 混算形态几乎一致**。

**结论（以后不纠结）**：
- **GPU（vulkan）执行相对 CPU 天生产生 ~80% token 专家翻转 + 数值漂移**，与是否 route-B 无关——
  连纯 upstream 原版在 GPU 上都不能复现 CPU 的逐字节/逐路由结果。**vk 没有绝对理想还原**。
- route-B 的 GPU 混算差异不是 route-B 引入的 bug，落在"任何 GPU vs CPU 后端"固有噪声框架内。
- K6 数值门**预期修正**：目标不是 "GPU == CPU IDENTICAL"，而是 "GPU 上 route-B 装载/寻址无 bug"。
  判据建议：同 GPU 下 route-B vs 纯 upstream 的差异应远小于 GPU-vs-CPU；或结构等价论证
  （w3d 壳 stride=col_stride 与 CPU 同构造、同代码路径）已足够。
- `moe_129_8192_vk` 基线（旧 vulkan build 产物）无需强求重建到与 CPU 一致——vk 基线只能
  追踪"同版本 GPU build 自洽"，且要同版本才能比较。
