# MOCK.md - mock 点登记（M1/M2 实施中"按最终设计但暂用现状顶着"的位置）

> 纪律：
> 1. mock = **能跑通正确路径的实现**（用真实单 CPU/单池现状），不是空壳/空 return；
>    只"结构上标注将来换多池/多 backend/固定区"。
> 2. 每次新增/移除 mock 点都更新本文件。
> 3. 全部代码逻辑通用（不硬编码 gemma 细节）；arch 差异靠主图拓扑天然带入（见
>    docs/ROUTE_B_GPU_PHASE.md §10.1/§10.5）。
> 4. 最终设计参考：docs/ROUTE_B_GPU_PHASE.md。

## 首轮 mock 点（计算线程私有链执行器）

| # | mock 点 | 现状（mock）实现 | 最终（替换目标） | 引入时机 |
|---|---|---|---|---|
| M1 | **专家视图 resolve**：执行器要 `(L,E) -> {pool, slot 基址/stride, 布局}` | 直接经 scheduler（branch_layout/group_of/subpool——现状单池）| device_pool[] 多池查目录 | G3 实现时 |
| M2 | **计算 backend**：mini-graph 委托目标 | 固定 CPU backend（moe_backend ctx 的 cpu）| 每 device 的 ggml backend（Vulkan…）| G3（CPU 正确性先过）|
| M3 | **私有 arena**：中间落区 | 单 ctx scratch arena（现状 moe_backend scratch_arena）| 每 device_pool 的固定执行区（staging/exec/result 分块）| G3 |
| M4 | **view 偏移衔接**：consumer 输入 = 上游 data+off | 直接在私有 arena 上算指针（现状 minigraph 已如此）| 同（C-style struct 记录 offset/stride）——无需换 | G3 |
| M5 | **moe_out 写回** | 单池 = 结果本地已有，直写主图 dst | lead 设备选择 + 跨 device 汇总（M3+）| G3 先直写 |
| M6 | **门控输入（ids/weights）来源** | 从主图读（dense 侧算好，sched 拷入 split）| 同（门控跟 dense device 已定，静态）——不 mock | G3 |

## 将来（M3+）才引入、现在不 mock 的
- device_pool[] 结构（backend 句柄/多基址/多固定区）
- 池本地驱逐多副本化
- 跨 device 分散 + lead/汇总（同层专家跨池）
- 设备迁移（move slot 原语）
- n_copies>1 时读 cur_copy 对应输入（编码注意）
