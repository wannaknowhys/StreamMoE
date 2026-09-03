# M2 设备执行器 - 按设备分派的专家列链

[English](M2_DEVICE_EXECUTOR.md) | [简体中文](M2_DEVICE_EXECUTOR.zh-CN.md)

> 状态：设计，2026-09。基于 `docs/ROUTE_B_GPU_PHASE.md`（整层自调度 §1/§3）与已落地的
> VRAM 数据层（docs/WORK_IN_PROGRESS.md J：host-map 通道、专家真驻留 device、demote 回 RAM）。
> 本文固定 M2 执行器形态 + 供未来预取策略消费的 profile 通道。

## 1. 目标

私有化 MoE 层按**专家列**跨真实设备执行：每个 MUL_MAT_ID 的激活专家按驻留 pool
分组，每设备在本地算自己专家的**完整列链**（mm -> silu/geglu -> down mm -> weighted
贡献），中间不出设备 arena，结果留各自乒乓，只有**出口节点**（moe_out）把各设备的列
汇总到外部消费方 buffer。同时执行过程产出"每设备每层"计时，供未来预取策略消费。

## 2. 执行单元：设备内专家列链

现状（`minigraph_exec.cpp`）每个私有化计算节点一次一个 CPU 单节点 mini-graph；隐藏输出
落全局两块 host 乒乓。M2 下 MUL_MAT_ID 的激活集按每个专家的驻留 **pool/设备** 划分：

- 每设备一个子计算 = 该设备专家 x 触碰它们的**整条列路径**（routed mm 列 + 层内跟随的
  per-expert view/silu/geglu/down 乘）；
- 子计算**在该设备内跑**（它的 ggml backend、它的 arena）；
- 子结果留该设备自己的乒乓；
- 列映射（哪个设备产出了全局列 (k,t)）由 route B 记录；只有出口节点做 scatter/merge（§5）。

门控段仍留 dense 侧（不变）。ids 按设备分发；激活 `cur` 每设备广播一次。

## 3. 并行与 graph_compute 契约

ggml 已核实事实：
- `ggml_backend_graph_compute_async` 直通 `backend->iface.graph_compute`
  （ggml-backend.cpp:450）——**没有通用异步层**；是否提前返回纯看后端实现。
- GPU 后端（Vulkan/CUDA/Metal）：提交设备队列即返回——真异步；`synchronize`/event 等待。
- CPU 后端：内部多线程算完才返回——同步，无自己的设备队列。

结论：跨后端真并行不会白送。设备子计算统一用 `ggml_backend_graph_compute_async` 提交；
vulkan 天然异步，CPU-pool 子计算需要**一条 route-B worker 线程**才能与 vulkan 重叠。
逐设备完成用 `ggml_backend_synchronize`（vulkan）+ CPU worker 同步收尾。

**契约（不变，必须）**：`graph_compute` 只在每个设备子计算完成后、且出口 merge 写了外部
dst 才返回（GPU_PHASE §4 同步纪律）。异步不得越过接管边界。

## 4. 每设备 arena 与乒乓

- verify（route_b_chain.cpp moe_chain_verify_graph）已算每层奇偶乒乓**预算**（奇偶各自最大
  计算节点输出）。运行时现状全局 host 两块；M2 下每设备拿**自己的**奇偶对，尺寸 = 该设备
  分派链所需（设备 arena：staging cur 拷贝 | 执行区 | 结果列）。
- **原地单 buffer 写普遍不可能**：链计算节点输出与输入尺寸不同（mul_mat_id dst != cur），
  且 reader 落后 writer 一步。长依赖检查已强制 liveness <= 1 计算步——**每执行设备两个
  乒乓区是理论最小**。
- vram arena buffer 是 host map（host-map 通道已落地）；dense 在 CPU 时出口 scatter 可从
  映射直读设备结果。dense 上 GPU 后同一 scatter 目标换 vram（§5）。

## 5. 出口 scatter 通用化

出口节点（moe_out）永不被隐藏——必须给 dense 侧产出真值。通用化：**scatter 到外部消费方
buffer 所在处**，不做 host 假设：
- route B 维护映射"设备乒乓列 -> 主图全局 (k,t) 列"（设备子结果局部列序 ≠ 主 ids 序）；
- 出口处各设备结果列按全局偏移拷入（scatter）出口节点入参区，随后出口链节点照旧执行
  （加和 + 写主 dst）；
- 现在 dense 在 CPU -> 目标是 CPU 主图 dst；dense 进 GPU -> 目标是 vram buffer。
  scatter 由外部数据位置参数化，不是 host 假设。
- host 阶段：逐列 memcpy（所有池 host 可读）。vulkan 阶段：真跨设备列搬运。

## 6. profile 通道（供未来预取策略）

双 ring，不用 tagged union：
- **alloc 请求 ring**（现 MPSC）保持单一职责；读专家申请附带 `total_tokens` + 批次
  `start_rdtsc`（扩 POD `slot_request_t`——它是原子整存 struct，加两字段不破坏该性质）。
- **profile ring**（新，非阻塞低水位投递）：计算线程的 profile 流量不得阻塞调度线程的
  alloc 处理。

聚合：**per-backend 聚合器挂 `moe_backend_ctx`**（不 globals），按设备记
`{本层已算专家数, 累计 delta_rdtsc}`；层出口节点每设备每层推**一次** profile 事件，
无每节点洪峰。

profile 事件负载（POD）：`{layer, device_id, n_expert, delta_cycles, batch_total_tokens,
batch_start_rdtsc}`。IO 完成事件已带 req->done rdtsc（async_load 字段）。策略消费者（未来）：
- 每层计算 lag（稳态，经 total_tokens 按 token 进度对齐）+ io lag -> 预取多少专家；
- 每设备每专家平均耗时（delta/n_expert，防忙设备被误判闲）-> 预取向闲设备倾斜。

策略本身后置（调度线程策略重构）。

## 7. 迁移步骤

1. **M2-1** 全专家 vram、单设备：真 vulkan 列链执行（vram 内设备 arena/乒乓，mini-leaf
   图提交 vulkan backend），对 vk 基线 IDENTICAL。数据层之上一次只引一套新变量（vulkan 执行）。
2. **M2-2** 双通道真并行：CPU-pool worker + vulkan async，子计算按 pool 拆；graph_compute
   全同步收尾。IDENTICAL。
3. **M2-3** 出口 scatter 通用化：写回外部数据位置（dense 位置变参数），列映射 + scatter。
   混合 RAM/vram 激活集（J6）在此框架自然落地。
4. **M2-4** profile 埋管：ctx 聚合、profile ring、事件 struct、io+device 完成时间戳。
   与 1-3 独立，可先落。

实现中遇到设计问题先问再写。
