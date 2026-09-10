# Route B 整层拥有 - 整层执行、设备本地 buffer、C1 动态搬运

[English](ROUTE_B_LAYER_OWNERSHIP.md) | [简体中文](ROUTE_B_LAYER_OWNERSHIP.zh-CN.md)

> 状态：**设计，2026-09-10**。由"cur 拷贝 / 接缝"讨论收敛而来。
> 取代 per-GPU MoE backend 草案（那个只解决静态放置）。相关：
> `ROUTE_B_GPU_PHASE.md`（整层自调度）、`M2_DEVICE_EXECUTOR.md`（每设备执行器 +
> arena）、`DENSE_PLACEMENT.md`（C1/C2 + Phase 2 迁移）、`EXPERT_MOVE_PIPELINE.md`
> （搬运机制）、`BUCKET_FAST_PATH.md`、`WORK_IN_PROGRESS.md` P（sync 实测）、
> `VRAM_DMA_MOVE.md`。

## 1. 目标

Route B 拥有**整层**：一层的每个计算节点（C1 dense、gating、MoE 链）都经我们的
backend 执行，且该层的激活值放在**我们自己的 buffer**里、落在该层所属设备上。这是
终态形态，带来三个"只拥有 MoE 闭包"做不到的东西：

1. **没有多余的接缝拷贝**。cur 和 moe_out 变成我们图内的、设备本地的张量；sched
   不再插 host 拷贝，执行器也不再重复上传。同设备桶原地读 cur，只有远端 pool 才拷。
2. **每层一个 backend split**（所有层都归我们 → 整图一个 split）：每层跨 backend 的
   submit/sync 消失。
3. **C1 动态搬运**。Route B 拥有 C1 放置权，于是 C1（及其 KV）能在运行时换设备，层图
   按新放置重建。

不在范围内：C2 放置策略（`--dense-placement` 已能表达）；自写 dense kernel（标准算子
委托给设备 backend）。

## 2. 背景：三个代价

### 2.1 接缝拷贝
MoE backend 声明的是 host compute buft（`STREAMMOE_HOST`，`moe_backend.cpp:264`）。
sched 只在**目标 backend 不支持源 buft** 时才插拷贝（`ggml-backend.cpp:1026`，拷贝
条件 `:1325`/`:1399`）。于是 C1 产出的 cur 被拷到 host，执行器再上传到设备 stage
（`bucket_upload_leaf`，`minigraph_exec.cpp:635`）；moe_out 又跨回 dense 设备。每层
两次不含信息的跨界。

### 2.2 每层 sync
dense（llama backend）与 MoE（我们的 backend）逐层交替，sched 在每个 backend 边界
同步。实测 decode `tail_sync 1.68 ms/层`（`WORK_IN_PROGRESS.md` P2）——正是这个固定
开销让 CPU 路径比设备路径还快。

### 2.3 静态 C1
`dev_layer` 加载时定死（`llama-model.cpp:1492-1494`），C1 搬不动。
`DENSE_PLACEMENT.md` §5 Phase 2：动态迁移要求 route B 拥有 C1 权重与执行。

## 3. 架构

### 3.1 整层闭包
把私有化集合从"MoE 链"扩到**整层**。层归属已有（`moe_chain_layer_of_node`），闭包
变成第 L 层的所有计算节点（dense、gating、MoE）。验证门（Check 1）变成：第 L 层的
任何中间量，除交给下一层/residual 的层输出外，没有层外消费者。

### 3.2 单 backend、内部放置
整图归我们的 backend，于是 sched 只产生**一个 split**。`graph_compute` 拿到整图，
**逐层**执行，内部把每层激活值放到该层设备上。sched 的分配退化为模型输入（embd，
被强制 CPU）与模型输出（logits）；每层激活值都是我们的。

这就是 per-GPU MoE backend **不必要**的原因：层激活值不再依赖 sched 的
per-backend compute buft。（在只有部分层归我们的过渡期，`supports_buft` 接受所有
设备 buft 仍有用。）

### 3.3 dense 委托
我们不写 dense kernel。每个 dense 节点克隆一份（新张量、同 op / op_params），在该层
设备 backend 上跑；dense 权重先原地引用（Phase 1），迁移阶段改为来自 route-B C1 池。
attention 原地引用 llama 的 KV 张量。MoE 链走既有桶引擎。

### 3.4 设备本地 buffer 与 fold
每层一个 per-device arena（既有 verify interval 布局，从 MoE 闭包扩到整层）。cur 是
层输入；moe_out 是层输出，落在该层设备上。专家 fold 在设备上写 moe_out（现在的 host
`layer_fold`，`minigraph_exec.cpp:492`，改为设备侧归约）。

### 3.5 跨设备搬运
专家按 pool pin；`build_mix_plan` 保证一个桶的专家就在该桶的 pool 上。唯一跨设备的是：

- **cur**：owner（层设备）→ 远端 pool。满宽桶 → 整份复制；subset 桶 → index-gather。
- **acc_d**：远端 pool → owner，用于设备侧 fold。

都是设备传输（D2D 或 transfer queue），不经 host staging。

### 3.6 ids 与那一次残余的 host 往返
调度（pin、`build_mix_plan`）在 host，需要当前 token 的 ids，而 ids 是设备算出来的。
这是唯一不可避免的每层 host 往返（ids 很小，贵的是"等"）。可用的杠杆：

- **预取 / 流水**：用历史路由提前装载专家，当前 ids 不进关键路径（只有 miss 才 stall）；
- **整层全驻留**：某层需要的专家已驻留时，绕过桶引擎，直接跑设备侧标准
  `MUL_MAT_ID`（ids 作为设备张量）——该层零 host 往返。

### 3.7 C1 与 C2
C1 是层内 dense；它的设备就是"层设备"，决定 cur/moe_out 落在哪。C2
（`token_embd` / `output` / `output_norm`）在层外，不碰接缝；它保持为放置策略
（`--dense-placement C2:<dev>`），只在净收益为正时放设备。

### 3.8 C1 动态搬运
Route B 用一个受管池拥有 C1 权重（复用 `EXPERT_MOVE_PIPELINE` 机制：move worker、
hysteresis、copy-then-release）。迁移决策发生时，C1 权重与该层 KV 一起搬，层图按新
放置重建。因为我们拥有该层的执行与 buffer，这只是个放置参数，不是 llama 的结构改动。

### 3.9 ids join：静态前缀 / 动态后缀

整层接管**不会**消除数据依赖 `gate -> ids -> 专家装载 / 槽映射`。所以每层天然是两段，
在 ids 处 join：

- **静态前缀**（可提前构建，不依赖 ids）：dense prefix（attention / norm）+ gating 到
  `topk`（产出 ids）。按 C1 的设备派发。
- **ids join**：路由 ids 是设备算出来的，必须进 host——因为 host 拥有 (a) 装载决策
  （pin 哪些专家）与 (b) expert-id → 池内 slot 映射（`pin_slot`，即 `w3d` 的槽轴）。
  这是唯一残余的每层 host 往返；ids 很小，贵的是"等"。
- **动态后缀**（依赖 ids）：`weights = get_rows(probs, ids)`、专家 mm、fold、tail
  （residual / 下一层前缀）。随 ids 重建。

为什么 host 往返是强制的：route B 把专家重排进有界的槽池、部分驻留，所以 ids 不能只靠
设备侧 `MUL_MAT_ID` 消费——槽映射与装载决策都在 host。唯一的逃逸是"专家全驻留且按原
GGUF 顺序"（有界池下不可行），或预取让装载决策离开关键路径。GPU 没有"查 ids、miss 就
搬进来"的机制；驻留由 host/DMA 管。

后果：

- "整层图" = **静态前缀图 + 运行时后缀图**，不是一张图；ids D2H 是 join 点。
- 单设备：后缀可以把 tail（甚至下一层的前缀）吸收进同一次提交，于是唯一的一次 sync
  就是等 ids（每层 1 次）。
- 多设备：远端 pool 的专家子图 + fold 多一次 converge（每层 1 次 ids + 1 次 converge）。

整层接管后**仍然需要**专家闭包分析，但角色变了：从"私有化边界 + 外部消费者校验"变成
**层内执行分区**（专家域 → 桶引擎，dense 域 → 委托设备 backend）；外部消费者校验简化
为"层内任何中间量不得逃出本层（除层输出）"。

## 4. 执行流（每 token）

1. `graph_compute` 对整图被调用一次。
2. 对每层 L，在 ids join 处切分：
   - **静态前缀**：在 L 的设备上跑 dense prefix（attention / norm）+ gating 到 `topk`
     （克隆 + 委托）；
   - **ids join**：D2H ids；pin/解析专家；构建 mix plan；
   - **动态后缀**：在 L 的设备上构建并提交后缀（路由权重、专家 mm、fold、tail）；
     各 pool 的专家图异步提交，调用线程跑 CPU pool 图；汇聚，在 L 的设备上 fold
     `acc_d`，写 moe_out，再跑 tail。
3. 输出 logits 到模型输出。

## 5. 里程碑

- **L1 - 整层捕获 + 验证**：闭包捕获/验证扩到整层；debug 门控下 dump。不改执行。
- **L2 - dense 委托**：层执行器在既有 MoE 爆发前后跑 dense head + tail（克隆 +
  委托）；ids D2H。门：backend gate 数值。
- **L3 - 设备本地 buffer + fold**：层激活值进该层设备上的 arena；设备侧 fold；跨设备
  cur/acc 搬运。
- **L4 - C1 动态搬运**：C1 权重池 + move + KV 跟随 + 重建。
- **L5 - ids 流水**：历史预取；全驻留绕过桶引擎。

## 6. 待定问题

1. dense 权重：L4 之前保持 llama 加载原地引用，还是 L2 就进 route-B 池？
2. KV：原地引用 llama 的 KV；确认 L4 的搬运路径。
3. dense 子图：逐节点克隆并委托（复用既有克隆机制），还是整段 dense 子图一次交给设备
   backend？
4. L2 过渡期：一次性认领整图，还是逐层认领？

## 7. 验证门

- 纯 RAM 路径保持 IDENTICAL（L2 落地前行为不变）。
- L2 之后：同 flavor、对当前 split 路径走宽松 backend gate（cos ~0.999x）；
  `SM_COPY_TMR` / `[seam]` 显示无 cur/moe_out 拷贝。
- 设备图节点数与逐层耗时用既有 TMR 钩子。

## 8. 风险

- 拥有 attention 意味着要负责 KV 顺序；出错是错数值而非崩溃。以纯 RAM 门为锚。
- 克隆整层会放大建图面；验证 Check 2（链完整性）必须扩到整层。
- buffer 尺寸：整层 interval 布局要重算（现在只覆盖 MoE 闭包）。
