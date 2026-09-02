[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B 多设备专家池设计方案（Phase B）

> 状态：设计，**2026-09 重写**（取代 2026-08-29 旧版——旧版的"cb_eval pin/unpin + 逐节点
> hijack + HOST_VISIBLE 放置"模型被下述"整层自调度"主线替换）。基于 route B 第三路径
> （官方 `ggml_mul_mat_id` 内核 + 均匀 stride 槽池，`docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7）。
> 保留的已验证事实：Vulkan 与 CUDA 原生支持 `MUL_MAT_ID` 与池槽 3D 布局；RX 590 暴露真实
> 8 GiB `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` 堆；GGML_VULKAN=ON 可干净构建。
> 相关：`docs/MULTI_MODEL_POOL.md`、`docs/MULTI_SUBPOOL.md`、`docs/Backend.md`。

## 1. 目标与主线

被选专家分布到多个设备池（CPU RAM + GPU 显存），各池由预算有界；GPU 是一等算力节点
（不是存储）。MoE 执行**不是**在 llama 图上逐节点：stream_moe backend 在 `graph_compute`
里**收整层 MoE** 并**自调度**——读 ids → 解析每个专家的池 → 按设备分派子集 → 每设备在
自己的固定执行区内跑专家链 → 汇总 → 写回层输出。

**2026-09 已确认主线**：compute 收整层、自调度到各设备——这是必选项不是可选项。

## 2. 现状机制（哪些保留、哪些扩展）

- **为何注册 backend**：注册 ggml backend/device 拿到调度器的正式 `graph_compute` 回调。
  `supports_op` 只认 `*_exps` 的 MUL_MAT_ID（+ 源在我们 host compute buft 的视图）；sched
  恰把这类节点交给我们（每节点一次、顺序正确）。不计划用图手术替代它。
- **mini-graph 现状**：每个原 MUL_MAT_ID 在私有 leaf-only mini-graph 里用**官方内核**重算：
  `w3d` leaf 把池的紧凑槽伪装成 `[ne0, ne1, group_slots]`（data=池基址+分支偏移，nb[2]=槽
  stride），`ids_slot` 把专家 id 翻译成槽索引，`b_leaf` 包主图激活，结果写主图 dst。
  CPU-only、零拷贝（同 RAM）。
- **通知协议**：compute→调度 = MPSC 入队 + 睡在 per-(L,E) 版本字；调度→compute = 版本
  bump + `WakeByAddressAll`/futex 唤醒（目录 set/clear + slot_meta 状态机）。该协议
  **与设备无关、保留**——compute 等"(L,E) 可用"，醒来重扫所有池。

到"整层自调度"的差距不是"多收节点"：而是**层内激活链（gate_up→view gate/up→silu→加权
→down）的执行权**——今天它在 llama 各 buffer 间跑。链不在手里，跨设备必然拷大中间。

## 3. 目标模型：整层自调度

llama 主图照常构建每层 MoE 节点（can_reuse/plan 稳定）。`supports_op` 扩展到整层 MoE
域（`blk.N.ffn_moe_*` 名字：gate_up/silu/down/视图/加/乘）。执行时 `graph_compute` 收到
一层的 MoE split，然后：

1. 读 ids（`ffn_moe_topk-N`；softmax/topk 门控留在图/dense 侧——ids 已算好）
2. 每个 (L,E) 解析到池槽（目录 scan，pin/wait 同现状）
3. 选该层**主执行设备**（§5）
4. 分派：非主设备各拿子集（ids 切片 + weights 切片）
5. 每设备：在自己的固定区内跑专家链（该 backend 的官方 mul_mat_id + silu/mul/add）；
   大中间不离开区
6. 汇总专家贡献（embd 维小量）回主设备，写主图 `ffn_moe_out-N` dst

动态（哪些专家、在哪个设备、主设备选择）全在执行层；图/reuse 形状静态。这是自调度路线
的架构核心收益。

**执行边界**：整段层内 MoE 链（gate_up→view→silu→加权→down→汇总）由我们执行、写进我们
自己的固定区；主图只保留两头——层输入（`cur`）与层输出（`ffn_moe_out-N`）。"跨 device
搬运"不是物理定律——只有当中间放在 llama 管 buffer、sched 按 backend 归属插 cpy 时才会
出现。中间在我们的区里永不拷贝（llama 不管理我们的区）；代价是链执行归我们编排，主图
在两头之间不能保留单独的 MoE 链节点。

## 4. device_pool[] + 每设备固定执行区

现状单池代码的目录已带 pool 维（entries 是 (L,E)×pool，scan 扫多池）——保留。要改的是
**物理槽寻址**（今天一个基址 carve 层组）：

```
device_pool[] = {
  ggml_backend 句柄,
  物理基址(显存/内存), 槽 stride, 容量,
  每池层组划分(subpools_), 每池槽 meta(state/refcount/generation) 或带 pool 维的全局索引,
  固定执行区,
}
```

**每设备固定执行区**（"一个可复用结果区"的点）：预分配一块，按该设备要处理的最坏层形状：

```
arena = [ staging(cur 广播副本) | exec 区(链中间 gate_up/silu/down 原位覆盖)
          | result(专家贡献) ]
```

价值**不在省分配**（llama gallocr + 我们的 scratch arena 已复用）——在于让每设备专家链
中间不离开这块，跨设备只搬小量（§5）。注意：原位覆盖顺序纪律（每步读旧值前不覆盖；层间
串行安全，但显式读写窗口仍要）。

## 5. 计算归属规则（数据流按构造最小化）

- **门控**（`cur×gate_proj`、softmax、topk）是 **dense 计算** → 在该层 **dense 设备**上
  算（静态，由层排布决定——dense 权重绝不为门控搬迁/拷贝）。ids + per-expert weights
  再分发到各执行池（小：n_used×tokens）。
- **主执行设备**（汇聚点）= 本次被选专家驻留**最多**的池。在**所有专家 pin/解析完之后**选
  （阶段 2）——本次 ids 的实际逐池分布已确定，主设备 = 对本次真实分布取 argmax（不需要
  启发式；门控位置才受门控先后影响，而它跟 dense device 走、本就不看分布）。多数专家贡献
  就地相加。
- **cur 广播**：每个要算专家的池都需要 cur——每个执行池拷一次（不可避免的最小量）。
- **汇总**：主设备把本地 + 非主设备送回（embd 维小）的贡献相加 → `ffn_moe_out-N` →
  送回 dense 设备给 residual/add/norm。

大中间（gate_up/silu/down 输出）永不跨设备；只搬 ids/weights/cur 广播/专家贡献/moe_out
——全小量。

## 6. dense/KV 按层分配（独立后续）

把 dense 权重按层剥离、每层指派设备（静态、不搬迁），KV 跟该层 dense 设备。例子：后 N 层
dense+KV 放 RX 590。价值：dense 设备 = 门控设备 = 该层理想主设备 → 这些层零跨设备。

两点：
- llama 原生层 offload 是**连续段**（`-ngl N`、split-mode layer）——任意层设备表
  （"layer 27-29 上 GPU"）要改 loader/`dev_layer`，是独立工程块。
- 只有专家池动态/可迁移；dense+KV 加载/初始化时定死。动手前先量每层 dense/KV 尺寸，
  算显存账。

不阻塞主线：先走 M1/M2，用原生连续 offload 摸交互，再决定自定义层表值不值。

## 7. 数值/正确性基准

- 单专家结果用官方内核 + 同权重 → 一致；专家间 add 顺序可能与官方图不同 → ulp。用
  v2/deepseek 对齐把关（路由 IDENTICAL + cos~ulp）——工具链现成。
- 池大小绝不改变数值：不同 `--moe-ram-pool` 必须导出 IDENTICAL（8192 vs 71680 已验证）
  ——GPU 阶段保持为回归护栏。
- 2026-09 分歧分析（docs/BACKEND_DIVERGENCE_ANALYSIS.md）：任意两后端约 5% 路由翻转 +
  hidden 放大是固有噪声非 bug——GPU 阶段同样会见到这一级噪声。

## 8. 里程碑（风险递减）

- **M1** — CPU-only 自调度骨架：两个"伪 device_pool"（都 CPU、独立池 + 执行区），钉死
  ids 分组/分派/每池子集链/汇总逻辑与数值 == 官方（v2 对齐）。不碰真 GPU；这一步定义
  自调度器接口。
- **M2** — device_pool[] 抽象落地（backend 句柄/槽寻址/固定区）；接一个真 Vulkan 设备 +
  CPU 混合（部分层上 GPU 走原生 offload），实跑"门控跟 dense + 汇总回主设备"。
- **M3** — 同层跨设备分散（2 GPU0 / 2 GPU1 / 3 CPU 那类）+ 固定区读写窗口纪律全速跑。
- **M4** — 设备间迁移（目录 set/clear 协议已支持）。

## 9. 保留的已验证事实（构建/硬件）

- Vulkan + CUDA 原生 `MUL_MAT_ID` + 槽 3D 布局（src0 ne[2]=n_slots、nb[2]=slot stride）
  ——w3d 伪装槽技术跨后端可移植。
- RX 590 8 GiB BAR1 是真 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` 堆——设备池成型后
  专家加载仍可零拷贝 DIO 进映射显存（fallback：RAM staging + 拷贝）。
- `GGML_VULKAN=ON` 干净构建（vendored CMake hook 传工具链）；`libomp.dll` 要放 exe 旁。
  现实目标不变：dense 上 Vulkan + 热专家上 GPU + 多数专家留 RAM。
