[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B 多设备专家池设计方案（Phase B）

> 状态：设计，**2026-09 多次修订**（取代 2026-08-29 旧版）。基于 route B 第三路径（官方
> `ggml_mul_mat_id` 内核 + 均匀 stride 槽池，`docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7）。
> 保留已验证事实：Vulkan/CUDA 原生支持 `MUL_MAT_ID` + 池槽 3D 布局；RX 590 真实 8 GiB
> `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` 堆；GGML_VULKAN=ON 干净构建。
> 相关：`docs/MULTI_MODEL_POOL.md`、`docs/MULTI_SUBPOOL.md`、`docs/Backend.md`。

## 1. 目标与主线

被选专家分布到多个设备池（CPU RAM + GPU 显存），各池预算有界；GPU 是一等算力节点。
MoE 执行**不是**在 llama 图上逐节点：stream_moe backend 在 `graph_compute` 收**整层 MoE**
并**自调度**——读 ids → 解析每个专家的池 → 按设备分派子集 → 每设备在固定执行区内跑专家链
→ 汇总 → 写回层输出。

**2026-09 已确认主线**：compute 收整层、自调度到各设备——必选项。

## 2. 现状机制（保留/扩展）

- **为何注册 backend**：拿调度器正式 `graph_compute` 回调。`supports_op` 认 MoE 专家
  `MUL_MAT_ID` 域（+ 源在我们 host buft 的视图）。
- **mini-graph 现状**：每 MUL_MAT_ID 在私有 leaf-only 图用**官方内核**重算：`w3d` 伪装池
  槽、`ids_slot` 专家 id→槽、`b_leaf` 包激活、写主图 dst。CPU-only、零拷贝。
- **通知协议**：compute→调度 MPSC+睡版本字；调度→compute 版本 bump+唤醒。**与设备无关、
  保留**。

## 3. 整层自调度

主图照常建每层 MoE 节点（reuse/plan 稳定）。`supports_op` 扩到整层链域。执行时
`graph_compute` 收到一层 MoE 链：

1. 读 ids（门控留在图/dense 侧——ids 已算好）
2. 每个 (L,E) 解析到池槽（目录 scan、pin/wait 同现状）
3. 选主执行设备（§5）
4. 非主设备各拿子集（ids/weights 切片）
5. 每设备在自己固定区跑专家链（官方 mul_mat_id + silu/mul/add）；大中间不离开区
6. 汇总专家贡献（embd 级小）回主设备，写 `ffn_moe_out-N` dst

动态全在执行层；图/reuse 静态。

**执行边界**：整段层内 MoE 链由我们执行、写我们自己的固定区；主图只留两头（`cur` 输入 /
`ffn_moe_out-N` 输出）。"跨 device 搬运"不是物理定律——只有中间放 llama 管 buffer、sched
按 backend 归属插 cpy 时才出现；中间在我们区永不拷贝。代价：链执行归我们编排，主图两头
之间不能保留单独 MoE 链节点。

### 3.1 私有化准入验证（fail-fast，无 fallback）

藏中间只在"该层 MoE 链之外无节点消费它们"时安全。大声验证、拒绝运行：

- **挂点**：三个 `model.build_graph()` 调用点后调 `stream_moe_verify_graph(gf)`（我们侧
  判定；只在图真重建时跑，can_reuse 跳过）。
- **检查 1——无外部消费者**：每个私有化中间的全图消费者必须在同层 MoE 链内（最终汇聚
  `ffn_moe_out-N`）。豁免：`ffn_moe_out-N`（输出端，被残差消费）；门控段（dense 域、不
  私有化、不验）。
- **检查 2——链完整性**：我们 `graph_compute` 自检收到的 split 确实含该层完整期望链
  （全部 expert-buft MUL_MAT_ID + 链内 view/silu/mul/add）。缺/错位 = sched 启发式把链
  节点分错侧——**大声失败，而非静默错数**。
- **失败模式**：任何违规 → log 完整上下文（中间/外部消费者或缺失节点/层/arch）后退出。
  **无静默 fallback、无逃生门。**
- **私有化集合从严**：拿不准是链还是门控的节点宁可不私有化。

### 3.2 为什么整层落一个 split（sched 机制，已查证）

ggml-backend-sched（`ggml/src/ggml-backend.cpp`）按连续 backend 切分：
- **pass1-4（backend 分配）**：有权重节点跟权重 buffer backend；无权重节点向相邻扩展
  （GPU 优先、CPU 最低），`supports_op` 把关。view/reshape/transpose 有**确定性规则**
  （pass4：view 永远跟 `view_src` 同 backend）。
- **pass5（切 split）**：同 backend 连续节点一段；backend 变化/不兼容权重输入处开新 split。
- **跨 split 输入**（约 1400-1420 行）：src 在不同不兼容 backend → 自动在 split backend
  建 copy 并把 `node->src` 重指。

推论：supports_op 全收整层链 → 连续 → **一个 split 进我们 graph_compute**。门控段留在
dense split；ids/weights/cur 作为我们 split 输入——**sched 自动拷，llama 层无需手动广播**。

### 3.3 两个端：cur / moe_out

- **cur**：sched 处理跨 split 输入拷贝。优化：只有 `ggml_backend_sched_buffer_supported`
  失败才拷——若我们 `supports_buft` 认 CPU host buft（都 host 内存），cur 零拷贝共享指针。
  自调度内向各 device 池广播 cur 是我们自己的 staging（llama 层之外）。
- **moe_out**：链输出端节点 → backend 归我们 → gallocr 把它的 dst 分配在我们 buffer →
  **我们写真实汇总结果进去**（图输出端必须真写，残差要读）。残差 add（dense backend）
  消费它，sched 处理跨 split（host 共享或小拷）。
- **藏节点**（链中间）：dst 不真算，但 gallocr 仍在我们 buffer 分配空间（空洞——host 内存
  便宜，可接受/后续优化）；检查 1 保证无外部读，空洞安全。

### 3.4 异步纪律：graph_compute 出口同步

sched 层跨 split 先后是**显式机制、与 copy 节点无关**——每个 split 边界同步前一 backend
异步工作（事件或 synchronize，因 allocator 跨 split 复用 buffer）。所以零拷贝共享 host
指针**不会**丢掉我们 split 与 dense split 之间的先后。

真实风险在**自调度器内部**：`graph_compute` 契约是"返回 = 本 split 完成、dst 就绪"。若
我们异步提交 per-device 子链（如 Vulkan queue）不等就返回，下个 split 会信一个未就绪的
dst。纪律（必选非测试）：**graph_compute 返回前，等所有内部提交的 per-device 工作**
（每设备事件/synchronize），再写 moe_out。CPU-only 现在天然同步；GPU 异步才触发——M3 留
"故意变慢验证正确性"回归。

## 4. device_pool[] + 每设备固定执行区

现状单池目录已带 pool 维（entries 是 (L,E)×pool、scan 扫多池）——保留。要改的是物理槽
寻址：

```
device_pool[] = {
  ggml_backend 句柄, 物理基址(显存/内存), 槽 stride, 容量,
  每池层组划分(subpools_), 每池槽 meta,
  固定执行区,
}
```

**每设备固定执行区**：预分配一块按最坏层形状：

```
arena = [ staging(cur 副本) | exec 区(链中间原位覆盖) | result(专家贡献) ]
```

价值不在省分配——在于链中间不离开这块，跨设备只搬小量（§5）。注意原位覆盖顺序（层间
串行安全；显式读写窗口仍要）。

### 4.1 多位置调度原语

目录本来就是"**非包含非独占的只读多位置缓存**"：entries `(layer, expert) × pool` 让同一
专家可同时在多个设备有位置（页缓存多副本语义）；scan 找第一个可用。调度范围 = 一个
(model, expert-kind) × 全部 `device_pool[]` 副本（含主/草稿池；异构专家种类保持独立池）。

原语（VRAM↔VRAM 搬迁刻意不做——GPU 间搬不如重读/DIO）：

| 原语 | 语义 | 实现本质 |
|---|---|---|
| `ensure(ram)` | 专家在 RAM 可用（pin）| 目录 RAM 条目 + DIO 加载 |
| `ensure(vram)` | 专家在 VRAM 可用 | 目录 VRAM 条目 + 加载（DIO-into-VRAM 或 RAM→拷）|
| `move ram→vram` | 驻留换到 VRAM | vram alloc + 内容拷/重读 + 双条目过渡 + 释放旧 |
| `move vram→ram` | 反向 | 反向 |

- `ensure` = read-to-X、幂等（= 现在 pin/wait + 目标 pool 参数）。
- `move` = 先建新位置副本再删旧（目录短暂双条目——非独占模型天然允许，任何时刻不丢）。
- 副本让"逐出"变"逐副本"——每池条目独立计频/逐出。

## 5. 门控 vs 专家链：分界判据

**分界判断**：门控段（`cur×gate_proj` = dense mm、softmax、topk）只碰 dense gate_proj；
topk 产 id 但**从不按 id 取专家权重**。链段（gate_up/up/down 的 `mul_mat_id(w, cur, ids)`）
**按 id 访问专家权重**——这才需要池。判据 = **"节点是否按 id 访问专家权重"**（不是"有没有
id 张量"）。

**实现**：不猜名字——看 `src[0]` 的 buffer 类型：expert-buft 的 src[0]（MUL_MAT_ID）→
链；其余（dense gate_inp 等）→ 门控。链内无权重的 view/silu/mul/add 靠 backend 扩展归属，
由检查 2（链完整性）把"希望它对"变成"确保它对"。

层内归属规则：

- **门控**是 dense 计算 → 在该层 **dense 设备**算（静态，由层排布决定；dense 权重绝不为
  门控搬）。ids + per-expert weights 再分发各执行池（小：n_used×tokens）。
- **主执行设备**（汇聚点）= 本次被选专家驻留**最多**的池——在**所有专家 pin/解析完**之后选
  （本次真实逐池分布已确定——argmax，无启发式）。
- **cur 广播**：每个执行池需要 cur——每池一份（最小量）。
- **汇总**：主设备加自己 + 送回贡献（embd 级小）→ `ffn_moe_out-N` → 回 dense 设备
  residual/add/norm。

数据流：大中间永不跨设备；只搬 ids/weights/cur 广播/专家贡献/moe_out——全小。

## 6. dense/KV 按层分配（独立后续）

dense 按层剥离、每层指派设备（静态、不迁移）；KV 跟层 dense 设备（例：后 N 层 dense+KV
放 RX 590）。价值：dense 设备 = 门控设备 = 该层理想主设备 → 零跨设备。

- llama 原生层 offload 是**连续段**（`-ngl N`）——任意层表（"layer 27-29 上 GPU"）要改
  loader/`dev_layer`（独立工程块）。
- 只有专家池动态/可迁移；dense+KV 加载/初始化定死。动手前量每层尺寸算显存账。

不阻塞主线：先 M1/M2，用原生连续 offload 摸交互再定自定义层表。

## 7. 数值/正确性基准

- 单专家结果官方内核同权重 → 一致；专家间 add 顺序可能 ulp。用 v2/deepseek 对齐把关。
- 池大小绝不改变数值（8192 vs 71680 已验 IDENTICAL）——保持回归护栏。
- 边界误分类由检查 2 在**产出任何数值前**结构性抓出（不是端到端数值兜底）。M3 补一个
  针对性回归（无权重节点同时邻 dense 与 expert backend）做数值后盾，但非主防线。
- 任意两后端约 5% 路由翻转 + hidden 放大是固有噪声（docs/BACKEND_DIVERGENCE_ANALYSIS.md）
  非 bug。

## 8. 里程碑（风险递减）

- **M1** — CPU-only 自调度骨架：两个伪 device_pool（都 CPU、独立池+执行区），钉死 ids
  分组/分派/每池子集链/汇总逻辑与数值==官方（v2 对齐）。定义自调度器接口。
- **M2** — device_pool[] 落地（backend 句柄/槽寻址/固定区）；一个真 Vulkan + CPU 混合，
  实跑门控跟 dense + 主设备汇总 + graph_compute 出口同步纪律。
- **M3** — 同层跨设备分散（2 GPU0/2 GPU1/3 CPU）+ 固定区读写窗口全速 + 故意变慢正确性回归。
- **M4** — 设备迁移（目录 set/clear + move 原语）。

## 9. 保留的已验证事实（构建/硬件）

- Vulkan + CUDA 原生 `MUL_MAT_ID` + 槽 3D 布局——w3d 伪装槽技术跨后端可移植。
- RX 590 8 GiB BAR1 是真 `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` 堆——设备池成型后专家
  加载仍可零拷贝 DIO 进映射显存。
- `GGML_VULKAN=ON` 干净构建；`libomp.dll` 放 exe 旁。现实目标不变：dense 上 Vulkan +
  热专家上 GPU + 多数专家留 RAM。
