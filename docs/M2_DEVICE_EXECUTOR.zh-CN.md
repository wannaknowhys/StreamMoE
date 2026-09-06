# M2 设备执行器 - 按设备分派的专家列链

[English](M2_DEVICE_EXECUTOR.md) | [简体中文](M2_DEVICE_EXECUTOR.zh-CN.md)

> 状态：设计，2026-09（修订：不写过渡代码，直接最终形态按组件推进）。基于
> `docs/ROUTE_B_GPU_PHASE.md`（整层自调度 §1/§3）与已落地 VRAM 数据层
> （docs/WORK_IN_PROGRESS.md J：host-map 通道、专家真驻留 device、demote 回 RAM）。

## 0. 2026-09-05 决策补充（用户）——布局由 verify 分析产出，而非执行路径拍脑袋；每设备整链到出口

历史教训：当前结果缓冲是一堆补丁。`hide_burst` 强制每节点全分配；ping-pong 曾被
"long-range dep: L0 compute#4 reads compute#1" 触发后直接关掉（`g_pingpong_ok=false`），
变成全 full-alloc。那是**数值错乱下瞎改**的补丁，真正根因疑似别处（DIO bug）。所以
缓冲布局问题必须靠**分析**回答，不在执行器里试补丁。

用户目标形态：

1. **布局分析放 build-time verify**（与已收集每层闭包 compute 序列同一次 pass）。对每个
   compute 节点的**结果**，从最后一个节点**倒序**回扫消费者找它的 free 时刻 = 最后一个
   还读它的节点（last-use）。得每个结果的活区间 `[produce, last_use]`。
2. **闭包内跨节点依赖图**（不是跨层——层间独立，Check1 已验证）。两种形态：
   - 链式/近邻（结果只被下一节点读）：乒乓（两块交替结果缓冲）够；
   - 长距（产者隔很多节点才被读，如 compute#4 读 compute#1）：需更大缓冲组或 interval 分配。
   先做 **DEBUG DUMP + exit 0**（打印每个结果的 free/last-use + 依赖图），再决定分配器。
3. **每设备整链算到闭包出口**：持有激活专家的每个设备把自己专家的**列 mini graph 一路算到
   per-expert 贡献/闭包出口**——不是"mm 在 GPU、再回读给 CPU 跑无权重链尾"。每设备算到最终出口。
4. 缓冲尺寸 + 每步输入输出**相对偏移**由 verify 分析导出（执行期不猜）。产物 = `moe_node_plan`：
   per-device arena 尺寸 + 每步 out_off / in(producer+offset)。
5. build-time 为每层备一张**拓扑模板**（闭包形状静态）。执行入口从 verify 产物 + **本次实际
   pin 到的专家分布**实例化（填 data 指针/ids/偏移，不改结构）。
6. 执行：异步提交其它 device 的实例化 mini graph，调用线程跑本地路，然后**汇聚节点强制全等待**
   （同步每 device）把各贡献汇总出闭包。

保留理由：设备 mini graph 在设备上算 per-expert 列路径；只有（小的）贡献/merge 跨界。
大中间量永不离开设备 arena。graph_compute 仅在层汇聚同步后才返回（§3.4）。

构建顺序注意：**分析（倒序 last-use dump）是第 1 步且纯观测**——先做、dump、`exit 0`，
用真实依赖数据决定分配器，再写任何执行器代码。

### 0.1 已落地布局 + 与理想差距实测（2026-09-05，commit e6995dd）

已落地：`moe_chain_assign_backend` 现在为每层算**节点级 interval 布局**（常算，非 debug）：
每个结果取最后 reader（last_use），按 exec 序贪心 first-fit——占用者已死（last_use < 当前序）
的 slot 可复用。产物 `moe_layer_exec_t.out_off[i]` + `result_bytes` + `layout_ok`；
exec `hide_burst` 把 compute[i] 输出写 `out_off[i]`（无布局时回落 per-node bump）。
验证：gemma + deepseek 对 CPU 基线 IDENTICAL；`STREAM_MOE_CAP_DUMP` 打印布局。

**布局随 ubatch 自动缩放**：布局存 `g_layer_exec`，每次 graph **重建**时由 assign 重算。
llama `process_ubatch` 调 `can_reuse(gparams)`——gparams 含 ubatch 尺寸，ubatch 变则
reuse 失败 → 重建 graph → 重跑 assign → 布局按新张量字节重算。同 shape decode 复用
graph（布局不变，而 shape 未变故正确）。实测 12610-token prefill-from（-b 13000、
-ub 2048、ctx 20000）：result_bytes 出现 ~766KB / 12MB / 62MB / 392MB 四档，跟随该图
构建时的 batch。slot **拓扑**（哪些节点共享 slot）只由依赖决定；只有 slot **字节尺寸**
随 ubatch 缩放。

**与理想差距（实测，gemma 14 节点/层）**：
- layout = 191488 B/层 vs full-alloc 参照 417312 B（省 54%）。
- 3 slot/层：slot0 90112B（gate_up 45K + down-mm 90K + weighted 90K）、slot1 90112B
  （geglu 22K + down_scaled 90K + 5 个小 ADD/scale 节点）、slot2 11264B（GET_ROWS 32B + 3 ADD）。
- 理想（best-fit / interval-coloring 最小化总面积）应把 ADD 链放一个 ~11K 小槽、大 mm 输出
  进大槽，可达 ~130K/层 → 当前高约 47%。
- 根因：贪心 first-fit 合法（无活区间冲突）但非最小——小 ADD 被并进大 slot，因为
  first-fit 从不考虑"与既有小槽共享"。分配器"依赖正确、面积非最优"；节点序（exec 序）决定适配。
- 注意：exec 一次只跑一层、arena 跨层复用 → 成本是"一层布局 × ubatch 缩放"，不是"层数×布局"。

**理想算法（后续）**：区间调度/一维装箱——把结果 buffer 看成栈，按 (produce, last_use)
分配并 best-fit-decreasing（大块先落位），即最小化"同时存活字节的峰值"，等价于按大小类
对 interval 图着色。当前 first-fit 是保守通用 DAG fallback；理想保留同一依赖分析，但装箱前
先按大小类分组。普适性：first-fit 对任意 DAG 形状安全（gemma fused gate_up + deepseek
独立 gate/up/down 均已验证）；理想装得更紧但需每 arch 的大小类分组启发。

## 1. 目标

私有化 MoE 层按**每设备专家列链**执行，从一开始就是最终形态（无 CPU-only 过渡执行器）。
有激活专家的每个设备拥有自己专家的**整列 mini graph**：vram 设备一次异步提交；CPU pool 的列在
graph_compute 主线程沿用现有逐节点路径算；只有出口 merge 跨设备同步。每设备计时在完成时刻记录，
供未来预取策略消费。

## 2. 执行单元：每设备专家列 mini graph

私有化链**按设备**执行，每个真正持有激活专家的设备一张 mini graph：

- 设备 mini graph 为该设备专家重建层的列路径：routed mm（gate_up/down）列 + 后续 per-expert
  view/silu/geglu/down/weighted 操作，所有活张量（权重视图壳、中间、per-expert contribution 输出）
  落在该设备自己的 arena buffer；
- 叶子 = 张量壳：权重壳指向驻留 pool buffer 槽（假 base 偏移）；cur/ids 指向 host-visible
  staging 拷贝（每层上传；dense 在 CPU，直至 dense offload 落地）；
- 设备链终点 = **per-expert contribution**（首个跨专家操作之前的最后一步专家独立操作）。
  跨专家求和留在各设备图之外；
- **CPU pool 列**在主线程沿用现有逐节点路径、限定于 CPU pool 的专家（**列片语义**——现
  执行器是"整链全权"，一旦同层激活集跨 pool 必须改成列片）。

列映射（哪个设备产出了全局列 (k,t)）由 route B 记录，出口 merge 消费。

### 2.1 层内收敛：匿名 per-k add 纳入私有链

勘察事实（gemma4）：层链每次 graph_compute **只收 1 个节点**（`n_nodes=1`）；weighted 各 k 视图的求和是**匿名 add 节点树**（无 `ffn_moe_` 名），llama 把它们留在 CPU 默认后端、按 host 指针读我们的隐藏中间。这只在隐藏输出在 host 内存时成立；weighted 列一旦在 vram（vulkan 壳 data），dense 侧匿名 add 会读假指针。决策：**匿名收敛 add 纳入私有化**——weightless op 且其 src（经 view 追溯）到私有化中间则同样私有化，整层收敛在我们 merge 内完成（经 moe_out 写主 dst 同现状）。privatizable 判定改为闭包/递归式，与 verify 的 chain BFS 一致（弃用名字启发）。

### 2.2 执行触发：首节点整层爆发

层按每节点一个 split 到达，执行期没有整层视图。决策：**build/verify 时缓存私有化层拓扑**（per-layer 节点序列/形状/边，gf 上已可遍历）；执行时该层**第一个私有化 split 触发整层**：建各设备列 mini graph、vram 异步提交、CPU 列主线程算、层尾 merge 写主 dst；同层后续私有化 split 空转返回（数据已被爆发产出）。爆发必须在该层 dense 侧输入（cur/ids）就绪后——它们在 llama 调用首个私有化 split 前已算好。

## 3. 并行与同步（无额外 worker 线程）

事实（已核实）：`ggml_backend_graph_compute_async` 直通 `iface.graph_compute`——异步由后端决定。
vulkan 提交设备队列即返回；CPU 同步算完（自身图调用内多线程）。

执行模型（最终形态）：
- vram 设备 mini graph：`ggml_backend_graph_compute_async(vulkan, mini)`——一次提交，GPU 图内无需
  逐节点同步；
- CPU pool 列：graph_compute 主线程算（不需要专门 CPU worker——CPU 工作本就在调用线程，
  GPU 与 CPU 部分天然重叠）；
- `graph_compute` 只在出口 merge 完成后返回：同步 vram 设备、读回 contribution 列、跨专家
  add 写外部 dst。异步不得越过接管边界。

## 4. 每设备 arena 与 staging

- vram arena buffer（vk buft）放 mini graph 的中间与 contribution 输出，尺寸来自 verify 每层
  乒乓预算。每执行设备两个乒乓区是理论最小（原地单 buffer 普遍不可行——输出与输入形状不同、
  reader 落后 writer 一步）。
- host-visible staging（vk host buft）每层收 cur + ids；层异步完成前必须稳定（层内不复用）。
- vram arena buffer 是 host map（host-map 通道已落地），dense 在 CPU 时出口 merge 可从映射
  直读 contribution 列。

## 5. 出口 merge 与 scatter（通用化）

- 出口（跨专家 add 到 moe_out）读每设备 contribution 列并写外部消费方 dst。目标 buffer =
  外部（dense）数据所在处——现在 CPU 主图 dst，dense offload 后 vram。不做 host 假设。
- route B 维护"设备乒乓列 -> 主图全局 (k,t) 列"映射；出口按正确全局偏移读/散布 contribution，
  随后链 add 节点照旧执行。
- vram contribution 经 host map 逐列读回，直到 dense 上 GPU。

## 6. profile 通道（供未来预取策略）

- alloc 请求 ring（现 MPSC）保持单一职责；读专家申请带 `total_tokens` + 批次 `start_rdtsc`
  （POD `slot_request_t` 加两字段）。
- profile ring（新，非阻塞）：计算流量不得拖慢调度线程的 alloc 处理。
- `moe_backend_ctx` 内 per-backend 聚合器按设备记 {n_expert, submit_rdtsc, done_rdtsc}；
  层出口每设备每层推**一次**事件，无每节点洪峰。
- 负载（POD）：{layer, device_id, n_expert, delta_cycles, batch_total_tokens,
  batch_start_rdtsc}。IO 事件已带 req->done rdtsc。策略消费者（未来）：每层 lag + io lag ->
  预取深度；每专家均耗时（delta/n_expert）-> 预取向闲设备倾斜。

## 7. 组件构建顺序（最终形态，无过渡代码）

1. **arena + 壳**：每 device arena buffer（vk buft，verify 预算）+ host-visible staging；
   张量壳 helper（假 base 偏移、buffer 绑定）。
2. **vram 列链 mini-graph builder**：在 arena 重建每设备列路径（权重视图壳、上传 cur/ids、
   arena 中间与 contribution 输出），每设备一张 cgraph，异步提交。
3. **CPU 列片**：CPU 执行器用现有逐节点路径跑自己 pool 的列（列片语义；contribution 输出到
   共享 merge 区）。
4. **出口 merge**：从每设备 contribution 列做跨专家 add（vram 经 host map 读回）写外部 dst，
   保持主图列序；graph_compute 收尾同步。
5. **数值门**：纯 CPU（单列片）与全 vram（单设备）都 IDENTICAL；混合 pool 过同一门。
6. **profile 埋管**：ctx 聚合 + profile ring + 事件 struct + IO/设备完成时间戳。

实现中遇到设计问题先问再写。

## 7.1 现状差距分析 -> 可执行 roadmap（2026-09-05）

现状代码：`exec_layer_burst` 每层在主线程**逐节点同步**跑。`MUL_MAT_ID` 经 `exec_mixed_mm`
（per-pool peel round——CPU 或 Vulkan，每 round 一次**同步** graph_compute + 回读）；weightless
链节点是 CPU 手动 1-node graph。结果缓冲 = §4.1 best-fit 布局的**整层一块**。无 async、无
per-device arena、无设备侧整链。

与最终设计（每设备专家列 mini graph，§1-§7）的差距，按依赖序列出：

**分析层（verify/assign）——大多已落地，需设备化**
- [x] 闭包收集 + 链内依赖/last-use + 布局（best-fit out_off/result_bytes）——完成（e6995dd, 488930f）。
- [ ] `moe_node_plan`（每步 in[prod+off] 相对偏移）——布局只覆盖每节点 out 偏移；输入仍由执行器临时解析。
- [ ] **每设备整链执行规划**：每设备拥有哪些专家 / contribution 列（mix_plan 的 pool 划分），
      按 compute 节点表达（该设备产生某节点输出的哪个切片、需要它哪些输入）。每设备对**自己
      的列**跑完整节点链（算到 contribution），不只 mm。

**Per-device arena（用户决策 2026-09-05：不做按设备收缩）**
- [ ] 参与一层的每个设备各申请**一整层结果块**（现有 best-fit 布局的 result_bytes）——**不**
      按设备列切 arena。理由：各设备布局几何相同（同一 out_off[]）、简单统一；设备列只是运行
      时划分，只影响每设备实际填哪些切片，不影响块几何。成本 = 每参与设备一块整层 best-fit 块
      （小；arena 跨层复用）。
- [ ] 桶执行：设备在自己块内**只算自己分到的桶（列）**，其余切片不动。

**执行器资源**
- [ ] per-device ping-pong / 事件跟踪（async GPU 不能让下一层覆写在飞结果，§3 同步纪律）。
- [ ] 执行入口从 verify 产物 + 本次 pin 分布**实例化模板**（填 data/ids/偏移，不改结构）。
- [ ] 每设备列执行：设备沿层内每个 compute 节点的**自己切片**走（同节点链、限自己列），
      设备侧直达 contribution，无 host 往返。

**异步执行骨架**
- [ ] `exec_round_vk` → 异步提交（`graph_compute_async`）+ 完成跟踪，取代每 round 同步+回读。
- [ ] CPU/VK 重叠：设备图异步提交后主线继续算 CPU 列，层尾 converge。
- [ ] converge 点：强制同步每设备，经 host map 读回 contribution，折进 moe_out（通用出口
      merge/scatter，§5）。
- [ ] 每个参与设备各持自己的**整层结果块**（§4.1 几何，每设备一块，不按列切）。

**验证门**
- [ ] 设备执行落地后纯设备数值门（K6 形态；注意 GPU 对 CPU 无绝对还原——验证结构等价而非字节
      一致，见 BACKEND_DIVERGENCE_ANALYSIS.md §6）。
- [ ] M8 UT（布局自检、设备规划、merge）——test 链接问题是 blocker（stmoe_vk_* 符号需
      ggml-vulkan 链接，B33）。

**profile（延后，§6）**
- [ ] profile ring + 每设备完成时间戳；slot_request_t 已带 total_tokens/start_rdtsc 字段。

建议下一步：让每个参与设备**用自己的桶（列）把整节点链算进自己的整层结果块**（同一 best-fit
布局，每设备一块）。build-time：把设备列划分表达成每节点的切片，供执行器实例化每设备整链模板；
exec-time：每设备异步提交、主线算 CPU 桶、层尾 converge。先做纯分析层增量（CAP_DUMP/CSV +
sim.js 可验证），再在其上加异步骨架。

## 7.2 桶链串行 + per-device 输出区（设计定稿，2026-09-06 用户决策）

多桶拼接长链 mini-graph（形态 B）的收敛设计。三个已确认决策：

1. **长图本身就是串行**——一个 device 一张图、同 backend 按拓扑序执行、无跨节点并行时，
   把各桶链拼进这一张图 = 严格串行：桶2 链只在桶1 链结束后才开始。无需每桶一次 submit，
   也无需人为加依赖边——单图单后端本身就是串行。
   **为什么成立（关键前提）**：执行器不走 ggml 自动 buffer 分配——是 `no_alloc` + tensor 壳 +
   手动 `nd->data = arena_base + out_off`（CPU 路径的 hide_burst 就是这套）。ggml/vulkan 自动
   allocator 看不到桶间复用（两链无数据依赖，会被当可并行、各给不重叠 buffer，arena = 所有桶之
   和）。**手动 out_off 才是让串行复用中间区安全又免费的原因**。这机制与设备无关：vulkan（和
   CUDA）以 `(data - buffer_base) + view_offs` 定位张量在 buffer 内的偏移，从不 deref `data`
   本身（`stmoe_vk_buffer_host_offset` 返回 `vk_ptr_base + off`；exec_round_vk 已这样用壳）。
   所以每节点 `data = arena_base + out_off` 在 vulkan/CUDA 上原样成立。
2. **每桶链尾把贡献 ADD 进本 device 自己的输出区**（同块 arena 内预留区 / per-device 累加器）。
   桶尾 add 消费掉结果后，该桶中间区即死，下一桶链复用**同一批 `out_off[]` 字节区**。最终输出 =
   层尾对各 device 输出区做**一次性读回 / 跨 device merge**（§5：只有 contribution 跨设备边界，
   且只在全部设备完成后一次）。
3. **per-device arena = 整层满宽块**（已落地 best-fit 布局的 result_bytes，几何与 CPU 相同，
   每参与设备一块）。这是最坏情况上界：单桶链峰值 live 字节 ≤ 整层峰值，故块内任意桶排列都安全，
   且跨设备几何逐字节一致（同一 `out_off[]`，无需按设备重算偏移）。简单统一，不做按设备列切片。

下游推论：
- moe_out / 匿名 per-topk 收敛 ADD **不放进任何桶链内跑**。多桶形态下 llama 那棵固定 ADD 树
  （按每 token k 连续搭的）对不上被拆散的 k 列，所以折叠是**汇聚步骤**：按原始 (t,k) 映射 gather
  各设备贡献列、加进外部 dst（§2.1/§5 的私有化汇聚图，phase 2）。CPU phase 1 安全恰恰因为
  scatter 在宿主 main dst 重新凑齐了每 token 的完整 k 行，让现有捕获的匿名 ADD 树原样跑。
- 设备端桶链用**降维列形状**（mm 输出 [d, w_b, n_active]，非满 [d, n_k, n_t]）；verify 算的
  满宽字节级 `out_off` 因此不直接是桶节点的字节布局。继承的是 interval/槽**结构**；跑降维形状时
  per-device arena 字节按设备列跨度线性缩放（推迟到 GPU 桶阶段——CPU phase 1 保持满宽只验数值）。
