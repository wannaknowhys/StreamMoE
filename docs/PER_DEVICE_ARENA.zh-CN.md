# 每设备 Arena 规划（carry 拆分、per-device compact、C1/C2 放置）

[English](PER_DEVICE_ARENA.md) | [简体中文](PER_DEVICE_ARENA.zh-CN.md)

> 状态：**设计，2026-09-14**。承接"整层所有权转正"（`LAYER_EXECUTOR_DESIGN.md`）。
> 现在的单 host arena 太大、且架空 `--dense-placement`；本文规划每设备 arena。
> 相关：`ROUTE_B_LAYER_OWNERSHIP.md`（目标形态 §3.4/§3.5/§3.8）、
> `M2_DEVICE_EXECUTOR.md`（设备执行器）、`DENSE_PLACEMENT.md`（C1/C2）、
> `GRAPH_PARTITION.md`（区域）、`LAYER_EXECUTOR_DESIGN.md`（§4.3 arena）。

## 1. 问题

整层所有权转正后，`layout_arena` 把**每个** compute 节点预分配进单个 host arena，
布局为 `[carry][compact][closure]`。ubatch 512 实测（host buft）：

| model    |   carry | compact | closure |       need |
| :------- | ------: | ------: | ------: | ---------: |
| olmoe    |   62 MB |  218 MB |  142 MB | 0.4–0.5 GB |
| gemma    |  160 MB | 2498 MB |  198 MB |     2.8 GB |
| deepseek | 2630 MB | 1300 MB |  248 MB |     4.1 GB |

三个具体缺陷：

1. **carry 逐层累加。** 每个跨层张量拿到唯一、永不复用的 offset。实测：olmoe 15 个
   `l_out-N`、gemma 29 个、deepseek 84 个（43 个 `l_last-N` + 41 个 hyper-connection
   `node_NNN`）全部共存 → ub512 时 deepseek 2.6 GB。
2. **compact 没打包。** 它是"最坏层所有节点字节之和"，没有层内 liveness 复用。
   gemma 的 2.5 GB 由 dense MLP + C2 的 lm_head logits 主导。
3. **C1/C2 放置被架空。** `--dense-placement C1:<dev>,C2:<dev>` 设了 buft，但
   `layout_arena` 把节点的 `buffer`/`data` 改写到 host arena，放置失效、logits 永远在 host。

## 2. carry 拆分：cross-1 vs cross-N

**分析（verify 期，动态）。** 对每个 captured 跨层张量，算
`distance = 最大消费层 - 生产层`（用 `collect_layer_nodes` 建的层归属）。

- `distance == 1` → **cross-1**：流水线张量（第 L 层输出被 L+1 消费）。任一时刻存活的
  数量有界 → 可复用。
- `distance > 1` → **cross-N**：跨多层存活 → 必须保留。

**实测（ub=1）：三模型全是 100% cross-1，cross-N = 0：**

| model    | cross-1                          | cross-N |
| :------- | :------------------------------- | :------ |
| olmoe    | 15 节点 / 122880 B（max 8192）   | 0       |
| gemma    | 29 节点 / 326656 B（max 11264）  | 0       |
| deepseek | 84 节点 / 5505024 B（max 65536） | 0       |

**布局。** `carry1` 按**层边界索引双缓冲**：第 L 层产出、被 L+1 消费的那组 cross-1 张量
打包进 buffer `L % 2`（两个 buffer，使 L+1 的 tail 处 `O_L` 与 `O_{L+1}` 共存——那里既读
残差又写下一层输出）。`carry1` 大小 = `2 × max over 边界 pack(边界集合)`。`carryN` 打包
一次并保留（当前三模型为空，保留通用性）。

ub512 投影：olmoe 62 MB → ~0.3 MB，gemma 160 MB → ~11 MB，deepseek 2630 MB → ~67 MB。

## 3. 每设备规划（2026-09-14 定稿）

`layout_arena` 改为每设备规划器。每设备一份 `region_plan_t`。区域按**生命周期**分，
不按阶段分：

```cpp
struct region_plan_t {
    std::string dev;                     // "" = host
    ggml_backend_buffer_t buf;           // 每设备一个 grow-only
    // carry：跨层流水（逻辑一份；物理按设备驻留）
    size_t carry1_bytes;                 // 2 × max 边界 pack（层奇偶双缓冲）
    size_t carryN_bytes;                 // 保留的 cross-N
    // scratch：该设备的全部层内临时量，合并：
    //   compact（dense head/tail）+ closure（MoE）
    size_t scratch_bytes;                // max over layers
    // xfer：跨设备生产者的消费者侧副本，每个 (生产者, 设备, stage) 一个 shell；
    //   跨层复用（取各层 max）
    size_t xfer_bytes;
    size_t off_carryN, off_scratch, off_xfer;
    std::unordered_map<const ggml_tensor*, size_t> carry1_off, carryN_off, scratch_off, xfer_off;
    std::set<const ggml_tensor*> cross_device;   // carry 被远端设备消费
};
```

**三个区域，按生命周期。**

- **carry**（跨层）：残差流 / 层输出。**逻辑上一份流水**；物理上每个设备各自
  grow-only 留好位置。当 layer L 的 carry 被 layer L+1（在别的设备）消费时，执行器在
  L+1 首次读之前**显式搬运**（D2D / transfer-queue；host 边界走 staging）——不隐式、
  不多份复制。CPU-only = 一个设备 = 零搬运。
- **scratch**（层内）：`compact`（dense head/tail）**与 `closure`（MoE 计算）合并**。
  两者都是层内临时量、跨层复用，所以在同一设备上就是一个池。执行器的 closure 偏移
  （`moe_chain_fullalloc_buffer`）改为索引这个合并池，`closure` 不再是独立区域。

**closure 天然 per-device。** MoE 计算在专家所在的设备上跑，而每个设备都有自己的专家
池，所以 closure 本来就是 per-device 的——不存在"closure 放哪个设备"的选择。

**节点 → 设备：**

- **C1 dense**（attention / norm / dense MLP / gating）→ 它的 `--dense-placement`
  设备。C1 可以跨设备分拆、可以搬迁（`DENSE_PLACEMENT.md`）。
- **C2 输出头**（`result_norm` / `result_output`，即 logits）→ `C2:<dev>`，整体
  （不分割）。
- **MoE closure** → 该层专家池所在设备。
- **carry** → 产生层的设备（跨边界按上面显式搬）。
- 默认（无 placement）：一个 host plan。

**设备解析（已落地）。** 被捕获节点的设备 = 拥有它权重操作数的设备（dense 层 → 其
placement 设备，取权重 buffer 的设备；专家 → 其池设备）；无权重节点从 producer 继承。
route B 自己的 host backend 和 CPU 设备都坍缩成 host plan。专家池在 `route_b_setup`
时记录。

**跨设备搬运（通用机制，2026-09-15 落地）。** 每条生产者/消费者设备不同的已捕获边，都在
**消费者设备的 `xfer` 区**分配一个**消费者侧副本**（shell），并把消费者的 `src` 改指它。
执行器按 **stage** 各跑一次 `ggml_backend_tensor_copy(src, dst)`：
`ROUTE_B_XFER_LAYER_FRONT`（跨层 carry，消费者层 head 之前）、`ROUTE_B_XFER_CLOSURE`
（如 `cur`，MoE 闭包之前）、`ROUTE_B_XFER_TAIL`（如 `moe_out`，dense tail 之前）。每个
`(生产者, 消费者设备, stage)` 只一个 shell，所以生产者在一层内被写多次也只**搬一次**，
不是每次写都跨设备。这是 D2D / transfer-queue，除 host 边界外绝不 host staging
（`ROUTE_B_LAYER_OWNERSHIP.md` §3.5）。carry、`cur`（C1 head → 专家池）、`moe_out`
（专家池 → C1 tail）统一走它。路由 **`ids`** 无条件读到 host（compact-ids 构建与 round
规划都在 host），不需要 shell。

## 4. 复用已有打包

把 `moe_chain_verify_graph` 里现有的 best-fit decreasing 区间打包（闭包结果布局，
`route_b_chain.cpp`）抽成 `pack(nodes, last_use) -> { offsets, size }`。合并后的
`scratch`（`compact` ∪ `closure`）、`carryN`、每个 `carry1` 边界集合都复用它。不写
第二个分配器。

## 5. C1/C2 放置语义

`--dense-placement C1:<dev>,C2:<dev>`（已解析）是 C1/C2 放置的唯一事实来源。设备的
arena 用该设备的 buffer type；被 own 的 C1/C2 节点从每设备 arena 取 `buffer`/`data`，
而不是固定 host arena。**logits 跟随 C2。**

分两层：

- **规划/分配层**（本文 phase 1-2）：现在就能落；CPU-only 时每设备坍缩成一个，数值不变。
- **执行层**：真在放置设备上跑 C1/C2 需要设备执行器（`run_dense_subgraph` 现在固定调 CPU
  backend）。即 `M2_DEVICE_EXECUTOR`（phase 3）。

## 6. 实施顺序

1. **carry 双缓冲 + 抽 `pack`。** 无设备依赖；立竿见影省内存；验证 olmoe/gemma/deepseek
   数值 + `run_baseline`。
2. **每设备 plan 结构。** 节点 → (设备, 区域, offset)、每设备大小、跨设备标记。CPU-only
   坍缩成一个设备（数值不变）。
3. **设备执行器。** 在放置设备上执行 C1/C2/闭包（跨设备用 D2D / transfer-queue）。独立大工程。

## 7. 状态（2026-09-14）

**Phase 1 —— carry 拆分：已落地。** `layout_arena` 把 carry 区拆开：cross-1 按层奇偶
双缓冲（`2 × max 边界`），cross-N 保留。闭包的 best-fit 区间打包抽成
`pack_interval(nodes, start, end)` 并复用于 `carryN`；`layout_arena` 现在在闭包结果布局
**之后**运行，用上打包后的闭包大小。

ubatch 512 实测（生产构建）：deepseek carry 2630 MB → ~67 MB，olmoe 62 MB → ~0.25 MB，
gemma 160 MB → ~11 MB。闭包块也降（olmoe ub1 278528 → 131072 B）。验证：生产
`run_baseline` PASS；olmoe / gemma / deepseek 整层 OK。

**compact 打包：不是数学 bug —— 它破坏的是 `--prefill-from` 的导出捕获。** 下面的
导出捕获修复落地后，打包后的 scratch 已成为**所有构建的生产布局**（原先"生产
`StreamMoE_dump` opt-in"的计划被取代，`AGENTS.md` 15）：`STREAM_MOE_TMP_COMPACT_PACK=0`
强制字节求和布局，仅用于 A/B 调试。

实测（gemma v2、129-token prefill-from、8 GB 池，对 `moe_129_8192_vk`）：

| compact 布局     | 导出的 embd cos（token 0） | top-4 logits |
| :--------------- | -------------------------: | :----------- |
| 字节求和（默认） |                    0.98631 | 完全一致     |
| 区间打包（`=1`） |                    0.01568 | 完全一致     |

区间打包的 "FAIL" 是**导出假象，不是推理 bug**：

- **生成不受影响。** `llama-cli` 在 pack 开 / 关两种下逐 token 输出相同（temp 0，
  都是同样的 "Paris."）。
- **模型输出完全一致。** 导出的 top-4 logits + logsumexp 在 pack / sum 之间
  **逐位相同**（0/129 id 不匹配，maxLogitDiff = 0，maxLseDiff = 0）；KV cache 也完全
  相同（base + swa，0 diff）。
- **只有导出的 `embd` / `hidden` 不同。** `embd` 是 `t_embd`（输入 token embedding），
  由 token id 决定、本应完全相同。在整层拥有下它是 layer-0 的一个 arena 节点，compact
  打包把它的槽复用给了更晚的 L0 节点：`embd` 与 `norm-0` 共享 `off=0` / 同一个
  `data=` 指针（见 `[cpack]` / `[stage]`），而字节求和给每个节点唯一槽。导出在 route B
  **复用之后**才读该槽，于是抓到的是覆盖者的值。

所以 compact 打包对执行是正确的；**prefill 导出捕获与 arena 槽复用不兼容**。此前
"打包是坏的"结论作废。（更早的 "RelWithDebInfo sum == pack，0 diff" 也是假象：所有
dbg 运行都开着 `STREAM_MOE_TMP_COMPACT_PACK`，其"sum"跑其实就是 pack 跑。）

**读到陈旧值的根因（2026-09-14 定位）。** 装了 `cb_eval` 后，sched 不再整段算一个 split，
而是**逐节点**走（`ggml-backend.cpp:1747`）——因为导出回调对所有节点都返回 `true`。
整层拥有下每个 1-node view 都让 route B 把它所在整层跑一遍（`exec_state` 是 per
graph_compute call 的），所以导出模式慢 ~8x（实测 prompt 4.2 → 0.5 t/s），且捕获发生在
整层已复用槽之后。

**修复 —— 已落地（2026-09-14）。**

1. 导出回调只对它真正读取的张量（`embd` / `hidden` / `logits` / 路由 `MUL_MAT_ID` 的
   ids）返回 `true`，于是 sched 只在那些点切块、route B 基本每层跑一次。导出开销回到
   ~1.4x（prompt 4.3 → 3.3 t/s；剩余来自每层的 MoE 切点）。
2. 这些张量在层捕获前声明给 route B（`route_b_set_export_retained`），`layout_arena`
   把它们移出复用池（并入保留区 `carryN`）。于是 split 之后的读能看到算出来的值。
   expert 历史的 ids 也需要（它最后一层的 ids 会被 tail 复用）。

门控：回调挂 `STREAM_MOE_PREFILL_EXPORT`，保留 API 挂
`STREAM_MOE_ROUTE_B && STREAM_MOE_PREFILL_EXPORT`，用不到的地方不编译。

验证（gemma v2、129-token prefill-from、pack 默认开）：导出的 `embd` / `hidden` / KV
与 expert history 在 pack vs sum 下 **IDENTICAL**；top-4 logits 仍逐位相同；pack 对
`moe_129_8192_vk` gate 通过；生产 `run_baseline`（`StreamMoE_dump`，pack 默认开）仍 PASS。

**复现。** `build.bat llamalibs StreamMoE_latest` 后跑 129-token prefill-from：不带 env
走打包路径、`=0` 走字节求和；两者现在产物完全一致。要看 `[cpack]` offset 计划 /
`[stage]` 节点值用 `StreamMoE_dump_dbg`
（+ `STREAM_MOE_TMP_STAGE_DUMP=1 STREAM_MOE_TMP_COMPACT_DEBUG=1`）。

**每设备规划：已落地（2026-09-14，commit `fdd772e`）。** `layout_arena` 现在每设备
一份 `region_plan_t`，各自一个 grow-only buffer：

- 两个区域按生命周期：`carry`（跨层流水）+ `scratch`（层内：dense head/tail 与 MoE
  closure 合并）。
- `carry1` 每设备（parity 双缓冲）、`carryN` 每设备（retained）。`cross_device` 标记
  消费者在别的设备的 carry 张量；执行器在 consumer 首次读之前搬运
  （`ggml_backend_tensor_copy`，`M2_DEVICE_EXECUTOR.md` §7.8）。carry1 和 carryN
  在**每个边界一起搬**（relay：第 L 层时的副本永远在 dev(L)，所以任何设备上的
  consumer 读到的都是对的那份）。
- closure 的 `ex.out_off` / `result_bytes` 改为索引合并后的 scratch；
  `moe_chain_fullalloc_buffer` 改为按设备（专家池设备）查，返回的正是 dense
  head/tail 用的同一个池。`route_b_in_arena` 检查所有设备 buffer。
- MoE closure 的设备 = 专家池设备（`route_b_closure_device`，在 `route_b_setup`
  记录）——closure 在专家所在的设备上跑。专家权重不在普通 buffer 里，所以 `dev_of`
  无法从权重操作数解析它。

实测（gemma v2，129-token prefill-from，ub 129）：decode build carry1 88 KB /
scratch 8 MB；prefill build carry1 1.4 MB / scratch 128 MB。（§1 的 ub-512 字节和
表里 compact 是 2.5 GB；合并+打包后的 scratch 小得多。）

验证（CPU-only 坍缩成 1 个 host plan，数值不变）：pack vs sum `IDENTICAL`
（embd / hidden / KV + 专家历史）；vk gate 121/129（93.8%）；生产
`run_baseline`（`StreamMoE_dump`）PASS。

**跨设备 carry relay：已落地（2026-09-14，commit `0ef71df`）。** carry 张量是
producer 节点的输出，所以留在 producer 设备上。当它的（最大）consumer 在别的设备
时，`layout_arena` 分配一份本地副本——在 consumer 层的 carry1/carryN 边界集合里预留
一个 shell 张量（与 carry 共用打包和生命周期）——并把 consumer 的 `src` 改指它。
`exec_layer_burst` 在 consumer 层最前面、head 读之前执行
`ggml_backend_tensor_copy(src, dst)`（同步；`M2_DEVICE_EXECUTOR.md` §7.8 的 transport）。
carry1 和 carryN 在每个边界一起搬。

单设备 / CPU-only：没有跨设备 carry → 没有 relay → 验证 `IDENTICAL`（pack vs sum +
专家历史）。

**phase 3（设备执行器）：已落地（2026-09-15）。** `route_b_setup` 记录
`设备名 → ggml_backend_t` 表；`layout_arena` 暴露 `节点 → 设备`；`exec_layer_burst`
把 dense head/tail 按 placement 设备分组、各在自己的 backend 上跑
（`run_dense_subgraph`，host → CPU）。`cur` 上传与 `moe_out` 回写都改为 backend 无关，
设备驻留的激活值绝不在 host 上解引用。carry 专用的 relay 已被上面的通用 stage 搬运
取代：carry、`cur`、`moe_out` 统一走 `xfer` 区。

仍待做：C1 **跨设备拆分**（同层 head → head 的边需要比 `LAYER_FRONT` 更细的 stage）；
设备侧验证（C1/C2 上 Vulkan）。注意：改指 `src` 发生在 scheduler 的 tensor-backend
pass 之后——route B 整层拥有时是安全的，但所有权模型若变必须重新检查。

## 8. 已定结论（2026-09-15）

1. **`carry1` 保持两个 buffer。** 放不进两个 parity buffer 的边界集合本就不是 cross-1，
   必须划归 **cross-N**（`carryN`）。cross-1/cross-N 的划分就是保证——不加第三个 buffer。
2. **`carry1` 边界集合：每个 parity 一个共享集合，按该 parity 各层的最大边界定容**
   （即当前布局）。不需要逐边界打包——总量很小（deepseek ub 512 约 67 MB）。
3. **C1/C2 放置严格尊重参数。** C1 与 C2 不同设备时，末层输出 → C2 必然搬运；不要求
   C2 == C1。每条跨设备边都插一个带 stage 的搬运（见 §3）：carry、`cur`、`moe_out`；
   路由 `ids` 读到 host（无 shell）。
4. **每设备一份 plan，按该设备实际拥有的节点定容。** 设备只为它有的节点（C1 / MoE closure /
   C2）预留区域；plan 由节点存在性导出，不是固定模板。
5. **每层设备 backend 注册表（phase 3）。** `route_b_setup` 记录一张
   `设备名 → ggml_backend_t` 表（与专家池设备记录放在一起）；执行器按 plan 的设备解析
   每个 head/tail/closure 的 backend。`run_dense_subgraph` 与 `moe_exec_mul_mat_id` 收
   解析出的 backend，而不是固定 CPU。
