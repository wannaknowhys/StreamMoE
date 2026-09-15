# Route B 层执行器设计（整层所有权）

[English](LAYER_EXECUTOR_DESIGN.md) | [简体中文](LAYER_EXECUTOR_DESIGN.zh-CN.md)

> 状态：**提案，2026-09-13**。取代临时的 debug 整层路径（`STREAM_MOE_TEMP`）。
> 相关：`ROUTE_B_LAYER_OWNERSHIP.md`（目标形态）、`M2_DEVICE_EXECUTOR.md`
> （每设备执行器 + arena）、`L2_WHOLE_LAYER_REVIEW.md`（bug 清单 + 已定位发散）、
> `DENSE_PLACEMENT.md`（C1/C2）。

## 1. 问题

debug 整层路径有三个拦路虎，根因相同：route B **通过 scheduler 认领节点**，而不是
**拥有层、在内部执行它**。

1. **DeepSeek OOM / 崩溃**：把每层节点都划给我们的 backend，scheduler 会为整图预留
   一整块 `STREAMMOE_HOST` compute buffer（约 97 GiB，且不做跨层复用）；叠上专家池
   超过内存 -> 分配失败（`--moe-ram-pool 81920`）或 `0xC0000005`（`8192`）。
2. **olmoe 发散**：最后一层含 `inp_out_ids` 输出 token 收窄（`get_rows`），它被捕获
   进层里并被 `run_dense_nodes` 执行，破坏输出。bisect：L0–L14 单独接管与基线一致，
   L15 单独接管发散；排除 L15 即恢复正常。
3. **flash attention 被关**：整层模式把 `FLASH_ATTN_EXT` 变成手动 `kq/kqv`，落地
   `O(B^2)` 的 score 矩阵（见 4.9）。

## 2. 目标 / 非目标

目标：route B 拥有每一层（后续阶段还包括 C2）；scheduler 只看到模型 I/O；每层用
**一块可复用的 arena** 内部执行；最后一层的收窄被显式处理；**保住 flash attention**。

非目标：自己写 dense 内核（dense 委托给设备/CPU backend）；改变 C1/C2 的放置语义。

## 3. 架构

- 我们的 backend 拥有整图 -> **一个 split**。
- `graph_compute` 按**逐层**方式走图，由构建期算好的 `LayerPlan` 驱动。
- 每层：`dense head -> MoE burst -> dense tail`，然后该层完成。
- **每设备一块 arena，按 worst-case（最大）一层 sizing，跨层复用**。下文说的
  "per-layer" 指**布局分析**（offset），不是每层各一块承载内存。
- 数据流：KV cache 常驻（归 llama，不在我们 arena）；我们的 arena 只放**当层临时
  激活**；跨层只传隐藏态 `X` 和残差。所以按最坏一层 sizing 就够。
- 为什么 debug 路径是 ~97 GiB：`ggml_backend_sched` 按**整图**算它的 compute buffer
  且不跨层复用。**我们不依赖 gallocr**：自己的 arena + 预分配 buffer（4.2）把内存
  限制在一层 worst-case。
- scheduler 的 compute buffer 只覆盖模型 I/O（`embd`、`logits`）。

## 4. 机制

### 4.1 LayerPlan（构建期）

用构建期一次算好的结构，取代运行期启发式（`collect_layer_nodes` + 名字后缀 +
`strstr("ffn_moe_out")` + `down[]` 传播 + `has_first`/`first_node`）：

```cpp
struct moe_layer_plan {
    int32_t layer;
    std::vector<ggml_tensor*> all;
    std::vector<ggml_tensor*> head;   // MoE 输入之前的 dense
    std::vector<ggml_tensor*> moe;    // MoE 闭包
    std::vector<ggml_tensor*> tail;   // MoE 输出之后的 dense
    ggml_tensor* input;               // 层输入
    ggml_tensor* moe_out;             // MoE 锚点（ffn_moe_out）
    ggml_tensor* output;              // 层输出
    bool host_owned;                  // dense 权重是否 host 常驻
};
```

事实来源：由 `llm_build_context` 在构建每层 / 每个 MoE 子图时注册的**显式 side
channel**（见 4.5），不靠字符串匹配。

### 4.2 预分配 buffer（解 DeepSeek）

scheduler 把**已经有 buffer** 的 tensor 当作 pre-allocated（`ggml-backend.cpp:911`
把它归给该 buffer 的 backend；`:927` 绝不为它分配）。

**原型结果**（`temp/proto_prealloc.cpp`，CPU backend，32 个 256 KiB 同时存活中间量）：
控制组 compute buffer 8,912,896 B -> 预分配后 786,432 B，且 32/32 preset `data`
指针在 `ggml_backend_sched_alloc_graph` 后全部保留。机制成立。

计划：

- 每设备一块 **arena** `ggml_backend_buffer`（buft 注册到我们的 backend）；
- 调度前，把每层节点的 `buffer`/`data` 指向该 arena；
- scheduler 把这些 tensor 归给我们，且不为它们分配；
- `graph_compute` 原地执行它们。

**CPU 注意**：现有闭包路径只设 `data`（`minigraph_exec.cpp:730`），没设 `buffer`。
两个都要设，否则 scheduler 仍会分配。设备路径本来就设了 `buffer`（`:724`）。

### 4.3 arena：三段（carry / compact / closure）

每设备一块 arena buffer，**一次 sizing、永不增长**（增长会让已设的 `data` 指针失效），
切成三个固定子区：

```
[ carry region（最前，固定 base）][ compact region ][ closure block ]
```

- **carry region（最前）**：所有 live range **跨层**（无论跨几层）的 tensor——残差流 /
  层输出，以及边界变换（如最后一层的 `inp_out_ids` 收窄）。地址固定、永不复用。**放最前**
  是为了 compact/closure 大小变化时**不会移动 carry 地址**（carry 指针跨层存活）。
- **compact region**：层内临时量，每层各 pack、**跨层复用**；大小 = 各层 max。每层地址相同。
- **closure block**：现有 MoE 闭包布局（`ex.out_off` / `ex.result_bytes`）+ bump，跨层复用；
  大小 = 各层 max。内部 per-bucket scratch 在多桶间**复用**，累加器**独立一个地址**
  （每桶算完累加进去）。

全局（整图）liveness **只用来分类** carry vs 层内，不用来全局分配地址。任何跨层 tensor
都进 carry region：若留在 compact region，下一层会在它被消费前覆盖它（顺序脆弱，否决）。

**复用现有分配函数**：把 `moe_chain_verify_graph` 里的 best-fit interval packing
（`route_b_chain.cpp:282-370`）抽成 `pack(nodes, last_use) -> offsets, size`，闭包段和
compact 段都调它。不写第二份分配器。

闭包独立的原因（执行特殊）：

- bucket 引擎每个 bucket/round 重建 mini-graph（临时 tensor，走 bump）；
- `ffn_moe_out` / fold 是专门输出；
- 匿名 per-topk add（`node_NNN ADD`）是闭包节点。

**异构层**：每层的 plan 和 packing 各自算；只有 region 大小共享（max / peak）。边界规则：

- 纯 dense 层（无 MoE）：closure 空，整层都是 dense；
- 无 `ffn_moe_out`：用闭包最后一个节点作锚点；
- 不同 attention 类型（滑动窗 / SSM）：head/tail split 只依赖闭包锚点，与 attention op 无关；
- 特殊层号：官方层号 channel 处理；
- 一层多个 MoE 子图：当前每层一个 `ex`；需要时把 plan 扩成列表（待定）。

**动机实例**（consumer 校验，R1）：最后一层的 `inp_out_ids` `get_rows(inpSA, ...)` 读上一层
输出、被最后一层消费；生产者传播把它归到上一层。它是 carry（跨 1 层），必须活到最后一层消费。

### 4.4 不 clone 的 dense 执行（解 olmoe）

执行**原始节点**（`nd`），而不是 `ggml_dup` 出来、再手工填 `data`/`nb`/`view` 的
clone。因为 route B 拥有整图（一个 split），scheduler 从不执行我们的节点；我们自己
跑，没有重复执行。用 `LayerExecutionState {NOT_STARTED, RUNNING, COMPLETE}` 保证
一次且仅一次，并在重复并发执行时 assert。

### 4.5 显式层边界（side channel）

`llm_build_context` 构建时知道层号和 MoE 段。每层注册
`layer_span_t {begin, moe_begin, moe_end, end}` 到 side channel。由此删掉
`name_layer_suffix`、`last_suffixed`、`moe_chain_layer_of_node`、`ffn_moe_out`
字符串匹配、以及 `has_first`/`first_node` 启发式（评审项 A1/A2/A5/E1/E2/E4）。

### 4.6 op 三分类

三类（吸收评审项 E3）：

- `PURE_ALIAS`：`VIEW / RESHAPE / TRANSPOSE / PERMUTE` —— 不 compute，须保证
  data/layout 正确。
- `MATERIALIZING`：`CONT / DUP / CAST` —— 真拷贝，必须执行。
- `COMPUTE`：其余。

### 4.7 最后一层输出 token 收窄（显式）

最后一层含（如 `olmoe.cpp`）：

```cpp
if (il == n_layer - 1 && inp_out_ids) {
    cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
```

这些 `GET_ROWS` 把 token 维收窄到 `n_outputs`。执行器**显式支持**：

- 收窄节点是 `COMPUTE`（不是 alias），在 head 里执行；
- 从收窄往后，该层用 `n_tokens = n_outputs` 跑（`ffn_inp`、MoE pin / mix-plan /
  arena、tail 都用收窄后的计数）；
- 该层的 arena / pin 预算按收窄后的计数算。

### 4.8 tracer 分离 + 分配失败干净报错

把 canary / dump / `fprintf` 从 `run_dense_nodes` / `exec_layer_burst` 抽出，做成可
注入的 tracer（默认 no-op）。arena / buffer 分配失败时返回清晰的
`GGML_STATUS_FAILED` + 可操作信息（绝不裸崩 `0xC0000005`）。

### 4.9 flash attention 必须保留

当前整层路径把 `FLASH_ATTN_EXT` 变成手动 `kq / softmax / kqv`，落地 `O(B^2)` 的
score 矩阵。每层估算（f32，`h` 个头）：

| 模型     | B    | head（linear + attn）    | closure | tail  |
| -------- | ---- | ------------------------ | ------- | ----- |
| gemma    | 2048 | **900 MB (132 + 768)**   | 77 MB   | 60 MB |
| olmoe    | 2048 | **864 MB (96 + 768)**    | 48 MB   | 40 MB |
| deepseek | 2048 | **3264 MB (192 + 3072)** | 96 MB   | 80 MB |

attention 项占大头且随 `B^2` 增长。route B 拥有层但**委托**它不实现的操作；
`FLASH_ATTN_EXT` 必须委托给设备/CPU backend，让 score 矩阵永不落地。这是硬要求
（也是 D 类项）。

### 4.10 C2 接管（后续阶段）

目前 `--dense-placement C2:<dev>` 只设放置 buft（`route_b_dense_device`，
`route_b_inject.cpp:445`）；C2 节点仍由 llama 执行，**没被 route B own**。所以
route-B 层 -> C2 那道缝，同 device 也可能拷。

目标：route B 也拥有 C2 激活（预分配进 arena），使 C1 与 C2 同 device 时无拷贝。
空间在统一 liveness 分析合并前各自独立（两者时间上不重叠）。

## 5. D 类问题（必须专门设计，不会自动消失）

| D 问题                                       | 本设计如何应对                                 |
| -------------------------------------------- | ---------------------------------------------- |
| 整图单 backend compute buffer                | 一块 worst-case 一层 arena、跨层复用（3、4.2） |
| 最后一层 `inp_out_ids` 语义                  | 显式收窄支持（4.7）                            |
| flash attention 必须保留                     | 委托 `FLASH_ATTN_EXT`（4.9）                   |
| C2 缝拷贝                                    | C2 接管（4.10）                                |
| `supports_buft` / `supports_op` 与 ownership | route B 声明 arena buft + expert buft          |
| dense 执行顺序 / data 就绪                   | 内部控制顺序 + `LayerExecutionState` + assert  |
| debug 脚手架焊在热路径                       | tracer 分离（4.8）                             |
| 分配失败崩溃                                 | 干净错误路径（4.8）                            |

## 6. 里程碑

- **R1** LayerPlan + 显式边界（行为不变；保持 MoE-only）。
- **R2** 不 clone 的 dense 执行 + `LayerExecutionState` -> 解 olmoe。
- **R3** 三段预分配 arena（carry 最前 / compact 复用 / closure block）-> 解 DeepSeek。
- **R4** 最后一层收窄显式支持。
- **R5** 保留 flash attention（委托 `FLASH_ATTN_EXT`）。
- **R6** C2 接管（own C2 激活）-> 同 device C1/C2 无拷贝。
- **R7** op 三分类 + tracer 分离。

每个里程碑都保持生产（MoE-only）路径数值 **IDENTICAL**。

**已落地（2026-09-13）：**

- **R1** 官方层号 channel + 构建期 LayerPlan + consumer gate：`6c1c99b`、`011948b`。
- **R2** 不 clone 的 dense 执行 + `LayerExecutionState`。dense head/tail 用**原主图节点**
  手工拼一个 `ggml_cgraph` 执行（不再 `ggml_dup` clone）；节点列表不必在主图里连续
  （MoE closure 与 dense head 交错，`ggml_graph_view` 连续区间不可用）。
  `LayerExecutionState` 按 **graph_compute 调用**重置、不是按 build：llama 跨 decode
  复用已建图（`llama-context.cpp:1379`）且不再跑 `moe_chain_assign_backend`，状态挂在
  构建期 plan 上会一直是 COMPLETE、后续 decode 全跳过。整层路径首见即执行（不再依赖
  "first node 恰好在这个 split"）；MoE-only 路径保留 first-node 触发。
- **R4** 收窄节点重归到消费者层：`cb3e59d` —— **修好了 olmoe 整层发散**（bisect：L0–L14
  正常，L15 单独发散；`get_rows(inpSA, inp_out_ids)` 读上一层输出、被生产者传播归到 L14，
  于是 L14 的 tail 执行了它；重归到 L15 后整层输出与基线一致）。

R2 验证：olmoe 逐层 bisect L0–L15 全部 SAME；整层全开 / `MAX=14` / `MAX=15` 全部 SAME；
gemma 整层 SAME；生产 `StreamMoE_dump` olmoe/gemma 不变。

**R1 归属改造 —— 所有 compute 节点无条件收归（修好 DeepSeek 整层）。** 原来的部分归属
在 producer 是 leaf/输入时会漏掉匿名 compute 节点（如 DeepSeek 的 `-INF` attention mask
FILL）。这些节点留在 scheduler 的 buffer 池里，`ggml_gallocr` 把一块仍被图输入
（`k_idxs`）使用的地址复用给了它，于是 mask FILL 覆盖了 KV 槽索引，末层 `set_rows` 断言。
`collect_layer_nodes` 现在把**每个 compute 节点**都归属：named → 层；producer 传播
（MAX producer 层）；**consumer 传播**（无归属匿名 → consumer 的层）；最近层兜底。
不再有任何 un-owned compute 节点，整图 = 一个 split、所有激活在我们的 arena。这同时把
embedding（层 0 之前）和 C2 输出头（末层之后）折进层 0 / 末层，scheduler 只管叶子
（输入 / KV / 权重）。验证：olmoe 逐层 bisect 全 SAME；gemma SAME；DeepSeek 整层
`-n 24` 对 baseline SAME、池退出干净（0 泄漏）；生产 `run_baseline` PASS。

**已转正。** 整层路径不再受 `STREAM_MOE_TEMP` 门控：`route_b_whole_layer_active()`
恒真、`moe_dev_supports_buft` 接受 host buft、`moe_dev_supports_op` 认领 fused op、
capture/arena/plan 调用无条件执行。调试 dump 仍留在 `#ifdef STREAM_MOE_TEMP`。生产
`run_baseline` 用整层执行器 PASS。

**临时激活空间（arena `need`，host buft）。** 用生产构建（`STREAM_MOE_TMP_DENSE_DEBUG=1`）、
按给定 ubatch 的 prompt prefill（`-ub N`）实测：

| ubatch |   olmoe |   gemma | deepseek |
| -----: | ------: | ------: | -------: |
|      1 | ~1.0 MB | ~5.6 MB |  ~9.8 MB |
|    512 | ~0.5 GB | ~2.8 GB |  ~4.1 GB |
|   2048 | ~2.0 GB |       - |        - |

ubatch 512 分解（carry / compact / closure）：olmoe 62/218/142 MB，
gemma 160/2498/198 MB，deepseek 2630/1300/248 MB。arena 装下每层全部激活：
`carry` = 所有跨层张量（不复用——DeepSeek 的残差流/hyper-connection 占大头），
`compact` = 最坏层的 dense head/tail（gemma 的 dense MLP + C2 lm_head logits 占大头），
`closure` = MoE 闭包块。这是当前保守布局；层内 liveness 打包、以及把 C2 logits 移出
arena 是明显的优化点。

## 7. 验证门

- 生产纯 RAM **IDENTICAL**（gemma、olmoe），走 `baseline_regression`。
- 整层 olmoe 与基线一致（输出正常，hidden cos ~1.0）。
- DeepSeek 整层能加载并通过 cos 门（`baseline/deepseek_hi_up`）。
- 打印的 scheduler compute-buffer 大小保持有界（只剩模型 I/O）。
- **构建期 debug log**：打印每层 head / closure / tail 字节（用于看三模型在指定
  ubatch 下的真实数值）。

## 8. 待定问题

1. arena 生命周期/大小：单个 grow-only arena vs 精确 worst-case 测量。
2. 设备侧 arena（DMA）用于 GPU 阶段 —— 复用 `M2_DEVICE_EXECUTOR`。
3. 多设备：每设备 arena + ids D2H join。
4. C2 接管：只 own C2 激活，还是连权重也进池（复用 C1 池）？
