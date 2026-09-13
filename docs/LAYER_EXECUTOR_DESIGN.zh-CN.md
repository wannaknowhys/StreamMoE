# Route B 层执行器设计（整层所有权）

[English](LAYER_EXECUTOR_DESIGN.md) | [简体中文](LAYER_EXECUTOR_DESIGN.zh-CN.md)

> 状态：**提案，2026-09-13**。取代临时的 debug 整层路径（`STREAM_MOE_TEMP`）。
> 相关：`ROUTE_B_LAYER_OWNERSHIP.md`（目标形态）、`M2_DEVICE_EXECUTOR.md`
> （每设备执行器 + arena）、`L2_WHOLE_LAYER_REVIEW.md`（bug 清单 + 已定位发散）、
> `DENSE_PLACEMENT.md`（C1/C2）。

## 1. 问题

debug 整层路径有两个拦路虎，根因相同：route B **通过 scheduler 认领节点**，而不是
**拥有层、在内部执行它**。

1. **DeepSeek OOM / 崩溃**：把每层节点都划给我们的 backend，scheduler 会为整图预留
   一整块 `STREAMMOE_HOST` compute buffer（约 97 GiB）；叠上专家池超过内存 ->
   分配失败（`--moe-ram-pool 81920`）或 `0xC0000005`（`8192`）。
2. **olmoe 发散**：最后一层含 `inp_out_ids` 输出 token 收窄（`get_rows`），它被捕获进
   层里并被 `run_dense_nodes` 执行，破坏输出。bisect：L0–L14 单独接管与基线一致，
   L15 单独接管发散；排除 L15 即恢复正常。

## 2. 目标 / 非目标

目标：route B 拥有每一层；scheduler 只看到模型 I/O；每层在自己的 arena 里内部执行；
最后一层的收窄被显式处理。

非目标：C2（`token_embd` / `output`）的放置策略；自己写 dense 内核（dense 委托给
设备/CPU backend）。

## 3. 架构

- 我们的 backend 拥有整图 -> **一个 split**。
- `graph_compute` 按**逐层**方式走图，由构建期算好的 `LayerPlan` 驱动。
- 每层：`dense head -> MoE burst -> dense tail`，然后该层完成。
- 层的激活住在 **route B 自己的 arena**，不在 scheduler 的 compute buffer 里。
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
channel**（见 4.4），不靠字符串匹配。

### 4.2 每层 arena + 预分配 buffer（解 DeepSeek）

scheduler 把**已经有 buffer** 的 tensor 当作 pre-allocated（`ggml-backend.cpp:911`
把它归给该 buffer 的 backend；`:927` 绝不为它分配）。因此：

- route B 拥有一个**每层 arena**（一个 `ggml_backend_buffer`，其 buft 注册到我们
  的 backend）；
- 调度前，route B 把每层节点的 `buffer`/`data` 指向该 arena；
- scheduler 把这些 tensor 归给我们，且**不在 compute buffer 里为它们分配** ->
  compute buffer 保持有界（只剩模型 I/O）；
- `graph_compute` 原地执行它们。

待定：arena 大小（层的生命周期）、单 arena vs 单个 grow-only arena、设备变体
（device-local arena，复用 `M2_DEVICE_EXECUTOR`）。

### 4.3 不 clone 的 dense 执行（解 olmoe）

执行**原始节点**（`nd`），而不是 `ggml_dup` 出来、再手工填 `data`/`nb`/`view` 的
clone。因为 route B 拥有整图（一个 split），scheduler 从不执行我们的节点；我们自己
跑，所以没有重复执行。用 `LayerExecutionState {NOT_STARTED, RUNNING, COMPLETE}`
保证一次且仅一次，并在重复并发执行时 assert。

这去掉了那套破坏最后一层归约的第二执行语义。

### 4.4 显式层边界（side channel）

`llm_build_context` 构建时知道层号和 MoE 段。每层注册
`layer_span_t {begin, moe_begin, moe_end, end}` 到一个 side channel。由此可删掉
`name_layer_suffix`、`last_suffixed`、`moe_chain_layer_of_node`、`ffn_moe_out`
字符串匹配、以及 `has_first`/`first_node` 的 split 启发式（评审项 A1/A2/A5/E1/E2/E4）。

### 4.5 op 三分类

三类（吸收评审项 E3）：

- `PURE_ALIAS`：`VIEW / RESHAPE / TRANSPOSE / PERMUTE` —— 不 compute，须保证
  data/layout 正确。
- `MATERIALIZING`：`CONT / DUP / CAST` —— 真拷贝，必须执行。
- `COMPUTE`：其余。

### 4.6 最后一层输出 token 收窄（显式）

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

这就是让最后一层整层所有权正确的关键。

### 4.7 tracer 分离

把 canary / dump / `fprintf` 从 `run_dense_nodes` / `exec_layer_burst` 里抽出，做成
可注入的 tracer（默认 no-op）。热路径保持干净，去掉长寿的 debug 全局量。

### 4.8 分配失败干净报错

arena / buffer 分配失败时返回清晰的 `GGML_STATUS_FAILED` + 可操作信息（绝不裸崩
`0xC0000005`）。

## 5. D 类问题（必须专门设计，不会自动消失）

| D 问题 | 本设计如何应对 |
|---|---|
| 整图单 backend compute buffer | 每层 arena 预分配 buffer（4.2）-> scheduler compute buffer 保持有界 |
| 最后一层 `inp_out_ids` 语义 | 显式收窄支持（4.6） |
| `supports_buft` / `supports_op` 与 ownership | route B 声明 arena buft + expert buft；不无条件接受 |
| dense 执行顺序 / data 就绪 | route B 内部控制执行顺序；`LayerExecutionState` + assert |
| debug 脚手架焊在热路径 | tracer 分离（4.7） |
| 分配失败崩溃 | 干净错误路径（4.8） |

## 6. 里程碑

- **R1** LayerPlan + 显式边界（行为不变；保持 MoE-only）。
- **R2** 不 clone 的 dense 执行 + `LayerExecutionState` -> 解 olmoe。
- **R3** 每层预分配 arena -> 解 DeepSeek。
- **R4** 最后一层收窄显式支持。
- **R5** op 三分类 + tracer 分离。

每个里程碑都保持生产（MoE-only）路径数值 **IDENTICAL**。

## 7. 验证门

- 生产纯 RAM **IDENTICAL**（gemma、olmoe），走 `baseline_regression`。
- 整层 olmoe 与基线一致（输出正常，hidden cos ~1.0）。
- DeepSeek 整层能加载并通过 cos 门（`baseline/deepseek_hi_up`）。
- 打印的 scheduler compute-buffer 大小保持有界（只剩模型 I/O）。

## 8. 待定问题

1. arena 生命周期/大小：每层一个 arena vs 单个 grow-only arena。
2. 设备侧 arena（DMA）用于 GPU 阶段 —— 复用 `M2_DEVICE_EXECUTOR`。
3. 大量 tensor 被预分配后 scheduler split 的行为。
4. 多设备：每设备 arena + ids D2H join。
