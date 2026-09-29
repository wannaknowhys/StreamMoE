[English](./DEVICE_SEGMENT_EXEC.md) | [简体中文](./DEVICE_SEGMENT_EXEC.zh-CN.md)

# 设备分段执行（路径 B）

## 问题

xfer 阶段机制（`run_layer_xfers`）假设同层 producer 在 `LAYER_FRONT` xfer
执行之前已经就绪。当 C2 ≠ C1 时，非层节点（embd、output_norm、output）通过回退
传播分配到相邻层，放置在 C2 设备上。xfer 在它们被计算**之前**就复制了输出，读到
了未初始化的设备内存 → 推理结果全乱。

## 设计

用一个**拓扑分段执行器**替代 `run_layer_xfers` + `run_dense_by_device`，在拓扑
序中交替执行设备段和跨设备拷贝。三段结构保持不变：

```
exec_layer_burst(layer):
  1. topo_segment_execute(dense_head)
  2. IDs 回读（硬同步）+ 专家 pin + 桶执行器（不变）
  3. topo_segment_execute(dense_tail)
```

### topo_segment_execute(nodes)

输入：拓扑序的张量列表，每个张量有已知的物理设备（`route_b_node_device`）。

算法：

1. 顺序遍历列表。跟踪"当前设备"。
2. 将连续同设备节点累积到当前**段**。
3. 设备切换时（或列表结束时）：
   a. **刷新**：用当前段构建 mini-graph，调用
   `ggml_backend_graph_compute(real_backend, mini_graph)`。
   b. **拷贝**：对下一段的每个节点，如果其任何 `src` 张量在*之前*的设备上
   产生且在当前设备上有 relay shell，调用
   `ggml_backend_tensor_copy(src, shell)`。
   c. 在新设备上开始新段。

这保证拷贝发生在 producer 段**之后**、consumer 段**之前** — 顺序由构造保证
正确。

### layout_arena 的变更

relay shell（`g_relays`）仍然创建：它们是消费者设备上 arena 分配的张量，接收跨
设备拷贝。变更点：

- **删除阶段标记**（`ROUTE_B_XFER_LAYER_FRONT`、`ROUTE_B_XFER_CLOSURE`、
  `ROUTE_B_XFER_TAIL`）。relay 执行不再由阶段驱动，而由段边界驱动。
- shell 分配和 src→shell 重连在 `layout_arena` 中保持不变。

### 非层节点

`embd`、`norm`（output_norm）、`result_norm`、`result_output` 通过
`collect_layer_nodes` 的回退传播分配到层。它们通过 `build_layer_plans` 进入层的
`head` 或 `tail` 列表。拓扑分段执行器自然处理它们：如果 `embd` 在 Vulkan0（C2）
而 head 的其余部分在 CPU（C1），执行器为 `embd` 创建一个 Vulkan0 段，将结果拷贝
到 CPU，然后执行 CPU 段。

### IDs 回读

路由 IDs 回读是 head 和闭包之间的硬同步点。host 必须检查 IDs 以决定要 pin 哪些
专家。闭包（桶执行器）走自己的专用路径 — 它 stage 叶子、运行逐桶链、写入
`moe_out`。这部分不变。

### 不变的部分

- Arena 布局：carry1/carryN/scratch BFD 打包、双缓冲流水线
- 每设备 plan buffer（host plan + 可选设备 plan）
- `layout_arena` 中的 relay shell 创建和 src 重连
- MoE 闭包的桶执行器
- `run_dense_subgraph`：仍然是在真实后端上执行 mini-graph 的叶函数

### 删除的部分

- `run_layer_xfers` 函数
- 阶段常量（`ROUTE_B_XFER_LAYER_FRONT`、`ROUTE_B_XFER_CLOSURE`、
  `ROUTE_B_XFER_TAIL`）
- `run_dense_by_device` 函数（被 `topo_segment_execute` 替代）
- `exec_layer_burst` 中的三个 `run_layer_xfers` 调用点

### 新增部分

- `topo_segment_execute(ctx, cpu_backend, nodes, layer, stage_name)`：上述段
  边界执行器。

## 不变量

1. **保持拓扑序**：`build_layer_plans` 的节点列表已按图拓扑序排列。段执行器尊重
   此顺序。
2. **拷贝先于消费**：跨设备拷贝只在 producer 段刷新之后、consumer 段开始之前
   执行。
3. **单设备无回归**：当 C1 == C2（或全 RAM）时，没有设备切换 → 单段、零拷贝。
   与当前路径完全一致。
4. **GPU 终态兼容**：每个设备得到一个连续图段，拷贝是显式数据搬运操作 — 这直接
   映射到真实 GPU 路径上的每 GPU 命令队列 + DMA 传输。
