# 多子池设计：按专家种类分组 (MULTI_SUBPOOL.md)

> 状态：**定案（2026-08-27）**。替代 `heterogeneous_layers` 排除方案（异构层也进池）。
> 相关：`docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7（第三路径：官方 `ggml_mul_mat_id` 内核 + 均匀 stride 槽）。

## 1. 问题

route B 第三路径要求**每专家等大**（均匀 stride 槽：`nb[2]=slot_size`）。但 MoE 模型可能**异构**：不同层用不同量化 → 每专家字节大小不同。

实测 gemma-4-26B-A4B：
- layer 0-28（29 层）：`gate_up Q4_K + down Q5_1`，每专家 **3717120 B**
- layer 29（1 层）：`gate_up Q4_K + down Q8_0`，每专家 **4336640 B**

同质假设下异构层无法进池（旧方案：排除走 mmap）。

## 2. 设计：按专家种类分子池

**专家种类 = 每专家布局（子张量集合 + 量化类型）相同的组**。每种类一个**子池**：

- **子池内**：槽等大（slot_size = 该组专家字节大小），均匀 stride，与现第三路径一致。
- **子池间**：不同 slot_size（各组的专家大小不同），独立管理。

### 2.1 组（expert_group）

```text
moe_model_topology_t:
  std::vector<expert_group_t> groups;
  struct expert_group_t {
      uint32_t               idx;
      std::vector<uint32_t>  layers;     // 本组覆盖的层
      size_t                 expert_size;// 每专家字节（组内均匀）
      uint32_t               n_experts;  // layers.size() * n_expert_per_layer
      size_t                 byte_fraction; // 组专家字节 / 总专家字节
  };
```

组识别：`parse_gguf_topology` 校验时，把每专家总字节相同的层归为一组（不再 throw/排除）。

### 2.2 预算分配（关键：槽数与池大小同时按比例成立）

总池预算 `P`（字节）按**组字节占比**切分给各子池：

```text
P_group = P * (group.byte_fraction)                     // 字节预算
n_slots_group = P_group / group.expert_size             // 槽数
```

由于组内专家等大，`byte_fraction = n_experts_group / n_experts_total`，所以：

- **槽数比例 = 专家数量比例**（`n_slots_group ∝ n_experts_group`）
- **池大小比例 = 字节比例**（`P_group ∝ byte_fraction`）

两者**自然同时成立**，且每种专家都保留"总量中同等比例"的常驻量 → 命中率跨组均衡。

> 局限（首版接受）：比例是**静态**的，访问负载不均时某组池满不能借用别组；后续可加"按访问热度动态调比例"。

### 2.3 scheduler / delegate 影响

- **scheduler**：每子池独立
  - `slot_meta[group][n_slots_group]`（64 位原子字）
  - `expert_directory[group]`（expert → slot 或 UNASSIGNED）
  - 驱逐/命中统计/EST1 按组独立
  - `branch_layout(layer, branch, name, off, bytes)`：按层定位组 → 组内 branch 偏移
  - `slot_size(layer)` / `pool_base(layer)`：按层所属组返回
- **minigraph delegate**：`w3d` 叶子按层查组，`nb[2]=组 slot_size`、`data=组 pool_base + off`（原逻辑改为按组）
- **pin/unpin**：`sched.pin_expert(layer, e)` 内部按层定位组，refcount 在组内

### 2.4 对 deepseek4（同质）的影响

deepseek4 只有一个组（全同质）→ 单子池 = 现行为，零改动路径。

## 3. 实施步骤

1. `moe_loader`：组识别（替代 heterogeneous_layers throw/排除），`topo.groups` 填充 + 字节占比。
2. `scheduler`：多子池控制面（slot_meta/expert_directory/驱逐/`branch_layout`/`slot_size(layer)`/`pool_base(layer)`）。
3. `minigraph_exec`：`w3d` 叶子按组 stride/base。
4. `route_b_inject`：分配改为按组预算。
5. 验证：gemma-4-26B-A4B `--expert-backend` 全层进池跑通 + 数值等价（与无 expert-backend 基线对比）；deepseek4 回归。

## 4. 验证记录

**2026-08-27**：gemma-4-26B-A4B-it-UD-Q4_K_M + `--expert-backend --moe-ram-pool 8192` ✅

- 组识别：2 组（group 0 = 29 层 Q4_K+Q5_1 3630KB；group 1 = layer 29 Q8_0 4235KB）。
- 池分配：2297 slots / 8187MB（组 0 2221 槽 7.88GB，组 1 76 槽 0.31GB）——槽数与池大小均按源模型专家组成比例。
- 全层（含异构 layer 29）进池，无排除；chat 正常返回。
- 修复记录：
  - 预算乘法 uint64 溢出（8GB×总字节）→ double 计算。
  - per-group staging 用组内 **max** 专家布局（文件偏移对齐导致 per-expert staging 需求不同，E0 的偏小会越界 NOACCESS）。
