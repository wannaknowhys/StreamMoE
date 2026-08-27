# StreamMoE 自有 GGUF 格式设计 (STREAMMOE_GGUF_FORMAT.md)

> 状态：**设计（v1/v2 定稿）**。目标：让 MoE 模型加载/装载全面走 DIO 且免 staging。
> 依据：ggml-org/ggml `docs/gguf.md`（`general.alignment` 可设 4096；`tensor_data` 是 arbitrary binary data、tensor 由显式 offset 定位、offset 必须 ALIGNMENT 倍数；社区 KV namespaced）。
>
> **关键结论**：GGUF 张量没有 stride/切片语义（`gguf_tensor_info_t` 只有 name/ne/type/offset，数据必须连续）——"专家物理连续"与"原版可读"**不可兼得**。因此两个版本：
> - **v1 = GGUF 超集（兼容，原版可读）**：每分支张量连续，dense/expert 分区。
> - **v2 = expert-blocks（自有格式，原版不可读）**：每专家一个紧凑块，一次 DIO 装载。

---

## 0. 多分片（shard）背景

`-00001-of-00005`：GGUF 多分片命名（`<ShardNum>-of-<ShardTotal>`，从 00001 起 5 位零填充）。**不是随便切**——按张量分布（`split.no`/`split.count`/`split.tensors.count`），llama.cpp 加载时合并全部 `splits`。deepseek 是 5 片（shard1 只含元数据 + 部分张量，00002+ 含权重）。

---

## 1. v1：GGUF 超集（兼容，原版可读）

### 1.1 目标
- dense 整段一次读入（替代逐张量 seek）；expert 段 4K 对齐 → DIO 免 staging。
- **原版 llama.cpp 可直接加载**（张量连续 + offset 显式）。

### 1.2 文件结构
```text
GGUF v3
├─ header + metadata
│   ├─ general.alignment = 4096
│   └─ stream_moe.layout = "sections-v1"
│      stream_moe.dense_section  = [off, size)     # tensor_data 内字节范围
│      stream_moe.expert_section = [off, size)
├─ tensor_infos（offset 显式，均 4K 对齐）
└─ tensor_data
   ├─ DENSE SECTION   ← 非 `_exps` 张量，模型构建顺序连续
   └─ EXPERT SECTION  ← `_exps` 张量，按层（blk.0..N）每分支张量连续
```

### 1.3 布局（专家三分支分散）
每层三个 3D 张量各占一块连续（`gate_up[128]` → `down[128]` → `scale[128]`，层内相邻）：
- 专家 e 的 gate_up 在 gate_up 块 `e*size`，down 在 down 块，scale 在 scale 块——**三分支分散**。
- 装载专家 e = **3 次独立 DIO**（三个分支各自 4K 对齐位置）。

### 1.4 张量重排
1. dense（非 `_exps`）按构建顺序 → dense 段，每张量 4K 对齐。
2. expert（`_exps`）按层（blk.0..N）→ 每层分支张量连续到 expert 段，每张量 4K 对齐。
3. 写 `general.alignment=4096` + `stream_moe.*` 分区表。

### 1.5 loader
- dense：检测 `stream_moe.layout` 且张量在 dense 段 → **整段一次读**到确定 buffer（`dense_section.size`），段内张量按 `offset - dense_section.off` 映射。
- expert：scheduler 用 `tensor_info.offset`（4K 对齐）DIO 读分支张量内的专家（`e * aligned_expert_size`，免 staging 或只处理首尾）。
- 无 `stream_moe.*` → 现有路径不变（兼容普通 GGUF）。

---

## 2. v2：expert-blocks（自有格式，专家紧凑块）

### 2.1 目标
**每个专家一个物理连续块**（gate_up+down+scale 拼一块），装载专家 e = **一次 DIO 读**，完美匹配 route B 槽布局。

### 2.2 结构（非 GGUF 超集）
```text
GGUF v3（复用 header/KV/tensor_infos，但 tensor_data 语义变）
├─ general.alignment = 4096
├─ stream_moe.layout = "expert-blocks-v2"
├─ stream_moe.dense_section      = [off, size)
├─ stream_moe.expert_sections[]  = {            # 数组，每元素一个专家块
│      { expert_block_offset, block_size, n_subtensors },
│      ...                                        # 块内：gate_up_e | down_e | scale_e 连续
│   }
└─ tensor_data
   ├─ DENSE SECTION
   └─ EXPERT SECTION（每专家一紧凑块，4K 对齐，块索引 = 专家全局索引）
```

### 2.3 每专家块
- 块内容：`gate_up_e + down_e + scale_e`（该专家三分支物理连续，4K 对齐）。
- `expert_sections[]` 数组记录每个块的 `offset + block_size`（**对齐后大小**，DIO 直用）。
- **异构支持**：不同层专家块大小不同 → 每块条目独立 `block_size`。

### 2.4 张量名处理（关键）
- 分支 3D 张量被**打散**（gate_up_exps 不再是连续 3D 张量）。
- 原版按 `blk.N.ffn_gate_up_exps.weight` 找连续张量 → **读不出**。
- **我们 loader 专用**：`llama_model_loader` 检测 `stream_moe.layout="expert-blocks-v2"` → 用 `expert_sections[]` 建专家块映射，`moe_backend` 的 delegate 从块读专家权重。
- 验证只能靠我们自己（数值等价回归），原版/量化工具不认。

### 2.5 loader
- dense：同 v1（整段读）。
- expert：scheduler 按 `expert_sections[i].offset` **一次 DIO 读整个块** → 槽（gate_up/up/down/scale 区域按块内布局）。
- 无 `stream_moe.*` → 现有路径。

---

## 3. 分片（shard）下的转换

- 输入分片（`-00001-of-00005`）：转换器读全部 `splits`，合并张量视图，重排到目标（v1/v2）。
- 输出：单文件（或同样分片——v1 可分片，v2 专家块分片可选）。

---

## 4. 转换器（stream_moe_convert）

- 输入：`-m <model.gguf>`（多分片自动合并）。
- 输出：`-o <out.gguf>` + `--format v1|v2`（默认 v1）。
- 流程：
  1. 解析 GGUF（header/KV/tensor_info，张量 offset/size/type/ne）。
  2. 分类张量（dense / `_exps`）。
  3. 重排写入目标（v1：dense 段 + 分支张量段；v2：dense 段 + 专家紧凑块）。
  4. 写 `general.alignment=4096` + `stream_moe.*` 分区/块表。
  5. 校验：转换后原版（v1）加载输出与转换前一致；v2 用我们 loader 数值等价回归。

## 5. 收益与代价

| | v1（超集） | v2（expert-blocks） |
|---|---|---|
| 原版可读 | ✅ | ❌ |
| dense 整段读 | ✅ | ✅ |
| 专家 DIO | 3 次/专家（分支分散）| **1 次/专家**（紧凑块）|
| 异构支持 | ✅（分支张量独立）| ✅（块大小独立）|
| 生态 | GGUF 兼容 | 自研 |
