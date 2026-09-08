# Route-B 加载器：GGUF 输入格式、差距与异步加载设计

[English](ROUTE_B_LOADER_FORMATS.md) | [简体中文](ROUTE_B_LOADER_FORMATS.zh-CN.md)

> 本文档的事实来源：`src/loader/model_builder.cpp` / `src/loader/model.h`（读写共用的 `model_t`）+ `src/convert/writer.cpp`（写入方向）。转换器已纯 C++ 化，与加载器共用 `model_t`。

> **2026-09 修订（v1 已删除）**：
> ggml-vulkan 将每专家的 stride 硬编码为单张量紧凑大小（`ne0*ne1`），因此 route B 正迁移至 **结构体数组池（SoA，每个张量一列）** 以及 v2/v3 块结构（**每个分支张量切片在块内部 4K 对齐**，参见 `STREAMMOE_GGUF_FORMAT.md` §2.6/§3）。对加载器的影响：
> 加载为 **每个 (expert, tensor-slice) 执行一次 DIO**，而非对整个块执行单次 DIO：若切片的 perExpert 为 4K 整数倍，则直接加载至对应的张量列（对齐源 + 对齐槽位）；否则 DIO 读取 4K 窗口至中转区后拷入对应列。列布局由 `src/loader` + `src/backend/scheduler` (SoA) 决定，独立于文件格式。

## 1. 输入格式

| 格式 | `stream_moe.layout` | `incomplete` | 专家布局 | 对齐 | 读取计划 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **原始 GGUF**（单文件或 `-00001-of-N.gguf`） | 缺省 / `"original"` | - | 按张量连续：专家切片 = `tensor.offset + e*perExpert`；每个专家 3 个子张量（gate/up/down 或 gate_up/down） | GGUF 默认（32B / 量化块） | 3 次扇区对齐读取至中转 buffer + memcpy 至槽位（需要中转） |
| **v2 expert-blocks-v2** | `"expert-blocks-v2"` | - | 按 (layer, expert) 块；分支（gate_up/gate/up/down）在块内按 `branchOff` 拼接；块大小 = alignUp(sum(branch perExpert), 4096) | 4096 块 | 1 次整块异步 DIO 直接读入槽位（块布局 == 槽位布局） |
| **v2 chunk** | `"expert-blocks-v2"` | `1` | 块条带分散在 N 个分片文件中（每文件 `chunk_slices`）；单个专家块跨越至多 N 个文件段 | 4096 条带 | **未实现**——加载器硬编码单文件（`shard_idx = 0`） |
| **v3 category-sections** | `"v3"` | - | 四个段：C2 全局 dense / C1 按层 dense / C4 专家小表 / C3 专家块（块布局同 v2）。按"是否被 MoE 闭包消费"划分（见 `STREAMMOE_GGUF_FORMAT.md` §3） | 4096 | **未实现**——每段对应一种驻留策略：C2 常驻、C1 按层流式、C4 每设备复制、C3 池化 / 跨设备 |

## 2. 布局 KV 语义 (`stream_moe.*`)

由 `src/convert/writer.cpp` 写入，由 `parse_model`（`src/loader/model_builder.cpp`）读取：

| KV | 含义 |
| :--- | :--- |
| `stream_moe.layout` | `"original"` / `"expert-blocks-v2"` / `"v3"` |
| `stream_moe.incomplete` | `1` = v2 chunk（分片文件）；`0`/缺省 = 单文件 |
| `stream_moe.dense_section` | `[0, denseEnd]` - 密集张量区（在专家块之前） |
| `stream_moe.expert_sections` | 每个块 `[off, size, nsub]`（共 nLayer*nExpert 个块） |
| `stream_moe.expert_branch_names` | 按层展平的完整分支张量名称 |
| `stream_moe.expert_branch_sizes` | 每个分支的 `perExpert` 字节数（已展平，与名称顺序一致） |
| `stream_moe.expert_branch_counts` | 每层的分支数量（支持异构非均匀 MoE 层） |
| `stream_moe.chunk_no` / `chunk_total` | v2 chunk 的分片索引 / 总分片数 |
| `stream_moe.chunk_slices` | 每个文件的 `[denseBlocks, blockSlices...]`——该文件持有的 4K 对齐条带 |
| `stream_moe.dense_global_section` (v3) | `[off, size]` - C2 全局 dense 区 |
| `stream_moe.dense_layer_sections` (v3) | `[layer, off, size, ...]` - C1 按层 dense 区 |
| `stream_moe.expert_meta_sections` (v3) | `[layer, off, size, ...]` - C4 每层专家小表（可为空） |

## 3. 当前差距（加载器 vs 转换器）

1. **v2/v3 chunk 读取（B37）。** `parse_model` 现已自动发现分片兄弟文件并把条带映射为多段 `src`；转换器往返逐字节一致。**引擎**的多段并发 DIO 在 N>1 分片时仍会非确定性损坏专家权重（1-chunk 逐字节正确；plan 正确；staging/批量提交无效）。见 `BUG_TRACKER.md` B37。

2. **异构专家。** 存在按专家大小分组（`topo.groups`，参见 `MULTI_SUBPOOL.md`）；读取计划按组构建。

3. **v3 驻留策略未接线。** `parse_model` 已理解 v3；引擎仍对所有 dense 一视同仁。四类各自需要策略：C2 常驻、C1 按层流式、C4 每设备复制、C3 池化。

## 4. 目标异步加载设计（概念——与 C++ 写入器对齐）

统一规划器 + 规范异步 DIO：

```
输入路径
  -> 格式检测（布局 KV + incomplete 标志，均位于文件头部）
  -> 按格式规划器 -> 统一读取计划：
       expert e -> [ { file, off, len, slot_off } ... ]   (1..N 个分段)
  -> 异步 DIO 引擎（IOCP / io_uring / io_submit 回退），支持 N 个 in-flight
  -> 完成 -> 槽位放置
```

各格式 DIO 特征：
- **original**：每个专家 3 次读取，每次读入 8K 填充的中转缓冲区（前后填充，大小为 size+2*4096），然后 memcpy 至槽位。按 4K 对齐以满足 DIO。
- **v2**：1 次整块异步 DIO 直接读入槽位。
- **v2/v3 chunk**：按文件条带读取（每专家 N 个分段），直接读入槽位。

所有读取均为异步 + 连续（每文件最大连续对齐区间），支持并发 in-flight。**专家异步头中的时间字段**：提交/DIO完成/就绪时捕获的原始 TSC（`uint64_t req_tsc` / `dio_tsc` / `done_tsc`），在性能分析时转换为 ns（启动时通过 chrono 校准一次 TSC 频率）。无需每专家 printf——该字段为惰性数据，供后续动态 profiling / 自适应预取使用。

## 5. 待决问题

- **Q1 original "中转" 语义**：中转路径为 3 次扇区对齐读取 + 复制；需确认 original 切片非 4K 对齐，即使填充后也无法直接读入槽位（即中转为必须流程，非可选）。
- **Q2 布局单一真实来源（已解决）**：转换器已纯 C++ 化，与 loader 共用 `src/loader/model.h::model_t` + `layout_math.h`，不再有 JS/C++ 双实现。
