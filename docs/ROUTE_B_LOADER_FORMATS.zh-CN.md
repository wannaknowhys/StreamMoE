# Route-B 加载器：GGUF 输入格式、差距与异步加载设计

[English](ROUTE_B_LOADER_FORMATS.md) | [简体中文](ROUTE_B_LOADER_FORMATS.zh-CN.md)

> 本文档的事实来源：`tools/stream_moe_layout.js`（转换器布局逻辑，写入方向）+ `src/loader/moe_loader.cpp` / `src/io/staging_reader.h`（加载器，读取方向）。转换器在加载器之后演进；本文档记录两者之间的分歧以及目标异步加载设计。

> **2026-09 修订（取代下文的 v1 超集和整块 DIO）**：
> ggml-vulkan 将每专家的 stride 硬编码为单张量紧凑大小（`ne0*ne1`），因此 route B 正迁移至 **结构体数组池（SoA，每个张量一列）** 以及 v2 块结构（**每个分支张量切片在块内部 4K 对齐**，参见 `STREAMMOE_GGUF_FORMAT.md` §2.6）。对加载器的影响：
> v1 sections-v1 已废弃（GGUF 张量偏移必须紧凑/单调——writeV1 的每专家重排产物会导致 llama 拒绝加载）。对于 v2，加载转变为 **每个 (expert, tensor-slice) 执行一次 DIO**，而非对整个块执行单次 DIO：若切片的 perExpert 为 4K 整数倍，则直接加载至对应的张量列（对齐源 + 对齐槽位）；否则 DIO 读取 4K 窗口至中转区后拷入对应列。列布局由 `src/loader` + `src/backend/scheduler` (SoA) 决定，独立于文件格式。

## 1. 输入格式

| 格式 | `stream_moe.layout` | `incomplete` | 专家布局 | 对齐 | 读取计划 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **原始 GGUF**（单文件或 `-00001-of-N.gguf`） | 缺省 / `"original"` | - | 按张量连续：专家切片 = `tensor.offset + e*perExpert`；每个专家 3 个子张量（gate/up/down 或 gate_up/down） | GGUF 默认（32B / 量化块） | 3 次扇区对齐读取至中转 buffer + memcpy 至槽位（需要中转） |
| **v1 sections-v1** | `"sections-v1"` | - | 相同的按张量切片，但张量由转换器按 4K 对齐 | 4096 | **设计预期**：3 次异步 DIO 直接读入槽位（免中转）。**未实现**——回退至原始中转路径 |
| **v2 expert-blocks-v2** | `"expert-blocks-v2"` | - | 按 (layer, expert) 块；分支（gate_up/gate/up/down）在块内按 `branchOff` 拼接；块大小 = alignUp(sum(branch perExpert), 4096) | 4096 块 | 1 次整块异步 DIO 直接读入槽位（块布局 == 槽位布局） |
| **v2 chunk** | `"expert-blocks-v2"` | `1` | 块条带分散在 N 个分片文件中（每文件 `chunk_slices`）；单个专家块跨越至多 N 个文件段 | 4096 条带 | **未实现**——加载器硬编码单文件（`shard_idx = 0`） |

## 2. 布局 KV 语义 (`stream_moe.*`)

由 `stream_moe_layout.js` 中的 `writeV2` / `writeV2chunk` 写入，由 `moe_loader.cpp` 中的 `build_v2_experts` 读取：

| KV | 含义 |
| :--- | :--- |
| `stream_moe.layout` | `"original"` / `"sections-v1"` / `"expert-blocks-v2"` |
| `stream_moe.incomplete` | `1` = v2 chunk（分片文件）；`0`/缺省 = 单文件 |
| `stream_moe.dense_section` | `[0, denseEnd]` - 密集张量区（在专家块之前） |
| `stream_moe.expert_sections` | 每个块 `[off, size, nsub]`（共 nLayer*nExpert 个块） |
| `stream_moe.expert_branch_names` | 按层展平的完整分支张量名称 |
| `stream_moe.expert_branch_sizes` | 每个分支的 `perExpert` 字节数（已展平，与名称顺序一致） |
| `stream_moe.expert_branch_counts` | 每层的分支数量（支持异构非均匀 MoE 层） |
| `stream_moe.chunk_no` / `chunk_total` | v2 chunk 的分片索引 / 总分片数 |
| `stream_moe.chunk_slices` | 每个文件的 `[denseBlocks, blockSlices...]`——该文件持有的 4K 对齐条带 |

## 3. 当前差距（加载器 vs 转换器）

1. **v1 4K 对齐未利用。** `moe_loader::parse_gguf_topology` 仅对 `sections-v1` 设置 `topo.layout = V1_SECTIONS`，随后直接回退到 ORIGINAL 按张量切片路径（中转 + 复制）。4K 对齐的 v1 布局应当允许 3 次直接异步 DIO 读入槽位（免中转、免复制）。需要确认：v1 中是否每个专家切片都满足 4K 对齐（即使张量起始 4K 对齐，perExpert 也不一定能被 4K 整除）？

2. **v2 chunk 暂不支持。** `build_v2_experts` 硬编码了 `shard_idx = 0`，仅从主文件读取整块，从未读取 `chunk_slices` / `incomplete`。对于 v2 chunk，每个专家块是分散在 N 个文件中的条带（类似转换器中的 `rangeToSegs`）——加载器必须将单个块映射到 N 个文件段并下发分文件 DIO。

3. **异构专家。** 存在按专家大小分组（`topo.groups`，参见 `MULTI_SUBPOOL.md`）；读取计划必须按组构建（每组拥有独立的中转大小 / 槽位大小 / DIO 次数），v2 满足此要求，但 v1/original 重构中也必须予以保留。

## 4. 目标异步加载设计（概念——与 convertd 对齐）

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
- **v1**：若切片为 4K 对齐，则 3 次异步 DIO 直接读入槽位（免中转）。
- **v2**：1 次整块异步 DIO 直接读入槽位。
- **v2 chunk**：按文件条带读取（每专家 N 个分段），直接读入槽位。

所有读取均为异步 + 连续（每文件最大连续对齐区间），支持并发 in-flight。**专家异步头中的时间字段**：提交/DIO完成/就绪时捕获的原始 TSC（`uint64_t req_tsc` / `dio_tsc` / `done_tsc`），在性能分析时转换为 ns（启动时通过 chrono 校准一次 TSC 频率）。无需每专家 printf——该字段为惰性数据，供后续动态 profiling / 自适应预取使用。

## 5. 待决问题

- **Q1 original "中转" 语义**：中转路径为 3 次扇区对齐读取 + 复制；需确认 original 切片非 4K 对齐，即使填充后也无法直接读入槽位（即中转为必须流程，非可选）。
- **Q2 v1 "3 reads" 划分**：gate_up / down / scale 分为三个独立的 4K 对齐区域？（scale 目前在转换器中视为密集张量——参见 `buildModel` `isScale`）。需确认确切的 v1 section 布局。
- **Q3 布局单一真实来源**：转换器布局逻辑位于 JS（`stream_moe_layout.js`）；C++ 加载器无法直接复用。备选项：(a) 将布局计算下沉至 C++ 共享模块，convertd 和 loader 共同链接；(b) 编写完整 C++ convertd（弃用 JS）；(c) 保持 JS 为主权威并在加载器中重新实现布局（存在分歧风险）。Q3 将决定该工作的工作量。
