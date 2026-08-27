# StreamMoE 自有 GGUF 格式设计 (STREAMMOE_GGUF_FORMAT.md)

> 状态：**设计（草案）**。目标：让 MoE 模型加载/装载全面走 DIO 且免 staging。
> 结论：**新格式 = GGUF 超集**——`general.alignment=4096` + **dense/expert 分区** + `stream_moe.*` 元数据。扩展名仍 `.gguf`（生态兼容），文件内一眼可认。
> 依据：ggml-org/ggml `docs/gguf.md`（`general.alignment` 可设 4096；`tensor_data` 是 arbitrary binary data、tensor 由显式 offset 定位、offset 必须 ALIGNMENT 倍数；社区 KV 允许 namespaced 扩展）。

---

## 1. 目标

| 现状问题 | 新格式解决 |
|---|---|
| `use_mmap=false` 后 dense 逐张量 `seek/read`（568 次，慢）| dense **整段一次读入**（大块 I/O）|
| 专家段非 4K 对齐 → DIO 需 staging（对齐 + 复制）| 专家段 **4K 对齐连续** → DIO 直读专家（免 staging 或大幅减少）|
| 无法区分 dense/expert 布局 | 分区表元数据 |

## 2. 文件结构（GGUF v3 超集）

```text
GGUF v3
├─ header（magic "GGUF" + version + alignment 字段 + tensor_count + kv_count）
├─ metadata
│   ├─ general.alignment = 4096                        ← 4K 全局对齐（8 的倍数，规范支持）
│   ├─ general.* / llama.* ...（原样保留）
│   └─ stream_moe.layout = "sections-v1"               ← 格式标识（一眼认出）
│      stream_moe.dense_section  = [off, size)        ← tensor_data 内字节范围
│      stream_moe.expert_section = [off, size)
├─ tensor_infos（offset 相对 tensor_data，均 4K 对齐）
└─ tensor_data
   ├─ DENSE SECTION   ← 所有非 `_exps` 张量，按模型构建顺序连续
   └─ EXPERT SECTION  ← 所有 `_exps` 张量，按层（blk.0..N）连续
```

## 3. 关键依据（gguf.md）

- `general.alignment`：uint32，默认 32，**必须是 8 的倍数**，设 4096 合法。
- `tensor_data`：**arbitrary binary data**，tensor 由 `tensor_info.offset` 显式定位（相对 tensor_data），offset 必须 ALIGNMENT 倍数——**规范不强制 tensor 连续或按顺序**。
- 社区 KV namespaced（`stream_moe.` 前缀），原版读器忽略未知键。

## 4. 分区表与 KV 定义

```text
stream_moe.layout         (string)  = "sections-v1"      # 格式版本标识
stream_moe.dense_section  (u64[2])  = [off, size)        # tensor_data 内 dense 段字节范围
stream_moe.expert_section (u64[2])  = [off, size)        # tensor_data 内 expert 段字节范围
```

- 张量精确定位仍走 `tensor_info.offset`（每个张量 4K 对齐）——分区段只是"整块读"与"确认连续对齐"的辅助。
- 未知 `stream_moe.*` 键被原版忽略 → **兼容**。

## 5. 张量重排规则（stream_moe_convert）

1. 读原 GGUF → 解析全部张量（name/type/ne/offset/size）。
2. 分类：
   - **dense**：名字不含 `_exps`（attention/embedding/norm/gate_inp/router...）。
   - **expert**：名字含 `_exps`（`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`/`ffn_gate_up_exps`/`_exps.scale`...）。
3. 写入：
   - dense 段：按模型构建顺序（blk.0.attn... → blk.0.norm... → ...），每个张量 `offset` 4K 对齐。
   - expert 段：按层（blk.0 的所有 `_exps` 连续 → blk.1 ... → blk.N），每张量 4K 对齐。
4. 写 `general.alignment=4096` + `stream_moe.*` 分区表。
5. 输出：`-s4k` 后缀的 `.gguf`（本体格式兼容）。

## 6. loader 改动点（llama.cpp / route B）

- `llama_model_loader`：读 `general.alignment`（gguf 已支持）+ `stream_moe.dense_section/expert_section`。
- **dense 装载**：非 mmap 分支改为——检测 `stream_moe.layout` 且张量在 dense 段 → **整段一次 DIO/read** 到确定大小 buffer（`dense_section.size`），段内张量按 `offset - dense_section.off` 映射；不再逐张量 seek/read。
- **expert 装载**：scheduler 读专家时用 `tensor_info.offset`（4K 对齐）→ **DIO 直接读专家**（目标 buffer 4K 对齐，免 staging 或只处理首尾）。
- 若文件无 `stream_moe.*`（普通 GGUF）→ 现有路径不变（兼容）。

## 7. 兼容性与验证

- **原版 llama.cpp**：offset 显式定位 → 能正常加载（只是不走分区 DIO）。
- **我们的 loader**：识别分区 → 整段 DIO。
- 验证：
  1. convert 后原版 `llama-cli` 加载输出与转换前一致（数值等价）。
  2. 我们的 loader dense 整段读 + expert DIO 装载，数值等价回归。
  3. 加载/首次推理耗时对比（逐张量 vs 整段）。

## 8. 收益与代价

- **收益**：dense 加载快（整段 I/O）；expert DIO 免 staging；MoE 全链路无 mmap。
- **代价**：convert 一次（离线）；文件需重排（一次性成本）；分区表维护。

## 9. 后续（可扩展）

- 每层 expert 段（更细分区）→ 更高并发装载。
- `stream_moe.lru_*` 等运行时提示 KV。
- 多分片（shard）分区表。
