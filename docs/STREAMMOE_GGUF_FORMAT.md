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

---

## 6. 多文件：v1 合分片/分片，v2 RAID0 切分

### 6.1 v1：合分片与张量级分片
- **合分片**（`-00001-of-00005` → 单文件）：读全部 `splits` 张量，重写为单文件（张量连续）。流式（`gguf_init_from_callback` 读 + 边写边拷，162GB 不全量进内存）。**无本质困难**。
- **分片**（单文件 → 文件系列）：**张量级**分片（文件1 含张量 0..k，文件2 含 k+1..m）——每个文件是完整可读 GGUF（类似原版 split），原版 `splits` 机制可合并。

### 6.2 v2：专家块 RAID0 切分（跨盘）
- **切分规则（固定，统一适用于所有专家，含异构）**：专家块**字节级**跨文件切（不考虑专家内三张量结构），按 4K 块**均匀分配余数**——每个专家块独立用同一条规则：
  - 块总数 `B`（每专家块自己的 4K 块数），N 个文件 → `base = B/N`（整除），`rem = B%N`；
    **前 rem 份各 `base+1` 个 4K，其余 `base` 个 4K**（余数摊到前面，不做"最后一份很小"）。
  - 段_i 块数 = `base + (i < rem ? 1 : 0)`；段_i 偏移 = `(i*base + min(i, rem)) * 4K`。
  - 例：B=3329, N=3 → 1110 / 1110 / 1109；B=7, N=3 → 3/2/2（12K/8K/8K）；B=203, N=10 → 21 21 21 20x7。
  - **异构无碍**：大专家块、小专家块各按自己的 B 用同一规则切，互不影响。
- **bookkeeping 最小**：每专家只需知道"块总数 B"（+ 固定文件数 N）——段边界由 `base/rem` 推导，无需逐段表。
- **比例切分（可选 `--ratio a:b:c`，默认均匀）**：
  - 均匀（默认）：`base = B/N`、`rem = B%N`、前 rem 份 +1（余数前摊，不做"最后一份很小"）。
  - 比例（`--ratio`）：**largest remainder（Hamilton 法）**——每份配额 `quota_i = B·r_i/Σr`（浮点），`base_i = floor(quota_i)`，余数 `diff = B - Σbase`；按 quota **小数部分降序（稳定排序）**，前 diff 份各 +1（"提取差值顺序"即小数降序的份索引，如 `8:9:9:7:9` → 23514）。
  - **校验（防御性编程，非性能关键路径）**：切分完成后断言 `Σ段块数 == B`（dense 与每个专家块都校验），不符即报错，不静默。
- **文件大小不要求相等**：每个文件 = 各专家在该文件片段的累积字节，天然不同——不影响（切分规则固定）。
- **装载**：专家 e = **N 次并发 DIO 直读**各文件片段（4K 对齐 start + 4K 对齐 len）→ 合并到槽。比非对齐 + staging 更简单。
- **文件结构**：**每个文件都是完整 GGUF**（header/KV/tensor_info），但含：
  - `stream_moe.format = "expert-blocks-v2"`
  - `stream_moe.chunk = <no>/<total>`（自编号）
  - `stream_moe.incomplete = 1`（**注明无法被原版单独读取**——张量被切分、文件内不完整，原版读会错）
  - metadata（KV）主要在第 1 个文件；后续文件含最小 header + 编号 KV + 数据段。
- **loader（common 侧）**：接受多文件输入（`--model` 主文件 + 按 `stream_moe.chunk`/`split.*` 自动发现兄弟文件），v2 下按固定切分规则合并装载专家。

---

## 7. 转换器架构（node 指挥 + C/C++ wrapper）

- **Node 无可靠 GGUF 读写库**（`@huggingface/gguf` 等只读 metadata、写/重排/大张量不可靠）→ 用 **ggml 的 gguf C 库**（`ggml/include/gguf.h`：`gguf_init_from_callback` 流式读 + `gguf_set_val_*`/`gguf_add_tensor`/`gguf_write_to_file` 写）。
- **wrapper（C/C++，`stream_moe_convertd`）**：**JSON 行管道**（stdin/stdout，每次转换跑一次）：
  - `open {path}` → 输出 metadata JSON（header/alignment/KV/tensor_info 摘要）。
  - `convert {format v1|v2, out, plan}` → 按 plan 读源张量、写目标（进度/完成/错误事件）。
  - `chunk {n_chunks, out_base}` → v2 RAID0 切分（或 v1 张量级分片）。
- **node（`tools/stream_moe_convert.js`）**：spawn wrapper → `open` 拿 metadata → 规划（分类/重排/切分/合分片）→ 发 `convert`/`chunk` → 汇总。
- 复用：vendored `ggml/src/gguf.cpp`（llama.cpp 已编译进 ggml 库，wrapper 链接 ggml.lib）。

### wrapper 编译（Windows，clang-cl + vendored ggml）
```bat
rem tools/stream_moe_convertd.cpp：JSON-lines 管道（stdin 命令 / stdout 响应）
F:\Dev\LLVM\bin\clang-cl.exe /std:c++17 tools\stream_moe_convertd.cpp /EHsc /MT ^
    "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/include" ^
    "/IF:/Dev/StreamMoE/third_party/llama.cpp/ggml/src" ^
    F:/Dev/StreamMoE/build/main/llama-build/ggml/src/ggml-base.lib ^
    F:/Dev/LLVM/lib/libomp.lib /Fe:temp/stream_moe_convertd.exe
rem 测试 open（读 GGUF metadata -> JSON）：
echo {"cmd":"open","path":"N:\\AI_LLM\\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"} | temp\stream_moe_convertd.exe
```

### wrapper 命令
- `open {path}` → metadata JSON（header/alignment/KV/tensor_info）——✅ 已实现（gemma 验证）。
- `convert {format v1|v2, in, out, plan}` → 读源张量写目标——TODO。
- `chunk {n_chunks, out_base}` → v2 RAID0 切分——TODO。

---

## 8. 转换器功能清单（v1/v2）【2026-08-29 更新为实际状态】

`tools/stream_moe_convertd.cpp`（JSON 行管道 wrapper）+ `tools/stream_moe_convert.js`（CLI 前端，多分片发现）：

- [x] **v1 sections**（`alignment=4096` + dense/expert 分区，原版可读）——`convert --format v1`。
- [x] **v2 expert-blocks**（专家紧凑块，自有 loader）——`convert --format v2`；已产出 gemma-4-26B-A4B `-v2.gguf` / `-v2-NOHOLE.gguf` 并做 bit 级验证。
- [x] **v1 合分片**（`-00001-of-00005` → 单文件，`in` 支持 `p1;p2` 多分片输入）。
- [x] **v2 RAID0 切分**（dense 段 + 每专家块按 4K 块切 N 份，均匀 base/rem 或 `--ratio` largest-remainder）——`chunk` 命令（`out1,out2,...` 逗号分隔 N 路径，完整 GGUF + `incomplete=1`）。
- [ ] **v1 张量级分片**（单文件 → 文件系列，原版可合并）——未实现。
- [ ] **v2 反重排还原**（v2 → v1/原版，经中间格式拆块）——未实现，见 §9。
- [x] **数值等价验证**（gemma v2 vs 原版逐位一致已做；v1 原版可读自证）。

**结论**：原版 → v1/v2 单文件 + v1 → v2 + v2 → chunk 已可用；**中间格式（§9）统一 parse/write 后矩阵全通**（含 v2 → v1/原版 反重排还原）。

---

## 9. 转换器中间格式（统一模型抽象）

> 目标：让**所有格式**（原版 / v1 / v2 / v2 切）都能解析到同一个内存中的模型抽象，再从该抽象写出任意目标（v1 / v2 / v2 切）——矩阵全通、单向可逆（v2 只是"打乱"而非"丢失"，可拆回）。这样**任意多方向转换可互相验证转换器无 bug**。

### 9.1 中间格式（JS 内存中的模型抽象，不是文件）

以 **v2 为中心**的块抽象：头块 + [dense 张量块][专家原子块]。

```js
{
  headerKV,                                 // 源 KV（写回时保留）
  dense:  [ { name, ne, type, src:{file,off,size} } ],  // dense 张量块（每张量一块）
  experts:[ { layer, branch, name, ne, type, perExpert,
              srcs: [每专家 {file, off, size}] } ],     // 专家原子块（每专家每分支）
}
```

- **原子块 = 每专家每分支的 `per_expert` 连续切片**——不携带源格式的对齐（切片语义）。
- **专家块（v2）= 该专家各分支原子块连续拼接**——块内分支**不各自对齐**（与 v2 真实布局一致），仅块尾 `align_up` 到 4K → **紧凑**。因此从 v1 读（三分量张量各自 4K 对齐）也不会让 v2 块不紧凑——对齐被"读取"吸收。
- **scale 走 dense 张量块**（不进专家块，v2 现状）。
- 转换 = 源格式解析成原子块列表 → 按目标布局重排（v1 张量连续 / v2 专家块 / v2 切 段）。

### 9.2 解析器（各格式 → 中间）

| 源格式 | 解析 |
| :--- | :--- |
| 原版 | open 张量列表 → dense/expert（name 含 `_exps` 且非 `.scale`）→ `src` = shard+off+size |
| v1 | 同原版（张量连续可读）+ `dense_section` 标记 |
| v2 | `expert_sections`（块 off/size）+ `expert_branch_names/sizes/counts`（块内布局）→ 每专家块拆出各分支区间 → 专家张量 `src` = 块内分支区间（**可逆关键**）|
| v2 切 | chunk 规则（base/rem 或 ratio）→ 专家张量 `src` = "文件 i 的段区间" |

### 9.3 写入器（中间 → 各格式）

| 目标 | 写入 |
| :--- | :--- |
| v1 | 张量重排（dense 段 + expert 段，4K 对齐）|
| v2 | dense 段 + 每专家块（从分支 `src` 拼）|
| v2 切 | dense/专家块按规则切 N 份（每份完整 GGUF + chunk KV）|

转换 = `parse(源) → 中间 → write(目标)`；执行层（convertd）只做"按 `src` 读区间 + 按布局写"。

### 9.4 矩阵（经中间格式全通）

| 源 \ 目标 | v1 | v2 | v2 切 |
| :--- | :--- | :--- | :--- |
| 原版 | ✓ | ✓ | ✓ |
| v1 | — | ✓ | ✓ |
| v2 | ✓（拆块还原）| — | ✓ |
| v2 切 | ✓（合并还原）| ✓ | 重切 |

- **可逆性**：v2 的专家数据是"打乱"（重排成块），张量 name/ne/type + branch 布局都在 KV/tensor_info——可拆回；真正丢失的只有多分片结构（`split.*`）与对齐 padding（有原始大小可去）。
- **验证方法**：`A→中间→B` 与 `A→B` 直转对比逐字节；`A→中间→v2→中间→v1` 回环与 `A→v1` 对比——多方向转换互相校验转换器。
