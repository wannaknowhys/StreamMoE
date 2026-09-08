# StreamMoE 自有 GGUF 格式设计 (STREAMMOE_GGUF_FORMAT.md)

> 状态：**设计（v1/v2/v3 定稿；v3 为目标格式，2026-09）**。目标：让 MoE 模型加载/装载全面走 DIO 且免 staging。
> v3 = 按**是否被 MoE 闭包消费**把张量分四类、各一个物理 section（见 §3）；v2 保留为旧格式（只写单文件）。
> **2026-09 决策**：转换器转**纯 C++**（复用 `model_t`，删 JS/convertd），**v1 彻底删除**（见 §3.5）。
> 依据：ggml-org/ggml `docs/gguf.md`（`general.alignment` 可设 4096；`tensor_data` 是 arbitrary binary data、tensor 由显式 offset 定位、offset 必须 ALIGNMENT 倍数；社区 KV namespaced）。
>
> **关键结论**：GGUF 张量没有 stride/切片语义（`gguf_tensor_info_t` 只有 name/ne/type/offset，数据必须连续）——"专家物理连续"与"原版可读"**不可兼得**。因此：
> - **v1 = GGUF 超集（兼容，原版可读）**：每分支张量连续，dense/expert 分区。**已废弃**（见下）。
> - **v2 = expert-blocks（自有格式，原版不可读）**：每专家一个紧凑块，一次 DIO 装载。旧格式。
> - **v3 = 按闭包四分类（目标格式，2026-09）**：C1 dense 按层 / C2 dense 与层无关 / C3 每专家 / C4 专家小表，各占一个 section（见 §3）。
>
> **2026-09 修正**：v1（sections-v1）因 GGUF tensor offset 必须紧凑单调（gguf reader 校验
> `ti.offset == 累计 padded size`，ggml gguf.cpp:774-794）而否决——无法在单张量内做 4K 专家切片
> stride（writeV1 的 per-expert reflow 产出非法 GGUF，llama 加载报 offset 不匹配）。v2 块内进一步
> 改为"每张量切片独立 4K 对齐"（供推理引擎按张量 DIO + SoA 槽执行，见 §2.6），此变体原版同样不可读；
> v3 在 v2 之上把 dense 段按闭包拆成三段（见 §3）。

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

### 2.6 块内张量切片独立 4K 对齐变体（2026-09，推理引擎 SoA 槽执行用）

> 背景：ggml-vulkan MUL_MAT_ID 专家步长硬编码 = 单张量紧凑大小（`ne0*ne1`），忽略 `nb[2]`。
> 推理引擎槽布局改为"每张量一列（struct-of-array）"，槽 = 某张量的单个专家切片，stride = 该张量
> 紧凑 perExpert。为让每个专家切片的 **DIO 源 offset 4K 对齐**，v2 块内不再是"分支紧凑拼接"
> （gate_up 2230272 半块会把 down 起点错开 2048B 非 4K），而是**每个分支起点独立 align_up 4K**：
>
> ```text
> block e 内部（相对块起点，块起点本身 4K 对齐）:
>   gate_up 切片起点 0            （4K 对齐）
>   down   切片起点 align_up(gate_up perExpert, 4096)   （4K 对齐）
>   块内各分支之间 pad 空洞（converter fill 补 0）
> ```
>
> - **文件不变量**：块内每个分支切片源 offset（块基址 + 分支起点）都是 4K 对齐 → 装载任一专家
>   任一张量切片都可用 DIO 直读。
> - **DIO 分流**：perExpert 为 4K 倍数 → 源(4K)+目标槽(4K) → DIO 直写槽；perExpert 非 4K → DIO 读
>   4K 窗口 → staging move。每个专家切片**独立一次 DIO**（不再是整块一次读）。
> - 数值上块 = 原紧凑块 + 分支间 pad（约 +2048B/块, gemma；deepseek perExpert 全 4K 无 pad）。
> - 兼容：tensor_info 占位、expert_sections 表、chunk 机制不变；仅"块内分支布局"变。loader 需按
>   `stream_moe.branch_align` 区分新旧（无/0 = 旧紧凑拼接，1 = 新分支 4K 对齐）。
>
> **执行侧（2026-09-08 落地）**：SoA 列 + 分支 4K 对齐的最终目的——ggml-vulkan 的
> `MUL_MAT_ID` 专家步长硬编码 `ne0*ne1` 与列 stride（= 单张量紧凑 perExpert）天然一致。
> per-device 整链设备图已落地（docs/M2_DEVICE_EXECUTOR.md §7.9）：VRAM round 的
> mm→weightless→fold→acc_d 全在 Vulkan 上跑，只回读 acc_d；RAM8G+Vulkan0:256M 设备混跑对
> 同分区 CPU cos 0.982（已知 flip 噪声量级，用户决定不追）。

---

## 3. v3：按闭包四分类（2026-09 定稿，目标格式）

> 动机：v2 只有 `dense` / `expert` 两桶，把两种完全不同的生命周期混在 `dense` 里（`token_embd`/`output`
> 每 token 必用、应永久常驻；逐层 `attn`/`norm` 是流式），并且把专家域的非每专家张量（router / shexp /
> scale）含混地塞进 dense。v3 以**是否被 MoE 闭包消费**为判据把张量分成四类，各占一个物理 section，
> 让 loader/executor 分别施加驻留与 DIO 策略。

### 3.1 四类定义

判据以运行期闭包为准（`src/backend/route_b_chain.cpp` 的 `collect_chain`：从 routed `MUL_MAT_ID`
锚点（权重名含 `_exps` 且不含 `_shexp`）沿消费者前向 BFS 到 `ffn_moe_out`；gating 段
`ffn_moe_logits`/`probs`/`argsort`/`topk`/`weights` 留在 dense 侧，不进闭包）。

| 类 | 名称 | 判据 | 基数 | 跨设备策略 |
| :-- | :-- | :-- | :-- | :-- |
| C1 | dense 按层 | 不被闭包消费 + `blk.N.*` | 每层一组 | 按层 DIO/预取，层后驱逐 |
| C2 | dense 与层无关 | 不被闭包消费 + 非 `blk.*` | 全局一个 | 一次载入，永久常驻 |
| C3 | 每专家独立 | 被闭包消费 + 按专家切片（`_exps.weight`，`ne[2] == n_expert`） | 每专家一块 | 专家池（SoA 列），跨设备分片读取 |
| C4 | 专家不按每专家 | 被闭包消费 + 非按专家切片（小表） | 每层一组 | 每设备复制一份（广播） |

**实测归属**（gemma4 / deepseek4）：

| 张量 | 类 | 说明 |
| :-- | :-- | :-- |
| `token_embd` / `output` / `output_norm` / `rope_freqs` / `output_hc_*` | C2 | 每 token 必用，常驻 |
| `attn_*` / `ffn_norm` / dense `ffn_gate/up/down` / `hc_*` / `indexer*` | C1 | 逐层 |
| `ffn_gate_inp`（router） | C1 | 每层一个；输出 `ffn_moe_logits` 属 gating 段，不在闭包 |
| `ffn_{gate,up,down}_shexp` | C1 | 每层一个；普通 `MUL_MAT`（`_shexp` 不含 `_exps`），不在闭包 |
| `ffn_*_exps.weight` | C3 | 专家池 |
| `ffn_down_exps.scale`（gemma） | C4 | 闭包 REPEAT/GET_ROWS 消费，512 B/层 |
| deepseek | C4 = 空 | 无 `_exps.scale`；`exp_probs_b` / `tid2eid` 属 gating，归 C1 |

> 关键结论：**两个模型都不存在"大的、被闭包消费、但非每专家"的张量**。大的要么是 C3（专家池），
> 要么不被闭包消费（C1/C2）。C4 只有"闭包消费的、按专家索引的小表"，目前仅 gemma 的 scale
> （512 B/层，30 层共 15 KB）。实测体积见下表。

| 类 | gemma4（30 层 / 128 专家） | deepseek4（43 层 / 256 专家） |
| :-- | :-- | :-- |
| C2 全局 | 748.0 MB | 2020.3 MB |
| C1 按层（含 router） | 1711.2 MB（54.6~69.2 MB/层） | 9920.6 MB（209.0~245.6 MB/层） |
| C3 每专家 | 3.5 MB/专家（末层 4.1），总 13.4 GB | 12.8 MB/专家，总 137.1 GB |
| C4 专家小表 | 15 KB | 空 |

### 3.2 物理布局

```text
GGUF v3
├─ metadata
│   general.alignment = 4096
│   stream_moe.layout = "v3"
│   stream_moe.dense_global_section = [off, size]
│   stream_moe.dense_layer_sections = [layer, off, size, ...]   # 每层一条
│   stream_moe.expert_meta_sections = [layer, off, size, ...]   # 每层一条（可空）
│   stream_moe.expert_sections      = [off, size, nsub, ...]    # 每 (layer,expert) 一块
│   stream_moe.expert_branch_names / expert_branch_sizes / expert_branch_counts
│   stream_moe.branch_align = 1
├─ tensor_infos（含 `_exps.weight` 占位，route B 不读其数据）
└─ tensor_data
   ├─ C2 GLOBAL-DENSE
   ├─ C1 LAYER-DENSE（按层）
   ├─ C4 EXPERT-META（按层）
   └─ C3 EXPERT-BLOCKS（每专家紧凑块，块内分支 4K 对齐，见 §2.6）
```

- section 顺序固定；GGUF tensor offset 必须单调（v1 栽在这），按 C2→C1→C4→C3 顺序写即可。
- **4K 对齐硬不变量**：每个 section 内每个张量起点、每个 (expert,tensor) 切片起点、每个 chunk unit
  边界都 4K 对齐。
  - C2/C1/C4 张量整体读入 4K 对齐的 RAM/VRAM buffer（读 `align_up(size)`），**免 staging**。
  - C3 免 staging 额外要求 `perExpert % 4096 == 0`（目标槽 stride 被 ggml-vulkan 硬编码 = perExpert，
    不能补齐）。实测：deepseek 全命中（4456448）；gemma `gate_up` 2230272 = 544.5×4096 不命中
    （每专家尾 2 KB 仍需 staging），`down` 命中。故 C3 免 staging 是"条件成立时"，不是普遍保证。
- `_exps.weight` 的 tensor_info 仍写占位 offset（图构建要名字/ne/type；route B 从 `expert_sections`
  读数据），与 v2 同。
- C3 块布局 = v2（`branch_align=1`，SoA 列 stride = perExpert）。

### 3.3 分类规则（转换器，名字/结构代理）

转换器不能跑图，用与闭包一致的名字/结构规则代理：

| 类 | 规则 |
| :-- | :-- |
| C2 | 名字不以 `blk.` 开头 |
| C3 | `blk.*_exps.weight` 且 `ne[2] == n_expert`（否则报错，不静默切片） |
| C4 | `blk.*_exps.*` 非 `.weight`（专家索引小表，如 `.scale`） |
| C1 | 其余 `blk.*` |

> 注意：router（`ffn_gate_inp`）与 shexp（`ffn_*_shexp`）都不含 `_exps`，天然落 C1；`_shexp` 的
> 子串是 `_shexp` 而非 `_exps`，不会被误判进 C3/C4。

### 3.4 与 v2 的关系

v3 = v2 的 dense 段拆成 C2/C1/C4 三段（各自 offset 表），C3 不变。v2 → v3 是纯重排，可逆；
v3 → v2 把三段合回一段。

### 3.5 实现决策（2026-09）

1. **纯 C++ 转换器，消灭 JS/convertd/TCP**：复用 `src/loader/model.h` 的 `model_t` + `parse_model`
   （`model.h:9` 已为此预留），布局数学沉到共享模块，loader 与 writer 唯一一份。
2. **v1 彻底删除**：`V1_SECTIONS`、v1 writer、convertd v1 reflow、`scripts/convert_v1.bat`、矩阵 v1 列，
   以及 **`patches/gguf-alignment.patch` 整个删除**（`STREAM_MOE_GGUF_ALIGN` 只服务 v1 reflow）。
3. **v2 只保留单文件写入**（v2chunk 写删除）；v2 / v2chunk 的**读**保留到 v2 退役，现有源仍可转 v3。
4. **v3chunk 全切**：统一入口——每个 section = 一串 unit（C2 整段一个、C1/C4 每层一个、C3 每专家块一个），
   每个 unit 按同一条 4K base/rem 规则切 N 份，读侧按 unit 合并段。N 可按 section 配置（默认同一 N）。

### 3.6 仍待定（open）

1. **C4 复制策略**：加载时每设备各留一份，还是每设备图引用同一 host 副本（设备 staging）？
2. **per-layer 边界表格式**：`[layer, off, size]` 扁平，还是按层索引的边界数组（缺层用 -1）？
3. **是否把 C4 并入 C1 段**（都按层、都小），只靠 category KV 区分，省一个 section？

---

## 4. 分片（shard）下的转换

- 输入分片（`-00001-of-00005`）：转换器读全部 `splits`，合并张量视图，重排到目标（v2/v3）。
- 输出：单文件（v3 专家块分片 = v3chunk 可选）。

---

## 5. 转换器（stream_moe_convert）

- 输入：`-m <model.gguf>`（多分片自动合并；chunk 源用 `;` 分隔全部文件）。
- 输出：`-o <out.gguf>` + `--format v2|v3`（默认 v3）；`--format v3chunk --chunks N [--ratio a:b:c]` 时 `-o <base>`，产出 `<base>-00001.gguf`、`<base>-00002.gguf`……（数字宽度取自源文件名的末尾数字，缺省 5 位补零，超出自然变长）。
- 流程：
  1. `parse_model` 解析 GGUF（header/KV/tensor_info，张量 offset/size/type/ne）→ `model_t`。
  2. 分类张量（C1/C2/C3/C4，见 §3.3）。
  3. 重排写入目标（v2：源顺序 dense + 专家紧凑块；v3：C2/C1/C4/C3 四段；v3chunk：按 unit 切条带）。
  4. 写 `general.alignment=4096` + `stream_moe.*` 分区/块表。
  5. 校验：v2 对旧 JS 输出逐字节一致；v3/v3chunk 用矩阵不变量 + loader 数值回归（见 §10.4）。

## 6. 收益与代价

|  | v2（expert-blocks） | v3（四类 section） |
| :--- | :--- | :--- |
| 原版可读 | ❌ | ❌ |
| dense 整段读 | ✅ | ✅（C2/C1/C4 三段，各按策略） |
| 专家 DIO | 1 次/专家（紧凑块） | 1 次/专家（同 v2） |
| 异构支持 | ✅（块大小独立） | ✅ |
| 生态 | 自研 | 自研 |

---

## 7. 多文件：合分片，v3chunk RAID0 切分

> v1 张量级分片已随 v1 删除。合分片（`-00001-of-00005` → 单文件）由 `parse_model`
> 自动完成（读全部 `splits`，重排到目标）。

### 7.1 v3chunk：全 section unit RAID0 切分（跨盘）

- **unit**：每个 section = 一串 unit（C2 整段一个、C1/C4 每层一个、C3 每专家块一个）；每个 unit 独立用下面同一条规则切 N 份。v2chunk 同理（unit = [合并 dense] + [每专家块]）。

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
  - `stream_moe.layout = "v3"`（或 `"expert-blocks-v2"`）
  - `stream_moe.chunk_no` / `chunk_total`（自编号）
  - `stream_moe.incomplete = 1`（**注明无法被原版单独读取**——张量被切分、文件内不完整，原版读会错）
  - `stream_moe.chunk_slices`（每文件每个 unit 的 4K 块数）
  - metadata（KV）每个文件都有（含完整 tensor_info 表）；数据段按 unit 条带分布。
- **loader（common 侧）**：`parse_model` 接受多文件输入（`;` 分隔或按 chunk 编号），按 `chunk_slices` 合并各 unit 条带为多段 `src`。

---

## 8. 转换器架构（纯 C++，2026-09 落地）

- **读**（`src/loader/model_builder.cpp`）：`parse_model(paths)` → `model_t`（§10）。任意源：官方分片 /
  原版 / v2 / v2chunk / v3 / v3chunk。与 loader 同一解析器（单一事实来源）。
- **写**（`src/convert/writer.cpp`）：`convert_model(model_t, opts, out)`：
  - **v2**：dense 源顺序 + 每专家块（`branch_align=1`）。
  - **v3**：C2/C1/C4/C3 四段（§3）。
  - **v3chunk**：统一 unit 切分（每段 = 一串 unit，同一条 4K base/rem 规则切 N 份）+ 逐条带 copy/fill。
  - **4K 对齐**：从内存 seed 上下文（含 `general.alignment=4096` 的最小 GGUF）初始化 `gguf_context`，
    `gguf_add_tensor` 即按 4K 布局——**不需要 vendored `gguf_set_alignment`**（`gguf-alignment.patch` 已删）。
- **CLI**（`src/convert/main.cpp`）：`stream_moe_convert -m <model> -o <out|base> [--format v2|v3|v3chunk] [--chunks N] [--ratio a:b:c]`（v3chunk 的 `-o` 是基名前缀，输出 `<base>-00001.gguf`…）。
- **构建**：`build.bat convert <tag>`（CMake target `stream_moe_convert`，链 ggml-base + `model_builder.cpp`）。
- **已删除**：`tools/stream_moe_layout.js` / `stream_moe_convert.js` / `stream_moe_convertd.cpp` /
  `stream_moe_model.js` / `convertd_call.js`、`scripts/convert_v1.bat`、`patches/gguf-alignment.patch`。

---

## 9. 转换器功能清单【2026-09：目标态 = 纯 C++，v1 删除】

- [x] **读：任意源 → `model_t`**（官方分片 / 原版 / v2 / v2chunk / v3 / v3chunk）——复用 `src/loader/model_builder.cpp`。
- [x] **v2 expert-blocks 单文件写入**（分支 4K 对齐 + 块尾 0 填充）；C++ 与旧 JS 输出**逐字节一致**（gemma 实测）。
- [x] **C++ writer**（`model_t` → GGUF 直写，`src/convert/writer.cpp`）。
- [x] **v3 四类 section**（C2/C1/C4/C3 + category KV + 4K 对齐，见 §3）；v3→v3 逐字节幂等。
- [x] **v3chunk 全切**（统一 unit 切分/合并，见 §3.5）；v3chunk→v3 逐字节等于单文件 v3。
- [x] **KV 消毒**（只复制模型信息 KV，布局 KV 全新生成）。
- [x] **矩阵回归**（`scripts/verify_convert_matrix.bat`，C++ 转换器 + v3 不变量）——待整轮跑。
- [删除] **v1 sections / v1 张量级分片 / `gguf-alignment.patch` / JS+convertd**。

**结论**：转换器收敛为「读 = `parse_model` → `model_t`，写 = `model_t` + 落地参数 → 文件」，读写同一份
`model_t` 与布局数学；原版 / v2 / v2chunk 任意源 → v2 / v3 任意目标全部字节可复现。

---

## 10. 转换器中间格式（C++ `model_t`，读写共用）

> 目标：所有格式（原版 / v2 / v2chunk / v3 / v3chunk）都解析到同一个 C++ `model_t`，再从它写出任意目标
> （v2 / v3 / v3chunk）——矩阵全通。**v3 按 category 重排 dense**，所以 v3→v2 与 原版→v2 不是逐字节相同
> （dense 顺序不同）；可逆性判据是 `v3→v2→v3 == v3`、`原版→v2→v3 == 原版→v3`。

### 10.1 中间格式（`src/loader/model.h::model_t`）

```cpp
struct model_t {
    arch; layout; n_layer; n_expert; n_expert_used; incomplete; files; data_offs;
    dense:  [ { name, ne[4], type, size, category, layer, srcs:[src_seg_t] } ],  // 源顺序
    expert: [ { name, ne[4], type, size, per_expert, branch, layer, branch_off,
                per_expert_srcs:[ [src_seg_t] x n_expert ] } ],                   // 按 (layer, ORDER) 排序
    // v2/v3 布局 KV 原样：dense_section / expert_sections / branch_* / chunk_slices /
    //                    dense_layer_sections / expert_meta_sections / branch_align
};
struct src_seg_t { uint32_t file; uint64_t off, len, in_off; };
```

- `srcs` / `per_expert_srcs` 是段列表：单文件 = 1 段；chunk 源 = N 段（条带跨 N 文件）。
- `category`（C1/C2/C3/C4，见 §3.3）由 `sm_classify`（`src/loader/layout_math.h`）按名字/结构判定。
- `dense` 保持源顺序；写 v3 时按 category 排序（C2→C1→C4），写 v2 时保持源顺序。

### 10.2 解析器（各格式 → `model_t`，`parse_model`）

| 源格式 | 解析 |
| :--- | :--- |
| 官方分片 / 原版 | 张量列表 → dense/expert（`_exps.weight` 且 `ne[2]==n_expert`）→ `src` = file+off+size；`split.count>1` 自动合并分片 |
| v2 | `expert_sections` + `expert_branch_*` → 每专家块拆出分支区间 |
| v2chunk | `chunk_slices` → 区间 × 各文件条带求交 → 多段 `src`（全部 N 文件同时传入） |
| v3 | 同 v2 + `dense_global_section` / `dense_layer_sections` / `expert_meta_sections`；dense 按 category 分类 |
| v3chunk | 同 v3 + `chunk_slices`（unit = [global] + [C1 层] + [C4 层] + [block]）→ 多段 `src` |

### 10.3 写入器（`model_t` → 各格式，`src/convert/writer.cpp`）

| 目标 | 写入 |
| :--- | :--- |
| v2 | 源顺序 dense + 每专家块（`branch_align=1`）→ 分支间/块尾 0 填充 |
| v3 | C2/C1/C4/C3 四段 + 各自 offset 表 |
| v3chunk | 每 unit 按 4K base/rem 切 N 份 → 逐条带 copy + 补零 |

### 10.4 矩阵不变量（`scripts/verify_convert_matrix.bat`）

- `v2→v2 == 原版→v2`；`v2→v3 == 原版→v3`。
- `v3→v3 == v3`；`v3→v2→v3 == v3`；`v3chunk→v3 == v3`；`v3chunk→v2→v3 == v3`。
- **不变量**：`v3→v2 ≠ 原版→v2`（v3 重排 dense）——设计如此，不是 bug。
- 验证工具：`tools/cmp_gguf.js` 逐字节比较。
