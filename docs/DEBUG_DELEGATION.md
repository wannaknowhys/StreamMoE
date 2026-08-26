# 委托数值 bug 调试记录 (DEBUG_DELEGATION.md)

> 2026-08-26 会话。问题：`--expert-backend` 下 L3+ 层 MoE 输出与基线分歧（L0-2 逐字节一致）。
> 方法：隔离复现（`temp/iso.cpp`，输入取自 trace + 文件权重）+ 内核 KDBG log 直写文件。

## 1. 确定没问题的（已实证，不再怀疑）

| # | 项 | 证据 |
|---|----|----|
| 1 | DIO 装载字节全等 | layer3 全部 256 专家切片校验和 == 文件 |
| 2 | 运行时槽内容正确 | s86-s115 全部 30 槽前 4 字节 == 文件；e235 全段校验和 `BCFB9B786148910B` 双端一致 |
| 3 | 路由 ids 一致 | base vs eb `#ids` 记录逐位一致（含 L0 与 L3） |
| 4 | gate 输入 cur 一致 | base vs eb `#cur` 逐字节一致；== ffn_norm-3；全 5 列 81920B 完整 |
| 5 | w3d 元数据 | off=0, nb1=2176, nb2=slot_size, pool 基址，type=39 |
| 6 | mm->nb == dst->nb | [4,8192,49152] 双端一致 |
| 7 | wdata 尺寸 | `ggml_graph_plan` 含 atomic_current_chunk，无溢出 |
| 8 | metadata 无关 | B==C 组输出逐位相同（ne2=256/stride 与 ne2=5621/slot 无差别） |
| 9 | 非竞态 | 单线程复现相同错误 |
| 10 | 隔离复现 | iso 独立程序复现：L0 gate 5 token 全对，L3 gate t0 对 t1-4 错 |
| 11 | 内核读偏移正确 | KDBG: cur_a=217 rm={0,1} src1_col_off=4352（t1 的 Q8_0 列）；dst_off=49152 |
| 12 | 内核读的 src1 字节正确 | Q8_0(cur col1) 实算 `C8 1E CE F6 CB FE 10 EF` == 内核读到；col0 同理 |
| 13 | 内核读的权重字节正确 | wb=`793A2B22C9E143D9` == e217 文件字节；e235 同理 |
| 14 | 转换环节正确 | 内核 F32→Q8_0 输出 == ggml_quantize_chunk；PRECONVERT_Q8 预转后仍错（排除转换） |
| 15 | 写回位置正确 | iso 输出 t1 k0 == 内核 dot tmp（-0.3502 写入 dst_off=49152），与 base 期望 -0.8902 不符 |

## 2. 确定有问题的（实证定位）

| # | 项 | 证据 |
|---|----|----|
| 1 | **L3 gate/up t>=1 输出数值错误**（t0 正确；t1/t3 的 k5 例外） | 隔离 + live 均复现；每 (k,t) 唯一专家 cne1=1 |
| 2 | **错误发生在 vec_dot 计算，不是写位置** | 内核 dot tmp=-0.3502 与期望 -0.8902 不符，而输入全部字节验证正确 |
| 3 | L0 正确 / L3 错误，同代码同数据正确 | 差异只在数据值本身，指向数据相关的 SIMD 行为 |

## 3. 调用链与 log 计划（按函数进出打 log）

```
ggml_backend_graph_compute(cpu, gf)                 [iso 入口]
  -> ggml_compute_forward_mul_mat_id                 [KDBG enter 已打: src1_cont/ne/nb/type/vdt/rowsize]
     -> wdata 布局分配                                [未打]
     -> src1 F32->Q8_0 转换循环                       [未打: 打转换后 wdata 首块字节]
     -> matrix_rows 构建 (ith==0)                     [未打: 打 e217/e235 的 {k,t} 条目]
     -> chunk 循环 -> one_chunk                       [KDBG chunk 已打: rm/src1_col_off/src1b/wb]
        -> vec_dot = ggml_vec_dot_mxfp4_q8_0          [x86 AVX2, 未打: 打 x/y 首块字节 + sumf]
```

打 log 位置：
- `ggml_vec_dot_mxfp4_q8_0`（ggml-cpu/arch/x86/quants.c）：入口打 x/y 首块字节、退出打 sumf。
- `ggml_vec_dot_mxfp4_q8_0_generic`（ggml-cpu/quants.c）：同样打（对照）。
- 主函数转换循环后：打 wdata 首块字节（验证 == Q8_0 实算）。
- matrix_rows 构建后：打 e217/e235 的条目。

## 4. 下一步（待验证假设）

- vdot.cpp 探针：真实权重行 + 真实 Q8_0 列，直接对比 x86 vs generic vec_dot，看哪个 == 期望。
- 若 x86 错 generic 对 -> x86 AVX2 的 `_mm_srli_epi16(q4bits,4)`（字移位而非字节移位）嫌疑最大。
- 同时验证 base 是否真的走了同一 x86 函数（若 base 对，则 base 的输入或路径有别的差异）。

## 5. 全量 dump 结果（2026-08-26，moe 侧）

工具：`dump_moe_node()`（minigraph_exec.cpp，STREAM_MOE_TEMP 下）写 `temp/moedump.bin`（MOEDUMP1 二进制）+ x86 vec_dot 写 `temp/vdotdump.bin`（前 256 次完整 x/y）。解析：`temp/nodedump.js`。

**Dump 到的记录**：L0 gate/up/down + L3 gate（4 条；L3 up/down 未入 dump，后续补）。

**验证结论（moe 侧 vec_dot 全部输入正确）：**
| 项 | 结果 |
|----|------|
| L3 gate 槽内容 == 文件 | e235→s86 FNV=`BCFB9B786148910B` 与此前一致；30 槽全部正确 |
| ids/idslot | L3: e235→s86, e217→s92, ..., e75→s115 全对；L0: 槽 0-29 |
| cur | L3 col0=-0.2272... 与 base trace 逐值一致 |
| w3d | off=0, nb2=13369344, pool=0x2019da70000, type=39 |

**闭环结论**：moe 侧 vec_dot 输入（槽内容 + cur Q8_0 + ids）全部 == 文件/trace 真值，且 vdot 探针证明这些字节的数学结果 = -0.3502（iso 值）≠ base -0.8902。→ **base 的 L3 vec_dot 输入必与这些字节不同**（唯一待确认项）。

## 6. 待完成的关键实验

- **base 在 L3 的 (k=0,t=1)（专家 217）vec_dot 输入 dump**：用 one_chunk 的 ALIGN 调试（cur_a==217）跑 base，对比 xb/yb 是否 == moe 槽 92 内容（FNV=17D7EC6C704682AE）与 Q8_0(cur col1)。若不同 → 找到 base 输入来源差异（repack？）；若相同 → 需复查 base 的 dst 写回或 trace 值。

## 7. ✅ 根因确认（2026-08-26，最终定位）

**根因：`minigraph_exec.cpp` 的 delegate 用紧凑索引读取路由 ids 张量，但 argsort 层（L3+）的 ids 张量 stride 是非紧凑的 `nb[1]=1024`。**

### 证据链

1. **base 的 L3 gate 真实 ids 是 stride 1024 布局**（`trace_base3.bin` 的 `blk.3.ffn_gate_exps.weight#ids`，nbytes=4120）：
   - t0（offset 0-23）: `235,117,129,68,47,71`
   - t1（offset 1024-1047）: `208,90,71,255,6,128`
   - t2（offset 2048-2071）: `174,230,29,90,59,246`
   - t3（offset 3072-3095）: `208,84,71,134,111,6`
   - t4（offset 4096-4119）: `235,129,208,119,68,71`
   - offset 1040（t1 k4）= 6，与内核 `KDBG EXP6 rm={4,1}` 完全吻合（base 内核按 `nb[1]=1024` 读，正确）。

2. **moe 的 delegate 用紧凑索引 `ids_data[t*n_ids+k]` 读取** → t≥1 读到 offset 24-119（t0 行 padding 里的假值）：
   - t1 读到 `217,161,143,176,118,90`（假值，≠ 真实 `208,90,...`）
   - t0 读到 `235,117,...`（两种布局 offset 0 重叠，碰巧正确）

3. **后果**：moe 把假专家号翻译成槽 → mini-graph 用**错误专家**计算 → t≥1 输出错。槽装载本身正确（槽内容 == 文件），错在"选了错误的专家"。

4. **L0 vs L3 差异解释**：L0-2（hash 层）ids 张量紧凑（nbytes=120, nb[1]=24）→ 紧凑读正确 → L0-2 正常。L3+（argsort 层）ids 非紧凑（nbytes=4120, nb[1]=1024）→ 紧凑读 t≥1 错。

5. **t0 对 t1-4 错解释**：t0 的 offset 0-23 在两种布局下相同 → t0 永远正确。

### 修复

delegate 读 ids 必须按张量真实 stride：
```cpp
int32_t e = *(const int32_t*)((const char*)ids->data + (size_t)t * ids->nb[1] + (size_t)k * ids->nb[0]);
```
（替换 `ids_data[t * n_ids + k]`，同位置还有 gate/up/down 的遍历与 ids_slot 翻译。）

## 8. ✅ 修复实施与验证（2026-08-26）

### 改动
- `src/backend/minigraph_exec.cpp`：新增 `MOE_ID_AT(ids,t,k)` 宏（按 `ids->nb[1]` 真实行步长读元素）；phase 1（pin 解析）、phase 2（ids_slot 翻译）、`[ids]` 调试、`dump_moe_node`、ABC 实验块全部改用它。gate/up/down 三处统一。

### 运行时验证（修复后 delegate 读到的 L3 gate ids）
```
[ids] blk.3.ffn_gate_exps.weight nb1=1024:
  t0 = e235,e117,e129,e68,e47,e71
  t1 = e208,e90,e71,e255,e6,e128      <- 修复前误读 e217,e161,...
  t2 = e174,e230,e29,e90,e59,e246
```
t1-t4 与 `trace_base3.bin` 的 stride1024 布局（offset 1024/2048/3072/4096）逐位一致。

### 端到端测试
`--expert-backend --temp 0 -p "Say hi." -n 8`：
- 修复前输出：`## Final answer\nSay hi.`（错误）
- **修复后输出：`Hi! How can I help you today`（与基线逐字一致）✅**

### 遗留说明
- 诊断代码仍在：`arch/x86/quants.c` / `ggml-cpu.c` 的 KDBG + vec_dot 逐行 detail（均 `#ifdef STREAM_MOE_KERNEL_DBG`，仅 kdbg-build 编译，不入主构建）；`minigraph_exec.cpp` 的 `[verify]` 打印（`STREAM_MOE_TEMP` 下）与 `dump_moe_node`（L0/L3 dump）。上下文压缩后清理。
- 数值等价回归（全命中/混合 vs 官方图逐元素 diff）待后续跑。



