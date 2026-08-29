# StreamMoE v2 转换器与运行时架构修正方案

> 状态：已审查落地（2026-08-29）。审查结论：DeepSeek 核对代码行号 + Claude 补充，方向一致——**P3 优先**。
> 关联代码：`tools/stream_moe_convertd.cpp`（v2 转换器）、`src/loader/moe_loader.cpp`（`build_v2_experts`）、`src/io/async_dio_win.cpp`（DIO）、`src/backend/moe_backend.cpp`（专家 buft）。

## 背景问题

1. **`_exps` offset 空洞**：v2 文件里 `_exps` 张量名义区留下大洞（gemma 实测 ~14GB，deepseek 按专家总量估算会更大，**以实际转换对比为准**）。
2. **bnames 越界**：loader 读 `expert_branch_names` 时只读出 3/90（定长除法假设每层 branch 数一致连续）。
3. **DIO 无 buffered fallback**：`CreateFileW(FILE_FLAG_NO_BUFFERING)` 失败仅 `LOG_ERROR`（网络盘/虚拟化磁盘直接失败）。
4. **`_exps` tensor->data 无防误用断言**：非 route B 路径摸到会静默读 0/垃圾。

## 根因（代码核对）

- `stream_moe_convertd.cpp:245-251`：`denseEnd` 遍历**所有张量（含 `_exps` 名义区）**取 offset 末尾；`:277` `cur = denseEnd` 作为 block 区起点 → block 在 `_exps` 虚假布局后写入 → 洞。
- **两套地址体系脱钩**：GGUF tensor_info 的 offset 是图构建占位符，与 Route B 的 Expert Storage（block 区）完全分离。`_exps` offset 不能参与物理文件布局计算。

## 布局修正（P3）

```
GGUF Header + KV Metadata
Tensor Info（含 _exps 占位 offset，图构建用，route B 不读数据）
Dense 数据（含 scale，真实写在 tensor_info offset）   <- denseDataEnd
Expert Blocks（blockStart = align_up(denseDataEnd, 4096)）
```

- **scale 确认**：`stream_moe_convertd.cpp:194-198` scale（`*_exps.scale`）进 **dense 桶**；`:335-340` dense 数据真实写 offset。消除洞后 block 从 denseDataEnd 开始，**对 scale 无冲突**。
- **修法**：`denseEnd` 改为**只统计 dense 组张量**（`dense` 容器），block 起点 = `align_up(denseEnd, ALIGN)`，复用 `_exps` 名义区，消除洞。

## bnames 元数据编码（P3）

- 弃用 `total / n_layer` 定长除法（假设每层 branch 数一致且连续）。
- 改为显式记录：`stream_moe.expert_branch_counts[n_layer]` 或 `pair<layer_idx, branch_name>`，兼容非均匀/非连续 MoE 层。

## 运行时防御（P1，绑定 mini-graph delegation 时间线）

- 对 `_exps` 张量：`tensor->data` / `ggml_backend_tensor_get` / `memset` 硬阻断（assert(false) / fatal log），杜绝静默返回 0 或垃圾。
- **不破坏 route B**：`moe_backend.cpp:45-48` `get_base` 返回哨兵指针、`memset_tensor` no-op——"reads slots, not tensor->data"。
- **约束（Claude 补充）**：P1 可以排在 P3/P4 之后，但**必须在 graph_compute 执行路径（mini-graph delegation）落地之前完成**，不能无限期漂在 backlog。

## DIO（P4）

- **扇区对齐不匹配**（扇区 > 4KB）：不退化，DIO 路径内向下取整 offset + 扩窗读取 + Staging Buffer 切出 payload。
- **介质不支持 DIO**（网络盘/虚拟化卷）：保留独立显式 Buffered Fallback 路径。两者不能混用同一套控制逻辑。
- **最低成本（顺手做）**：`async_dio_win.cpp:98-100` 创建失败时给出清晰报错（路径 + errno + "检查是否支持无缓冲 IO"提示），而不是裸 `LOG_ERROR`。

## 已确认现状（代码核对）

| 项 | 结论 |
|---|---|
| scale 在 dense 桶、真实写 offset | ✓（:194-198, :335-340） |
| moe_backend 不摸 tensor->data | ✓（哨兵 + no-op） |
| DIO 无 buffered fallback | ⚠️ 现有 CreateFileW(NO_BUFFERING) 失败仅 LOG_ERROR |

## To-Do（按优先级）

1. **P3**：消除洞（denseDataEnd 只算 dense）+ bnames 元数据编码（显式 per-layer counts）。
2. **P4 最低成本**：DIO 创建失败清晰报错。
3. **P1**：`_exps` 访问硬阻断断言（绑定 mini-graph delegation 动工）。
4. **Backlog**：expert_section 拆 `payload_size`/`physical_size`（SSD->VRAM/RAM 分层缓存用）、scale 独立分区评估、RAII 资源管理（`copy_from` seek 检查 + FILE*/gguf_context* 泄漏）。
