# 设计记录矛盾与待确认问题 (DESIGN_REVIEW.md)

> 对照：Backend.md / EXPERT_OFFLOAD_INTEGRATION.md / LLAMA_MOE_NO_MMAP_RESEARCH.md / TODO.md / README(.zh-CN).md / BUG_TRACKER.md / 实际 llama.cpp 代码 @ f280b2698。
> 每条给：原文出处 → 矛盾/疑点 → 建议。

---

## §1 槽数据布局：Backend.md 与 llama.cpp 图需求不兼容 ⚠️ 最重要

> ✅ **已定案（2026-08-26 落地）**：最终采纳**第三路径**——槽池 = 单块连续内存 + 均匀 stride（`nb[2]=slot_size`），每个 MUL_MAT_ID 用官方 `ggml_mul_mat_id` 内核直接执行（叶子 `[ne00,ne01,num_slots]` + ids 槽号翻译 + b_leaf 包装）。槽 = stride 区域三元组（gate/up/down 三个 branch offset），装载 = 3 次 DIO。`expert_directory`/`slot_meta`/`refcount` 控制面原样成立。详见 `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7。

- **Backend.md §3**：槽 = 每个专家 gate+up+down **紧凑拼接**（与旧 StreamMoE `moe_loader` 的 `expert_slot_size` 布局一致）。
- **llama.cpp 现实**（LLAMA_MOE_NO_MMAP_RESEARCH §3.1）：`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` 是**三个独立连续 3D 张量**，各含全部 256 专家；图内核按 `e*nb[2]` 索引。自定义 buft 只能提供"连续大张量"，无法提供"每专家紧凑三合一"。
- **后果**：Backend.md 的 slot 语义必须重定义为"**三层三区域的 (gate[e], up[e], down[e]) 逻辑三元组**"，装载 = 3 次并行 DIO 到 3 个区域偏移。旧 `moe_loader` 的紧凑 read plan 废弃，需新布局。
- **问题**：确认接受此布局修正？`expert_directory`/`slot_meta`/`refcount` 控制面（Backend.md §4）可在新布局上原样成立，仅"slot"指三元组。

## §2 Backend.md 要求自定义 ggml_backend，研究显示 CPU-only 可更薄

> ✅ **已定案（2026-08-26 落地）**：route B 采用自定义 backend（`src/backend/`：moe_backend + minigraph_exec + scheduler），CPU-only 阶段即走自定义 backend 路线；GPU 混合池为 Phase B。

- **Backend.md §1**：必须实现自定义 backend 的 `supports_op(MUL_MAT_ID)` + `graph_compute`，以支持"同一次调用 CPU/GPU 双池并发"。
- **LLAMA_MOE_NO_MMAP_RESEARCH §1.6**：CPU-only 阶段只需**自定义 buft + cb_eval 钩子**，CPU backend 直接消费 host buffer，**无需自定义 backend**；自定义 backend 只在 GPU 混合池阶段（Backend.md Phase A 目标）才需要。
- **问题**：是否接受分两段——先落地 buft+cb_eval（CPU-only，较快，可立即跑基准），GPU 混合后端后置？

## §3 Backend.md 的 GPU 等待原语是 CUDA 专属，本机是 Vulkan

- **Backend.md §1**：GPU 组异步发射"优先用 `cuStreamWaitValue64` 之类的流内内存等待"；§5 又写"Phase A 以 CPU dispatch 线程 park+转发为基线"。
- **现实**：本机 RX 590 8GB 只有 Vulkan（无 CUDA）；Vulkan 无 `cuStreamWaitValue64` 等价物。
- **建议**：确认 Phase A 统一用 dispatch 线程 park+转发；`cuStreamWaitValue64` 标记为 CUDA 后端专属优化（Phase C，ReBAR 一并记录）。Backend.md 措辞建议加"CUDA 专属"限定。

## §4 "不做物理载入"与 llama.cpp 加载路径的衔接已被研究证实，无需改 loader

- 早前你问"是不是没有 loader 改动"；研究（§1.2-1.4）确认：`tensor_buft_overrides` 指向非默认 buft → 自动走真实分配 + `set_tensor` 路径 → 我们的 no-op `set_tensor` 即跳过物理读入。**零 loader 源码改动**。
- 需确认：是否接受用 `cb_eval`（公开 context 参数）承担就绪等待，而不用 Backend.md §4 的 MPSC 分配请求队列（二者可共存：cb_eval 只是挂载点，内部仍可用 MPSC）。

## §5 旧 `expert_pool` / `moe_scheduler` 的保留价值

- BUG_TRACKER 把 B14-B17/B21/B24/B25 标为 SUPERSEDED（由 Backend.md 控制面替代）。研究 §3.3 显示 `expert_pool`/`moe_scheduler` 的驱逐/热度/单 worker 调度逻辑可复用，仅存储布局改为大区域分片。
- **问题**：是重构复用旧模块，还是按 Backend.md §2-§5 全新实现（slot_meta/expert_directory/MPSC）？建议：新布局下全新实现控制面，旧模块仅参考算法。

## §6 README 宣称与现状不符

- **README(.zh-CN).md**：声称"Instant Engine Startup (< 0.15s)"、已可跑 150GB+ 模型。实测：真实引擎 mmap 冷启动 1-6s、私有池方案后模型不整体加载但首次推理受冷页/冷 DIO 影响。0.15s 是 mock 时代的数字。
- **建议**：README 改为"秒级启动 + 专家按需流式"，或注明目标值。

## §7 速度归因与基准声明

- BUG_TRACKER 性能记录将慢速归因于 iSCSI，实际是 USB 转接 NVMe（已更正）。但仍实测 decode 0.3~2 tok/s。
- **疑点**：USB-NVMe 理论带宽足够，0.3~2 tok/s 是否还含其他开销（专家 slice 大小 ~13MB × 8 × 43 ≈ 4.4GB/token 冷读，NVMe 顺序也需 ~5-10s？）。私有池 + DIO 后应重点测 IO 吞吐与命中率，建议基准时记录每 token 冷读字节数。

## §8 TODO.md Phase 4 测试矩阵与本机硬件约束

- TODO 4.1-4.6（RAM 池容量扫描、KV RAM/VRAM、QD 扫描、投机步长、4KB 对齐、多槽并发）部分依赖 GPU（KV vram 对比）与多实例。本机 Vulkan-only 且 8GB。
- **问题**：GPU 相关条目是否全部推迟到 Vulkan 后端接入后？本轮（CPU-only 私有池）先做 4.1（RAM 池容量 vs 命中率/TPS 曲线）即可？

## §9 子模块升级策略

- vendored llama.cpp 锁定 f280b2698。扩展点（tensor_buft_overrides / cb_eval / buft iface）均为该版本签名。
- **问题**：是否允许周期升级子模块并在升级时跑数值等价回归？还是本轮完全冻结？

## §10 与 EXPERT_OFFLOAD_INTEGRATION.md 的张力

- 该文档 L0（mmap 基线）被标记为"可用基线"，与用户"消灭 mmap"终态目标不同。L0 现仅作对照组/性能参照，不构成最终形态——文档措辞建议补一句"L0 非终态"。

---

## 待你拍板的事项（按优先级）

1. **§1 布局修正**：池 = 三层三连续区域，槽 = (gate[e], up[e], down[e]) 三元组。接受？
2. **§2 分两段**：先 CPU-only（buft+cb_eval，无 llama.cpp 改动），再 GPU 混合自定义 backend。接受？
3. **§3**：GPU 等待原语确认 Vulkan 用 dispatch 线程 park+转发，CUDA stream-wait 后置。
4. **§4**：就绪等待挂 cb_eval（内部可走 MPSC）。接受？
5. **§5**：控制面全新实现（Backend.md §2-§5），旧模块仅参考算法。接受？
6. **§8**：GPU 相关基准推迟到 Vulkan 接入后；本轮先跑 RAM 池容量曲线。
7. **§9**：子模块本轮冻结升级。
