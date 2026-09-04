# VRAM DMA 迁移管线（transfer queue 上的 demote / r2v）

[English](VRAM_DMA_MOVE.md) | [简体中文](VRAM_DMA_MOVE.zh-CN.md)

> 状态：**设计定稿，2026-09**。工作会话实测了 RX590 的 transfer 路径并敲定方案；
> 实现随后进行。建立在 `docs/EXPERT_MOVE_PIPELINE.md`（M4 异步 move worker +
> (L,E) 键驱逐）与 VRAM 数据层（docs/WORK_IN_PROGRESS.md J 节）之上。

## 1. 问题

Demote（vram→RAM 专家驱逐）是 decode 瓶颈。此硬件（Radeon RX 590，独显，rebar
BAR1）上 **从 vram host-map 做 host 侧 memcpy 读只有 ~0.02 GB/s**——单个 gemma
专家（3.63 MB）要 ~158 ms，而带 vram 压力的 prefill/decode 一遍要做几百次
demote。

按内存类型实测 host 读带宽（直连 vulkan 测试）：

| 内存类型 | heap | host 读 | 备注 |
|---|---|---|---|
| DEVICE_LOCAL \| HOST_VISIBLE (rebar) | vram 8 GB | **0.02 GB/s** | 现 move 源——不可用 |
| HOST_VISIBLE \| COHERENT | 系统 RAM 64 GB | 0.27 GB/s | 非 cached |
| HOST_VISIBLE \| COHERENT \| **CACHED** | 系统 RAM 64 GB | **21-27 GB/s** | ggml sync_staging heap |

## 2. Transfer-queue DMA 是解法

在带 transfer 能力的队列（RX590 有纯 transfer 队列族）上 `vkCmdCopyBuffer`
把 vram 拷到 host buffer **~14 GB/s**，随后 CPU 以 cached 速度读该 host buffer：

| 路径 | 3.63 MB 专家 |
|---|---|
| 从 vram rebar map CPU memcpy（现状）| ~158 ms |
| `vkCmdCopyBuffer` vram → CACHED staging | ~0.38 ms |
| + memcpy CACHED staging → RAM 槽 | ~0.17 ms |
| 总计（DMA + memcpy）| **~0.5 ms（快 ~300×）** |

一块固定 staging buffer 上 10 专家并发**无带宽损失**：逐专家顺序提交 9.5
GB/s、10 条 copy 一次提交 10.7 GB/s、流水 10 次提交 10.2 GB/s（都 ~0.35-0.39
ms/专家）。

## 3. 否决的替代（已实测）

- **整块 RAM 池 import 成 vulkan 内存**（VK_EXT_external_memory_host）：驱动每
  进程只接受**一块 ~2 GB** host import；10×2 GB 普通分配也在第一块后失败。故专家池
  无法做成多块 vulkan buffer（其 host map 无法落在快 heap 上）。
- **直接 vkCmdCopyBuffer 进普通 malloc 池槽**：需要池内存是 vulkan buffer
  （import，被上面否决）。

Host-map 带宽是**不对称的**：vram rebar CPU **写 ~8 GB/s**（PCIe posted write）
但 CPU **读 ~0.02 GB/s**。后果：
- r2v（RAM/磁盘字节载入 vram = CPU 写 vram host-map）**不需要 staging**——现状
  DIO 直写路径已跑在快的写速度上。
- 只有 v2r demote（把 vram 读回 RAM）慢；这是唯一需要下面 DMA/staging 修复的路径。

结论：**一块固定 2 GB CACHED staging buffer**（heap0 / memtype 7）作 **v2r 专用**
DMA 跳板，按 ~10 专家槽位定尺以支持并发，再快速 memcpy 进普通 RAM 槽。RAM 槽寻址
（base + 偏移）不变。

## 4. 设计

### 4.1 分层（保持 scheduler 无 ggml）

`expert_scheduler` 是纯 C++（无 ggml/vulkan）。它不该学 vulkan。设备侧 DMA
服务由 route B / moe_backend 持有（它们握着 vulkan device/queue/buffers），
以普通函数指针 + 不透明 ctx 的形式注入 scheduler（挂在 model pool 上）。

```
route_b_inject (vulkan device/queue/buffers)
   |  提供: vram_dma 服务（函数指针 + 不透明 ctx）
   v
expert_scheduler.move_worker   （控制面不变）
   |  每个 move 任务:
   |    1. dma_service->download(vram_buf, src_slot_off, bytes, staging_slot_ptr)
   |       = vkCmdCopyBuffer(vram_buf -> staging buf 的槽位)  [transfer queue]
   |    2. memcpy(RAM 槽 dst, staging_slot_ptr, bytes)        [worker CPU]
   |    3. 分阶段报 rdtsc delta
```

- scheduler 控制面（submit_move / 状态机 / drain_moves）**不变**。
- move_task 保留 per-column (src,dst) 范围；worker 对设备源改走 DMA 服务，不再
  memcpy rebar map。
- **worker 线程保留**（仍做 staging→RAM memcpy + fence wait）；只是不再读慢的
  rebar map。

### 4.2 Staging 布局

- 一块固定 CACHED staging buffer，~10 个专家槽位（gemma 专家 3.63 MB →
  ~37 MB），若模型专家更大则在驱动 2 GB 窗口内按需增长。
- 槽 stride = expert_size（紧凑）。worker 轮转使用 staging 槽；只有该槽的
  memcpy 到 RAM 槽完成后才复用，保证一个槽不会有两个 in-flight DMA 写。

### 4.3 DMA 服务接口（moe_backend 侧）

```cpp
// 由 moe_backend/route_b 持有；每个 model pool 注册一次
struct vram_dma_ctx;  // opaque
using vram_dma_download_t = bool (*)(vram_dma_ctx* c, uint32_t pool,
                                     void* vram_buf, size_t vram_off,
                                     uint8_t* staging_dst, size_t bytes);
```

scheduler 在 model pool 上存 `{ctx, fn}`；`move_worker_main` 在 `src_pool != 0`
时调 `fn`，RAM 源回退普通 memcpy。

### 4.4 计时（rdtsc，STREAM_MOE_LOG=debug 门控）

每个 move 记：
- `dma_us` = vkCmdCopyBuffer submit → queue idle（下载时间）
- `mc_us`  = memcpy staging → RAM 槽（搬迁时间）
- 加既有 queued 总时间

据此可重新评估"删掉 memcpy worker"：若 `dma_us + mc_us` 很小、worker 不再是
瓶颈，worker 可能并入调度线程（在那 submit + fence wait）。等数字落地再定。

## 5. 构建顺序

1. moe_backend：首次使用时建 cached staging buffer + transfer queue；暴露
   `vram_dma_download`（vkCmdCopyBuffer + fence）。
2. scheduler：在 model pool 注入 DMA 服务；move_worker 对设备源用它、RAM 源
   保持 memcpy；加分阶段 rdtsc 日志。
3. 用现有 VRAM 配置验证（RAM 1GB + Vulkan0 64MB）：demote 风暴应从 ~158
   ms/专家降到 ~0.5 ms；之前几乎不可用的 decode 应变得可用。
4. 回归：纯 RAM 路径必须逐字节 IDENTICAL（无 DMA 服务 → 纯 memcpy）。重跑
   mixed 执行数值门。
5. r2v 无需改动：vram host-map 写已 ~8 GB/s（DIO 直载路径），DMA 服务只管 v2r。

## 6. 涉及文件

- `src/backend/scheduler.h` / `scheduler.cpp` - 注入 DMA 服务；worker 路径
- `src/backend/moe_backend.h` / `moe_backend.cpp` - cached staging buffer +
  transfer queue + `vram_dma_download`
- `src/server/route_b_inject.cpp` - 每池创建并注册服务
- `docs/WORK_IN_PROGRESS.md`、`docs/CHECKPOINT.md` - 状态
- 新 UT（可选）：staging 槽生命周期
