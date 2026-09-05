# VRAM DMA 迁移管线（transfer queue 上的 demote / r2v）

[English](VRAM_DMA_MOVE.md) | [简体中文](VRAM_DMA_MOVE.zh-CN.md)

> 状态：**已实现 + 已验证，2026-09**。工作会话实测了 RX590 的 transfer 路径、
> 敲定方案并随 M4 move worker 落地（commit 816f8aa / 717bac8；各步状态见 §5）。
> 建立在 `docs/EXPERT_MOVE_PIPELINE.md`（M4 异步 move worker + (L,E) 键驱逐）与
> VRAM 数据层（docs/WORK_IN_PROGRESS.md J 节）之上。注意：本工作**推翻了**
> EXPERT_MOVE_PIPELINE §5.2 的"v2r 只能 CPU memcpy"结论——见该节修订框。

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

`expert_scheduler` 是纯 C++（无 ggml/vulkan）。它不该学 vulkan 类型。落地实现把
DMA 读导出为**一个全 void 参数的普通函数**，由 `ggml-vulkan.cpp` route-B 锚点
include 的 frag 定义（锚点处内部 vulkan 类型与 TU-static `ggml_vk_buffer_read`
可见）。scheduler 以 `extern` 声明并直接调用；任何 vulkan/ggml 类型都不跨界。

```
ggml-vulkan.cpp（vendored，已 patch）
   + route-B 锚点 include stmoe_routeb_vk_dma.frag
      导出: stmoe_vk_dma_read(void*, size_t, void*, size_t)
            stmoe_vk_dma_available()
expert_scheduler.move_worker   （控制面不变）
   |  每个 move 任务的每个列:
   |    src_pool != 0 && copy.dev_buf -> stmoe_vk_dma_read(dev_buf, dev_off,
   |                                                        dst, bytes)  [v2r]
   |    否则                        -> 普通 memcpy（RAM 源；r2v 仍 DIO 直写——
   |                                  host-map 写本身快）
   |  分阶段报 rdtsc delta
```

- frag 包装 ggml-vulkan 内部 `ggml_vk_buffer_read`（锚点上方 static）：同步
  transfer-queue `vkCmdCopyBuffer` 到 ggml 的 HOST_CACHED staging，再 memcpy 到
  `dst`。staging buffer 与 fence wait 归 ggml-vulkan 管——route B 不自建。
- scheduler 控制面（submit_move / 状态机 / drain_moves）**不变**。
  `move_task_t::copy_t` 只新增 `dev_buf`（void* 设备区句柄）+ `dev_off`（区内
  字节偏移），submit 时由区的 vk buffer 填入（scheduler.cpp:257）。
- **worker 线程保留**（仍发 DMA + 做 fence wait / 任何 staging 拷贝）；只是不再
  读慢的 rebar map。

### 4.2 Staging

Staging 是 ggml-vulkan 内部 `sync_staging`（HOST_CACHED，heap0 / memtype 7）。
每次 `dma_read` 一次同步提交：transfer-queue 拷 vram → staging → memcpy staging
→ `dst`，每 call 一次 fence wait。

早期设想的"固定多槽 staging ring（~10 槽，轮转复用）"**没有做**：实测 ~0.5-1
ms/专家（§5）已比 rebar 读快 ~150-300×，move worker 不再是 decode 瓶颈。流水式
staging ring（多个 copy 在飞）只摊薄每次 call 的 fence wait；等 move worker 再上
profile 再说。

### 4.3 导出接口（frag 侧）

```cpp
// patches/route-b/common/stmoe_routeb_vk_dma.frag - 在 ggml-vulkan.cpp route-B
// 锚点 include；包装 TU-static ggml_vk_buffer_read。
void stmoe_vk_dma_read(void * buffer /* ggml_backend_buffer_t */, size_t off,
                       void * dst, size_t bytes); // 同步
bool stmoe_vk_dma_available(void);                // 仅诊断
```

调用侧 `buffer` 是不透明句柄（vram 区的 `ggml_backend_buffer_t`，以 `void*`
存在区上并拷进 move 任务）。无需 moe_backend/route_b 注册对象——符号与区 buffer
同在一个 ggml-vulkan TU，直接链接。

### 4.4 计时（rdtsc，STREAM_MOE_LOG=debug 门控）

每个 move 记：
- `dma_us` = vkCmdCopyBuffer submit → queue idle（下载时间）
- `mc_us`  = staging → RAM 槽拷贝时间（内部 staging 路径下为 0——已含在 dma_read）
- 加既有 queued 总时间

据此可重新评估"删掉 memcpy worker"：若 `dma_us + mc_us` 很小、worker 不再是
瓶颈，worker 可能并入调度线程（在那 submit + fence wait）。当前 ~1 ms/专家，
说明暂仍留 worker。

## 5. 构建顺序

1. 为 route B 包装 ggml-vulkan 内部 device→host 路径：route-B 锚点的 frag 导出
   `stmoe_vk_dma_read`（包装 TU-static `ggml_vk_buffer_read`）。**[done - 816f8aa]**
2. scheduler：move worker 对设备源用 DMA、RAM 源保持 memcpy；加分阶段 rdtsc 日志。
   **[done - copy_t.dev_buf/dev_off + worker DMA 分支 + dma/mc 计时 - 717bac8]**
3. 用现有 VRAM 配置验证（RAM 1GB + Vulkan0 64MB）：demote 风暴应从 ~158
   ms/专家降到 ~0.5 ms；之前几乎不可用的 decode 应变得可用。
   **[done - 4-token 64MB：238 次 DMA demote ~0.9-1.5 ms（首个 ~10 ms = staging
   懒初始化）；129-token 64MB 跑通（~14 s，冷 DIO + CPU 计算主导）]**
4. 回归：纯 RAM 路径必须逐字节 IDENTICAL（RAM 源 dev_buf==null → 纯 memcpy，
   路径不变）。重跑 mixed 执行数值门。
   **[done - 4-token DMA demote 遍 vs 纯 RAM golden：64 列，0 BAD（22 bit / 42
   ulp，符合 mixed CPU/GPU 切分预期）]**
5. r2v 无需改动：vram host-map 写已 ~8 GB/s（DIO 直载路径），DMA 只管 v2r。
   **[done - r2v 仍 DIO 直写]**

## 6. 涉及文件

- `patches/route-b/common/stmoe_routeb_vk_dma.frag` - 新增：导出
  `stmoe_vk_dma_read` / `stmoe_vk_dma_available`；在 vendored ggml-vulkan.cpp
  的 route-B 锚点 include（记录于 `streammoe-macros.patch`，一行 include）。
- `src/backend/scheduler.h` / `scheduler.cpp` - `move_task_t::copy_t` 与
  region.dev_buf 新增 `dev_buf`/`dev_off`；move worker 对设备源分支到
  `stmoe_vk_dma_read`（extern 声明，全 void 参数）；dma/mc rdtsc 计时（STREAM_MOE_LOG=debug 门控）。
- `docs/WORK_IN_PROGRESS.md`、`docs/CHECKPOINT.md` - 状态（M 节、当前状态）。
- 测量探针（temp/，仅本地）：vram_bw / vram_heaps / vram_dma / demote10 /
  rebar_write——§1-§2 数字来源。
- 新 UT（可选，未写）：staging 槽生命周期——后置（未建专用 staging ring，§4.2）。
