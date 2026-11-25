# StreamMoE 后端审计：CPU / Vulkan / CUDA 三后端重构

审计日期 2026-11-24，基于 vendored llama.cpp HEAD `f280b2698` + 工作区已 apply 的 3 patch 栈。

## 0. 结论先行

- 是，现在要做的就是这个。P100 定在 TCC 模式后，Vulkan 在这台机器上走不通（TCC 下无图形 API），P100 的唯一 GPU 路是 CUDA；CPU 要保持可用（开发/数值对照）；Vulkan 要保留（RX 590 和以后可能切 WDDM 的机器）。所以重构成 cpu/vulkan/cuda 都支持是对的。
- 好消息：上游侵入比看上去小得多。`git diff --stat` 显示 57k+/57k- 全是换行符 churn，忽略 CRLF 后真实改动只有 23 个文件、337 行新增、16 行删除。而且涉 Vulkan 的上游改动只有 1 处锚点（7 行）+ 2 个 frag。
- 坏消息在自有代码：Vulkan 耦合集中在 3 个文件（minigraph_exec、moe_backend、route_b_inject）+ 1 个 args frag，核心是 `stmoe_vk_buffer_host_offset` 的 fake-base 指针约定，这是 Vulkan 专属的，CUDA 不能照搬。

## 1. 上游改动全貌（third_party/llama.cpp，HEAD f280b2698）

机制见 `patches/README.md`：vendored 永不 commit，改动以 3 个 patch + frag 记录，按序 `git apply`：`streammoe-macros.patch`（phase 1 锚点）→ `tsc_timer.patch` → `route-b-inject.patch`（phase 2a，只剩改既有逻辑行的部分）。frag 全部在主仓库 `patches/<phase>/`，经 features 宏（`STREAM_MOE_ROUTE_B` / `STREAM_MOE_PREFILL_EXPORT`）+ 全局 include 路径展开。

真实 diff（`--ignore-cr-at-eol`）：23 文件，+337/-16。

| 文件 | +行 | 类别 | 涉 Vulkan？ |
|---|---|---|---|
| `CMakeLists.txt` | +19 | features 宏块（全局宏 + frag include 路径） | 否（但 CUDA 要在这里加门控分支） |
| `common/CMakeLists.txt` | +3 | include `src/cmake/stmoe_routeb_sources.cmake` | 否 |
| `common/arg.cpp` | +16 | route-b/prefill 参数锚点 + 追踪/placement 检查 | 部分（placement 字符串见 §3） |
| `common/common.cpp` | +13 | 锚点 | 否 |
| `common/common.h` | +7 | 锚点 | 否 |
| `common/preset.cpp` | +3 | 锚点 | 否 |
| `common/speculative.cpp` | +12 | draft 池/统计锚点 | 否 |
| `common/speculative.h` | +4 | 锚点 | 否 |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | +7 | 唯一 Vulkan 锚点 | **是（§2）** |
| `include/llama.h` | +15 | prefill 3 锚点 | 否 |
| `src/llama-context.cpp` | +57 | route-b 5 锚点 + prefill 10 锚点 | 否（锚点本身与后端无关） |
| `src/llama-context.h` | +5 | 锚点 | 否 |
| `src/llama-kv-cache-dsv4.cpp` | +6 | 物理设备选择 | 否（后端无关的逻辑，见 §4） |
| `src/llama-kv-cache.cpp` | +10 | 物理设备选择 + `get_v_storage` 锚点 | 否 |
| `src/llama-kv-cache.h` | +3 | 锚点 | 否 |
| `src/llama-memory-recurrent.cpp` | +6 | 物理设备选择 | 否 |
| `src/llama-model-loader.cpp` | +59 | bounds check skip | 否（后端无关） |
| `src/llama-model-loader.h` | +8 | 同上 | 否 |
| `src/llama-model.cpp` | +71 | 逻辑/物理设备分离 + dense placement | 否（重构要复用的抽象，见 §4） |
| `src/llama-model.h` | +2 | `dev_layer_physical` / `route_b_enabled` 声明 | 否 |
| `src/llama.cpp` | +6 | 逻辑改行 | 否 |
| `tools/server/server-context.cpp` | +10 | route-b spec 统计 + prefill nout | 否 |
| `tools/server/server.cpp` | +11 | prefill 锚点 | 否 |

结论：上游 23 个文件里，只有 `ggml-vulkan.cpp` 涉 Vulkan；其余全是后端无关的锚点或逻辑改行。重构基本不用再动上游（除非给 ggml-cuda 加一个同类锚点，见 §5）。

## 2. 涉 Vulkan 的上游部分（全部）

### 2.1 锚点：`ggml/src/ggml-vulkan/ggml-vulkan.cpp`，+7 行

`STREAM_MOE_ROUTE_B` 保护的 `#include "stmoe_routeb_vk_hostmap.frag"`（末尾 18878-79 行，dma frag 同批 include）。无宏构建时预处理跳过 = 纯上游。

### 2.2 frag：`patches/route-b/common/`（2 个，约 3.4 KB）

- `stmoe_routeb_vk_hostmap.frag`（1186 B）：定义 `stmoe_vk_buffer_host_ptr`（host-visible 时返回 `ctx->dev_buffer->ptr`，否则 nullptr）与 `stmoe_vk_buffer_host_offset`（返回 `vk_ptr_base + off` 的 fake-base 指针；Vulkan kernel 用 `tensor->data - vk_ptr_base` 算 buffer 内偏移）。
- `stmoe_routeb_vk_dma.frag`（2342 B）：定义 `stmoe_vk_dma_read`（包一层 TU-static 的 `ggml_vk_buffer_read`：transfer queue + HOST_CACHED staging；注释写明 rebar 直读约 0.02 GB/s，DMA 约 14 GB/s）与 `stmoe_vk_dma_available`（恒 true，占位诊断）。

其余 44 个 frag 与 Vulkan 无关（route-b common 22 个是参数/上下文/放置/spec 统计；prefill-export 22 个是导出体）。

## 3. 自有代码使用 Vulkan 的清单（`src/`，共 4 文件 + 1 frag）

| 文件 | 位置 | 用途 | CUDA 对等物 |
|---|---|---|---|
| `src/backend/minigraph_exec.cpp` | :43-44 声明；:1543, 1556, 1599, 1650, 1844, 1930, 2414, 2452, 2453, 2464 共 10 处调用 `host_offset` | 给 device 子图的每个 tensor 造 data 指针（fake-base 约定）；:87 注释：绕过 ggml-vulkan 未 pin 时 `get_async`/`set_async` 串行化 | **最深耦合**。CUDA 下 `tensor->data` 是设备指针，host 不能解引用；这 10 处要改成“后端给地址，minigraph 只传地址不解引用”的模式 |
| `src/backend/moe_backend.cpp` | :18 声明；:446, :488 调用 `host_ptr` | 读 VRAM 池的 host 映射（判空 = 非 host-visible 时回退）；:491 注释提到 `C1:Vulkan0` | CUDA：设备内存一般无 host 映射，判空逻辑要改成“始终走 DMA/拷贝路径” |
| `src/server/route_b_inject.cpp` | :24-25 声明；:284 调用 `host_ptr`；:306-307 按 `seg.dev` 前缀 `"Vulkan"` 分发 `set_pool_dma_read(stmoe_vk_dma_read)` | 注册 expert 池的 DMA reader；:243 注释提到 `Vulkan_Host` buft（DEVICE_LOCAL\|HOST_VISIBLE） | 加 `"CUDA"` 分支，分发 `stmoe_cu_dma_read`；`Vulkan_Host` buft 对应 CUDA 的 host-pinned staging |
| `src/backend/moe_backend.h` | :55 注释 | route_b 在 vram pool 就绪后注册 backend + buffer 类型 | 注册点要按后端分发（Vulkan0/CUDA0/CPU） |
| `patches/route-b/common/stmoe_routeb_args.frag` | :41, :56 硬编码 `"Vulkan0:"`；:103, :112, :133 help 文本 | 解析 `--moe-ram-pool` 等放置参数（`<device>:<MB>`，如 `RAM:8192,Vulkan0:5120`） | 解析要接受 `CUDA0`（及 RAM/CPU），`Vulkan0` 保留 |
| `src/backend/scheduler.h` | :76, :188, :368 注释 | 已预留设计：vulkan transfer-queue 与 `cudaMemcpyDtoH` 各提供一个 reader，可共存 | 注释变实现：scheduler 的 DMA reader 注册表按后端 kind 分发 |

另：`src/` 里 `cuda|CUDA` 只有 5 处，全是上述注释（scheduler.h 3 处、route_b_inject.cpp:297 1 处 “when it lands”、scheduler.cpp:313 1 处）。即 CUDA 在自有代码里零实现。

## 4. 与后端无关、重构要保留的逻辑改动（route-b-inject.patch）

- `src/llama-model.cpp` / `.h`（+71/+2）：逻辑设备 vs 物理设备分离（`dev_layer()` 返回 route-b 调度设备，`dev_layer_physical()` 返回真实设备）+ dense placement。这是三后端复用的关键抽象：placement 字符串（`C1:Vulkan0` / `C1:CUDA0` / `C1:RAM`）落到这里，不要另起一套。
- `src/llama-model-loader.cpp` / `.h`（+59/+8）：expert 权重 bounds check skip（route-b 接管加载）。
- `src/llama-kv-cache*.cpp`、`src/llama-memory-recurrent.cpp`、`src/llama.cpp`：物理设备选择与冲突拒绝（KV、DSV4 compressor、recurrent state 用物理设备）。
- 这些改动不感知 Vulkan，CUDA 复用时不用重写，只需确认 `CUDA0` 能走通同一条 placement 路径。

## 5. CUDA 缺口映射（vk → cu）

| Vulkan 现状 | CUDA 对等物 | 难度/备注 |
|---|---|---|
| `ggml-vulkan.cpp` 锚点 include hostmap+DMA frag | 可能不需要 vendored 锚点。CUDA 的 DMA 读就是 `cudaMemcpyDtoH`，尽量用公开 API + 自有 `src/` 实现，零上游侵入；只有当需要 ggml-cuda 内部 context 时才考虑加锚点 | 待查：从 `ggml_backend_buffer_t` 拿 CUDA 设备指针的公开办法（§7 阶段 3 调研点） |
| `stmoe_vk_buffer_host_ptr` | 无直接对等物。CUDA 设备内存 host 不可见（除 managed/UVA）；moe_backend 的“判空回退”分支在 CUDA 下恒走拷贝路径 | 中 |
| `stmoe_vk_buffer_host_offset`（fake-base） | 无对等物，且概念要换：minigraph 的 `tensor->data` 在 CUDA 下就是设备地址；host 侧代码一律不解引用 | **难，是重构核心** |
| `stmoe_vk_dma_read`（transfer queue + staging） | `stmoe_cu_dma_read`：`cudaMemcpyDtoH`（同步）或 `cudaMemcpyAsync` + stream；staging 用 `cudaHostAlloc` pinned | 易 |
| `stmoe_vk_dma_available` | `stmoe_cu_dma_available`：恒 true（DtoH 一定可用）或按设备属性 | 易 |
| 放置字符串 `Vulkan0` | `CUDA0`；解析、help、CHECKPOINT/文档里的 `Vulkan0` 用例都要补 `CUDA0` | 易 |
| `build.bat`：`GGML_CUDA` 默认 OFF；StreamMoE tag 默认 `GGML_VULKAN=ON` | 加 CUDA tag（或 `GGML_CUDA=ON` 的变体），features 矩阵扩展 | 易 |
| B33（main tag CPU-only 时无条件引用 `stmoe_vk_*` 导致链接失败） | 重构顺带修掉：后端符号一律走分发表/条件编译，CPU 构建不链接任何 vk/cu 符号 | 验收项 |

## 6. 重构目标架构

新增一层后端无关的 device 接口（建议放 `src/backend/device_iface.h/.cpp`，自有代码，不碰上游）：

- `enum stmoe_dev_kind { CPU, VULKAN, CUDA }`；placement 解析产出 `(kind, index, MB)`，`"Vulkan0"` / `"CUDA0"` / `"RAM"` 都走同一解析。
- 接口（每个后端提供一套）：`buffer_host_ptr`（CPU：直接返回指针；Vulkan：现有 hostmap；CUDA：返回 nullptr，语义 = 不可 host 直访）、`buffer_host_offset`（CPU：base+off；Vulkan：现有 fake-base；CUDA：设备 base+off，只作 kernel 地址，host 不解引用）、`dma_read`（CPU：memcpy；Vulkan：现有 transfer-queue；CUDA：`cudaMemcpyDtoH`）、`dma_available`。
- 调用方（minigraph_exec / moe_backend / route_b_inject）只调接口，不再出现 `stmoe_vk_*` 名字、不再按 `"Vulkan"` 前缀字符串分发。
- 上游保持现状：`ggml-vulkan.cpp` 锚点不动；CUDA 争取零上游改动；route-b-inject.patch 的逻辑/物理设备分离复用。

## 7. 分阶段计划（每阶段可独立验收，不动下一阶段代码）

- **Phase 0（本审计）**：清单 + 目标 + 待确认问题。你点头后进入 Phase 1。
- **Phase 1 接口层**：抽 `device_iface`，先接 CPU + Vulkan 两套实现，行为零变化。验收：现有 Vulkan 回归（RX 590 上 `C1:Vulkan0` 的 CHECKPOINT 用例）+ CPU tag 数值一致；B33 在 main tag 上消失。
- **Phase 2 CUDA DMA 与放置**：`stmoe_cu_dma_read` + `CUDA0` 解析 + `build.bat` CUDA 变体；先不碰 minigraph 的 10 处 `host_offset`（CUDA0 的 expert 池只走 DMA 路径，device 子图仍跑 CPU/Vulkan）。验收：P100（TCC）上 expert 权重能经 CUDA DtoH 进出 RAM 池。
- **Phase 3 minigraph 解耦**：10 处 `host_offset` 改走接口，host 侧不解引用设备指针。验收：`CUDA0` device 子图跑通；Vulkan/CPU 回归。
- **Phase 4 全量验证**：三后端 ×（DeepSeek-V4 43 层 / 256 专家）数值对照（per-element，与 stock graph 一致）；更新 CHECKPOINT 与 BUG_TRACKER。
- **Phase 5 清理**：删除失效注释（"when it lands" 等），补文档（placement 字符串、构建矩阵）。

## 8. 前置条件与风险（审计时已核实）

- **CUDA 工具链未装**：`nvcc` 不存在，`CUDA_PATH` 为空，`C:\Program Files\NVIDIA GPU Computing Toolkit` 不存在。编 ggml-cuda 必须先装 CUDA Toolkit；按你的规矩，装到 C 盘要先问你，且版本要带 sm_60 支持（12.6 是稳妥选择，12.8+ 对 Pascal 的支持状态装之前再核实一次）。
- **P100 保持 TCC**：TCC 下 CUDA 正常，且无 TDR 超时风险，这是 CUDA 路线的优势；Vulkan 仍不可用，与本次重构无关，不用再动驱动。
- **ggml-cuda 的 sm_60**：llama.cpp 的 ggml-cuda 需要 `CMAKE_CUDA_ARCHITECTURES` 含 60，否则编出来 P100 跑不了；这是 Phase 2 的检查项。
- **上游改动纪律**（`patches/README.md` 铁律）：不得不直接改 vendored 时先问你。Phase 2/3 如需给 ggml-cuda 加锚点，会先提出来，不直接加。

## 9. 待你确认

1. 是否按 §7 分阶段做，Phase 1 先抽接口（CPU+Vulkan 行为不变）？
2. CUDA Toolkit 装哪个版本、装哪（默认 C 盘，按规矩要你点头）？我建议先不动，等 Phase 1 验收完再装，避免一次变两个东西。
3. 放置字符串保留 `"Vulkan0"` / `"CUDA0"` / `"RAM"` 这套命名吗？
4. 重构期间 main tag（CPU-only）是否保持可构建作为回归基线？

---

审计依据（2026-11-24 实测）：`git -C third_party/llama.cpp diff --ignore-cr-at-eol --stat`（23 文件 +337/-16）；逐文件 added-line 标记统计；`src/` 全量 grep `vulkan|Vulkan|stmoe_vk|ggml_backend_vk|vk_ptr|Vulkan0`；`nvcc` / `CUDA_PATH` / Toolkit 目录检查；`patches/README.md`（2026-09-03 版）。
