[English](ROUTE_B_GPU_PHASE.md) | [简体中文](ROUTE_B_GPU_PHASE.zh-CN.md)

# Route B GPU 混合池设计方案（Phase B）

> 状态：设计（2026-08-29）。基于 route B 第三路径（官方 `ggml_mul_mat_id` 内核 + 均匀 stride 槽池，`docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` §7）。
> 已在 vendored llama.cpp @ 5ab785cf8 核实：Vulkan 与 CUDA 均原生支持 `MUL_MAT_ID` 与池槽 3D 布局。
> 相关：`docs/MULTI_MODEL_POOL.md`（每模型池）、`docs/MULTI_SUBPOOL.md`（按专家种类子池）、`docs/Backend.md` §1（CPU/GPU 双池并发）。

## 目标

扩展 route B，让部分专家可驻留 GPU 池（VRAM），其余留在 CPU 池；物理内存仍受池预算约束（`--moe-ram-pool` / `--moe-vram-pool`）。模型无关，不改 llama.cpp 数学 / 图拓扑 / 调度。

## 架构：三根柱子，各管一摊

| 柱子 | 归属 | 职责 |
| :--- | :--- | :--- |
| 计算劫持 | `stream_moe` backend | 所有消费 ids 的 `MUL_MAT_ID` 被劫持——src0 就地改指池槽布局，官方内核执行。 |
| pin/unpin 生命周期 | `cb_eval` 钩子 | 产生 ids 处 pin、专家用完处（汇合节点）unpin。与计算解耦。 |
| 异步加载引擎 | scheduler | IOCP（Win）/ io_uring（Linux）/ io_submit fallback——并发 in-flight 读、wait-on-address 唤醒。 |

## 1. 计算劫持（就地池槽改写）

MoE 节点保留在主图（`build_moe_ffn` 的 `ggml_mul_mat_id`）。backend 的 `graph_compute` 收集它们，用官方内核按池槽布局执行：

- src0 就地改三字段：`ne[2]: n_expert→n_slots`、`nb[2]: 专家stride→槽stride`、`data: dummy→池基址+分支off`（`ne[0]/ne[1]` 不变——每专家权重形状相同；官方内核按 `e*nb[2]` 索引，正好吃槽号）。
- ids 翻译：专家 id → 槽索引（pin 时映射已定）。
- `ggml_backend_graph_compute(backend, 子图)` 官方内核执行。

CPU-only 今天：一个子图，委托 `cpu_backend`。GPU 混合：按专家所在池把 ids 拆成 `cpu_ids` / `gpu_ids` -> **每池一个子图**（单个 src0 无法同时描述两个池）——CPU 子图到 `cpu_backend`、GPU 子图到 `vulkan_backend`，按 token/专家归属拼回 dst，两组并发。

已在 vendored llama.cpp @ 5ab785cf8 核实的关键事实：

- Vulkan：`supports_op` 处理 `MUL_MAT_ID`（`ggml-vulkan.cpp:18120`）；AMD vendor 分支启用 `mul_mat_id_s/m`（small/medium warp tile，`mul_mat_id_l=false`）；src0 类型覆盖 Q4_K / Q5_1 / Q8_0（gemma 异构全类型）。
- CUDA：原生 `MUL_MAT_ID` 支持（`ggml-cuda.cu:2252` supports_op + 专用 `mul_mat_id` / up-gate 融合内核 `:1684`）。
- **注意**：后端的 `supports_buft` 不认 stream_moe 池 buft——所以 GPU 子图 src0 必须是**后端自己的 buffer（GPU 池）**，不是池 buft。GPU 池 = 一块 vulkan buffer（DEVICE_LOCAL 或 HOST_VISIBLE），槽权重拷入（DIO→host→device，或 HOST_VISIBLE 直写）。

## 2. pin/unpin 生命周期（cb_eval）

pin 需要 ids 的**值**（运行时张量）——所以"产生 ids 处 pin"实际是"ids 就绪的图执行点 pin"：

- **hash 层**（`il < dsv4_hash_layer_count`，`deepseek4.cpp:1279`——确定性 `get_rows(ffn_gate_tid2eid, tokens)`）：token 已知即 ids 已知 -> **构建期**就异步 pin（免费完美预取）。注意：hash 是浅层，argsort 是深层。
- **argsort 层**：`cb_eval` 没有"执行后"回调，所以 pin 落在**该层第一个 `MUL_MAT_ID`**（拓扑序保证 ids 已就绪）：读 `src[2]`、对未预取的专家异步补 pin、等 ready（wake-on-ready，非忙等）、放行。
- **unpin 挂汇合节点**（`mul(expert_out, weights)` / `add`）：这些是消费 down 输出的 CPU 逐元素节点——sched 的边界事件同步保证 GPU 子图在它们执行前已完成——所以在此 unpin 是 GPU 完成的**权威确认点**（搭 sched 事件同步便车，无需额外 `backend_synchronize`）。unpin 略晚安全（只是多占一会槽）。

`cb_eval` 是每节点回调且强制同步——CPU-only 无妨，GPU 混合会吃掉异步收益。GPU 阶段 pin/unpin 可能回归 `graph_compute` 内部（天然有每 split 的前后位置、能与后端事件同步耦合），异步加载引擎保留。

## 3. 异步加载引擎（scheduler）

pin = 提交异步读（scheduler MPSC 队列）立即返回——计算线程绝不阻塞在 IO 上。scheduler 跑 IOCP（Win）/ io_uring（Linux）/ io_submit fallback，多 in-flight 并发（QD 扫描：1/4/16/64）。IO 完成写槽 READY（memory_order_release）+ WakeByAddressAll / futex_wake；计算线程 wait-on-address 等 `slot_meta[slot]`（64 位原子字：state+refcount+generation）变 READY（acquire），绝不忙等。GPU 槽在 IO 后多一步 host→device 拷贝（或直写 HOST_VISIBLE VRAM，ReBAR 属 Phase C）。

## 4. 多 GPU

`ggml_backend` 原生多设备（sched `n_backends`，vulkan 枚举 `physical_devices.size()`）。池加设备维度：**每模型 × 每设备一个池实例**。池选择（pin 时）决定专家进哪个 GPU（空闲 / 热度 / 层偏好）。每个 GPU 一个 vulkan backend，MoE 子图按专家分配到的 GPU 委托对应 backend。dense 的 `-ngl` 分片机制不动。

## 5. 通用 ggml 后端抽象

整个方案走 `ggml_backend_graph_compute(backend, 子图)`——GPU 阶段只是把 `cpu_backend` 换成 `vulkan_backend`。不写任何 shader / 内核 / 后端专属内存代码：池（槽 / DIO / pin / 翻译 / 池选择）是 backend 无关的，在 stream_moe 层。

性能要点：

- **权重搬运是主要开销**：GPU 池槽权重来自 DIO 读，要拷进 vulkan buffer。用 HOST_VISIBLE（AMD 直写）缓解 + EST1 热度让高频专家常驻 VRAM；ReBAR（HOST_VISIBLE|DEVICE_LOCAL 零拷贝）是 Phase C。
- **每层委托调度开销**：每个 MoE 子图一次 vulkan `graph_compute`（命令提交 / descriptor 绑定）——llama.cpp vulkan 后端有 pipeline 缓存；对计算密集的大 GEMM，GPU 收益远大于调度开销。
- 代价：放弃融合/定制 shader（如 swiglu 融合进 GEMM）——可接受，符合"不改 llama 数学 / 依赖通用后端"的取向。

## 6. 异构专家

`docs/MULTI_SUBPOOL.md` 的按专家种类分组（子池、预算按字节占比）继续适用。GPU 侧加两点：

- GPU 池也要按组；池选择（pin 时）要带"组量化 × 目标后端能力"（某组可能只进 CPU 池——vulkan 不认其类型）。
- src0 的 `nb[2]`（槽 stride）按层/组解析（已有 `branch_layout(layer,...)`）；CPU/GPU 槽各自按组/后端对齐。

## 7. 数值等价

GPU 内核浮点累加路径不同——预期 ulp 级差异（与 repack-vs-普通 同类）。回归必须覆盖三种分布（`docs/Backend.md` 上线门槛）：全 CPU / 全 GPU / 混合。

## 8. 构建前置（环境）

- Vulkan SDK（glslc shader 编译器、vulkan.hpp、vulkan-1.lib）——构建期必需；build.bat 已透传 `GGML_VULKAN=ON`（默认 OFF）。运行时只需 GPU 驱动（vulkan-1.dll 已存在 1.3.250.0）。
- RX 590 8GB 仅 Vulkan；VRAM 池预算因此很小——现实目标是 dense 上 Vulkan + 小部分专家上 GPU + 大部分专家留 RAM。
