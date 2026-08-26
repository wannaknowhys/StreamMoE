# DeepSeek4 MoE 自定义 Backend / Expert Pool 调度设计

## 保留 (不动)

* **`src/models/deepseek4.cpp` / `llama-graph.cpp` 里的全部数学**——MLA、DSA indexer、hyper-connections、Sinkhorn 归一化、gating、top-k 选择、hash-routed 确定性映射、shared expert 相加、swiglu clamp，以及下游的 weighted aggregation 节点 (`selected_weights` 加权求和)，一行不改。

* **`ggml-backend.cpp` 的调度器本身**——split 划分、跨 backend 事件同步机制，复用；这套机制也是我们 unpin 时机、以及 GPU 异步完成确认的天然落点，不用另起一套 join 逻辑。

* **不复用 `copy_experts`**——原因同前，我们自定义 backend 内部直接对接 slot 池，不走标准 host→device 拷贝这条路径。

* **一张完整的 `ggml_cgraph`**——构图时机、拓扑结构与 stock llama.cpp 完全一致，不因命中数、也不因专家分布在哪个池，产生不同的图。

---

## 新增

### 1. 自定义 Backend 与 执行机制

* 自定义 `ggml_backend/buffer_type`，实现 `supports_op(MUL_MAT_ID + 用到的量化类型)` 和 `graph_compute`。

* `graph_compute` 必须原生支持"同一次调用里，选中的专家分布在 CPU 池和 GPU 池两边"：按 expert 当前所在池分组 (`cpu_ids`/`gpu_ids`) → 两组并发 pin → GPU 组异步发射（优先用 `cuStreamWaitValue64` 之类的流内内存等待做到零 CPU 线程占用，不支持则退化成独立 dispatch 线程 park+转发）→ CPU 组本线程同步算 → 函数立即返回，不等 GPU，把"这个 split 真正完事"这件事交给 sched 已有的事件同步机制。CPU/GPU 两部分结果只是按 token 归属拼进同一张输出张量，不涉及任何加权求和数学。

### 2. Slot 状态与位分配 (64 位原子字)

结构为 `state(3 bit) + refcount(29 bit) + generation(32 bit)`，pin/unpin/驱逐判定全部走 CAS。

```text
bit 63........32 | 31........3 | 2 1 0
    generation   |  refcount   | state
```

* **state (3 bit)**: `000=EMPTY`, `001=IO_INFLIGHT`, `010=READY`, `011=EVICTING`, `100=FAILED`。（`FAILED` 必须显式区分，DIO 失败时 CAS 到 `FAILED`，等待线程醒来直接向上抛错，防止变成"看起来像死锁"的卡死）

* **refcount (29 bit)**: 够用，不会有上亿个并发 pin。

* **generation (32 bit)**: 每次 slot 被重新装填不同专家时自增，用于防 ABA。

#### 监听与唤醒路径 (低 32 位):

* **Windows 路径**: `WaitOnAddress` 支持 1/2/4/8 字节，直接对整个 64 位字做 `WaitOnAddress(addr, &expected_snapshot, 8, ...)`，拿完整的 64 位快照做比较，不用拆字，最简单。

* **Linux + futex2** (`内核 ≥ 6.7`, `FUTEX2_SIZE_U64`): 同样直接对整个 64 位字操作，不用拆。

* **Linux fallback** (经典 32 位 `futex()`): 只能对 32 位字操作。要等待的条件（`IO_INFLIGHT → READY`）只跟低 32 位有关。直接把 `futex()` 地址指向该 64 位字的低 32 位那一半（小端序 x86/ARM 上低 32 位即地址本身；代码加 `static_assert` 兜底）。苏醒后**必须重新读完整的 64 位字**核对 generation 是否还是 pin 前记下的那个，不对就回头找调度线程重新要地址（防 ABA）。

* **唤醒时机**: 调度线程/loader 把 state 从 `IO_INFLIGHT` CAS 成 `READY` 后，紧接着调用 `WakeByAddressAll` 或 `futex_wake`（唤醒全部而不是唤醒一个，因为可能有多个层/并发请求在等同一个专家）。

* **无忙等**: I/O 毫秒级与睡眠唤醒微秒级差 3 个数量级，忙等无收益。

### 3. 驱逐（Eviction）状态发布顺序

必须严格遵循：

```text
READY  --CAS-->  EVICTING  ----->  清空 expert_directory  ----->  等 refcount == 0  ----->  复用 Slot
```

**先进入 `EVICTING` 阻止新的 pin，再清 directory**。若反过来先清 directory，持旧 slot index 的计算线程会在窗口期通过旧 index 成功 pin。

### 4. 线程间通信与 Residency 数据结构

两张独立的原子数组，**属于 Residency / Scheduling Control Plane，与 LRU/热度淘汰数据结构完全解耦**：

* `slot_meta[n_slots]`: 64 位原子字，描述每个物理 slot 的状态。

* `expert_directory[n_expert][n_pools]`: **二维 32 位原子数组（外层是专家号，内层是池子数量）**。存储的是 `slot 索引` 或 `UNASSIGNED`。只有调度线程写，计算线程只读。

* 查找方式：拿到专家号 `e` 后，计算线程直接遍历 `expert_directory[e][0..n_pools-1]` 扫描所有池子。

* `expert_directory_version[n_expert]`: 32 位/64 位原子自增版本号，每专家一个。用于解决二维 directory 下的等待问题。

#### 计算线程侧查找流程：

1. 读 `expert_directory[e]` 遍历各池子。

2. **命中** (找到合法 slot 索引) → 对 `slot_meta[slot]` 走 generation-aware CAS pin（无锁快路径，不打扰调度线程）。

3. **未命中** (所有池均为 `UNASSIGNED`) → 先记下当前 `snapshot = expert_directory_version[e]`，走 MPSC 队列向调度线程发起分配请求，然后 wait 在 `expert_directory_version[e]` 这个地址上。被唤醒后重读 directory 并进入 `slot_meta` 的 pin/等待流程。

#### 通信信道明确分三条：

1. **分配请求**: 计算线程 → 调度线程，有界 MPSC 队列，满时生产者阻塞等 (`wait-on-address` 等有空位)，不丢弃请求。

2. **命中统计**: 计算线程 → 调度线程，不经过队列，每专家独立 atomic 计数器，pin 成功直接 `fetch_add`，供 EST1/Phase C 热度决策使用。

3. **分配结果/状态变化**: 调度线程 → 计算线程，不是消息，是共享数组写入 + `version[e]` 自增 + `WakeByAddressAll`/`futex_wake` 广播，等待方醒来后自己重读判断。

### 5. 调度线程与 Loader

* **调度线程**: 与计算线程同进程，无 IPC。拆分为"非阻塞决策快循环"与"异步 I/O 执行"。异步 I/O 走 Windows IOCP / Linux `io_uring`，fallback 到 Linux Native AIO (`io_submit`)。

* **Phase C 迁移**: 热度感知的 CPU ↔ GPU 搬家决策单独拎到 Phase C。迁移是"改变专家的稳态归属"，与 Phase A/B 的"运行时按当前归属并发读两个池"明确区分。

* **Loader 改动**: 接线用 `tensor_buft_overrides` 把 `blk.*.ffn_*_exps.*` 指向自定义 buft。MoE 的 `ffn_*_exps.*` 无视 `mmap` 开关强制走 pool、不做物理载入；按 buft 类型判断，不与 `use_mmap` 分支混在一起。

* **GPU 侧等待**: Phase A 以 CPU dispatch 线程 park+转发为基线。ReBAR 记录为 Phase C 候选优化（HOST_VISIBLE | DEVICE_LOCAL 堆 + compute shader 轮询），不在 Phase A 绑入关键路径。

---

## 上线前必须补的一项

* **数值等价性回归测试**——同一份 ids/weights，分别跑 stock llama.cpp 图和接了自定义 backend 后的图，逐元素 diff。必须覆盖三种专家分布组合：全部命中在 CPU 池、全部在 GPU 池、以及同一次调用里 CPU/GPU 混合命中（第三种是新增逻辑，风险最高，必须单独起一组测试用例）。

---

## 后续修订记录

* 2026-08-26：补充到 route B 文档体系。与 `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` 的对应关系：
  - §1 自定义 backend `graph_compute` = 研究文档 §4.1/§4.3（mini-graph 委托执行 MUL_MAT_ID / MUL_MAT）。
  - §2 slot 64 位原子字 = 研究文档 §4.4（slot_meta/expert_directory 复用）。
  - §4.5 pin 生命周期：**首触 pin、末触 unpin**（split B pin、split D unpin，同一份 ids 自洽，无状态表）见研究文档 §4.8。
  - §4.9 shared expert：`ffn_*_shexp` 纳入 buft，backend 额外支持普通 `MUL_MAT`。
