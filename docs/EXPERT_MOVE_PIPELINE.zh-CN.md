# 专家迁移管线 + (L,E) 键驱逐 - 设计

[English](EXPERT_MOVE_PIPELINE.md) | [简体中文](EXPERT_MOVE_PIPELINE.zh-CN.md)

> 状态：**设计（2026-09）** - 一次工作会话敲定下一个 scheduler 重构。已落地的
> 批量 pin 层（slot_request bitmap / wake-once，见 WORK_IN_PROGRESS L）是本设计的基座。
> 本文档捕获**完整设计**，避免会话间丢失。
> 相关：`docs/WORK_IN_PROGRESS.md` L / J 节、`docs/Backend.md` §2/§4、
> `docs/EXPERT_SCHEDULER_DESIGN.md`、`src/backend/scheduler.{h,cpp}`、`src/backend/slot.h`。

## 1. 动机 / 为什么重构

在尝试于 VRAM 上验证 SoA 列池时观察到的现象：

1. **逐专家阻塞 pin 串行化 DIO** 并在 exec/scheduler 间乒乓——已由批量 pin 修复
   （WORK_IN_PROGRESS L，已落地）：129-token prefill 在纯 RAM 路径从几十秒降到
   0.11s，逐字节 IDENTICAL。
2. **VRAM 运行仍卡**（129-token prefill ~900+ 次 demote，从未到达 `exec_mm_vk`）：
   每次向满的设备区装载新专家都在调度线程**同步** demote 一个 victim
   （整专家 VRAM→RAM memcpy，并递归为 RAM 腾位）。批量 pin 改完后暴露了这点：
   DIO 串行没了，demote 串行成了新瓶颈。
3. **demote 必须异步化**（worker 做 memcpy，调度线程永不阻塞），且同一 worker
   必须同时服务 **v2r（VRAM→RAM）和 r2v（RAM→VRAM）**——一个统一的 move 管线，
   不是 demote 特例。

次要驱动：**驱逐键化**。今天驱逐扫描池的槽并用 `owner_[]` 反查 (L,E)。重构后
驱逐以 (L,E) 为键并带层距偏好（先逐刚算完的层），从而**完全不需要 `owner_`**。

## 2. 已落地基座（WORK_IN_PROGRESS L）——不可回归

- `slot_request_t` 96B POD：`{layer, total_tokens, start_rdtsc, seq(=批目标计数),
  batch_ready ptr, needed[8]=512bit bitmap}`。
- `mpsc_alloc_queue`：普通 POD ring + 每槽发布 generation（release/acquire），
  多生产者安全，无 ABA。绝不用 `std::atomic<96B>`。
- `expert_scheduler::pin_layer(layer, bitmap, await, out)`：扫 dir → pin 已驻留，
  missing 子集一次提交，wake-once 在计数词上，一轮重试。`pin_expert`/`wait_ready` 已删。
- `accept_requests`：pop 一个层请求，遍历 bitmap，device-first 放置，逐位批 bump，
  requeue 未放置位。
- `drain_completions`：settle bump 批计数 + wake。
- init 在 `n_expert > MAX_EXPERTS_PER_LAYER (512)` 时 fail-fast。
- 验证：gemma v2align prefill-from IDENTICAL、0.11s（纯 RAM）；5/5 UT。

## 3. Directory 作为 (L,E) 键生命周期状态表

### 3.1 现状（3 个并行数组，均按 (L,E) 索引）

```
entries_  [(L,E)*n_pools + pool]  -> slot          （专家 -> 每池槽）
versions_ [(L,E)]                 -> u32           （等待/唤醒版本）
last_used_[(L,E)]                 -> u64           （recency，跨池）
stats_.adaptive_scores_[(L,E)]    -> double        （EST1 freq，scheduler 私有）
owner_    [slot]                  -> (L,E)         （槽 -> 专家，反向）
```

今天的访问模式：exec 按 (L,E) 扫；驱逐扫槽后用 `owner_` 拿 (L,E)，再按 (L,E)
索引 last_used_/freq。所以热路径里唯一的**遍历**是驱逐的槽扫描。

### 3.2 问题：半途状态对 exec 不可见

`entries_` 只记录 READY 驻留（`dir_->set` 在 mark_ready 时触发）。在"alloc 预约槽"
与"mark_ready"之间，槽是 IO_INFLIGHT 且 `owner_[slot]` 已写 (L,E)，但
`entries_[(L,E)]` 还是 UNASSIGNED。于是计算线程的扫描看不到**正在装载**的专家，
调度线程的请求也无法区分"正在装"和"缺失"——两者经 entries_ 看一模一样。
这造成重复装载窗口（exec 请求一个 mid-load 的专家；accept 再 alloc 一个槽；
两次装载竞争，一个成为孤儿 READY 槽）。

### 3.3 提案：per (L,E) x pool 状态

保持扁平 (L,E) 索引布局（对主导的随机访问 cache 友好），但每个条目加宽以携带
专家生命周期相位。两种布局：

- **A1（推荐）：per (L,E,p) 两个并行原子**
  `state_[(L,E)*n_pools+pool]`（u32）+ `slot_[(L,E)*n_pools+pool]`（u32）。
  状态迁移频繁（move），槽一旦 set 稳定。读多写少。
- **A2：per (L,E,p) 单 u64 `{state(4) | slot(28) | gen(32)}`**
  扫描一次原子读，但每次状态改变都要重写整词。

尺寸：deepseek 43 层 × 256 专家 × ~2 池 = ~22k 条目；任一布局几百 KB。可忽略。

状态（per (L,E) × pool；一个专家可同时占 RAM + VRAM 副本）：

```
ABSENT       无槽      不在本池
LOADING      槽 Y      已预约，DIO 在飞（槽 IO_INFLIGHT），未 READY
READY        槽 Y      可 pin（现行为）
MOVING_OUT   槽 Y      这份正被搬走（v2r/r2v 源）
MOVING_IN    槽 Y      这份正被搬入（目标），不可 pin
FAILED       无槽      装载/迁移失败
```

一次 move = 两个池两个条目同时：`MOVING_OUT(源池) + MOVING_IN(目标池)`；
完成后源 → ABSENT，目标 → READY。

**完整描述住哪里——不在 directory。** Directory 只回答计算侧问题"(L,E) 这里可用吗、
在哪"。完整 move 描述（源池/槽、目标池/槽、列）住在 move 任务对象里（见 §5）。
cq 携带字节少，但 `drain_completions` 从 `done[i]->user_data` 拿回任务指针，任务里什么都有。

**单写者纪律不变**：只有调度线程改这些状态；计算线程只读。现有 `version[(L,E)]`
每次迁移 bump + wake 保留——任何半途状态改变都会唤醒等待者重扫。

### 3.4 顺序不变量（必须钉死）

调度线程决定把 (L,E) 装进池 p 槽 s 时：

```
dir(L,E,p) = LOADING(slot=s)   // 先发布意图（seq-cst store = 可见点）
owner 侧：（已删，见 §4）
slots_[s].begin_reload()       // 物理槽 -> IO_INFLIGHT
```

反序（先预约槽、后发布）会留下"槽在装但 directory 看不见"的窗口。固定顺序消灭
重复装载窗口。

## 4. 删 owner_（槽 -> (L,E)）

证据：注释 `owner_` 后恰好 8 个编译错误，**全在 `alloc_or_evict` 内**
（登记 :258/:328、打分读 :278/:281、victim 反查 :293-294）。无其他文件/函数碰它。

一旦驱逐改成 (L,E) 键（下节），调度线程永不需要问"槽 i 里住着哪个专家？"：
- 装载完成 `dir_->set(t->layer, t->expert, ...)` 已从 async_load_t 带 (L,E)；
- 驱逐按 (L,E) 选，经 `entries_` 读槽。

故删除 `owner_`。槽侧只剩物理状态（slot_meta）。

## 5. Move 管线：异步 v2r + r2v worker

### 5.1 原则：demote 是缓存迁移，尽力而为

专家权重只读；VRAM 内容 == 磁盘内容。搬到 RAM 是加分（省一次未来盘读），绝非必须。
故 move 管线是低优先级后台任务，调度线程绝不因它阻塞。
（工作决定：**等 copy 完成再复用源槽——不在 copy 中途 drop**。drop 曾考虑过，因
状态清晰度被否；`MOVING` 状态 + 调度线程 in-flight 记录保持模型干净。）

### 5.2 为什么 v2r 不用 GPU/vulkan DMA

`vkCmdCopyBuffer` / copy engine 只能做 device↔device 与
device↔host-visible-vulkan-buffer。独显（RX 590）的 host-visible heap 仍是显存
（rebar BAR1）；128GB 系统 RAM 池不在 GPU 地址空间，copy engine 写不到它。
独显上不存在真正的 `vkCmdCopyBuffer` vram→系统 RAM 目标（只有 UMA/APU 有）。
跨后端 `ggml_backend_tensor_copy` 同样回落到 memcpy，或要求一个占显存的
host-visible 目标 buffer。结论：本硬件上 v2r 到普通 malloc RAM 只能 CPU memcpy。
异步 CPU memcpy worker 是解；GPU copy-engine DMA 记为未来 UMA/APU 目标。

### 5.3 任务形态（move_task）

一个专用 copy worker（数量参数化，先 1）。每专家迁移一个任务；批量提交。worker
只做纯逐列 memcpy，绝不碰控制面。

```cpp
struct move_task_t {
    // kind: MOVE_V2R（src=vram 池, dst=RAM 池）或 MOVE_R2V（反向）
    uint32_t layer, expert;
    uint32_t src_pool, src_slot;
    uint32_t dst_pool, dst_slot;
    // 逐列指针由 worker 经 slot_col_mem() 推导；列数/stride 取自组拓扑（src/dst 同组）
    expert_scheduler* owner;         // 完成分派回来
    uint64_t req_tsc, done_tsc;      // [profile]
};
```

move 任务 ring，仿 async-load ring buffer 模式。

### 5.4 通讯（两个队列，仿 DIO）

- **提交队列**（调度线程 -> worker）：move_tasks。调度线程一次批量提交若干
  victim move（`submit_moves(tasks, n)`）。
- **完成队列**（worker -> 调度线程）：完成的任务。worker 只 memcpy 后把任务推回；
  调度线程做控制面收尾：dst `mark_ready` + `dir_->set`（MOVING_IN → READY）+
  version bump + wake + 批计数；随后源槽 `begin_reload()`（现在可装新专家）。

### 5.5 一次 v2r move 的状态舞步

```
调度线程（设备区满，需给新专家 X 腾槽）：
  按 (L,E) 驱逐（§6）选 victim V
  entries_(V, 源池)    : READY  -> MOVING_OUT
  dst = alloc_or_evict(RAM)          // dst IO_INFLIGHT（排他）
  entries_(V, RAM)     : ABSENT -> MOVING_IN
  enqueue move{V vram, RAM dst}      // 非阻塞
  源槽保持 EVICTING（不 reload）直到 cq

worker：逐列 memcpy(RAM dst, vram src)；推完成

调度线程 drain：
  dst slot mark_ready()              // IO_INFLIGHT -> READY
  entries_(V, RAM)    : MOVING_IN -> READY   （+ version bump + wake + 批计数）
  源槽 begin_reload()                // 复用于 X
  entries_(V, 源池)   : MOVING_OUT -> ABSENT（或直接 reload 成 X）
```

r2v 是镜像；worker 代码方向无关（src/dst 互换）。

### 5.6 排他：用槽状态，不用 refcount

今天驱逐会 drain refcount（`while (refcount != 0) yield`）等计算读者放完。若 move
worker 也在源槽上取一个 refcount pin，那个 drain 会自锁（refcount 语义是
"计算读者共享计数"，不是"迁移锁"）。排他来自状态：源 = EVICTING / MOVING_OUT，
目标 = IO_INFLIGHT / MOVING_IN——两者都拒绝 `try_pin`（要求 READY）。两个槽都不加
额外 refcount。

## 6. 驱逐以 (L,E) 为键 + 层距偏好

### 6.1 语义

驱逐回答"本池要为 layer-L 批腾 K 个空槽"。victim 从最靠近 L 的层选（decode 内
刚算完的层最可能已陈旧），再往远层，score 阈值按层距放宽：

```
for delta = 1, 2, ...:
  for each 本池驻留的 (L-delta, e):
     if 槽 READY && refcount==0 && score(e) <= threshold(delta): 候选
  从产出候选的最近层里选最低 score 那个
```

工作示例（已定）：需要 K 槽 -> 取 L-1 层最低 score 驻留专家的 top K/2、
L-2 的 top K/2，……（score 低者先逐；score 线估算后置，见 §8）。

### 6.2 如何枚举本池驻留的 (L-delta) 层专家

两个选项：

- **A：逐层扫**——对层 L-delta 每个专家 e 查 `entries_[(L-delta, e)][pool]`；
  驻留则查槽 refcount + score。成本 O(每层 n_expert)（gemma 128 / deepseek 256），
  且只扫少数近层。简单，无额外索引。
- **B：per-(pool, layer) 驻留列表**——增量维护。驱逐更快，但每次 set/clear 付记账。

决定：先用 **A**（近层少，每层 ~256 查，可忽略）。B 留作将来驱逐上 profile 后的优化。

### 6.3 批量规划 vs 逐专家 alloc

现在 `accept_requests` 逐专家调 `alloc_or_evict`。(L,E) 层距方案天然想要**批量规划**：
收到 layer-L 批后，算本池需几个新槽，一次规划逐 K 个 victim（L-1 的 top K/2 等），
再装载。这是目标终态；per-expert `alloc_or_evict` 包装可作为底下单槽原语保留。

## 7. 批进度记账移到调度线程侧

已定：这层批"还剩几个在装"的记账属于调度线程（它反正逐位处理）。exec 应在整批
READY 时**只醒一次**，而不是醒 N 次去比较一个它不拥有的计数。

- 调度线程维护 per-batch 进度（如小的 batch-slot 数组或并入 in-flight
  move/load 记账）：`remaining = seq`；每个处理位（装载 READY 或 skip in-flight）
  递减；到 0 时写一次完成信号并 wake 一次。
- exec 无论如何保留它的 `pins[]` handle 数组（层计算靠它 resolve 槽、层尾靠它
  unpin）——那个记账独立。

## 8. Open questions / 后置

1. **score 线估算**："需要 K 槽时，层 L-delta 该用什么 score 阈值？"——后置
   （用户：可将来再说）。先按"最近层优先、按 score 取每层 top K/2"细化。
2. **A1 vs A2** directory 布局：推荐 A1（两个并行原子）。
3. **in-flight 重复请求处理**：LOADING/MOVING_IN 的 (L,E) 收到第二个请求——skip 且
   仍计入批目标（调度线程负责让该专家 READY；在途算它的义务）——需在
   accept_requests 里精确表述。
4. **move worker 数量**：先 1，参数化。
5. `dir_->set` 语义是否彻底改成"预约 = LOADING、完成 = READY"两段（见 §3.4 顺序）。

## 9. 构建顺序（终态，按组件）

1. Directory：加宽为 (L,E) 键状态（A1）；owner_ 留到第 3 步编译通过。
2. 装载路径：begin_reload 前先发布 LOADING（顺序不变量）；accept 加 in-flight
   skip（消灭重复装载窗口）。
3. 驱逐：alloc_or_evict 内改 (L,E) 键层距选择；删 owner_（8 处用，全在此函数，
   已枚举）。
4. Move 管线：move_task ring + 1 worker（v2r + r2v）+ 完成 drain；把 v2r 接进
   device victim 的驱逐；r2v 留给将来放置策略。
5. 批进度记账放调度线程（只醒一次）。
6. 验证：纯 RAM IDENTICAL（如今天）；VRAM 单设备运行到达 `exec_mm_vk`
   （这是 K6/L6b 阻塞项）；directory 状态 + move ring + 驱逐顺序的 UT。

## 10. 涉及文件

- `src/backend/slot.h`——directory 状态加宽（A1）或新增 move-task 结构
- `src/backend/scheduler.h` / `scheduler.cpp`——驱逐、删 owner_、move worker + ring、批记账
- `src/backend/minigraph_exec.cpp`——预期不动（exec API 不变：仍是 pin_layer），
  除非批信号移到调度线程侧
- `tests/test_slot.cpp`、`tests/test_scheduler.cpp`——新状态/move/驱逐 UT
- `docs/WORK_IN_PROGRESS.md`——任务追踪（新 M 节）
