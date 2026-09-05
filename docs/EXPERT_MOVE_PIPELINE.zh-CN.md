# 专家迁移管线 + (L,E) 键驱逐 - 设计

[English](EXPERT_MOVE_PIPELINE.md) | [简体中文](EXPERT_MOVE_PIPELINE.zh-CN.md)

> 状态：**设计（2026-09）** - 一次工作会话敲定下一个 scheduler 重构。已落地的
> 批量 pin 层（slot_request bitmap / wake-once，见 WORK_IN_PROGRESS L）是本设计的基座。
> 本文档捕获**完整设计**，避免会话间丢失。
> 相关：`docs/WORK_IN_PROGRESS.md` L / J 节、`docs/Backend.md` §2/§4、
> `docs/EXPERT_SCHEDULER_DESIGN.md`、`src/backend/scheduler.{h,cpp}`、`src/backend/slot.h`。
>
> **2026-09-04 修订**：构建顺序 1-4 已落地（M1-M4，commit 6b6300e / dbaff8f /
> 45b14de / fdf4982）。M5（活跃槽记账）与 M7（文档收敛）未做；M6 部分。步骤 4 的
> v2r 现走 transfer-queue DMA，**推翻了 §5.2 的"只能 CPU memcpy"结论**——见该节
> 修订框与 `docs/VRAM_DMA_MOVE.md`。

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

- `slot_request_t` 96B POD：`{layer, total_tokens, start_rdtsc, n_load_target(=批装载数),
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
MOVING_OUT   槽 Y      这份正被搬走（v2r/r2v 源），不可 pin
                       （源槽保持 EVICTING 直到 move settle）
MOVING_IN    槽 Y      这份正被搬入（目标），不可 pin；槽 Y 是**已占用的
                       目标物理槽**（IO_INFLIGHT）——随状态一起发布，绝非占位
FAILED       无槽      装载/迁移失败
```

状态条目要点：
- 携带槽的状态发布时，`slot` 必须是**真实已预约的物理槽号**（LOADING = 其
  IO_INFLIGHT 槽；MOVING_IN = 已预约的目标槽）。与 §3.4 同一可见性规则：
  先发布带合法槽的意图、worker 再行动——绝不发布占位槽。
- **MOVING_OUT 与 MOVING_IN 都不可 pin**（try_pin 要求 READY），对称——见 §5.6。

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

**完成侧写序是镜像，同样必须遵守。** 计算线程扫到 `dir = READY` 时必须能 pin
物理槽。故完成收尾为：

```
slots_[s].mark_ready()            // 物理槽 -> READY（释放读者）
memory fence (seq-cst)
dir(L,E,p) : LOADING -> READY     // 至此计算线程扫到 READY 才合法
```

反序（dir 先 READY、物理还 IO_INFLIGHT）会让 scan 见 READY 去 try_pin 一个
仍在 IO_INFLIGHT 的槽——假唤醒 / pin 失败。装载完成、move 完成
（MOVING_IN -> READY）、demote 到 RAM 都用此顺序。

### 3.5 计算侧 pin 语义：无状态 exec

exec **刻意对驻留无状态**：它不读状态迁移、不区分 ABSENT / LOADING / MOVING_IN。
它只尝试 pin，把失败的重新请求：

```
exec pin_layer(bitmap):
  loop:
    for each needed (L,E):
      找 READY 副本（dir scan）-> try_pin -> 成功：记 handle
      否则（ABSENT / LOADING / MOVING_IN / pin 竞争）-> 记 still-need
    if 无 still-need: return 全部 handle
    提交一条请求 { layer, still-need bitmap, n_load_target = count }
    睡在批完成信号上（全部 still-need READY 时 wake-once）
    // 调度线程保证每个 still-need 专家变 READY（自己装 或 已有的在途装载），
    // exec 醒后重 pin；仍有竞争就循环
```

幂等、自愈：重试轮重测"仍非 READY"的集合，以**当轮重测数**重新提交
（绝不是 round-1 总数）。exec 不需要知道某专家是否已被人（含预取）装载——调度
保证它变 READY，届时唤醒 exec。

FAILED（装载/move 硬失败）作为错误上抛，不无限重试。

> 决策沿革：早前"决策 6a"（exec 因能看到 LOADING 而**不提交**在途专家）被取代。
> exec 在提交时无法可靠区分"会由我自己的请求装载"与"已为别人在装"（竞争 +
> 预取），故保持无状态，由调度拥有"让每个 still-need 变 READY"的职责。

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

### 5.2 v2r 设备读："只能 CPU memcpy"已被实测推翻（经 cached staging 走 DMA）

保留的原始论证（历史）：`vkCmdCopyBuffer` / copy engine 只能做 device↔device 与
device↔host-visible-vulkan-buffer。独显（RX 590）的 host-visible heap 仍是显存
（rebar BAR1）；128GB 系统 RAM 池不在 GPU 地址空间，copy engine 写不到它。
独显上不存在真正的 `vkCmdCopyBuffer` vram→系统 RAM 目标（只有 UMA/APU 有）。
跨后端 `ggml_backend_tensor_copy` 同样回落到 memcpy，或要求一个占显存的
host-visible 目标 buffer。当时结论：本硬件上 v2r 到普通 malloc RAM 只能 CPU
memcpy；GPU copy-engine DMA 记为未来 UMA/APU 目标。

> **2026-09-04 修订——该结论对"两步路径"是错的。** copy engine 确实写不进
> *普通 malloc* RAM（上面正确），但能写 **CACHED host-visible *vulkan* host
> buffer**（heap0 / memtype 7），再从该 host buffer 以 cached 速度读回很快。
> RX590 实测：`vkCmdCopyBuffer` vram → cached staging ~14 GB/s、staging → RAM 槽
> memcpy ~21+ GB/s ⇒ 单个 demote 从 ~158 ms（rebar CPU 读 0.02 GB/s）降到 ~0.5 ms。
> v2r 现为 transfer-queue DMA 经 ggml-vulkan 的 cached staging
> （`stmoe_vk_dma_read`，包装其内部 `ggml_vk_buffer_read`）+ 之后 memcpy 进 RAM 槽。
> 完整设计与被否决的整池 import 方案：`docs/VRAM_DMA_MOVE.md`。

### 5.3 任务形态（move_task）

一个专用 copy worker（数量参数化，先 1）。每专家迁移一个任务；批量提交。worker
只做纯逐列 memcpy，绝不碰控制面。

为什么只一个 worker（不用池）：v2r 的瓶颈是 **设备→host 链路带宽**，不是 CPU
核数——多 worker 不加带宽只加并发度。DMA 前是 rebar map 的 PCIe 读（0.02
GB/s）；按 §5.2 修订后是 transfer-queue DMA + staging memcpy（~14 GB/s 链路），
仍非 CPU 瓶颈。一个 worker（submit + fence wait + staging memcpy）就饱和；
数量留参数，将来链路更快或 UMA 目标再调。

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
  begin_evict(源槽)                 // READY -> EVICTING（挡住新 try_pin）
  dir_->clear(V, 源池)              // 不再可 pin
  while (源槽 refcount != 0) yield  // 等既有计算读者放完（通常 0）——
                                     // worker 绝不能读计算线程还持有的槽
  entries_(V, 源池)    : READY  -> MOVING_OUT
  dst = alloc_or_evict(RAM)          // dst IO_INFLIGHT（排他）
  entries_(V, RAM)     : ABSENT -> MOVING_IN
  enqueue move{V vram, RAM dst}      // 非阻塞；worker 拿到任务时源槽必已
                                     // EVICTING 且 refcount==0
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

**跨层 score 可比性警告**：`adaptive_scores_` 的分布可能逐层不同（某些层是全局
热点层）。**绝不跨层裸比 score 绝对值**。工作示例已规避这点——**层内**排序、
每层取 top K/2——保持。将来若改成一锅排序，先层内归一/排名（见 §8）。

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

## 7. 调度侧：per-model 单活跃请求槽

已定：批进度记账在调度线程；exec 在其请求整批 READY 时**只醒一次**。调度线程把
exec 请求串行为**每模型一个活跃请求**——任一时刻每个 `expert_scheduler`（本就
per-model，MULTI_MODEL_POOL）只服务一个 exec 请求。于是"谁在等"是每模型单个对象，
无需 waiter 列表。

### 7.1 活跃槽纪律

```
调度循环（global worker 轮询各 per-model scheduler；模型间互不阻塞——
每个有自己的活跃槽）：
  per model:
    if model.active 空:
      pop 一条 exec 请求 -> active  （记 still-need bitmap + n_load_target）
      for each still-need (L,E):
        ABSENT                      -> 发布 LOADING，启动自己的装载
        LOADING / MOVING_IN（在途，如预取或先前竞争）-> 不启动第二次装载；
                                      它已是别人的在途装载，会经 drain 到 READY
    else:
      继续 drain（active 进行中）；不 pop 新请求
  drain：每个 (L,E) settle READY（自装载 / 预取 / move 都算）：
      mark_ready + 状态 -> READY
      if (L,E) ∈ model.active.still-need: bump active.counter（一次）
      profile delta rdtsc（将来做，并行附加——不是完成的唯一职责）
  active.counter == 0 -> 唤醒 exec 一次 -> exec 重 pin -> 清空 active
```

因为 drain 是**任何槽变 READY 的唯一地点**，一个属于 active 请求的在途专家
（由预取或先前竞争装载）恰在它真正 READY 时 bump active 计数。任何地方都没有
"skip 算完成"，也不需要 per-(L,E) waiter 列表——active 请求是本模型唯一 waiter。

### 7.2 预取不是"没有完整下半的独立事件"——复用完整 DIO 路径

预取（调度主动的预测装载）只是同一 `async_dio_engine` 上的另一次 submit；其完成
落进**同一个 `drain_completions`**，执行完整 settle（mark_ready + dir set +
version bump + wake），外加 drain 级的 profile delta-rdtsc 记账作为并行附加。
它**绝不是"只记 rdtsc、无下半的独立事件"**。推论：

- 预取也必须先发布 `dir = LOADING` 再 submit（否则 exec scan 看不到"在途"、
  会重复装载；§3.4 顺序适用）；
- 预取的专家若在 settle 时属于 active 请求，经同一 drain 规则 bump active
  计数——不需要专门的"预取→active 通知通道"。

### 7.3 重试轮重置（exec 侧）

`pin_layer` 重试时，重新提交的 `n_load_target` 必须精确等于**当轮 scan 重测的
still-need 数**——绝不是 round-1 总数。调度线程按当轮自己的 n_load_target 倒计数。

## 8. Open questions / 后置

1. **score 线估算**："需要 K 槽时，层 L-delta 该用什么 score 阈值？"——后置
   （用户：可将来再说）。先按"最近层优先、按 score 取每层 top K/2"细化。
   **跨层裸比 score 绝对值保持禁止**（见 §6.1 警告）。
2. **A1 vs A2** directory 布局：推荐 A1（两个并行原子）。
3. `dir_->set` 语义是否彻底改成"预约 = LOADING、完成 = READY"两段（见 §3.4 顺序）。
4. **单活跃边界**：§7 模型把 exec 请求按模型串行（每模型一个活跃槽）。这匹配
   今天单 ctx 单 decode 线程的现实（exec 阻塞到本层请求完成才发下一条）。若同一
   模型将来跑多个并发 decode context，需要多活跃槽（每 ctx 一个）或 waiter 集
   ——标记，现在不建。

> 本会话已解决：move worker 数量 = 1（见 §5.3）。exec 无状态（见 §3.5）；预取
> 复用完整 DIO 路径（见 §7.2）；in-flight 重复无需专门处理——活跃槽 + drain
> 规则即覆盖（§7）。

## 9. 构建顺序（终态，按组件）

1. Directory：加宽为 (L,E) 键状态（A1）；owner_ 留到第 3 步编译通过。**[done - M1, 6b6300e]**
2. 装载路径：begin_reload 前先发布 LOADING（顺序不变量）；每模型单活跃请求槽
   （消灭重复装载窗口；§7.1）。**[先发 LOADING + state-aware accept 已做 - M2,
   dbaff8f；单活跃槽未做——属 M5]**
3. 驱逐：alloc_or_evict 内改 (L,E) 键层距选择；删 owner_（8 处用，全在此函数，
   已枚举）。**[done - M3, 45b14de]**
4. Move 管线：move_task ring + 1 worker（v2r + r2v）+ 完成 drain；把 v2r 接进
   device victim 的驱逐；r2v 留给将来放置策略。**[done - M4, fdf4982；v2r 现为
   DMA（§5.2 修订）- 717bac8]**
5. 调度侧记账：活跃槽计数 + drain 统一 bump + wake-once（§7）；exec 侧改无状态
   pin+resubmit（§3.5）。**[未做 - M5 open]**
6. 验证：纯 RAM IDENTICAL（如今天）；VRAM 单设备运行到达 `exec_mm_vk`
   （这是 K6/L6b 阻塞项）；directory 状态 + move ring + 驱逐顺序的 UT。
   **[部分 - M6：纯 RAM 路径不变（RAM 源 dev_buf==null → memcpy）、DMA 内容门
   0 BAD、VRAM demote 风暴消除；exec_mm_vk 到达 + 状态/move UT 未做]**

## 10. 涉及文件

- `src/backend/slot.h`——directory 状态加宽（A1）或新增 move-task 结构
- `src/backend/scheduler.h` / `scheduler.cpp`——驱逐、删 owner_、move worker + ring、批记账
- `src/pool/expert_stats.h`——仅审查：打分本就按 (L,E) 键
  （`stats_.adaptive_scores_[(L,E)]`）；(L,E) 键驱逐直接读它，无需槽反查
- `src/backend/minigraph_exec.cpp`——预期不动（exec API 不变：仍是 pin_layer），
  除非批信号移到调度线程侧
- `tests/test_slot.cpp`、`tests/test_scheduler.cpp`——新状态/move/驱逐 UT
- `docs/WORK_IN_PROGRESS.md`——任务追踪（新 M 节）
