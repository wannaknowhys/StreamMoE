# Route B 专家调度器设计 (EXPERT_SCHEDULER_DESIGN.md)

> 现状：已实现（`src/backend/scheduler.h/.cpp` + `src/backend/slot.h`）。本文记录控制面/计算面的数据结构与驱逐策略，供改动（异步 IO、GPU 池、cb_eval 生命周期）参考。
> 相关：`docs/Backend.md` §2/§4（slot_meta / expert_directory / MPSC）、`docs/MULTI_SUBPOOL.md`（按专家种类子池）、`docs/ROUTE_B_GPU_PHASE.md`（GPU 混合扩展）。

---

## 1. 概览：控制面 vs 计算面

- **每池一个 scheduler 实例**（MULTI_MODEL_POOL：主模型/draft 各一个；将来每 GPU 一个）。
- **计算面**（`graph_compute` / cb_eval 回调，单 decode 线程）：`pin_expert` / `wait_ready` / `unpin`。
- **控制面**（scheduler 后台线程）：`scheduler_loop`（pop 请求 → `alloc_or_evict` → `load_slot` DIO 读入 → `mark_ready`）。
- 通信：计算线程 → 控制线程走 **MPSC 有界队列**；控制线程 → 计算线程走 **共享数组写 + version 自增 + WakeByAddressAll/futex**。

## 2. 每池实例的数据结构（`expert_scheduler`）

| 成员 | 类型 | 语义 |
| :--- | :--- | :--- |
| `slots_` | `slot_meta[]`（64 位原子字数组）| **每 pool 一个**，长度 = `num_slots`（跨子池全局索引；异构组用 `subpool_t.slot_begin/n_slots` 分段）|
| `owner_[]` | `(layer, expert)` 对数组 | 槽 → 专家的**反向映射**（scheduler 私有，驱逐打分用）|
| `dir_` | `expert_directory` | **专家的数组**：`(layer, expert) → slot 或 UNASSIGNED`（`entries_` 原子 u32）+ `versions_`（每专家版本号）|
| `stats_` | `expert_stats_tracker` | EST1 热度（`get_adaptive_frequency`，读时归一化）|
| `staging_per_group_[]` | 每子池一块 staging 缓冲 | DIO 扇区对齐读的中转（组内专家布局相同，一块够）|
| `requests_` | `mpsc_alloc_queue`（4096）| 计算线程的分配请求（layer, expert, seq）|
| `subpools_` | `subpool_t[]` | 异构组（slot_begin / n_slots / expert_size / base）|

**回答：是不是"每 pool 一个 slots 数组 + 一个专家数组"——是。**
- `slots_` 按 pool（scheduler 实例）一个，子池只是同一个数组里的分段。
- `dir_` 是专家 → 槽的正向映射（`(layer,expert)` 维），`owner_` 是槽 → 专家反向映射，双向都有。

## 3. slot_meta：64 位原子字 + 状态机（`slot.h`）

```
bit 63........32 | 31........3 | 2 1 0
    generation   |  refcount   | state
```

状态：`EMPTY(0) / IO_INFLIGHT(1) / READY(2) / EVICTING(3) / FAILED(4)`。

| 转移 | 方法 | 说明 |
| :--- | :--- | :--- |
| 分配槽 | `begin_reload()` | → IO_INFLIGHT，generation++ |
| IO 完成 | `mark_ready()` | IO_INFLIGHT→READY（release），`WakeByAddressAll` |
| IO 失败 | `mark_failed()` | →FAILED，等待线程醒后抛错 |
| 计算面 pin | `try_pin()` | 仅 READY 可 pin，CAS refcount++，返回 generation（防御 ABA）|
| 计算面释放 | `unpin()` | refcount-- |
| 驱逐 | `begin_evict()` | READY→EVICTING（阻止新 pin）|
| 等待 | `wait_ready(gen)` | wait-on-address 等 READY 且 generation 匹配 |

等待原语：Windows `WaitOnAddress/WakeByAddressAll`；Linux futex（64 位字取低 32 位，醒后重读全字防 ABA）；其他 yield。**非忙等**。

## 4. expert_directory（`slot.h`）

- `entries_[(layer)*n_expert + expert]`：原子 u32，`slot` 或 `SLOT_UNASSIGNED`。计算线程只读，scheduler 线程写。
- `versions_[]`：每次映射变化 `fetch_add(1)` + wake——等待线程 `wait_version` 醒后重扫，无锁。
- `set()` = 写 slot（release）+ version++ + wake。

## 5. MPSC 分配请求队列

有界环形队列（cap 4096）。生产者满时 yield 等空间；消费者空时 sleep。`slot_request_t{layer, expert, seq}`。

## 6. 计算面 API（供 graph_compute / cb_eval）

```
pin_expert(layer, expert)  → expert_handle_t{slot, generation, ...}
  1. 查 dir → 命中 → slots_[s].try_pin()（READY 且 CAS refcount++）→ 返回 (slot, gen)
  2. UNASSIGNED → push 请求 → wait_version → 重查（循环，上限 100000 次）
  3. FAILED → 抛错
wait_ready(layer, expert)  → 已 pin 过（down 角色）：等 READY（不改变 refcount）
unpin(handle)              → slots_[slot].unpin()（refcount--，0 后可驱逐）
```

## 7. 调度面：驱逐与装载（`scheduler.cpp`）

**`alloc_or_evict(layer, expert)`**（组内）：
1. 先找组内 `SLOT_EMPTY` 槽 → `begin_reload` 占用。
2. 没有空槽 → **线性扫描**组内所有 `READY && refcount==0` 的槽，算 hybrid score 取最小 → `begin_evict` → `clear_directory` → drain refcount（yield 等归零）→ `begin_reload`。
3. 全在 pin / in-flight → 返回 -1，请求重入队（sleep 1ms 重试）。

**驱逐打分**（当前）：
```
score = 0.5 * freq + 0.5 * (1.0 - generation / 1e9)
freq  = stats_.get_adaptive_frequency(owner)   // EST1，读时按当前最大分归一化（B27 已修）
```

**回答：score 是纯每次循环算再取最小，没有树/堆。** `alloc_or_evict` 里 `for i = lo..hi` 线性遍历组内全部槽，逐槽算 score 保留最小 victim——O(n_slots)（当前 642~2297 槽，可接受）。无优先队列/树结构。

**`load_slot`**：`read_expert_sync`（同步阻塞 DIO）→ 成功 `mark_ready` + `dir_->set`；失败 `mark_failed`。

**`scheduler_loop`**：单 worker 线程。pop → 去重（dir 已有 skip）→ alloc_or_evict → load_slot。空队列 sleep 1ms。

## 8. 已知问题与改动路线

| 问题 | 现状 | 方向 |
| :--- | :--- | :--- |
| score 魔数 | `1e9` 归一化，generation<<1e9 时 recency 区分度≈0（实际只看 freq）| 改 `last_used_token` recency + 动态 α（REVIEW_2026_08_28）|
| 驱逐忙等无超时 | `while(refcount != 0) yield()` 无界自旋 | 加超时 + 日志（L/E + refcount）|
| 单线程 + 同步读 | scheduler_loop 单线程，`read_expert_sync` 阻塞，跨专家串行 | **异步 IO 引擎**（IOCP 并发 in-flight / io_uring / io_submit fallback），pin 提交后立即返回，IO 完成 wake（ROUTE_B_GPU_PHASE §3）|
| 线性扫描 | 槽数几千时 OK | GPU/多池后槽数×设备，考虑按组索引/堆优化驱逐候选 |
| 生命周期位置 | pin/unpin 在 `graph_compute` 按批猜角色 | 迁 cb_eval（hash 构建期预取 / argsort 首节点 wait / 汇合节点 unpin），见 ROUTE_B_GPU_PHASE §2 |
| 多池（GPU） | 每模型一个 scheduler | 每模型×每设备一个 scheduler；池选择（pin 时按热度/空闲/后端能力）|

## 9. 文件对照

- `src/backend/slot.h`：slot_meta / expert_directory（二维：`(layer,expert)×(pool)` + last_used_token）/ mpsc_alloc_queue / wait-wake。
- `src/backend/scheduler.h/.cpp`：expert_scheduler（init / pin / wait_ready / unpin / alloc_or_evict / scheduler_loop / load_slot / subpool 布局）。
- `src/io/async_dio_win.cpp` / `async_dio_posix.cpp`：DIO（Win IOCP 真异步 / POSIX 同步 pread，io_uring 待办 B22）。
- `src/pool/expert_stats.h/.cpp`：EST1 热度。

## 10. 频率与放置（GPU Phase 扩展）

专家存储整体视为**只读、可抛弃的多级缓存**（磁盘=源，CPU RAM=大慢层，GPU VRAM=快小层）——池里内容丢了不丢数据，只浪费一次重读。因此迁移可以"先拷贝后释放"保持全程合法，无需独占/双份。

### 10.1 每专家 EMA 实时频率

新增每 `(layer,expert)` 的 **EMA 实时频率**（pin 时更新，比 EST1 平滑、实时）：

```
freq_ema[e] = lambda * freq_ema[e] + (1 - lambda) * 1    // lambda ~ 0.95
```

放在 `expert_directory`（正向，跨池共享属性，与 last_used_token 并列）。

### 10.2 放置决策（pin 时，全局调度线程做）

新专家读入时按 `freq_ema` 决定进哪层：

- `freq_ema` 高 → 放进 **GPU 池**（IOCP DirectIO 直读 HOST_VISIBLE VRAM——反正高频复用，值得占显存）
- `freq_ema` 低 → 放进 **CPU RAM 池**

### 10.3 迁移（全局调度线程驱动，Phase C）

- 判据：GPU 最冷专家（freq_ema 最低） vs CPU 某专家（freq_ema 高出一个阈值）——满足则互换。
- **滞回阈值 + 冷却期**：热度差要超过固定 margin 才触发；每专家迁移后冷却 N token，防来回搬（抖动耗 PCIe）。
- **先拷贝后释放**：device→host 读（或 host→device 写）完成后再释放/覆盖槽——非独占 cache 语义，全程合法；用 slot_meta 的 generation 绑定读操作防脏读。
- **紧急 pin 优先**：后台迁移的每步可中断；锁定"最冷"时若发现它正被真实请求 pin（refcount>0）→ 放弃这次迁移，让位给紧急 pin。
- **攒批评估**：每 N decode step 或某侧命中率下滑时评估一次，非常驻高频轮询。

### 10.4 与 EST1 的关系

`get_adaptive_frequency`（EST1，读时归一化）用于驱逐打分；EMA 频率用于放置/迁移判据。两者并存：打分用 EST1 热度 + recency，放置用 EMA 实时频率。

