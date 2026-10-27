# 设备侧 fold + 提交单元 arena + head 融合（设计）

[English](DEVICE_FOLD_HEAD_FUSION.md) | [简体中文](DEVICE_FOLD_HEAD_FUSION.zh-CN.md)

> 状态：**A 已确认（2026-10-26）。B 的选型待定（2026-10-27）**：提交单元窗口需要先把
> 同步边界张量钉住（见 §4.1b），所以 B0 拆成 **B0a = 按层 parity**（最小改动，推荐）与
> **B0b = 同步边界 carry + 单元窗口**（更省内存、改动更大，延后）。针对唯一执行器
> `exec_layer_burst_chain_buckets`（`src/backend/minigraph_exec.cpp`）的两项改动：
>
> - **A. 设备侧 fold**（已确认）：把每池 partial accumulator 与 `moe_out` 留在 tail 所在
>   设备，不再把每个设备 partial 读回 host 做 host fold（只影响 multi-target 路径）。
> - **B. 提交单元 arena + 每层单次提交**：以提交单元
>   `[closure(L) + tail(L) + head(L+1)]` 为单位做 liveness 打包（单元内按活跃区间复用，
>   单元之间复用同一窗口），再把这个单元融成一张设备图。这取代了之前的"按层 parity"：
>   parity 用 2×scratch 绕开别名，而单元规划是**证明**复用合法，且不额外占内存。
>
> 相关：`M2_DEVICE_EXECUTOR.md`（§7.8 出口、§7.9 设备 round）、`PER_DEVICE_ARENA.md`
> （§6 区域、§7 open items）、`DEVICE_DENSE_CLOSURE.md`（3.6 每设备 `moe_out`、
> 3.8 grow-only）、`ASYNC_PIPELINE_ROADMAP.md`（Phase 3）、`BUCKET_FAST_PATH.md`
> （P1-b 单 target）、`BENCHMARK.md`（2026-10-26 dispatch 实测）。

## 1. 为什么：实测数字

2026-10-26（RX590 Vulkan，olmoe，23 个 decode token，16 层）：

| 配置 | eval/token | head | fused tail+head | MoE 闭包 | relay |
| :--- | ---: | ---: | ---: | ---: | ---: |
| place-cpu | 20.6 ms | 0.4 | 5.5（15×0.37） | 12.7（16×0.79） | 0 |
| place-c1 | 75.7 ms | 2.7 | 49.3（15×3.29） | 21.2（16×1.33） | 4.4 |
| place-exp5 | 60.0 ms | 1.1 | 19.4（15×1.29） | 37.3（16×2.33） | 0 |
| place-c1c2-exp5 | 80.8 ms | 40.8（16×2.55） | -（Tier-2） | 40.1（16×2.50） | 0.1 |

提交微基准（`diagnostics/vk_launch_probe.cpp`）：一次 Vulkan 图提交 ≈ 0.17 ms 固定 +
**每节点 ≈ 39 µs GPU + 17 µs CPU 录制**。小算子上逐节点开销占主导（256² MUL_MAT 约
13%，2048² 约 0.5%）。结论：

- A/B 都**不能**减少逐节点成本，它们省的是**同步往返、每层一次提交、host fold**——
  在 RX590 上约 5~10%，在 deepseek/P100 上比例更小（节点更大）。
- 仍然值得做：它们是 (a) 混合池目标（DeepSeek on RAM + P100）和 (b) CUDA 上的
  每层图重放的结构前提，后者的收益会大得多。

## 2. 当前代码事实（今天真正跑的路径）

`exec_layer_burst_chain_buckets` 出口：

- `direct_out`（单池单满 round）：`per_token` 就是答案；`moe_out_same_dev` 时原地写
  （[:2406-2413](src/backend/minigraph_exec.cpp:2406)、
  [:2632-2642](src/backend/minigraph_exec.cpp:2632)）；`moe_out` 在 host 时一次 D2H。
- `single_dev_target`（全部 round 在同一设备、无 CPU round）：`t.acc` 即完整结果，
  原地/D2H/D2D（[:2643-2656](src/backend/minigraph_exec.cpp:2643)）。
- **其余（multi-target：RAM round + 设备 round）**：每设备
  `ggml_backend_tensor_get(t.acc, dev_accs[di])`
  （[:2658-2670](src/backend/minigraph_exec.cpp:2658)），再 `layer_fold()`
  （[:2727-2728](src/backend/minigraph_exec.cpp:2727)）在 host 求和并
  `tensor_write_host(moe_out, ...)`（[:1290-1310](src/backend/minigraph_exec.cpp:1290)）。
  CPU round 累加在 `c.acc_d`（[2437-2441](src/backend/minigraph_exec.cpp:2437)）。

`moe_out`（`ffn_moe_out`）是被捕获节点；闭包设备是真实设备时，其 `buffer`/`data`
指向该设备 `plan.buf`（[route_b_chain.cpp:640-646](src/backend/route_b_chain.cpp:640)）。
所以 GPU 路径上的 fold = D2H + host fold + H2D 写回设备 buffer。

scratch：`plan.scratch_size = max over layers`，每层 offset 都从 0 在**同一窗口**里打包
（[route_b_chain.cpp:506-578](src/backend/route_b_chain.cpp:506)）；节点 data =
`base + carry1 + carryN + scratch_off`；`ex->result_bytes` 就是这个层窗口（`lb`），而闭包
运行期 bump（`pool_bump`，[:2195-2211](src/backend/minigraph_exec.cpp:2195)）从
`base = result_bytes` 开始，即**在静态区之后**。这既是跨层融合别名的原因，也说明运行期
bump 本来就在静态布局之外。

Tier-2 在"单设备 target"时已融合 `MoE(L)+Tail(L)`
（[minigraph_exec.cpp:2464-2530](src/backend/minigraph_exec.cpp:2464)）；head 融合被硬关
（[:2526](src/backend/minigraph_exec.cpp:2526)），`out_fused_head_next` 从未置 true，所以
[:3348](src/backend/minigraph_exec.cpp:3348) 的 `continue` 是死代码。

carry：`carry1` 是 cross-1 边界张量集合、**按层奇偶双缓冲**
（[route_b_chain.cpp:462-481](src/backend/route_b_chain.cpp:462)）；`carryN` 作为 retained
打包（[483-494](src/backend/route_b_chain.cpp:483)）；两者都在设备边界 relay。KV 归
llama，不在 arena 里。

## 3. 设计 A —— 设备侧 fold（已确认）

**决定（2026-10-26）：按原设计通过，先于 B 实现。**

### 3.1 目标 / 非目标

目标：一层的 round 同时落在 host 与一个或多个设备上时，在 **tail 所在设备**上产出
`moe_out`，不做 host fold 往返，并把设备同步从 2 次降到 1 次。

非目标：改单 target 路径（已 elide）；P2P 设备互拷（本栈没有 semaphore 暴露）；把 CPU
round 搬到设备。

### 3.2 owner 选择

`owner = dev_of[moe_out]`（tail 设备），前提 `moe_out->buffer` 是设备 buffer。
若 `moe_out` 在 host（C1 在 CPU），保留当前 host fold（该路径本来就只有 D2H、没有 H2D）。
若本层没有设备 round，不变。

### 3.3 非 owner partial 的传输

- CPU partial（`c.acc_d`）：owner 设备上的**主机可见 staging tensor**（与 `g_pinned_h2d`
  同 host buft，grow-only，每层一片，不与传输 staging ring 共用）。CPU 图跑完后 host
  memcpy 写入；fold 图把它当普通 operand 读。Vulkan 支持 host-buft operand（现在的 stage
  leaf 就是这么用的），所以**不需要 `set_tensor`，也不额外同步队列**。
- 另一块**物理设备**的 partial：提交 fold 图前读回主机可见 tensor（D2H + 该设备 fence），
  fold 图直接读它。这是没有 semaphore 时不可避免的一次同步；但仍比今天省（不再有
  `moe_out` 的 H2D、不再 host fold，且 tail 可以并进同一张图）。
- 同一物理设备上的多个 chunk/pool：acc 在同一设备 buffer 里，fold 图直接读，无需拷贝。

### 3.4 图的形状

```
owner D 上的 fold(L)：
    moe_out = acc_D
            + staged_cpu_partial              (host-visible operand)
            + staged_other_dev_partial_i ...  (host-visible operands)
```

用 `ggml_add` 组成一个小 `ggml_cgraph`，追加到该设备 target 的图（tail 融合时就是
`single_dt->gf`；未融合时单独提交）。节点数很小（1 + N_partials），逐节点成本不是问题；
收益在去掉 host fold 和一次同步。

`direct_out` / `single_dev_target` 已有 elision：当 owner accum 已经等于 `moe_out` 时
**不要**再跑 fold 图。

### 3.5 控制流（multi-target + 单物理设备）

今天：

```
提交设备 MoE 图（async）
跑 CPU 图                     -> c.acc_d
sync 设备                     # 1
每设备 acc D2H                # 拷贝
host fold -> host vector
tensor_write_host(moe_out)    # H2D
提交 tail+head 图（Tier-1）
sync                          # 2
```

改造后：

```
提交设备 MoE 图（async）
跑 CPU 图                     -> c.acc_d
把 c.acc_d 写进 host-visible partial（memcpy，无设备工作）
[其他物理设备：D2H 其 acc -> host-visible partial + fence]
在 D 上提交 fold+tail(+head) 图   # 读 acc_D 与 host partials
sync                          # 1
```

### 3.6 数据结构

```cpp
struct fold_plan_t {
    int device_pool = -1;            // owner pool（-1 = 走现在的 host fold）
    bool needs_cpu_partial = false;  // c.acc_d -> host-visible staging
    std::vector<int> foreign_pools;  // 需要先读回的设备 pool
    // 图片段直接追加进 owner target，不引入新 buffer 类型
};
```

Staging：每设备一块 grow-only 主机可见 buffer，大小
`n_d_out * n_t * 4 * (1 + n_foreign)`；每层复位（调用线程串行执行层）。

### 3.7 正确性不变量

1. 每层 `moe_out` 只被写一次：owner fold 图（或现有 elision）。
2. 所有参与 partial 在 fold 图提交前写完：设备 partial 由 D 队列顺序保证；外部设备由
   显式 fence 保证；CPU partial 由提交前 memcpy 保证。
3. fold 图在 D 队列上排在 MoE 图之后 → 顺序由队列保证，不靠 wall clock。
4. 求和顺序变化（设备 partial + staged partials，而不是 CPU 优先的 host 顺序）。ulp 级
   差异，走项目宽松 gate（maxAbs ≤ 1e-5 / cos ≈ 1.0），不追字节一致。
5. 末层/导出：最终 `moe_out`（及被导出张量）仍照今天读回 host；A 不改变最后一次 D2H。
6. 谓词失败（owner 在 host、`moe_out` 不连续、operand 设备不支持）→ 该层回退 host fold；
   这是**结构性回退**，不是编译期开关。

### 3.8 失败模式与处置

| 失败 | 处置 |
| :--- | :--- |
| staging 分配失败 | 该层回退 host fold，log 一次 |
| 外部设备 acc 读回失败 | 上抛 GGML_STATUS_FAILED |
| fold 节点设备不支持（`supports_op`） | host fold 回退 |
| `moe_out` 不连续 / 在 host | host fold 回退（当前路径） |
| 外部物理设备漏 fence | 必须是断言：plan 携带 `foreign_pools`，提交路径逐个 fence |

### 3.9 验证

- `build.bat llamalibs dual` + `build.bat test main`（6/6）。
- `run_baseline.bat moe_129_8192_vk`：moe/upstream gate 仍 PASS；vk gate 是宽松 cos，
  允许 fold 顺序变化，但比例不能明显变化。
- `verify_places --flavor StreamMoE`：12 拓扑 0 ERROR；8 个逐字一致的拓扑必须保持逐字
  一致**除**多池拓扑（fold 顺序变化）——记录在案，不能新增 DIFF。
- TMR 前后对比：`tail_read`/`tail_fold` 应降到 ~0，`tail_sync` 次数减少 1。
- expert history 不变（路由与 fold 无关）。

## 4. 设计 B —— 提交单元 arena + head 融合（B0 选型待定）

### 4.1 真正的阻塞

`layout_arena` 给每层同一个 scratch 窗口（`scratch_size = max`、offset 从 0）。把
`MoE(L)+Tail(L)` 与 `Head(L+1)` 并进一张图，两层临时量落进同一 offset；2026-10-07 的
CHECKPOINT 记录过这种并图造成 GPU 地址碰撞（输出乱码）。**router ids 回读不是阻塞**：
ids(L+1) 在图返回后才读，正好是当前每层已有的那次同步点。

### 4.1b 发现（2026-10-27）：ids barrier 切在层内部

`layout_arena` 按层打包，是因为**整层就是一个分配窗口**：head、closure、tail 共用它，
所以跨 ids barrier 的张量（`ffn_norm-*` = 闭包的 `cur`、`ffn_moe_*` ids/topk/weights、
残差）就是普通 scratch 节点，一个 offset 搞定。

而单元 `[closure(L) + tail(L) + head(L+1)]` 把 `head(L+1)` 和 `closure(L+1)` 分到了**不同
单元**，于是 head(L+1) 产出、closure(L+1) 消费的张量要跨单元边界。它们今天**不是**
`carry`（carry 的定义是"被不同层消费"，[route_b_chain.cpp:277-287](src/backend/route_b_chain.cpp:277)），
也没有 relay——就是普通 scratch。单元窗口跨单元复用，就会在 closure(L+1) 读之前被
unit k+1 的临时量覆盖。

plan dump 证据（`STREAM_MOE_TMP_DUMP_PLAN=1`，olmoe，place-cpu）：

```
[ 30] ffn_norm-0        MUL  L0  scratch off=41943040 dev=CPU(host)
[ 38] ffn_moe_gate-0    MUL_MAT_ID L0  scratch off=8388608 ...
```

`ffn_norm-0`（闭包的 `cur`）是 L0 scratch 节点，消费者 `ffn_moe_gate-0` 也在 L0 闭包——
所以 `carry` 里没有它。

因此 B0 有两个可行方案。

### 4.2a Option B0a —— 按层 parity（推荐）

保留现有按层窗口，层 L 用槽位 `L & 1`：

- `stride = max over layers(层打包大小)`；`scratch_size = 2 * stride`；
- 节点 offset = `off_scratch + (L & 1) * stride + scratch_off_L[node]`；
- `ex->result_bytes = 2 * stride`（运行期 bump 基址必须在**两个槽位之上**，否则层 L 的
  prefill 级 bump 会溢出到 head(L+1) 的槽位）；
- `plan.need` 随 `scratch_size` 变大；CPU full-alloc block 因为索引同一份 `out_off`，
  会跟着覆盖 parity 偏移；
- 只对真实设备 plan 生效（`D != ""`）；host plan 保持 1×（没有设备图可融合，纯 CPU 不变）。

代价：每设备 2× 静态 scratch（olmoe prefill 71MB→142MB；gemma prefill 128MB→256MB；
bump 不变）。不重分类、不改 relay，布局改动逐字节一致（只动 offset）。

### 4.2 Option B0b —— 提交单元窗口（延后：需要 4.1b）

**前置（4.1b）**：必须先把 head→closure 集合重分类为"同步边界 carry"并钉住，否则窗口
复用会覆盖它。这个集合不大但不少：`cur`（norm 之后的 hidden，`ffn_norm-*`）、router
ids/topk/weights（`ffn_moe_*`）、以及 tail 消费的残差流。它们今天经 xfer shell 跨设备，
挪进 `carry1` 会连带改 carry/relay 集合——这就是延后的部分。

执行器每 token 的结构是：

```
U0      = [ head(L0) ]
U_k     = [ closure(L_k) + tail(L_k) + head(L_k+1) ]      k = 0 .. n-2
U_last  = [ closure(L_{n-1}) + tail(L_{n-1}) ]
```

每个单元以 ids 回读（每层唯一 host break）结尾，即结束在**同步点**。单元内节点严格按
closure -> tail -> head 的顺序执行；单元之间可以复用同一窗口，因为同步已经回收所有在飞
读取。

于是规划变成：

- 每个单元的非 carry、非 view 节点，沿用现有 `pack_interval` 做 liveness 打包（按设备），
  时间轴跨整个单元（`head(L+1)` 的节点排在 `tail(L)` 之后）；
- `unit_stride = max over units(打包大小 + 运行期 bump 上界)`；
- `scratch_size = unit_stride`，每个单元都用同一窗口（offset 0），和今天按层复用完全
  同构，只是"单元"更大；
- 闭包的 `out_off` 与 `result_bytes` 取单元中闭包那一段的打包结果；运行期 bump
  （`pool_bump`）仍从 `base = unit_static` 开始（静态区之后），与今天一致；
- CPU full-alloc（`moe_chain_set_full_alloc`）用同一个单元窗口尺寸。

### 4.3 为什么合法（以及为什么 B0a 是安全子集）

- **单元之间**复用只有在"跨边界的张量全部被钉住"时才合法。ids barrier 就是边界，而
  head→closure 集合今天**没有**被钉住（4.1b），所以 B0b 必须先补这个钉住。
- 钉住之后，单元边界由**放置和图结构**（每层一个 head、每层一个 ids barrier）决定，
  **不由运行时路由决定**：一个 token 的专家落 RAM 还是 VRAM 只改变跑哪些 round，不改变
  节点集合与顺序。所以单元规划在 build 期确定、跨 token 稳定。
- B0a（parity）是同一不变量的无重分类版本：两个窗口让融合对天然不相交。它作为第一步，
  因为逐字节一致且不碰 relay 路径。

### 4.4 单元规划覆盖什么

| buffer | 在单元内？ | 由谁处理 |
| :--- | :--- | :--- |
| dense head/tail 临时量 | 是 | 单元 liveness 打包 |
| 闭包静态节点（`out_off`、`result_bytes`） | 是 | 单元 liveness 打包 |
| 闭包运行期 bump（`pool_bump`：孪生/gather/fold） | 是（黑盒） | 预留 `unit_static` 之后，按设备取上界 |
| tail(L) 产出、head(L+1) 消费的张量 | **是** | 单元 scratch（今天它是 carry；B0b 里变为内部，同设备时 relay 消失） |
| tail/head 跨设备边 | 否（两张设备 plan） | xfer shell（目前独立区域；以后可并入单元打包） |
| closure(L) 的 `cur`（由 head(L) 产出） | 否（跨 ids barrier） | **今天只是普通 scratch——B0b 必须先变成"同步边界 carry"** |
| router ids/topk/weights（head → closure） | 否（同一 barrier） | **同上：B0b 要先钉住** |
| cross-N retained 状态 | 否 | `carryN` |
| KV cache | 否 | llama 管理 |

也就是说：**生命周期完整落在单元内的东西全部并入同一份 plan**；只有跨 ids barrier 的
`carry1` 与 retained 的 `carryN` 保持特殊。这与"除了 cross buffer 之外都一次性搞定"一致
——另外要补充：cross-1 是被 ids barrier 逼出来的，不是 plan 的缺陷。

### 4.5 多残差流

plan 是逐张量的，所以 k 条残差流就是每个边界 k 个张量：

- 单元内消费完的流 → 单元 scratch；
- 跨 ids barrier 的流 → `carry1`，它的边界集合本来就是"集合"（parity 作用于整个集合，
  是集合 2×，不是每条流 2×）；
- retained 的流 → `carryN`。

不需要新机制。但本设计之外仍有两个假设：**每层一个 MoE 闭包**（`g_layer_exec` 每层一个
`moe_layer_exec_t`，闭包取第一个 `MUL_MAT_ID`），以及每层一个 accumulator
（`acc_d`/`moe_out`）。如果将来某架构每层有多个 MoE 块，那是另一项改动。

### 4.6 注意 / 非目标

1. **运行期 bump 区**不在静态 liveness 模型里。保留今天的规则
   （`base = unit_static + acc`，bump 在后），保证它不可能压到 tail/head 静态区；每设备
   上界 = `max over layers(pool_bump)`。
2. **每设备两块 buffer**：设备 arena 与 CPU full-alloc block 都按 `out_off` 索引，两者
   都要用单元窗口定尺寸。
3. **xfer shell** 第一步不吸收（保持独立区域，按 consumer layer 打包，与今天一致）。
4. 单元规划是**纯布局**改动，必须在当前"两图/两次提交"路径上验证**逐字节一致**之后，
   才允许打开融合。

### 4.7 head 融合的构造（单元规划落地之后）

- 在 Tier-2 块（[:2464-2530](src/backend/minigraph_exec.cpp:2464)）里，append 完 tail
  节点后，按拓扑序 append `Head(L+1)` 的 dense 节点（它们已在单元窗口内，offset 已经
  规划好）。
- **提交前**派发 `Head(L+1)` 的 incoming relay（同设备放置时多数是内部边；外部的是
  KV/positions，已在设备上）。图里不能出现"源在本图更晚才产出"的 relay。
- 置 `out_fused_head_next = true`；调用侧循环已有
  `fused_tail && fused_head_next -> continue`（[:3348](src/backend/minigraph_exec.cpp:3348)）。
- ids(L+1) 由下一轮 `exec_moe_closure(L+1)` 在本层 sync 之后读取；pin 路径不变。
- 保留 Tier-1 作为回退。

### 4.8 资格谓词（每层、运行时结构判定）

```
fuse_head_next(L) =
    Tail(L)、Head(L+1)、moe_out 的放置设备同为 D
    && Head(L+1) 存在且其全部非 alias 节点都在 D
    && 这些节点都通过 ggml_backend_supports_op(D)
    && Head(L+1) 没有"由本图自己产出"之外的外部跨设备输入
    && 不是最后一层
```

不满足就保持今天的行为（Tier-1 融合 tail+next head，或分开 head）。

### 4.9 正确性不变量

1. 单元内任意两节点不得共享 scratch 字节，除非在单元执行顺序下活跃区间不相交；跨单元
   边界复用必须被同步分隔（按构造它一定被分隔）。
2. 融合节点表按拓扑序；tail(L) 在 head(L+1) 之前，且 head 节点不得读取尚未发出的 tail
   节点。
3. `executed_relays` 记账在融合调用内共享，保证 tail/head relay 只派发一次。
4. 每层 Tier-1 与 Tier-2 互斥（Tier-1 分支加断言 `!(fused_tail && fused_head_next)`）。
5. 最后一层没有 `Head(L+1)`；`exec_moe_closure(-1)` 路径不变。
6. 数值：单元规划（只改布局）与融合（只改提交分组）都不改变算术 → 两者都应与当前路径
   **逐字节一致**。出现差异一律当 bug。

### 4.10 失败模式与处置

| 失败 | 处置 |
| :--- | :--- |
| head 节点在 D 上不支持 | 谓词 false → Tier-1 回退 |
| 单元窗口分配失败（设备 OOM） | 明确失败；绝不静默别名 |
| relay 的源在图中更晚产出 | 谓词 false（建图前检查） |
| 融合图超过 `gf->size` | 扩容（禁止今天的静默丢节点；[:2508](src/backend/minigraph_exec.cpp:2508) 要改成硬错） |
| ids(L+1) 缺失 / 0-token | 保留 B38 的 0-token 提前返回 |

### 4.11 验证

- 步骤 B0a（只做 parity、不融合）：`run_baseline` PASS 且 moe/upstream **IDENTICAL**；
  `verify_places` 不变（8 逐字 + 同样 4 个已知 DIFF）；smoke PASS。这是强门禁：布局改动
  不得移动数值。
- 步骤 B1（融合）：gemma + deepseek 逐层 `STREAM_MOE_TMP_DUMP` 的 `moe_out`/hidden 对
  B0a 必须逐字节一致；TMR `head` 桶降到 1 次/token、每层提交数变为 1；12/12 拓扑 0 ERROR。
- TMR 前后：`chain_tail`/`tail_sync` 与提交次数。

## 5. 顺序、成本、预期收益

| 步骤 | 触及 | 风险 | 预期 |
| :--- | :--- | :--- | :--- |
| B0a. 按层 parity（纯布局、不融合） | `layout_arena` scratch 段（槽位 + `scratch_size`）、`result_bytes = 2*stride`、设备 `arena_used` | 低；逐字节一致 | 解除别名约束；每设备 2× 静态 scratch；不谈性能 |
| B0b. 同步边界 carry + 单元窗口（可选、延后） | carry 分类 + relay 集合 + `layout_arena` | 高（relay 路径） | 窗口 1× 而不是 2×；carry 每个边界多约 1 个 hidden |
| A. 设备侧 fold | 出口块 + fold plan + staging | 中（multi-target 路径） | 混合池每层省 1 次 sync、1 次 D2H/H2D（olmoe ~0.2-0.4ms/层；deepseek 更大） |
| B1. head 融合 | Tier-2 块 + 循环 | 中（B0a 之后别名按构造已消） | 每层少 1 次提交（RX590 ~0.2-0.3ms/层；CUDA replay 后更大） |

B0a 是 B1 的前置，可先落地（行为中性）；A 独立、混合池必需；B1 把目前死掉的
`can_fuse_head_next` 路径接通；B0b 是以后的内存优化。三者都不替代 `BENCHMARK.md` 测到的
逐节点 dispatch 成本：RX590 上即使零断点，decode 上限也就 20~40 t/s。目标（P100/CUDA）
必须先用同一个 probe 测，再决定 B1 要走多远。

## 6. 待确认

1. 多物理设备的 foreign partial 路径（A）现在就做，还是只做"同一物理设备"？
   （目标是单 P100 + RAM，第一里程碑不需要前者。）
2. B0a 还是 B0b：接受 parity（2× 静态 scratch、不碰 relay），还是等同步边界 carry 重构
   （窗口 1×，但要改 carry/relay 集合）？（建议先 B0a；只有 P100 显存吃紧才做 B0b。）
3. B0 是否顺手把 `xfer` shell 吸收进单元打包，还是保持独立 xfer 区域、以后再并？
   （建议以后——一次只改一件事。）
4. B0 是否顺手把 tail(L)→head(L+1) 的内部 carry 降级为单元 scratch？
   （单元建好后很便宜，但会改 relay 集合；建议等 B0 验证逐字节一致后单独一步。）
5. 将来 CUDA replay：round-shape cache 用 `(layer, stage, width, n_active)` 作键够不够？
   （decode 的 width ∈ [0..n_k]。）
