# 设备侧 fold + 每层单次提交（设计）

[English](DEVICE_FOLD_HEAD_FUSION.md) | [简体中文](DEVICE_FOLD_HEAD_FUSION.zh-CN.md)

> 状态：**设计提案，2026-10-26；未实现。** 针对唯一执行器
> `exec_layer_burst_chain_buckets`（`src/backend/minigraph_exec.cpp`）的两项改动：
>
> - **A. 设备侧 fold**：把每池 partial accumulator 与 `moe_out` 留在 tail 所在设备，
>   不再把每个设备 partial 读回 host 做 host fold（只影响 multi-target 路径）。
> - **B. 每层单次提交**：把 `[MoE(L) + Tail(L) + Head(L+1)]` 合成一张设备图；需要
>   **per-layer scratch parity**，因为当前 arena 把所有层共用同一段 scratch 窗口。
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
（[route_b_chain.cpp:565-578](src/backend/route_b_chain.cpp:565)）；节点 data =
`base + carry1 + carryN + scratch_off`。这就是跨层融合会别名的原因。

Tier-2 在"单设备 target"时已融合 `MoE(L)+Tail(L)`
（[minigraph_exec.cpp:2464-2530](src/backend/minigraph_exec.cpp:2464)）；head 融合被硬关
（[:2526](src/backend/minigraph_exec.cpp:2526)），`out_fused_head_next` 从未置 true，所以
[:3348](src/backend/minigraph_exec.cpp:3348) 的 `continue` 是死代码。

## 3. 设计 A —— 设备侧 fold（multi-target）

### 3.1 目标 / 非目标

目标：当一层的 round 同时落在 host 与一个或多个设备上时，在 **tail 所在设备**上产出
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

## 4. 设计 B —— 每层单次提交（scratch parity + head fusion）

### 4.1 真正的阻塞

`layout_arena` 给每层同一个 scratch 窗口（`scratch_size = max`、offset 从 0）。把
`MoE(L)+Tail(L)` 与 `Head(L+1)` 并进一张图，两层临时量落进同一 offset；2026-10-07 的
CHECKPOINT 记录过这种并图造成 GPU 地址碰撞（输出乱码）。**router ids 回读不是阻塞**：
ids(L+1) 在图返回后才读，正好是当前每层已有的那次同步点。

### 4.2 parity 布局

- `plan.scratch_stride = max_L(lb_L)`（今天的 `scratch_size`）。
- `plan.scratch_size = 2 * scratch_stride`（parity 0/1）。
- 层 L 的基址 = `off_scratch + (L & 1) * scratch_stride`；`scratch_off[node]` 相对该基址。
- 闭包节点 `ex->out_off[i]` 保持相对；执行器的 `moe_chain_fullalloc_buffer`/`bind_fresh`
  基址加上同一 parity 偏移，`result_bytes = scratch_stride`。
- CPU full-alloc block（`moe_chain_set_full_alloc`）同样按 2× 申请（保持一致；将来 CPU
  侧也可能并图）。
- `xfer` shell 已按 consumer layer 键控，不受影响。
- 内存成本：每设备 +scratch_stride（olmoe decode 8MB→16MB，prefill 128MB→256MB；
  gemma/deepseek 同量级）。可接受。

### 4.3 资格谓词（每层、运行时结构判定）

```
fuse_head_next(L) =
    Tail(L)、Head(L+1)、moe_out 的放置设备同为 D
    && Head(L+1) 存在且其全部非 alias 节点都在 D
    && 这些节点都通过 ggml_backend_supports_op(D)
    && Head(L+1) 没有"由本图自己产出"之外的外部跨设备输入
    && 不是最后一层
```

不满足就保持今天的行为（Tier-1 融合 tail+next head，或分开 head）。

### 4.4 构造

- 在 `exec_layer_burst_chain_buckets` 的 Tier-2 块
  （[:2464-2530](src/backend/minigraph_exec.cpp:2464)）里，append 完 tail 节点后，按拓扑序
  append `Head(L+1)` 的 dense 节点。
- **提交前**派发 Head(L+1) 的 incoming relay（同设备放置时多数是内部边；外部的是
  KV/positions，已在设备上）。图里不能出现"源在本图更晚才产出"的 relay。
- 置 `out_fused_head_next = true`；调用侧循环已有
  `fused_tail && fused_head_next -> continue`（[:3348](src/backend/minigraph_exec.cpp:3348)）。
- ids(L+1) 由下一轮 `exec_moe_closure(L+1)` 在本层 sync 之后读取；pin 路径不变。
- 保留 Tier-1 作为回退：head 不可融合的层仍走 `exec_fused_tail_and_head`。

### 4.5 与 A 的关系

- 同设备（单池）：A 是 no-op（已 elide），B 可单独落地。
- 混合池 + 单物理设备：B 需要 A（fold 必须在 D 上、先于 tail，而 tail 在融合图里）。
  先 A 后 B，B 的谓词再纳入混合情况。
- 多物理设备：B 的谓词保持 false（head 可能在别的设备），继续 Tier-1。

### 4.6 正确性不变量

1. 融合图内任何两节点不得共享 scratch 字节，除非可证明同时死亡。parity 按层保证；层内
   仍按现有 `pack_interval` 顺序。
2. 融合节点表按拓扑序；tail(L) 在 head(L+1) 之前，且 head 节点不得读取尚未发出的 tail 节点。
3. `executed_relays` 记账必须在融合调用内共享，保证 tail/head relay 只派发一次。
4. 每层 Tier-1 与 Tier-2 互斥（Tier-1 分支加断言 `!(fused_tail && fused_head_next)`）。
5. 最后一层没有 `Head(L+1)`；`exec_moe_closure(-1)` 路径不变。
6. 数值：并图不改变任何算术，只改变提交分组 → 输出应与两图路径**逐字节一致**（同内核、
   同顺序）。出现差异一律当 bug。

### 4.7 失败模式与处置

| 失败 | 处置 |
| :--- | :--- |
| head 节点在 D 上不支持 | 谓词 false → Tier-1 回退 |
| scratch parity 未分配到（设备 OOM） | 谓词 false + log（绝不静默别名） |
| relay 的源在图中更晚产出 | 谓词 false（建图前检查） |
| 融合图超过 `gf->size` | 扩容（禁止今天的静默丢节点；[:2508](src/backend/minigraph_exec.cpp:2508) 要改成硬错） |
| ids(L+1) 缺失 / 0-token | 保留 B38 的 0-token 提前返回 |

### 4.8 验证

- 与 A 相同的门禁；另外用 `STREAM_MOE_TMP_DUMP` 对 gemma + deepseek 逐层比对
  `moe_out`/hidden（B 预期逐字节一致）。
- TMR：`head` 桶应降到 1 次/token（c1c2exp5 现在是 16 次），每层提交数应为 1。
- `verify_places` 12/12；`run_baseline` PASS；`smoke_cli` PASS。

## 5. 顺序、成本、预期收益

| 步骤 | 触及 | 风险 | 预期 |
| :--- | :--- | :--- | :--- |
| A. 设备侧 fold | 出口块 + fold plan + staging + arena 尺寸 | 中（multi-target 路径） | 混合池每层省 1 次 sync、1 次 D2H/H2D（olmoe ~0.2-0.4ms/层；deepseek 更大） |
| B. parity + head fusion | `layout_arena` + Tier-2 块 + 循环 | 中高（别名证明） | 每层少 1 次提交（RX590 ~0.2-0.3ms/层；CUDA replay 后更大） |

A 是混合池上 B 的前置；单设备放置时 B 可独立落地。两者都**不替代** `BENCHMARK.md` 测到的
逐节点 dispatch 成本：RX590 上即使零断点，decode 上限也就 20~40 t/s。目标（P100/CUDA）
必须先用同一个 probe 测，再决定 B 要走多远。

## 6. 待确认

1. 多物理设备的 foreign partial 路径现在就做，还是只做"同一物理设备"？
   （目标是单 P100 + RAM，第一里程碑不需要前者。）
2. parity 取 2 还是 N（N = 最大融合深度）？看一眼层就够；只有将来做跨层批处理才需要 N>2。
3. 是否需要 TEMP dump（`STREAM_MOE_TMP_FOLD_DUMP`）逐层对比设备 fold 与 host fold？
   建议要，仅 bring-up 用。
4. 将来 CUDA replay：round-shape cache 用 `(layer, stage, width, n_active)` 作键够不够？
   （decode 的 width ∈ [0..n_k]。）
