# 每设备 Arena 规划（carry 拆分、per-device compact、C1/C2 放置）

[English](PER_DEVICE_ARENA.md) | [简体中文](PER_DEVICE_ARENA.zh-CN.md)

> 状态：**设计，2026-09-14**。承接"整层所有权转正"（`LAYER_EXECUTOR_DESIGN.md`）。
> 现在的单 host arena 太大、且架空 `--dense-placement`；本文规划每设备 arena。
> 相关：`ROUTE_B_LAYER_OWNERSHIP.md`（目标形态 §3.4/§3.5/§3.8）、
> `M2_DEVICE_EXECUTOR.md`（设备执行器）、`DENSE_PLACEMENT.md`（C1/C2）、
> `GRAPH_PARTITION.md`（区域）、`LAYER_EXECUTOR_DESIGN.md`（§4.3 arena）。

## 1. 问题

整层所有权转正后，`layout_arena` 把**每个** compute 节点预分配进单个 host arena，
布局为 `[carry][compact][closure]`。ubatch 512 实测（host buft）：

| model | carry | compact | closure | need |
| :-- | --: | --: | --: | --: |
| olmoe | 62 MB | 218 MB | 142 MB | 0.4–0.5 GB |
| gemma | 160 MB | 2498 MB | 198 MB | 2.8 GB |
| deepseek | 2630 MB | 1300 MB | 248 MB | 4.1 GB |

三个具体缺陷：

1. **carry 逐层累加。** 每个跨层张量拿到唯一、永不复用的 offset。实测：olmoe 15 个
   `l_out-N`、gemma 29 个、deepseek 84 个（43 个 `l_last-N` + 41 个 hyper-connection
   `node_NNN`）全部共存 → ub512 时 deepseek 2.6 GB。
2. **compact 没打包。** 它是"最坏层所有节点字节之和"，没有层内 liveness 复用。
   gemma 的 2.5 GB 由 dense MLP + C2 的 lm_head logits 主导。
3. **C1/C2 放置被架空。** `--dense-placement C1:<dev>,C2:<dev>` 设了 buft，但
   `layout_arena` 把节点的 `buffer`/`data` 改写到 host arena，放置失效、logits 永远在 host。

## 2. carry 拆分：cross-1 vs cross-N

**分析（verify 期，动态）。** 对每个 captured 跨层张量，算
`distance = 最大消费层 - 生产层`（用 `collect_layer_nodes` 建的层归属）。

- `distance == 1` → **cross-1**：流水线张量（第 L 层输出被 L+1 消费）。任一时刻存活的
  数量有界 → 可复用。
- `distance > 1` → **cross-N**：跨多层存活 → 必须保留。

**实测（ub=1）：三模型全是 100% cross-1，cross-N = 0：**

| model | cross-1 | cross-N |
| :-- | :-- | :-- |
| olmoe | 15 节点 / 122880 B（max 8192） | 0 |
| gemma | 29 节点 / 326656 B（max 11264） | 0 |
| deepseek | 84 节点 / 5505024 B（max 65536） | 0 |

**布局。** `carry1` 按**层边界索引双缓冲**：第 L 层产出、被 L+1 消费的那组 cross-1 张量
打包进 buffer `L % 2`（两个 buffer，使 L+1 的 tail 处 `O_L` 与 `O_{L+1}` 共存——那里既读
残差又写下一层输出）。`carry1` 大小 = `2 × max over 边界 pack(边界集合)`。`carryN` 打包
一次并保留（当前三模型为空，保留通用性）。

ub512 投影：olmoe 62 MB → ~0.3 MB，gemma 160 MB → ~11 MB，deepseek 2630 MB → ~67 MB。

## 3. 每设备规划

`layout_arena` 改为每设备规划器。每设备一份 `region_plan_t`：

```
struct region_plan_t {
    size_t carry1_bytes;   // 2 × max 边界 pack
    size_t carryN_bytes;   // 保留
    size_t compact_bytes;  // max over layers of pack(该层 dense 节点)
    size_t closure_bytes;  // max over layers of pack(该层 MoE 闭包)
    std::unordered_map<const ggml_tensor*, size_t> off;   // 节点 -> offset
    ggml_backend_buffer_t buf;                            // 该设备 arena
};
```

**节点 → 设备：**

- **C1 dense**（attention / norm / dense MLP / gating）→ `C1:<dev>`
  （`--dense-placement C1`）；该设备同时拥有该层的 `carry`。
- **C2 输出头**（`result_norm` / `result_output`，即 logits）→ `C2:<dev>`
  （`--dense-placement C2`）。
- **MoE 闭包** → 该层专家池所在设备（`--moe-expert-pools <dev>:<MB>`）。
- 默认（无 placement）：全部 CPU/RAM。

**每设备大小**按该设备上有什么算：有 C1（→ carry + compact）、有专家池（→ closure）、
有 C2（→ C2 激活）。产出的 plan 同时指导 C1/C2 的 arena 和专家闭包布局。

**跨设备张量**（消费者在别的设备：`cur` 进远端池、`acc` 回 owner、跨设备 `carry`）在
plan 里标记，供执行器用 D2D / transfer-queue 搬运（`ROUTE_B_LAYER_OWNERSHIP.md` §3.5），
绝不 host staging。

## 4. 复用已有打包

把 `moe_chain_verify_graph` 里现有的 best-fit decreasing 区间打包（闭包结果布局，
`route_b_chain.cpp`）抽成 `pack(nodes, last_use) -> { offsets, size }`。`compact`、
`closure`、`carryN`、每个 `carry1` 边界集合都复用它。不写第二个分配器。

## 5. C1/C2 放置语义

`--dense-placement C1:<dev>,C2:<dev>`（已解析）是 C1/C2 放置的唯一事实来源。设备的
arena 用该设备的 buffer type；被 own 的 C1/C2 节点从每设备 arena 取 `buffer`/`data`，
而不是固定 host arena。**logits 跟随 C2。**

分两层：

- **规划/分配层**（本文 phase 1-2）：现在就能落；CPU-only 时每设备坍缩成一个，数值不变。
- **执行层**：真在放置设备上跑 C1/C2 需要设备执行器（`run_dense_subgraph` 现在固定调 CPU
  backend）。即 `M2_DEVICE_EXECUTOR`（phase 3）。

## 6. 实施顺序

1. **carry 双缓冲 + 抽 `pack`。** 无设备依赖；立竿见影省内存；验证 olmoe/gemma/deepseek
   数值 + `run_baseline`。
2. **每设备 plan 结构。** 节点 → (设备, 区域, offset)、每设备大小、跨设备标记。CPU-only
   坍缩成一个设备（数值不变）。
3. **设备执行器。** 在放置设备上执行 C1/C2/闭包（跨设备用 D2D / transfer-queue）。独立大工程。

## 7. 状态（2026-09-14）

**Phase 1 —— carry 拆分：已落地。** `layout_arena` 把 carry 区拆开：cross-1 按层奇偶
双缓冲（`2 × max 边界`），cross-N 保留。闭包的 best-fit 区间打包抽成
`pack_interval(nodes, start, end)` 并复用于 `carryN`；`layout_arena` 现在在闭包结果布局
**之后**运行，用上打包后的闭包大小。

ubatch 512 实测（生产构建）：deepseek carry 2630 MB → ~67 MB，olmoe 62 MB → ~0.25 MB，
gemma 160 MB → ~11 MB。闭包块也降（olmoe ub1 278528 → 131072 B）。验证：生产
`run_baseline` PASS；olmoe / gemma / deepseek 整层 OK。

**compact 打包：已实现、已确认是坏的、且在最新功能构建里保持默认开。** 生产
`StreamMoE_dump` 保持 opt-in（`STREAM_MOE_TMP_COMPACT_PACK`，默认关 = 字节求和），
回归保持绿；最新功能构建 `StreamMoE_latest` / `StreamMoE_dump_dbg` 默认**开**（按
`AGENTS.md` 15：已落地的功能绝不回滚），让 bug 保持可复现；在那里 `=0` 可关。

生产构建（`StreamMoE_dump`）gemma v2、129-token prefill-from、8 GB 池、ubatch 512，
对 `moe_129_8192_vk` 基线实测：

| compact 布局 | embd cos（token 0） | gate（>=0.99） |
| :-- | --: | :-- |
| 字节求和（默认） | 0.98631 | 121/129 PASS |
| 区间打包（`=1`） | 0.01568 | 0/129 FAIL |

即打包后的 compact 区会损坏输出。本轮排查已确定：

- **区间打包本身不是元凶。** `[cpack]` plan 自检报告：无任何时间区间重叠的节点共享
  字节；首个损坏节点（L0 head 的 `inp_scaled`）位于**独占** offset（`off=131072`），
  没有任何其他节点用它 —— 不是"重叠覆盖"。
- **liveness 模型不是（全部）原因。** `layout_arena` 把执行时间轴建模为
  `head → 闭包屏障 → tail`，并合法地把死节点的槽给更晚的节点复用（`embd` end=1 /
  `norm-0` start=2 共享 `off=0`）。该复用按模型合法，但打包运行在**独占槽**上也读到了
  陈旧值（`inp_scaled = SCALE(embd)` 既不是正确 `embd`，也不是覆盖者的值）。
- **已排除 CPU 融合。** 手搓 cgraph（`run_dense_subgraph`）从不填
  `visited_hash_set`/`use_counts`，`ggml_cpu_try_fuse_ops` 看到 `use_count=0`，不融合。
  `GGML_TENSOR_FLAG_COMPUTE` 也已被主图 build 置位，无节点被跳过。

**注意 —— debug 构建在此不可作为判据。** 早前"RelWithDebInfo sum == pack，0 diff"
的结论是假象：所有 `StreamMoE_dump_dbg` 运行实际上都开着
`STREAM_MOE_TMP_COMPACT_PACK`（全部 `[cpack]` 日志都是打包 offset，如
`inp_scaled off=131072`），其"sum"跑其实就是 pack 跑。生产构建才正确遵守该 env
（`sum → off=45056` 顺序、`pack → off=131072` 复用）。本问题一律用生产构建排查。

下一步：在**生产**构建上、同一 ubatch 做 1-node vs multi-node 对比（dump 手搓 cgraph
的真实执行顺序，与 `plan->head` 对照），判断损坏在打包还是多节点 CPU 执行。

**复现（bug 态）。** `build.bat llamalibs StreamMoE_latest` 后跑 129-token
prefill-from：不带 env 就走坏的打包路径（`embd cos 0.01568`），带
`STREAM_MOE_TMP_COMPACT_PACK=0` 走字节求和（`cos 0.98631`）。对 vk 基线验证：

```bat
build\StreamMoE_latest\llama-build\bin\llama-server.exe ^
  -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf ^
  --prefill-from baseline_regression\baseline\upstream_129\tokens_id.bin ^
  --export-dir baseline_regression\temp_latest\moe ^
  -c 2048 -t 16 --expert-backend --moe-ram-pool 8192 --fit off --no-warmup
node baseline_regression\tools\verify_prefill.js ^
  baseline_regression\baseline\moe_129_8192_vk\prefill_export_main.bin ^
  baseline_regression\temp_latest\moe\prefill_export_main.bin --cos-floor 0.99 --min-ratio 0.9
```

要看 `[cpack]` offset 计划 / `[stage]` 节点值用 `StreamMoE_dump_dbg`
（+ `STREAM_MOE_TMP_STAGE_DUMP=1 STREAM_MOE_TMP_COMPACT_DEBUG=1`）。

## 8. 待定问题

1. `carry1` 两个 buffer 够吗，还是某些层需要更多（边界集合在 head 和 tail 都读，外加同层
   第二个边界）？
2. `carry1` 边界集合：逐边界打包，还是用一个共享集合按全局 max 定大小（更简单、略大）？
3. C2 与 C1 不同设备时：末层输出 → C2 需要跨设备搬运；保持为 plan 标记的搬运，还是要求
   C2 == C1 的设备以零拷贝（设计 §4.10 C2 接管）？
4. 每设备一个 grow-only buffer，还是每区域一个 buffer？
5. C1 在设备上时，`run_dense_subgraph` 要调该设备的 backend；`moe_exec_mul_mat_id` 现在
   收 `cpu_backend` —— 每层设备 backend 怎么穿进来？
