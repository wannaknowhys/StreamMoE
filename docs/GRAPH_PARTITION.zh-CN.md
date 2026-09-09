# StreamMoE 整图分区 - 全图地图与闭包切面

[English](GRAPH_PARTITION.md) | [简体中文](GRAPH_PARTITION.zh-CN.md)

> 状态：**分析，2026-09-09**。把 olmoe / gemma-4-26B-A4B / deepseek-v4-flash 的
> 完整 llama.cpp 计算图映射成 route B 拥有的几个区域，给每个 leaf 标出 buffer
> 类型，并框定下一步执行器方向（同设备接缝、C1 闭包化）。
>
> 相关：`docs/ROUTE_B_GPU_PHASE.md`（闭包/私有化）、`docs/DENSE_PLACEMENT.md`
> §2/§7（C1/C2 分类、闭包分析）、`docs/STREAMMOE_GGUF_FORMAT.md` §3
>（C1/C2/C3/C4）、`docs/M2_DEVICE_EXECUTOR.md`（每设备执行器）、
> `docs/BUCKET_EXEC_TOKEN_SUBSET.md`。

> **铁律 - 数据搬运（2026-09-09）**：永远不要在 host 上解引用或 `memcpy` 一个
> `ggml_tensor::data`。张量的数据可能在任何后端（设备独显存，或 host 映射）。一律用
> 后端无关拷贝：`ggml_backend_tensor_get` / `ggml_backend_tensor_set`（及 `_2d` 变体）；
> mid-graph 用 `_async` + `ggml_backend_synchronize`，后端用
> `ggml_backend_sched_get_tensor_backend` 取（见 `llama-context.cpp:1806`
> `export_capture_experts` 的写法）。裸 `memcpy` 只对**我们自己拥有、且不是 ggml 张量**
> 的字节区（专家池槽、staging、scratch）以及后端自己的 `iface` 实现合法。来源见 §7.4。

## 1. 地图怎么来的

`moe_chain_verify_graph(gf)` 在 route-b 的 `graph_reserve` frag
（`patches/route-b/common/stmoe_routeb_lctx_reserve.frag`）里调用，因此能拿到
scheduler 切分之前的完整 `ggml_cgraph`。`STREAM_MOE_TEMP` 门控的 dump
（`STREAM_MOE_TMP_GRAPH_DUMP=1`，仅 dbg 编译）打印：

- 每个 compute 节点：`N idx op name ne bytes chain=<L|-1> buft=<name|-|>`
- 每个 leaf（权重/输入/KV）：`L idx op name ne bytes buft=<name>`

`chain=L` 标出第 L 层被私有化的专家闭包（见 `collect_chain`，
`src/backend/route_b_chain.cpp`）。`buft` 在张量已有 buffer 时通过
`ggml_backend_buffer_get_type` 解析（权重有；compute 节点要到 sched 切分后才有）。
复现：

```
STREAM_MOE_TMP_GRAPH_DUMP=1 build\StreamMoE_dump_dbg\llama-build\bin\llama-cli.exe ^
  -m <model> -p hi -n 1 -c 2048 -t 16 --expert-backend --fit off ^
  --moe-expert-pools RAM:<N> --dense-placement C1:RAM,C2:RAM --no-warmup < nul > dump.txt 2>&1
node temp/analyze_gdump.js dump.txt
```

## 2. 四个区域

整图分成四块。第 (2) 与第 (3) 之间的切面是**硬切**（`moe_chain_verify_graph`
用纯拓扑证明：闭包外的节点绝不消费闭包中间量，只有 `ffn_moe_out` 例外）。

| # | 区域 | 内容 | 执行者 |
|:-:|:-----|:-----|:-------|
| 1 | **Dense trunk = C1** | 逐层 attention（GQA 或 MLA+DSA）、norm、KV 写入、dense MLP（gemma/deepseek）、共享专家（deepseek `_shexp`）、hyper-connection（deepseek） | llama.cpp 原生（只做放置） |
| 2 | **Gating** | `ffn_moe_logits/probs/argsort/topk/weights`（gemma/deepseek 另有权重归一化） | llama.cpp 原生（dense 侧） |
| 3 | **专家闭包** | routed `MUL_MAT_ID` + swiglu/geglu + down + weighted + 收敛 ADD + `ffn_moe_out`；权重 = 池里的 `_exps.weight` | **route B**（私有化） |
| 4 | **Output head = C2** | final norm + lm_head | llama.cpp 原生（只做放置） |

区域 3 就是闭包：从 routed `MUL_MAT_ID` 锚点前向 BFS 到 `ffn_moe_out`
（`collect_chain`）。Gating（2）在锚点上游，留在 dense 侧。闭包包含其生产者输出
的 view/reshape 别名（它们不是 compute）。

## 3. 每模型结构

用上面的 dump 实测（reserve 图，`-c 2048`；闭包字节是该层 hidden 中间块）。

| | olmoe-1b-7b | gemma-4-26B-A4B | deepseek-v4-flash |
|:--|--:|--:|--:|
| 层数 | 16 | 30 | 43 |
| compute 节点 | 934 | 2644 | 8521 |
| leaf | 234 | 729 | 1606 |
| 闭包节点 | 320（20/层） | 660（22/层） | 774（18/层） |
| 闭包 hidden / 层 | 344 KB | 578 KB | 608 KB |
| `MUL_MAT_ID` / 层 | 3（gate/up/down） | 2（gate_up, down） | 3（gate/up/down） |
| C3 专家权重 | 48 / 3720 MB | 60 / 13688 MB | 129 / 140352 MB |
| C1 逐层 dense | 144 / 160.8 MB | 565 / 1711.2 MB | 1193 / 11993.5 MB |
| C2 embedding（`token_embd`） | 1 / 55.3 MB | 2 / 1496 MB | 1 / 1010 MB |
| C4 scale | - | 30 / 15 KB | - |

模型差异：

- **olmoe**：纯 MoE，无 dense MLP；C1 只有 attention + norm + router。
- **gemma**：每层 dense MLP（`ffn_gate/up/geglu/mlp`）+ MoE；C4
  `ffn_down_exps.scale`（512 B/层）复制到 `STREAMMOE_HOST`。
- **deepseek**：MLA + DSA lightning-indexer + hyper-connection（`DSV4_HC_*`）+
  共享专家（`ffn_*_shexp`，dense/C1）+ routed 专家（3 mm/层）。

## 4. 接缝（闭包 external leaves）

`STREAM_MOE_CAP_DUMP=1` 打印每层的外部叶子——闭包从 dense 侧取的东西，以及它交回
的唯一产物：

| role | 张量 | 生产者 | 消费者 |
|:-----|:-----|:-------|:-------|
| `w` | `blk.L.ffn_*_exps.weight` | 专家池（scheduler pin） | 闭包的 `MUL_MAT_ID` |
| `cur` | `ffn_norm-L`（norm 后 hidden） | C1 norm | 第一个 `MUL_MAT_ID` |
| `ids` | `ffn_moe_argsort-L` | gating | `MUL_MAT_ID` src[2] |
| `scale` | `ffn_moe_weights-L`（路由权重） | gating | `ffn_moe_weighted` |
| `scale` | `blk.L.ffn_down_exps.scale`（仅 gemma，C4） | C4 常驻 | down 的无权重 op |
| **out** | `ffn_moe_out-L` | 闭包 | dense 残差 ADD |

所以 dense<->闭包 的全部接口就是：**`cur` 进、`ids` 进、路由权重 `scale` 进、专家
权重进、`ffn_moe_out` 出**。别的都不跨。

## 5. buffer 类型标记

| buft 名 | 含义 | 出现位置 |
|:--------|:-----|:---------|
| `STREAMMOE_EXPERT` | 专家池 buft；所有 routed `_exps.weight` | C3，闭包的权重叶子 |
| `STREAMMOE_HOST` | C4 复制小叶子的 host-mapped buft | gemma `_exps.scale` |
| `STREAMMOE_DENSE` | v2-chunk dense 条带 buft（仅 `topo.incomplete`） | 这三个完整 GGUF 没出现 |
| `STREAMMOE`（backend） | 接收私有化闭包 split 的设备 | 区域 3 的 `graph_compute` |

`--dense-placement C1/C2` 与这些标记**正交**：它把 dense 权重放到设备 buft
（如 `Vulkan0`），名字不是 STREAMMOE。

## 6. C1 / C2 / embedding

`docs/DENSE_PLACEMENT.md` §2 定义 **C2 = 全局 dense，非 `blk.*`：`token_embd`、
`output`、`output_norm`、`rope_freqs`（deepseek 另有 `output_hc_*`）**。
所以**按分类 embedding 属于 C2**，但 C2 的两个成员位于图的两端：

- `token_embd` 是**输入**（上游：token -> embedding -> 各层）。llama.cpp 把输入层
  硬钉在 CPU（`llama-model.cpp:1489`，"little benefit to offloading the input
  layer"），所以无论 `C2:<dev>` 怎么设，`token_embd` 都在 CPU。
- `output` + `output_norm` 是**输出头**（下游）。它们跟随输出层设备
  （`route_b_dense_device(spec, il == n_layer_all)`），即 C2。

它们同属一类只是因为都"与层无关"；二者不相邻、没有直接的生产/消费关系。（有些
模型权重共享；这三个模型的 `token_embd` 与 `output` 是各自独立的张量。）

## 7. 设计含义（执行器下一步）

### 7.1 同设备接缝：不出设备

今天的每设备执行器（`M2_DEVICE_EXECUTOR.md` §7.9）把 `cur`/`ids` 从 host 搬到
device，层尾再把 `acc_d` 从 device 读回 host，即使 C1 与池在同一设备上。要加的
规则：**当生产者和消费者在同一设备时，接缝张量绝不离开该设备。**

- `cur` `[d, T]` 是最贵的 H2D。若 C1 的 norm 在池所在设备，闭包可直接绑定 norm
  的输出。
- `ffn_moe_out` `[d, T]` 是最贵的 D2H：把 `acc_d` 留在设备上，在设备侧做 fold +
  残差 ADD（M2-3 的推广）。
- `ids` 是控制数据：host 侧 round 规划器（`build_mix_plan`、`scatter_plan`）要读它
  才能给 token 分桶。让 `ids` 不出设备，需要把 round 规划搬到设备侧，或使用固定/
  已知路由。**注意**：`cur` 和 `ffn_moe_out` 是纯数据，今天就能常驻；`ids` 在规划器
  搬走之前不行。

回退：生产者/消费者设备不同时，按今天的方式 staging。设备身份要按**物理设备**判定
（llama `dev_layer` vs 池设备），不能按 `ggml_backend_t` 指针。

### 7.2 C1 闭包化（整层闭包）

C1 本身**不是**锚点驱动的闭包：它的节点与专家链交织（pre-attn norm 是 C1，链在
中间，尾部残差 ADD 是 C1 且消费 `ffn_moe_out`）。正确的单位是**整层**
（`docs/DENSE_PLACEMENT.md` §7）：取所有第 L 层节点，seed = 层输入，stop = 层输出。
这样 `cur`/`ids`/`ffn_moe_out` 变成图内边，接缝从结构上消失。

- 代价：route B 必须自己执行 attention + norm + KV + dense MLP（今天 llama.cpp
  做）。这是 Phase 2 的"managed C1"，不是小重构。
- 落地时的机械重构：把 `collect_chain` 泛化成
  `collect_closure(gf, seed_pred, stop_pred, ...)`；专家闭包 =
  `(is_routed_mm, is_output_name)`，整层闭包 = `(is_layer_node, layer_output)`。
  verify 变成"除层输出外，没有任何第 L 层中间量被外部消费"。
- 顺序：7.1 对**静态全驻留 C1**（单设备、零往返）就能拿到大部分收益，且不用接管
  C1；7.2 是**动态/拆分 C1** 与多设备的终态，那时层已经是放置单位。

### 7.3 多出口整层包

C1 每层是原子、**不跨设备**（`DENSE_PLACEMENT §3.2`）；专家**跨设备**（RAM + Vulkan0 +
…）。所以整层包是扇出/扇入形状：

```
C1 前段（设备A） --cur--> [ 专家闭包 pool0（设备A） ]--\
                 \--cur--> [ 专家闭包 pool1（设备B） ]--+--> C1 尾段（残差，设备A）
                 \--cur--> [ 专家闭包 pool2（RAM）  ]--/
```

- `cur` 是**多出口**：C1 前段只写一次；每个 per-device 专家子包消费一份副本（设备相同
  时就是同一块设备内 buffer）。
- `ffn_moe_out` 是**多入口**：每个子包 fold 自己的 `acc_d`，C1 尾段把各部分相加
  （M2-2 现有 per-pool `acc_d` fan-in 的推广）。
- N=1（C1 与全部专家同设备）退化为"同设备接缝、零往返"（§7.1）。
- 无论 N 是多少，`ids` 仍是控制数据（host 侧 round 规划），整层包不改变这一点，除非
  规划器设备侧化。

### 7.4 B39（已修）：对设备 leaf 的裸 memcpy

上面的铁律来自一次真实崩溃。gemma `C1:Vulkan0` + Vulkan 专家池在装载时 `0xC0000005`。
lldb 栈：`memcpy` <- `stream_moe_backend_replicate_leaf`（`moe_backend.cpp:478`）<-
`moe_chain_assign_backend`（C4 复制）<- `graph_reserve`。C4 leaf
（`blk.N.ffn_down_exps.scale`）在 C1 上 GPU 时被分到 Vulkan，而代码在 host 上做
`memcpy(dev, t->data, bytes)`。**不是显存问题**：128 MB 池 + `-ub 16` 仍崩。C1:RAM 时
leaf 在 host、没有 Vulkan 池时复制循环为空，两者都正常——所以只有 `C1:Vulkan0 + 池` 中招。
修复：`ggml_backend_tensor_get(t, dev, 0, bytes)`。两个原崩溃配置
（`place-c1-exp2`、`place-c1c2-exp2`）现在都能干净装载。

裸 memcpy 审计（2026-09-09，`src/`）：只有 `moe_backend.cpp:478` 不安全。其余都合法：
后端 `iface` 实现（`moe_backend.cpp:98-108`）、非张量内存
（`moe_backend.cpp:190/199/327`、`async_dio_win.cpp:191`）、名字切片
（`route_b_chain.cpp:536`）、我们自己的字节区——专家池槽与 staging
（`scheduler.cpp:322/724`、`staging_reader.cpp:114`），以及
`STREAM_MOE_TMP_DEVDBG` 诊断读设备 arena（`minigraph_exec.cpp:1256`）。staging 上传
`minigraph_exec.cpp:564`（`memcpy(stage_map + off, host_data, bytes)`）写的是 host
映射的 stage buffer；等接缝改成张量后应改用 `ggml_backend_tensor_set`。

## 8. 待定问题

1. 设备身份映射：如何跨 llama `dev_layer` 与 route-B 池设备证明"同一设备"，并在
   两个 backend 里绑定同一块 buffer。
2. `ids` 常驻：token-subset 规划器能否设备侧跑，还是只接受这个小 int 张量走一次
   host 往返。
3. 整层闭包 vs llama.cpp dense 执行：attention/KV 有多少能按"每设备图"直接复用，
   多少必须重写。
4. gemma `C1:Vulkan0 + 池` load 崩溃（BUG_TRACKER B39）——**2026-09-09 已修**，
   靠后端无关拷贝铁律（§7.4）；接缝现在可以在共享设备上验证。
