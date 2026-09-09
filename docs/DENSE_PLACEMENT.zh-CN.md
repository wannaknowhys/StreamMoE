# Dense 位置管理 - C1/C2 驻留与迁移设计

[English](DENSE_PLACEMENT.md) | [简体中文](DENSE_PLACEMENT.zh-CN.md)

> 状态：**Phase 1a 已落地**（2026-09-08）。route B 已接管专家放置；本文把 dense
> 放置也变成显式的、参数控制的事情。**静态放置（Phase 1）是主路径；C1 动态迁移
> （Phase 2）是容量受限时的兜底。**
>
> 相关：`docs/STREAMMOE_GGUF_FORMAT.md` §3（C1/C2/C3/C4）、`docs/ROUTE_B_GPU_PHASE.md`
> §6（逐层 dense/KV）、`docs/M2_DEVICE_EXECUTOR.md` §7.8（最终 fold 跟随 dense）、
> `docs/TODO.md` 阶段 7、`docs/EXPERT_MOVE_PIPELINE.md`（搬运机制）、
> `docs/VRAM_DMA_MOVE.md`（设备带宽实测）。

## 1. 目标

`--expert-backend` 目前强制 `no_op_offload=true` + `n_gpu_layers=0`
（`patches/route-b/common/stmoe_routeb_args.frag`）：专家池由 route B 放置，
dense 放置则完全交给用户显式 `-ngl`（连续、从尾部开始，`llama-model.cpp:1455`）。

本设计把 dense 放置变成与 `--moe-expert-pools` 同风格的一等参数控制项，四条规则：

1. **C2 静态、永不迁移。**
2. **C1 按层组织，允许在层粒度上拆分到不同设备；KV 跟随该层 C1。**
3. **只有当 lag 比较认为值得时才迁移该层 C1，否则保持原位（惰性迁移）。**
4. **`--expert-backend` 下 `-ngl` 直接报错退出**：route B 拥有 dense 放置权，
   dense 只通过 `--dense-placement` 表达。

## 2. 分类与实测体积

来自 `docs/STREAMMOE_GGUF_FORMAT.md` §3.1/§3.1 表（实测）：

| 类  | 含义                                                                                         | deepseek4（43 层 / 256 专家）     | gemma4（30 / 128）         |
| :-- | :------------------------------------------------------------------------------------------- | :-------------------------------- | :------------------------- |
| C1  | 按层 dense（`blk.N.*` 非专家：attn、norm、dense ffn、router、shexp、indexer/hc）             | 9920.6 MB 合计，209.0~245.6 MB/层 | 1711.2 MB，54.6~69.2 MB/层 |
| C2  | 全局 dense（非 `blk.*`：`token_embd`、`output`、`output_norm`、`rope_freqs`、`output_hc_*`） | 2020.3 MB                         | 748.0 MB                   |
| C3  | 每专家（`_exps.weight`，`ne[2]==n_expert`）                                                  | 137.1 GB（12.8 MB/专家）          | 13.4 GB                    |
| C4  | 闭包消费的非每专家小表（gemma `_exps.scale`）                                                | 空                                | 15 KB                      |

**关键事实**：deepseek C1+C2 = 11940.9 MB ≈ 11.66 GiB，**放得下 128 GiB RAM**。
所以 dense 流式对 RAM 而言**不是容量必需**，而是"要不要上 GPU 加速"的决策。
（`docs/TODO.md` 阶段 5 里"dense 162 GB"是旧说法——162 GB 是专家总量。）

## 3. 放置原则

### 3.1 C2 全局静态

C2（`token_embd`/`output`/`output_norm`/`rope_freqs`）每 token 必用，永不迁移。
`output` 是每 token 一次的大 matmul，是把 dense 钉到 GPU 上收益最高的张量。注意：
llama 强制输入层留 CPU（`llama-model.cpp:1470`，"offload 输入层收益极小"），所以
`token_embd` 按设计留 CPU；`output` 跟输出层设备。

### 3.2 C1 逐层原子，只在层粒度拆分

C1 以"一层一个单位"组织，允许不同层分到不同设备。**不要层内拆**：attention 放
一个设备、shexp 放另一个，会把无权重算子的跨设备拷贝问题重新引入（见 §7）。
层是原子放置单位。

### 3.3 KV 跟随 C1

KV cache 是该层 attention 产生并消费的层内状态，必须与该层同设备
（`llama-kv-cache.cpp:215`：`offload_kqv` 打开时 `dev = model.dev_layer(il)`）。
迁移 C1 层时 KV 随行，这份 KV 代价计入迁移成本（§4）。

### 3.4 C1 不做带驱逐的池

C1 每 token、每层必用，**层内零复用**、严格顺序访问。LRU/驱逐池会每层都 miss、
抖动。所以 C1 只有三种状态：

- **静态常驻**（RAM，放得下时 VRAM）——decode 默认；
- **prefill 顺序流式**（双缓冲，算 L 时预取 L+1）——仅 prefill，这是流不是缓存池；
- **动态迁移**（Phase 2 兜底）——只有当某层放不下、且 lag 比较认为划算时才动。

## 4. 判据：计算 lag vs 搬迁 lag

对一层 C1，设参数 `Np`、每参数字节 `b`、CPU/GPU 有效算力 `P_cpu`/`P_gpu`、
链路带宽 `BW`、决策窗口 token 数 `T`：

```
省下的计算  = T * 2 * Np * (1/P_cpu - 1/P_gpu)
搬迁代价    = (Np*b + KV_bytes) / BW
迁移  <=>  T > (b + KV_bytes/Np) / (2 * BW * (1/P_cpu - 1/P_gpu))
```

`b ≈ 1`、`P_cpu ≈ 0.5 TFLOPS`、`P_gpu ≈ 3 TFLOPS`、`BW = 21 GB/s` 时
`T* ≈ 14 token`；按 RX590 实测 ReBAR 写 8 GB/s（`docs/VRAM_DMA_MOVE.md`）
则 `T* ≈ 38 token`。这就是为什么 prefill（几百到几千 token）该搬、decode（T=1）不搬。

三个细化：

1. **KV 计入搬迁代价。** deepseek MLA 的 KV 项很小（~1.15 KB/token/层，43 层约
   49.5 KB/token）；GQA 模型 KV 大得多，可能主导。
2. **滞回 + 冷却。** 阈值附近判据会抖。复用 `EXPERT_MOVE_PIPELINE` 的机制
   （margin + 冷却 + 先拷贝后释放），不要另造一套。
3. **窗口 = batch/phase，不是 per-token。** "不值得搬迁就不搬迁"是惰性迁移：
   一旦放好，只有来回都划算才动。逐 token 决策会抖。

预取会改变判据：算 L 时预取 L+1，搬运被藏住，有效成本更低（跨层流水）。

## 5. 实现

### Phase 1 - 静态放置（主路径，小改动）

**`--expert-backend` 下 `-ngl` 直接报错退出。** 加载器的连续 `-ngl` 与 route B
自管的 dense 放置冲突，所以 `--expert-backend` 下若给了 `-ngl`/`--gpu-layers`，
route B 报警并退出。检测方式：`--expert-backend` 的 handler **不要**再把
`n_gpu_layers` 覆盖成 0，保持默认 `-1`，解析后统一校验——只要不是 `-1` 就说明
用户动过 `-ngl`。边界：显式 `-ngl auto` 与默认同值，视为没给。

**Phase 1a（第一里程碑）：整体选设备。**

```
--dense-placement <spec>
  spec  := item[,item...]
  item  := C1:<dev>            # 全部 C1 层（整体）
         | C2:<dev>            # 全部 C2（整体；别名 GLOBAL）
  dev   := RAM | CPU | Vulkan0 | CUDA0 | ...
```

- 示例：`--dense-placement C1:Vulkan0,C2:Vulkan0`（dense 全上 GPU）、
  `--dense-placement C2:Vulkan0,C1:RAM`（C2 上 GPU、C1 留 CPU）。
- 不给 `--dense-placement` 时默认全部 dense 留 CPU（即现状，纯 RAM 遍不分配
  GPU dense buffer）。

**已落地（2026-09-08）**：`--dense-placement C1:<dev>,C2:<dev>` 填
`llama_model_params.dense_placement`；`get_layer_buft_list` 调
`stream_moe::route_b_dense_device`（route-b 钩子在 `llama-model.cpp`）。
`--expert-backend` 下 `-ngl` 报错退出；设备名非法也报错退出。gemma-4-26B
（129-token prefill-from）实测：默认与 `C1:RAM,C2:RAM` 逐字节 IDENTICAL；
`C1:Vulkan0,C2:Vulkan0` 把 30 个 C1 + 1 个输出层放 Vulkan0，差异在已知后端噪声
量级（hidden cos ~0.986）；`C1:RAM,C2:Vulkan0` / `C1:Vulkan0,C2:RAM` 差异 ~0.9999。
DeepSeek 的 C1 放不进 RX590 8G（dense 11.66 GB > 8 GB）——目标是 P100 16G。

**Phase 1b（后续）：逐层表。**

```
  item  := ... | L<a>[-<b>]:<dev> | LAYER:<dev> | OUTPUT:<dev>
```

- P100 16G 全放 dense：`C1:Vulkan0,C2:Vulkan0`（Phase 1a 已覆盖）。
- 部分放置：`C2:Vulkan0,LAYER:RAM,L27-42:Vulkan0`（Phase 1b）。

实现（1a/1b 共用）：在 `get_layer_buft_list`（`llama-model.cpp:1457`）加 route-b
钩子，查设备表返回 `{dev, gpu_buft_list}`；输出层用 `OUTPUT:`/`C2:` 设备。dense
执行仍走 llama 原生，我们只决定"放哪"。这同时自动搬 KV（`offload_kqv`）。

- **不要用 `-ot` 做这件事**：`-ot` 只改权重 buft、不改 `dev_layer`，KV 和无权重
  算子会留 CPU，每个 attention 节点都跨设备。
- `op_offload` 交互：route B 强制 `no_op_offload=true` 是为了纯 RAM 遍不占 GPU。
  一旦有 dense 层在 GPU 上，无权重 dense 算子（rope/softmax…）必须能跟随，所以
  该 run 必须放开 `op_offload`。规则：`op_offload = (存在 dense 层在 GPU 设备上)`。
  接受随之而来的 Vulkan compute buffer（~1.3 GB）和 GPU 数值形态。

### Phase 2 - C1 动态迁移（兜底，大改动）

- 动态迁移无法用 llama 静态 `dev_layer` 表达；C1 权重必须由 route B 自己持有并
  搬运（受管 C1 权重池），执行也由我们的执行器接管该层。复用
  `EXPERT_MOVE_PIPELINE`（move worker、滞回、先拷贝后释放、每池本地驱逐）。
- 入口条件：仅当 C1 放不下目标设备（或 KV 增长吃光预算）时启用。放得下就优先
  Phase 1 全常驻。
- attention + KV 一起搬；层保持原子。

## 6. 容量估算（deepseek4）

| 配置                | 容量          | C1+C2 = 11.66 GB 放得下？ | 说明                                   |
| :------------------ | :------------ | :------------------------ | :------------------------------------- |
| 单 P100 16G         | 可用 ~15.5 GB | **能**                    | 剩 ~3.8 GB 给 KV + 小专家池            |
| 单 RX590 8G         | 可用 ~7 GB    | **不能**                  | C2（2.02 GB）+ ~20 层 C1，或全给专家池 |
| RX590 8G + P100 16G | ~24 GB        | **轻松能**                | dense 全放 P100，590 给专家池          |

一旦有 P100 16G，**dense 应整体静态常驻 P100**（HBM2 ~732 GB/s vs PCIe 21 GB/s），
prefill 和 decode 的 dense 瓶颈同时消失。动态迁移此时只在 dense 放不下时才有意义
（例如只有 590，或超长上下文把 P100 的余量吃光）。

P100 上的 KV 预算（deepseek MLA，~1.15 KB/token/层 × 43 ≈ 49.5 KB/token）：

| 上下文 |      KV | C1+C2+KV | 15.5 GB 余量 |
| -----: | ------: | -------: | -----------: |
|     4k | 0.20 GB | 11.86 GB |      ~3.6 GB |
|    32k | 1.62 GB | 13.28 GB |      ~2.2 GB |
|    64k | 3.24 GB | 14.90 GB |      ~0.6 GB |
|   128k | 6.48 GB | 18.14 GB |       放不下 |

Pascal 注意：P100 是 CC 6.0、无 tensor core、无显示输出；要验证 ggml-vulkan 在
它上面能跑 `MUL_MAT_ID` 和量化内核，并保留一块显示卡（RX 590）带桌面。

## 7. verify / 闭包分析

**决策：现在不改 `moe_chain_verify_graph` / `collect_chain`，也不做"两个前向闭包"。**

- `collect_chain` 是锚点驱动的前向 BFS：从 expert `MUL_MAT_ID` 到 `ffn_moe_out`
  （`src/backend/route_b_chain.cpp:552`）。C1 不是这种闭包：它是层内 dense 节点，
  与专家链交织（前段 attn/norm 是 C1，中间是专家链，末尾 residual add 又是 C1、
  消费 `moe_out`）。没有单一锚点、也没有单一方向。
- verify 的职责只是证明"被私有化的中间量没有外部消费者"。Phase 1 静态放置**不
  私有化 C1**（llama 执行它），所以 verify 不用动。
- 等 Phase 2 接管 C1 执行时，正确抽象是**整层闭包**（`layer[]` 已给出层归属，
  取层内全部节点），verify 变成"层内中间量无层外消费者（层输出除外）"——正好收敛
  到 `docs/ROUTE_B_GPU_PHASE.md` §3 的 whole-layer self-scheduling。
- 届时唯一值得做的机械重构：把 BFS 骨架抽成
  `collect_closure(gf, seed_pred, stop_pred, out_chain, out_layer)`；专家闭包 =
  `(is_routed_mm, is_output_name)`。**等 C1 接管时再做。**

## 8. 待定问题

1. **`-ngl` 与 `--dense-placement`**（已定）：`--expert-backend` 下 `-ngl` 报错
   退出；dense 只通过 `--dense-placement` 表达。见 §5。
2. **部分静态放置**（部分层 VRAM、部分 RAM）Phase 1b 表允许；确认每层边界的残差
   隐藏态跨设备拷贝可接受（就一个 `[d, T]` 张量）。
3. **P100 的 Vulkan 可行性**（Pascal 上的 `MUL_MAT_ID` + 量化内核）——在把"dense
   全放 P100"定为主路径前必须实测。
4. **KV slot 分页**（单 slot 进 VRAM、禁止跨 slot 拼 ubatch）是独立的 KV 特性，
   本文不覆盖。见 `docs/TODO.md` 阶段 5。
