# 设备端 dense 整层闭包（C1 / MoE / C2 各在自己的设备上）

[English](DEVICE_DENSE_CLOSURE.md) | [简体中文](DEVICE_DENSE_CLOSURE.zh-CN.md)

> 状态：**设计 + 实施进行中，2026-09-15**。把整层拥有（`ROUTE_B_LAYER_OWNERSHIP.md`、
> `LAYER_EXECUTOR_DESIGN.md`）扩展到 **dense 权重在设备上** 的层
> （`--dense-placement C1:<dev>`），并让所有跨设备数据移动显式化。相关：
> `PER_DEVICE_ARENA.md`（每设备 arena）、`M2_DEVICE_EXECUTOR.md`（设备执行器）、
> `DENSE_PLACEMENT.md`（C1/C2）、`GRAPH_PARTITION.md`（铁律）。

## 1. 目标

整层拥有目前只在层内 dense 权重 **host 驻留** 时才捕获该层
（`moe_chain_assign_backend`）。用 `--dense-placement C1:Vulkan0` 时该层不被捕获，
dense 走 llama sched、MoE 闭包走 route B——两条执行路径。目标是让设备端 dense 的层也
被捕获，由执行器把 **dense head（C1）、MoE 闭包、dense tail（C2）各在自己的 placement
设备上跑**，所有跨设备移动显式、在图内完成（GPU 等价，中途无 host 往返）。

## 2. 放宽过滤后暴露的缺陷

放宽捕获过滤后暴露了一串问题（均已修或在修）：

1. **sched 把设备权重 dup 到我们的 host buft。** 因为 `moe_dev_supports_buft` 拒绝设备
   buft，`ggml_backend_sched_split_graph` 为每个设备权重插了 `%s#%s#%d` 副本
   （`ggml-backend.cpp:1405`）到 `STREAMMOE_HOST`；设备子图随后拿到 host 操作数
   （Vulkan subbuffer 非法）。
2. **KV cache 落错设备。** 整层拥有时 `llama_model::dev_layer` 恒返回 `STREAMMOE`，
   于是 llama 把 KV cache（和 fused op）放在 host，而 C1 权重在设备。
3. **桶引擎的叶子读取假设 host。** 若干处把 `m->data` 当 host 指针传给
   `bucket_upload_leaf`；源在设备时那是假的 `0x1000+off` 指针。
4. **scratch/staging 尺寸是手算的**，没覆盖"设备端 dense + 混合池"隐含的设备<->host 读。

## 3. 设计

### 3.1 buft 伪装（`moe_dev_supports_buft`）

接受我们会执行其上的设备 buft（已注册的 device-exec arena/stage buft）。这能阻止 sched
的"权重→host" dup。安全的前提是执行器把每个节点放到其操作数实际所在的设备上跑。

### 3.2 `dev_layer` 跟随 placement

`llama_model::dev_layer(il)` 在层的 placement 是真设备时返回该设备，只有 host dense 的层
才返回 `STREAMMOE`。于是 KV cache 与 fused op 跟随 `--dense-placement C1:<dev>`。

### 3.3 `ids` 强制到 RAM

路由 `ids` 一律读到 host（mix plan 在 host 构建：`build_mix_plan`）。这是强制 D2H；
每设备图用 host 构建的 index，不用设备上的 ids。

### 3.4 跨设备移动：方案 1（最前面整拷贝 + 本设备 gather）

一个 ggml 图只在 **一个** backend 上跑，所有操作数必须已在它上面；不存在跨设备
`get_rows` / `cpy` 节点。所以：

- **最前面整拷贝**（执行器级 `ggml_backend_tensor_copy`，D2D / D2H / H2D）每层把每个闭包
  输入搬到该设备一次，摊销到该设备的所有 round；
- 之后 **gather**（`get_rows`）是本设备图节点，读设备本地内存。

"全部走跨设备 gather"（方案 2）不可行且更慢：会退化成逐行/逐元素拷贝或 host 往返。

### 3.5 每设备的闭包输入/输出 buffer

每设备用 `ggml_backend_buft_alloc_buffer`（设备 buft）分配闭包的 **输入 buffer** 和它的
`moe_out` 部分量，并 **在每层开始时清零**（部分量不会累积旧数据）。中间计算节点仍用
verify 构建的 plan（`ex->out_off` / `layout_ok`）。

### 3.6 每设备 `moe_out` 部分量 + 匿名 add

闭包内部已经完成累加/折叠。每个设备把自己的桶折叠进 **自己的 `moe_out` 部分量**。
消费 `ffn_moe_out` 的 **匿名 add** 在闭包 **之外**
（`collect_chain` 在 `moe_out` 的消费者处停止，`route_b_chain.cpp:1746`），是 **tail**
节点，所以跑在 C1 的设备上。它拿到 **每个设备部分量各一个额外 `src`** 并把它们相加
（单设备 => 一个输入，等于不多做）。这取代了原来对每设备 `acc_d` 的 host 侧
`layer_fold`。

### 3.7 backend 无关的叶子读取

桶引擎通过唯一助手 `bucket_source_leaf` 读所有源叶子（设备目标走 staging，CPU 目标走
host scratch，两者都用 backend 无关的 `tensor_read_host`）。任何地方都不得在 host 上解引用
`src->data`（铁律，`GRAPH_PARTITION.md`）。

### 3.8 独立 grow-only buffer

scratch / staging 改为独立 grow-only buffer，取代手算尺寸的 `af32`/`a32`/`stage_used`
bump，消除越界这一类问题。增长（重新分配）永远发生在 **绑定当前 build 指针之前**。

### 3.9 严格的跨设备审计

`layout_arena` 审计每个已捕获节点的操作数：任何操作数所在设备 != 节点设备且没有搬运，即
硬错误。它打印问题节点与全图，然后生产版 `exit(1)`；`STREAM_MOE_TEMP` +
`STREAM_MOE_TMP_AUDIT_CONTINUE` 下打印并继续（开发版排查）。

### 3.10 保留原始节点（不克隆 dense）

dense head/tail 继续用 **原始** 图节点（`run_dense_subgraph` 的做法），输出绑到每设备
arena。不克隆：MoE 桶引擎克隆是为了形状收窄（分桶），不是 buft 原因；dense 路径没有形状
收窄。

## 4. 状态（2026-09-15）

- **已落地**：buft 伪装（3.1）、`dev_layer`（3.2）、通用 xfer（覆盖叶子/无 buffer、排除闭包
  节点）、`bucket_source_leaf`（3.7）、view `buffer` 传播、CPU 闭包路径的 host full-alloc、
  严格审计（3.9）。
- **进行中**：每设备闭包输入/输出 buffer（3.5）、最前面整拷贝（3.4）、每设备 `moe_out` +
  匿名 add（3.6）、grow-only buffer（3.8）。
- **未做**：设备端 dense 数值验证（`C1:Vulkan0` + Vulkan 池）。

## 5. 涉及文件

- `src/backend/route_b_chain.cpp` / `.h` - 捕获、dev_of、每设备 plan、通用 xfer、审计。
- `src/backend/minigraph_exec.cpp` - 桶引擎（`bucket_source_leaf`、每设备 round、出口合并）。
- `src/backend/moe_backend.cpp` - `moe_dev_supports_buft`。
- `third_party/llama.cpp/src/llama-model.cpp` - `dev_layer`（经
  `route-b-inject.patch`）。
