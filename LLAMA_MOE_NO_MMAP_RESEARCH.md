# MoE 去 mmap 可行性研究 (LLAMA_MOE_NO_MMAP_RESEARCH.md)

> 目标问题：**不改（或极小改）llama.cpp 的前提下，能否让 MoE 专家权重完全不经过 mmap、全部走自研 expert pool，而 dense 维持 llama.cpp 默认行为？**
> 基于 vendored llama.cpp @ f280b2698。
> 修订记录：
> - §7 v2（2026-08-26）：根据"自定义 backend 接管 MUL_MAT_ID"的设计修正结论——紧凑槽方案可行且零 llama.cpp 改动，代价是**必须自行实现 MUL_MAT_ID 计算路径**（mini-graph 委托给 ggml-cpu 或原生内核）。
> - 编码修复（2026-08-26）：整文件重写为纯 UTF-8；此前 PowerShell 追加段落导致编码混合。

---

## 1. llama.cpp 权重加载/内存管线逐层分析

### 1.1 张量 → buffer type 的选择 (`create_tensor`)

`llama-model-loader.cpp` 为每个权重调用 `select_weight_buft(...)`，且**优先查询 `tensor_buft_overrides`**（llama-model-loader.cpp:1180-1203）：

```cpp
for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
    std::regex pattern(overrides->pattern);
    if (std::regex_search(tensor_name, pattern)) {
        if (overrides->buft == ggml_backend_cpu_buffer_type()) {
            buft = select_weight_buft(...);            // 覆盖为 CPU buft 时仍走常规选择
        } else {
            buft = overrides->buft;                    // 自定义 buft 直接生效
        }
        break;
    }
}
```

- 覆盖为**自定义 buft**（非 CPU buft）→ 直接采用，不再走常规列表。这是 MoE 去 mmap 的第一个开关。

### 1.2 mmap 零拷贝的成立条件

`llama-model.cpp` 创建后端 buffer（1666-1754）：

```cpp
bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;   // CPU: true
bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);        // 只有默认 buft 才成立
if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
    // 零拷贝：把 mmap 页直接包成 buffer，张量 data 指向文件映射
    buf = ggml_backend_dev_buffer_from_host_ptr(dev, addr+first, last-first, max_size);
} else {
    // 真实分配 buffer + 逐个张量物理读入
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}
```

**关键：零拷贝 mmap 只对"默认 buft"张量成立。** 任何非默认 buft（含自定义 pool buft）自动落入 `else` 分支 = 真实分配。

### 1.3 数据搬运的最终分叉 (`load_all_data`, llama-model-loader.cpp:1556-1583)

```cpp
if (use_mmap) {
    data = mapping->addr() + weight->offs;               // 从文件映射读源
    if (buf_mmap && cur->data == nullptr) {
        ggml_backend_tensor_alloc(buf_mmap, cur, data); // 零拷贝：指针指向映射
    } else {
        ggml_backend_tensor_set(cur, data, 0, n_size);  // 物理拷贝，走 buffer iface
    }
}
```

对自定义 buft 的张量：`alloc_ctx_tensors_from_buft` 已赋 `cur->data`（非 nullptr），`buf_mmap` 非空 → 必走 `else` → 调用 **`buf->iface.set_tensor`**（ggml-backend.cpp:342）。

### 1.4 `set_tensor` 是我们可以完全接管的点

`ggml_backend_tensor_set` → `buf->iface.set_tensor(buf, tensor, data, offset, size)`。
自定义 buffer 的 `set_tensor` 写成一个 **no-op（只登记张量到文件映射，不拷贝）**，则 150GB 专家权重在加载阶段**零物理读入**——这正是"不做物理载入"的天然落点，且**不需要改任何 llama.cpp 文件**。

### 1.5 就绪等待的备选钩子：`cb_eval`（route B 不需要，仅供参考）

`llama_context_params` 有公开字段 `cb_eval` / `cb_eval_user_data`，llama.cpp 内部将其接到 sched（llama-context.cpp:1354）：

```cpp
ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

sched 在 `compute_splits` 中对每个节点分两段回调（ggml-backend.cpp）：

```cpp
bool need = sched->callback_eval(t, true,  ...);   // ask=true  : "该节点需要同步执行吗"
...
if (need && !sched->callback_eval(t, false, ...)) // ask=false : 节点执行前、后端已同步
    break;                                         // 返回 false 取消整个计算
```

route B（自定义 backend 接管 MUL_MAT_ID）中**不需要 cb_eval**——就绪等待发生在我们的 `graph_compute` 内部。cb_eval 只在 route A（官方内核执行 MUL_MAT_ID）时需要。

### 1.6 算子归属：自定义 backend 如何接管 MUL_MAT_ID（已核实）

`ggml_backend_sched_backend_from_buffer`（ggml-backend.cpp:885-905）：

```cpp
for (int i = 0; i < sched->n_backends; i++) {
    if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&   // 支持权重 buffer 的 buft
        ggml_backend_supports_op(sched->backends[i], op)) {               // 且支持该算子
        return i;    // 按优先级返回第一个同时满足的 backend
    }
}
```

sched 的后端列表来自 `model.devices` + CPU（llama-context.cpp:330-357）：`devices` 为 NULL（默认）时枚举全部注册设备，CPU 最后追加。

**结论**：只要我们的自定义 backend 在模型加载前注册为设备（`ggml_backend_reg` 动态注册机制），它就会进入 `model.devices` → sched 后端列表（优先级在 CPU 之前）。`tensor_buft_overrides` 把 exps 权重挂到我们的 buft 后，sched 找到"第一个同时 `supports_buft(我们的buft)` + `supports_op(MUL_MAT_ID)`"的后端 = 我们 → **`MUL_MAT_ID` 整体派给我们的 `graph_compute`，官方 `ggml-cpu.c` 内核不参与**。

---

## 2. 结论总览

| 需求 | 实现途径 | 是否改 llama.cpp |
|---|---|---|
| MoE 专家权重不经过 mmap 数据路径 | `tensor_buft_overrides` → 自定义 buft；非默认 buft 走真实分配；`set_tensor` no-op 跳过物理读入 | **否** |
| 专家按需装载 + 紧凑槽 | 自定义 backend 接管 MUL_MAT_ID 执行；`graph_compute` 内查 `expert_directory`、就绪等待、从槽内存计算 | **否** |
| dense 维持 llama.cpp 默认 | 不覆盖 dense 张量的 buft → 默认 mmap 零拷贝不变 | **否** |
| 计算正确性 | 我们的 compute 路径必须数值等价于 build_moe_ffn → 数值等价回归兜底 | 责任在我们 |

**诚实的边界说明**：llama.cpp 仍会为分片文件创建 mmap 映射对象（dense 张量需要），但专家权重区域的页**永远不会被 fault**——专家数据 100% 走私有槽内存 + DIO。

---

## 3. 三条路线最终对比

| 维度 | 路线 A：Fork 1（RESERVE 区域 + 官方内核 + cb_eval）| 路线 B：紧凑槽 + 自定义 backend（Backend.md 原设计）| 路线 C：紧凑槽 + 改 llama.cpp 建 view |
|---|---|---|---|
| llama.cpp 改动 | **零** | **零**（注册 backend + buft）| **必须**（load_arch_tensors 建 view）|
| MUL_MAT_ID 执行 | 官方 ggml-cpu 内核 | **我们实现**（mini-graph 委托或原生内核）| 官方内核（stride view 适配）|
| 专家物理布局 | 大区域三区域 slice（非紧凑）| 紧凑 [gate\|up\|down] 槽 | 紧凑槽 |
| ids | 专家 id，无需翻译 | 专家 id，图内不翻译（graph_compute 内私下查表）| 需翻译成 slot 索引（有 get_rows 冲突）|
| 虚拟地址 | RESERVE 147GB（免费）| 池大小（预算）| 池大小 |
| 实现工作量 | 小（buft + cb_eval + scheduler）| **中-大**（backend 骨架 + MUL_MAT_ID 计算实现）| 中（改 llama.cpp + 翻译）|
| 正确性风险 | 无（全官方内核）| 我们的 compute 需数值等价 | 翻译 hack 风险 |

---

## 4. route B（紧凑槽 + 自定义 backend）的实现要点

### 4.1 backend 与 buft 双组件

- **backend（设备）**：注册一个 `ggml_backend`，`device` 提供默认 buft（compute buffer，供 cur/ids/dst 等节点输出使用）。实现 `supports_op`（对 MUL_MAT_ID + 所用量化类型返回 true）、`graph_compute`。
- **weight buft**：独立 buft，供 `tensor_buft_overrides` 引用。`device` 指向我们的 backend 设备。
- `graph_compute` 收到含 MUL_MAT_ID 的 split 子图时：对该节点做"槽装载 + 计算"。

### 4.2 alloc_buffer 轻量句柄（已核实）

`ggml-backend-impl.h` 的 buffer/buft iface 契约：

- `alloc_buffer(buft, size)` 不必真正申请 `size` 字节。返回一个 buffer，`size` 字段 = 请求的总大小（147GB 的"虚拟值"），`get_base` 返回**非空** dummy 指针（`ggml_backend_buffer_get_base` 断言 base!=NULL），`set_tensor`/`memset_tensor`/`get_tensor` 全部 no-op。
- `ggml_backend_tensor_alloc`（gallocr 调用）的断言：`addr >= get_base`、`addr + alloc_size <= base + size` → 用 dummy base + size=总大小即可全部满足。张量 `data` 变为悬空指针，**只有我们的 backend 会碰它**（实际我们不读 `data`，走槽内存）。

### 4.3 MUL_MAT_ID 的计算实现：mini-graph 委托（真正的成本所在）

官方内核不参与，我们必须实现。推荐 **mini-graph 委托**（复用官方内核的正确性）：

```text
graph_compute 处理 MUL_MAT_ID 节点（src0=权重[我们的buft], src1=cur, src2=ids）：
  1. 读 ids（专家 id，原样）
  2. 对每个选中专家 e：
     - expert_directory[e] 查当前槽；未就绪 → DIO 装载 + 等待就绪（Backend.md 控制面）
     - 把槽内该专家 gate/up/down 连续内存包成 2D 量化 ggml 张量（外部数据指针，type/shape 正确）
     - cur 按 token 分组，组出该专家的输入列
  3. 构造 mini-graph：k 个 mul_mat 节点（每专家），
     用 ggml_backend_graph_compute(cpu_backend / vulkan_backend, mini_graph) 执行
  4. 结果按 token/专家写入 dst
```

- 这就是原 repo"subgraph_executor 指针重绑定"概念的正式落地。
- 备选：原生量化 GEMM 内核（工作量巨大，不推荐首期）。
- **数值等价回归（Backend.md 上线前要求）必须覆盖**：全命中 / 混合分布 / 与官方图逐元素 diff。

### 4.4 与 Backend.md 控制面的衔接

- `expert_directory` / `slot_meta`（64 位原子字）/ `refcount` / `generation` 原样复用。
- "一个专家一个数据结构" = 紧凑槽（gate/up/down 物理连续一块），比 route A 更贴合 Backend.md 槽模型。
- 槽复用/驱逐在 `graph_compute` 内部做就绪等待，无图/ids 改动。

### 4.5 mini-graph 的复用：预分配 scratch arena

mini-graph 节点的**张量形状是动态的**：按 token 分组（每专家只算路由到它的 token，忠实复刻官方 MUL_MAT_ID 语义）时，`cur_e = [d_in, n_tokens_e]` 每一步都不同 → **每次调用都必须重建 mini-graph**，不能"构一次复用到底"。

为消除每层每次前向的 malloc/布局规划开销，使用**预分配、可复用的 scratch context/arena**：

- 每个 device 一块（CPU arena、Vulkan arena 各一），容量按**最坏情况 top-k 个节点**预留。
- `ggml_init` 用固定的外部 `mem_buffer`；每次调用 `ggml_free(ctx)` + 重新 `ggml_init(同一块 buffer)`，只重置内部簿记、**不释放/不重新 malloc 内存**。
- 每次调用实际选中 k 个专家（k 从 1 到 top-k），就把 k 个节点摆进这块内存的前 k 个槽位，其余槽位闲置；摆节点的代码逻辑只有一套，按参数 k 复用。
- arena 只存在于我们 `graph_compute` 函数体内部，对 ggml 调度器完全不可见。

### 4.6 split 机制：graph_compute 的调用粒度

主图 → sched 先给每个节点定 backend（带权重的跟权重走；逐元素无权重跟输入走）→ **把连续同 backend 的节点打包成一段 = split** → `graph_compute` **按 split 被调用**（每次拿到一个子图，含该 split 的连续节点）。跨界处插事件同步/输入拷贝。

DeepSeek4 一层（示意，最终以实际图序核实）：

```text
attn/MLA/DSA + router/weights     → CPU    ┐ 合并成 split A（CPU）
mul_mat_id(up_exps)                → 我们   │
mul_mat_id(gate_exps)              → 我们   ├─ split B（我们）→ 一次 graph_compute 拿到 gate+up
swiglu/clamp + 加权求和            → CPU    ┐ split C（CPU）
mul_mat_id(down_exps)              → 我们   ├─ split D（我们）→ 一次 graph_compute 单独拿 down
```

即：**每层预期 2 次调用我们**（gate+up 一次、down 一次），不是 1 次也不是 3 次；具体切法照实际节点序核实。

**unpin 挂 split 边界**：我们 split 的输出被后续 CPU split 消费 → sched 在边界做事件同步保证"我们的 split 真算完" → **unpin 挂在这个边界确认点上**（搭 sched 的便车），不是我们 dispatch 返回那一刻。

### 4.7 GPU 委托目标与完成时机

- **GPU 组必须委托给已注册的 Vulkan backend 的 graph_compute，不是 cpu_backend**——否则 GPU 池名存实亡（显存白放、矩阵却在 CPU 算、还多一次隐式搬运）。CPU 组委托 cpu_backend，GPU 组委托 vulkan backend，两组在同一个我们的 graph_compute 里并发处理（与 Backend.md §1"同一次调用 CPU/GPU 双池并发"对齐）。
- **完成时机**：Vulkan 的 graph_compute 语义是"提交命令、返回，不保证跑完"。unpin 不能挂函数返回。方案：① 简单版：`ggml_backend_synchronize(vulkan_backend)` 等真完成再 unpin（同步等待，牺牲异步收益）；② 正解：unpin 挂 sched 的 **split 边界事件**（下一个消费我们输出的 split 开始前的 event wait 即确认我们 split 已完成）——不额外 synchronize、真正异步。GPU 组的异步收益靠②拿到。

---

## 5. 分阶段落地（推荐 route B）

- **Phase A（CPU-only 自定义 backend）**：
  1. 注册 backend + weight buft（轻量句柄，no-op set_tensor）。
  2. `tensor_buft_overrides` 指向 weight buft。
  3. `graph_compute` 实现 MUL_MAT_ID（scratch arena + mini-graph 委托 ggml-cpu，先单专家，再 k 专家批处理）。
  4. scheduler（DIO + EST1）+ 槽控制面（slot_meta/expert_directory/MPSC）。
  5. 数值等价回归 vs 官方图。
- **Phase B（GPU 混合池）**：Backend.md §1——graph_compute 内 CPU/GPU 双池并发（Vulkan 先行，dispatch 线程 park+转发；ReBAR/CUDA stream-wait 后置）。GPU 组委托 vulkan backend，unpin 挂 split 边界事件。
- 主线合并后：`git merge main` 进 `debug/memguard` 用 memwatch 验证槽池提交内存有界。

---

## 6. 风险与注意事项

1. **MUL_MAT_ID 计算实现是我们的责任**：mini-graph 委托的每次构造/执行有开销，用 scratch arena 消除 malloc；节点构造逻辑需预分配复用（指针重绑定）。
2. **跨后端输出缓冲**：MUL_MAT_ID 的 dst 在 split 里被后续 CPU 节点消费 → sched 会做 split 输入拷贝（CPU 读我们的 host compute buffer）。我们的 compute buft 需 is_host=true 以减少拷贝。
3. **90GB 哨兵仍有效**：槽池提交内存 = 池预算，超限即触发。
4. **子模块锁定**：vendored llama.cpp 固定 f280b2698；升级需评审 buft/backend 接口签名。
5. **数值等价是硬门槛**：mini-graph 的精度必须与官方 `build_moe_ffn` 一致（含 swiglu clamp、权重缩放、per-token 聚合）。
6. **文档编辑纪律**：追加文档段落一律用 write/edit 工具（UTF-8）；严禁用 PowerShell `Set-Content` 追加中文（默认 ANSI/CP936 编码会破坏 UTF-8 文件）。
