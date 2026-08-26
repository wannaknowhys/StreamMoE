# MoE �?mmap 可行性研�?(LLAMA_MOE_NO_MMAP_RESEARCH.md)

> 目标问题�?*不改（或极小改）llama.cpp 的前提下，能否让 MoE 专家权重完全不经�?mmap、全部走自研 expert pool，�?dense 维持 llama.cpp 默认行为�?*
> 结论�?*能。零 llama.cpp 源码改动**，只靠已有的公开扩展�?+ 自研新组件。本文给出完整依据与最小架构�?> 基于 vendored llama.cpp @ f280b2698�?> 修订：�? 回答"紧凑�?+ 把位置塞�?mul_mat_id"是否可行�?
---

## 1. llama.cpp 权重加载/内存管线逐层分析

### 1.1 张量 �?buffer type 的选择 (`create_tensor`)

`llama-model-loader.cpp` 为每个权重调�?`select_weight_buft(...)`，且**优先查询 `tensor_buft_overrides`**（llama-model-loader.cpp:1180-1203）：

```cpp
for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
    std::regex pattern(overrides->pattern);
    if (std::regex_search(tensor_name, pattern)) {
        if (overrides->buft == ggml_backend_cpu_buffer_type()) {
            buft = select_weight_buft(...);            // 覆盖�?CPU buft 时仍走常规选择
        } else {
            buft = overrides->buft;                    // �?自定�?buft 直接生效
        }
        break;
    }
}
```

- 覆盖�?*自定�?buft**（非 CPU buft）→ 直接采用，不再走常规列表。这�?MoE �?mmap 的第一个开关�?
### 1.2 mmap 零拷贝的成立条件

`llama-model.cpp` 创建后端 buffer�?666-1754）：

```cpp
bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;   // CPU: true
bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);        // 只有默认 buft 才成�?if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
    // �?零拷贝：�?mmap 页直接包�?buffer，张�?data 指向文件映射
    buf = ggml_backend_dev_buffer_from_host_ptr(dev, addr+first, last-first, max_size);
} else {
    // �?真实分配 buffer + 逐个张量物理读入
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}
```

**关键：零拷贝 mmap 只对"默认 buft"张量成立�?* 任何非默�?buft（含自定�?pool buft）自动落�?`else` 分支 = 真实分配�?
### 1.3 数据搬运的最终分�?(`load_all_data`, llama-model-loader.cpp:1556-1583)

```cpp
if (use_mmap) {
    data = mapping->addr() + weight->offs;               // 从文件映射读�?    if (buf_mmap && cur->data == nullptr) {
        ggml_backend_tensor_alloc(buf_mmap, cur, data); // 零拷贝：指针指向映射
    } else {
        ggml_backend_tensor_set(cur, data, 0, n_size);  // �?物理拷贝，走 buffer iface
    }
}
```

对自定义 buft 的张量：`alloc_ctx_tensors_from_buft` 已赋 `cur->data`（≠nullptr），`buf_mmap` 非空 �?必走 `else` �?调用 **`buf->iface.set_tensor`**（ggml-backend.cpp:342）�?
### 1.4 `set_tensor` 是我们可以完全接管的�?
`ggml_backend_tensor_set` �?`buf->iface.set_tensor(buf, tensor, data, offset, size)`�?自定�?buffer �?`set_tensor` 写成一�?**no-op（只登记张量→文件映射，不拷贝）**，则 150GB 专家权重在加载阶�?*零物理读�?*——这正是"不做物理载入"的天然落点，�?*不需要改任何 llama.cpp 文件**�?
### 1.5 计算就绪等待的公开钩子：`cb_eval`

`llama_context_params` 有公开字段 `cb_eval` / `cb_eval_user_data`，llama.cpp 内部将其接到 sched（llama-context.cpp:1354）：

```cpp
ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

sched �?`compute_splits` 中对每个节点分两段回调（ggml-backend.cpp）：

```cpp
bool need = sched->callback_eval(t, true,  ...);   // ask=true  : "该节点需要同步执行吗"
...
if (need && !sched->callback_eval(t, false, ...)) // ask=false : 节点执行前、后端已同步
    break;                                         // 返回 false 取消整个计算
```

- `ask=true`：我们只�?`GGML_OP_MUL_MAT_ID` 且权重在我们�?buft 的节点返�?true（隔离出同步点），其余节点返�?false 保持批量异步�?- `ask=false`�?*该节点执行前**（此�?`src[2]` ids 张量已由前面�?argsort/topk 节点产出、CPU 可读），正是"检查所选专家是否已就绪 �?未就绪则阻塞等待 DIO 流式装载 �?就绪后返�?true"的落点�?
### 1.6 计算本身：CPU backend 直接从我们的 host buffer �?
- 我们�?buft �?buffer �?`is_host=true` �?sched �?算子跟随权重所在后�?规则（ggml-backend.cpp:940-971）把 `mul_mat_id` 指派�?CPU backend�?- CPU backend 不需要任何改动：只要 `cur->data`�? 我们大区域内的偏移）在算子执行时数据有效，它就能正常反量�?GEMM�?- **不需要自定义 ggml_backend**（CPU-only 阶段）。自定义 backend 只在"同一次调用里 CPU/GPU 双池混合"（Backend.md §1）时才需要�?
---

## 2. 可行性结�?
| 需�?| 实现途径 | 是否�?llama.cpp |
|---|---|---|
| MoE 专家权重不经�?mmap 数据路径 | `tensor_buft_overrides` 指向自定�?buft；非默认 buft 自动走真实分配；`set_tensor` no-op 跳过物理读入 | **�?*（公开参数 + 新组件）|
| 专家权重按需装载 | 自研 scheduler + DIO + pool，经 `cb_eval` �?`mul_mat_id` 前阻塞等待就�?| **�?*（公开 `llama_context_params.cb_eval`）|
| dense 维持 llama.cpp 默认 | 不覆�?dense 张量�?buft �?默认 mmap 零拷贝不�?| **�?* |
| 计算正确�?| CPU backend 直接消费 host buffer；数值等价回归兜�?| **�?* |

**一句话**：llama.cpp �?非默�?buft �?真实分配 + `set_tensor` 路径 + `cb_eval` 执行前钩�?这三个既有机制，恰好拼出�?MoE �?mmap 的完整缝隙，全部通过公开扩展点注入，**不需要修�?llama.cpp 任何源文�?*�?
**诚实的边界说�?*：llama.cpp 仍会为分片文件创�?mmap 映射对象（dense 张量需�?+ 作为 set_tensor 的拷贝源），但专家权重区域的�?*永远不会�?fault**——专家数�?100% 走我们私有提交内存池 + DIO 读取。因�?MoE �?mmap"�?*数据路径**意义上成立（无页缓存驻留、无 lazy fault）�?
---

## 3. 最小架构（全部为新组件，零 llama.cpp 改动�?
```
┌──────────────────────────── StreamMoE ───────────────────────────�?�? engine/llama_engine (已有)                                       �?�?   ├─ llama_model_params.tensor_buft_overrides =                 �?�?   �?   pattern "blk\..*\.ffn_.*_exps\.weight"  �?moe_expert_buft�?�?   �?   pattern "blk\..*\.ffn_.*_shexp\.weight" �?moe_expert_buft�?�?   ├─ llama_context_params.cb_eval = on_mul_mat_id_ready()       �?�?   └─ 其余 dense 参数默认                                          �?�?                                                                   �?�? backend/moe_expert_buft (�?   �?一�?ggml_backend_buffer_type  �?�?   ├─ alloc_buffer(total) : VirtualAlloc MEM_RESERVE（~147GB 地址�?�?   �?  空间，零物理内存；按专家 slice 逐段 MEM_COMMIT�?           �?�?   ├─ iface.is_host = true  (CPU backend 可直接消�?              �?�?   ├─ iface.set_tensor = no-op + 登记 (张量名→shard/offset/size)  �?�?   └─ iface.memset_tensor/get_tensor = 直通或 no-op               �?�?                                                                   �?�? scheduler/expert_scheduler (新，复用�?moe_scheduler/async_dio/  �?�?                                  expert_stats)                   �?�?   ├─ cb_eval ask=false：解�?ids �?未就绪专家入�?�?阻塞等待     �?�?   ├─ DIO(Windows IOCP) �?(shard,offset,size) 读到大区域偏�?     �?�?   └─ EST1/LRU 驱逐：VirtualFree(MEM_DECOMMIT) 释放旧专家页       �?└────────────────────────────────────────────────────────────────────�?```

### 3.1 数据布局（与 Backend.md 槽模型的关键差异�?
llama.cpp 图要求每�?`ffn_gate_exps` �?*一个连�?3D 张量** `[n_embd, n_ff, n_expert]`，gate/up/down 各自独立连续。因�?pool �?*每层三个连续子区�?*�?
```text
big_region (VirtualAlloc RESERVE, 私有, 非文件映�?
 ├─ blk.0.ffn_gate_exps [1.14GB]  �?256 专家 gate slice 连续
 ├─ blk.0.ffn_up_exps   [1.14GB]
 ├─ blk.0.ffn_down_exps [1.14GB]
 ├─ blk.1.ffn_gate_exps ...
 └─ ...
```

"装载专家 e" = 并行 DIO 三片 (gate[e], up[e], down[e]) 到各自区域偏�?`e*slice_bytes`�?*Backend.md �?紧凑 gate+up+down 三合一 slot"布局�?llama.cpp 不兼�?*，需改为上述三区域布局（详�?DESIGN_REVIEW.md §1）�?
### 3.2 �?token 阻塞语义与重�?
- 每层�?token �?3 �?`mul_mat_id` 节点（gate/up/down），cb_eval 对同 ids 只做一次就绪检查（缓存上次结果）�?- 路由 ids(L+1) 依赖 L 层输出，**非投机前瞻无法跨层预�?*（与 Backend.md 一致）�?- 可重叠来源：�?�?`dsv4_hash_layer_count` 层路由由 token 决定 �?**免费完美预取**；② EST1 历史/上步路由做投机预取；�?层内 attention/shared-expert/HC �?MoE IO 天然并行（不同后�?同后端不同节点）�?
### 3.3 与现有模块的复用

| 旧模�?| 复用方式 |
|---|---|
| `moe_loader` read plan | 需新增"�?(tensor, expert) �?目标区域偏移"的布局（原紧凑布局废弃）|
| `async_dio_win` (IOCP) | 直接复用 |
| `expert_pool` / `expert_stats` (EST1) | 驱�?热度逻辑复用，存储改为大区域分片 |
| `expert_scheduler` | �?worker 升级�?Backend.md 决策快循�?+ IOCP 执行 |

---

## 4. 分阶段落�?
- **Phase A1（CPU-only，无 llama.cpp 改动�?*：moe_expert_buft + tensor_buft_overrides + cb_eval 就绪等待 + scheduler/DIO/EST1。产出：MoE 全私有池、dense 默认 mmap、memwatch 验证 FILEBK 不再增长�?- **Phase A2**：数值等价回归（stock �?vs 接入 pool 的图逐元�?diff，覆盖全命中/混合），Backend.md 上线前要求�?- **Phase B（GPU 混合池）**：此时才需�?Backend.md §1 的自定义 ggml_backend（graph_compute �?CPU/GPU 双池并发），Vulkan 先行、dispatch 线程 park+转发为基线，ReBAR/CUDA stream-wait 为后续候选�?
---

## 5. 风险与注意事�?
1. **cb_eval 同步开销**：设置回调后 sched 走逐节点路径，但只�?`need=true` 的节点（我们�?mul_mat_id）才同步；其余批量异步。影响有限，实测验证�?2. **VirtualAlloc RESERVE 147GB 地址空间**�?4 位进程地址空间 128TB，安全；�?slice COMMIT �?64KB 粒度约束�?3MB slice 无浪费�?3. **90GB 哨兵仍有�?*：私有池提交内存 = 池预算（�?70GB），超过即触发哨兵，作为机器保护�?4. **文件 mapping 对象仍创�?*：不 fault 专家页，�?mapping 本身占虚拟地址；若想彻底不映射分片文件，需另走（超出本方案，不作默认）�?5. **子模块锁�?*：vendored llama.cpp 需固定 commit（现 f280b2698），扩展点签名变化需升级评审�?
---

## 6. ffn_gate_exps / ffn_up_exps / ffn_down_exps 结构详解 �?"一专家一数据结构" 设计空间

### 6.1 这三个张量是什�?
MoE FFN（SwiGLU）�?token 计算�?
```text
对每个被路由选中的专�?e�?  gate = x @ W_gate_e        # [n_embd] -> [n_ff]
  up   = x @ W_up_e          # [n_embd] -> [n_ff]
  act  = silu(gate) * up     # SwiGLU（DeepSeek-V4 还带 per-layer clamp�?  y    = act @ W_down_e      # [n_ff] -> [n_embd]
加权求和所有选中专家�?y
```

llama.cpp **�?*为每专家单独建张量，而是每层每支路一�?**3D 大张�?*，把全部专家的矩阵沿�?3 维堆叠：

| 张量 | 形状 | 含义 | 本模型实�?|
|---|---|---|---|
| `blk.l.ffn_gate_exps.weight` | `[n_embd, n_ff, n_expert]` | 全部专家�?gate 矩阵 | **1,140,850,688 B �?1.14 GB** |
| `blk.l.ffn_up_exps.weight`   | `[n_embd, n_ff, n_expert]` | 全部专家�?up 矩阵 | 1.14 GB |
| `blk.l.ffn_down_exps.weight` | `[n_ff, n_embd, n_expert]` | 全部专家�?down 矩阵 | 1.14 GB |

专家 e 的数�?= 各自张量内的 2D 切片 `[:, :, e]`，物理位�?`e * expert_slice_bytes`。本模型
`expert_slice_bytes = 1.14GB / 256 �?4.45 MB`，三层三片合�?�?**13.4 MB/专家/�?*（与�?StreamMoE �?13,056 KB slot 一致）�?
### 6.2 为什�?llama.cpp 必须把它们打包成一�?3D 张量

因为算子 **`ggml_mul_mat_id`** 就是�?一张权重含全部专家 + 运行期按 ids 选专�?设计的：

```text
ggml_mul_mat_id(w_exps[n_embd, n_ff, n_expert], x, ids[n_expert_used, n_tokens])
  = 对每�?token，按 ids 指出的专家切片做 matmul，一次内核调用完�?```

- 图是**静态拓�?*：ids 是运行期数据，不能决定节点结构，所以节点数固定 = 3（gate/up/down），专家选择是内核内部的数据依赖 gather�?- 若拆�?256×3 个独�?2D 张量：要么图爆炸（~33k 权重节点），要么�?token 重建图，要么预建全量+掩码�?× 浪费）。全部违�?llama.cpp 设计�?- **内核要求**：执行时整张 3D 权重（全部专家切片）�?*同一连续地址�?*内可寻址（`e*nb[2]` 指针算术）�?
### 6.3 两个内存模型分叉

| | Fork 1：虚拟连�?+ 按需物理驻留（本研究报告方案�?| Fork 2：每专家独立物理张量 |
|---|---|---|
| llama.cpp 改动 | **�?* | 必须重写 `build_moe_ffn`（或自建子图执行器，= 研究里的 L2）|
| 3D 张量物理布局 | 一�?RESERVE 大虚拟区域，专家 slice 落位其中 | 256×3 独立 tensor + �?token 动态装�?|
| 是否"全量在内�? | **�?*：虚拟全量（免费）、物理按需、池预算封顶 | 否，但图/内核重写成本巨大 |
| "一个专家一个数据结�? | �?slot 描述符（�?6.4�?| �?物理级别 |
| 正确性背�?| 上游 MLA/DSA/HC 全保�?| 放弃上游图，全部自证 |
| �?token 动态拓�?| 无（graph 不变�?| 有（重建/掩码）|

### 6.4 澄清�?一个专家一个数据结�? �?Fork 1 下如何成�?
关键区分三个概念�?
```text
�?全量物理驻留（llama.cpp mmap 默认 + 工作集增�?�?INC-3 的病根）�?我们要消�?�?全量虚拟地址空间（VirtualAlloc MEM_RESERVE，只占地址空间、不�?RAM）✓ 免费、安�?�?按需物理驻留（按专家 slice MEM_COMMIT / MEM_DECOMMIT，池预算封顶）✓ 我们的目�?```

Fork 1 = �?+ ③�?*"一个专家一个数据结�?在存�?调度层成�?*�?
```cpp
struct expert_slot_t {          // 每专家一个，独立数据结构
    uint32_t layer, expert;
    void *   slice_gate;        // big_region 内偏移（内存由我们按需 COMMIT�?    void *   slice_up;
    void *   slice_down;
    // Backend.md 64 位原子字：state(3b)|refcount(29b)|generation(32b)
    uint64_t ctrl;              // EMPTY/IO_INFLIGHT/READY/EVICTING/FAILED
    uint64_t last_access;       // LRU/EST1
};
```

物理上这 3 �?slice 位于同一 RESERVE 区域的不同偏移（llama.cpp 图需要连续可寻址）；**逻辑�?* gate/up/down 是同一个专家数据结构的三个成员，装�?驱�?refcount 都以专家为单位原子操作。llama.cpp 的图对这个细节完全无感——它只看�?那个 3D 张量此刻是完整可读的"，�?此刻完整"由我们的 cb_eval 就绪等待保证（未就绪 �?阻塞直到 3 slice DIO 装载完成）�?
### 6.5 结论

- llama.cpp �?全量在内�?只体现在**图层的虚拟连续性要�?*，不�?物理全部驻留"。用 RESERVE 大区域满足虚拟连续性，用按专家 slice �?COMMIT/DECOMMIT 实现物理有界，两者解耦�?- "一个专家一个数据结�? 在存�?调度层达成（slot 描述符），不必、也不应该在图层拆散 3D 张量——拆�?= 重写 llama.cpp �?+ 放弃上游正确性，且每 token 动态拓扑在静态图模型下无解�?- **推荐 Fork 1**（本报告 §3 架构），�?llama.cpp 改动，物理内存恒 �?池预算�?

## 7. ��Ĺ���������"���ղ� + ��λ������ ggml_mul_mat_id" ������

> ������ķ����������� expert pool����ַ/����ԭ����ע��� ggml ��ΪȨ�� buffer��ÿ��ר�ҵ� gate/up/down ����һ������ slot �����������"��õ�λ��"ι�� `ggml_mul_mat_id`�����������ַ/����������

### 7.1 mul_mat_id �ں˵���ʵѰַ��ʽ���Ѻ�ʵ ggml-cpu.c:1454-1681��

```c
const int n_ids = ids->ne[0];   // ÿ token ѡ k ��ר��
const int n_as  = ne02;         // Ȩ������ dim-2 ��С
// �׶�һ���� ids ��ֵ�� ARRAY INDEX ���ַ���
for (iid1 in tokens)
  for (id in 0..n_ids-1) {
    int32_t i02 = ids[iid1, id];
    assert(i02 >= 0 && i02 < n_as);              // �� ids ������ [0, n_as) ��С����
    matrix_rows[i02][...] = {id, iid1};          // �� i02 ֱ����Ϊ���� n_as �������±�
  }
// �׶ζ�����ר�� id ����ȡȨ��
src0_cur = src0->data + cur_a * nb02;            // �� Ȩ����Ƭ = data + id * nb02������
```

**���ۣ�`mul_mat_id` ����������ָ��/λ�á�** ids �� int32 С������ͬʱ�䵱"�����±�"��"���Բ�������"��Ȩ����Ƭ��ַ = `data + id*nb02`��`nb02` ����������ʱ�̶��� dim-2 ������

### 7.2 ��ķ���Ҫ����������ͬʱ����

1. **���������� stride view**���� `nb02 = slot_size`��`data = pool_base + branch_offset`��`ne[2] = n_slots`������"ids = slot ����"ʱ `data + slot_idx*nb02` ǡ���䵽�� slot��
   �� ֻ�� `ggml_view_3d` ���Զ��� `nb02`��`ggml_new_tensor_3d` �Ĳ����������ġ���ģ�������� `load_arch_tensors`��deepseek4.cpp������ �� **����� llama.cpp**��ÿ�� 3 �а� exps ���� view�������� pool buffer����`tensor_buft_overrides` ֻ�� buffer ���ͣ�**�Ĳ�����������**��
2. **ids ������ slot ��������ר�� id**����������Ѱַָ������ slot������Ҫ"ר�� id �� ��ǰ slot ����"���롣
3. **�۸��ã�5400 slot < 11008 expert����̶� stride view ��ͻ**��view ��Ѱַ�Ǿ�̬�ģ�������/���õ� slot ��ַ���ٶ�Ӧԭר�� �� ֻ�ܿ� (2) ������ʱ���롣

### 7.3 ����Ŀӣ����� ids ����

`build_moe_ffn`��llama-graph.cpp����ͬһ�� `selected_experts` ����ι�����������ߣ�

```c
weights = ggml_get_rows(probs, selected_experts);   // ��Ҫ"ר�� id"������ probs[n_expert]��
ggml_mul_mat_id(up_exps, cur, selected_experts);    // ��Ҫ"slot ����"��Ѱַ view��
```

������ͬ �� �͵ظ�д���ƻ� `get_rows` ��·��Ȩ�ء������ֶΣ������ɾ�����
- **(a) ִ��˳�� hack**��get_rows �������������ڵ�һ�� mul_mat_id���� ids ֮���ٱ��� �� �� cb_eval ��һ�� mul_mat_id �ڵ�͵ظ�д ids������ִ��˳�򣬴������� get_rows ���ڱ�� backend ִ�и�����ʧЧ��
- **(b) ͼ�Ķ�**������"�������"�ڵ㣨`translated = ggml_get_rows(slot_table[256], selected_experts)`��slot_table ÿ��һ�š��ɵ��������£��� ����� `build_moe_ffn`���� slot_table ��ÿ token ͬ����

### 7.4 �� Fork 1 �ĶԱ�

| ά�� | ��Ľ��ղ۷��� | Fork 1����Ķ��� |
|---|---|---|
| llama.cpp �Ķ� | **����**��view ������ + ����ڵ��ִ���� hack��| **��** |
| �����ַ | �ش�С��~70GB��| RESERVE ģ��ȫ����~147GB��**���**��64 λ��ַ�ռ� 128TB��|
| ����פ�� | ���� COMMIT��Ԥ��ⶥ | ��ר�� slice COMMIT��Ԥ��ⶥ��һ����|
| ÿר�����ݽṹ | ���� [gate\|up\|down] һ�� | slot �������� 3 �� slice ָ�� |
| ids | �跭��Ϊ slot �������� get_rows ��ͻ��| ����ר�� id����Ȼ���� |
| ���� | �� | �� |
| ��ȷ�Է��� | ���� hack / ͼ�Ķ� | �� |

### 7.5 ��ʵ����

- **"��λ������ mul_mat_id" �����������ϲ�����**���ں�ֻ�� `[0, n_as)` ���� id + �̶�����������ָ�롣
- **���ղ� + ����Ҫ����������� llama.cpp**��stride view + ids ���룩���ҷ����빲�� ids ������ͻ����Ҫ hack ��ͼ�Ķ���������"����� llama.cpp"Ŀ����㣡�
- **Fork 1 ��"������� RESERVE + ��ר�� slice ����פ��"�ﵽ��ȫ��ͬ��Ŀ��**���� mmap�������н硢��������ÿר�Ҷ������ݽṹ��������ֻ�Ƕ�ռ ~77GB �����ַ�ռ䣨��ռ�κ������ڴ棩��**ǿ�ҽ��� Fork 1��**
- �����ֽ��������ۣ���Ϊ GPU �ϴ����Ȼ� Backend.md ��ģ��һ���ԣ�������һ��"�� llama.cpp��С�ģ�+ ����"�Ķ���·�ߣ��ҿ��Լ���ϸ����ķ���view ������ + ִ������ + ��ֵ�ȼۻع飩������������"��Ķ�"��