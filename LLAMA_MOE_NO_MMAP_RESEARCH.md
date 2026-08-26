# MoE �?mmap 可行性研�?(LLAMA_MOE_NO_MMAP_RESEARCH.md)

> 目标问题�?*不改（或极小改）llama.cpp 的前提下，能否让 MoE 专家权重完全不经�?mmap、全部走自研 expert pool，�?dense 维持 llama.cpp 默认行为�?*
> 基于 vendored llama.cpp @ f280b2698�?
> 修订记录�?
> - §7 v2�?026-08-26）：根据"自定�?backend 接管 MUL_MAT_ID"的设计修正结论——紧凑槽方案可行且零 llama.cpp 改动，代价是**必须自行实现 MUL_MAT_ID 计算路径**（mini-graph 委托�?ggml-cpu 或原生内核）�?

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
bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);        // 只有默认 buft 才成�?
if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
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
    data = mapping->addr() + weight->offs;               // 从文件映射读�?
    if (buf_mmap && cur->data == nullptr) {
        ggml_backend_tensor_alloc(buf_mmap, cur, data); // 零拷贝：指针指向映射
    } else {
        ggml_backend_tensor_set(cur, data, 0, n_size);  // �?物理拷贝，走 buffer iface
    }
}
```

对自定义 buft 的张量：`alloc_ctx_tensors_from_buft` 已赋 `cur->data`（≠nullptr），`buf_mmap` 非空 �?必走 `else` �?调用 **`buf->iface.set_tensor`**（ggml-backend.cpp:342）�?

### 1.4 `set_tensor` 是我们可以完全接管的�?

`ggml_backend_tensor_set` �?`buf->iface.set_tensor(buf, tensor, data, offset, size)`�?
自定�?buffer �?`set_tensor` 写成一�?**no-op（只登记张量→文件映射，不拷贝）**，则 150GB 专家权重在加载阶�?*零物理读�?*——这正是"不做物理载入"的天然落点，�?*不需要改任何 llama.cpp 文件**�?

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

- `ask=true`：对 `GGML_OP_MUL_MAT_ID` 且权重在我们�?buft 的节点返�?true（隔离出同步点），其余节点返�?false 保持批量异步�?
- `ask=false`�?*该节点执行前**（此�?`src[2]` ids 张量已由前面�?argsort/topk 节点产出、CPU 可读），�?检查所选专家是否已就绪 �?未就绪则阻塞等待 DIO 流式装载 �?就绪后返�?true"的落点�?

### 1.6 算子归属：自定义 backend 如何接管 MUL_MAT_ID（Claude 细节#2，已核实�?

`ggml_backend_sched_backend_from_buffer`（ggml-backend.cpp:885-905）：

```cpp
for (int i = 0; i < sched->n_backends; i++) {
    if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&   // 支持权重 buffer �?buft
        ggml_backend_supports_op(sched->backends[i], op)) {               // 且支持该算子
        return i;    // 按优先级返回第一个同时满足的 backend
    }
}
```

sched 的后端列表来�?`model.devices` + CPU（llama-context.cpp:330-357）：`devices` �?NULL（默认）时枚举全部注册设备，CPU 最后追加�?

**结论**：只要我们的自定�?backend 在模型加载前注册为设备（`ggml_backend_reg` 动态注册机制），它就会进入 `model.devices` �?sched 后端列表（优先级�?CPU 之前）。`tensor_buft_overrides` �?exps 权重挂到我们�?buft 后，sched 找到"第一个同�?`supports_buft(我们的buft)` + `supports_op(MUL_MAT_ID)`"的后�?= 我们 �?**`MUL_MAT_ID` 整体派给我们�?`graph_compute`，官�?`ggml-cpu.c` 内核不参�?*�?

---

## 2. 结论总览

| 需�?| 实现途径 | 是否�?llama.cpp |
|---|---|---|
| MoE 专家权重不经�?mmap 数据路径 | `tensor_buft_overrides` �?自定�?buft；非默认 buft 走真实分配；`set_tensor` no-op 跳过物理读入 | **�?* |
| 专家按需装载 + 紧凑�?| 自定�?backend 接管 MUL_MAT_ID 执行；`graph_compute` 内查 `expert_directory`、就绪等待、从槽内存计�?| **�?* |
| dense 维持 llama.cpp 默认 | 不覆�?dense 张量�?buft �?默认 mmap 零拷贝不�?| **�?* |
| 计算正确�?| 我们�?compute 路径必须数值等价于 build_moe_ffn �?数值等价回归兜�?| 责任在我�?|

**诚实的边界说�?*：llama.cpp 仍会为分片文件创�?mmap 映射对象（dense 张量需要），但专家权重区域的页**永远不会�?fault**——专家数�?100% 走私有槽内存 + DIO�?

---

## 3. 三条路线的最终对�?

| 维度 | 路线 A：Fork 1（RESERVE 区域 + 官方内核 + cb_eval）| 路线 B：紧凑槽 + 自定�?backend（Backend.md 原设计）| 路线 C：紧凑槽 + �?llama.cpp �?view |
|---|---|---|---|
| llama.cpp 改动 | **�?* | **�?*（注�?backend + buft）| **必须**（load_arch_tensors �?view）|
| MUL_MAT_ID 执行 | 官方 ggml-cpu 内核 | **我们实现**（mini-graph 委托或原生内核）| 官方内核（stride view 适配）|
| 专家物理布局 | 大区域三区域 slice（非紧凑）| 紧凑 [gate\|up\|down] �?| 紧凑�?|
| ids | 专家 id，无需翻译 | 专家 id，图内不翻译（graph_compute 内私下查表）| 需翻译�?slot 索引（有 get_rows 冲突）|
| 虚拟地址 | RESERVE 147GB（免费）| 池大小（预算）| 池大�?|
| 实现工作�?| 小（buft + cb_eval + scheduler）| **�?�?*（backend 骨架 + MUL_MAT_ID 计算实现）| 中（�?llama.cpp + 翻译）|
| 正确性风�?| 无（全官方内核）| 我们�?compute 需数值等�?| 翻译 hack 风险 |

---

## 4. 路线 B（紧凑槽 + 自定�?backend）的实现要点

### 4.1 backend �?buft 双组�?

- **backend（设备）**：注册一�?`ggml_backend`，`device` 提供默认 buft（compute buffer，供 cur/ids/dst 等节点输出使用）。实�?`supports_op`（对 MUL_MAT_ID + 所用量化类型返�?true）、`graph_compute`�?
- **weight buft**：独�?buft，供 `tensor_buft_overrides` 引用。`device` 指向我们�?backend 设备�?
- `graph_compute` 收到�?MUL_MAT_ID �?split 图时：对该节点做"槽装�?+ 计算"�?

### 4.2 alloc_buffer 轻量句柄（Claude 细节#1，已核实�?

`ggml-backend-impl.h` �?buffer/buft iface 契约�?

- `alloc_buffer(buft, size)` 不必真正申请 `size` 字节。返回一�?buffer，`size` 字段 = 请求的总大小（147GB �?虚拟�?），`get_base` 返回**非空** dummy 指针（`ggml_backend_buffer_get_base` 断言 base!=NULL），`set_tensor`/`memset_tensor`/`get_tensor` 全部 no-op�?
- `ggml_backend_tensor_alloc`（gallocr 调用）的断言：`addr >= get_base`、`addr + alloc_size <= base + size` �?�?dummy base + size=总大小即可全部满足。张�?`data` 变为悬空指针�?*只有我们�?backend 读它**（实际我们不�?`data`，走槽内存）�?

### 4.3 MUL_MAT_ID 的计算实现（真正的成本所在）

官方内核不参与，我们必须实现。推�?**mini-graph 委托**（复用官方内核的正确性）�?

```text
graph_compute 处理 MUL_MAT_ID 节点（src0=权重[我们的buft], src1=cur, src2=ids）：
  1. �?ids（专�?id，原样）
  2. 对每个选中专家 e�?
     - expert_directory[e] 查当前槽；未就绪 �?DIO 装载 + 等待就绪（Backend.md 控制面）
     - 把槽内该专家 gate/up/down 连续内存包成 2D 量化 ggml 张量（外部数据指针，type/shape 正确�?
     - cur 包成 2D 输入视图
  3. 构�?mini-graph：k �?mul_mat 节点（每专家 gate/up/down 依节点类型而定），
     �?ggml_backend_graph_compute(cpu_backend, mini_graph) 执行（复用官方反量化/GEMM�?
  4. 结果�?token/专家写入 dst
```

- 这就是原 repo"subgraph_executor 指针重绑�?概念的正式落地�?
- 备选：原生量化 GEMM 内核（工作量巨大，不推荐首期）�?
- **数值等价回归（Backend.md 上线前要求）必须覆盖**：全命中 / 混合分布 / 与官方图逐元�?diff�?

### 4.4 �?Backend.md 控制面的衔接

- `expert_directory` / `slot_meta`�?4 位原子字�? `refcount` / `generation` 原样复用�?
- "一个专家一个数据结�? = 紧凑槽（gate/up/down 物理连续一块），比路线 A 更贴�?Backend.md 槽模型�?
- 槽复�?驱逐在 `graph_compute` 内部做就绪等待，无图/ids 改动�?

---

## 5. 分阶段落地（推荐路线 B�?

- **Phase A（CPU-only 自定�?backend�?*�?
  1. 注册 backend + weight buft（轻量句柄，no-op set_tensor）�?
  2. `tensor_buft_overrides` 指向 weight buft�?
  3. `graph_compute` 实现 MUL_MAT_ID（mini-graph 委托 ggml-cpu，先单专家，�?k 专家批处理）�?
  4. scheduler（DIO + EST1�? 槽控制面（slot_meta/expert_directory/MPSC）�?
  5. 数值等价回�?vs 官方图�?
- **Phase B（GPU 混合池）**：Backend.md §1——graph_compute �?CPU/GPU 双池并发（Vulkan 先行，dispatch 线程 park+转发；ReBAR/CUDA stream-wait 后置）�?
- 主线合并后：`git merge main` �?`debug/memguard` �?memwatch 验证槽池提交内存有界�?

---

## 6. 风险与注意事�?

1. **MUL_MAT_ID 计算实现是我们的责任**：mini-graph 委托的每次图构�?执行有开销，需预分配复用（指针重绑定）�?
2. **跨后端输出缓�?*：MUL_MAT_ID �?dst �?split 里被后续 CPU 节点消费 �?sched 会做 split 输入拷贝（CPU 读我们的 host compute buffer）。我们的 compute buft 需 is_host=true 以减少拷贝�?
3. **90GB 哨兵仍有�?*：槽池提交内�?= 池预算，超限即触发�?
4. **子模块锁�?*：vendored llama.cpp 固定 f280b2698；升级需评审 buft/backend 接口签名�?
5. **数值等价是硬门�?*：mini-graph 的精度必须与官方 `build_moe_ffn` 一致（�?swiglu clamp、权重缩放、per-token 聚合）�?

### 4.5 scratch arena��Q1������"�� k ��ͼ������"����"һ�鸴���ڴ� + һ�װڽڵ����"��

**���ۣ�scratch arena ����˵��"�� 1 ����� k ÿ device ��һ����ͼ����"����һ���¡�**

- ���"ͼ������"��Ϊ k=1..max ��׼��**һ������ͼ����**���ṹ��ͬ�����ˣ�������ʱ����Ӧ��һ�ݡ�
- scratch arena��**ֻ��һ��**��������top-k �ڵ㣩Ԥ���� ggml context �ڴ� + **һ��**�ڽڵ��߼���ÿ�ε��ð�ʵ��ѡ�е� k ���ڵ�ڽ�����ڴ棬û�õĲ�λ���á��ڴ��ͷ��β��һ�飬������ malloc��

**Ϊʲô������ arena ������"ͼ����"**��mini-graph �ڵ��**������״�Ƕ�̬��**���� token ���飨ÿר��ֻ��·�ɵ����� token����ʵ���̹ٷ� MUL_MAT_ID ���壩ʱ��`cur_e = [d_in, n_tokens_e]` ÿ����ͬ �� Ԥ���õ�ͼ����**����û������**����״�Բ��ϣ������� k ��ôѡ��������ȷ������ÿ���ؽ������ؽ���**Ԥ���� arena** ������`ggml_init` �ù̶��ⲿ `mem_buffer`��ÿ�ε��� `ggml_free(ctx)` + ���� `ggml_init(ͬһ�� buffer)` ֻ���ò��ǡ����ͷ��ڴ棩������������ÿ��ÿ��ǰ��� malloc/���ֹ滮������

**���������**��ÿ�� device һ�飨CPU arena��Vulkan arena ��һ���������� worst-case top-k Ԥ����arena ֻ���������� backend �� `graph_compute` �������ڣ��� sched ��ȫ���ɼ���

### 4.6 split ���ƣ�Q2��graph_compute �ĵ������ȣ�

��ͼ �� sched �ȸ�ÿ���ڵ㶨 backend����Ȩ�صĸ�Ȩ���ߣ���Ԫ����Ȩ�ظ������ߣ��� **������ͬ backend �Ľڵ�����һ�� = split** �� `graph_compute` **�� split ������**��ÿ���õ�һ����ͼ������ split �������ڵ㣩����紦���¼�ͬ��/���뿽����

DeepSeek4 һ�㣨ʾ�⣬������ʵ��ͼ���ʵ����
- attn/dense ȫ����CPU���� **split A (CPU)**
- router/gating/weights��CPU���� ���� A
- `mul_mat_id(up_exps)`��`mul_mat_id(gate_exps)` ���ڣ����ǵ� buft���� **split B (����)**��һ�� graph_compute �õ� gate+up �����ڵ�
- swiglu/clamp����Ԫ�� �� CPU���� **split C (CPU)**
- `mul_mat_id(down_exps)`�����ǵ� buft���� **split D (����)**��һ�� graph_compute ������ down
- ��Ȩ��ͣ�CPU���� split C ��

����**ÿ��Ԥ�� 2 �ε�������**��gate+up һ�Ρ�down һ�Σ������� 1 ��Ҳ���� 3 �Σ�������ʵ�ʽڵ�˳���ʵ��**��Ҳ��ζ�� route B ��������Ҫ cb_eval** ���� �����ȴ����������ǵ� graph_compute �ڲ����� Fork 1 ���ɾ���

**unpin �� split �߽�**������ split ����������� CPU split ���� �� sched �ڱ߽����¼�ͬ����֤"���ǵ� split ������" �� **unpin ��������߽�ȷ�ϵ���**���� sched �ı㳵������������ dispatch ������һ�̡�

### 4.7 GPU ί��Ŀ�������ʱ����Q3/Q4��

- **Q3��GPU �����ί�и���ע��� Vulkan backend �� graph_compute������ cpu_backend**�������� GPU ������ʵ�����Դ�׷š�����ȴ�� CPU �㡢����һ����ʽ���ˣ���CPU ��ί�� cpu_backend��GPU ��ί�� vulkan backend��������ͬһ�����ǵ� graph_compute �ﲢ���������� Backend.md ��1"ͬһ�ε��� CPU/GPU ˫�ز���"���룩��
- **Q4�����ʱ��**��Vulkan �� graph_compute ������"�ύ������أ�����֤����"��unpin ���ܹҺ������ء��������� �򵥰棺`ggml_backend_synchronize(vulkan_backend)` ��������� unpin��ͬ���ȴ��������첽���棩���� ���⣺unpin �� sched �� **split �߽��¼�**����һ��������������� split ��ʼǰ�� event wait ��ȷ������ split ����ɣ����������� synchronize�������첽��GPU ����첽���濿���õ���