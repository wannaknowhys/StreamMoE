# llama.cpp 架构深度调研与 StreamMoE 改造/复用评估报告

> **版本**：v1.0.0  
> **关联工程**：StreamMoE (OffloadMoE)  
> **定位**：指导 StreamMoE 对 `third_party/llama.cpp` 的依赖裁剪、接口对接、核心重写及后续上游同步 (Upstream Tracking)。

---

## 1. `llama.cpp` 仓库架构全景拆解

`llama.cpp` 整体架构清晰地分为三个层次：**底层计算与硬件抽象层 (`ggml/`)**、**LLM 拓扑与执行运行时 (`src/`)**、**上层应用与投机解码工具 (`common/`)**。

```
llama.cpp /
├── ggml/                       # [底层算子与硬件后端]
│   ├── include/                # ggml.h, gguf.h, ggml-backend.h, ggml-alloc.h
│   └── src/                    # ggml.c, ggml-quants.c, ggml-backend.cpp, ggml-vulkan/, ggml-cpu/
├── src/                        # [模型拓扑、KV 缓存与计算图]
│   ├── llama-model.cpp / .h    # 模型骨架与权重加载入口
│   ├── llama-model-loader.cpp  # GGUF 权重到后端 Buffer 的映射与分配
│   ├── llama-graph.cpp / .h    # 全局计算图构建器 (含 Attention / RoPE / build_moe_ffn)
│   ├── llama-kv-cache.cpp / .h # KV Cache 内存与生命周期管理
│   ├── llama-mmap.cpp / .h     # OS mmap 内存映射实现 (需被 StreamMoE 替换)
│   └── models/                 # 150+ 种具体模型架构定义 (qwen2moe, deepseek2, llama 等)
└── common/                     # [上层调度与投机解码]
    ├── speculative.cpp / .h    # 投机解码 (Speculative Decoding) 框架
    ├── sampling.cpp / .h       # Logits 采样器 (Top-K, Top-P, Temperature)
    └── arg.cpp / .h            # 命令行参数解析
```

---

## 2. 核心问题解答：`models/*.cpp` (150+ 架构) 如何处理？

### 2.1 源码机制剖析
在 `src/models/*.cpp` 中，每个模型架构主要实现三个核心虚函数：
1. `load_arch_hparams()`：解析该模型专有的 GGUF 超参数（如 `n_expert`, `n_expert_used`）。
2. `load_arch_tensors()`：声明该模型包含的 Tensor（如 `ffn_gate_exps`, `attn_qkv` 等）。
3. `build_arch_graph()`：构建该模型的前向计算图。

### 2.2 结论与应对策略：**无需修改 150+ 文件，采用“中心网关拦截模式”**

| 模型分类 | 代表架构 | 处理策略 | 为什么无需侵入单个模型文件？ |
| :--- | :--- | :--- | :--- |
| **Dense 模型**<br>(全量平权计算) | `llama.cpp`, `qwen2.cpp`, `gemma.cpp`, `dspark.cpp` 等 | **100% 直接复用** | 仅包含 Attention / Norm / MLP，完全无 MoE 逻辑。直接作为 Dense 主干或 Dense 草稿模型运行。 |
| **MoE 模型**<br>(稀疏专家选通) | `qwen2moe.cpp`, `deepseek2.cpp`, `deepseek32.cpp`, `bailingmoe.cpp`, `mixtral` 等 | **通过核心网关拦截，保持文件原貌** | 所有的 MoE 模型在 `build_arch_graph` 中最终都统一调用 **`llm_graph_context::build_moe_ffn()`** 统一网关！ |

#### 中心网关拦截设计 (Central Gateway Interception)：
* **切入点 1 (`llama-model-loader`)**：识别到模型属于 MoE 架构时，跳过对 MoE 专家大 Tensor 的全量物理显存/内存分配，仅记录其 GGUF 偏移与元数据，交给 StreamMoE 的 `expert_pool_t` 管理。
* **切入点 2 (`build_moe_ffn`)**：所有 MoE 模型构建 MoE 分支时都汇聚在 `llama-graph.cpp` 的 `build_moe_ffn()`。我们只需在此处拦截：
  * Dense 骨干部分正常前向并计算出 Gating Router Logits / Top-K 选通列表；
  * 将选通列表直接交由 StreamMoE 的双线程调度器（`moe_scheduler`）与静态子图执行器（`subgraph_executor`）接管；
  * 计算完成后将专家合并结果返回给主图继续后续 LayerNorm / Output Head。
* **收益**：**完全不修改 `models/*.cpp` 中的任何一个单独模型文件**，后续 `llama.cpp` 官方新增 10 个新 MoE 架构，StreamMoE 可以零成本直接兼容！

---

## 3. 模块处置全景矩阵 (Reuse / Adapt / Rewrite)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       StreamMoE 模块处置全景矩阵                                │
├──────────────────────┬──────────────────────────┬───────────────────────────────┤
│    A. 完全直接复用   │   B. 借鉴并裁剪/修改     │      C. 彻底重写 / 纯自研     │
│   (Directly Reuse)   │   (Adapt & Refactor)     │     (Completely Rewrite)      │
├──────────────────────┼──────────────────────────┼───────────────────────────────┤
│ 1. ggml 计算与量化   │ 1. Dense 骨干前向图      │ 1. Direct I/O 异步读盘引擎    │
│    (GEMM / Q4_K 反量化)│    (剥离 MoE 分支)       │    (Windows IOCP/io_uring)    │
│ 2. GGUF 元数据解析器 │ 2. KV-Cache 内存管理     │ 2. Pinned RAM/VRAM 内存池     │
│    (gguf.cpp / Shape)│    (默认保留在 Host RAM) │    (4KB 对齐紧凑 Slot 数组)   │
│ 3. ggml-backend 通用 │ 3. 投机推理框架          │ 3. 双线程协同调度器           │
│    硬件后端 (Vulkan) │    (暴露路由并广播预取)  │    (Compute + Scheduler 流水线)│
│ 4. Tokenizer / Vocab │ 4. Dense 草稿模型运行体  │ 4. Sub-Graph 指针重绑定执行器 │
│ 5. Sampler 采样器    │    (纯 Dense 轻量 Context│    (预建候选子图+指针注入)    │
│ 6. 150+ models/*.cpp │                          │ 5. 5维资源自适应决策状态机    │
│    模型架构定义文件  │                          │    (防抖动与紧急复位)         │
└──────────────────────┴──────────────────────────┴───────────────────────────────┘
```

---

## 4. 详细模块级处置说明

### 4.1 直接复用模块 (Group A)
1. **`ggml/src/ggml.c` & `ggml-quants.c` & `ggml-cpu`**：
   * **价值**：包含了经过高度手写 SIMD 优化的 AVX2/AVX-512/Neon 矩阵乘法、激活函数与数十种量化格式（Q4_K_M, Q8_0, IQ4_XS 等）的高性能解码内核。
   * **策略**：作为静态基础库直接编译链接。
2. **`ggml/src/ggml-vulkan/`**：
   * **价值**：成熟跨平台的 Vulkan 计算着色器与管线，完美适配 AMD RX590、Intel 及 Nvidia 显卡。
   * **策略**：作为 GPU 计算后端直接复用。
3. **`ggml/src/gguf.cpp` (`gguf.h`)**：
   * **价值**：标准化 GGUF 解析，直接提供 `gguf_find_tensor`、`gguf_get_tensor_offset` 等。
   * **策略**：直接调用，用于在引擎初始化时提取每个 Expert Sub-Tensor 的物理文件偏移。
4. **`src/llama-vocab.cpp` & `common/sampling.cpp`**：
   * **价值**：处理所有主流分词算法（BPE, WordPiece, SentencePiece）与采样策略。
   * **策略**：直接复用。
5. **`src/models/*.cpp` (全部模型定义)**：
   * **策略**：保持原貌，不做侵入式修改。

### 4.2 借鉴并裁剪修改模块 (Group B)
1. **Dense 骨干网络执行 (`llama-graph.cpp`)**：
   * **改造点**：保留 Embedding、Attention（RoPE、KV-Cache 写入、Softmax、Wo 投影）、LayerNorm、Output Head 等，在 `build_moe_ffn` 处接入 StreamMoE 调度入口。
2. **KV 缓存管理器 (`llama-kv-cache.cpp`)**：
   * **改造点**：默认将 KV 缓存锁定在系统 RAM（`--no-kv-offload`），确保宝贵的 VRAM 全部留给 MoE VRAM Pool。
3. **投机推理引擎 (`common/speculative.cpp`)**：
   * **改造点**：保留标准草稿循环，但在目标模型批量验证 $K$ 个 Token 时，将各层 Gating 预测出的专家集合提前向 `moe_scheduler` 进行广播，触发跨步批量预取。

### 4.3 彻底重写 / 纯自研模块 (Group C)

#### 1. 异步 Direct I/O 引擎 (`src/io/`)
* **替代目标**：彻底废弃 `llama-mmap.cpp`。
* **技术实现**：
  * Windows：`CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` + `IOCP` + `GetQueuedCompletionStatusEx`。
  * Linux：`io_uring`（支持 fallback 到 `io_submit`）。
  * 单专家动态临时缓冲区：容量为 $\sum (\text{TensorSize}_i + 8\text{KB})$，自动执行 4KB 扇区对齐直刷并 `memcpy` 汇聚进紧凑 Slot。

#### 2. Pinned RAM / VRAM 内存池 (`src/pool/`)
* **替代目标**：替代 `llama-alloc.c` 对 MoE 专家的管理。
* **技术实现**：
  * 预分配静态连续 4KB 对齐大块内存（Windows `VirtualAlloc` 锁页 / Vulkan Device Local Memory）。
  * 维护固定 Slot 数组，包含 `SLOT_PIN_LOCKED`, `SLOT_IO_INFLIGHT`, `SLOT_READY` 等状态机，结合 LRU 时钟与 `EST1` 频次。

#### 3. 双线程协同调度器 (`src/scheduler/`)
* **技术实现**：
  * **推理线程**：下发 Hit GEMM 算子 $\to$ 等待 Miss Ready 事件 $\to$ 补算合并。
  * **调度线程**：监听 Miss 列表 $\to$ 驱逐非保护 Slot $\to$ 下发异步 DIO 批量读盘 $\to$ 下发 Vulkan Staging DMA 传输 $\to$ 触发 Ready 事件。

#### 4. 静态子图指针重绑定执行器 (`src/engine/`)
* **替代目标**：替代 `ggml_mul_mat_id` 运行时动态大图。
* **技术实现**：
  * 引擎初始化时预编译 $1 \dots K$ 专家的固定子图（`gpu_subgraphs[K]`, `cpu_subgraphs[K]`）。
  * 推理时仅修改 `tensor->data = slot->raw_ptr + tensor_offset`，零动态分配与零建图开销。

#### 5. 5维资源自适应决策机 (`src/engine/state_machine.cpp`)
* **技术实现**：
  * 实时采集 CPU、GPU、PCIe、Disk I/O、Draft 接受率指标，在 9 种状态之间平滑流转（如自动降阶 CPU GEMM、限制 DMA 速率、换页抖动时紧急复位）。

---

## 5. 后续跟踪上游 `llama.cpp` 更新的维护指南 (Upstream Tracking)

为了让 StreamMoE 能够长期轻松同步 upstream `llama.cpp` 的最新优化（如新的量化格式、算子加速、新模型架构）：

1. **Submodule 纯净性原则**：
   * 严禁直接在 `third_party/llama.cpp` 源码树中散落修改。
   * 如果必须微调上游代码，通过轻量的 `include` 拦截、继承包装或集中在 `src/adapter/` 补丁层中。
2. **更新同步流程**：
   ```bash
   cd third_party/llama.cpp
   git fetch origin
   git merge origin/master
   cd ../..
   # 运行 StreamMoE 的单元测试套件验证兼容性
   .\build.bat test
   ```
3. **接口版本隔离**：
   * StreamMoE 核心模块（`src/io/`, `src/pool/`, `src/scheduler/`）只依赖标准的 C 结构与 `ggml-backend.h` / `gguf.h` 稳定接口，避免依赖 `llama.cpp` 内部易变的私有实现。