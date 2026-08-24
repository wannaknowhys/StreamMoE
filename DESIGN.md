# StreamMoE (OffloadMoE) 架构与详细设计文档

> **版本**：v0.1.0-draft  
> **更新时间**：2026-08-24  
> **文档维护原则**：本项目核心架构、关键数据结构、调度策略及接口规范的唯一权威设计文档。随项目迭代和设计变更持续同步更新。

---

## 1. 项目定位与核心需求 (Core Intent)

* **引擎目标**：开发名为 **StreamMoE**（或 **OffloadMoE**）的高性能、内存极致优化的 MoE 推理引擎。
* **核心痛点**：突破现有引擎（如默认 `llama.cpp`）物理 RAM/VRAM 必须容纳全量权重的限制。允许在 **物理内存远小于模型体积**（例如 32GB RAM / 8GB VRAM 跑 100GB+ MoE）的环境下，实现低延迟、高吞吐的混合推理。

---

## 2. 核心物理观察与基础假设 (Core Hypotheses)

1. **专家局域性 (Expert Locality)**：
   * Decode 阶段具有极强路由局部性，少量 Hot Experts 承担绝大部分计算需求，具备高 Cache Hit Rate 空间。
2. **Dense 与 MoE 拓扑物理彻底解耦**：
   * Dense 骨干（Attention / LayerNorm / Embedding / Final Head / Shared MLP）是每 Token 必经的平权计算；
   * MoE 专家是稀疏动态选通的。两者物理内存管理与生命周期彻底分离。
3. **同构专家与动态 Tensor 描述 (Homogeneous Expert & Dynamic Sub-Tensors)**：
   * 单层单个专家作为一个 4KB 对齐的 `expert_blob_t` 结构体。
   * **动态 Tensor 组成**：单个专家内部包含的 Tensor 数量（如 SwiGLU 结构的 `gate`, `up`, `down` 3 个 Tensor，或融合结构的 2 个 Tensor）在模型加载时**动态解析，严禁代码写死**。
   * **初始版本同构假设**：同模型内所有同层/异层专家具有严格相同的总字节尺寸与量化类型。程序启动时检测各专家尺寸，若发现大小不一立即报错阻断。
4. **硬件 Direct I/O (DIO) 优于 OS Page Fault**：
   * 彻底放弃 OS 默认 `mmap` 的 4KB 缺页中断机制，避免 Page Fault 导致的内核线程上下文颠簸与不可控 I/O 阻塞。
   * 采用用户态自研 Pinned Pool，结合 4KB 硬件 Sector 对齐的异步 Direct I/O 直刷与零拷贝 DMA（Host to Device）。

---

## 3. 分层存储池与加载架构 (Memory Hierarchy & Pools)

### 3.1 三级存储层次 (L0 / L1 / L2) 与多设备协作 (Multi-Device Co-op)

* **存储拓扑性质**：纯只读的 **NINEC (Non-Inclusive Non-Exclusive Cache)** 系统。
  * 各层各设备缓存独立管理，允许同一高频 Expert 在多设备（如 GPU 0 VRAM、GPU 1 VRAM、CPU RAM）中存在只读多副本。
  * 抽象为统一的 `multi_device_coop` 拓扑：`Device 0 (GPU Vulkan) + Device 1 (GPU Vulkan) + Host (CPU AVX2/OpenMP)`。

```
┌───────────────────────────────────────────────────────────────┐
│              L0: Pinned VRAM Pool (Per-GPU Vulkan)            │
│  - 预分配静态 Slot 数组 [0 ... N_vram_slots - 1]              │
│  - 存放高频专家，GPU GEMM 直接执行 (支持多 GPU 副本)          │
└───────────────────────────────▲───────────────────────────────┘
                                │ (PCIe DMA, Vulkan Staging/Buffer Copy 异步提交)
┌───────────────────────────────┴───────────────────────────────┐
│                    L1: Pinned RAM Pool                        │
│  - 4KB 对齐用户态锁页大块内存 (VirtualAlloc / mlock)          │
│  - 预分配紧凑静态 Slot 数组 [0 ... N_ram_slots - 1]           │
│  - 既作为 CPU GEMM 计算区，又作为各 GPU 的 DMA Staging 缓冲区 │
└───────────────────────────────▲───────────────────────────────┘
                                │ (Direct I/O 异步扇区直读 + 动态 Multi-Tensor Staging)
┌───────────────────────────────┴───────────────────────────────┐
│                    L2: NVMe SSD / Disk                        │
│  - GGUF / .smoe 模型物理文件，按 4KB 扇区对齐异步直读         │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 动态 Multi-Tensor Direct I/O 扇区对齐与 Staging 机制

#### 原生 GGUF 物理现状：
* 在原生 GGUF 文件中，单层单个专家由 $N$ 个独立的 Sub-Tensor 组成（例如 $N=3$：`ffn_gate_ex`, `ffn_up_ex`, `ffn_down_ex`，或者 $N=2$：`gate_up`, `down` 等，根据模型架构动态确定）。
* 各 Sub-Tensor 在 GGUF 文件内部的物理偏移并不连续，且各自起始位置未必恰好落在 4KB 扇区边界上。

#### 动态 Staging Buffer 容量计算与读取机制：
1. **单 Tensor 扇区对齐计算**：
   对于单专家内的第 $i$ 个 Tensor（文件偏移 $\text{offset}_i$，大小 $\text{size}_i$）：
   $$\text{aligned\_start}_i = \lfloor \text{offset}_i / 4096 \rfloor \times 4096$$
   $$\text{aligned\_end}_i = \lceil (\text{offset}_i + \text{size}_i) / 4096 \rceil \times 4096$$
   $$\text{read\_len}_i = \text{aligned\_end}_i - \text{aligned\_start}_i \le \text{size}_i + 8\text{KB}$$
2. **总 Staging Buffer 需求**：
   单个专家读取所需的临时 Direct I/O 缓冲区大小为所有 Sub-Tensor 对齐需求之和：
   $$\text{StagingBufferSize} = \sum_{i=1}^{N} (\text{size}_i + 8\text{KB}) = \text{total\_expert\_size} + N \times 8\text{KB}$$
3. **批量 Direct I/O 直刷与组装**：
   * 调度线程针对这 $N$ 个 Tensor 一次性生成 $N$ 个 `aio_req_t` 请求并批量提交 IOCP / io_uring。
   * DIO 完成后，分别将每个 Tensor 有效载荷从 Staging 缓冲区 `memcpy` 复制到目标 Slot 对应的 `slot_offset_i` 处。
   * **彻底维持 Slot 内部内存紧凑且 4KB 对齐，动态适应任意 MoE 架构（2/3/4 个 Tensor）**。

#### 未来演进（专属转换格式 `.smoe`）：
* 专属离线转换工具将单专家的 $N$ 个 Tensor 提前合并打包为连续且整体 4KB 对齐的单一 Blob，实现真正的 **Zero-Copy 单次 DIO 直刷**。

### 3.3 专家物理 Slot 与元数据描述 (C-Style Struct)

```cpp
enum slot_status_flags : uint32_t {
    SLOT_EMPTY       = 0,
    SLOT_PIN_LOCKED  = 1 << 0, // 推理线程正在使用，禁止驱逐
    SLOT_IO_INFLIGHT = 1 << 1, // 异步 DIO / DMA 正在传输
    SLOT_READY       = 1 << 2, // 数据有效，可供 GEMM 计算
};

#define MAX_TENSORS_PER_EXPERT 8

// 单个 Tensor 切片的物理描述
struct expert_tensor_desc_t {
    uint64_t file_offset;     // GGUF 文件中实际物理字节偏移
    uint64_t byte_size;       // 实际数据字节大小
    uint64_t slot_offset;     // 在内存 Slot 中的紧凑相对偏移
    int32_t  ggml_type;       // 量化格式 (Q4_K, Q8_0, F16 等)
    int64_t  ne[4];           // 维度形状 (ne[0]...ne[3])
};

// 专家整体元数据规格 (模型初始化时动态计算，严禁硬编码)
struct expert_layout_meta_t {
    uint32_t             num_sub_tensors;  // 单专家包含的 Sub-Tensor 数量 (动态检测)
    size_t               total_slot_size;  // Slot 紧凑总大小 (向上对齐到 4KB)
    size_t               dio_staging_size; // 单专家 DIO 临时缓冲区需求: sum(size_i + 8KB)
    expert_tensor_desc_t tensor_descs[MAX_TENSORS_PER_EXPERT];
};

// 单个专家在内存池中的槽位元数据
struct expert_slot_t {
    int32_t  layer_idx;        // 所属层号 (-1 表示空闲)
    int32_t  expert_idx;       // 专家编号 (-1 表示空闲)
    uint32_t flags;            // slot_status_flags
    uint64_t last_access_seq;  // LRU 访问时钟/序号
    uint64_t access_count;     // 累积调用频次 (结合 EST1 统计)
    void*    raw_ptr;          // 4KB 对齐的物理数据缓冲区指针 (Host Pinned 或 Device Pointer)
};

// 预分配的连续内存池
struct expert_pool_t {
    size_t               slot_size;    // 单个 Expert Blob 规格 (4KB 对齐)
    uint32_t             num_slots;    // 槽位总数
    uint8_t*             base_ptr;     // 大内存块基地址 (Pinned Memory)
    expert_slot_t*       slots;        // 槽位描述符数组
    expert_layout_meta_t layout_meta;  // 专家统一物理布局
};
```

### 3.4 四阶段预加载策略 (`--moe-preload`)

| 模式 | 启动行为 | 适用场景 | 启动耗时 |
| :--- | :--- | :--- | :--- |
| `all` | 启动时将全量 MoE 专家预载入 RAM/VRAM Pool | 物理 RAM 充裕环境 | 较慢 (全量读盘) |
| `vram` | 启动时仅填满 VRAM Pool (根据历史热度)，其余留空 | 追求首 Token 命中率 | 中等 |
| `ram` | 启动时仅填满 RAM Pool，VRAM 留空等待运行时 DMA | 平衡型 | 中等 |
| `none` | **秒级启动模式**：仅加载 GGUF Header 与 Dense 骨干，MoE 槽位全空，推理时按需冷加载 | 极致冷启动/内存极限压榨 | **< 1 秒** |

---

## 4. 异步 I/O 引擎抽象 (Async Direct I/O Engine)

### 4.1 平台原生异步驱动
* **Windows**: `IOCP (I/O Completion Ports)` + `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` + `GetQueuedCompletionStatusEx` 批量事件收取。
* **Linux**: `io_uring`（高性能异步环），回退机制支持 `io_submit` (Linux Native AIO)。

### 4.2 接口抽象：批处理提交与同步模型 (Batch Submit & Wait)

```cpp
struct aio_req_t {
    void*    file_handle;   // 打开的 DIO 文件句柄
    uint64_t file_offset;   // 4KB 对齐的文件偏移
    void*    aligned_buf;   // 4KB 对齐的目标内存地址
    uint32_t aligned_len;   // 4KB 对齐的读取长度
    void*    user_data;     // 关联的 slot / tensor 或回调上下文
    uint32_t bytes_read;    // 实际读取字节数
    int32_t  error_code;    // 0 表示成功
};

struct async_io_engine_t {
    // 批量提交请求: 提交 tensor 1, 提交 tensor 2, 提交 tensor 3...
    int32_t submit_batch(aio_req_t* reqs, uint32_t count);

    // 批量等待完成: 支持等待至少 min_complete 个请求完成，或按超时返回
    uint32_t wait_events(aio_req_t** out_completed, uint32_t max_events, uint32_t min_complete, uint32_t timeout_ms);
};
```

---

## 5. 双线程协同与 Prefill/Decode 执行流水线

### 5.1 线程拓扑与协同模式

* **计算线程 (Compute Thread)**：负责 Tokenizer、Dense 骨干前向、Gating 路由判定、Hit 专家 GEMM 算子提交、Miss 专家补算合并。
* **调度线程 (Scheduler Thread)**：监听 Miss List 请求、执行 LRU+频次槽位驱逐、下发异步 DIO 读盘任务、执行 PCIe DMA H2D 搬运、通知 Ready 信号。

```
Compute Thread (推理主线)                     Scheduler Thread (调度搬运)
      │                                                │
      ├─► 1. Dense 骨干计算 (Attention, Norm)          │
      │                                                │
      ├─► 2. Gating 路由计算                           │
      │    - 产出 Target Experts                       │
      │    - 拆分 [Hit List] 与 [Miss List]            │
      │                                                │
      ├─► 3. 主动报备与锁定 (Pin Lock) ───────────────►│
      │    - 标记 Hit Slots 禁止驱逐                   │
      │                                                ├─► 4. Slot 分配与 LRU 驱逐
      ├─► 5. 提交 Hit Experts 计算 (异步 GEMM)         │    - 寻找空闲或非保护 Slot
      │    - 结果写入 Layer 临时累加 Buffer            ├─► 6. 批量发起异步 DIO (L2 -> L1 Multi-Tensor Staging)
      │                                                │    - 批量提交各 Tensor 的 aio_req_t
      │                                                ├─► 7. wait_events 完成后 memcpy 各 Tensor 至 Slot
      │                                                ├─► 8. (可选) PCIe DMA 搬运 (L1 -> L0 Vulkan)
      │                                                │
      ├─► 9. 等待 Miss Ready 信号 ◄────────────────────┴─► 10. 标记 READY 状态并发出 Event
      │                                                
      ├─► 11. 提交 Miss Experts 补算并累加             
      │                                                
      ▼ 12. 进入下一层 / Output Head                   
```

### 5.2 Sub-Graph 指针重绑定 (Pointer Rebind)

* **彻底放弃运行时动态建图**（避免运行时动态分配与图解析开销）。
* **静态预建候选子图**：在引擎初始化时，针对 1 到 K 个并发专家的拓扑，预编译好固定的计算子图数组 (`cpu_graphs[K]`, `gpu_graphs[K]`)。
* **运行时指针注入**：
  ```cpp
  // 运行时无需重新分配 Tensor，仅做指针重绑定
  subgraph->nodes[i]->data = pool_slots[slot_id].raw_ptr + tensor_offset;
  ggml_backend_graph_compute(backend, subgraph);
  ```

---

## 6. 草稿模型与投机推理协同机制 (Speculative Decoding)

### 6.1 草稿模型 (Draft Model) 角色定位
* 草稿模型采用紧凑的 **Dense 模型**（如 10G dspark），接受与原生 `llama.cpp` 一致的标准参数。
* 草稿模型自身不含 MoE 路由，无法直接预测目标模型的专家编号。

### 6.2 投机验证与专家路由的协同机制
1. **草稿生成阶段**：草稿模型在 CPU/GPU 上独立快速执行 $K$ 步生成 $K$ 个候选 Token $[T_1, T_2, \dots, T_K]$。
2. **目标模型批量前向 (Target Verification Batch)**：将这 $K$ 个候选 Token 作为一个 Batch 输入 StreamMoE 目标模型进行并行前向验证。
3. **专家路由提前暴露**：
   * 在目标模型执行第 $L$ 层 Dense 骨干计算时，Gating 模块即可一次性计算出该层**这 $K$ 个 Token 所需的所有专家并集**。
   * 这为调度线程提供了多 Token 跨步的“多专家批量 Prefetch”机会。
4. **验证与截断回退**：
   * 目标模型计算完成后，比对 logits，确定接受的前 $M$ 个 Token ($0 \le M \le K$)。
   * 保留接受的 $M$ 个 Token 的 KV Cache，若非 EOS，将最后一个有效 Token 作为下一轮草稿生成的起始前缀。
   * 若接受率持续偏低，触发状态机 **1.5 状态**（缩减 $K$ 或暂停草稿）。

---

## 7. 参数体系与命令行设计

### 7.1 Dense 与 MoE 参数彻底分离

```bash
# 启动示例
./stream_moe \
  -m models/DeepSeek-V2-Lite-Q4_K_M.gguf \
  -ngl 28 \                      # 仅控制 Dense 骨干网络的 GPU 卸载层数 (Vulkan)
  --moe-vram-pool 6144 \         # MoE VRAM Pinned Pool 上限 (6GB)
  --moe-ram-pool 24576 \         # MoE Host RAM Pinned Pool 上限 (24GB)
  --moe-preload none \           # 预加载策略: none | vram | ram | all
  --draft-model models/dspark.gguf \ # 草稿模型 (Dense 骨干)
  --threads 16
```

### 7.2 参数职责划分

* **`-ngl / --gpu-layers`**：**仅针对 Dense 结构**（Embedding、Self-Attention、LayerNorm、Output Head、Dense MLP），不计入 MoE 专家。
* **`--moe-vram-pool <MB>`**：分配给 MoE Pinned VRAM Slot 的专属显存池大小。
* **`--moe-ram-pool <MB>`**：分配给 MoE Pinned RAM Slot 的专属系统内存池大小。
* **`--moe-preload <policy>`**：控制启动时 MoE 权重的预加载深度（`none`/`ram`/`vram`/`all`）。
* **`--no-kv-offload`**：将 KV Cache 保留在 RAM 中，为 MoE VRAM Pool 挤出宝贵的显存。

---

## 8. 专家热度持久化与多维资源约束状态机

### 8.1 专家热度持久化 (`expert_stats`)

* **存储路径**：`~/.llama/expert_<model_name>.bin`
* **二进制文件格式**：
  ```
  [0x00 - 0x03]: Magic (0x31545345, "EST1")
  [0x04 - 0x07]: uint32_t n_layer
  [0x08 - 0x0B]: uint32_t n_expert
  [0x0C - ....]: uint64_t counts[n_layer][n_expert]
  ```
* **作用**：冷启动时自动读取历史统计，结合 `--moe-preload vram/ram` 在启动时即形成先验最优缓存分布。

### 8.2 运行时状态决策矩阵 (State-Decision Matrix)

| 状态 ID | CPU | GPU | PCIe | Disk | Draft 收益 | 核心瓶颈 | 自动化自适应调优动作 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0** | 满载 | 空闲 | 空闲 | 空闲 | 正 | CPU 算力 | 激进将 L1 专家 DMA 升阶到 VRAM；将 Draft 模型卸载到 GPU |
| **1.5** | 满载 | 空闲 | 空闲 | 空闲 | 负/低 | Draft 反噬 | 缩减投机步长 $K$ 或暂停草稿；收回 CPU 算力全量计算 Gemm |
| **1.6** | 满载 | 空闲 | 满载 | 空闲 | - | DMA 抢总线 | 搬运线程主动 Sleep 降速，降低 DMA 频率，减少总线竞争 |
| **2.0** | 闲/中 | 停顿 | 满载 | 空闲 | - | PCIe 带宽 | 缩小 VRAM 动态槽位；部分专家放弃 DMA，由 CPU 在 L1 就地计算 |
| **2.1** | 闲/中 | 中等 | 满载 | 空闲 | 负 | 换页抖动 | 冻结 A/B 槽位切换，锁定当前 Cache 周期 (50~100 Token) |
| **3.0** | 空闲 | 满载 | 空闲 | 空闲 | - | GPU 算力 | 将次热点专家下放给 CPU (AVX2) 异构并行分流；增大 Draft 步长 |
| **4.0** | 空闲 | 空闲 | 空闲 | 满载 | - | Disk I/O | 增加预取深度；触发冷专家异步近似/Shared 兜底逻辑 |
| **4.1** | 伪闲 | 空闲 | 空闲 | 空闲 | - | 内存带宽 | 降低 CPU Gemm 线程数以减少 DDR 通道争抢；采用更低 bit 量化 |
| **5.1** | 满载 | 满载 | 满载 | 满载 | 负 | 系统抖动 | **紧急复位 (Emergency Reset)**：清空预取队列，退回纯静态安全模式 |

---

## 9. 代码架构与工程规范 (Engineering Standards)

1. **编程范式**：
   * **C-Style C++**：核心以 `struct`、平铺数据结构、函数式 API 为主，避免深层继承与过度虚函数封装。
   * **RAII & Modern Idioms**：关键系统资源（文件句柄、Direct I/O 缓冲区、线程、同步事件）使用 RAII 结构管理，杜绝资源泄漏。
2. **依赖集成范围**：
   * 通过 Git Submodule 引入 `third_party/llama.cpp`。
   * **仅保留与提取**：GGUF Header 解析器、Tensor 物理描述、`ggml-backend` (Vulkan/CPU) 泛型硬件接口。
   * **彻底重写**：接管 `ggml_mul_mat_id` 与 MoE 算子，自研 Direct I/O 读取器、Pinned Pool 管理器及双线程流水线。
3. **构建体系**：
   * 跨平台 `Makefile` + Windows 本地环境专用的 `build.bat`（集成 LLVM OpenMP `clang++` 与 MSVC CRT）。
   * 严格遵循 Win32 Unicode (`_UNICODE`, `CreateFileW`) 与 64 位原生编译规范。