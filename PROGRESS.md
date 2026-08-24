# StreamMoE (OffloadMoE) 研发进度与任务清单 (PROGRESS.md)

> **版本**：v0.1.0  
> **最后更新**：2026-08-24  
> **状态标记**：`[ ]` 待办 | `[>]` 进行中 | `[x]` 已完成 | `[-]` 已跳过

---

## 1. 测试模型资产与环境登记 (Test Assets)

| 资产类型 | 本地物理路径 | 规格与特性 | 主要测试用途 |
| :--- | :--- | :--- | :--- |
| **小体积 Dense 模型** | `F:\Dev\computer-use\Qwen3-VL-2B-Instruct-Q4_K_M.gguf` | 1.03 GB, Dense 结构 | 快速端到端测试、Dense 骨干推理验证 |
| **多模态投影文件** | `F:\Dev\computer-use\mmproj-F16.gguf` | 0.76 GB | GGUF 解析与基准数据读取 |
| **超大 MoE 分片权重** | `N:\AI_LLM\DeepSeek-V4-Flash-0731\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-*.gguf` | 5 分片共 ~150 GB (Header 5MB, Shards 各 ~46GB) | **严禁全量载入内存**！专用于 Direct I/O 局部扇区直读、多分片定位与冷热池命中/驱逐测试 |
| **Dense 草稿模型** | `N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` | 10.15 GB, 纯 Dense 结构 | 投机推理 (Speculative Decoding) 协同测试 |

---

## 2. 研发里程碑与任务清单 (Milestones & Roadmap)

### Phase 0: 架构设计、上游调研与构建体系
- [x] 确立核心架构方案与物理假设（Dense/MoE 分离、4KB 扇区对齐 DIO、Pinned Pool、NINEC Vulkan 拓扑）
- [x] 编写详细架构设计规范：[`DESIGN.md`](DESIGN.md)
- [x] 引入 `third_party/llama.cpp` 子模块
- [x] 深度调研 `llama.cpp` 算子库与 150+ 模型架构，确立“中心网关拦截”模式：[`LLAMA_CPP_ANALYSIS.md`](LLAMA_CPP_ANALYSIS.md)
- [x] 建立进度跟踪与测试资产清单：[`PROGRESS.md`](PROGRESS.md)
- [x] 搭建项目标准构建工具链（`Makefile` + 基于 LLVM OpenMP 的 `build.bat`）
- [x] 设立隔离的临时编译输出区（`build/` 和 `temp/`）

### Phase 1: 异步 Direct I/O (DIO) 引擎与多 Tensor Staging 机制
- [x] **Windows IOCP 驱动实现** (`src/io/async_dio_win.cpp`):
  - [x] 封装 `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)`
  - [x] 实现 `submit_batch(aio_req_t*, count)` 批量投递
  - [x] 实现 `wait_events()` 基于 `GetQueuedCompletionStatusEx` 批量收取
- [x] **多 Tensor 扇区对齐 Staging Buffer 组装器** (`src/io/staging_reader.cpp`):
  - [x] 动态计算单专家 $N$ 个 Sub-Tensor 的 `offset` / `size` 扇区对齐范围
  - [x] 批量异步 DIO 直刷入 `TotalSize + N*8KB` 临时缓冲区
  - [x] 有效载荷切片 `memcpy` 复制进 4KB 对齐的紧凑内存 Slot
- [x] **Linux `io_uring` / POSIX 驱动骨架** (`src/io/async_dio_posix.cpp`)
- [x] **单元测试 (UT 1)** (`tests/test_async_dio.cpp`):
  - [x] Mock 文件测试：4KB 边界、跨扇区偏移、大块随机读写准确性校验
  - [x] 真实 DeepSeek MoE 分片文件无缓冲 Direct I/O 读取局部 Tensor 校验
  - [x] 16 批次并发异步 IO 投递与收取校验

### Phase 2: Pinned RAM / VRAM 专家槽位内存池
- [ ] **Pinned Host RAM Pool 管理器** (`src/pool/expert_pool.cpp`):
  - [ ] Windows `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)` + `VirtualLock` 锁页大块连续物理内存
  - [ ] 静态 Slot 槽位数组分配与生命周期状态机 (`EMPTY`, `PIN_LOCKED`, `IN_FLIGHT`, `READY`)
- [ ] **VRAM Pool 描述符与 Vulkan Staging 传输通路**:
  - [ ] 映射 `ggml-backend-vulkan` 的 Buffer 与异步 `tensor_set_async` 传输
- [ ] **混合驱逐策略与统计持久化**:
  - [ ] 实现基于 LRU 时钟时序与 `EST1` 频次加权的槽位驱逐选择器
  - [ ] 实现 `save_expert_stats` / `load_expert_stats` (`~/.llama/expert_<model>.bin`)
- [ ] **单元测试 (UT 2)** (`tests/test_expert_pool.cpp`):
  - [ ] 槽位锁定保护（Pin Lock 禁止驱逐）正确性
  - [ ] 槽位并发状态流转与高频命中/换入换出模拟

### Phase 3: GGUF MoE 拓扑动态解析与同构性检查
- [ ] **GGUF MoE 元数据解析器** (`src/loader/moe_loader.cpp`):
  - [ ] 解析 GGUF 中的各层各专家 Sub-Tensor 列表（`gate`, `up`, `down` 等）
  - [ ] 动态提取各 Tensor 偏移、形状与量化类型，生成 `expert_layout_meta_t`
  - [ ] **同构性硬性检查**：遍历所有专家尺寸，若发现尺寸不一致立即报错阻断
- [ ] **单元测试 (UT 3)** (`tests/test_moe_loader.cpp`):
  - [ ] 使用真实 DeepSeek MoE 头部解析各专家 Tensor 偏移与尺寸同构性

### Phase 4: 双线程流水线调度器与静态子图指针重绑定
- [ ] **双线程协同调度核心** (`src/scheduler/moe_scheduler.cpp`):
  - [ ] 推理线程 (Compute Thread) 与 调度线程 (Scheduler Thread) 无锁/轻量事件同步
  - [ ] 选通路由拆分 `Hit List` 与 `Miss List`，主动报备 Pin 锁定
  - [ ] Hit 专家异步 GEMM 提交与 Miss 专家异步 DIO/DMA 搬运重叠执行
  - [ ] Miss 完成 Ready 信号通知与补算合并
- [ ] **静态候选子图与指针重绑定执行器** (`src/engine/subgraph_executor.cpp`):
  - [ ] 初始化时预建 $1\dots K$ 并发专家子图 (`cpu_graphs[K]`, `gpu_graphs[K]`)
  - [ ] 运行时零动态建图，直接进行 `tensor->data` 指针重绑定
- [ ] **单元测试 (UT 4)** (`tests/test_scheduler.cpp`):
  - [ ] 多线程并发下 Hit 计算与 Miss 搬运的时序重叠验证

### Phase 5: 草稿模型投机推理与 5 维资源状态决策机
- [ ] **投机推理验证集成** (`src/engine/speculative_engine.cpp`):
  - [ ] 驱动 Dense 草稿模型运行 $K$ 步
  - [ ] 目标模型批处理验证，提取各层 Gating 专家并集并提前预取
  - [ ] 依据前缀接受率动态调整投机步长 $K$
- [ ] **5 维自适应状态机** (`src/engine/state_machine.cpp`):
  - [ ] 监控 CPU/GPU/PCIe/Disk/Draft 收益
  - [ ] 实现 CPU Gemm 降阶、DMA 限速、以及 Thrashing 换页抖动时的**紧急复位 (Emergency Reset)**
- [ ] **单元测试 (UT 5)** (`tests/test_state_machine.cpp`):
  - [ ] 模拟各种系统瓶颈场景下的状态流转与策略动作触发

### Phase 6: 端到端 CLI 与性能基准测试
- [ ] 整合命令行应用 `stream_moe` (`src/main.cpp`)
- [ ] 支持参数：`-m`, `-ngl`, `--moe-vram-pool`, `--moe-ram-pool`, `--moe-preload`, `--draft-model`
- [ ] 在 32GB RAM / RX590 真实环境下针对 DeepSeek MoE 运行 Benchmark

---

## 3. 测试与编译规范 (UT & Build Protocol)

1. **临时编译隔离**：
   * 所有中间编译文件 (`.obj`, `.o`) 和测试执行文件生成在 `temp/` 或 `build/` 目录下，严禁污染源码树。
2. **测试驱动与 Mocking 原则**：
   * 每个功能模块均配有独立的单元测试源文件 (`tests/test_*.cpp`)。
   * 无外部文件依赖的测试采用 Mock 数据生成器（如生成伪造的 4KB 对齐 Tensor 文件）。
   * 涉及真实物理 GGUF 文件的测试，采用只读小切片/只读 Header，严禁把超大模型全量加载进内存。