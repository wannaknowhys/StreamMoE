# StreamMoE Expert Offload x llama.cpp 集成架构分析 (EXPERT_OFFLOAD_INTEGRATION.md)

> 目标：回答"llama.cpp 现有图/执行模型能否承载 StreamMoE 的动态专家调度"，给出最小改动方案。
> 基于 vendored llama.cpp @ f280b2698（已完整支持 `deepseek4` / `dflash`）。

---

## 1. 双侧执行路径实测追踪

### 1.1 llama.cpp DeepSeek4 前向路径

```
llama_model_deepseek4::build_arch_graph()                 src/models/deepseek4.cpp:182
  -> graph ctor (deepseek4.cpp:1219) 逐层构建:
       build_attention   : MLA(q_a/q_b latent) + DSA indexer(ratio=4层)
                           + 压缩KV状态机(compress_ratios) + attn_sinks
       build_hc_pre/post : hyper-connections(多流) + Sinkhorn 归一化
       build_moe_ffn     (deepseek4.cpp:1284 -> llama-graph.cpp:1941):
           logits = mul_mat(ffn_gate_inp [n_embd, n_expert], cur)
           gating = sqrt(softplus(logits))          # DSV4 专用 GGML_PREC_F32
           前 dsv4_hash_layer_count 层:
               selected = get_rows(ffn_gate_tid2eid, tokens)   # token->expert 确定性映射!
           其余层: top-k(argsort_top_k(selection_probs + exp_probs_b))
           up   = ggml_mul_mat_id(ffn_up_exps,   cur, ids)
           gate = ggml_mul_mat_id(ffn_gate_exps, cur, ids)
           swiglu(clamp per-layer swiglu_clamp_exp/shexp, deepseek4 split-swiglu 路径)
           out  = ggml_mul_mat_id(ffn_down_exps, act, ids)
       shared expert: build_ffn(ffn_*_shexp) 后与 moe_out 相加
```

关键事实：

| # | 事实 | 证据 |
|---|------|------|
| F1 | 图是**静态拓扑**，路由 ids 是运行期数据张量，宿主在图内无法插桩 | `ggml_argsort_top_k` -> `ids` 仅作为 `mul_mat_id` 的 src[2] |
| F2 | 每个 `ffn_{gate,up,down}_exps` 是**单一 3D 张量** `[d_in, d_ff, n_expert]`，expert e 位于字节偏移 `e*nb[2]`；`ggml_mul_mat_id` 要求整张量在执行后端可寻址 | llama-graph.cpp:2119/2138/2151 |
| F3 | **算子跟随权重所在后端**："operations with weights are preferably run on the same backend as the weights"；`--cpu-moe/-cmoe` 通过 `tensor_buft_overrides` 把 `blk.*.ffn_*_exps.*` 放入 CPU buffer，其余放 GPU | ggml-backend.cpp:940-971, common/arg.cpp:2740 |
| F4 | **上游已有"选择性专家搬运"缝隙**：当 host 权重 buffer 喂给设备端执行的 MUL_MAT_ID 时，调度器回读 ids、bitset 已用专家、**只拷贝被选中专家的切片** host->device | ggml-backend.cpp:1641-1725 (`copy_experts`) |
| F5 | split 顺序执行，跨后端用 event 同步；图内部无用户回调；层间无钩子 | ggml-backend.cpp:1594-1740 |

### 1.2 StreamMoE 侧路径

```
moe_loader::parse_gguf_topology      src/loader/moe_loader.cpp:93
  扫描全部分片 -> 定位 blk.N.ffn_{gate,up,down}_exps
  按 total_size/n_expert 切片 -> (shard, abs_offset, bytes) -> 4KB 对齐 read_plan
expert_pool                          src/pool/expert_pool.cpp:17
  VirtualAlloc + VirtualLock 固定内存 slots；EMPTY/IO_INFLIGHT/READY/PIN_LOCKED 生命周期
moe_scheduler                        src/scheduler/moe_scheduler.cpp
  route_and_prefetch(hit=pin / miss=evict+queue) -> 单 worker read_expert_sync(DIO)
  -> staging memcpy 进 slot -> mark_ready；wait_miss_ready 阻塞计算线程
async_dio_win                        Windows IOCP 真异步；posix 版为同步 pread
subgraph_executor                    标量乘 mock（未接主路径）
```

---

## 2. 核心问题的回答

### Q1: 不改图的前提下，llama.cpp 能否承载动态专家调度？

**能。** 图本身（MLA/DSA/HC/Sinkhorn/routing/shared-experts 全部数学）一行都不需要改。
调度语义存在于图**之下**的 buffer/backend/sched 层，且上游已留好两个缝：
- F3：权重的物理位置决定算子落点 —— 专家权重放哪是自由变量；
- F4：ids 在 split 边界可被宿主读取，且上游已经实现"按需只搬被选中的专家切片"。

### Q2: 只需 expert-ID/slot 重映射适配器，还是必须自建子图执行器？

**分层结论：**

| 级别 | 内容 | 改动面 | 结论 |
|------|------|--------|------|
| L0 | mmap + OS page cache 当池（`--cpu-moe --mlock`） | 0 行代码 | 可用的基线；无 EST1/DIO 控制 |
| L1 | **自定义 ggml backend/buffer-type**：把 `ffn_*_exps` 的存储绑定到 StreamMoE slot 池，并在该后端内实现 `MUL_MAT_ID`（直接从 slot 反量化 GEMM，miss 时阻塞等待 READY，后台线程预取） | 新增文件 + 通过公开扩展点 `params.tensor_buft_overrides` 注入 buft | **推荐的适配器路线**。零图改动、命中零拷贝、且在我们的 mul_mat_id 实现里能直接看到真实路由 ids（src[2]），喂给 EST1/speculative 预取器 |
| L1' | 修补 vendored `ggml-backend.cpp::copy_experts`，让拷贝源经过 slot 查找（hit=slot 地址，miss=先同步加载） | ~40 行，动上游文件 | 备选；每步仍要 PCIe 拷贝，不如 L1 |
| L2 | 自建子图执行器（schedule→load→wait→GEMM→merge→续图） | fork 整个执行模型 | **不需要**。只会丧失上游 MLA/DSA/HC 的正确性背书 |

### Q3: CPU 加载与 GPU 计算能否安全重叠？

**能，但有明确的语义边界：**
- 跨层前瞻（L+1 路由未知时预取 L+1）：**不可能非投机地做到** —— ids(L+1) 依赖 L 的输出，F1/F5 决定了图内无观察点。可行替代：
  1. **投机预取**：EST1 历史 + 上一步路由 + hash 层确定性（前 `dsv4_hash_layer_count` 层 token->expert 映射完全确定，可免费完美预取）驱动后台 DIO 线程；
  2. 当前 step 内天然重叠：某层 MoE miss 阻塞只卡该层的 mul_mat_id，attention/shared-expert/HC 仍在别的后端跑。
- 安全条件（StreamMoE 池原语已具备，但当前有泄漏 bug，见 BUG_TRACKER B14-B17）：
  - 计算侧使用期间 PIN_LOCKED 防驱逐；
  - READY 以 release 语义发布、消费侧 acquire 后才允许 kernel 读；
  - loader 线程永不写 pinned-in-use 的 slot。

### Q4: 必须改动的 llama.cpp 抽象清单（L1 路线）

| 抽象 | 是否改 | 说明 |
|------|--------|------|
| `src/models/deepseek4.cpp` / `llama-graph.cpp` 数学 | 否 | 保持上游正确性 |
| `ggml-backend.cpp` 调度器 | 否 | 自定义后端声明 host + 支持 MUL_MAT_ID 后权重原地不动 |
| `llama-model-loader` buffer 选择 | 否 | 复用公开 API `common_params.tensor_buft_overrides`（pattern -> 自定义 buft） |
| 新增 `ggml-backend-reg` 动态注册的自定义后端 | **新增文件** | supports_op(MUL_MAT_ID + Q8_K 等 quant 类型)；buffer 从 slot 池划拨 |
| 预取/EST1 驱动线程 | **新增文件** | 观测 src[2] ids + 投机预测 -> moe_scheduler DIO |

即：**不改任何上游数学/图/调度代码，只新增两个编译单元并经由现有公开扩展点接线。**

---

## 3. 分阶段落地

- **Phase A（本轮）**：链接 vendored libllama 作为计算内核；StreamMoE server/CLI/bench/profiler 外壳保留；CLI 参数映射（`--moe-ram-pool` -> mlock/no-mmap 行为，`--moe-vram-pool`/`-ngl` -> GPU offload 预算，全部参数化不写死）；修复 BUG_TRACKER P1/P2；跑通 long_horizon_prompts 输出可读。
- **Phase B（下轮）**：L1 自定义后端适配器，slot 池真正接管专家存储；真实命中率/EST1 闭环；对比 mmap 基线。

## 4. 环境备注（2026-08-25 实测）

- RAM 128GB（空闲 99GB）—— 70GB 参数可行。
- GPU: Radeon RX 590 **8GB**（WMI AdapterRAM 32 位截断显示 4GB，注册表 qwMemorySize 实为 8GB）；AMD 无 CUDA -> Vulkan 后端；**Vulkan SDK 未安装**（无 glslc/glslangValidator），Phase A 先 CPU-only，SDK 就绪后以参数开启 Vulkan。
- 模型: `N:\AI_LLM\DeepSeek-V4-Flash-0731\` 5 分片共 ~162GB（shard1 仅元数据 5MB 属正常 split 布局）+ dflash 草稿模型 10GB（支持 MTP 投机解码）。
