# StreamMoE Bug 追踪清单 (BUG_TRACKER.md)

> 来源：`Bugs.txt` 外部审计 + 2026-08-25/26 对当前 main 分支源码逐条复核（全部属实，已标注 file:line）。
> 状态图例：`OPEN` 未修 / `FIXED` 已修并提交 / `WONTFIX` 有意保留 / `GONE` 由架构替换消除 / `SUPERSEDED` 由新架构整体取代 / `PARTIAL` 部分解决。

---

## 0. 数值类 bug 排查方法论（2026-08-26 立）

> 适用：mini-graph 委托 / 数值等价回归类问题（"数据全对但输出错"这种看似悖论的情况）。
> 铁律：**先打 log 后思考；以事实/测试为基准，不做无限猜想。**

1. **遇到费解的问题，第一步永远是"全量 log + 全量 dump"**：把所有能够 log 的地方都打上 log，并 dump 所有重要的相关数据（输入张量字节、权重字节、ids、中间量、输出），宁可多不可漏。log/dump 直接 `fopen` 写文件，不要用 stderr（见第 3 条）。
2. **用二分法缩小错误范围**：对比"正确的基准（base）"与"出错的对象（moe）"，从整条调用链两端往中间收拢——先确认数据/装载/路由/元数据哪些一致、哪些不一致，把范围一分为二，重复到锁定单个函数。
3. **避开 PowerShell 陷阱**：PS 5.1 的 `2>` 会把原生 stderr 转成 UTF-16 ErrorRecord 并**截断长行**（本次 `[ids]`/`[slot]` 行被截到 ~100-120 字符，误导结论）。用 **cmd .bat 重定向**（`> file 2> file`）或程序 `fopen` 直写，再用 opencode 原生 Read/Grep 查看。
4. **锁定到某个函数后，逐行打 log**：把**每个输入、每个输出、每一行表达式**都 log 下来，只要硬盘装得下、执行时间合理。SIMD 变量用 `_mm_storeu_*` 转数组打印；不要凭读代码猜测表达式语义。
5. **能脱离主程序就脱离**：把复现挪进独立诊断程序（如 `temp/iso.cpp`，输入从 trace 读、权重从文件读），脱离 sched/多线程环境；先单线程跑排除竞态。
6. **单点执行优先于猜想**：能单线程/单步/调试器执行就单点执行，逐点看值；无法用调试器时就用第 4 条的逐行 log 等效替代。
7. **以事实为基准，不做无限思考**：优先用测试建立**确定的、不可辩驳的基准**（如"base 的参考输出"、"文件字节校验和"、"trace 记录逐位对比"），用实验结果推演下一步；当多个猜想无法用推理区分时，直接做判别式实验（改变一个变量看结果是否随之改变）。
8. **修复后复用同一套 log/dump 逐项对照确认**，再做端到端测试（输出逐字 vs 基准），最后清理诊断代码。

> 本次实战记录：`docs/DEBUG_DELEGATION.md`（2026-08-26，从"数据全对但输出错"到根因"delegate 紧凑读 stride=1024 的 ids"的完整过程）。



## P0 推理真实性（generation 为模拟，输出不可信）

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B01 | FIXED | tokenize() 是贪心最长匹配，未执行 BPE merge；`bpe_ranks_` 为死代码，prompt token IDs 错误 | src/tokenizer/tokenizer.cpp:154-194 | ✅ 新引擎使用 libllama 内置 tokenizer + 模型自带 chat template（两段式读取）|
| B02 | FIXED | 输出 token ID = `(step*17+100) % vocab_size` 公式伪造 | src/main.cpp:191 | ✅ 真实 logits 采样链（top-k/top-p/temp/dist，元数据默认值可覆盖）|
| B03 | FIXED | expert routing = `(step*3+l*7+k*13) % n_expert` 公式伪造 | src/main.cpp:161 | ✅ 模型真实 router（sqrt-softplus gating + hash 层 tid2eid）|
| B04 | FIXED | prefill = `sleep_for(5ms)`，prefill TPS 无意义 | src/main.cpp:146 | ✅ 真实 prefill 计时 |
| B05 | FIXED | decode 每 token `sleep_for(8ms)`，TPS 含人为延迟 | src/main.cpp:195 | ✅ 真实 decode 计时 |
| B06 | FIXED | subgraph_executor 是标量乘 mock：`out += in * w`，slot.raw_ptr 未参与计算 | src/engine/subgraph_executor.cpp:58-66 | ✅ 已从仓库删除；route B 以自定义 backend 的 mini-graph 委托取代 |
| B07 | FIXED | 无 embedding/attention/RoPE/router/LM head/sampling 前向链 | 全仓 | ✅ 上游 deepseek4 完整图 |
| B08 | FIXED | KV cache 只有 storage/snapshot，与前向零集成 | src/kv/* | ✅ libllama KV 管理（含 dsv4 SWA/压缩 KV）+ 跨轮前缀复用 |
| B09 | FIXED | MLA 仅 metadata 识别，无推理实现 | src/loader/moe_loader.cpp:139-143 | ✅ 上游 deepseek4 MLA/DSA indexer/hyper-connections/Sinkhorn |
| B10 | PARTIAL | GPU 执行不存在：`prof.gpu_hits=0` 写死；无任何 GPU kernel | src/main.cpp:207 | `-ngl/--kv-placement/--moe-vram-pool` 已接线；Vulkan 后端待接入（route B Phase B）|
| B11 | OPEN | speculative decoding 未实现（mock 引擎已删）| 已删（原 src/engine/speculative_engine.cpp）| 待接 libllama --model-draft (dflash/MTP)；上游已支持，草稿模型在 N:\AI_LLM\DeepSeek-V4-Flash-0731\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf |
| B12 | FIXED | 遥测硬编码 cpu_load=0.70/gpu_load=0.40/cache_hit_rate=0.85 | src/engine/state_machine.cpp | ✅ /stats 与 profile JSONL 全部真实计时计数 |
| B13 | FIXED | state machine 无遥测闭环 | src/main.cpp | ✅ 真实引擎路径不再依赖状态机模拟 |

### INC 期间新增

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B33 | FIXED | 加载 162GB UD 量化模型内存爆满：repack extra bufts 对 Q4_K/Q5_K/Q6_K/Q2_K/Q4_0 张量整体物理拷贝 | llama_model_params.use_extra_bufts 默认 true | ✅ 引擎强制关闭；全部权重零拷贝 mmap（见 INC-1）|
| B34 | FIXED | 覆盖 llama_batch.seq_id 上游指针 -> llama_batch_free 重复释放堆损坏 c0000374 | src/engine/llama_engine.cpp | ✅ 只填值不换指针（见 INC-2）|
| B35 | FIXED | chat template 元数据超 1024 字节被截断导致 apply_template 失败 | read_meta_str 固定缓冲 | ✅ 两段式按需长度读取 |

## P1 正确性/资源管理 bug（独立于推理路线）

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B14 | SUPERSEDED | `--moe-preload` 未读 GGUF 即 mark_ready，空内存当合法权重 | 已删（原 src/main.cpp:325-337）| route B 槽控制面替代 |
| B15 | SUPERSEDED | IO 失败后 slot 永久 IO_INFLIGHT\|PIN_LOCKED | 已删（原 src/scheduler/moe_scheduler.cpp）| Backend.md slot_meta 含 FAILED 态替代 |
| B16 | SUPERSEDED | wait_miss_ready 超时 slot 永久 PIN | 已删（原 src/scheduler/moe_scheduler.cpp）| Backend.md refcount/generation 替代 |
| B17 | SUPERSEDED | timeout 按 expert 各自消耗 | 同 B16 | 同上 |
| B18 | FIXED | VirtualLock/mlock 返回值忽略 | src/pool/expert_pool.cpp:39,45 | ✅ 检查返回值+提升工作集配额+is_pinned() |
| B19 | FIXED | shard 缺失仅 WARN 继续 | src/loader/moe_loader.cpp:186-190 | ✅ 分片缺失/打不开/数量不符一律 throw |
| B20 | FIXED | expert 切片假设等大连续布局未校验 | src/loader/moe_loader.cpp:249-250 | ✅ 三重校验（整除/ne[2]/quant block 对齐）|
| B21 | SUPERSEDED | route_and_prefetch 重复专家 double-pin | 已删（原 src/scheduler/moe_scheduler.cpp）| Backend.md MPSC/directory 替代 |
| B22 | OPEN | POSIX "async DIO" 实为同步 pread，无 io_uring | src/io/async_dio_posix.cpp | route B 规划 io_uring/io_submit fallback |
| B23 | FIXED | POSIX completed_queue_ 无锁非线程安全 | src/io/async_dio_posix.cpp | ✅ completed_mutex_ |
| B24 | SUPERSEDED | scheduler 单 worker，expert 间串行 | 已删（原 src/scheduler/moe_scheduler.cpp）| Backend.md 调度线程替代 |
| B25 | FIXED | compute/IO 重叠未接入主路径 | src/main.cpp | ✅ 真实引擎替换 mock 主路径；双池并发由 Backend.md 阶段实现 |
| B26 | FIXED | `-ngl/--gpu-layers`/`--moe-vram-pool` 未接线 | src/main.cpp:34-35 | ✅ -ngl/offload_kqv/load_mode 已映射；--moe-vram-pool 待 GPU 池 |

## P2 统计与测试可信度

| ID | 状态 | 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B27 | FIXED | hybrid 打分 freq_val 无 [0,1] 归一化，退化为 frequency dominance | src/pool/expert_pool.cpp:187-189 | ✅ 读时按当前最大分归一化 |
| B28 | FIXED | 历史 global_counts 与 session 分数尺度混合 | src/pool/expert_stats.cpp | ✅ 同 B27 机制 |
| B29 | FIXED | "EMA" 语义混叠 | src/pool/expert_stats.cpp | ✅ 澄清为 recency-weighted decaying counter |
| B30 | PARTIAL | cache hit rate 基于 fake routing | src/main.cpp | ✅ fake 指标随 mock 路径移除；真实命中率待 route B 提供路由观测 |
| B31 | PARTIAL | 测试断言过弱 | tests/* | ✅ 端到端真实输出验证已建立；单元级黄金用例待补 |
| B32 | FIXED | prompt_tokens 分母不可信 | src/main.cpp:130 | ✅ libllama tokenizer 计数 |

## P3 保留资产

GGUF 元数据/多分片发现、per-expert offset/read plan、4KB sector staging reader、Windows IOCP DIO、pin/unpin 生命周期语义、LRU/LFU/EST1 驱逐算法、RDTSCP profiler + JSONL、HTTP server、bench_agent.js 测评线。详见 `docs/CODEBASE_AUDIT.md`。

---

## 修复批次记录

| 批次 | Commit | 内容 | 关联 Bug |
|------|--------|------|----------|
| 0 | 92ff91d | 建立追踪清单 + 架构分析文档 | - |
| 1 | 6f47eaf | 保留模块修复（pool pin 校验、shard 严格校验、切片三重验证、POSIX 队列加锁、EST1 归一化）| B18 B19 B20 B23 B27 B28 B29 |
| 2 | af1c595 | 新引擎落地（libllama deepseek4 真实推理核心、CLI 参数映射）| B01-B07 B09 B12 B13 B26 B30 B32 |
| 3 | 234b8e0 | 内存爆掉根因修复（repack 关闭）+ 堆损坏修复（seq_id）+ 模板两段读取 + idx=-1 | B33 B34 B35 |

## 事故记录

### INC-1: 加载 162GB 模型内存爆满（已修复）
- **现象**：加载 UD-Q8_K_XL 全量分片后提交内存冲满 128GB。
- **根因**：`use_extra_bufts=true` 使 repack buffer type 对 Q4_K/Q5_K/Q6_K/Q2_K/Q4_0 张量在加载时整体物理拷贝；UD 动态量化正是这些类型混合。
- **修复**：引擎强制 `use_extra_bufts=false`，全部权重零拷贝 mmap（实测加载秒级、WS 52MB）。

### INC-2: llama_batch.seq_id 指针覆盖 -> 堆损坏 c0000374（已修复）
- **根因**：`llama_batch_init` 已为每槽 calloc seq_id[i]，我方用共享数组覆盖，`llama_batch_free` 逐槽 free 重复释放。
- **教训**：llama_batch 所有权归上游，调用方只填值不换指针。

### INC-3: memwatch 哨兵实测 —— 内存增长真因 = file-backed working set
- **方法**：vendored llama.cpp 4 个 ggml 分配点插桩（见 `patches/README.md`），WS>90GB 自动退出。
- **实测（单 prompt 24 token）**：107,369 条记录；max WS=42.1GB、max PRIV=1.7GB、max FILEBK=41.5GB。
- **结论**：内存增长几乎全为 mmap 模型的 file-backed 页驻留（128GB 无压力系统不回收）；私有大拷贝/泄漏排除。
- **主线修复方向**：route B 自定义 backend + 槽池（私有提交内存、物理有界），dense 保持默认 mmap。

## 性能基准记录（2026-08-26）

### 环境事实
- 模型盘 N: 为 **USB 转接 NVMe**（早期 CIM 探测误判为 iSCSI，已纠正）。
- 模型 162GB + 草稿 10GB 放在 N:。

### 实测（70GB RAM 参数 / mmap 页缓存基线 / 16 线程 CPU / temp=0.6 top_p=0.95）
> 注：temp=0.6 为早期实测采样设置；当前建议遵循 `docs/SAMPLING.md`（agentic：temp=1.0 top_p=0.95）。
- EN/ZH 双语输出正确且可读（`benchmark/results/conversation_real_*.txt`）。
- decode TPS 0.3~2 tok/s，由 N: 冷专家页拉取主导；页缓存随运行变热。
- route B 槽池（DIO 预取 + EST1 驻留）正是针对此瓶颈的架构解；mmap 基线为其对照组。
