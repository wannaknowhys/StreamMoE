# StreamMoE Bug 追踪清单 (BUG_TRACKER.md)

> 来源：`Bugs.txt` 外部审计 + 2026-08-25 对当�?main 分支源码逐条复核（全部属实，已标�?file:line）�?> 状态图例：`OPEN` 未修 / `FIXED` 已修并提�?/ `WONTFIX` 有意保留 / `GONE` 由架构替换消除�?
---

## P0 推理真实性（generation 为模拟，输出不可信）

| ID | 状�?| 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B01 | FIXED | tokenize() 是贪心最长匹配，未执�?BPE merge；`bpe_ranks_` 为死代码，prompt token IDs 错误 | src/tokenizer/tokenizer.cpp:154-194 | �?新引擎使�?libllama 内置 tokenizer + 模型自带 chat template（两段式读取）|
| B02 | FIXED | 输出 token ID = `(step*17+100) % vocab_size` 公式伪�?| src/main.cpp:191 | �?真实 logits 采样链（top-k/top-p/temp/dist，元数据默认值可覆盖）|
| B03 | FIXED | expert routing = `(step*3+l*7+k*13) % n_expert` 公式伪�?| src/main.cpp:161 | �?模型真实 router（sqrt-softplus gating + hash �?tid2eid）|
| B04 | FIXED | prefill = `sleep_for(5ms)`，prefill TPS 无意�?| src/main.cpp:146 | �?真实 prefill 计时 |
| B05 | FIXED | decode �?token `sleep_for(8ms)`，TPS 含人为延�?| src/main.cpp:195 | �?真实 decode 计时 |
| B06 | FIXED | subgraph_executor 是标量乘 mock：`out += in * w`，slot.raw_ptr 未参与任何计算；�?dequant/SwiGLU/gate/up/down GEMM | src/engine/subgraph_executor.cpp:58-66 | �?已移出主路径；真�?MUL_MAT_ID �?libllama CPU 内核执行。Backend.md 自定义后端阶段将以槽池内核取�?|
| B07 | FIXED | �?embedding/attention/RoPE/router/LM head/sampling 前向�?| 全仓 | �?上游 deepseek4 完整�?|
| B08 | FIXED | KV cache 只有 storage/snapshot，与前向零集�?| src/kv/* | �?libllama KV 管理（含 dsv4 SWA/压缩 KV�? 跨轮前缀复用（seq_rm 截断）|
| B09 | FIXED | MLA �?metadata 识别，无推理实现 | src/loader/moe_loader.cpp:139-143 | �?上游 deepseek4 MLA/DSA indexer/hyper-connections/Sinkhorn |
| B10 | PARTIAL | GPU 执行不存在：`prof.gpu_hits=0` 写死；无任何 GPU kernel | src/main.cpp:207 | `-ngl/--kv-placement/--moe-vram-pool` 参数已接线到 llama_model/context params；Vulkan 后端�?SDK 就绪后编译接�?|
| B11 | OPEN | speculative decoding 仅接口模拟：load_draft_model 只查文件存在；acceptance=`step%4` 公式 | src/engine/speculative_engine.cpp, src/main.cpp:185 | 待接 libllama --model-draft (dflash/MTP)；上游已支持 |
| B12 | FIXED | 遥测硬编�?cpu_load=0.70/gpu_load=0.40/cache_hit_rate=0.85 | src/engine/state_machine.cpp �?| �?/stats �?profile JSONL 全部真实计时计数 |
| B13 | FIXED | state machine 无遥测闭环（主循环不�?update_telemetry�?| src/main.cpp | �?真实引擎路径不再依赖状态机模拟；真实指标直�?|

### INC 期间新增

| ID | 状�?| 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B33 | FIXED | 加载 162GB UD 量化模型内存爆满：repack extra bufts �?Q4_K/Q5_K/Q6_K/Q2_K/Q4_0 张量整体物理拷贝 | llama_model_params.use_extra_bufts 默认 true | �?引擎强制关闭；全部权重零拷贝 mmap（加�?6.3s/WS 52MB）。见 INC-1 |
| B34 | FIXED | 覆盖 llama_batch.seq_id 上游指针 -> llama_batch_free 重复释放堆损�?c0000374 崩溃 | src/engine/llama_engine.cpp | �?只填值不换指针；所有权归上游。见 INC-2 |
| B35 | FIXED | chat template 元数据超 1024 字节被截断导�?apply_template 失败 | read_meta_str 固定缓冲 | �?两段式按需长度读取 |

## P1 正确�?资源管理 bug（独立于推理路线，需修复�?
| ID | 状�?| 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B14 | SUPERSEDED | `--moe-preload ram/all` 未读 GGUF �?mark_ready，空内存被当合法权重且后�?find_slot 永久 hit、永不触发磁盘读 | src/main.cpp:325-337 | �?main/pool/scheduler 将被新引�?Backend.md 控制面整体替换，不再单独修补 |
| B15 | SUPERSEDED | IO 失败�?worker �?LOG 不复�?slot -> 永久 IO_INFLIGHT\|PIN_LOCKED，泄漏且永不驱逐，可耗尽整池 | src/scheduler/moe_scheduler.cpp:158-175 | 同上：Backend.md slot_meta(�?FAILED �? 替代 |
| B16 | SUPERSEDED | wait_miss_ready 超时�?miss 不进返回列表�?slot �?pin -> 永久 PIN_LOCKED 泄漏 | src/scheduler/moe_scheduler.cpp:100-130, src/main.cpp:171-177 | 同上：Backend.md refcount/generation 设计替代 |
| B17 | SUPERSEDED | timeout_ms 被每�?miss 单独消耗，非整�?deadline | �?B16 | 同上 |
| B18 | FIXED | VirtualLock/mlock 返回值忽略，"pinned" 只是尝试而非确认，误�?benchmark | src/pool/expert_pool.cpp:39,45 | �?检查返回值；Windows 先提升工作集配额；失败时明确告警 pool �?pin；新�?is_pinned() |
| B19 | FIXED | shard 缺失�?WARN 继续�?50GB 模型静默缺数�?| src/loader/moe_loader.cpp:186-190 | �?分片缺失/打不开/数量�?split.count 不符一�?throw |
| B20 | FIXED | expert 切片假设等大连续布局，未校验 shape/quant block 对齐 | src/loader/moe_loader.cpp:249-250 | �?三重校验：total_size 整除 n_expert、ne[2]==n_expert 交叉验证、slice �?quant block 对齐 |
| B21 | SUPERSEDED | route_and_prefetch 对重�?expert id �?double-pin 并重复入�?| src/scheduler/moe_scheduler.cpp:62-89 | Backend.md MPSC/directory 设计替代 |
| B22 | OPEN | POSIX "async DIO" 实为同步 pread，README 宣称 io_uring 不实（Windows IOCP 为真�?| src/io/async_dio_posix.cpp | Backend.md Phase A 规划 io_uring/io_submit fallback |
| B23 | FIXED | POSIX completed_queue_ 无锁非线程安全；max_in_flight_ 未使�?| src/io/async_dio_posix.cpp | �?completed_mutex_ 保护队列 |
| B24 | SUPERSEDED | scheduler �?worker：expert �?batch 并发、expert 间串行，无跨 expert 流水 | src/scheduler/moe_scheduler.cpp:30-34 | Backend.md 调度线程设计替代 |
| B25 | SUPERSEDED | compute/IO 重叠未接入主路径（executor 从未�?generation loop 调用�?| src/main.cpp | 新引�?+ Backend.md 双池并发设计替代 |
| B26 | OPEN | `-ngl/--gpu-layers`、`--moe-vram-pool` 解析后完全未接线 | src/main.cpp:34-35 | 新引擎参数映射中接线 |

## P2 统计与测试可信度

| ID | 状�?| 问题 | 位置 | 处置 |
|----|------|------|------|------|
| B27 | FIXED | hybrid 打分 freq_val �?[0,1] 归一化，可到 ~50，LRU 项失效退化为 frequency dominance | src/pool/expert_pool.cpp:187-189, src/pool/expert_stats.cpp | �?get_adaptive_frequency 读时按当前最大分归一化到 [0,1] |
| B28 | FIXED | 历史 global_counts 初始化与 session 自增分数尺度不一致，无重归一�?| src/pool/expert_stats.cpp | �?�?B27 同一机制解决（读时归一化，冷启动种子仍来自持久化计数）|
| B29 | FIXED | "EMA" 实为 score *= pow(0.999,n) �?+1 的累加衰减，语义混叠 | src/pool/expert_stats.cpp | �?语义澄清：确认为 recency-weighted decaying counter（合法），头文件注释已更正；配合读时归一化后 hybrid 打分数学成立 |
| B30 | OPEN | cache hit rate 基于 fake routing，只能回�?人工访问模式下缓存表�? | src/main.cpp | Phase A 后自动转为真�?routing 数据 |
| B31 | OPEN | 测试断言过弱：tokenizer 只验 roundtrip 含子串；scheduler 测试�?shard_files �?miss 直接 mark_ready 不做 IO；overlap 测试无时间验�?| tests/test_tokenizer.cpp, tests/test_scheduler.cpp | 修复：加入参�?token IDs 黄金用例 + 真实�?GGUF IO 用例 |
| B32 | OPEN | prompt_tokens 分母不可信（依赖 B01），污染全部 TPS 统计 | src/main.cpp:130 | Phase A |

## P3 保留资产（审计确认真实可用）

GGUF 元数�?多分片发现、per-expert offset/read plan�?KB sector staging reader、Windows IOCP DIO、VirtualAlloc+VirtualLock 槽池、pin/unpin 生命周期、LRU/LFU/EST1 驱逐骨架、scheduler 骨架、RDTSCP profiler + JSONL、HTTP server 骨架、bench_agent.js 测评线。详�?EXPERT_OFFLOAD_INTEGRATION.md�?
---

## 修复批次记录

| 批次 | Commit | 内容 | 关联 Bug |
|------|--------|------|----------|
| 0 | 92ff91d | 建立追踪清单 + 架构分析文档 | - |
| 1 | 6f47eaf | 保留模块修复：pool pin 校验、shard 严格校验、切片三重验证、POSIX 队列加锁、EST1 读时归一化；�?pool/scheduler 缺陷标记 SUPERSEDED（由 Backend.md 控制面替代）| B18 B19 B20 B23 B27 B28 B29；B14-B17/B21/B24/B25 标记 SUPERSEDED |
| 2 | af1c595 | 新引擎落地：vendored libllama deepseek4 真实推理核心（chat template/KV prefix 复用/采样�?SSE），CLI 参数映射，build.bat 链接预编�?llama �?| B01-B07 B09 B12 B13 B26 B30 B32 |
| 3 | (本批) | 内存爆掉根因修复：use_extra_bufts=false 禁用 repack 整体拷贝；llama_batch seq_id 指针覆盖导致的堆损坏修复；chat template 两段式读取；logits idx=-1 语义 | B33 B34 B35 |

## 事故记录

### INC-1: 加载 162GB 模型把内存占满（已修复）
- **现象**：stream_moe.exe 加载 UD-Q8_K_XL 全量分片后提交内存冲�?128GB 物理内存�?- **根因 1（主�?*：`llama_model_default_params().use_extra_bufts=true` �?ggml-cpu repack buffer type �?Q4_K/Q5_K/Q6_K/Q2_K/Q4_0 张量在加载时**整体物理拷贝**进私�?buffer。UD 动态量化正是这些类型的混合体，命中部分达数�?GB�?- **修复**：引擎强�?`mparams.use_extra_bufts=false`，全部权重走默认 CPU buft �?`buffer_from_host_ptr` 零拷�?mmap 路径（实�?162GB 模型加载 6.3s、工作集 52MB）。注意：这满�?Backend.md "保留" 一节——但用户要求�?ffn_*_exps 强制�?pool buft 跳过物理载入"的独立优先判断逻辑属于 Backend.md 自定�?buft 阶段（见 EXPERT_OFFLOAD_INTEGRATION.md L1），当前 mmap 零拷贝是等价�?OS 级实现，EST1/DIO 控制随后续阶段接入�?
### INC-2: llama_batch.seq_id 指针覆盖 -> 堆损�?c0000374（已修复�?- **现象**：首�?decode �?`RtlFreeHeap` �?heap corruption 进程崩溃�?- **根因**：`llama_batch_init` 已为每个槽位 `calloc` �?`seq_id[i]` 数组；我方代码用单个共享数组指针覆盖全部槽位，`llama_batch_free` 逐槽 `free()` 时重复释放同一地址�?- **教训**：llama_batch 的所有权归上游，调用方只填值不换指针。最小复现程序因漏调 llama_batch_free 而未暴露，cdb 符号化栈定位�?decode_tokens:183�?

## ���ܻ�׼��¼��2026-08-26��

### ������ʵ
- ģ���� N: Ϊ **iSCSI ����洢**��LIO-ORG target��200GB�������С���ӳٸߡ�
- ģ�� 162GB �޷���������κα�����ʣ��ռ䣨F: ʣ 279GB ��ģ��Ŀ¼���ݸ干 ~173GB���ҿ����ɱ��ߣ���

### ʵ�⣨70GB RAM ���� / mmap ҳ������� / 16 �߳� CPU / temp=0.6 top_p=0.95��
| �ִ� | prompt tok | decode TPS | ��ע |
|------|-----------|-----------|------|
| EN turn1��ȫ�䣩| 154 | ~2.0 tok/s | �� token ǰ�� 77s prefill����ҳ��ȡ��|
| EN turn2 | 298 | ~1.8 tok/s | |
| ZH turn1-3 | 154-487 | 0.3-2.0 tok/s | �����뵱������δפ��ר�ҵ���ҳ�������� |

### ����
- �����ȷ����ɶ��ԣ�**EN/ZH ˫������**���� benchmark/conversation_real_en3.txt / _zh3.txt����
- ��ǰ decode TPS �� N: ����ר��ҳ��ȡ������ҳ�����������𽥱��ȡ�
- Backend.md �Զ���۳أ�DIO Ԥȡ + EST1 פ����������Դ�ƿ���ļܹ��⣻mmap ����Ϊ������顣