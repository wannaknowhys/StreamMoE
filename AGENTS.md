# StreamMoE AGENTS.md - 用户思维与协作偏好（先读，做选择时参考）

> 本文是用户与 AI 协作的**偏好指南**。开始任何工作前先读本文件；做技术选型、方案取舍、
> 或不确定怎么做时，参考这里的偏好；**拿不定主意就问用户**（用 question 工具）。

## 用户是谁

用户是这套 StreamMoE 引擎的架构师与工程主导者：用缓存理论想问题、用手搓代码落地、用粗判据把关、用完整记录兜底。下列偏好是从长期协作中提炼的，逐条对应实际例子。

## 一、思维方式

1. **抽象与务实并存，切换极快**
   - 高层用抽象心智（如"专家池 = 只读多级缓存 NINEC"）统摄 CPU/GPU/磁盘。
   - 落地用粗判据快速把关（如"cos 只要一直 0.9 就没大错"、"真 short 不会只少 1M"）。
   - 做判断时两者都要给：既有结构视角，也有数量级/阈值视角。

2. **追问本质，但接受多因并存**
   - 会追问到根因（如"是 ulp 差异还是少一个 repack"），但接受"直接原因 + 放大机制"的复合解释，不强迫单一归因。

3. **偏好"改函数/改代码路径"，而非引入抽象机制**
   - 倾向直接改 llama.cpp 的函数（build_moe_ffn、路由、内核），而不是伪造数据结构 / buffer 机制（extra_buffer_type 等）。
   - 报告方案时优先给"直接改代码路径"的走法；只有确实需要才提机制层方案。

4. **通用性 > 单例修复**
   - "我不是单独给 gemma 的补丁，我是为了所有 MoE 通用"——修复/功能要落到模型无关的通用机制，不迁就眼前模型。

## 二、工程偏好

5. **先聊透设计，再动手**
   - 重大改动前：把方案、挂点、数据结构、取舍先摆出来讨论，确认后才实现。用户会说"你先给我讲讲""咱们先商量一下""聊完设计你就改"。

6. **用户自己掌控执行**
   - 给清晰的命令行（bat/node），用户自己跑、自己看输出。不要替用户跑长任务（除非明确要求）。

7. **一切可回放、可追溯**
   - 数据、模型输出、prompt、命令历史都要留档（bin / chat.json / prompt 快照）。模型吐出的**全部文本**（thinking + content，含截断/ctx 耗尽）都要记录。
   - 排查问题时追数据来源、追历史命令。

8. **性能底线不可妥协（异步/零拷贝）**
   - 异步加载（IOCP / io_uring / io_submit）、并发 in-flight、零拷贝直读是前提不是可选项。
   - 报性能方案时默认按"异步 + 并发 + 免 staging"设计。

9. **规范与防污染**
   - vendored `third_party/llama.cpp` **永不 commit**，改动只留工作区、靠 `patches/*.patch` 记录；主仓库文件正常 git add/commit。
   - 多 commit、每步成功就 commit；临时调试代码用完清理；临时文件进 temp/（gitignored）。
   - **commit 顺便 push**——本地提交后立即 `git push origin main`，保持远程同步（用户会看 GitHub 确认进度）。

10. **文档双语文档 + 落地**
    - 设计文档写英文 + 简体中文两版（`[English](x.md) | [简体中文](x.zh-CN.md)`）；讨论定的方案要落地成 md 并提交。

11. **代码只写给终态，CPU 阶段必须以 GPU 等价为准绳（2026-09-06 用户立）**
    - 核心判据：**这个方案在 GPU 上（vulkan/CUDA）能不能按同样的结构跑？**
      - 能（中间量留在设备端、每设备一张图、无 host 往返）→ 写；
      - 不能、只能靠 host<->device 来回搬运数据来"假装能跑" → **一个字都别写**。
    - 一个将来必然推倒的中间方案 = 过度代码。别写过渡执行器、别写注定被换掉的 buffer/路径。
    - 只有两类中间代码值得写：**调试类**（log / dump / debugger / 离线比对脚本）与**验证类**（数值门、对拍基线）——它们服务于"看哪里错了"，用后即弃或留作回归，不进入终态路径。
    - CPU 阶段的工作若要与 GPU 等价，就照着 GPU 的结构写：链在 arena 上跑、节点 data 钉死 offset、一次图整层算；"主节点只在入口唤醒、其余 no-op" 这类结构本身就是 GPU 形态，可以直接落。
    - 这条优先于"先 CPU 验证再迁移"的惯性：**验证手段宁可换成 dump/log 对拍，也不要为 CPU 另造一套将来不用的执行结构。**

12. **数值验证用宽松 gate，不追 bit 级 IDENTICAL（2026-09-06 用户立）**
    - 重构/桶化/并行化会改变浮点累加顺序，产生正常的 ulp 级误差——**只要在 gate 内（实测标定：maxAbs ≤ 1e-5 / cos ≈ 1.0，见 M2_DEVICE_EXECUTOR.md §7.3），且整体逻辑正确，即视为通过**。
    - 别为"逐字节一致"去硬凑中间顺序、别把 ulp 差当 bug 排查。判据是结构等价 + 量级 gate，不是 bit 复刻。

13. **用户明确要求的内容即使会让整体跑不对，也照写（2026-09-06 用户立）**
    - 当用户明确要求的改动可能导致整体无法正确运行/当前不完整时：**先一句话告知可能的后果**，然后**按"当前就是无法正确运行"去写**——不要为了完整性/正确性去补、去绕、去加保护，不要自作主张做超出的部分。
    - 像 REPL 写脚本一样：写一步看一步、接受中间态跑不对，按用户给的步骤逐行推进。
    - 只有用户自己叫停或要求修正时才停。

14. **需求实现途中不搞中间回归验证，直奔终态实现，完工后再做回归（2026-09-08 用户立）**
    - 实现需求途中不进行任何中间数值回归或过渡验证，代码不打折扣地直接按最终目标形态写到底；若实现过程中遇到做不到或不明确的问题直接停下问用户，禁止私自引入过渡妥协代码；全部需求代码写完闭环后再统一进行回归验证。

15. **开发新功能遇到 bug 严禁回滚功能，只需前进不许后退（2026-09-14 用户立）**
    - 开发新功能过程中一旦发现 bug，**禁止用回滚/关闭功能来规避**：不得把新功能默认关（env 默认 off）、不得用 `#ifdef` 把新路径切回旧路径、不得删掉新功能"退回绿"。功能一旦开始落地就保持默认开启，bug 只能**向前修**。
    - 回归失败的正确解读是"当前这个功能就是坏的（bug 态）"，而不是"关掉它"。排查 bug 时保留 bug 态可复现（见下条）。
    - 唯一例外：**用户明确下令**临时关闭某功能；AI 不得自作主张 gate off。
    - 配套构建（见 build.bat / PROJECT_STRUCTURE.md §10）：`StreamMoE_latest` 与 `StreamMoE_dump_dbg` 都是"最新功能"构建（`STREAM_MOE_LATEST`，新功能默认开），区别仅是 `dbg` 多带 `STREAM_MOE_TEMP` 诊断 dump/print；专门用于**体验/复现 bug 态**。生产构建 `StreamMoE_dump` 保持稳定（新功能 opt-in），三者并存。

## 修订（2026-09-17）

- **编译唯一入口：永远只使用 `build.bat`**。包括主程序、库、测试、临时 harness、增量编译和 clean rebuild；禁止直接调用 CMake、Ninja、clang-cl、cl、lib 等工具绕过入口，禁止手工拆库或混链旧产物。入口不支持所需目标时，先讨论并补齐 `build.bat` 支持，不另造编译命令。
- Gemma prefill 回归使用固定 129-token 输入，按整个序列的 cosine 分布判定；首 token 的已知偏差不单独判失败。当前采用 `cos >= 0.99` 的 token 比例至少 90%，同时报告 hidden 分布和低于 0.9 的比例，不追 bit 一致。

## 项目必知速记（2026-09-17 阅读重要文档后整理）

- 文档包含历史方案和未同步 TODO，不能把旧命令、旧阶段结论直接当成当前实现。优先采用用户最新决定、本文修订、`docs/CHECKPOINT.md` 的最新验证记录；发生矛盾先核实，不凭旧稿回退功能。
- 推理入口是 vendored `llama-cli` / `llama-server`；自研 CLI/server/engine 已删除。Route B 沿用官方算子，专家权重不走 mmap，使用 SoA 张量列槽池和唯一紧凑桶执行器；真实 token 子集的 gather、scatter 必须保持同一 tight order。
- `dev_layer()` 表达逻辑 STREAMMOE 归属，`dev_layer_physical()` 表达物理 C1 设备。KV、DSV4 compressor、recurrent state 跟 C1；FA 纳入整层捕获，算子能力按实际设备检查，不支持就明确报错，不静默 CPU fallback。
- 每设备 arena 按生命周期组织 carry 与 scratch；xfer 的 shell 身份必须包含 producer、consumer device、stage、consumer layer。允许跨层复用存储，不允许按首次层分配的同一个 shell 跨消费层复用。导出保留张量不能提前被 arena 覆盖。
- 2026-09-17 固定 129-token 验证：设备 dense 对本次 RAM 参考为 123/129（95.3%）cos >= 0.99，RAM 对冻结基线为 121/129（93.8%）；embd/hidden 均无 token 低于 0.9。证据在 `temp/embedding_probe_2026-09-17T19-07-02-705Z/`。这只覆盖已测配置，不代表所有设备路径均已验证。
- vendored 改动留工作区，主仓库 frag/patch 负责记录；补丁顺序 macros → tsc_timer → route-b → prefill。重放区分原始字节一致和仅 CRLF 规范化后一致，不能混称。临时脚本、harness、日志、重放 clone 放 `temp/`，正式编译产物放 `build/<tag>/`。
- 异步 IO、并发 in-flight、避免冗余搬运是终态底线；文件切片 4K 对齐不等于所有目标布局天然免 staging。Linux 真异步 DIO、层内跨设备细粒度 stage、profile/并发验收仍需按最新源码与状态核实，不能把设计文档当作完成证明。

## 三、协作规则

- 开始任务前：先读本文件 + `docs/CHECKPOINT.md`（当前状态）+ `docs/PROJECT_STRUCTURE.md`（结构）。
- 做技术选型/方案取舍时：按上述偏好给推荐（改函数优先、通用性优先、性能底线、可回放），并说明取舍。
- **拿不定主意 / 有重大分叉（改动面大、影响架构、删除保留等）→ 用 question 工具问用户**，不要自作主张。
- 用户会纠正——纠正意见记进本文件（追加"修订"段，保持本文件与用户当前想法同步）。

## 四、docs 文档地图（按需读取）

### 重要必读（会话开始 / 大改动前）

| 文档                                 | 内容                                                      |
| :----------------------------------- | :-------------------------------------------------------- |
| `docs/CHECKPOINT.md`                 | 当前状态、下一步、验证命令（会话恢复先读）                |
| `docs/PROJECT_STRUCTURE.md`          | 目录/产物/规范、vendored patch 纪律                       |
| `docs/LLAMA_MOE_NO_MMAP_RESEARCH.md` | route B 核心设计（第三路径：官方内核 + 均匀 stride 槽池） |
| `docs/Backend.md`                    | 自定义 backend / expert pool 调度设计                     |
| `docs/VENDORED_MODIFICATIONS.md`     | vendored 改动汇总 + patch 记录                            |

### 按场景读取

| 场景                                              | 文档                                                                                                     |
| :------------------------------------------------ | :------------------------------------------------------------------------------------------------------- |
| 调度/池（dir 二维、异步装载、全局线程、驱逐打分） | `docs/EXPERT_SCHEDULER_DESIGN.md`                                                                        |
| GPU/多设备（vulkan、HOST_VISIBLE、EMA 放置）      | `docs/ROUTE_B_GPU_PHASE.md`                                                                              |
| dense 放置与驻留管理（C1/C2 策略、迁移判据）      | `docs/DENSE_PLACEMENT.md`                                                                                |
| 设备端 dense 闭包与跨设备流转（C1/MoE/C2）        | `docs/DEVICE_DENSE_CLOSURE.md`                                                                           |
| 每设备 arena 规划（区间打包、carry 管线）         | `docs/PER_DEVICE_ARENA.md`                                                                               |
| 整图分区（四区域、接缝 leaves、buft 标记）        | `docs/GRAPH_PARTITION.md`                                                                                |
| route B 整层拥有（整层执行、跨设备搬运）          | `docs/ROUTE_B_LAYER_OWNERSHIP.md`                                                                        |
| 层执行器设计（三段 arena、轻量 mini-graph）       | `docs/LAYER_EXECUTOR_DESIGN.md`                                                                          |
| L2 整层执行审查（A~E 风险清单与防御）             | `docs/L2_WHOLE_LAYER_REVIEW.md`                                                                          |
| M2 设备执行器设计（设备 mini-graph、异步骨架）    | `docs/M2_DEVICE_EXECUTOR.md`                                                                             |
| 紧凑多桶快速路径 / 算子融合                       | `docs/BUCKET_FAST_PATH.md`                                                                               |
| 桶执行 token 子集紧凑收集                         | `docs/BUCKET_EXEC_TOKEN_SUBSET.md`                                                                       |
| token 子集 scatter-add 规划                       | `docs/SCATTER_PLAN.md`                                                                                   |
| 专家移动流水线与 (L,E) 驱逐设计                   | `docs/EXPERT_MOVE_PIPELINE.md`                                                                           |
| VRAM DMA 搬运与 staging 规约                      | `docs/VRAM_DMA_MOVE.md`                                                                                  |
| 吞吐基准评测（单次/多轮、参数矩阵）               | `docs/BENCHMARK.md`                                                                                      |
| 多模型池 / 异构子池                               | `docs/MULTI_MODEL_POOL.md`、`docs/MULTI_SUBPOOL.md`                                                      |
| GGUF 格式 v1/v2 / RAID0 分片                      | `docs/STREAMMOE_GGUF_FORMAT.md`                                                                          |
| Route-B 加载器与 GGUF 输入格式                    | `docs/ROUTE_B_LOADER_FORMATS.md`                                                                         |
| 图构建输出点与阶段宏补丁规则                      | `docs/GRAPH_BUILD_OUTPUT.md`                                                                             |
| 后端数值差异分析与基线控制                        | `docs/BACKEND_DIVERGENCE_ANALYSIS.md`                                                                    |
| prefill 交叉验证 / 专家历史模拟 / repack 排查     | `docs/PREFILL_CROSS_VALIDATION.md`、`docs/EXPERT_TRACE_SIMULATION.md`、`docs/REPACK_DIVERGENCE_DEBUG.md` |
| delegate 排查方法论 / bug 清单                    | `docs/DEBUG_DELEGATION.md`、`docs/BUG_TRACKER.md`                                                        |
| patch 拆分/更新踩坑                               | `docs/PATCH_SPLITTING_PITFALLS.md`                                                                       |
| 迁移上游工具 / 可执行程序路线                     | `docs/UPSTREAM_TOOLS_MIGRATION.md`、`docs/LLAMA_EXE_ROADMAP.md`                                          |
| 冒烟/测试/采样                                    | `docs/SMOKE_TESTING.md`、`docs/TEST_FLOW.md`、`docs/SAMPLING.md`                                         |
| Mock 规范与测试桩原则                             | `docs/MOCK.md`                                                                                           |
| ASan 构建                                         | `docs/ASAN_BUILD.md`                                                                                     |
| v2 架构修正                                       | `docs/V2_ARCHITECTURE_REVISION.md`                                                                       |

### 可以不读（参考/历史）

| 文档                        | 内容                            |
| :-------------------------- | :------------------------------ |
| `docs/REVIEW_2026_08_28.md` | 早期审查对照（结论已并入代码）  |
| `docs/LLAMA_MMAP_CALLS.md`  | mmap 调用点调试地图（低优先级） |
